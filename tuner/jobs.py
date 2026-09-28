"""Supabase jobs: pending configs and evaluation rows."""
from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

from tuner.supabase_scores import rest_json

CONFIG_TABLE = "gatk_configs"
EVAL_TABLE = "gatk_evaluations"
SCORE_VIEW = "gatk_config_scores"


def config_table() -> str:
    return (os.environ.get("SUPABASE_CONFIG_TABLE") or CONFIG_TABLE).strip()


def eval_table() -> str:
    return (os.environ.get("SUPABASE_EVAL_TABLE") or EVAL_TABLE).strip()


def score_view() -> str:
    return (os.environ.get("SUPABASE_SCORE_VIEW") or SCORE_VIEW).strip()


def insert_pending_config(
    *,
    experiment: str,
    updates: Dict[str, Any],
    box: Dict[str, Any],
    trial_number: int,
    gatk_config: Optional[Dict[str, Any]] = None,
    worker_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    rounds_target: Optional[int] = None,
) -> Optional[str]:
    row = {
        "experiment": experiment,
        "tool_name": "gatk",
        "gatk_updates": updates,
        "gatk_config": gatk_config or {},
        "study_name": box.get("study_name") or f"gatk-v2-{box.get('search_category')}",
        "search_category": box.get("search_category"),
        "suggested_by": box.get("suggested_by") or "optuna",
        "hypothesis": box.get("hypothesis"),
        "optuna_trial_number": trial_number,
        "status": "pending",
    }
    if worker_id:
        row["worker_id"] = str(worker_id)
    if batch_id:
        row["batch_id"] = str(batch_id)
    if rounds_target is not None:
        row["rounds_target"] = int(rounds_target)
    inserted = rest_json("POST", config_table(), body=row)
    return _row_id(inserted)


def list_pending(limit: int = 1) -> List[Dict[str, Any]]:
    """Unassigned pending rows. Fleet jobs carry a worker_id and are left for main.py."""
    query = "&".join([
        "select=*",
        "status=eq.pending",
        "worker_id=is.null",
        "order=created_at.asc",
        f"limit={int(limit)}",
    ])
    data = rest_json("GET", config_table(), query=query)
    return data if isinstance(data, list) else []


def list_jobs_for_workers(
    worker_ids: List[str], statuses: List[str]
) -> Optional[List[Dict[str, Any]]]:
    if not worker_ids or not statuses:
        return []
    workers = ",".join(urllib.parse.quote(str(w), safe="") for w in worker_ids)
    states = ",".join(urllib.parse.quote(str(s), safe="") for s in statuses)
    query = "&".join([
        "select=*",
        f"worker_id=in.({workers})",
        f"status=in.({states})",
        "order=created_at.asc",
    ])
    data = rest_json("GET", config_table(), query=query)
    if data is None:
        return None
    return data if isinstance(data, list) else []


def claim_worker_job(worker_id: str) -> Optional[Dict[str, Any]]:
    """Take the oldest pending row for this VPS. Empty when another claim won."""
    worker = urllib.parse.quote(str(worker_id), safe="")
    query = "&".join([
        "select=*",
        "status=eq.pending",
        f"worker_id=eq.{worker}",
        "order=created_at.asc",
        "limit=1",
    ])
    data = rest_json("GET", config_table(), query=query)
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    config_id = str(data[0].get("id") or "")
    if not config_id:
        return None
    claimed = rest_json(
        "PATCH",
        config_table(),
        query="&".join([
            f"id=eq.{urllib.parse.quote(config_id, safe='')}",
            "status=eq.pending",
        ]),
        body={"status": "running"},
    )
    if isinstance(claimed, list) and claimed and isinstance(claimed[0], dict):
        return claimed[0]
    return None


def config_for_trial(study_name: str, trial_number: int) -> Optional[Dict[str, Any]]:
    query = "&".join([
        "select=*",
        f"study_name=eq.{urllib.parse.quote(study_name, safe='')}",
        f"optuna_trial_number=eq.{int(trial_number)}",
        "order=created_at.desc",
        "limit=1",
    ])
    data = rest_json("GET", config_table(), query=query)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def patch_config(config_id: str, fields: Dict[str, Any]) -> bool:
    query = f"id=eq.{urllib.parse.quote(str(config_id), safe='')}"
    updated = rest_json("PATCH", config_table(), query=query, body=fields)
    return updated is not None


def fetch_score_row(config_id: str) -> Optional[Dict[str, Any]]:
    query = "&".join([
        "select=*",
        f"config_id=eq.{urllib.parse.quote(str(config_id), safe='')}",
        "limit=1",
    ])
    data = rest_json("GET", score_view(), query=query)
    if isinstance(data, list) and data:
        return data[0] if isinstance(data[0], dict) else None
    return None


def insert_evaluations(config_id: str, scores: List[Dict[str, Any]]) -> bool:
    rows = [evaluation_row(config_id, score) for score in scores]
    if not rows:
        return True
    inserted = rest_json("POST", eval_table(), body=rows)
    return inserted is not None


def load_score_details(practice_dir: Path) -> List[Dict[str, Any]]:
    scores: List[Dict[str, Any]] = []
    if not practice_dir.is_dir():
        return scores
    for json_path in sorted(practice_dir.glob("*/score_run/score_details.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"   WARNING: skip {json_path}: {e}", flush=True)
            continue
        if isinstance(data, dict):
            scores.append(data)
    return scores


def evaluation_row(config_id: str, score: Dict[str, Any]) -> Dict[str, Any]:
    metrics = score.get("metrics") if isinstance(score.get("metrics"), dict) else {}
    folder_path = score.get("folder")
    folder_name = Path(str(folder_path)).name if folder_path else None
    return {
        "config_id": config_id,
        "folder": folder_name,
        "folder_path": folder_path,
        "region": _pick(score, metrics, "region"),
        "variant_count": _pick(score, metrics, "variant_count"),
        "scorer": _pick(score, metrics, "scorer") or "AdvancedV2",
        "scoring_version": _pick(score, metrics, "scoring_version") or "v2",
        "scoring_status": _pick(score, metrics, "scoring_status"),
        "ok": score.get("ok"),
        "error": score.get("error"),
        "would_record": score.get("would_record"),
        "score": _pick(score, metrics, "score", "combined_final"),
        "combined_final": _pick(score, metrics, "combined_final"),
        "advanced_score": _pick(score, metrics, "advanced_score"),
        "core": _pick(score, metrics, "core"),
        "germline": _pick(score, metrics, "germline"),
        "fp_per_target": _pick(score, metrics, "fp_per_target"),
        "snp_fp_per_target": _pick(score, metrics, "snp_fp_per_target"),
        "snp_final": _pick(score, metrics, "snp_final", "f1_snp"),
        "indel_final": _pick(score, metrics, "indel_final", "f1_indel"),
        "weighted_f1": _pick(score, metrics, "weighted_f1"),
        "f1_snp": _pick(score, metrics, "f1_snp"),
        "precision_snp": _pick(score, metrics, "precision_snp"),
        "recall_snp": _pick(score, metrics, "recall_snp"),
        "tp_snp": _pick(score, metrics, "tp_snp"),
        "fp_snp": _pick(score, metrics, "fp_snp"),
        "fn_snp": _pick(score, metrics, "fn_snp"),
        "truth_total_snp": _pick(score, metrics, "truth_total_snp"),
        "query_total_snp": _pick(score, metrics, "query_total_snp"),
        "target_total_snp": _pick(score, metrics, "target_total_snp"),
        "frac_na_snp": _pick(score, metrics, "frac_na_snp"),
        "query_unk_snp": _pick(score, metrics, "query_unk_snp"),
        "f1_indel": _pick(score, metrics, "f1_indel"),
        "precision_indel": _pick(score, metrics, "precision_indel"),
        "recall_indel": _pick(score, metrics, "recall_indel"),
        "tp_indel": _pick(score, metrics, "tp_indel"),
        "fp_indel": _pick(score, metrics, "fp_indel"),
        "fn_indel": _pick(score, metrics, "fn_indel"),
        "truth_total_indel": _pick(score, metrics, "truth_total_indel"),
        "query_total_indel": _pick(score, metrics, "query_total_indel"),
        "target_total_indel": _pick(score, metrics, "target_total_indel"),
        "frac_na_indel": _pick(score, metrics, "frac_na_indel"),
        "query_unk_indel": _pick(score, metrics, "query_unk_indel"),
        "region_fp_snp": _pick(score, metrics, "region_fp_snp"),
        "region_fp_indel": _pick(score, metrics, "region_fp_indel"),
        "region_fp_total": _pick(score, metrics, "region_fp_total"),
        "overcall_penalty": _pick(score, metrics, "overcall_penalty"),
        "titv_query_snp": _pick(score, metrics, "titv_query_snp"),
        "titv_truth_snp": _pick(score, metrics, "titv_truth_snp"),
        "hethom_query_snp": _pick(score, metrics, "hethom_query_snp"),
        "hethom_truth_snp": _pick(score, metrics, "hethom_truth_snp"),
        "hethom_query_indel": _pick(score, metrics, "hethom_query_indel"),
        "hethom_truth_indel": _pick(score, metrics, "hethom_truth_indel"),
        "weights": _json_safe(score.get("weights") if isinstance(score.get("weights"), dict) else None),
        "difficulty_class_counts": _json_safe(
            score.get("difficulty_class_counts")
            if isinstance(score.get("difficulty_class_counts"), dict)
            else None
        ),
        "metrics": _json_safe(metrics),
    }


def _row_id(inserted: Any) -> Optional[str]:
    row = None
    if isinstance(inserted, list) and inserted and isinstance(inserted[0], dict):
        row = inserted[0]
    elif isinstance(inserted, dict):
        row = inserted
    if not row or not row.get("id"):
        return None
    return str(row["id"])


def _pick(score: Dict[str, Any], metrics: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in score and score[key] is not None:
            return _pg_value(score[key])
        if key in metrics and metrics[key] is not None:
            return _pg_value(metrics[key])
    return None


def _pg_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value
