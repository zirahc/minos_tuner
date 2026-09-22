"""Propose the next GATK search box from gatk_config_scores.

Tuner server only. Does not run GATK or edit minos_subnet.

  python -m tuner.agent
  python -m tuner.agent --heuristic
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

TUNER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TUNER_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(TUNER_ROOT / ".env")
    load_dotenv(TUNER_ROOT / ".env.tuner")
except ImportError:
    pass

from tuner.search_box import space_catalog, specs_to_space_json, validate_search_box, write_search_box
from tuner.spaces import space_for
from tuner.supabase_scores import fetch_config_scores

DEFAULT_OUT = TUNER_ROOT / "search_box.json"
HISTORY_LIMIT = 40

SYSTEM_PROMPT = """You are a Minos subnet 107 GATK practice-tuning analyst.

You propose ONE search box for Optuna. You do not invent GATK flags.
You do not tune a live miner. You only choose a category and bounds.

v2 ranking number is avg_combined_final:
  combined_final = 0.70 * core + 0.30 * exp(-fp_per_target / 8)
SNP/INDEL F1 (snp_final, weighted_f1) is diagnostic, not the objective.

Rules:
- Optimize avg_combined_final.
- Open exactly one search_category from the catalog.
- space keys must be a subset of that category. Do not add other GATK keys.
- Bounds must stay inside the catalog low/high (or choices).
- n_trials is 4..8.
- If target F1 is already high and fp_per_target is high, open quality_filters
  (calling confidence / mapping quality), not heterozygosity priors.
- If INDEL F1 lags SNP F1, consider pcr or assembly, not SNP quality alone.
- One category per round.
- Failed / missing scores are not a GATK failure; ignore them for ranking.
- Write a short hypothesis that a later review can confirm or reject.

Return ONLY a JSON object with keys:
  search_category, hypothesis, space, constraints, n_trials, optimize
optimize must be "avg_combined_final".
constraints may include min_avg_f1_snp, min_avg_f1_indel, min_avg_combined_final.
"""


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    rows = fetch_config_scores(category=None, scored_only=not args.include_failed)
    if rows is None:
        return 2

    history = _compact_history(rows)
    print(f"   history rows for agent: {len(history)}", flush=True)

    if args.heuristic or not _has_llm_key():
        if not args.heuristic and not _has_llm_key():
            print(
                "   no ANTHROPIC_API_KEY or OPENAI_API_KEY — using heuristic box. "
                "Do not paste keys here.",
                flush=True,
            )
        try:
            box = heuristic_search_box(history)
            box = validate_search_box(box)
        except ValueError as e:
            print(f"ERROR: heuristic box invalid: {e}", flush=True)
            return 2
        source = "heuristic"
    else:
        try:
            raw = _llm_search_box(history)
            box = validate_search_box(raw)
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: LLM search box rejected: {e}", flush=True)
            return 2
        source = "llm"

    box["suggested_by"] = f"agent+{source}"
    out_path = Path(args.out).resolve() if args.out else DEFAULT_OUT
    write_search_box(box, out_path)
    print(f"   wrote {out_path}", flush=True)
    print(json.dumps(box, indent=2, sort_keys=True), flush=True)
    return 0


def heuristic_search_box(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in history if r.get("avg_combined_final") is not None]
    latest = scored[-1] if scored else {}
    f1_snp = _num(latest.get("avg_f1_snp"))
    f1_indel = _num(latest.get("avg_f1_indel"))
    fp = _num(latest.get("avg_fp_per_target"))

    if f1_snp is not None and f1_snp >= 0.98 and fp is not None and fp >= 1.0:
        category = "quality_filters"
        hypothesis = (
            "Target SNP F1 is already high; v2 looks limited by fp_per_target. "
            "Search quality filters and calling confidence."
        )
    elif (
        f1_indel is not None
        and f1_snp is not None
        and f1_indel + 0.02 < f1_snp
    ):
        category = "pcr"
        hypothesis = (
            "INDEL F1 lags SNP F1. Search the PCR indel model before touching SNP priors."
        )
    elif not scored:
        category = "quality_filters"
        hypothesis = (
            "No scored history yet. Start with quality filters; they are the usual "
            "lever for extra region calls."
        )
    else:
        category = "quality_filters"
        hypothesis = (
            "Default next box: quality filters, one category, maximize avg_combined_final."
        )

    constraints: Dict[str, float] = {}
    if f1_snp is not None:
        constraints["min_avg_f1_snp"] = round(max(0.0, min(1.0, f1_snp - 0.02)), 4)
    if f1_indel is not None:
        constraints["min_avg_f1_indel"] = round(max(0.0, min(1.0, f1_indel - 0.02)), 4)

    return {
        "search_category": category,
        "hypothesis": hypothesis,
        "space": specs_to_space_json(space_for(category)),
        "constraints": constraints,
        "n_trials": 6,
        "optimize": "avg_combined_final",
    }


def _compact_history(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for row in rows[-HISTORY_LIMIT:]:
        compact.append({
            "config_id": row.get("config_id"),
            "experiment": row.get("experiment"),
            "search_category": row.get("search_category"),
            "suggested_by": row.get("suggested_by"),
            "hypothesis": row.get("hypothesis"),
            "status": row.get("status"),
            "n_scored": row.get("n_scored"),
            "avg_combined_final": row.get("avg_combined_final"),
            "avg_core": row.get("avg_core"),
            "avg_germline": row.get("avg_germline"),
            "avg_fp_per_target": row.get("avg_fp_per_target"),
            "avg_f1_snp": row.get("avg_f1_snp"),
            "avg_f1_indel": row.get("avg_f1_indel"),
            "gatk_updates": row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {},
        })
    return compact


def _llm_search_box(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    user = json.dumps(
        {
            "catalog": space_catalog(),
            "history": history,
            "instruction": (
                "Choose the next search box from catalog + history. "
                "Return JSON only."
            ),
        },
        default=str,
    )
    provider = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if provider == "openai" or (not provider and os.environ.get("OPENAI_API_KEY")):
        text = _openai_chat(SYSTEM_PROMPT, user)
    else:
        text = _anthropic_chat(SYSTEM_PROMPT, user)
    return _parse_json_object(text)


def _has_llm_key() -> bool:
    return bool(
        (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        or (os.environ.get("OPENAI_API_KEY") or "").strip()
    )


def _anthropic_chat(system: str, user: str) -> str:
    key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    model = (os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-6").strip()
    body = {
        "model": model,
        "max_tokens": 1500,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    )
    raw = _http_json(req)
    blocks = raw.get("content") if isinstance(raw, dict) else None
    if not isinstance(blocks, list):
        raise RuntimeError("unexpected Anthropic response")
    parts = [b.get("text", "") for b in blocks if isinstance(b, dict)]
    return "\n".join(parts)


def _openai_chat(system: str, user: str) -> str:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    model = (os.environ.get("OPENAI_MODEL") or "gpt-4.1").strip()
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        },
    )
    raw = _http_json(req)
    try:
        return raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError("unexpected OpenAI response") from e


def _http_json(req: urllib.request.Request) -> Any:
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            if not (200 <= resp.status < 300):
                raise RuntimeError(f"LLM HTTP {resp.status}")
            return json.loads(payload)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"LLM HTTP {e.code}: {err}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM request failed: {e.reason}") from e


def _parse_json_object(text: str) -> Dict[str, Any]:
    blob = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", blob, re.DOTALL)
    if fenced:
        blob = fenced.group(1)
    else:
        start = blob.find("{")
        end = blob.rfind("}")
        if start >= 0 and end > start:
            blob = blob[start : end + 1]
    try:
        data = json.loads(blob)
    except ValueError as e:
        raise ValueError("model did not return JSON") from e
    if not isinstance(data, dict):
        raise ValueError("model JSON was not an object")
    return data


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Propose the next GATK Optuna search box from Supabase history.",
    )
    p.add_argument(
        "--heuristic",
        action="store_true",
        help="Skip the LLM and emit the rule-based box.",
    )
    p.add_argument(
        "--include-failed",
        action="store_true",
        help="Include configs with no scored combined_final in the history payload.",
    )
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Write JSON here (default {DEFAULT_OUT})",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
