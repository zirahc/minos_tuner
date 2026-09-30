#!/usr/bin/env python3
"""GATK VPS loop, or a one-shot update file.

24/7 (default): wait until the tuner assigns this WORKER_ID a setting,
score that setting on 15 practice rounds, POST the rows, then wait again.

  set WORKER_ID=1
  python main.py

One-shot file (unchanged sweep):

  python main.py --updates gatk_updates.json

Run gatk_tuning.sql in the Supabase SQL editor before the first POST.
This does not submit to the live Minos miner API.
Set SUPABASE_URL and SUPABASE_KEY in .env. Do not paste those values here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent


def _load_env_file() -> None:
    """Load .env even when python-dotenv is not installed. Never print values."""
    env_path = REPO_ROOT / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except ImportError:
        pass
    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if name and name not in os.environ:
            os.environ[name] = value


_load_env_file()

GATK_CONF = REPO_ROOT / "configs" / "gatk.conf"
SCORER = REPO_ROOT / "score_gatk_folders.py"
PRACTICE_DIR = REPO_ROOT / "datasets" / "practice"
UPDATES_FILE = REPO_ROOT / "gatk_updates.json"
DEFAULT_CONFIG_TABLE = "gatk_configs"
DEFAULT_EVAL_TABLE = "gatk_evaluations"
TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def main() -> int:
    args = _parse_args()
    if not args.updates:
        return run_wait_loop(args)
    updates_file = Path(args.updates)
    if not updates_file.is_absolute():
        updates_file = REPO_ROOT / updates_file

    print("=" * 72, flush=True)
    print("  GATK UPDATE → SCORE → POST", flush=True)
    print("=" * 72, flush=True)

    dest = _supabase_dest()
    if dest is None:
        return 2
    print(
        f"   Supabase tables: {dest[2]} (config), {dest[3]} (evaluations)",
        flush=True,
    )

    if not GATK_CONF.exists():
        print(f"ERROR: GATK config not found: {GATK_CONF}", flush=True)
        return 2
    if not SCORER.exists():
        print(f"ERROR: scorer not found: {SCORER}", flush=True)
        return 2

    try:
        experiments = load_update_lists(updates_file)
    except (OSError, ValueError) as e:
        print(f"ERROR: could not load {updates_file}: {e}", flush=True)
        return 2

    if not experiments:
        print(f"ERROR: no update lists in {updates_file}", flush=True)
        return 2

    original_conf, backup = _original_gatk_conf()
    if original_conf is None:
        return 2
    print(
        f"   Original GATK snapshot: {backup.name} "
        f"(restored after every trial)",
        flush=True,
    )

    print(f"   Update lists: {len(experiments)} from {updates_file.name}", flush=True)
    for name, _updates, meta in experiments:
        cat = meta.get("search_category") or ""
        who = meta.get("suggested_by") or ""
        extra = "  ".join(x for x in (cat, who) if x)
        print(f"     - {name}" + (f"  ({extra})" if extra else ""), flush=True)

    worst_rc = 0
    try:
        for i, (name, updates, meta) in enumerate(experiments, 1):
            print(f"\n{'#' * 72}", flush=True)
            print(f"  [{i}/{len(experiments)}] {name}", flush=True)
            print(f"{'#' * 72}", flush=True)
            try:
                if not _restore_gatk_conf(original_conf):
                    return 2
                rc = run_one(
                    name, updates, meta,
                    sweep_index=i, sweep_total=len(experiments),
                )
            finally:
                _restore_gatk_conf(original_conf)
            if rc > worst_rc:
                worst_rc = rc
            print(
                f"   Sweep {i}/{len(experiments)} finished "
                f"(exit={rc}). Next value starts from original gatk.conf.",
                flush=True,
            )
    finally:
        _restore_gatk_conf(original_conf)

    print(
        f"\n   Finished {len(experiments)} update list(s). "
        f"{GATK_CONF.name} restored from {backup.name}. worst exit={worst_rc}",
        flush=True,
    )
    return worst_rc


def run_wait_loop(args: argparse.Namespace) -> int:
    """Claim this VPS's next tuner job, score FLEET rounds, POST, repeat."""
    worker_id = str(args.worker or "").strip()
    if not worker_id:
        print("ERROR: set WORKER_ID or pass --worker.", flush=True)
        return 2
    dest = _supabase_dest()
    if dest is None:
        return 2
    root, conf, scorer, practice = _gatk_paths()
    if not conf.exists() or not scorer.exists():
        print(
            f"ERROR: {root} is missing configs/gatk.conf or score_gatk_folders.py. "
            "Set MINOS_SUBNET to the GATK checkout.",
            flush=True,
        )
        return 2
    from tuner.jobs import claim_worker_job, patch_config, touch_worker

    print("=" * 72, flush=True)
    print("  GATK VPS  waiting for tuner updates", flush=True)
    print("=" * 72, flush=True)
    print(f"   worker={worker_id}  MINOS={root}  poll={args.poll}s", flush=True)
    print("   this machine posts itself so the fleet can assign work", flush=True)
    print("   stop with Ctrl+C", flush=True)
    while True:
        try:
            if not touch_worker(worker_id):
                print(
                    f"   WARNING: worker={worker_id} could not post online status",
                    flush=True,
                )
            job = claim_worker_job(worker_id)
            if not job:
                print(f"   worker={worker_id} waiting for a GATK update ...", flush=True)
                time.sleep(max(1, args.poll))
                continue
            _run_assigned_job(
                job,
                conf=conf,
                scorer=scorer,
                practice=practice,
                root=root,
                rounds_override=args.rounds,
                patch_config=patch_config,
            )
        except KeyboardInterrupt:
            print(f"\n   worker={worker_id} stopped", flush=True)
            return 0
        except Exception as e:
            print(
                f"   ERROR: {type(e).__name__}: {e} — worker={worker_id} "
                "retries in 60s",
                flush=True,
            )
            time.sleep(60)


def _run_assigned_job(
    job: Dict[str, Any],
    *,
    conf: Path,
    scorer: Path,
    practice: Path,
    root: Path,
    rounds_override: int,
    patch_config: Any,
) -> int:
    config_id = str(job.get("id") or "")
    updates = job.get("gatk_updates") if isinstance(job.get("gatk_updates"), dict) else {}
    rounds = int(rounds_override or job.get("rounds_target") or 15)
    offset = int(job.get("rounds_offset") or 0)
    experiment = job.get("experiment") or config_id
    print(
        f"\n   job {config_id}  {experiment}  rounds={rounds}  "
        f"offset={offset}  updates={updates}",
        flush=True,
    )
    try:
        original = conf.read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: read {conf}: {e}", flush=True)
        patch_config(config_id, {"status": "failed"})
        return 1
    try:
        if updates:
            changed = update_gatk_conf(conf, updates)
            if changed is None:
                patch_config(config_id, {"status": "failed"})
                return 1
            for key, old, new in changed:
                print(f"     {key}: {old} -> {new}", flush=True)
        gatk_params = read_gatk_conf(conf)
        stage = root / "datasets" / "practice_fleet"
        try:
            n_rounds = stage_practice_rounds(practice, stage, rounds, offset)
        except OSError as e:
            print(f"ERROR: could not stage {rounds} rounds: {e}", flush=True)
            patch_config(config_id, {"status": "failed", "gatk_config": gatk_params})
            return 1
        if n_rounds <= 0:
            print(f"ERROR: no complete practice rounds under {practice}", flush=True)
            patch_config(config_id, {"status": "failed", "gatk_config": gatk_params})
            return 1
        print(f"   scoring {n_rounds} practice round(s)", flush=True)
        started = time.time()
        scored = subprocess.run(
            [sys.executable, str(scorer), "--practice-dir", str(stage), "--config", str(conf)],
            cwd=str(root),
        )
        scores = _scores_since(stage, started)
        print(
            f"   posting {len(scores)} individual round row(s) to gatk_evaluations",
            flush=True,
        )
        scored_ok = [s for s in scores if s.get("ok") and s.get("combined_final") is not None]
        posted = post_evaluations(config_id, scores)
        status = "scored" if posted and scored_ok else "failed"
        patch_config(config_id, {"status": status, "gatk_config": gatk_params})
        patch_config(config_id, {"rounds_done": len(scored_ok)})
        print(
            f"   job {config_id} {status}  rounds_done={len(scored_ok)}  "
            f"scorer_exit={scored.returncode}",
            flush=True,
        )
        return 0 if status == "scored" else 1
    finally:
        try:
            conf.write_text(original, encoding="utf-8")
            print(f"   restored {conf.name}", flush=True)
        except OSError as e:
            print(f"ERROR: could not restore {conf}: {e}", flush=True)


def stage_practice_rounds(practice: Path, stage: Path, limit: int, offset: int = 0) -> int:
    """Link one shared slice of practice rounds so every VPS scores the same set."""
    if not practice.is_dir():
        print(f"ERROR: practice directory not found: {practice}", flush=True)
        return 0
    if stage.exists():
        for child in list(stage.iterdir()):
            _remove_link(child)
    else:
        stage.mkdir(parents=True, exist_ok=True)
    chosen = [folder for folder in sorted(practice.iterdir()) if _is_round_folder(folder)]
    start = max(0, int(offset))
    chosen = chosen[start : start + max(1, int(limit))]
    for folder in chosen:
        _link_dir(folder.resolve(), stage / folder.name)
    if len(chosen) < limit:
        print(
            f"   WARNING: only {len(chosen)} practice round(s); requested {limit}",
            flush=True,
        )
    return len(chosen)


def _is_round_folder(folder: Path) -> bool:
    if not folder.is_dir() or folder.name.startswith("."):
        return False
    if folder.name == "round_2aeeddc1d86288f3":
        return False
    bam = list(folder.glob("*.bam"))
    truth = list(folder.glob("truth*.vcf*")) or list(folder.glob("*truth*.vcf*"))
    mutations = list(folder.glob("mutations*.vcf*")) or list(folder.glob("*mut*.vcf*"))
    return bool(bam and truth and mutations)


def _link_dir(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        _remove_link(dest)
    try:
        os.symlink(src, dest, target_is_directory=True)
        return
    except OSError:
        pass
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(dest), str(src)],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not dest.exists():
        detail = (completed.stderr or completed.stdout or "").strip()
        raise OSError(detail or f"could not link {src} -> {dest}")


def _remove_link(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "rmdir", str(path)], capture_output=True)
        return
    path.unlink()


def _scores_since(practice_dir: Path, started: float) -> List[Dict[str, Any]]:
    """One score_details.json per practice round. Does not average them."""
    scores: List[Dict[str, Any]] = []
    missing: List[str] = []
    if not practice_dir.is_dir():
        return scores
    children = sorted(
        child for child in practice_dir.iterdir()
        if child.is_dir()
    )
    for child in children:
        json_path = child / "score_run" / "score_details.json"
        if not json_path.is_file():
            missing.append(child.name)
            continue
        try:
            if json_path.stat().st_mtime + 1 < started:
                missing.append(child.name)
                continue
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"   WARNING: skip {json_path}: {e}", flush=True)
            missing.append(child.name)
            continue
        if isinstance(data, dict):
            data.setdefault("folder", str(child))
            data.setdefault("score_details_path", str(json_path))
            scores.append(data)
    if missing:
        print(
            f"   WARNING: {len(missing)} round(s) had no new score file: "
            + ", ".join(missing),
            flush=True,
        )
    return scores


def _gatk_paths() -> Tuple[Path, Path, Path, Path]:
    root = Path(os.environ.get("MINOS_SUBNET") or REPO_ROOT).resolve()
    return (
        root,
        root / "configs" / "gatk.conf",
        root / "score_gatk_folders.py",
        root / "datasets" / "practice",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Wait for a tuner GATK update and score it, or score one updates file.",
    )
    p.add_argument(
        "--updates",
        default=None,
        help="Score this JSON file once and exit. Omit to wait for the tuner.",
    )
    p.add_argument(
        "--worker",
        default=os.environ.get("WORKER_ID") or "",
        help="VPS id that matches TUNER_WORKERS (default: WORKER_ID).",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Practice rounds to score. 0 uses the job's rounds_target (15).",
    )
    p.add_argument("--poll", type=int, default=int(os.environ.get("FLEET_POLL_SEC") or 20))
    return p.parse_args()


def _original_gatk_conf() -> Tuple[Optional[str], Path]:
    backup = GATK_CONF.with_suffix(GATK_CONF.suffix + ".bak")
    if backup.exists():
        try:
            return backup.read_text(encoding="utf-8"), backup
        except OSError as e:
            print(f"ERROR: could not read backup {backup}: {e}", flush=True)
            return None, backup
    try:
        original_conf = GATK_CONF.read_text(encoding="utf-8")
        backup.write_text(original_conf, encoding="utf-8")
        print(f"   Backup written: {backup.name}", flush=True)
        return original_conf, backup
    except OSError as e:
        print(f"ERROR: could not snapshot {GATK_CONF}: {e}", flush=True)
        return None, backup


def _restore_gatk_conf(original_conf: str) -> bool:
    try:
        GATK_CONF.write_text(original_conf, encoding="utf-8")
        return True
    except OSError as e:
        print(f"ERROR: could not restore {GATK_CONF}: {e}", flush=True)
        return False


def run_one(
    name: str,
    updates: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    sweep_index: int = 1,
    sweep_total: int = 1,
) -> int:
    if updates:
        changed = update_gatk_conf(GATK_CONF, updates)
        if changed is None:
            return 2
        print(f"\n   Updated {GATK_CONF.name} for {name}:", flush=True)
        for key, old, new in changed:
            print(f"     {key}: {old} -> {new}", flush=True)
        if not changed:
            print("     (listed params already matched; file unchanged)", flush=True)
    else:
        print(f"\n   {name} has empty updates — leaving gatk.conf as-is.", flush=True)

    gatk_params = read_gatk_conf(GATK_CONF)
    print(
        f"\n   POST sweep {sweep_index}/{sweep_total} config now "
        f"(before GATK). Refresh gatk_configs to see this row.",
        flush=True,
    )
    config_id = post_config(name, updates, gatk_params, meta, status="running")
    if not config_id:
        return 1

    print(f"\n   Running {SCORER.name} ...", flush=True)
    scored = subprocess.run([sys.executable, str(SCORER)], cwd=str(REPO_ROOT))
    if scored.returncode != 0:
        print(
            f"   WARNING: scorer exited {scored.returncode} — still posting any scores that exist.",
            flush=True,
        )

    scores = load_score_details(PRACTICE_DIR)
    print_results(name, updates, gatk_params, scores, meta)
    posted = post_evaluations(config_id, scores)
    scored_ok = [
        s for s in scores
        if s.get("ok") and s.get("combined_final") is not None
    ]
    _patch_config_status(config_id, "scored" if scored_ok else "failed")
    if not posted:
        return 1
    print(
        f"   Sweep {sweep_index}/{sweep_total} evaluations posted "
        f"config_id={config_id}",
        flush=True,
    )
    return 0 if scored.returncode == 0 else scored.returncode


PROVENANCE_KEYS = frozenset({
    "name",
    "updates",
    "sweep",
    "search_category",
    "hypothesis",
    "suggested_by",
    "study_name",
    "parent_config_id",
    "optuna_trial_number",
    "status",
})


def load_update_lists(path: Path) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Load one or more named GATK update dicts from JSON.

    Accepted shapes:
      [{"name": "...", "updates": {"param": value, ...}}, ...]
      [{"name": "...", "sweep": {"param": "...", "start": 1, "stop": 100, "step": 1}}]
      [{"name": "...", "sweep": {"param": "...", "values": ["NONE", "HOSTILE"]}}]
      [{"param": value, ...}, ...]
      {"name": "...", "updates": {...}}
      {"param": value, ...}

    Optional provenance on an object with "updates" or "sweep":
      search_category, hypothesis, suggested_by, study_name,
      parent_config_id, optuna_trial_number.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("experiments"), list):
        raw = raw["experiments"]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("expected a JSON array of update lists, or one update object")

    experiments: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"entry {i} is not an object")
        experiments.extend(_expand_item(item, i))
    return experiments


def _expand_item(item: Dict[str, Any], index: int) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    if "sweep" in item:
        return _expand_sweep(item, index)
    if "updates" in item:
        name = str(item.get("name") or f"update-{index}")
        updates = item["updates"]
        if not isinstance(updates, dict):
            raise ValueError(f"{name}: 'updates' must be an object of param -> value")
        return [(name, updates, _provenance_meta(item, name))]
    name = str(item.get("name") or f"update-{index}") if "name" in item else f"update-{index}"
    updates = {k: v for k, v in item.items() if k not in PROVENANCE_KEYS}
    return [(name, updates, _provenance_meta(item, name))]


def _expand_sweep(item: Dict[str, Any], index: int) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    sweep = item["sweep"]
    if not isinstance(sweep, dict):
        raise ValueError(f"entry {index}: 'sweep' must be an object")
    param = sweep.get("param")
    if not param or not isinstance(param, str):
        raise ValueError(f"entry {index}: sweep.param is required")
    values = _sweep_values(sweep, index)
    base_name = str(item.get("name") or param)
    hypothesis = item.get("hypothesis") or (
        f"One-parameter sweep of {param}; all other GATK params stay at original gatk.conf."
    )
    expanded: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for value in values:
        name = f"{base_name}-{value}"
        meta = _provenance_meta(item, name)
        meta["hypothesis"] = f"{hypothesis} value={value}"
        expanded.append((name, {param: value}, meta))
    return expanded


def _sweep_values(sweep: Dict[str, Any], index: int) -> List[Any]:
    if "values" in sweep:
        values = sweep["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"entry {index}: sweep.values must be a non-empty list")
        return values
    if "start" not in sweep or "stop" not in sweep:
        raise ValueError(f"entry {index}: sweep needs values, or start and stop")
    start, stop = sweep["start"], sweep["stop"]
    step = sweep.get("step", 1)
    if isinstance(start, bool) or isinstance(stop, bool) or isinstance(step, bool):
        raise ValueError(f"entry {index}: sweep start/stop/step must be numeric")
    if not isinstance(start, (int, float)) or not isinstance(stop, (int, float)):
        raise ValueError(f"entry {index}: sweep start/stop must be numeric")
    if not isinstance(step, (int, float)) or step == 0:
        raise ValueError(f"entry {index}: sweep step must be a non-zero number")
    if (stop - start) * step < 0:
        raise ValueError(f"entry {index}: sweep step has the wrong sign")
    values: List[Any] = []
    current = start
    int_range = all(isinstance(x, int) and not isinstance(x, bool) for x in (start, stop, step))
    while (step > 0 and current <= stop) or (step < 0 and current >= stop):
        values.append(current if not int_range else int(current))
        current = current + step
        if int_range:
            current = int(current)
        if len(values) > 10_000:
            raise ValueError(f"entry {index}: sweep produced more than 10000 values")
    if not values:
        raise ValueError(f"entry {index}: sweep produced no values")
    return values


def _provenance_meta(item: Dict[str, Any], name: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key in (
        "search_category",
        "hypothesis",
        "suggested_by",
        "study_name",
        "parent_config_id",
        "optuna_trial_number",
    ):
        value = item.get(key)
        if value is not None and value != "":
            meta[key] = value
    if not meta.get("search_category"):
        meta["search_category"] = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "unspecified"
    if not meta.get("suggested_by"):
        meta["suggested_by"] = "human"
    if not meta.get("study_name"):
        meta["study_name"] = f"gatk-v2-{meta['search_category']}"
    return meta


def update_gatk_conf(path: Path, updates: Dict[str, Any]) -> List[Tuple[str, str, str]] | None:
    """Replace matching key=value lines. Returns (key, old, new) for real changes."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"ERROR: could not read {path}: {e}", flush=True)
        return None

    pending = dict(updates)
    changed: List[Tuple[str, str, str]] = []
    out_lines: List[str] = []
    for raw in text.splitlines(keepends=True):
        stripped = raw.strip()
        newline = "\n" if raw.endswith("\n") else ""
        body = stripped[:-1] if stripped.endswith("\r") else stripped
        if body and not body.startswith("#") and "=" in body:
            key, _, current = body.partition("=")
            key = key.strip()
            if key in pending:
                new_val = _format_conf_value(pending.pop(key))
                old_val = current.strip()
                indent = raw[: len(raw) - len(raw.lstrip())]
                out_lines.append(f"{indent}{key}={new_val}{newline}")
                if old_val != new_val:
                    changed.append((key, old_val, new_val))
                continue
        out_lines.append(raw)

    if pending:
        if out_lines and not out_lines[-1].endswith("\n"):
            out_lines[-1] += "\n"
        out_lines.append("\n# Added by main.py\n")
        for key, value in pending.items():
            new_val = _format_conf_value(value)
            out_lines.append(f"{key}={new_val}\n")
            changed.append((key, "(missing)", new_val))

    try:
        path.write_text("".join(out_lines), encoding="utf-8")
    except OSError as e:
        print(f"ERROR: could not write {path}: {e}", flush=True)
        return None
    return changed


def read_gatk_conf(path: Path) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        params[key] = _parse_conf_value(val)
    return params


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
            data.setdefault("score_details_path", str(json_path))
            scores.append(data)
    return scores


def print_results(
    name: str,
    updates: Dict[str, Any],
    gatk_params: Dict[str, Any],
    scores: List[Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    meta = meta or {}
    print(f"\n{'=' * 72}", flush=True)
    print(f"  RESULTS  {name}", flush=True)
    print(f"{'=' * 72}", flush=True)

    print("\n  PROVENANCE", flush=True)
    print(f"    study_name        {meta.get('study_name') or 'n/a'}", flush=True)
    print(f"    search_category   {meta.get('search_category') or 'n/a'}", flush=True)
    print(f"    suggested_by      {meta.get('suggested_by') or 'n/a'}", flush=True)
    print(f"    hypothesis        {meta.get('hypothesis') or 'n/a'}", flush=True)

    print("\n  GATK UPDATES", flush=True)
    if updates:
        for key in sorted(updates):
            print(f"    {key}={updates[key]}", flush=True)
    else:
        print("    (none)", flush=True)

    print(f"\n  GATK CONFIG PARAMS  ({len(gatk_params)})", flush=True)
    if gatk_params:
        width = max(len(k) for k in gatk_params)
        for key in sorted(gatk_params):
            print(f"    {key:<{width}}  {gatk_params[key]}", flush=True)
    else:
        print("    (none)", flush=True)

    print(f"\n  V2 SCORING  ({len(scores)} folder(s))", flush=True)
    if not scores:
        print("    no score_details.json files found", flush=True)
        print(f"{'=' * 72}\n", flush=True)
        return

    print(
        f"    {'folder':32s} {'ok':4s} {'vars':>6s} {'snp_f1':>10s} "
        f"{'indel_f1':>10s} {'core':>10s} {'v2':>10s} {'final':>10s}",
        flush=True,
    )
    for score in scores:
        folder = Path(str(score.get("folder") or "")).name[:32] or "?"
        ok = "yes" if score.get("ok") else "no"
        nvars = score.get("variant_count", "")
        metrics = _score_metrics(score)
        print(
            f"    {folder:32s} {ok:4s} {str(nvars):>6s} "
            f"{_fmt(metrics.get('f1_snp', score.get('snp_final'))):>10s} "
            f"{_fmt(metrics.get('f1_indel', score.get('indel_final'))):>10s} "
            f"{_fmt(score.get('core')):>10s} "
            f"{_fmt(score.get('advanced_score')):>10s} "
            f"{_fmt(score.get('combined_final')):>10s}",
            flush=True,
        )
        if score.get("error"):
            print(f"      error: {score['error']}", flush=True)

    for score in scores:
        folder = Path(str(score.get("folder") or "")).name or "?"
        metrics = _score_metrics(score)
        print(f"\n    --- {folder} ---", flush=True)
        print(f"      region           {_fmt(score.get('region'))}", flush=True)
        print(f"      tool_name        {_fmt(score.get('tool_name'))}", flush=True)
        print(f"      scorer           {_fmt(score.get('scorer'))}", flush=True)
        print(f"      scoring_version  {_fmt(score.get('scoring_version'))}", flush=True)
        print(f"      scoring_status   {_fmt(score.get('scoring_status'))}", flush=True)
        print(f"      snp_final        {_fmt(score.get('snp_final'))}", flush=True)
        print(f"      indel_final      {_fmt(score.get('indel_final'))}", flush=True)
        print(f"      score            {_fmt(score.get('score'))}", flush=True)
        print(f"      combined_final   {_fmt(score.get('combined_final'))}", flush=True)
        print(f"      advanced_score   {_fmt(score.get('advanced_score'))}", flush=True)
        print(f"      core             {_fmt(score.get('core'))}", flush=True)
        print(f"      germline         {_fmt(score.get('germline'))}", flush=True)
        print(f"      fp_per_target    {_fmt(score.get('fp_per_target'))}", flush=True)
        print(f"      would_record     {_fmt(score.get('would_record'))}", flush=True)
        weights = score.get("weights") if isinstance(score.get("weights"), dict) else {}
        if weights:
            print(
                f"      weights          core={_fmt(weights.get('core'))}  "
                f"germline={_fmt(weights.get('germline'))}",
                flush=True,
            )
        counts = score.get("difficulty_class_counts") if isinstance(score.get("difficulty_class_counts"), dict) else {}
        if counts:
            print("      difficulty class counts:", flush=True)
            for cls, vals in counts.items():
                if not isinstance(vals, dict):
                    print(f"        {cls:16s} {vals}", flush=True)
                    continue
                print(
                    f"        {cls:16s} tp={vals.get('tp', 0)}  "
                    f"fn={vals.get('fn', 0)}  fp={vals.get('fp', 0)}",
                    flush=True,
                )
        print("      hap.py metrics:", flush=True)
        if not metrics:
            print("        (none)", flush=True)
        else:
            width = max(len(str(k)) for k in metrics)
            for key in sorted(metrics):
                value = metrics[key]
                if isinstance(value, dict):
                    print(f"        {key:<{width}}  {json.dumps(value, default=str)}", flush=True)
                else:
                    print(f"        {key:<{width}}  {_fmt(value)}", flush=True)

    print(f"{'=' * 72}\n", flush=True)


def _score_metrics(score: Dict[str, Any]) -> Dict[str, Any]:
    metrics = score.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def post_config(
    name: str,
    updates: Dict[str, Any],
    gatk_params: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    status: str = "running",
) -> Optional[str]:
    dest = _supabase_dest()
    if dest is None:
        return None
    url, key, config_table, _eval_table = dest
    meta = dict(meta or {})
    config_row = {
        "experiment": name,
        "tool_name": "gatk",
        "gatk_updates": _json_safe(updates if isinstance(updates, dict) else {}),
        "gatk_config": _json_safe(gatk_params if isinstance(gatk_params, dict) else {}),
        "study_name": meta.get("study_name"),
        "search_category": meta.get("search_category"),
        "suggested_by": meta.get("suggested_by") or "human",
        "hypothesis": meta.get("hypothesis"),
        "parent_config_id": meta.get("parent_config_id"),
        "optuna_trial_number": meta.get("optuna_trial_number"),
        "status": status,
    }
    print(
        f"   POST config → {config_table}  experiment={name}  "
        f"category={config_row['search_category']}  status={status}",
        flush=True,
    )
    inserted = _supabase_insert(url, key, config_table, config_row)
    if inserted is None:
        return None
    config_id = _inserted_id(inserted)
    if not config_id:
        print("   ERROR: config insert returned no id", flush=True)
        return None
    print(f"   inserted config id={config_id}", flush=True)
    return config_id


def post_evaluations(config_id: str, scores: List[Dict[str, Any]]) -> bool:
    dest = _supabase_dest()
    if dest is None:
        return False
    url, key, _config_table, eval_table = dest
    if not scores:
        print("   no score_details.json files — skipping evaluation insert", flush=True)
        return True
    eval_rows = [evaluation_row(config_id, score) for score in scores]
    print(
        f"   POST evaluations → {eval_table}  rows={len(eval_rows)}  config_id={config_id}",
        flush=True,
    )
    inserted_evals = _supabase_insert(url, key, eval_table, eval_rows)
    if inserted_evals is None:
        return False
    n = _count_inserted(inserted_evals)
    if n is not None:
        print(f"   inserted {n} evaluation row(s)", flush=True)
    return True


def _patch_config_status(config_id: str, status: str) -> bool:
    dest = _supabase_dest()
    if dest is None:
        return False
    url, key, config_table, _eval_table = dest
    endpoint = (
        f"{url}/rest/v1/{urllib.parse.quote(config_table, safe='')}"
        f"?id=eq.{urllib.parse.quote(str(config_id), safe='')}"
    )
    body = json.dumps({"status": status}, default=str).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="PATCH",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f"   PATCH {config_table} status={status}  HTTP {resp.status}", flush=True)
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        print(f"   WARNING: could not PATCH status={status}: HTTP {e.code}", flush=True)
        return False
    except (TimeoutError, urllib.error.URLError, OSError) as e:
        reason = getattr(e, "reason", e)
        print(f"   WARNING: could not PATCH status: {reason}", flush=True)
        return False


def evaluation_row(config_id: str, score: Dict[str, Any]) -> Dict[str, Any]:
    metrics = _score_metrics(score)
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


def _supabase_dest() -> Optional[Tuple[str, str, str, str]]:
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (
        os.environ.get("SUPABASE_KEY")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_ANON_KEY")
        or ""
    ).strip()
    config_table = (os.environ.get("SUPABASE_CONFIG_TABLE") or DEFAULT_CONFIG_TABLE).strip()
    eval_table = (os.environ.get("SUPABASE_EVAL_TABLE") or DEFAULT_EVAL_TABLE).strip()
    if not url or not key:
        print(
            "ERROR: set SUPABASE_URL and SUPABASE_KEY in .env "
            "(project URL + service-role key). Do not paste those values here.",
            flush=True,
        )
        return None
    if not url.startswith("https://"):
        print("ERROR: SUPABASE_URL must be an https:// project URL.", flush=True)
        return None
    if not TABLE_NAME_RE.match(config_table):
        print(f"ERROR: invalid SUPABASE_CONFIG_TABLE name: {config_table!r}", flush=True)
        return None
    if not TABLE_NAME_RE.match(eval_table):
        print(f"ERROR: invalid SUPABASE_EVAL_TABLE name: {eval_table!r}", flush=True)
        return None
    return url, key, config_table, eval_table


def _supabase_insert(url: str, key: str, table: str, payload: Any) -> Optional[Any]:
    body = json.dumps(payload, default=str).encode("utf-8")
    endpoint = f"{url}/rest/v1/{urllib.parse.quote(table, safe='')}"
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Prefer": "return=representation",
        },
    )
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                print(f"   HTTP {resp.status}  {table}  {len(body)} bytes", flush=True)
                if not (200 <= resp.status < 300):
                    preview = raw[:500] + ("..." if len(raw) > 500 else "")
                    if preview:
                        print(f"   response: {preview}", flush=True)
                    return None
                try:
                    return json.loads(raw) if raw else []
                except ValueError:
                    print("   ERROR: insert response was not JSON", flush=True)
                    return None
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")[:500]
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < 3:
                print(
                    f"   WARNING: HTTP {e.code} posting {table}; attempt {attempt}/3",
                    flush=True,
                )
                time.sleep(5 * attempt)
                continue
            print(f"   ERROR: HTTP {e.code} {e.reason}  table={table}", flush=True)
            if err_body:
                print(f"   {err_body}", flush=True)
            if e.code == 404:
                print(
                    f"   Hint: create table {table} in the Supabase SQL editor.",
                    flush=True,
                )
            if e.code in (401, 403):
                print(
                    "   Hint: use the service-role key in SUPABASE_KEY, or add an "
                    "INSERT policy on the table. Do not paste the key here.",
                    flush=True,
                )
            return None
        except (TimeoutError, urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            print(
                f"   WARNING: post {table} failed ({reason}); attempt {attempt}/3",
                flush=True,
            )
            if attempt < 3:
                time.sleep(5 * attempt)
                continue
            print(f"   ERROR: could not reach Supabase: {reason}", flush=True)
            return None
    return None


def _inserted_id(inserted: Any) -> Optional[str]:
    row = None
    if isinstance(inserted, list) and inserted and isinstance(inserted[0], dict):
        row = inserted[0]
    elif isinstance(inserted, dict):
        row = inserted
    if not row:
        return None
    value = row.get("id")
    return str(value) if value else None


def _count_inserted(inserted: Any) -> Optional[int]:
    if isinstance(inserted, list):
        return len(inserted)
    if isinstance(inserted, dict):
        return 1
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        if abs(value) >= 1 and value == int(value):
            return str(int(value))
        return f"{value:.6f}"
    return str(value)


def _format_conf_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_conf_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


if __name__ == "__main__":
    sys.exit(main())
