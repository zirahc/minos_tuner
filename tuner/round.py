"""Optuna ask → pending Supabase job → tell avg_combined_final.

Tuner server: python -m tuner.round
Scoring server must be running: python -m tuner.worker
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

TUNER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TUNER_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(TUNER_ROOT / ".env")
    load_dotenv(TUNER_ROOT / ".env.tuner")
except ImportError:
    pass

from tuner.adapter import (
    DEFAULT_STORAGE,
    catalog_categorical_distributions,
    import_rows,
    open_study,
    reconcile_categorical_study,
)
from tuner.jobs import fetch_score_row, insert_pending_config
from tuner.search_box import load_search_box, optuna_distributions
from tuner.spaces import default_study_name
from tuner.supabase_scores import fetch_config_scores

DEFAULT_BOX = TUNER_ROOT / "search_box.json"


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    box_path = Path(args.box)
    try:
        box = load_search_box(box_path)
    except (OSError, ValueError) as e:
        print(f"ERROR: could not load search box {box_path}: {e}", flush=True)
        return 2

    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        print("ERROR: pip install -r requirements.txt (optuna)", flush=True)
        return 2

    category = box["search_category"]
    space = box["space"]
    n_trials = int(args.n_trials or box["n_trials"])
    study_name = args.study or default_study_name(category)
    storage = (os.environ.get("OPTUNA_STORAGE") or DEFAULT_STORAGE).strip()
    distributions = catalog_categorical_distributions(
        category, optuna_distributions(space, optuna), optuna
    )
    keys = list(space)

    study = open_study(optuna, study_name, storage, reset=args.reset)
    study = reconcile_categorical_study(
        study, optuna, study_name, storage, distributions
    )
    history = fetch_config_scores(category=category, scored_only=True)
    if history is None:
        return 2
    added, skipped = import_rows(
        study, history, category, distributions, optuna, keys=keys
    )
    print(f"   warm-start imported={added} skipped={skipped}", flush=True)

    timeout = int(os.environ.get("ROUND_TRIAL_TIMEOUT_SEC") or args.timeout)
    poll = int(os.environ.get("ROUND_POLL_SEC") or args.poll)
    print(
        f"   category={category}  n_trials={n_trials}  "
        f"timeout={timeout}s  poll={poll}s",
        flush=True,
    )
    print("   scoring server must run: python -m tuner.worker", flush=True)

    for i in range(1, n_trials + 1):
        try:
            trial = study.ask(fixed_distributions=distributions)
        except TypeError:
            trial = study.ask(distributions)
        params = dict(trial.params)
        experiment = f"optuna-{category}-t{trial.number}"
        print(f"\n   [{i}/{n_trials}] ask trial={trial.number} {params}", flush=True)
        config_id = insert_pending_config(
            experiment=experiment,
            updates=params,
            box=box,
            trial_number=int(trial.number),
        )
        if not config_id:
            print("   ERROR: could not insert pending config", flush=True)
            study.tell(trial, state=TrialState.FAIL)
            continue
        print(f"   pending config_id={config_id}", flush=True)
        row = _wait_for_score(config_id, timeout=timeout, poll=poll)
        if row is None:
            print("   timeout / no score — telling FAIL", flush=True)
            study.tell(trial, state=TrialState.FAIL)
            continue
        value = row.get("avg_combined_final")
        try:
            score = float(value)
        except (TypeError, ValueError):
            print(f"   no avg_combined_final (status={row.get('status')}) — FAIL", flush=True)
            study.tell(trial, state=TrialState.FAIL)
            continue
        ok, why = _constraints_ok(row, box.get("constraints") or {})
        if not ok:
            print(f"   constraint miss: {why} (still telling real score {score:.6f})", flush=True)
        study.tell(trial, score)
        print(f"   tell {score:.6f}", flush=True)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if completed:
        print(
            f"\n   best avg_combined_final={study.best_value:.6f}  "
            f"params={study.best_params}",
            flush=True,
        )
    return 0


def _wait_for_score(config_id: str, timeout: int, poll: int) -> Optional[Dict[str, Any]]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = fetch_score_row(config_id)
        if row:
            status = str(row.get("status") or "")
            n_scored = row.get("n_scored") or 0
            if status == "failed":
                return row
            if status in ("scored", "rejected_constraint") and n_scored:
                return row
            if row.get("avg_combined_final") is not None and n_scored:
                return row
        time.sleep(max(1, poll))
        print(f"   waiting for {config_id} ...", flush=True)
    return None


def _constraints_ok(row: Dict[str, Any], constraints: Dict[str, Any]) -> tuple:
    checks = (
        ("min_avg_f1_snp", "avg_f1_snp"),
        ("min_avg_f1_indel", "avg_f1_indel"),
        ("min_avg_combined_final", "avg_combined_final"),
    )
    for ckey, rkey in checks:
        floor = constraints.get(ckey)
        if floor is None:
            continue
        try:
            value = float(row.get(rkey))
            limit = float(floor)
        except (TypeError, ValueError):
            return False, f"{rkey} missing"
        if value < limit:
            return False, f"{rkey}={value:.4f} < {limit:.4f}"
    return True, ""


def _parse_args(argv: Optional[list]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ask Optuna trials, enqueue pending configs, tell scores.",
    )
    p.add_argument("--box", default=str(DEFAULT_BOX), help="search_box.json from the agent")
    p.add_argument("--study", default=None)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--timeout", type=int, default=7200, help="seconds to wait per trial")
    p.add_argument("--poll", type=int, default=20, help="seconds between score polls")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
