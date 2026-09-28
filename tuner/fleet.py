"""24/7 tuner: 4 settings, one per GATK VPS, then the next batch.

  python -m tuner.fleet

Each cycle asks the agent for a search box, asks Optuna for 4 settings,
and writes one pending gatk_configs row per WORKER id. It waits until
every VPS has finished, records avg_combined_final, then starts again.

Stop with Ctrl+C. GATK machines run: python main.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
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

from tuner.adapter import DEFAULT_STORAGE, import_rows, open_study
from tuner.agent import main as agent_main
from tuner.jobs import (
    config_for_trial,
    fetch_score_row,
    insert_pending_config,
    list_jobs_for_workers,
)
from tuner.search_box import load_search_box, optuna_distributions
from tuner.spaces import default_study_name
from tuner.supabase_scores import fetch_config_scores

DEFAULT_BOX = TUNER_ROOT / "search_box.json"
TERMINAL = ("scored", "failed")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    workers = _workers(args.workers)
    if len(workers) != args.batch:
        print(
            f"ERROR: TUNER_WORKERS has {len(workers)} id(s) but the batch size is {args.batch}.",
            flush=True,
        )
        return 2
    print(
        f"   fleet workers={','.join(workers)}  batch={args.batch}  "
        f"rounds={args.rounds}  poll={args.poll}s",
        flush=True,
    )
    print("   GATK VPS: set WORKER_ID and run python main.py", flush=True)
    print("   stop with Ctrl+C", flush=True)
    try:
        while True:
            rc = run_once(args, workers)
            if rc == 2:
                print("   setup error — retrying in 60s", flush=True)
                time.sleep(60)
                continue
            if rc != 0:
                print(f"   batch returned {rc} — next cycle anyway", flush=True)
    except KeyboardInterrupt:
        print("\n   tuner stopped", flush=True)
        return 0


def run_once(args: argparse.Namespace, workers: List[str]) -> int:
    open_jobs = list_jobs_for_workers(workers, ["pending", "running"])
    if open_jobs is None:
        return 2
    if open_jobs:
        print(f"   {len(open_jobs)} job(s) still in flight — waiting, not asking for new settings", flush=True)
        if not _wait_jobs(open_jobs, poll=args.poll):
            return 2
        return _tell_jobs(open_jobs)

    rc = agent_main([])
    if rc != 0:
        print("   agent did not write a search box", flush=True)
        return 2
    try:
        box = load_search_box(Path(args.box))
    except (OSError, ValueError) as e:
        print(f"ERROR: search box: {e}", flush=True)
        return 2

    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        print("ERROR: pip install -r requirements.txt (optuna)", flush=True)
        return 2

    category = box["search_category"]
    study_name = args.study or default_study_name(category)
    box["study_name"] = study_name
    storage = (os.environ.get("OPTUNA_STORAGE") or DEFAULT_STORAGE).strip()
    distributions = optuna_distributions(box["space"], optuna)
    study = open_study(optuna, study_name, storage, reset=False)
    history = fetch_config_scores(category=category, scored_only=True)
    if history is None:
        return 2
    added, skipped = import_rows(
        study, history, category, distributions, optuna, keys=list(box["space"])
    )
    print(f"   warm-start imported={added} skipped={skipped}", flush=True)
    _settle_running(study, study_name, TrialState)

    batch_id = str(uuid.uuid4())
    jobs: List[Dict[str, Any]] = []
    for i, worker_id in enumerate(workers, 1):
        try:
            trial = study.ask(fixed_distributions=distributions)
        except TypeError:
            trial = study.ask(distributions)
        params = dict(trial.params)
        experiment = f"optuna-{category}-t{trial.number}"
        print(f"   [{i}/{args.batch}] worker={worker_id} trial={trial.number} {params}", flush=True)
        config_id = insert_pending_config(
            experiment=experiment,
            updates=params,
            box=box,
            trial_number=int(trial.number),
            worker_id=worker_id,
            batch_id=batch_id,
            rounds_target=args.rounds,
        )
        if not config_id:
            print(
                "   ERROR: could not insert job. Re-run gatk_tuning.sql so "
                "worker_id, batch_id, rounds_target exist.",
                flush=True,
            )
            _tell_state(study, trial.number, TrialState.FAIL)
            continue
        jobs.append({"id": config_id, "worker_id": worker_id, "optuna_trial_number": trial.number})
        print(f"   pending config_id={config_id} → worker {worker_id}", flush=True)

    if not jobs:
        return 1
    if not _wait_jobs(jobs, poll=args.poll):
        return 2
    return _tell_jobs(jobs)


def _wait_jobs(jobs: List[Dict[str, Any]], poll: int) -> bool:
    pending = {str(job.get("id") or job.get("config_id")): job for job in jobs}
    pending.pop("", None)
    print(f"   waiting for {len(pending)} VPS to finish", flush=True)
    while pending:
        still: Dict[str, Dict[str, Any]] = {}
        for config_id, job in pending.items():
            row = fetch_score_row(config_id)
            status = str((row or {}).get("status") or "")
            if status in TERMINAL:
                n_scored = (row or {}).get("n_scored")
                score = (row or {}).get("avg_combined_final")
                print(
                    f"   worker={job.get('worker_id')} {status} "
                    f"n_scored={n_scored} avg={score}",
                    flush=True,
                )
                continue
            still[config_id] = job
            print(
                f"   waiting worker={job.get('worker_id')} status={status or 'pending'}",
                flush=True,
            )
        pending = still
        if pending:
            time.sleep(max(1, poll))
    return True


def _tell_jobs(jobs: List[Dict[str, Any]]) -> int:
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        print("ERROR: pip install -r requirements.txt (optuna)", flush=True)
        return 2

    told: Dict[str, Any] = {}
    for job in jobs:
        config_id = str(job.get("id") or job.get("config_id") or "")
        row = fetch_score_row(config_id) if config_id else None
        if not row:
            continue
        study_name = str(row.get("study_name") or "")
        if not study_name:
            continue
        told.setdefault(study_name, []).append(row)

    storage = (os.environ.get("OPTUNA_STORAGE") or DEFAULT_STORAGE).strip()
    for study_name, rows in told.items():
        study = open_study(optuna, study_name, storage, reset=False)
        for row in rows:
            number = row.get("optuna_trial_number")
            if number is None:
                continue
            _tell_score(study, int(number), row, TrialState)
        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if completed:
            print(
                f"   best avg_combined_final={study.best_value:.6f}  "
                f"params={study.best_params}",
                flush=True,
            )
    return 0


def _settle_running(study: Any, study_name: str, trial_state: Any) -> None:
    for trial in study.get_trials(deepcopy=False):
        if trial.state not in (trial_state.RUNNING, trial_state.WAITING):
            continue
        config = config_for_trial(study_name, int(trial.number))
        if config and str(config.get("status") or "") in ("pending", "running"):
            continue
        if not config:
            _tell_state(study, trial.number, trial_state.FAIL)
            continue
        score_row = fetch_score_row(str(config.get("id")))
        _tell_score(study, int(trial.number), score_row or config, trial_state)


def _tell_score(study: Any, number: int, row: Dict[str, Any], trial_state: Any) -> None:
    status = str(row.get("status") or "")
    if status in ("pending", "running"):
        return
    value = row.get("avg_combined_final")
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = None
    if status == "scored" and score is not None:
        _tell_state(study, number, None, score)
        print(f"   tell trial={number} {score:.6f}", flush=True)
        return
    _tell_state(study, number, trial_state.FAIL)
    print(f"   tell trial={number} FAIL status={status or 'missing'}", flush=True)


def _tell_state(study: Any, number: int, state: Any = None, value: Optional[float] = None) -> None:
    try:
        if state is not None:
            study.tell(int(number), state=state, skip_if_finished=True)
        else:
            study.tell(int(number), value, skip_if_finished=True)
    except TypeError:
        if state is not None:
            study.tell(int(number), state=state)
        else:
            study.tell(int(number), value)
    except Exception as e:  # noqa: BLE001
        print(f"   WARNING: tell trial={number} failed: {e}", flush=True)


def _workers(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the 4-VPS GATK tuning loop until Ctrl+C.")
    p.add_argument("--box", default=str(DEFAULT_BOX))
    p.add_argument("--study", default=None)
    p.add_argument(
        "--workers",
        default=os.environ.get("TUNER_WORKERS") or "1,2,3,4",
        help="WORKER_ID values, one per GATK VPS (default 1,2,3,4).",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=int(os.environ.get("FLEET_BATCH") or 4),
        help="Settings per cycle. Must match the number of workers.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=int(os.environ.get("FLEET_ROUNDS") or 15),
        help="Practice rounds each VPS scores for one setting.",
    )
    p.add_argument("--poll", type=int, default=int(os.environ.get("FLEET_POLL_SEC") or 20))
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
