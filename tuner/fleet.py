"""24/7 tuner: several tests at once, a few VPS per test.

  python -m tuner.fleet

Workers are split into groups (default 2). Each group is one test: one
Optuna setting per VPS in the group. All groups run at the same time.
When a group finishes, those VPS get the next settings immediately.
The other groups are not held back.

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
    groups = _groups(workers, args.batch)
    if not groups:
        return 2
    labels = ["[" + ",".join(group) + "]" for group in groups]
    print(
        f"   fleet workers={','.join(workers)}  tests={len(groups)}  "
        f"vps_per_test={args.batch}  rounds={args.rounds}  poll={args.poll}s",
        flush=True,
    )
    print(f"   parallel tests: {' '.join(labels)}", flush=True)
    print("   a test starts again when its own VPS finish", flush=True)
    print("   GATK VPS: set WORKER_ID and run python main.py", flush=True)
    print("   stop with Ctrl+C", flush=True)
    state: Dict[str, Any] = {"since_agent": None, "box": None, "study": None, "study_name": None}
    try:
        while True:
            rc = run_fleet(args, groups, state)
            if rc == 2:
                print("   setup error — retrying in 60s", flush=True)
                time.sleep(60)
                continue
            if rc != 0:
                print(f"   fleet returned {rc} — next cycle anyway", flush=True)
    except KeyboardInterrupt:
        print("\n   tuner stopped", flush=True)
        return 0


def run_fleet(
    args: argparse.Namespace,
    groups: List[List[str]],
    state: Dict[str, Any],
) -> int:
    """Fill idle groups, then return after one group finishes."""
    workers = [worker_id for group in groups for worker_id in group]
    open_jobs = list_jobs_for_workers(workers, ["pending", "running"])
    if open_jobs is None:
        return 2

    busy: Dict[str, List[Dict[str, Any]]] = {}
    idle: List[List[str]] = []
    for group in groups:
        jobs = _jobs_for_group(open_jobs, group)
        if jobs:
            busy[_group_key(group)] = jobs
        else:
            idle.append(group)

    if idle:
        if not _ready_study(args, state):
            return 2
        for index, group in enumerate(groups, 1):
            if group not in idle:
                continue
            jobs = _assign_group(args, state, group, index, len(groups))
            if not jobs:
                return 2
            busy[_group_key(group)] = jobs

    if not busy:
        print("   no groups to run", flush=True)
        return 1
    _wait_first_group(busy, poll=args.poll, state=state)
    return 0


def _ready_study(args: argparse.Namespace, state: Dict[str, Any]) -> bool:
    box = state.get("box") if isinstance(state.get("box"), dict) else None
    try:
        limit = int((box or {}).get("n_trials") or 8)
    except (TypeError, ValueError):
        limit = 8
    since = state.get("since_agent")
    if box is None or since is None or int(since) >= limit:
        rc = agent_main([])
        if rc != 0:
            print("   agent did not write a search box", flush=True)
            return False
        try:
            box = load_search_box(Path(args.box))
        except (OSError, ValueError) as e:
            print(f"ERROR: search box: {e}", flush=True)
            return False
        state["box"] = box
        state["since_agent"] = 0
        state["study"] = None
        state["study_name"] = None

    study_name = args.study or default_study_name(box["search_category"])
    if state.get("study") is not None and state.get("study_name") == study_name:
        _settle_running(state["study"], study_name, state["TrialState"])
        return True

    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        print("ERROR: pip install -r requirements.txt (optuna)", flush=True)
        return False

    box["study_name"] = study_name
    storage = (os.environ.get("OPTUNA_STORAGE") or DEFAULT_STORAGE).strip()
    distributions = optuna_distributions(box["space"], optuna)
    study = open_study(optuna, study_name, storage, reset=False)
    history = fetch_config_scores(category=box["search_category"], scored_only=True)
    if history is None:
        return False
    added, skipped = import_rows(
        study, history, box["search_category"], distributions, optuna, keys=list(box["space"])
    )
    print(f"   warm-start imported={added} skipped={skipped}", flush=True)
    _settle_running(study, study_name, TrialState)
    state["study"] = study
    state["study_name"] = study_name
    state["distributions"] = distributions
    state["TrialState"] = TrialState
    return True


def _assign_group(
    args: argparse.Namespace,
    state: Dict[str, Any],
    group: List[str],
    index: int,
    group_count: int,
) -> List[Dict[str, Any]]:
    box = state["box"]
    study = state["study"]
    distributions = state["distributions"]
    trial_state = state["TrialState"]
    category = box["search_category"]
    batch_id = str(uuid.uuid4())
    jobs: List[Dict[str, Any]] = []
    print(f"   test {index}/{group_count} vps={','.join(group)}", flush=True)
    for worker_id in group:
        try:
            trial = study.ask(fixed_distributions=distributions)
        except TypeError:
            trial = study.ask(distributions)
        params = dict(trial.params)
        experiment = f"optuna-{category}-t{trial.number}"
        print(f"   worker={worker_id} trial={trial.number} {params}", flush=True)
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
            _tell_state(study, trial.number, trial_state.FAIL)
            continue
        jobs.append({"id": config_id, "worker_id": worker_id, "optuna_trial_number": trial.number})
        print(f"   pending config_id={config_id} → worker {worker_id}", flush=True)
    return jobs


def _wait_first_group(
    groups: Dict[str, List[Dict[str, Any]]],
    poll: int,
    state: Dict[str, Any],
) -> None:
    """Block until one group is fully finished. Tell each VPS as it finishes."""
    print(f"   {len(groups)} test(s) in flight", flush=True)
    while groups:
        finished_keys: List[str] = []
        for key, jobs in list(groups.items()):
            still: List[Dict[str, Any]] = []
            for job in jobs:
                config_id = str(job.get("id") or job.get("config_id") or "")
                row = fetch_score_row(config_id) if config_id else None
                status = str((row or {}).get("status") or "")
                if status in TERMINAL:
                    n_scored = (row or {}).get("n_scored")
                    score = (row or {}).get("avg_combined_final")
                    print(
                        f"   worker={job.get('worker_id')} {status} "
                        f"n_scored={n_scored} avg={score}",
                        flush=True,
                    )
                    _tell_jobs([job])
                    state["since_agent"] = int(state.get("since_agent") or 0) + 1
                    continue
                still.append(job)
            if still:
                waiting = ",".join(str(job.get("worker_id")) for job in still)
                print(f"   waiting test vps={waiting}", flush=True)
                groups[key] = still
            else:
                print(f"   test vps={key} finished — next settings for this pair only", flush=True)
                finished_keys.append(key)
        for key in finished_keys:
            groups.pop(key, None)
        if finished_keys:
            return
        time.sleep(max(1, poll))


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


def _groups(workers: List[str], size: int) -> List[List[str]]:
    if size < 1:
        print("ERROR: vps per test must be at least 1.", flush=True)
        return []
    if not workers or len(workers) % size != 0:
        print(
            f"ERROR: {len(workers)} worker(s) cannot be split into tests of {size}. "
            "TUNER_WORKERS length must be a multiple of FLEET_GROUP.",
            flush=True,
        )
        return []
    return [workers[i : i + size] for i in range(0, len(workers), size)]


def _group_key(group: List[str]) -> str:
    return ",".join(group)


def _jobs_for_group(open_jobs: List[Dict[str, Any]], group: List[str]) -> List[Dict[str, Any]]:
    wanted = set(group)
    return [job for job in open_jobs if str(job.get("worker_id") or "") in wanted]


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run parallel GATK tests until Ctrl+C.")
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
        default=int(os.environ.get("FLEET_GROUP") or 2),
        help="VPS per test. Other tests keep running when this test finishes (default 2).",
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
