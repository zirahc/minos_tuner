"""24/7 tuner: a stack of trials, one per free GATK VPS.

  python -m tuner.fleet

The agent chooses one category to vary. Each trial writes every catalog
parameter: the best scored config, with that category's new values on top.
Each trial first scores FLEET_ROUNDS practice rounds.
The top quarter of that stack then scores the next FLEET_ROUNDS
folders. Optuna is told after that check. A free VPS takes the next
trial immediately. When the stack and its checks are done, the agent
builds the next stack.

Stop with Ctrl+C. GATK machines run: python main.py
A machine posts its WORKER_ID while main.py is running. The fleet
gives the next trial to any machine that posted recently.
"""
from __future__ import annotations

import argparse
import math
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

from tuner.adapter import (
    DEFAULT_STORAGE,
    catalog_categorical_distributions,
    import_rows,
    open_study,
    reconcile_categorical_study,
)
from tuner.agent import main as agent_main
from tuner.jobs import (
    assign_queued,
    config_for_trial,
    fetch_score_row,
    insert_pending_config,
    list_online_workers,
    list_open_jobs,
    list_queued,
    requeue_confirmation,
)
from tuner.search_box import MAX_TRIALS, load_search_box, optuna_distributions
from tuner.spaces import COMBO_CATEGORY, best_base_updates, categories_touched, default_study_name
from tuner.steps import MIN_GAIN
from tuner.supabase_scores import fetch_config_scores

DEFAULT_BOX = TUNER_ROOT / "search_box.json"
TERMINAL = ("scored", "failed")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.batch < 1:
        print("ERROR: FLEET_BATCH must be at least 1.", flush=True)
        return 2
    print(
        f"   fleet reads online workers from gatk_workers  stack={args.batch}  "
        f"rounds={args.rounds}  poll={args.poll}s",
        flush=True,
    )
    print("   a machine that runs main.py is given the next trial", flush=True)
    print(
        f"   every trial scores {args.rounds} round(s); the top quarter then "
        f"scores the next {args.rounds}",
        flush=True,
    )
    print("   the agent runs again after those checks finish", flush=True)
    print("   GATK VPS: set WORKER_ID and run python main.py", flush=True)
    print("   stop with Ctrl+C", flush=True)
    while True:
        try:
            rc = run_once(args)
        except KeyboardInterrupt:
            print("\n   tuner stopped", flush=True)
            return 0
        except Exception as e:
            print(f"   ERROR: {type(e).__name__}: {e} — retrying in 60s", flush=True)
            time.sleep(60)
            continue
        if rc == 2:
            print("   setup error — retrying in 60s", flush=True)
            time.sleep(60)
            continue
        if rc != 0:
            print(f"   stack returned {rc} — next cycle anyway", flush=True)


def run_once(args: argparse.Namespace) -> int:
    queued = list_queued()
    inflight = list_open_jobs()
    if queued is None or inflight is None:
        return 2
    if not queued and not inflight:
        if not _build_stack(args):
            return 2
        queued = list_queued()
        inflight = list_open_jobs()
        if queued is None or inflight is None:
            return 2
    watching = {str(job.get("id")): job for job in inflight if job.get("id")}
    screened: Dict[str, Dict[str, Any]] = {}
    seen_online: Optional[List[str]] = None
    print(
        f"   stack={len(queued)}  running={len(watching)}",
        flush=True,
    )
    while True:
        workers = list_online_workers(max(180, args.poll * 6))
        if workers is None:
            return 2
        if workers != seen_online:
            print(
                f"   online workers={','.join(workers) if workers else '-'}",
                flush=True,
            )
            seen_online = workers
        busy = {
            str(job.get("worker_id"))
            for job in watching.values()
            if job.get("worker_id")
        }
        free = [worker_id for worker_id in workers if worker_id not in busy]
        for worker_id in free:
            if not queued:
                break
            row = queued.pop(0)
            config_id = str(row.get("id") or "")
            assigned = assign_queued(config_id, worker_id) if config_id else None
            if not assigned:
                print(f"   could not assign {config_id} to worker {worker_id}", flush=True)
                continue
            watching[config_id] = assigned
            print(
                f"   worker={worker_id} trial={assigned.get('optuna_trial_number')} "
                f"config_id={config_id}  stack_left={len(queued)}",
                flush=True,
            )
        still: Dict[str, Dict[str, Any]] = {}
        for config_id, job in watching.items():
            row = fetch_score_row(config_id)
            status = str((row or {}).get("status") or "")
            if status in TERMINAL:
                score = (row or {}).get("avg_combined_final")
                offset = int(job.get("rounds_offset") or (row or {}).get("rounds_offset") or 0)
                print(
                    f"   worker={job.get('worker_id')} {status} "
                    f"n_scored={(row or {}).get('n_scored')} avg={score} offset={offset}",
                    flush=True,
                )
                if status == "scored" and offset <= 0:
                    screened[config_id] = job
                    continue
                _tell_jobs([job])
                continue
            still[config_id] = job
        watching = still
        if not watching and not queued:
            if screened:
                queued = _confirm_high_scores(screened, args.rounds)
                screened = {}
                if queued:
                    print(
                        f"   confirming {len(queued)} high score(s) on the next "
                        f"{args.rounds} round(s)",
                        flush=True,
                    )
                    continue
            print("   stack finished — agent will set the next trials", flush=True)
            return 0
        waiting = ",".join(str(job.get("worker_id")) for job in watching.values())
        if queued and not free:
            print(
                f"   stack_left={len(queued)}  waiting workers={waiting or '-'}  "
                "no free machine online",
                flush=True,
            )
        else:
            print(
                f"   stack_left={len(queued)}  waiting workers={waiting or '-'}",
                flush=True,
            )
        time.sleep(max(1, args.poll))
        refreshed = list_queued()
        if refreshed is None:
            return 2
        assigned_ids = set(watching)
        queued = [row for row in refreshed if str(row.get("id") or "") not in assigned_ids]


def _confirm_high_scores(
    screened: Dict[str, Dict[str, Any]],
    rounds: int,
) -> List[Dict[str, Any]]:
    """Confirm only a screen that beats the current best by more than MIN_GAIN.

    A gap of 0.01 or less is told from the first rounds and the category ends.
    """
    ranked: List[tuple] = []
    for config_id, job in screened.items():
        row = fetch_score_row(config_id) or {}
        try:
            score = float(row.get("avg_combined_final"))
        except (TypeError, ValueError):
            _tell_jobs([job])
            continue
        ranked.append((score, config_id, job))
    if not ranked:
        return []
    ranked.sort(key=lambda item: item[0], reverse=True)
    baseline = _baseline_score(set(screened))
    best = ranked[0][0]
    if baseline is not None and best <= baseline + MIN_GAIN:
        print(
            f"   screen best={best:.4f} is within {MIN_GAIN:.2f} of {baseline:.4f}; "
            "not continuing this experiment",
            flush=True,
        )
        for _score, _config_id, job in ranked:
            _tell_jobs([job])
        return []
    raised = [
        item for item in ranked
        if baseline is None or item[0] > baseline + MIN_GAIN
    ]
    keep = max(1, math.ceil(len(ranked) * 0.25))
    raised = raised[:keep]
    base_txt = f"{baseline:.4f}" if baseline is not None else "none"
    print(
        f"   screen best={best:.4f} beats {base_txt} by more than {MIN_GAIN:.2f}; "
        f"confirming {len(raised)} trial(s)",
        flush=True,
    )
    queued: List[Dict[str, Any]] = []
    confirm_ids = {config_id for _score, config_id, _job in raised}
    for score, config_id, job in ranked:
        if config_id not in confirm_ids:
            _tell_jobs([job])
            continue
        updated = requeue_confirmation(config_id, rounds)
        if not updated:
            print(f"   could not requeue {config_id}; telling the first screen", flush=True)
            _tell_jobs([job])
            continue
        print(f"   confirm trial={job.get('optuna_trial_number')} avg={score:.4f}", flush=True)
        queued.append(updated)
    return queued


def _baseline_score(exclude: set) -> Optional[float]:
    """Best avg_combined_final outside the trials just screened."""
    rows = fetch_config_scores(
        category=None,
        scored_only=True,
        limit=30,
        order="avg_combined_final.desc",
    )
    if not rows:
        return None
    for row in rows:
        if str(row.get("config_id") or "") in exclude:
            continue
        try:
            return float(row.get("avg_combined_final"))
        except (TypeError, ValueError):
            continue
    return None


def _build_stack(args: argparse.Namespace) -> bool:
    rc = agent_main([])
    if rc != 0:
        print("   agent did not write a search box", flush=True)
        return False
    try:
        box = load_search_box(Path(args.box))
    except (OSError, ValueError) as e:
        print(f"ERROR: search box: {e}", flush=True)
        return False
    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        print("ERROR: pip install -r requirements.txt (optuna)", flush=True)
        return False

    if box.get("run_experiment") is False:
        print(
            f"   no category beat the best by more than {MIN_GAIN:.2f}; "
            "not starting another flat experiment",
            flush=True,
        )
        return False
    category = box["search_category"]
    study_name = args.study or default_study_name(category)
    box["study_name"] = study_name
    storage = (os.environ.get("OPTUNA_STORAGE") or DEFAULT_STORAGE).strip()
    distributions = catalog_categorical_distributions(
        category, optuna_distributions(box["space"], optuna), optuna
    )
    study = open_study(optuna, study_name, storage, reset=False)
    study = reconcile_categorical_study(
        study, optuna, study_name, storage, distributions
    )
    history = fetch_config_scores(category=None, scored_only=True)
    if history is None:
        return False
    if category == COMBO_CATEGORY:
        print(
            "   combo: varying "
            + ", ".join(box["space"])
            + " across categories; other parameters stay at the best config",
            flush=True,
        )
        added, skipped = 0, 0
    else:
        matched = [row for row in history if category in categories_touched(row)]
        print(
            f"   {category} experiments={len(matched)} "
            f"(counted from the parameters each row set)",
            flush=True,
        )
        added, skipped = import_rows(
            study, matched, category, distributions, optuna, keys=list(box["space"])
        )
    print(f"   warm-start imported={added} skipped={skipped}", flush=True)
    _settle_running(study, study_name, TrialState)

    budget = max(1, min(MAX_TRIALS, int(box.get("n_trials") or 1)))
    distinct = _fixed_choice_count(box["space"])
    target = budget if distinct is None else min(budget, distinct)
    if distinct is not None and distinct < budget:
        print(
            f"   {category} has {distinct} setting(s); stack uses {target} "
            "so the same option is not scored twice",
            flush=True,
        )
        for params in _enumerate_settings(box["space"]):
            try:
                study.enqueue_trial(params)
            except Exception as e:  # noqa: BLE001
                print(f"   WARNING: could not enqueue {params}: {e}", flush=True)
                break
    else:
        print(f"   stack target={target}", flush=True)

    best_rows = fetch_config_scores(
        category=None,
        scored_only=True,
        limit=1,
        order="avg_combined_final.desc",
    )
    base, best_row = best_base_updates(best_rows if best_rows else history)
    if best_row is None:
        print(f"   base config is the catalog default ({len(base)} parameters)", flush=True)
    else:
        print(
            f"   base config {best_row.get('experiment')} "
            f"avg_combined_final={float(best_row.get('avg_combined_final')):.4f} "
            f"({len(base)} parameters); this stack varies {category}",
            flush=True,
        )

    batch_id = str(uuid.uuid4())
    placed = 0
    seen = set()
    attempts = 0
    while placed < target and attempts < target * 4:
        attempts += 1
        try:
            trial = study.ask(fixed_distributions=distributions)
        except TypeError:
            trial = study.ask(distributions)
        except ValueError as e:
            if "dynamic value space" not in str(e):
                raise
            print(f"   ERROR: {e}", flush=True)
            return False
        params = dict(trial.params)
        signature = tuple(sorted((key, str(value)) for key, value in params.items()))
        if signature in seen:
            _tell_state(study, trial.number, TrialState.FAIL)
            continue
        seen.add(signature)
        updates = dict(base)
        updates.update(params)
        experiment = f"optuna-{category}-t{trial.number}"
        print(
            f"   stack [{placed + 1}/{target}] trial={trial.number} "
            f"vary={params}  full_keys={len(updates)}",
            flush=True,
        )
        config_id = insert_pending_config(
            experiment=experiment,
            updates=updates,
            box=box,
            trial_number=int(trial.number),
            batch_id=batch_id,
            rounds_target=args.rounds,
            status="queued",
        )
        if not config_id:
            print(
                "   ERROR: could not insert job. Re-run gatk_tuning.sql so "
                "worker_id, batch_id, rounds_target exist.",
                flush=True,
            )
            _tell_state(study, trial.number, TrialState.FAIL)
            continue
        placed += 1
    if placed == 0:
        return False
    print(f"   placed {placed} trial(s) on the stack", flush=True)
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
        if config and str(config.get("status") or "") in ("queued", "pending", "running"):
            continue
        if not config:
            _tell_state(study, trial.number, trial_state.FAIL)
            continue
        score_row = fetch_score_row(str(config.get("id")))
        _tell_score(study, int(trial.number), score_row or config, trial_state)


def _tell_score(study: Any, number: int, row: Dict[str, Any], trial_state: Any) -> None:
    status = str(row.get("status") or "")
    if status in ("queued", "pending", "running"):
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


def _fixed_choice_count(space: Dict[str, Any]) -> Optional[int]:
    """Choice product when every parameter is categorical. None if a numeric range remains."""
    total = 1
    for spec in space.values():
        if not isinstance(spec, dict):
            continue
        kind = spec.get("type")
        choices = list(spec.get("choices") or [])
        numeric = kind in ("int", "float") and not (kind == "float" and spec.get("log") and choices)
        if numeric:
            try:
                if float(spec["high"]) > float(spec["low"]):
                    return None
            except (KeyError, TypeError, ValueError):
                return None
        if len(choices) > 1:
            total *= len(choices)
    return max(1, total)


def _grid_values(spec: Dict[str, Any]) -> Optional[List[Any]]:
    """Coarse values for one parameter. None when the range is still continuous."""
    kind = spec.get("type")
    choices = spec.get("choices")
    if kind == "categorical" or choices:
        values = list(choices or [])
        return values or None
    if kind not in ("int", "float"):
        return None
    try:
        low = float(spec["low"])
        high = float(spec["high"])
        step = float(spec["step"])
    except (KeyError, TypeError, ValueError):
        if kind == "int":
            try:
                return list(range(int(spec["low"]), int(spec["high"]) + 1))
            except (KeyError, TypeError, ValueError):
                return None
        return None
    if step <= 0 or high < low:
        return None
    values = []
    n = 0
    while n < 10000:
        value = low + n * step
        if value > high + 1e-9:
            break
        values.append(int(round(value)) if kind == "int" else value)
        n += 1
    return values or None


def _distinct_settings(space: Dict[str, Any]) -> Optional[int]:
    """Number of coarse settings. None when a continuous float range remains."""
    total = 1
    for spec in space.values():
        if not isinstance(spec, dict):
            continue
        values = _grid_values(spec)
        if values is None:
            return None
        if len(values) > 1:
            total *= len(values)
        if total > 100000:
            return None
    return max(1, total)


def _enumerate_settings(space: Dict[str, Any]) -> List[Dict[str, Any]]:
    dims: List[tuple] = []
    for key, spec in space.items():
        if not isinstance(spec, dict):
            continue
        values = _grid_values(spec)
        if values is None:
            return []
        dims.append((key, values))
    if not dims:
        return []
    rows: List[Dict[str, Any]] = [{}]
    for key, values in dims:
        rows = [{**row, key: value} for row in rows for value in values]
    return rows


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the GATK tuning stack until Ctrl+C.")
    p.add_argument("--box", default=str(DEFAULT_BOX))
    p.add_argument("--study", default=None)
    p.add_argument(
        "--workers",
        default=os.environ.get("TUNER_WORKERS") or "",
        help="Ignored. Online machines are read from gatk_workers.",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=int(os.environ.get("FLEET_BATCH") or 8),
        help="Trials placed on the stack before the agent runs again.",
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
