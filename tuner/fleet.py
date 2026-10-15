"""24/7 tuner: a stack of trials, one per free GATK VPS.

  python -m tuner.fleet

experiments.json is checked first. Those experiments run before the agent.
The agent chooses one category to vary when that list is empty or already
scored. Each trial writes every catalog
parameter: the best scored config, with that category's new values on top.
Each trial scores every practice round in the folder. Optuna is told
when that score is in. A free VPS takes the next trial immediately.
When the stack is done, the agent builds the next stack.

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
    param_signature,
    reconcile_categorical_study,
    remember_scores,
)
from tuner.agent import main as agent_main
from tuner.experiments import load_experiments
from tuner.jobs import (
    assign_queued,
    config_for_trial,
    fetch_score_row,
    insert_pending_config,
    list_online_workers,
    list_open_jobs,
    list_queued,
    release_offline_job,
)
from tuner.search_box import MAX_TRIALS, load_search_box, optuna_distributions
from tuner.spaces import (
    best_base_updates,
    full_params_from_row,
    categories_touched,
    category_of,
    coerce_param,
    default_study_name,
    held_at_default,
    is_agent_experiment,
    spec_of,
)
from tuner.steps import MIN_GAIN, _row_informs_param
from tuner.supabase_scores import fetch_config_scores

DEFAULT_BOX = TUNER_ROOT / "search_box.json"
DEFAULT_EXPERIMENTS = TUNER_ROOT / "experiments.json"
TERMINAL = ("scored", "failed")


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.batch < 1:
        print("ERROR: FLEET_BATCH must be at least 1.", flush=True)
        return 2
    print(
        f"   fleet reads online workers from gatk_workers  stack={args.batch}  "
        f"poll={args.poll}s",
        flush=True,
    )
    print("   a machine that runs main.py is given the next trial", flush=True)
    print("   every trial scores every practice round in the folder", flush=True)
    print("   experiments.json runs before the agent when it lists experiments", flush=True)
    print("   the agent runs again after the stack finishes", flush=True)
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
        online = set(workers)
        for config_id, job in list(watching.items()):
            holder = str(job.get("worker_id") or "")
            if not holder or holder in online:
                continue
            released = release_offline_job(config_id)
            if not released:
                continue
            watching.pop(config_id, None)
            queued.append(released)
            print(
                f"   worker={holder} offline; trial "
                f"{released.get('optuna_trial_number')} returned to the stack",
                flush=True,
            )
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
                print(
                    f"   worker={job.get('worker_id')} {status} "
                    f"n_scored={(row or {}).get('n_scored')} avg={score}",
                    flush=True,
                )
                _tell_jobs([job])
                continue
            still[config_id] = job
        watching = still
        if not watching and not queued:
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


def _build_stack(args: argparse.Namespace) -> bool:
    try:
        listed = load_experiments(Path(args.experiments))
    except ValueError as e:
        print(f"ERROR: {e}", flush=True)
        return False
    for item in listed:
        print(
            f"   file experiment {item['name']}: {item['hypothesis']}",
            flush=True,
        )
        if item["kind"] == "fixed":
            outcome = _enqueue_fixed(args, item)
        else:
            outcome = _enqueue_box(args, item["box"])
        if outcome is True:
            return True
        if outcome is False:
            return False
        print(
            f"   {item['name']} is already scored on the current base",
            flush=True,
        )
    if listed:
        print(
            "   file experiments are already scored on this base; asking the agent",
            flush=True,
        )
    return _enqueue_box(args, None) is True


def _enqueue_fixed(args: argparse.Namespace, item: Dict[str, Any]) -> Optional[bool]:
    """Queue one exact config. None means that config is already scored or running."""
    past = fetch_config_scores(category=None, scored_only=False)
    if past is None:
        return False
    history = [row for row in past if _score_of(row) is not None]
    best_rows = fetch_config_scores(
        category=None,
        scored_only=True,
        limit=1,
        order="avg_combined_final.desc",
    )
    base, best_row = best_base_updates(best_rows if best_rows else history)
    updates = dict(base)
    updates.update(item["updates"])
    if _config_seen(past, updates):
        return None
    if best_row is None:
        print(f"   base config is the catalog default ({len(base)} parameters)", flush=True)
    else:
        print(
            f"   default config is the highest avg_combined_final: "
            f"{best_row.get('experiment')} "
            f"{float(best_row.get('avg_combined_final')):.4f}. "
            f"{item['name']} sets {item['updates']}.",
            flush=True,
        )
    box = {
        "search_category": item["category"],
        "hypothesis": item["hypothesis"],
        "suggested_by": "experiments.json",
        "study_name": item["name"],
    }
    config_id = insert_pending_config(
        experiment=item["name"],
        updates=updates,
        box=box,
        trial_number=0,
        status="queued",
    )
    if not config_id:
        print(
            "   ERROR: could not insert job. Re-run gatk_tuning.sql so "
            "worker_id and batch_id exist.",
            flush=True,
        )
        return False
    print(f"   placed {item['name']} on the stack  config_id={config_id}", flush=True)
    return True


def _config_seen(rows: List[Dict[str, Any]], updates: Dict[str, Any]) -> bool:
    """True when this full config is already scored or still in the stack."""
    wanted = param_signature({key: value for key, value in updates.items() if spec_of(key)})
    for row in rows:
        if _score_of(row) is None and str(row.get("status") or "") not in ("queued", "pending", "running"):
            continue
        full = full_params_from_row(row)
        current = param_signature({key: value for key, value in full.items() if spec_of(key)})
        if current == wanted:
            return True
    return False


def _enqueue_box(
    args: argparse.Namespace,
    preset: Optional[Dict[str, Any]],
) -> Optional[bool]:
    """Queue one search box. None means that box is already scored on this base."""
    if preset is None:
        rc = agent_main([])
        if rc != 0:
            print("   agent did not write a search box", flush=True)
            return False
        try:
            box = load_search_box(Path(args.box))
        except (OSError, ValueError) as e:
            print(f"ERROR: search box: {e}", flush=True)
            return False
    else:
        box = dict(preset)
        box["space"] = dict(preset.get("space") or {})
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
    box["space"] = {
        key: spec for key, spec in box["space"].items() if not held_at_default(str(key))
    }
    if not box["space"]:
        print("   search box has no numeric parameters left to vary", flush=True)
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
    past = fetch_config_scores(category=None, scored_only=False)
    if past is None:
        return False
    history = [row for row in past if _score_of(row) is not None]
    space_keys = list(box["space"])
    if is_agent_experiment(category):
        print(
            f"   {category}: varying "
            + ", ".join(space_keys)
            + "; other parameters stay at the best config",
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
            study, matched, category, distributions, optuna, keys=space_keys
        )
    print(f"   warm-start imported={added} skipped={skipped}", flush=True)
    _settle_running(study, study_name, TrialState)
    if is_agent_experiment(category) and not _trials_in_flight(study, TrialState) and (
        _only_failed_trials(study, TrialState) or _stored_range_is_narrower(study, distributions)
    ):
        print(
            f"   {category}: rebuilding the study on the current range",
            flush=True,
        )
        study = open_study(optuna, study_name, storage, reset=True)
        study = reconcile_categorical_study(
            study, optuna, study_name, storage, distributions
        )

    best_rows = fetch_config_scores(
        category=None,
        scored_only=True,
        limit=1,
        order="avg_combined_final.desc",
    )
    base, best_row = best_base_updates(best_rows if best_rows else history)
    known = _known_settings(study, past, space_keys, base)
    if is_agent_experiment(category):
        remembered = remember_scores(
            study,
            [(params, score) for params, score in known.values() if score is not None],
            distributions,
            optuna,
        )
        if remembered:
            print(f"   remembered {remembered} scored setting(s) from history", flush=True)
            known = _known_settings(study, past, space_keys, base)
    print(
        f"   {len(known)} setting(s) already scored on the current base; "
        "the same values on an older base are run again",
        flush=True,
    )

    budget = max(1, min(MAX_TRIALS, int(box.get("n_trials") or 1)))
    distinct = _distinct_settings(box["space"])
    fresh: List[Dict[str, Any]] = []
    if distinct is not None and distinct <= 100000:
        fresh = [
            params for params in _enumerate_settings(box["space"])
            if param_signature(params) not in known
        ]
        chosen = _spread_settings(fresh, budget)
        target = len(chosen)
        print(
            f"   {category} has {distinct} coarse setting(s), "
            f"{len(fresh)} not scored on the current base; stack uses {target}",
            flush=True,
        )
        for params in chosen:
            try:
                study.enqueue_trial(params)
            except Exception as e:  # noqa: BLE001
                print(f"   WARNING: could not enqueue {params}: {e}", flush=True)
                break
    else:
        target = budget
        print(f"   stack target={target}", flush=True)
    if target <= 0:
        print(
            "   every setting in this experiment is already scored on the current base; "
            "it runs again after another category changes that base",
            flush=True,
        )
        return None

    if best_row is None:
        print(f"   base config is the catalog default ({len(base)} parameters)", flush=True)
    else:
        print(
            f"   default config is the highest avg_combined_final: "
            f"{best_row.get('experiment')} "
            f"{float(best_row.get('avg_combined_final')):.4f} "
            f"({len(base)} parameters). This stack varies {category}; "
            "every other parameter stays on that config.",
            flush=True,
        )

    batch_id = str(uuid.uuid4())
    placed = 0
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
        signature = param_signature(params)
        if signature in known:
            previous = known[signature][1]
            if previous is not None:
                _tell_state(study, trial.number, None, previous)
                print(
                    f"   skip trial={trial.number} already scored {previous:.4f} {params}",
                    flush=True,
                )
            else:
                _tell_state(study, trial.number, TrialState.FAIL)
                print(f"   skip trial={trial.number} already tried {params}", flush=True)
            continue
        known[signature] = (params, None)
        updates = dict(base)
        updates.update(params)
        label = str(box.get("manual_name") or f"optuna-{category}")
        experiment = f"{label}-t{trial.number}"
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
            status="queued",
        )
        if not config_id:
            print(
                "   ERROR: could not insert job. Re-run gatk_tuning.sql so "
                "worker_id and batch_id exist.",
                flush=True,
            )
            _tell_state(study, trial.number, TrialState.FAIL)
            continue
        placed += 1
    if placed == 0:
        return False
    print(f"   placed {placed} trial(s) on the stack", flush=True)
    return True


def _known_settings(
    study: Any,
    history: List[Dict[str, Any]],
    keys: List[str],
    base: Dict[str, Any],
) -> Dict[tuple, tuple]:
    """Varied settings already scored on this base.

    The value is (params, score). A past quality-filter value does not count
    after assembly or priors change the other parameters: that is a new config.
    A finished Optuna trial stores only the varied keys, so it cannot decide
    this. Open trials still count, so the same stack is not queued twice.
    """
    varied = set(keys)
    known: Dict[tuple, tuple] = {}
    cache: Dict[str, set] = {}
    for row in history:
        if not _on_base(row, base, varied):
            continue
        if not keys or not all(
            _row_informs_param(history, row, key, category_of(key), cache) for key in keys
        ):
            continue
        params = _params_for_keys(row, keys)
        if params is None:
            continue
        _remember_setting(known, params, _score_of(row))
    for trial in study.get_trials(deepcopy=False):
        if getattr(trial.state, "name", "") not in ("RUNNING", "WAITING"):
            continue
        raw = dict(getattr(trial, "params", None) or {})
        if not raw or any(key not in raw for key in keys):
            continue
        _remember_setting(known, {key: raw[key] for key in keys}, None)
    return known


def _on_base(row: Dict[str, Any], base: Dict[str, Any], varied: set) -> bool:
    """True when every parameter outside this experiment still matches the base."""
    full = full_params_from_row(row)
    held_row = {key: value for key, value in full.items() if key not in varied and spec_of(key)}
    held_base = {key: value for key, value in base.items() if key not in varied and spec_of(key)}
    return param_signature(held_row) == param_signature(held_base)


def _remember_setting(
    known: Dict[tuple, tuple], params: Dict[str, Any], score: Optional[float]
) -> None:
    signature = param_signature(params)
    previous = known.get(signature)
    if previous is None or (previous[1] is None and score is not None):
        known[signature] = (params, score)


def _params_for_keys(row: Dict[str, Any], keys: List[str]) -> Optional[Dict[str, Any]]:
    updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
    config = row.get("gatk_config") if isinstance(row.get("gatk_config"), dict) else {}
    params: Dict[str, Any] = {}
    for key in keys:
        spec = spec_of(key)
        if spec is None:
            return None
        if key not in updates and key not in config:
            return None
        raw = updates[key] if key in updates else config[key]
        try:
            if spec.kind == "categorical":
                params[key] = coerce_param(spec, raw)
            elif spec.kind == "int":
                params[key] = int(round(float(raw)))
            else:
                params[key] = float(raw)
        except (TypeError, ValueError):
            return None
    return params


def _score_of(row: Dict[str, Any]) -> Optional[float]:
    try:
        value = float(row.get("avg_combined_final"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


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


def _trials_in_flight(study: Any, trial_state: Any) -> bool:
    return any(
        trial.state in (trial_state.RUNNING, trial_state.WAITING)
        for trial in study.get_trials(deepcopy=False)
    )


def _stored_range_is_narrower(study: Any, distributions: Dict[str, Any]) -> bool:
    """True when this study was built on a tighter range than the new box."""
    for trial in study.get_trials(deepcopy=False):
        stored = getattr(trial, "distributions", None) or {}
        if not stored:
            continue
        for key, dist in distributions.items():
            old = stored.get(key)
            if old is None or not hasattr(old, "low") or not hasattr(dist, "low"):
                continue
            if float(old.low) > float(dist.low) + 1e-9 or float(old.high) < float(dist.high) - 1e-9:
                return True
        return False
    return False


def _only_failed_trials(study: Any, trial_state: Any) -> bool:
    """True when every stored trial failed and none are still running."""
    trials = study.get_trials(deepcopy=False)
    if not trials:
        return False
    for trial in trials:
        if trial.state in (trial_state.COMPLETE, trial_state.RUNNING, trial_state.WAITING):
            return False
    return True


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


def _spread_settings(settings: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    """Take `limit` settings spread across the grid, not the first corner."""
    if limit <= 0 or not settings:
        return []
    if len(settings) <= limit:
        return list(settings)
    stride = len(settings) / limit
    return [settings[int(i * stride)] for i in range(limit)]


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
    p.add_argument(
        "--experiments",
        default=str(DEFAULT_EXPERIMENTS),
        help="JSON list of experiments to run before the agent.",
    )
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
    p.add_argument("--poll", type=int, default=int(os.environ.get("FLEET_POLL_SEC") or 20))
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
