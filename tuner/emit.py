"""Ask Optuna for the next trials and write gatk_updates.json for main.py.

One VPS loop:
  python -m tuner.agent
  python -m tuner.emit --out /path/to/minos_subnet/gatk_updates_optuna.json
  python main.py --updates gatk_updates_optuna.json

Does not run GATK. Does not edit gatk.conf. Postgres history is the warm-start.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from tuner.search_box import load_search_box, optuna_distributions
from tuner.spaces import default_study_name
from tuner.supabase_scores import fetch_config_scores

DEFAULT_BOX = TUNER_ROOT / "search_box.json"


def main(argv: Optional[List[str]] = None) -> int:
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
    _fail_dangling(study, TrialState)
    history = fetch_config_scores(category=category, scored_only=True)
    if history is None:
        return 2
    added, skipped = import_rows(
        study, history, category, distributions, optuna, keys=keys
    )
    print(f"   warm-start imported={added} skipped={skipped}", flush=True)

    experiments: List[Dict[str, Any]] = []
    for i in range(1, n_trials + 1):
        try:
            trial = study.ask(fixed_distributions=distributions)
        except TypeError:
            trial = study.ask(distributions)
        params = {key: _native(trial.params[key]) for key in keys if key in trial.params}
        name = f"optuna-{category}-t{trial.number}"
        experiments.append({
            "name": name,
            "search_category": category,
            "suggested_by": box.get("suggested_by") or "optuna",
            "study_name": study_name,
            "hypothesis": box.get("hypothesis"),
            "optuna_trial_number": int(trial.number),
            "updates": params,
        })
        print(f"   [{i}/{n_trials}] ask trial={trial.number} {params}", flush=True)
        # Leave RUNNING until the next emit. After main.py POSTs, emit fails
        # dangling asks and re-imports completed rows from Postgres.

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(experiments, indent=2) + "\n", encoding="utf-8")
    print(f"   wrote {len(experiments)} experiment(s) → {out_path}", flush=True)
    print(
        f"   next: python main.py --updates {out_path.name}",
        flush=True,
    )
    return 0


def _fail_dangling(study: Any, trial_state: Any) -> None:
    dangling = [
        t for t in study.get_trials(deepcopy=False)
        if t.state in (trial_state.RUNNING, trial_state.WAITING)
    ]
    for trial in dangling:
        try:
            study.tell(trial, state=trial_state.FAIL)
        except Exception as e:  # noqa: BLE001
            print(f"   WARNING: could not fail dangling trial {trial.number}: {e}", flush=True)
    if dangling:
        print(f"   failed {len(dangling)} dangling ask(s) from a prior emit", flush=True)


def _native(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return json.loads(json.dumps(value, default=str))


def _default_out() -> str:
    subnet = (os.environ.get("MINOS_SUBNET") or "").strip()
    if subnet:
        return str(Path(subnet) / "gatk_updates_optuna.json")
    return str(TUNER_ROOT / "gatk_updates_optuna.json")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ask Optuna and write gatk_updates.json for main.py.",
    )
    p.add_argument("--box", default=str(DEFAULT_BOX), help="search_box.json from the agent")
    p.add_argument(
        "--out",
        default=_default_out(),
        help="Write experiments JSON here (default MINOS_SUBNET/gatk_updates_optuna.json)",
    )
    p.add_argument("--study", default=None)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete the local Optuna study before warm-start (Postgres history is kept).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
