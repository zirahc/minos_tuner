"""Warm-start an Optuna study from gatk_config_scores.

Tuner server only. Does not run GATK. Scoring stays on the other machine.

Examples:
  python -m tuner --list
  python -m tuner --list --category quality_filters
  python -m tuner --warm-start --category quality_filters
"""
from __future__ import annotations

import argparse
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

from tuner.spaces import (
    default_study_name,
    known_categories,
    params_from_row,
    space_for,
)
from tuner.supabase_scores import fetch_config_scores

DEFAULT_STORAGE = f"sqlite:///{(TUNER_ROOT / 'optuna.db').as_posix()}"


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.category and args.category not in known_categories():
        print(
            f"ERROR: unknown category {args.category!r}. "
            f"known: {', '.join(known_categories())}",
            flush=True,
        )
        return 2

    rows = fetch_config_scores(category=args.category, scored_only=not args.include_failed)
    if rows is None:
        return 2
    if args.list or not args.warm_start:
        _print_rows(rows, args.category)
        if not args.warm_start:
            return 0

    try:
        import optuna
    except ImportError:
        print(
            "ERROR: optuna is not installed. On the tuner server: "
            "pip install -r requirements-tuner.txt",
            flush=True,
        )
        return 2

    category = args.category
    if not category:
        print("ERROR: --warm-start requires --category", flush=True)
        return 2

    study_name = args.study or default_study_name(category)
    storage = (os.environ.get("OPTUNA_STORAGE") or DEFAULT_STORAGE).strip()
    spec_map = space_for(category)
    distributions = {key: _to_distribution(optuna, spec) for key, spec in spec_map.items()}

    study = open_study(optuna, study_name, storage, reset=args.reset)
    added, skipped = import_rows(study, rows, category, distributions, optuna)
    print(
        f"   imported {added} completed trial(s), skipped {skipped}, "
        f"study now has {len(study.trials)} trial(s)",
        flush=True,
    )
    if study.trials:
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            print(
                f"   best avg_combined_final={study.best_value:.6f}  "
                f"params={study.best_params}",
                flush=True,
            )
    return 0


def open_study(optuna: Any, study_name: str, storage: str, reset: bool = False) -> Any:
    if reset:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"   deleted study {study_name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"   no existing study to delete ({e})", flush=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, constant_liar=True),
    )
    # TPE uses past scores and handles categorical parameters.
    # constant_liar keeps a parallel batch from all asking the same point.
    # CMA-ES and GP are a poor fit: several GATK parameters are categorical.
    # Hyperband-style pruning needs an epoch loop; the 0.01 screen is the early stop.
    study.sampler = optuna.samplers.TPESampler(seed=42, constant_liar=True)
    print(
        f"   study={study_name}  sampler=TPE  storage={_storage_label(storage)}",
        flush=True,
    )
    return study


def import_rows(
    study: Any,
    rows: List[Dict[str, Any]],
    category: str,
    distributions: Dict[str, Any],
    optuna: Any,
    keys: Optional[List[str]] = None,
) -> tuple:
    from optuna.trial import TrialState, create_trial

    key_tuple = tuple(keys) if keys else None
    # Callers still pass the tightened box. Import ignores it and uses the catalog.
    del distributions
    existing = {
        str(t.user_attrs.get("config_id"))
        for t in study.trials
        if t.user_attrs.get("config_id")
    }
    added = 0
    skipped = 0
    for row in rows:
        config_id = str(row.get("config_id") or "")
        if not config_id:
            skipped += 1
            continue
        if config_id in existing:
            skipped += 1
            continue
        value = row.get("avg_combined_final")
        try:
            score = float(value)
        except (TypeError, ValueError):
            skipped += 1
            continue
        try:
            params = params_from_row(row, category, keys=key_tuple)
        except (TypeError, ValueError) as e:
            print(f"   skip {config_id}: {e}", flush=True)
            skipped += 1
            continue
        trial_distributions = {
            key: _to_distribution(optuna, space_for(category)[key])
            for key in params
        }
        try:
            trial = create_trial(
                params=params,
                distributions=trial_distributions,
                values=[score],
                state=TrialState.COMPLETE,
                user_attrs={
                    "config_id": config_id,
                    "experiment": row.get("experiment"),
                    "avg_core": row.get("avg_core"),
                    "avg_germline": row.get("avg_germline"),
                    "avg_fp_per_target": row.get("avg_fp_per_target"),
                    "avg_f1_snp": row.get("avg_f1_snp"),
                    "avg_f1_indel": row.get("avg_f1_indel"),
                    "hypothesis": row.get("hypothesis"),
                    "suggested_by": row.get("suggested_by"),
                },
            )
        except ValueError as e:
            print(f"   skip {config_id}: {e}", flush=True)
            skipped += 1
            continue
        study.add_trial(trial)
        existing.add(config_id)
        added += 1
    return added, skipped


def remember_scores(
    study: Any,
    scored: List[tuple],
    distributions: Dict[str, Any],
    optuna: Any,
) -> int:
    """Add completed trials for settings already scored inside this box.

    Each item is (params, score). A value outside the current distributions
    is skipped so the study range does not change.
    """
    from optuna.trial import TrialState, create_trial

    existing = {
        param_signature(trial.params)
        for trial in study.get_trials(deepcopy=False)
        if getattr(trial, "params", None)
    }
    added = 0
    for params, score in scored:
        if not isinstance(params, dict) or not params:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        signature = param_signature(params)
        if signature in existing:
            continue
        if any(
            key not in distributions or not _value_in_distribution(distributions[key], val)
            for key, val in params.items()
        ):
            continue
        try:
            trial = create_trial(
                params=dict(params),
                distributions={key: distributions[key] for key in params},
                values=[value],
                state=TrialState.COMPLETE,
            )
            study.add_trial(trial)
        except (TypeError, ValueError) as e:
            print(f"   skip remembered setting {params}: {e}", flush=True)
            continue
        existing.add(signature)
        added += 1
    return added


def param_signature(params: Dict[str, Any]) -> tuple:
    """Stable identity of one trial's parameter values."""
    return tuple(sorted((str(key), _norm_value(value)) for key, value in params.items()))


def _norm_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        number = float(value)
        if number == int(number):
            return str(int(number))
        return str(number)
    return str(value)


def _value_in_distribution(dist: Any, value: Any) -> bool:
    kind = dist.__class__.__name__
    if kind == "CategoricalDistribution":
        return any(_norm_value(choice) == _norm_value(value) for choice in dist.choices)
    if kind in ("IntDistribution", "FloatDistribution"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return float(dist.low) - 1e-9 <= number <= float(dist.high) + 1e-9
    return False


def catalog_categorical_distributions(
    category: str,
    distributions: Dict[str, Any],
    optuna: Any,
) -> Dict[str, Any]:
    """Keep every categorical on the catalog choice list.

    Int and float ranges may shrink. A categorical may not: Optuna raises
    ``CategoricalDistribution does not support dynamic value space``.
    """
    from tuner.spaces import is_agent_experiment

    if is_agent_experiment(category):
        return dict(distributions)
    catalog = space_for(category)
    fixed = dict(distributions)
    for key in list(fixed):
        spec = catalog.get(key)
        if spec is not None and spec.kind == "categorical":
            fixed[key] = _to_distribution(optuna, spec)
    return fixed


def reconcile_categorical_study(
    study: Any,
    optuna: Any,
    study_name: str,
    storage: str,
    distributions: Dict[str, Any],
) -> Any:
    """Rebuild a local study whose categorical choices differ from the catalog.

    Scored history is imported again by the caller. Trials still running keep
    the choice list already stored, so their tell still matches.
    """
    stored = _stored_distributions(study)
    conflicts = _distribution_conflicts(stored, distributions)
    if not conflicts:
        return study
    if _has_open_trials(study):
        print(
            "   categorical choices are locked by trials still in this study; "
            "using the stored choice list until those trials finish",
            flush=True,
        )
        for key, old, new in conflicts:
            print(f"   {key}: study={old} catalog={new}", flush=True)
            distributions[key] = stored[key]
        return study
    for key, old, new in conflicts:
        print(
            f"   {key} choices in {study_name} are {old}; catalog is {new}. "
            "Rebuilding this study from scored history.",
            flush=True,
        )
    return open_study(optuna, study_name, storage, reset=True)


def _stored_distributions(study: Any) -> Dict[str, Any]:
    stored: Dict[str, Any] = {}
    for trial in study.trials:
        for name, dist in getattr(trial, "distributions", {}).items():
            stored[name] = dist
    return stored


def _distribution_conflicts(
    stored: Dict[str, Any], distributions: Dict[str, Any]
) -> List[tuple]:
    conflicts = []
    for key, dist in distributions.items():
        old = stored.get(key)
        if old is None:
            continue
        if old.__class__.__name__ != dist.__class__.__name__:
            conflicts.append((key, old.__class__.__name__, dist.__class__.__name__))
            continue
        if dist.__class__.__name__ == "CategoricalDistribution":
            if list(old.choices) != list(dist.choices):
                conflicts.append((key, list(old.choices), list(dist.choices)))
            continue
        old_step = getattr(old, "step", None)
        new_step = getattr(dist, "step", None)
        if old_step != new_step or bool(getattr(old, "log", False)) != bool(getattr(dist, "log", False)):
            conflicts.append((key, old_step, new_step))
    return conflicts


def _has_open_trials(study: Any) -> bool:
    for trial in study.trials:
        if getattr(trial.state, "name", "") in ("RUNNING", "WAITING"):
            return True
    return False


def _to_distribution(optuna: Any, spec: Any) -> Any:
    from tuner.spaces import coarse_levels

    if spec.kind == "int":
        return optuna.distributions.IntDistribution(int(spec.low), int(spec.high))
    if spec.kind == "float" and spec.log:
        levels = coarse_levels(spec)
        if levels:
            return optuna.distributions.CategoricalDistribution(list(levels))
        return optuna.distributions.FloatDistribution(
            float(spec.low), float(spec.high), log=True
        )
    if spec.kind == "float":
        return optuna.distributions.FloatDistribution(float(spec.low), float(spec.high))
    if spec.kind == "categorical":
        return optuna.distributions.CategoricalDistribution(list(spec.choices or ()))
    raise ValueError(f"unknown spec kind {spec.kind!r}")


def _storage_label(storage: str) -> str:
    if storage.startswith("sqlite"):
        return "sqlite (local tuner file)"
    if "postgres" in storage or "supabase" in storage:
        return "postgres"
    return "custom"


def _print_rows(rows: List[Dict[str, Any]], category: Optional[str]) -> None:
    print("=" * 72, flush=True)
    print("  GATK CONFIG SCORES  (history for Optuna)", flush=True)
    print("=" * 72, flush=True)
    if category:
        print(f"  category: {category}", flush=True)
    print(
        f"  {'experiment':24s} {'n':>4s} {'final':>10s} {'core':>10s} "
        f"{'germline':>10s} {'fp/tgt':>10s}",
        flush=True,
    )
    for row in rows:
        name = str(row.get("experiment") or row.get("config_id") or "?")[:24]
        print(
            f"  {name:24s} {_fmt(row.get('n_scored')):>4s} "
            f"{_fmt(row.get('avg_combined_final')):>10s} "
            f"{_fmt(row.get('avg_core')):>10s} "
            f"{_fmt(row.get('avg_germline')):>10s} "
            f"{_fmt(row.get('avg_fp_per_target')):>10s}",
            flush=True,
        )
        if category:
            try:
                params = params_from_row(row, category)
                shown = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"      {shown}", flush=True)
            except (TypeError, ValueError) as e:
                print(f"      params unavailable: {e}", flush=True)
    print("=" * 72, flush=True)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Load gatk_config_scores into an Optuna study (tuner server).",
    )
    p.add_argument(
        "--category",
        default=None,
        help="search_category / study box. Required for --warm-start. "
        f"known: {', '.join(known_categories())}",
    )
    p.add_argument(
        "--study",
        default=None,
        help="Optuna study name (default gatk-v2-<category>).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print history rows (default if --warm-start is omitted).",
    )
    p.add_argument(
        "--warm-start",
        action="store_true",
        help="Import scored rows as completed Optuna trials.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Delete the local study before importing.",
    )
    p.add_argument(
        "--include-failed",
        action="store_true",
        help="Also fetch configs with no scored combined_final (still skipped on import).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
