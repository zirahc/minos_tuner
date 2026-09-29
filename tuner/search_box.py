"""Validate and serialize the agent → Optuna search-box contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from tuner.spaces import SPACES, ParamSpec, known_categories, space_for

ALLOWED_OPTIMIZE = "avg_combined_final"
MIN_TRIALS = 1
MAX_TRIALS = 8


def specs_to_space_json(specs: Mapping[str, ParamSpec]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, spec in specs.items():
        item: Dict[str, Any] = {"type": spec.kind}
        if spec.kind in ("int", "float"):
            item["low"] = spec.low
            item["high"] = spec.high
            if spec.log:
                item["log"] = True
        if spec.kind == "categorical":
            item["choices"] = list(spec.choices or ())
        if spec.note:
            item["note"] = spec.note
        out[key] = item
    return out


def space_catalog() -> Dict[str, Any]:
    return {name: specs_to_space_json(specs) for name, specs in SPACES.items()}


def validate_search_box(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("search box must be a JSON object")
    category = str(raw.get("search_category") or "").strip()
    if category not in SPACES:
        raise ValueError(
            f"search_category must be one of: {', '.join(known_categories())}"
        )
    hypothesis = str(raw.get("hypothesis") or "").strip()
    if not hypothesis:
        raise ValueError("hypothesis is required")

    optimize = str(raw.get("optimize") or ALLOWED_OPTIMIZE).strip()
    if optimize != ALLOWED_OPTIMIZE:
        raise ValueError(f"optimize must be {ALLOWED_OPTIMIZE!r}")

    n_trials = int(raw.get("n_trials") or 6)
    if n_trials < MIN_TRIALS or n_trials > MAX_TRIALS:
        raise ValueError(f"n_trials must be {MIN_TRIALS}..{MAX_TRIALS}")

    allowed = space_for(category)
    space_in = raw.get("space")
    if not isinstance(space_in, dict) or not space_in:
        raise ValueError("space must be a non-empty object of GATK keys")
    space_out: Dict[str, Any] = {}
    for key, spec_in in space_in.items():
        if key not in allowed:
            raise ValueError(f"key {key!r} is not in category {category}")
        if not isinstance(spec_in, dict):
            raise ValueError(f"space.{key} must be an object")
        space_out[key] = _validate_spec(key, allowed[key], spec_in)

    constraints_in = raw.get("constraints") if isinstance(raw.get("constraints"), dict) else {}
    constraints = _validate_constraints(constraints_in)

    return {
        "search_category": category,
        "hypothesis": hypothesis,
        "space": space_out,
        "constraints": constraints,
        "n_trials": n_trials,
        "optimize": ALLOWED_OPTIMIZE,
    }


def load_search_box(path: Path) -> Dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("search_box.json must be an object")
    return validate_search_box(raw)


def optuna_distributions(space: Mapping[str, Any], optuna: Any) -> Dict[str, Any]:
    dists: Dict[str, Any] = {}
    for key, spec in space.items():
        kind = spec.get("type")
        if kind == "int":
            dists[key] = optuna.distributions.IntDistribution(int(spec["low"]), int(spec["high"]))
        elif kind == "float":
            dists[key] = optuna.distributions.FloatDistribution(
                float(spec["low"]), float(spec["high"]), log=bool(spec.get("log"))
            )
        elif kind == "categorical":
            dists[key] = optuna.distributions.CategoricalDistribution(list(spec.get("choices") or []))
        else:
            raise ValueError(f"unsupported space type for {key}: {kind!r}")
    return dists


def _validate_spec(key: str, allowed: ParamSpec, spec_in: Mapping[str, Any]) -> Dict[str, Any]:
    kind = str(spec_in.get("type") or allowed.kind)
    if kind != allowed.kind:
        raise ValueError(f"{key}: type must be {allowed.kind!r}")
    if kind == "categorical":
        allowed_choices = list(allowed.choices or ())
        choices = spec_in.get("choices", allowed_choices)
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{key}: choices must be a non-empty list")
        for item in choices:
            if item not in allowed_choices:
                raise ValueError(f"{key}: choice {item!r} not allowed")
        # Optuna cannot change a categorical choice list after the first trial.
        return {"type": "categorical", "choices": allowed_choices}
    low = spec_in.get("low", allowed.low)
    high = spec_in.get("high", allowed.high)
    try:
        low_f = float(low)
        high_f = float(high)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key}: low/high must be numeric") from e
    if allowed.low is not None and low_f < float(allowed.low):
        raise ValueError(f"{key}: low {low_f} below allowed {allowed.low}")
    if allowed.high is not None and high_f > float(allowed.high):
        raise ValueError(f"{key}: high {high_f} above allowed {allowed.high}")
    if low_f > high_f:
        raise ValueError(f"{key}: low > high")
    out: Dict[str, Any] = {"type": kind, "low": int(low_f) if kind == "int" else low_f, "high": int(high_f) if kind == "int" else high_f}
    if allowed.log or spec_in.get("log"):
        out["log"] = True
    return out


def _validate_constraints(raw: Mapping[str, Any]) -> Dict[str, float]:
    allowed = ("min_avg_f1_snp", "min_avg_f1_indel", "min_avg_combined_final")
    out: Dict[str, float] = {}
    for key in allowed:
        if key not in raw or raw[key] is None:
            continue
        try:
            value = float(raw[key])
        except (TypeError, ValueError) as e:
            raise ValueError(f"constraints.{key} must be numeric") from e
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"constraints.{key} must be in [0, 1]")
        out[key] = value
    return out


def write_search_box(box: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(box), indent=2) + "\n", encoding="utf-8")
