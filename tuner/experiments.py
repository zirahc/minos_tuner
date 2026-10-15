"""Experiments listed in experiments.json run before the agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tuner.search_box import enough_trials, specs_to_space_json, validate_search_box
from tuner.spaces import SPACES, category_of, coerce_param, held_at_default, space_for, spec_of


def load_experiments(path: Path) -> List[Dict[str, Any]]:
    """Pending experiments. An empty or missing file means the agent chooses."""
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"could not read {path.name}: {e}") from e
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = raw.get("experiments", [])
    else:
        raise ValueError(f"{path.name} must be a list or an object with experiments")
    if not isinstance(items, list):
        raise ValueError("experiments must be a list")
    return [_one(index, item) for index, item in enumerate(items, start=1)]


def _one(index: int, item: Any) -> Dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"experiment {index} must be an object")
    name = str(item.get("name") or f"experiment-{index}").strip()
    if not name:
        raise ValueError(f"experiment {index} needs a name")
    hypothesis = str(item.get("hypothesis") or "").strip()
    if not hypothesis:
        raise ValueError(f"{name}: hypothesis is required")
    updates = item.get("updates")
    category = str(item.get("search_category") or "").strip()
    if isinstance(updates, dict) and updates:
        coerced = {str(key): _exact_value(str(key), value) for key, value in updates.items()}
        return {
            "kind": "fixed",
            "name": name,
            "hypothesis": hypothesis,
            "category": category or _category_for(name, coerced),
            "updates": coerced,
        }
    if category:
        return {
            "kind": "screen",
            "name": name,
            "hypothesis": hypothesis,
            "box": _screen_box(name, category, hypothesis),
        }
    raise ValueError(f"{name}: set updates or search_category")


def _category_for(name: str, updates: Dict[str, Any]) -> str:
    owners = {category_of(key) for key in updates}
    if len(owners) == 1:
        owner = next(iter(owners))
        if owner:
            return owner
    return name


def _screen_box(name: str, category: str, hypothesis: str) -> Dict[str, Any]:
    if category not in SPACES:
        known = ", ".join(sorted(SPACES))
        raise ValueError(f"{name}: unknown search_category {category!r}. known: {known}")
    space = {
        key: spec
        for key, spec in specs_to_space_json(space_for(category)).items()
        if not held_at_default(key)
    }
    if not space:
        raise ValueError(
            f"{name}: {category} has no numeric parameter. "
            "Set that choice under updates."
        )
    box = validate_search_box({
        "search_category": category,
        "hypothesis": hypothesis,
        "space": space,
        "constraints": {},
        "n_trials": enough_trials(space),
        "optimize": "avg_combined_final",
    })
    box["suggested_by"] = "experiments.json"
    box["manual_name"] = name
    return box


def _exact_value(key: str, raw: Any) -> Any:
    """Keep the value written in the file. Do not snap it onto the search step."""
    spec = spec_of(key)
    if spec is None:
        raise ValueError(f"unknown parameter {key}")
    if spec.kind == "categorical":
        return coerce_param(spec, raw)
    try:
        value = int(round(float(raw))) if spec.kind == "int" else float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key}: {raw!r} is not a number") from e
    if spec.low is not None and value < float(spec.low):
        raise ValueError(f"{key}: {value} is below {spec.low}")
    if spec.high is not None and value > float(spec.high):
        raise ValueError(f"{key}: {value} is above {spec.high}")
    return value
