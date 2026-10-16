"""Experiments listed in experiments.json run before the agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tuner.search_box import enough_trials, specs_to_space_json, validate_search_box
from tuner.spaces import (
    SPACES,
    catalog_defaults,
    category_of,
    coerce_param,
    held_at_default,
    space_for,
    spec_of,
)


def parse_gatk_config(name: str, raw: Any) -> Dict[str, Any]:
    """Held GATK parameters. Omitted keys stay at the catalog default."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{name}: gatk_config must be an object of GATK parameters")
    params = catalog_defaults()
    for key, value in raw.items():
        if str(key) == "note":
            continue
        params[str(key)] = _exact_value(str(key), value)
    return params


def load_gatk_config(path: Path) -> Dict[str, Any]:
    """Base GATK parameters for experiments.json. Missing keys stay at the catalog default."""
    if not path.is_file():
        raise ValueError(
            f"{path.name} was not found. experiments.json trials use that file as the base config."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"could not read {path.name}: {e}") from e
    if isinstance(raw, dict) and isinstance(raw.get("gatk_config"), dict):
        raw = raw["gatk_config"]
    return parse_gatk_config(path.name, raw)


def experiment_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """One experiments-table row, in the same shape as an experiments.json entry."""
    payload: Dict[str, Any] = {
        "name": row.get("name"),
        "hypothesis": row.get("hypothesis") or "",
    }
    if row.get("search_category"):
        payload["search_category"] = row.get("search_category")
    space = row.get("space")
    updates = row.get("updates")
    if isinstance(space, dict) and space:
        payload["space"] = space
    if isinstance(updates, dict) and updates:
        payload["updates"] = updates
    if isinstance(row.get("gatk_config"), dict):
        payload["gatk_config"] = row.get("gatk_config")
    item = _one(1, payload)
    experiment_id = str(row.get("id") or "")
    item["experiment_id"] = experiment_id
    if item["kind"] == "screen":
        item["box"]["experiment_id"] = experiment_id
    return item


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
    space_in = item.get("space")
    gatk_config = parse_gatk_config(name, item.get("gatk_config"))
    if isinstance(updates, dict) and updates:
        if isinstance(space_in, dict) and space_in:
            raise ValueError(f"{name}: set updates or space, not both")
        coerced = {str(key): _exact_value(str(key), value) for key, value in updates.items()}
        return {
            "kind": "fixed",
            "name": name,
            "hypothesis": hypothesis,
            "category": category or _category_for(name, coerced),
            "updates": coerced,
            "gatk_config": gatk_config,
        }
    if isinstance(space_in, dict) and space_in:
        category = category or _category_for(name, space_in)
    if category:
        parsed = {
            "kind": "screen",
            "name": name,
            "hypothesis": hypothesis,
            "box": _screen_box(name, category, hypothesis, space_in),
            "gatk_config": gatk_config,
        }
        return parsed
    raise ValueError(f"{name}: set updates, space, or search_category")


def _category_for(name: str, updates: Dict[str, Any]) -> str:
    owners = {category_of(key) for key in updates}
    if len(owners) == 1:
        owner = next(iter(owners))
        if owner:
            return owner
    return name


def catalog_name(label: str) -> str:
    """quality_filters_1 and quality_filters_20261015T211500Z both use quality_filters."""
    text = str(label or "").strip()
    if text in SPACES:
        return text
    found = ""
    for name in SPACES:
        if text.startswith(name + "_") and len(name) > len(found):
            found = name
    return found


def _identity(name: str, category: str, catalog: str) -> str:
    """A suffix or timeslot is its own pass. The bare catalog name stays shared."""
    for label in (category, name):
        if catalog_name(label) == catalog and label != catalog:
            return label
    return catalog


def _screen_box(
    name: str,
    category: str,
    hypothesis: str,
    space_in: Any = None,
) -> Dict[str, Any]:
    catalog = catalog_name(category) or catalog_name(name)
    if not catalog:
        known = ", ".join(sorted(SPACES))
        raise ValueError(f"{name}: unknown search_category {category!r}. known: {known}")
    identity = _identity(name, category, catalog)
    space = _experiment_space(name, catalog, space_in)
    if not space:
        raise ValueError(
            f"{name}: {catalog} has no numeric parameter. "
            "Set that choice under updates."
        )
    box = validate_search_box({
        "search_category": catalog,
        "hypothesis": hypothesis,
        "space": space,
        "constraints": {},
        "n_trials": enough_trials(space),
        "optimize": "avg_combined_final",
    })
    box["space"] = space
    box["n_trials"] = enough_trials(space)
    box["search_category"] = identity
    box["catalog_category"] = catalog
    if identity != catalog:
        box["pass_id"] = identity
    box["suggested_by"] = "experiments"
    box["manual_name"] = name
    box["study_name"] = name
    return box


def _experiment_space(name: str, catalog: str, space_in: Any) -> Dict[str, Any]:
    """File ranges replace the catalog grid. Omitted space keeps the catalog steps."""
    allowed = space_for(catalog)
    if not isinstance(space_in, dict) or not space_in:
        return {
            key: spec
            for key, spec in specs_to_space_json(allowed).items()
            if not held_at_default(key)
        }
    out: Dict[str, Any] = {}
    for key, raw in space_in.items():
        if key not in allowed:
            raise ValueError(f"{name}: {key} is not in {catalog}")
        if not isinstance(raw, dict):
            raise ValueError(
                f"{name}: space.{key} must be an object with type and values, "
                "or type, low, high, and step"
            )
        spec = allowed[key]
        if spec.kind == "categorical":
            raise ValueError(f"{name}: set {key} under updates. It is a choice, not a range.")
        if "values" in raw:
            out[key] = _value_list(name, key, spec, raw)
        else:
            out[key] = _range_item(name, key, spec, raw)
    return out


def _value_list(name: str, key: str, spec: Any, raw: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(raw.get("type") or "").strip().lower()
    if kind not in ("int", "float"):
        raise ValueError(f"{name}: {key} needs a type of int or float")
    if kind != spec.kind:
        raise ValueError(f"{name}: {key} is {spec.kind} in the catalog, not {kind}")
    raw_values = raw.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"{name}: {key} values must be a non-empty list")
    values: List[Any] = []
    seen = set()
    for item in raw_values:
        try:
            number = float(item)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name}: {key} value {item!r} is not a number") from e
        if kind == "int" and number != int(number):
            raise ValueError(f"{name}: {key} value {item!r} must be a whole number")
        value: Any = int(number) if kind == "int" else number
        if spec.low is not None and float(value) < float(spec.low):
            raise ValueError(f"{name}: {key} value {value} is below {spec.low}")
        if spec.high is not None and float(value) > float(spec.high):
            raise ValueError(f"{name}: {key} value {value} is above {spec.high}")
        if value in seen:
            raise ValueError(f"{name}: {key} value {value} is listed more than once")
        seen.add(value)
        values.append(value)
    return {
        "type": kind,
        "values": values,
        "low": min(values),
        "high": max(values),
    }


def _range_item(name: str, key: str, spec: Any, raw: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(raw.get("type") or "").strip().lower()
    if kind not in ("int", "float"):
        raise ValueError(f"{name}: {key} needs a type of int or float")
    if kind != spec.kind:
        raise ValueError(f"{name}: {key} is {spec.kind} in the catalog, not {kind}")
    low = raw.get("low", spec.low)
    high = raw.get("high", spec.high)
    try:
        low_f = float(low)
        high_f = float(high)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: {key} low and high must be numbers") from e
    if spec.low is not None and low_f < float(spec.low):
        raise ValueError(f"{name}: {key} low {low_f} is below {spec.low}")
    if spec.high is not None and high_f > float(spec.high):
        raise ValueError(f"{name}: {key} high {high_f} is above {spec.high}")
    if low_f > high_f:
        raise ValueError(f"{name}: {key} low is above high")
    if kind == "int":
        low_v: Any = int(round(low_f))
        high_v: Any = int(round(high_f))
    else:
        low_v = low_f
        high_v = high_f
    item: Dict[str, Any] = {"type": kind, "low": low_v, "high": high_v}
    if "step" not in raw or raw.get("step") is None:
        raise ValueError(f"{name}: {key} needs a step")
    try:
        step = float(raw["step"])
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name}: {key} step must be a number") from e
    if step <= 0:
        raise ValueError(f"{name}: {key} step must be greater than 0")
    if kind == "int" and step != int(step):
        raise ValueError(f"{name}: {key} step must be a whole number")
    item["step"] = int(step) if kind == "int" else step
    return item


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
