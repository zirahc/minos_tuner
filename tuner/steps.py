"""Multi-step search box: diagnose, review, choose, tighten.

Pure functions. No Supabase and no LLM. tuner.agent runs this, then may
ask the model to revise the draft.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tuner.search_box import specs_to_space_json
from tuner.spaces import SPACES, ParamSpec, space_for
from tuner.v2_score import component_losses

# Order is the next lever for that limiter. A plateaued category is skipped.
BOTTLENECK_PLAN: Dict[str, Tuple[str, ...]] = {
    "no_history": ("quality_filters",),
    "false_positives": ("quality_filters", "downsampling", "emit", "pair_hmm"),
    "indel": ("pcr", "assembly", "pair_hmm", "priors"),
    "sensitivity": ("active_region", "assembly", "priors", "quality_filters"),
    "broad": (
        "assembly",
        "active_region",
        "priors",
        "pair_hmm",
        "downsampling",
        "pcr",
        "emit",
        "quality_filters",
    ),
}

IMPROVE_MARGIN = 0.0015
EXHAUST_MIN_TRIALS = 4
SHRINK_MIN_TRIALS = 3
TOP_K = 3


def multistep_search_box(
    history: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (search box, report). The box still needs validate_search_box."""
    report = diagnose(history)
    category, reason, mode = choose_category(report)
    scored = _by_category(history).get(category, [])
    shrink = mode == "refine" and len(scored) >= SHRINK_MIN_TRIALS
    space = _space_json(category, scored if shrink else [])
    if shrink and not _searchable(space):
        category, reason, mode = _next_searchable(report, category)
        scored = _by_category(history).get(category, [])
        shrink = mode == "refine" and len(scored) >= SHRINK_MIN_TRIALS
        space = _space_json(category, scored if shrink else [])
        if not _searchable(space):
            space = specs_to_space_json(space_for(category))
    bounds = "shrunk" if space != specs_to_space_json(space_for(category)) else "full"
    n_trials = 4 if bounds == "shrunk" else 6
    hypothesis = _hypothesis(report, category, reason, bounds)
    report.update({
        "choice": category,
        "reason": reason,
        "mode": mode,
        "bounds": bounds,
        "n_trials": n_trials,
        "hypothesis": hypothesis,
    })
    box = {
        "search_category": category,
        "hypothesis": hypothesis,
        "space": space,
        "constraints": _constraints(report),
        "n_trials": n_trials,
        "optimize": "avg_combined_final",
    }
    return box, report


def diagnose(history: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in history if _num(row.get("avg_combined_final")) is not None]
    grouped = _by_category(scored)
    categories: Dict[str, Any] = {}
    for name in SPACES:
        rows = grouped.get(name, [])
        best = _best_score(rows)
        categories[name] = {
            "n": len(rows),
            "best": best,
            "state": _category_state(rows),
        }
    best_row = _best_row(scored)
    metrics = _metrics(best_row)
    losses = component_losses(metrics)
    bottleneck = _bottleneck(losses, has_scores=bool(scored))
    last = scored[-1] if scored else {}
    last_category = last.get("search_category") if last.get("search_category") in SPACES else None
    return {
        "bottleneck": bottleneck,
        "n_scored": len(scored),
        "best_avg_combined_final": _num(best_row.get("avg_combined_final")) if best_row else None,
        "best_category": best_row.get("search_category") if best_row else None,
        "metrics": metrics,
        "losses": losses,
        "categories": categories,
        "last_category": last_category,
        "last_state": categories[last_category]["state"] if last_category else None,
        "last_hypothesis": (last.get("hypothesis") or "") if last else "",
    }


def choose_category(report: Mapping[str, Any]) -> Tuple[str, str, str]:
    """Return (category, reason, mode). mode is explore or refine."""
    categories: Mapping[str, Any] = report["categories"]
    last = report.get("last_category")
    last_state = report.get("last_state")
    if report["bottleneck"] == "no_history" or not last:
        return "quality_filters", "no scored history; start with quality filters", "explore"
    if last_state in ("improving", "partial"):
        mode = "refine" if last_state == "improving" else "explore"
        return last, f"{last} is {last_state}; keep searching it", mode

    plan = BOTTLENECK_PLAN[report["bottleneck"]]
    for name in plan:
        state = categories[name]["state"]
        if state in ("untried", "partial", "improving"):
            mode = "refine" if state == "improving" else "explore"
            return (
                name,
                f"{report['bottleneck']} limiter; {last} plateaued, next is {name} ({state})",
                mode,
            )

    ranked = sorted(
        plan,
        key=lambda name: categories[name]["best"] if categories[name]["best"] is not None else -1.0,
        reverse=True,
    )
    winner = ranked[0]
    return (
        winner,
        f"every {report['bottleneck']} category plateaued; refine the best, {winner}",
        "refine",
    )


def _hypothesis(report: Mapping[str, Any], category: str, reason: str, bounds: str) -> str:
    best = report.get("best_avg_combined_final")
    best_txt = f"{best:.4f}" if isinstance(best, float) else "none"
    return (
        f"v2 point loss {_fmt_losses(report.get('losses'))}. "
        f"Bottleneck={report['bottleneck']} (best avg_combined_final={best_txt}). "
        f"{reason}. Bounds={bounds}. Confirm or reject against the next scored rows."
    )


def _constraints(report: Mapping[str, Any]) -> Dict[str, float]:
    metrics = report.get("metrics") or {}
    out: Dict[str, float] = {}
    for src, dest in (
        ("avg_f1_snp", "min_avg_f1_snp"),
        ("avg_f1_indel", "min_avg_f1_indel"),
    ):
        value = metrics.get(src)
        if value is None:
            continue
        out[dest] = round(max(0.0, min(1.0, float(value) - 0.02)), 4)
    return out


def _bottleneck(losses: Mapping[str, Any], has_scores: bool) -> str:
    if not has_scores:
        return "no_history"
    largest = losses.get("largest")
    if largest == "indel":
        return "indel"
    if largest == "fp":
        return "false_positives"
    if largest in ("snp", "core"):
        return "sensitivity"
    return "broad"


def _fmt_losses(losses: Optional[Mapping[str, Any]]) -> str:
    source = losses or {}
    parts = []
    for key in ("indel", "snp", "fp"):
        value = source.get(key)
        parts.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}=na")
    return " ".join(parts)


def _category_state(rows: Sequence[Mapping[str, Any]]) -> str:
    scores = [_num(row.get("avg_combined_final")) for row in rows]
    scores = [score for score in scores if score is not None]
    n = len(scores)
    if n == 0:
        return "untried"
    if n < EXHAUST_MIN_TRIALS:
        return "partial"
    earlier, recent = scores[:-3], scores[-3:]
    if earlier and max(recent) >= max(earlier) + IMPROVE_MARGIN:
        return "improving"
    return "exhausted"


def _next_searchable(report: Mapping[str, Any], blocked: str) -> Tuple[str, str, str]:
    plan = BOTTLENECK_PLAN[str(report["bottleneck"])]
    categories: Mapping[str, Any] = report["categories"]
    for name in plan:
        if name == blocked:
            continue
        state = categories[name]["state"]
        if state in ("untried", "partial", "improving"):
            mode = "refine" if state == "improving" else "explore"
            return name, f"{blocked} has no room left; next is {name} ({state})", mode
    for name in plan:
        if name != blocked:
            return name, f"{blocked} has no room left; refine {name}", "refine"
    return blocked, f"{blocked} has no room left; reopen the full catalog", "explore"


def _searchable(space: Mapping[str, Any]) -> bool:
    for spec in space.values():
        kind = spec.get("type")
        if kind == "categorical" and len(spec.get("choices") or []) > 1:
            return True
        if kind in ("int", "float"):
            try:
                if float(spec["high"]) > float(spec["low"]):
                    return True
            except (KeyError, TypeError, ValueError):
                continue
    return False


def _space_json(category: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    specs = space_for(category)
    if not rows:
        return specs_to_space_json(specs)
    top = _top_rows(rows)
    out: Dict[str, Any] = {}
    for key, spec in specs.items():
        if spec.kind == "categorical":
            choices = _shrink_choices(rows, key, spec.choices or ())
            out[key] = {"type": "categorical", "choices": choices}
            continue
        values = _param_values(top, key, spec)
        out[key] = _shrunk_spec(spec, values) if values else _full_spec(spec)
    return out


def _shrunk_spec(spec: ParamSpec, values: Sequence[Any]) -> Dict[str, Any]:
    if spec.kind == "categorical":
        return {"type": "categorical", "choices": list(spec.choices or ())}
    low = float(spec.low if spec.low is not None else 0)
    high = float(spec.high if spec.high is not None else low)
    if spec.kind == "int":
        new_lo, new_hi = _shrink_int([float(v) for v in values], int(low), int(high))
        return {"type": "int", "low": new_lo, "high": new_hi}
    new_lo, new_hi = _shrink_float([float(v) for v in values], low, high, spec.log)
    item: Dict[str, Any] = {"type": "float", "low": new_lo, "high": new_hi}
    if spec.log:
        item["log"] = True
    return item


def _full_spec(spec: ParamSpec) -> Dict[str, Any]:
    return specs_to_space_json({"": spec})[""]


def _shrink_choices(
    rows: Sequence[Mapping[str, Any]], key: str, allowed: Sequence[Any]
) -> List[Any]:
    allowed_list = list(allowed)
    best: Dict[Any, float] = {}
    for row in rows:
        updates = row.get("gatk_updates")
        if not isinstance(updates, dict) or key not in updates:
            continue
        matched = _match_choice(updates[key], allowed_list)
        score = _num(row.get("avg_combined_final"))
        if matched is None or score is None:
            continue
        best[matched] = max(score, best.get(matched, score))
    if len(best) < 2:
        return allowed_list
    ranked = sorted(best, key=lambda choice: best[choice], reverse=True)
    top = best[ranked[0]]
    kept = [choice for choice in ranked if best[choice] >= top - 0.01]
    if len(kept) < 2:
        kept = ranked[:2]
    ordered = [choice for choice in allowed_list if choice in kept]
    return ordered or allowed_list


def _shrink_int(values: Sequence[float], low: int, high: int) -> Tuple[int, int]:
    center = int(round(sorted(values)[len(values) // 2]))
    span = high - low
    observed = int(round(max(values) - min(values))) if len(values) > 1 else 0
    half = max(int(round(span * 0.2)), observed, 1)
    new_lo = max(low, center - half)
    new_hi = min(high, center + half)
    if new_hi - new_lo < 2 and high - low >= 2:
        new_lo = max(low, min(center - 1, high - 2))
        new_hi = min(high, max(new_lo + 2, center + half))
    if new_lo > new_hi:
        return low, high
    return int(new_lo), int(new_hi)


def _shrink_float(
    values: Sequence[float], low: float, high: float, log: bool
) -> Tuple[float, float]:
    if log and low > 0 and high > 0 and all(v > 0 for v in values):
        logged = [math.log(v) for v in values]
        new_lo, new_hi = _shrink_linear(logged, math.log(low), math.log(high))
        return _round_float(math.exp(new_lo)), _round_float(math.exp(new_hi))
    new_lo, new_hi = _shrink_linear(list(values), low, high)
    return _round_float(new_lo), _round_float(new_hi)


def _shrink_linear(values: Sequence[float], low: float, high: float) -> Tuple[float, float]:
    ordered = sorted(values)
    center = ordered[len(ordered) // 2]
    span = high - low
    observed = ordered[-1] - ordered[0] if len(ordered) > 1 else 0.0
    half = max(span * 0.2, observed, span * 0.05 if span else 0.0)
    new_lo = max(low, center - half)
    new_hi = min(high, center + half)
    if new_hi <= new_lo:
        return low, high
    return new_lo, new_hi


def _param_values(
    rows: Sequence[Mapping[str, Any]], key: str, spec: ParamSpec
) -> List[Any]:
    values: List[Any] = []
    for row in rows:
        updates = row.get("gatk_updates")
        if not isinstance(updates, dict) or key not in updates:
            continue
        raw = updates[key]
        if spec.kind == "categorical":
            matched = _match_choice(raw, spec.choices or ())
            if matched is not None:
                values.append(matched)
            continue
        number = _num(raw)
        if number is None:
            continue
        values.append(int(round(number)) if spec.kind == "int" else number)
    return values


def _match_choice(raw: Any, choices: Sequence[Any]) -> Optional[Any]:
    if isinstance(raw, str) and raw.lower() in ("true", "false"):
        raw = raw.lower() == "true"
    if raw in choices:
        return raw
    as_str = str(raw)
    for choice in choices:
        if str(choice) == as_str:
            return choice
    return None


def _by_category(
    history: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in history:
        name = row.get("search_category")
        if name not in SPACES:
            continue
        if _num(row.get("avg_combined_final")) is None:
            continue
        grouped.setdefault(str(name), []).append(row)
    return grouped


def _top_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    ranked = sorted(rows, key=lambda row: _num(row.get("avg_combined_final")) or -1.0, reverse=True)
    return list(ranked[:TOP_K])


def _best_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    if not rows:
        return None
    return max(rows, key=lambda row: _num(row.get("avg_combined_final")) or -1.0)


def _best_score(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    scores = [_num(row.get("avg_combined_final")) for row in rows]
    scores = [score for score in scores if score is not None]
    return max(scores) if scores else None


def _metrics(row: Optional[Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    source = row or {}
    return {
        "avg_f1_snp": _num(source.get("avg_f1_snp")),
        "avg_f1_indel": _num(source.get("avg_f1_indel")),
        "avg_fp_per_target": _num(source.get("avg_fp_per_target")),
        "avg_core": _num(source.get("avg_core")),
    }


def _round_float(value: float) -> float:
    return float(f"{value:.6g}")


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
