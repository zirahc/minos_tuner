"""Multi-step search box: diagnose, review, choose, tighten.

Pure functions. No Supabase and no LLM. tuner.agent runs this, then may
ask the model to revise the draft.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tuner.search_box import enough_trials, specs_to_space_json
from tuner.spaces import (
    SPACES,
    coarse_step,
    ParamSpec,
    categories_touched,
    category_instruction,
    category_of,
    coerce_param,
    full_pass_category,
    is_agent_experiment,
    space_for,
    spec_of,
)
from tuner.v2_score import component_losses

# Order is the next lever for that limiter. A plateaued category is skipped.
BOTTLENECK_PLAN: Dict[str, Tuple[str, ...]] = {
    "no_history": ("quality_filters",),
    "false_positives": (
        "quality_filters",
        "calling_confidence",
        "pair_hmm",
        "priors",
        "downsampling",
    ),
    "indel": ("pcr", "assembly", "pair_hmm", "priors"),
    "sensitivity": (
        "active_region",
        "calling_confidence",
        "assembly",
        "priors",
        "quality_filters",
    ),
    "broad": (
        "pcr",
        "assembly",
        "active_region",
        "calling_confidence",
        "priors",
        "pair_hmm",
        "quality_filters",
        "downsampling",
    ),
}

# A category that does not beat the current best by more than this
# is not given more trials. The first goal is to move 0.75 toward 0.9.
MIN_GAIN = 0.01
IMPROVE_MARGIN = MIN_GAIN
EXHAUST_MIN_TRIALS = 4
SHRINK_MIN_TRIALS = 3
TOP_K = 3
# One lucky row cannot decide the side of a range. A setting needs this
# many scores before it can pull the experiment away from the best config.
MIN_BIN_ROWS = 3


def multistep_search_box(
    history: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (search box, report). The box still needs validate_search_box."""
    report = diagnose(history)
    category, reason, mode = choose_category(report)
    if mode == "experiment":
        built = _v2_experiment(history, report)
        if built is None:
            mode = "stop"
            category = (report.get("visit_order") or ["quality_filters"])[0]
            reason = (
                "no history setting moved avg_combined_final by more than "
                f"{MIN_GAIN:.2f} on the remaining v2 loss"
            )
            space = specs_to_space_json(space_for(category))
        else:
            category, space, reason = built
    scored = _rows_for_category(history, category) if mode != "experiment" else []
    shrink = mode == "refine" and len(scored) >= SHRINK_MIN_TRIALS
    if mode != "experiment":
        space = _space_json(category, scored if shrink else [])
    if shrink and not _searchable(space):
        category, reason, mode = _next_searchable(report, category)
        scored = _rows_for_category(history, category)
        shrink = mode == "refine" and len(scored) >= SHRINK_MIN_TRIALS
        space = _space_json(category, scored if shrink else [])
        if not _searchable(space):
            space = specs_to_space_json(space_for(category))
    if is_agent_experiment(category):
        bounds = "directed"
    else:
        bounds = "shrunk" if space != specs_to_space_json(space_for(category)) else "full"
    n_trials = enough_trials(space)
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
        "run_experiment": mode != "stop",
    }
    return box, report


def diagnose(history: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scored = [row for row in history if _num(row.get("avg_combined_final")) is not None]
    grouped = _by_category(scored)
    categories: Dict[str, Any] = {}
    # Catalog categories first. An agent experiment that is not one of those
    # categories (combo, or a later name) is added as its own row.
    names = list(SPACES)
    for name in grouped:
        if name not in SPACES:
            names.append(name)
    for name in names:
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
    full_passes = {name: 0 for name in names}
    for row in scored:
        passed = full_pass_category(row)
        if passed:
            full_passes[passed] = full_passes.get(passed, 0) + 1
    visit = _visit_order(bottleneck)
    needs_screen = {
        name: _category_needs_screen(scored, name) for name in SPACES
    }
    open_experiment = _open_v2_experiment(scored)
    last = scored[-1] if scored else {}
    touched = categories_touched(last) if last else ()
    stored = str(last.get("search_category") or "")
    if stored in touched:
        last_category = stored
    elif touched:
        last_category = touched[0]
    else:
        last_category = None
    return {
        "bottleneck": bottleneck,
        "n_scored": len(scored),
        "best_avg_combined_final": _num(best_row.get("avg_combined_final")) if best_row else None,
        "best_category": best_row.get("search_category") if best_row else None,
        "metrics": metrics,
        "losses": losses,
        "categories": categories,
        "last_category": last_category,
        "last_state": (categories.get(last_category) or {}).get("state") if last_category else None,
        "last_hypothesis": (last.get("hypothesis") or "") if last else "",
        "full_passes": full_passes,
        "visit_order": visit,
        "needs_screen": needs_screen,
        "open_experiment": open_experiment,
    }


def choose_category(report: Mapping[str, Any]) -> Tuple[str, str, str]:
    """One category per stack, on top of the best full config.

    A stack varies one category. The other parameters stay at the best
    scored values. After that category has a full-config stack, the next
    category in the visit order is opened. Old rows that changed only a
    few keys do not count as that pass.
    """
    order = list(report.get("visit_order") or _visit_order(str(report.get("bottleneck") or "broad")))
    passes = report.get("full_passes") if isinstance(report.get("full_passes"), dict) else {}
    if report.get("bottleneck") == "no_history" or not any(int(passes.get(name) or 0) for name in order):
        first = order[0]
        return (
            first,
            f"full-config pass starts at {first}; the other parameters stay at the best config",
            "explore",
        )
    open_name = report.get("open_experiment")
    if isinstance(open_name, str) and open_name.startswith("v2_"):
        return (
            open_name,
            f"continue {open_name} until it has {EXHAUST_MIN_TRIALS} scores",
            "experiment",
        )
    needs = report.get("needs_screen") if isinstance(report.get("needs_screen"), dict) else {}
    for name in order:
        if needs.get(name):
            done = int(passes.get(name) or 0)
            return (
                name,
                f"keep the best config and vary {name} ({done} full-config scores so far)",
                "explore",
            )
    return (
        order[0],
        "category screens are done; the next experiment comes from the largest "
        "v2 loss and the settings that raised avg_combined_final",
        "experiment",
    )


def _visit_order(bottleneck: str) -> List[str]:
    """Limiter categories first, then every remaining catalog category."""
    plan = list(BOTTLENECK_PLAN.get(bottleneck) or BOTTLENECK_PLAN["broad"])
    for name in SPACES:
        if name not in plan:
            plan.append(name)
    return plan


def _category_needs_screen(
    history: Sequence[Mapping[str, Any]], category: str
) -> bool:
    """True until this category has a screen, or the best config changed under it.

    A finished screen within MIN_GAIN of the best does not get more trials.
    It is tried again only after another category raises the best by more than MIN_GAIN.
    """
    full = [
        row for row in history
        if full_pass_category(row) == category and _num(row.get("avg_combined_final")) is not None
    ]
    if len(full) < EXHAUST_MIN_TRIALS:
        return True
    cat_best = _best_score(full)
    prior_scores = [
        score for row in history
        if full_pass_category(row) != category
        for score in [_num(row.get("avg_combined_final"))]
        if score is not None
    ]
    prior = max(prior_scores) if prior_scores else None
    if cat_best is None:
        return True
    if prior is None or cat_best + MIN_GAIN >= prior:
        return False
    best_row = _best_row(history)
    if best_row is None or full_pass_category(best_row) == category:
        return False
    latest = max(str(row.get("created_at") or "") for row in full)
    return latest < str(best_row.get("created_at") or "")


def _v2_experiment(
    history: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """A new experiment aimed at the largest v2 loss.

    Parameters are catalog settings whose history values already separated
    avg_combined_final. Numeric ranges sit on the side that scored higher.
    """
    open_name = _open_v2_experiment(history)
    if open_name:
        parsed = _keys_from_experiment(open_name)
        if parsed:
            loss, keys = parsed
            space = _directed_space(history, keys)
            if _searchable(space):
                return (
                    open_name,
                    space,
                    f"continue {open_name}; it is still the v2 experiment for {loss}",
                )
    ranked = _ranked_movers(history)
    if not ranked:
        return None
    losses = report.get("losses") if isinstance(report.get("losses"), dict) else {}
    for loss in _loss_priority(losses):
        preferred = set(_loss_categories(loss))
        ordered = [item for item in ranked if category_of(item[1]) in preferred]
        ordered.extend(item for item in ranked if category_of(item[1]) not in preferred)
        if not any(category_of(item[1]) in preferred for item in ordered):
            continue
        for start in range(len(ordered)):
            picked = ordered[start : start + 4]
            if not picked or not any(category_of(item[1]) in preferred for item in picked):
                continue
            keys = [item[1] for item in picked]
            name = _experiment_name(loss, keys)
            if _experiment_done(history, name):
                continue
            space = _directed_space(history, keys)
            if not _searchable(space):
                continue
            return name, space, _experiment_reason(loss, losses.get(loss), picked)
    return None


def _loss_priority(losses: Mapping[str, Any]) -> List[str]:
    named = []
    for key in ("indel", "fp", "snp"):
        value = losses.get(key)
        if isinstance(value, float):
            named.append((value, key))
    named.sort(reverse=True)
    order = [key for _value, key in named]
    largest = losses.get("largest")
    if isinstance(largest, str) and largest in order:
        order.remove(largest)
        order.insert(0, largest)
    return order or ["indel", "fp", "snp"]


def _loss_categories(loss: str) -> Tuple[str, ...]:
    if loss == "fp":
        return BOTTLENECK_PLAN["false_positives"]
    if loss == "snp":
        return BOTTLENECK_PLAN["sensitivity"]
    return BOTTLENECK_PLAN["indel"]


def _ranked_movers(
    history: Sequence[Mapping[str, Any]],
) -> List[Tuple[float, str, ParamSpec, Dict[str, List[float]]]]:
    cache: Dict[str, set] = {}
    ranked = []
    for specs in SPACES.values():
        for key, spec in specs.items():
            bins = _param_bins(history, key, spec, cache)
            means = list(_supported_means(bins).values())
            if len(means) < 2:
                continue
            spread = max(means) - min(means)
            if spread > MIN_GAIN:
                ranked.append((spread, key, spec, bins))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def _directed_space(
    history: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> Dict[str, Any]:
    cache: Dict[str, set] = {}
    specs: Dict[str, ParamSpec] = {}
    bins_by_key: Dict[str, Dict[str, List[float]]] = {}
    for key in keys:
        spec = spec_of(key)
        if spec is None:
            continue
        specs[key] = spec
        bins_by_key[key] = _param_bins(history, key, spec, cache)
    space = specs_to_space_json(specs)
    for key, spec in specs.items():
        space[key] = _directed_item(
            spec,
            bins_by_key.get(key) or {},
            space[key],
            _best_setting(history, key, spec),
        )
    return space


def _directed_item(
    spec: ParamSpec,
    bins: Mapping[str, Sequence[float]],
    full: Dict[str, Any],
    anchor: Optional[float] = None,
) -> Dict[str, Any]:
    """Keep the catalog grid, but drop the numeric side that scored worse.

    The best config's value stays inside the range. A new experiment that
    excludes it changes every trial, and those trials come back failed.
    """
    if spec.kind not in ("int", "float") or spec.log or spec.low is None or spec.high is None:
        return full
    means = _supported_means(bins)
    if len(means) < 2:
        return full
    winner_label = max(means, key=lambda label: means[label])
    winner_mean = means[winner_label]
    kept = [
        float(label)
        for label, mean in means.items()
        if mean + MIN_GAIN >= winner_mean
    ]
    if not kept:
        return full
    low = min(kept)
    high = max(kept)
    catalog_low = float(spec.low)
    catalog_high = float(spec.high)
    step = coarse_step(spec) or 0.0
    if high <= low and step > 0:
        winner = float(winner_label)
        if winner <= catalog_low:
            low, high = winner, min(catalog_high, winner + step)
        elif winner >= catalog_high:
            low, high = max(catalog_low, winner - step), winner
        else:
            low, high = max(catalog_low, winner - step), min(catalog_high, winner + step)
    if anchor is not None:
        low = min(low, float(anchor))
        high = max(high, float(anchor))
    low, high = _snap_span(low, high, step, catalog_low, catalog_high)
    if high < low:
        return full
    item = dict(full)
    if spec.kind == "int":
        item["low"] = int(round(low))
        item["high"] = int(round(high))
    else:
        item["low"] = _round_float(low)
        item["high"] = _round_float(high)
    return item


def _experiment_name(loss: str, keys: Sequence[str]) -> str:
    return "v2_" + loss + "__" + "__".join(keys)


def _keys_from_experiment(name: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
    if not name.startswith("v2_") or "__" not in name:
        return None
    loss, _, tail = name[3:].partition("__")
    keys = tuple(part for part in tail.split("__") if part and spec_of(part) is not None)
    if not loss or not keys:
        return None
    return loss, keys


def _open_v2_experiment(history: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for row in reversed(list(history)):
        name = str(row.get("search_category") or "").strip()
        if not name.startswith("v2_"):
            continue
        if _experiment_done(history, name):
            return None
        return name
    return None


def _experiment_done(history: Sequence[Mapping[str, Any]], name: str) -> bool:
    rows = [
        row for row in history
        if str(row.get("search_category") or "").strip() == name
        and _num(row.get("avg_combined_final")) is not None
    ]
    return len(rows) >= EXHAUST_MIN_TRIALS


def _supported_means(bins: Mapping[str, Sequence[float]]) -> Dict[str, float]:
    return {
        label: _mean(values)
        for label, values in bins.items()
        if len(values) >= MIN_BIN_ROWS
    }


def _best_setting(
    history: Sequence[Mapping[str, Any]], key: str, spec: ParamSpec
) -> Optional[float]:
    """Snapped value of this parameter on the highest-scoring row."""
    best_score: Optional[float] = None
    best_value: Optional[float] = None
    for row in history:
        score = _num(row.get("avg_combined_final"))
        if score is None:
            continue
        updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
        config = row.get("gatk_config") if isinstance(row.get("gatk_config"), dict) else {}
        if key not in updates and key not in config:
            continue
        raw = updates[key] if key in updates else config[key]
        try:
            snapped = coerce_param(spec, raw)
        except (TypeError, ValueError):
            continue
        if isinstance(snapped, (bool, str)):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_value = float(snapped)
    return best_value


def _snap_span(
    low: float, high: float, step: float, catalog_low: float, catalog_high: float
) -> Tuple[float, float]:
    """Coarse grid points inside the range. Do not pull the range back to a worse end."""
    low = max(catalog_low, min(catalog_high, low))
    high = max(catalog_low, min(catalog_high, high))
    if high < low:
        return catalog_low, catalog_high
    if step <= 0:
        return low, high
    limit = int(math.floor((catalog_high - catalog_low) / step + 1e-9))
    inside = []
    for n in range(limit + 1):
        value = catalog_low + n * step
        if low - 1e-9 <= value <= high + 1e-9:
            inside.append(value)
    if len(inside) >= 2:
        return inside[0], inside[-1]
    if not inside:
        n = int(round(((low + high) / 2 - catalog_low) / step))
        n = max(0, min(limit, n))
        center = catalog_low + n * step
    else:
        center = inside[0]
    neighbors = [center]
    if center - step >= catalog_low - 1e-9:
        neighbors.append(center - step)
    if center + step <= catalog_high + 1e-9:
        neighbors.append(center + step)
    return min(neighbors), max(neighbors)


def _experiment_reason(loss: str, loss_value: Any, picked: Sequence[Tuple[Any, ...]]) -> str:
    amount = f"{loss_value:.4f}" if isinstance(loss_value, float) else "na"
    directions = []
    for spread, key, _spec, bins in picked:
        means = _supported_means(bins)
        winner = max(means, key=lambda label: means[label]) if means else "?"
        directions.append(f"{key} toward {winner} (spread {spread:.4f})")
    return (
        f"v2 {loss} loss is {amount}. History raised avg_combined_final with "
        + "; ".join(directions)
        + ". This experiment varies those settings."
    )


def _param_bins(
    history: Sequence[Mapping[str, Any]],
    key: str,
    spec: ParamSpec,
    cache: Dict[str, set],
) -> Dict[str, List[float]]:
    owner = category_of(key)
    bins: Dict[str, List[float]] = {}
    for row in history:
        if not _row_informs_param(history, row, key, owner, cache):
            continue
        updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
        config = row.get("gatk_config") if isinstance(row.get("gatk_config"), dict) else {}
        if key not in updates and key not in config:
            continue
        raw = updates[key] if key in updates else config[key]
        try:
            snapped = coerce_param(spec, raw)
        except (TypeError, ValueError):
            continue
        score = _num(row.get("avg_combined_final"))
        if score is None:
            continue
        bins.setdefault(str(snapped), []).append(score)
    return bins


def _row_informs_param(
    history: Sequence[Mapping[str, Any]],
    row: Mapping[str, Any],
    key: str,
    owner: Optional[str],
    cache: Dict[str, set],
) -> bool:
    """Use a row only when this key was part of the experiment, not a held base value."""
    passed = full_pass_category(row)
    if passed in SPACES:
        return passed == owner
    if passed:
        named = _keys_from_experiment(passed)
        if named is not None:
            return key in named[1]
        return key in _varied_keys(history, passed, cache)
    return bool(owner) and owner in categories_touched(row)


def _varied_keys(
    history: Sequence[Mapping[str, Any]], name: str, cache: Dict[str, set]
) -> set:
    if name in cache:
        return cache[name]
    grouped: Dict[str, set] = {}
    for row in history:
        if str(row.get("search_category") or "").strip() != name:
            continue
        updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
        config = row.get("gatk_config") if isinstance(row.get("gatk_config"), dict) else {}
        source = updates or config
        for param, raw in source.items():
            grouped.setdefault(str(param), set()).add(str(raw))
    cache[name] = {param for param, values in grouped.items() if len(values) > 1}
    return cache[name]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _rows_for_category(
    history: Sequence[Mapping[str, Any]], category: str
) -> List[Mapping[str, Any]]:
    full = [row for row in history if full_pass_category(row) == category]
    if full:
        return full
    return _by_category(history).get(category, [])


def _hypothesis(report: Mapping[str, Any], category: str, reason: str, bounds: str) -> str:
    best = report.get("best_avg_combined_final")
    best_txt = f"{best:.4f}" if isinstance(best, float) else "none"
    return (
        f"v2 point loss {_fmt_losses(report.get('losses'))}. "
        f"Bottleneck={report['bottleneck']} (best avg_combined_final={best_txt}). "
        f"{reason}. {category_instruction(category)} "
        f"Bounds={bounds}. Confirm or reject against the next scored rows."
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
    order = list(report.get("visit_order") or _visit_order(str(report.get("bottleneck") or "broad")))
    passes = report.get("full_passes") if isinstance(report.get("full_passes"), dict) else {}
    for name in order:
        if name == blocked:
            continue
        if int(passes.get(name) or 0) < EXHAUST_MIN_TRIALS:
            return name, f"{blocked} has no room left; next is {name}", "explore"
    for name in order:
        if name != blocked:
            return name, f"{blocked} has no room left; vary {name} again on the best config", "refine"
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
    # A shorter choice list is a different CategoricalDistribution. Optuna
    # rejects that on a study that already has the full list.
    del rows, key
    return list(allowed)


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
        if _num(row.get("avg_combined_final")) is None:
            continue
        for name in categories_touched(row):
            grouped.setdefault(name, []).append(row)
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
