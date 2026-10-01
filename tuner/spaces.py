"""GATK search spaces: one category = one Optuna study.

Ranges match the validator tool schema. A submitted value outside these
bounds is defaulted, so Optuna never suggests past them.
sample_ploidy and dont_use_soft_clipped_bases have only one allowed value,
so they stay at the GATK default and are not searched.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

Kind = str  # "int" | "float" | "categorical"


@dataclass(frozen=True)
class ParamSpec:
    kind: Kind
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[Tuple[Any, ...]] = None
    note: str = ""


# One sentence the agent uses as the instruction for that category.
CATEGORY_GUIDE: Dict[str, str] = {
    "quality_filters": (
        "Moves core F1 and fp_per_target. Lower base-quality cuts keep more "
        "bases (higher recall, more noise). Higher mapping quality cuts "
        "false positives from mismapped reads."
    ),
    "calling_confidence": (
        "Moves core F1 and fp_per_target. A lower phred threshold emits more "
        "true variants and more false positives. emit_ref_confidence changes "
        "which sites are written, from variants only to every position."
    ),
    "pcr": (
        "Moves indel core F1, especially 1bp indels (40% of core). NONE keeps "
        "more indel calls. HOSTILE drops the most indel artifacts and can drop true indels."
    ),
    "assembly": (
        "Moves indel and complex-allele core F1, and fp_per_target. Lower "
        "pruning and shorter dangling branches keep weak paths. Higher pruning "
        "or LOD cuts noise and can drop true variants."
    ),
    "active_region": (
        "Moves core F1. A site that is not marked active is never called. "
        "Region size and padding change recall for variants near a boundary."
    ),
    "pair_hmm": (
        "Moves long-indel core F1 (4bp and longer) and fp_per_target. A higher "
        "gap penalty hurts long indels. A lower mismapping rate trusts reads less."
    ),
    "priors": (
        "heterozygosity moves snp_het core F1 (18% of core). "
        "indel_heterozygosity moves indel core F1 (80% of core). "
        "contamination_fraction_to_filter moves fp_per_target and can also cut true alt support."
    ),
    "downsampling": (
        "Moves core F1 where coverage is high. A lower cap drops reads that "
        "start at the same position. The floor is 25, so downsampling stays on."
    ),
    "combo": (
        "Varies only parameters that already moved avg_combined_final, "
        "including parameters from different categories. The rest stay at the best config."
    ),
}

DEFERRED: Tuple[Dict[str, str], ...] = ()
# Cross-category experiment. Not a GATK parameter group.
COMBO_CATEGORY = "combo"


SPACES: Dict[str, Dict[str, ParamSpec]] = {
    "quality_filters": {
        "min_base_quality_score": ParamSpec(
            "int", 10, low=10, high=50,
            note=(
                "Bases below this are ignored. Lower raises core F1 recall and "
                "fp_per_target. Higher can drop real signal from core F1."
            ),
        ),
        "min_mapping_quality_score": ParamSpec(
            "int", 20, low=0, high=60,
            note=(
                "Reads with mapQ below this are discarded. Higher cuts fp_per_target "
                "from mismapped reads. Lower keeps more repeat noise in fp_per_target."
            ),
        ),
        "base_quality_score_threshold": ParamSpec(
            "int", 18, low=0, high=50,
            note=(
                "Bases below this are floored, not removed. It softens their pull on "
                "core F1 and fp_per_target, together with min_base_quality_score."
            ),
        ),
    },
    "calling_confidence": {
        "standard_min_confidence_threshold_for_calling": ParamSpec(
            "float", 30.0, low=30.0, high=100.0,
            note=(
                "Minimum phred confidence to emit a call. 30 is 99.9% confidence. "
                "Lower raises core F1 recall and fp_per_target. Higher cuts both."
            ),
        ),
        "emit_ref_confidence": ParamSpec(
            "categorical",
            "NONE",
            choices=("NONE", "GVCF", "BP_RESOLUTION"),
            note=(
                "NONE writes variant sites only. GVCF and BP_RESOLUTION write more "
                "sites, which can change fp_per_target and the callset the gate sees."
            ),
        ),
    },
    "pcr": {
        "pcr_indel_model": ParamSpec(
            "categorical",
            "CONSERVATIVE",
            choices=("NONE", "HOSTILE", "AGGRESSIVE", "CONSERVATIVE"),
            note=(
                "PCR indel artifact filter. NONE keeps more indel calls in core F1. "
                "HOSTILE drops the most indel false positives and can drop true 1bp indels."
            ),
        ),
    },
    "assembly": {
        "min_pruning": ParamSpec(
            "int", 2, low=2, high=10,
            note=(
                "Minimum reads for a path to survive. Lower raises core F1 on weak "
                "alleles and fp_per_target. Higher demands more evidence."
            ),
        ),
        "max_alternate_alleles": ParamSpec(
            "int", 6, low=1, high=20,
            note=(
                "Maximum alternate alleles at one site. Higher can raise core F1 at "
                "multi-allelic sites. Lower keeps only the top alleles."
            ),
        ),
        "min_dangling_branch_length": ParamSpec(
            "int", 4, low=2, high=20,
            note=(
                "Shortest dangling branch to recover. Lower raises indel core F1 from "
                "partial reads. Higher cuts short-branch false positives."
            ),
        ),
        "recover_all_dangling_branches": ParamSpec(
            "categorical",
            False,
            choices=(False, True),
            note=(
                "True recovers every dangling branch. That raises indel core F1 and "
                "can raise fp_per_target."
            ),
        ),
        "max_num_haplotypes_in_population": ParamSpec(
            "int", 128, low=8, high=128,
            note=(
                "Maximum haplotypes evaluated. Higher can raise core F1 in complex "
                "regions. Lower can miss those alleles."
            ),
        ),
        "adaptive_pruning_initial_error_rate": ParamSpec(
            "float", 0.001, low=0.0001, high=0.1, log=True,
            note=(
                "Starting error rate for pruning. Higher prunes harder, cutting "
                "fp_per_target and possibly core F1. Lower keeps more paths."
            ),
        ),
        "pruning_lod_threshold": ParamSpec(
            "float", 2.302585, low=0.5, high=10.0,
            note=(
                "Log-odds cut for pruning. The default ~2.3 is 10:1 odds. Higher "
                "cuts fp_per_target and can cut core F1. Moves with the error rate."
            ),
        ),
    },
    "active_region": {
        "active_probability_threshold": ParamSpec(
            "float", 0.002, low=0.001, high=0.05, log=True,
            note=(
                "Minimum probability for a site to be assembled. Lower raises core F1 "
                "by examining more sites. Higher leaves those sites uncalled."
            ),
        ),
        "min_assembly_region_size": ParamSpec(
            "int", 50, low=1, high=300,
            note=(
                "Regions shorter than this are extended. Size changes core F1 for an "
                "isolated variant by adding or withholding context."
            ),
        ),
        "max_assembly_region_size": ParamSpec(
            "int", 300, low=100, high=700,
            note=(
                "Regions longer than this are split. A split can drop core F1 when "
                "a variant, especially an indel, crosses the boundary."
            ),
        ),
        "assembly_region_padding": ParamSpec(
            "int", 100, low=0, high=500,
            note=(
                "Flanking bases around each region. Higher raises core F1 for variants "
                "near a boundary. Lower gives the assembler less context."
            ),
        ),
    },
    "pair_hmm": {
        "pair_hmm_gap_continuation_penalty": ParamSpec(
            "int", 10, low=1, high=30,
            note=(
                "Penalty for extending a gap. Higher lowers core F1 for indels of "
                "4bp and longer. Lower tolerates those long gaps."
            ),
        ),
        "phred_scaled_global_read_mismapping_rate": ParamSpec(
            "int", 45, low=10, high=60,
            note=(
                "Assumed mismapping rate, phred scale. 45 is about 1 in 30,000. "
                "Lower cuts fp_per_target and can cut SNP core F1. Higher trusts reads more."
            ),
        ),
    },
    "priors": {
        "heterozygosity": ParamSpec(
            "float", 0.001, low=0.0001, high=0.01, log=True,
            note=(
                "Prior for a heterozygous SNP. Higher raises snp_het core F1 "
                "(18% of core). Lower is more conservative. Default 0.001 is about 1 per 1000 bases."
            ),
        ),
        "indel_heterozygosity": ParamSpec(
            "float", 0.000125, low=0.00001, high=0.001, log=True,
            note=(
                "Prior for a heterozygous indel. Higher raises indel core F1 "
                "(80% of core). Lower makes those calls more conservative."
            ),
        ),
        "contamination_fraction_to_filter": ParamSpec(
            "float", 0.0, low=0.0, high=0.05,
            note=(
                "Estimated contaminating fraction. Above 0, alt support is reduced, "
                "which lowers fp_per_target and can also lower core F1 recall."
            ),
        ),
    },
    "downsampling": {
        "max_reads_per_alignment_start": ParamSpec(
            "int", 50, low=25, high=300,
            note=(
                "Maximum reads kept per alignment start. Too low can drop core F1 "
                "at high depth. The minimum is 25, so downsampling stays on."
            ),
        ),
    },
}


def known_categories() -> Tuple[str, ...]:
    return tuple(sorted(SPACES))


def full_pass_category(row: Mapping[str, Any]) -> Optional[str]:
    """Category varied on a full config. Partial old rows return None.

    A search_category outside the catalog is its own experiment. Combo is
    one of those. It is counted under that name, not under a GATK category.
    """
    updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
    found: List[str] = []
    for key in updates:
        for name, specs in SPACES.items():
            if key in specs and name not in found:
                found.append(name)
    stored = str(row.get("search_category") or "").strip()
    if stored and stored not in SPACES:
        return stored
    if stored in SPACES and len(found) > 2:
        return stored
    return None


def categories_touched(row: Mapping[str, Any]) -> Tuple[str, ...]:
    """Current categories an experiment belongs to.

    A full-config row counts only as the category it varied. Older rows
    that set a few keys are counted from those keys.
    """
    passed = full_pass_category(row)
    if passed:
        return (passed,)
    updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
    found: List[str] = []
    for key in updates:
        for name, specs in SPACES.items():
            if key in specs and name not in found:
                found.append(name)
    if found:
        return tuple(found)
    stored = str(row.get("search_category") or "").strip()
    if stored:
        return (stored,)
    return ()


def catalog_defaults() -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for specs in SPACES.values():
        for key, spec in specs.items():
            params[key] = spec.default
    return params


def full_params_from_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Every catalog parameter on the config that was actually scored.

    gatk_updates wins, then gatk_config, then the catalog default.
    Keys outside the catalog are copied from gatk_config so the file matches that run.
    """
    updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
    config = row.get("gatk_config") if isinstance(row.get("gatk_config"), dict) else {}
    params = catalog_defaults()
    for key, value in config.items():
        if key not in params:
            params[key] = value
    for specs in SPACES.values():
        for key, spec in specs.items():
            if key in updates:
                raw = updates[key]
            elif key in config:
                raw = config[key]
            else:
                continue
            try:
                params[key] = coerce_param(spec, raw)
            except (TypeError, ValueError):
                params[key] = spec.default
    return params


def best_base_updates(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Optional[Mapping[str, Any]]]:
    """Full parameter set from the highest avg_combined_final row."""
    best_row: Optional[Mapping[str, Any]] = None
    best_score: Optional[float] = None
    for row in rows:
        try:
            score = float(row.get("avg_combined_final"))
        except (TypeError, ValueError):
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return catalog_defaults(), None
    return full_params_from_row(best_row), best_row


def space_for(category: str) -> Dict[str, ParamSpec]:
    if category not in SPACES:
        known = ", ".join(known_categories())
        raise ValueError(f"unknown search_category {category!r}. known: {known}")
    return SPACES[category]


def default_study_name(category: str) -> str:
    return f"gatk-v2-{category}"


def is_agent_experiment(category: str) -> bool:
    """True for a new experiment name. Catalog categories stay in SPACES."""
    name = str(category or "").strip()
    return bool(name) and name not in SPACES


def category_instruction(category: str) -> str:
    if category in CATEGORY_GUIDE:
        return CATEGORY_GUIDE[category]
    if str(category or "").startswith("v2_"):
        return (
            "Large experiment toward avg_combined_final 0.9. Vary several catalog "
            "parameters at once. The same experiment with different values is allowed. "
            "Do not repeat a setting that history already scored. Other parameters "
            "stay at the best config."
        )
    return ""


def agent_reference() -> Dict[str, Any]:
    """Catalog, notes, and deferred flags for the agent prompt."""
    categories: Dict[str, Any] = {}
    for name, specs in SPACES.items():
        categories[name] = {
            "instruction": category_instruction(name),
            "parameters": {
                key: _spec_note(spec) for key, spec in specs.items()
            },
        }
    return {
        "categories": categories,
        "deferred": list(DEFERRED),
        "deferred_rule": "Every catalog parameter may be searched. sample_ploidy and dont_use_soft_clipped_bases stay at the GATK default.",
    }


def _spec_note(spec: ParamSpec) -> Dict[str, Any]:
    item: Dict[str, Any] = {"type": spec.kind, "default": spec.default}
    if spec.kind in ("int", "float"):
        item["low"] = spec.low
        item["high"] = spec.high
    if spec.log:
        item["log"] = True
    if spec.kind == "categorical":
        item["choices"] = list(spec.choices or ())
    if spec.note:
        item["note"] = spec.note
    return item


def coarse_step(spec: ParamSpec) -> Optional[float]:
    """Large step for the first search. A step of 1 or 2 is not used on a wide range.

    standard_min_confidence_threshold_for_calling is 30, 40, 50, ... not 32, 34, 36.
    """
    if spec.kind not in ("int", "float") or spec.log:
        return None
    if spec.low is None or spec.high is None:
        return None
    span = float(spec.high) - float(spec.low)
    if span <= 0:
        return None
    if spec.kind == "float" and span >= 50:
        return 10.0
    target = span / 4
    nice = (2, 4, 5, 10, 15, 20, 25, 30, 40, 50, 100)
    if spec.kind == "int":
        target = max(2.0, target)
        if span >= 20:
            target = max(target, 5.0)
        if span >= 40:
            target = max(target, 10.0)
        step = min(nice, key=lambda item: (abs(item - target), -item))
        while step > span / 2 and step > 2:
            smaller = [item for item in nice if item < step]
            if not smaller:
                break
            step = smaller[-1]
        return float(int(step))
    return float(target)


def coarse_levels(spec: ParamSpec) -> Optional[Tuple[float, ...]]:
    """Four log-spaced levels. Adjacent floats on a log range are not separate trials."""
    if not spec.log or spec.low is None or spec.high is None:
        return None
    low = float(spec.low)
    high = float(spec.high)
    if low <= 0 or high <= low:
        return None
    levels: List[float] = []
    for index in range(4):
        weight = index / 3
        value = math.exp(math.log(low) + weight * (math.log(high) - math.log(low)))
        levels.append(_round_sig(value))
    ordered: List[float] = []
    for value in levels:
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


def spec_of(key: str) -> Optional[ParamSpec]:
    for specs in SPACES.values():
        if key in specs:
            return specs[key]
    return None


def category_of(key: str) -> Optional[str]:
    for name, specs in SPACES.items():
        if key in specs:
            return name
    return None


def snap_to_grid(spec: ParamSpec, value: float) -> Any:
    levels = coarse_levels(spec)
    if levels:
        def distance(level: float) -> float:
            return abs(math.log(max(level, 1e-12)) - math.log(max(value, 1e-12)))
        return min(levels, key=distance)
    step = coarse_step(spec)
    if step is None or spec.low is None or spec.high is None:
        return int(round(value)) if spec.kind == "int" else float(value)
    low = float(spec.low)
    high = float(spec.high)
    max_n = int(math.floor((high - low) / step + 1e-9))
    n = int(round((value - low) / step))
    n = max(0, min(max_n, n))
    snapped = low + n * step
    if spec.kind == "int":
        return int(round(snapped))
    return float(snapped)


def _round_sig(value: float) -> float:
    if value == 0:
        return 0.0
    digits = math.floor(math.log10(abs(value)))
    return round(value, -int(digits))


def coerce_param(spec: ParamSpec, raw: Any) -> Any:
    """Map a stored gatk_config value onto the coarse Optuna grid for this spec."""
    if spec.kind == "int":
        value = int(round(float(raw)))
        if spec.low is not None:
            value = max(int(spec.low), value)
        if spec.high is not None:
            value = min(int(spec.high), value)
        return snap_to_grid(spec, float(value))
    if spec.kind == "float":
        value = float(raw)
        if spec.low is not None:
            value = max(float(spec.low), value)
        if spec.high is not None:
            value = min(float(spec.high), value)
        return snap_to_grid(spec, value)
    if spec.kind == "categorical":
        choices = spec.choices or ()
        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            raw = raw.lower() == "true"
        if raw in choices:
            return raw
        as_str = str(raw)
        for choice in choices:
            if str(choice) == as_str:
                return choice
        raise ValueError(f"{raw!r} not in {choices}")
    raise ValueError(f"unknown spec kind {spec.kind!r}")


def params_from_row(
    row: Mapping[str, Any],
    category: str,
    keys: Optional[Tuple[str, ...]] = None,
) -> Dict[str, Any]:
    """Build the Optuna param dict from a gatk_config_scores row.

    Prefers gatk_updates for keys in this category, then full gatk_config,
    then the space default. Every requested space key is always present.
    """
    spec_map = space_for(category)
    if keys:
        spec_map = {k: spec_map[k] for k in keys if k in spec_map}
    updates = row.get("gatk_updates") if isinstance(row.get("gatk_updates"), dict) else {}
    config = row.get("gatk_config") if isinstance(row.get("gatk_config"), dict) else {}
    params: Dict[str, Any] = {}
    for key, spec in spec_map.items():
        raw = updates[key] if key in updates else config.get(key, spec.default)
        params[key] = coerce_param(spec, raw)
    return params
