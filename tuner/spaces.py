"""GATK search spaces: one category = one Optuna study.

Only these keys are imported from historical gatk_config rows.
Other GATK params stay frozen for that study.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

Kind = str  # "int" | "float" | "categorical"


@dataclass(frozen=True)
class ParamSpec:
    kind: Kind
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    log: bool = False
    choices: Optional[Tuple[Any, ...]] = None


SPACES: Dict[str, Dict[str, ParamSpec]] = {
    "quality_filters": {
        "min_base_quality_score": ParamSpec("int", 10, low=0, high=50),
        "min_mapping_quality_score": ParamSpec("int", 20, low=0, high=60),
        "base_quality_score_threshold": ParamSpec("int", 18, low=0, high=50),
        "standard_min_confidence_threshold_for_calling": ParamSpec(
            "float", 30.0, low=10.0, high=50.0
        ),
    },
    "pcr": {
        "pcr_indel_model": ParamSpec(
            "categorical",
            "CONSERVATIVE",
            choices=("NONE", "HOSTILE", "AGGRESSIVE", "CONSERVATIVE"),
        ),
    },
    "emit": {
        "emit_ref_confidence": ParamSpec(
            "categorical",
            "NONE",
            choices=("NONE", "GVCF", "BP_RESOLUTION"),
        ),
    },
    "assembly": {
        "min_pruning": ParamSpec("int", 2, low=1, high=10),
        "max_alternate_alleles": ParamSpec("int", 6, low=1, high=20),
        "min_dangling_branch_length": ParamSpec("int", 4, low=1, high=20),
        "recover_all_dangling_branches": ParamSpec(
            "categorical", False, choices=(False, True)
        ),
        "max_num_haplotypes_in_population": ParamSpec("int", 128, low=8, high=512),
        "adaptive_pruning_initial_error_rate": ParamSpec(
            "float", 0.001, low=0.0001, high=0.1, log=True
        ),
        "pruning_lod_threshold": ParamSpec("float", 2.302585, low=0.5, high=10.0),
    },
    "active_region": {
        "active_probability_threshold": ParamSpec(
            "float", 0.002, low=0.0001, high=0.05, log=True
        ),
        "min_assembly_region_size": ParamSpec("int", 50, low=1, high=300),
        "max_assembly_region_size": ParamSpec("int", 300, low=100, high=1000),
        "assembly_region_padding": ParamSpec("int", 100, low=0, high=500),
    },
    "pair_hmm": {
        "pair_hmm_gap_continuation_penalty": ParamSpec("int", 10, low=1, high=30),
        "phred_scaled_global_read_mismapping_rate": ParamSpec("int", 45, low=10, high=60),
    },
    "priors": {
        "heterozygosity": ParamSpec("float", 0.001, low=0.0001, high=0.01, log=True),
        "indel_heterozygosity": ParamSpec(
            "float", 0.000125, low=0.00001, high=0.001, log=True
        ),
        "sample_ploidy": ParamSpec("int", 2, low=1, high=10),
        "contamination_fraction_to_filter": ParamSpec("float", 0.0, low=0.0, high=0.5),
    },
    "downsampling": {
        "max_reads_per_alignment_start": ParamSpec("int", 50, low=0, high=1000),
        "dont_use_soft_clipped_bases": ParamSpec(
            "categorical", False, choices=(False, True)
        ),
    },
}


def known_categories() -> Tuple[str, ...]:
    return tuple(sorted(SPACES))


def space_for(category: str) -> Dict[str, ParamSpec]:
    if category not in SPACES:
        known = ", ".join(known_categories())
        raise ValueError(f"unknown search_category {category!r}. known: {known}")
    return SPACES[category]


def default_study_name(category: str) -> str:
    return f"gatk-v2-{category}"


def coerce_param(spec: ParamSpec, raw: Any) -> Any:
    """Map a stored gatk_config value onto the Optuna type for this spec."""
    if spec.kind == "int":
        value = int(round(float(raw)))
        if spec.low is not None:
            value = max(int(spec.low), value)
        if spec.high is not None:
            value = min(int(spec.high), value)
        return value
    if spec.kind == "float":
        value = float(raw)
        if spec.low is not None:
            value = max(float(spec.low), value)
        if spec.high is not None:
            value = min(float(spec.high), value)
        return value
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
