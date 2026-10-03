"""How v2 turns component metrics into avg_combined_final.

combined_final = 0.70 * core + 0.30 * exp(-fp_per_target / 8)
core is difficulty-weighted F1. SNP classes are 20% of that weight
(snp_hom 0.02, snp_het 0.18). INDEL classes are 80%
(indel_1bp 0.40, indel_2_3bp 0.16, indel_4_7bp 0.10, indel_8bp 0.14).

GATK settings move f1_snp, f1_indel, or fp_per_target. The average score
moves only through those terms. The losses below are the points each term
still has left, using aggregate F1 as a stand-in for the per-class F1s.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

CORE_WEIGHT = 0.70
GERMLINE_WEIGHT = 0.30
FP_SCALE = 8.0
SNP_CORE_SHARE = 0.20
INDEL_CORE_SHARE = 0.80
# Ignore a component until it can move the average by this much.
MIN_LOSS = 0.005

V2_BRIEF = """Decision instruction. The only ranking number is avg_combined_final.
combined_final = 0.70 * core + 0.30 * exp(-fp_per_target / 8).
core is difficulty-weighted F1, not a count of variants:
  snp_hom 0.02, snp_het 0.18, indel_1bp 0.40, indel_2_3bp 0.16,
  indel_4_7bp 0.10, indel_8bp 0.14.
1bp indels are the heaviest class. SNP hom is almost never the reason to open a category.
There is no completeness score and no Ti/Tv quality penalty. A failed ti/tv or het/hom
gate zeros the round; do not spend a search trying to nudge those ratios.
Points still on the table from the best row:
  indel_loss = 0.70 * 0.80 * (1 - f1_indel)
  snp_loss   = 0.70 * 0.20 * (1 - f1_snp)
  fp_loss    = 0.30 * (1 - exp(-fp_per_target / 8))
A parameter changes avg_combined_final only by moving core F1 or fp_per_target.
Read the parameter note for which of those it moves, and in which direction.
Open one catalog category. Do not invent a search category.
indel_loss: pcr, assembly, pair_hmm, priors.
fp_loss: quality_filters, calling_confidence, pair_hmm, priors, downsampling.
snp_loss: active_region, calling_confidence, assembly, priors, quality_filters.
The hypothesis must name that metric and the direction you will move the parameter.
"""


def component_losses(metrics: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """Point losses on the 0-1 combined_final scale. largest is snp, indel, or fp."""
    snp = _gap_loss(metrics.get("avg_f1_snp"), CORE_WEIGHT * SNP_CORE_SHARE)
    indel = _gap_loss(metrics.get("avg_f1_indel"), CORE_WEIGHT * INDEL_CORE_SHARE)
    fp = _fp_loss(metrics.get("avg_fp_per_target"))
    core = _gap_loss(metrics.get("avg_core"), CORE_WEIGHT)
    named = {
        name: value
        for name, value in (("indel", indel), ("fp", fp), ("snp", snp))
        if value is not None
    }
    largest: Optional[str] = None
    if named:
        # INDEL wins a tie: it is 80% of core, so an equal gap is the larger lever.
        winner = max(named, key=lambda name: named[name])
        if named[winner] >= MIN_LOSS:
            largest = winner
    elif core is not None and core >= MIN_LOSS and (fp is None or core >= fp):
        largest = "core"
    return {
        "snp": snp,
        "indel": indel,
        "fp": fp,
        "core": core,
        "largest": largest,
    }


def _gap_loss(raw: Any, weight: float) -> Optional[float]:
    value = _unit(raw)
    if value is None:
        return None
    return weight * (1.0 - value)


def _fp_loss(raw: Any) -> Optional[float]:
    try:
        rate = float(raw)
    except (TypeError, ValueError):
        return None
    if rate != rate or rate < 0 or rate == float("inf"):
        return None
    germline = math.exp(-rate / FP_SCALE)
    return GERMLINE_WEIGHT * (1.0 - germline)


def _unit(raw: Any) -> Optional[float]:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:
        return None
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
