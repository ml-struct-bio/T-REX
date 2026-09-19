"""T-ReX success criteria — single source of truth.

The strict_success rule comes from the Complexa paper (Proteina). Same
rule applies to all backend families because evaluation is performed on
the final generated structure + sequence, regardless of how it was
produced:

  - complexa_*: structure + sequence generated jointly by the diffusion
                model (variant = inference-time search strategy)
  - proteinmpnn_redesign: re-sequences an existing scaffold; the final
                scored structure comes from downstream re-folding
                (structure_refilter)
  - bindcraft: hallucinate via AF2 gradients
  - boltzgen: BoltzGen generator (structure + sequence)

In all cases, the metrics evaluated against thresholds below come from
the FINAL structure scored by a verifier (typically an AF2-class scorer
in the canonical structure_refilter verifier).

Foldseek deduplication is applied AFTER strict_success filtering to
produce the run-level structurally-unique success (SU) count. Strict
successes that fall into the same Foldseek cluster collapse to a single
SU entry.

DO NOT define thresholds in any other module. Import from here:
    from trex.success_criteria import STRICT_SUCCESS, NEAR_PASS_MARGINS
"""

from __future__ import annotations

import math
from typing import Literal

# ---------------------------------------------------------------------------
# Strict success rule -- AF2-Multimer protein-complex gate (3 axes below).
# Applied identically to canonical AF2 verifier metrics. Complexa families emit
# these axes directly from canonical af2folding; diagnostic-only generators
# (BindCraft/BoltzGen/ProteinMPNN) chain through structure_refilter before they
# can mint strict/SU credit.
# ---------------------------------------------------------------------------
#
# A result is `strict_success` iff ALL three axes pass:
#
#   pLDDT          >= 90.0       (predicted Local Distance Difference Test;
#                                 AlphaFold-style 0-100 confidence, >= 90 = very-high)
#   iPAE           <= 7/31       (=0.2258..; normalized AF2-Multimer interface PAE;
#                                 lower is better)
#   binder_scRMSD  <  1.5  (Å)   (binder backbone Cα RMSD to predicted complex;
#                                 STRICT "<" per official Complexa scRMSD_ca op)
#
# Per-axis thresholds + directions:
#

STRICT_SUCCESS: dict[str, tuple[float, Literal["increase", "decrease"]]] = {
    "pLDDT":         (90.0,   "increase"),  # higher-is-better
    "iPAE":          (7 / 31, "decrease"),  # lower-is-better, == 0.22580645...
    "binder_scRMSD": (1.5,    "decrease"),  # lower-is-better, in Å
}


def is_strict_success(metrics: dict[str, float]) -> bool:
    """Apply the Complexa strict_success rule to a metrics dict.

    Missing axes are treated as failures (cannot prove success without
    a verifier score on every axis). This matches the V6.3 behavior
    where un-scored candidates are not counted as SU.
    """
    p = metrics.get("pLDDT")
    i = metrics.get("iPAE")
    r = metrics.get("binder_scRMSD")
    if p is None or i is None or r is None:
        return False
    return (
        p >= STRICT_SUCCESS["pLDDT"][0]
        and i <= STRICT_SUCCESS["iPAE"][0]
        and r < STRICT_SUCCESS["binder_scRMSD"][0]   # official op is strict "<"
    )


# ---------------------------------------------------------------------------
# Near-miss / near-pass margins
# ---------------------------------------------------------------------------
#
# A result is "near_pass" on a given axis iff it fails strict but the
# deficit is within `NEAR_PASS_MARGINS[axis]`. A result is "near_miss"
# (overall) iff all three strict axes are present, the result is not strict,
# and at most ONE axis is in the hard-fail region (the others are pass or
# near_pass). A single hard fail is bounded to ten near-pass margins so a
# catastrophic one-axis failure is not mislabeled as a near miss.
#
# Margins are anchored to typical noise scales for each metric:
#
#   pLDDT margin = 5.0   one AlphaFold pLDDT confidence-band width;
#                        pLDDT 85-90 is "almost very-high".
#   iPAE  margin = 0.05  ~22% relative buffer below the 7/31 cut;
#                        iPAE 0.226-0.276 is "almost passing interface".
#   scRMSD margin = 0.3  ~sub-Å Cα backbone noise;
#                        scRMSD 1.5-1.8 Å is "almost passing geometry".
#

NEAR_PASS_MARGINS: dict[str, float] = {
    "pLDDT":         5.0,
    "iPAE":          0.05,
    "binder_scRMSD": 0.3,
}

# Preserve useful single-axis blocker signal for hard targets, but do not call a
# geometrically disconnected or otherwise catastrophic candidate a near miss.
# Larger failures remain visible in axis_stats and failure feedback for exploration.
NEAR_MISS_MAX_HARD_FAIL_MARGIN_MULTIPLIER = 10.0


def is_near_miss(metrics: dict[str, float]) -> bool:
    """A result is near_miss iff all three axes are present, it is NOT
    strict_success, and at most ONE axis is in the 'fail' region
    (deficit > near_pass_margin); the rest are pass or near_pass. A lone hard
    fail must remain within ``NEAR_MISS_MAX_HARD_FAIL_MARGIN_MULTIPLIER``
    margins. More distant blockers remain available through axis statistics and
    failure feedback but do not drive near-miss rescue accounting.

    Captures both:
      - "obvious-blocker" near-miss (exactly 1 axis fail, others pass/near_pass)
      - "barely-not-strict" near-miss (0 axes fail; one axis near_pass blocks strict)

    This matches the user-intended notion "iPAE/pLDDT/scRMSD criteria에
    거의 다가가는" — a result that almost passes the strict gate.

    Single definition prevents the live_tick vs _classify_result drift
    that was caught in beam_trap smoke 8721974.

    scRMSD boundary note: the strict gate uses ``r < 1.5`` (strict), while the
    near-miss margin logic below treats deficit ``d == 0`` (i.e. ``r == 1.5``)
    as non-fail. So a design at EXACTLY scRMSD 1.5 (pLDDT/iPAE passing) is
    not-strict but counts as a 0-axis-fail near_miss — the intended
    "barely-not-strict" case, not a bug. Immaterial for float metrics, which
    ~never land on the boundary; documented only to flag the deliberate
    asymmetry between ``is_strict_success`` and the margin comparator.
    """
    if is_strict_success(metrics):
        return False
    if any(metrics.get(axis) is None for axis in STRICT_SUCCESS):
        return False
    hard_fail_deficits: list[float] = []
    for axis, (thr, direction) in STRICT_SUCCESS.items():
        v = metrics.get(axis)
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return False
        if direction == "increase":
            d = max(0.0, thr - v)
        else:
            d = max(0.0, v - thr)
        if d > NEAR_PASS_MARGINS[axis]:
            hard_fail_deficits.append(d / NEAR_PASS_MARGINS[axis])
    if not hard_fail_deficits:
        return True
    return (
        len(hard_fail_deficits) == 1
        and hard_fail_deficits[0] <= NEAR_MISS_MAX_HARD_FAIL_MARGIN_MULTIPLIER
    )


def is_joint_fail(metrics: dict[str, float]) -> bool:
    """A result is joint_fail iff pLDDT and iPAE BOTH fail (regardless
    of scRMSD). This is a strong pivot/caution signal, not proof that every
    parent-consuming rescue is impossible.
    """
    p = metrics.get("pLDDT")
    i = metrics.get("iPAE")
    if p is None or i is None:
        return False
    p_fail = p < STRICT_SUCCESS["pLDDT"][0] - NEAR_PASS_MARGINS["pLDDT"]
    i_fail = i > STRICT_SUCCESS["iPAE"][0] + NEAR_PASS_MARGINS["iPAE"]
    return p_fail and i_fail
