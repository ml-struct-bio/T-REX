"""HypothesisCard lifecycle arithmetic.

Pure functions. See plan §8.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .schemas import (
    HypothesisCard,
    PredictedChange,
    ResultRecord,
)
from .success_criteria import STRICT_SUCCESS, is_strict_success


@dataclass(frozen=True)
class LifecycleConfig:
    min_supporting_descendants: int = 2
    min_contradicting_descendants: int = 3
    contradiction_fail_fraction: float = 0.70
    contradiction_rdr_max: float = 0.05
    eps_deficit: float = 0.01
    default_min_abs_delta: float = 1.0
    # support_points += panel_ready_bonus × N(supporting panel-ready descendants).
    # Calibrated to 1.0 (was 2.0) by lifecycle simulation 2026-05-26 (plan §21 Q29):
    # W=1 vs W=2 yield indistinguishable true_support (94.8% vs 92.0%) and
    # false_support (8.7% vs 9.5%) at TTL=10 — the ×2 multiplier was unjustified.
    panel_ready_bonus: float = 1.0


def deficit(value: float, threshold: float, direction: str) -> float:
    if direction == "decrease":  # lower-is-better
        return max(0.0, value - threshold)
    return max(0.0, threshold - value)


def rdr(
    baseline_d: float, descendant_d: float, eps: float = 0.01
) -> float | None:
    """Relative deficit reduction. Returns None if baseline is already passing."""
    if baseline_d <= eps:
        return None
    return (baseline_d - descendant_d) / baseline_d


def signed_delta(
    descendant_value: float, baseline_value: float, direction: str
) -> float:
    """Improvement delta in the direction of the predicted change."""
    if direction == "decrease":
        return baseline_value - descendant_value
    return descendant_value - baseline_value


def supports_descendant(
    descendant: ResultRecord,
    baseline: ResultRecord,
    predicted: PredictedChange,
    axis_thresholds: dict[str, tuple[float, str]],
    cfg: LifecycleConfig,
) -> tuple[bool, str]:
    """Return (supports, reason). reason is empty when supports=True."""
    axis = predicted.axis
    if axis == "diversity":
        # Legacy archive guard only. Production Planner validation now rejects
        # diversity as a PredictedChange axis because diversity is panel/route
        # evidence, not a per-result lifecycle metric. Keep old cards from
        # silently supporting while making the non-production status explicit.
        return False, "legacy_diversity_axis_not_supported"
    if axis not in axis_thresholds:
        return False, f"missing_threshold:{axis}"

    thr, direction = axis_thresholds[axis]
    if predicted.direction != direction:
        return False, f"direction_mismatch:{predicted.direction}!={direction}"
    bv = baseline.metrics.get(axis)
    dv = descendant.metrics.get(axis)
    if bv is None and dv is not None:
        # Diagnostic-generator baselines (BindCraft/BoltzGen/MPNN) can lack the
        # strict AF2 axes until a downstream structure_refilter converts them.
        # A strict canonical refilter descendant is direct support for that
        # chained hypothesis even though relative deficit cannot be computed.
        if descendant.backend_family == "structure_refilter" and is_strict_success(descendant.metrics):
            return True, ""
    if bv is None or dv is None:
        return False, f"missing_metric:{axis}"

    bd = deficit(bv, thr, direction)
    dd = deficit(dv, thr, direction)
    r = rdr(bd, dd, eps=cfg.eps_deficit)
    if r is not None:
        if r < predicted.min_relative_deficit_reduction:
            return False, f"rdr_below({r:.3f}<{predicted.min_relative_deficit_reduction})"
    else:
        # baseline already passing → absolute delta path
        absolute = max(0.0, signed_delta(dv, bv, direction))
        min_abs = predicted.min_absolute_delta or cfg.default_min_abs_delta
        if absolute < min_abs:
            return False, f"abs_delta_below({absolute:.3f}<{min_abs})"
    return True, ""


def contradicts_descendant(
    descendant: ResultRecord,
    baseline: ResultRecord,
    predicted: PredictedChange,
    axis_thresholds: dict[str, tuple[float, str]],
    cfg: LifecycleConfig,
) -> bool:
    axis = predicted.axis
    if axis == "diversity" or axis not in axis_thresholds:
        return False  # cannot contradict what we cannot measure
    thr, direction = axis_thresholds[axis]
    if predicted.direction != direction:
        return False
    bv = baseline.metrics.get(axis)
    dv = descendant.metrics.get(axis)
    if bv is None or dv is None:
        return False
    bd = deficit(bv, thr, direction)
    dd = deficit(dv, thr, direction)
    r = rdr(bd, dd, eps=cfg.eps_deficit)
    if r is None:
        # baseline already passing — cannot contradict an "improve" hypothesis
        # by measuring tiny absolute delta; skip
        return False
    return r <= cfg.contradiction_rdr_max


def violates_preserve(
    descendant: ResultRecord,
    baseline: ResultRecord,
    hyp: HypothesisCard,
    axis_thresholds: dict[str, tuple[float, str]],
    cfg: LifecycleConfig,
) -> bool:
    for pc in hyp.preserve_constraints:
        axis = pc.axis
        if axis not in axis_thresholds:
            continue
        thr, direction = axis_thresholds[axis]
        bv = baseline.metrics.get(axis)
        dv = descendant.metrics.get(axis)
        if bv is None or dv is None:
            continue
        bd = deficit(bv, thr, direction)
        dd = deficit(dv, thr, direction)
        if bd <= cfg.eps_deficit:
            # was passing; preserve = stays passing
            if dd > cfg.eps_deficit:
                return True
        else:
            # was failing; preserve = not worse than baseline by max_relative_deficit_increase
            if dd > bd * (1.0 + pc.max_relative_deficit_increase):
                return True
    return False


def supports_hyp(
    descendant: ResultRecord,
    baseline: ResultRecord,
    hyp: HypothesisCard,
    axis_thresholds: dict[str, tuple[float, str]],
    cfg: LifecycleConfig,
) -> tuple[bool, list[str]]:
    """All predicted changes must support; no preserve violation."""
    reasons: list[str] = []
    for pc in hyp.predicted_metric_changes:
        ok, why = supports_descendant(descendant, baseline, pc, axis_thresholds, cfg)
        if not ok:
            reasons.append(why)
    if violates_preserve(descendant, baseline, hyp, axis_thresholds, cfg):
        reasons.append("preserve_violated")
    return len(reasons) == 0, reasons


def contradicts_hyp(
    descendant: ResultRecord,
    baseline: ResultRecord,
    hyp: HypothesisCard,
    axis_thresholds: dict[str, tuple[float, str]],
    cfg: LifecycleConfig,
) -> bool:
    """All predicted changes must contradict; otherwise not contradiction."""
    if not hyp.predicted_metric_changes:
        return False
    for pc in hyp.predicted_metric_changes:
        if not contradicts_descendant(descendant, baseline, pc, axis_thresholds, cfg):
            return False
    return True


def update_hypothesis(
    hyp: HypothesisCard,
    *,
    healthy_descendants: list[tuple[ResultRecord, ResultRecord]],
    # ↑ list of (descendant, baseline) pairs; every predicted change is evaluated
    # against the candidate's validated shared comparison baseline.
    current_tick: int,
    axis_thresholds: dict[str, tuple[float, str]],
    cfg: LifecycleConfig | None = None,
) -> HypothesisCard:
    """Update a hypothesis card based on this tick's evidence.

    Returns a new card with updated points and status. Frozen-dataclass safe.
    """
    cfg = cfg or LifecycleConfig()

    if hyp.status in ("supported", "contradicted", "retired"):
        # Terminal states are immutable; only an active card can later retire by TTL.
        return hyp

    s = 0
    c = 0
    supporting_panel_ready = 0
    for descendant, baseline in healthy_descendants:
        ok, _why = supports_hyp(descendant, baseline, hyp, axis_thresholds, cfg)
        if ok:
            s += 1
            if descendant.panel_ready:
                supporting_panel_ready += 1
        if contradicts_hyp(descendant, baseline, hyp, axis_thresholds, cfg):
            c += 1

    n_eval = len(healthy_descendants)

    new_support = s + cfg.panel_ready_bonus * supporting_panel_ready
    new_contra = c

    new_status = hyp.status
    if new_support >= cfg.min_supporting_descendants:
        new_status = "supported"
    elif (
        n_eval >= cfg.min_contradicting_descendants
        and (c / max(1, n_eval)) >= cfg.contradiction_fail_fraction
    ):
        new_status = "contradicted"
    elif current_tick - hyp.tick_created >= hyp.ttl_ticks:
        # TTL retirement; do NOT override supported (already handled above)
        if new_status not in ("supported", "contradicted"):
            new_status = "retired"

    return replace(
        hyp,
        support_points=new_support,
        contradiction_points=new_contra,
        descendants_evaluated=n_eval,
        last_evaluated_tick=current_tick,
        status=new_status,
    )


def default_axis_thresholds(
    pLDDT: float | None = None,
    iPAE: float | None = None,
    binder_scRMSD: float | None = None,
) -> dict[str, tuple[float, str]]:
    """Defaults come from `success_criteria.STRICT_SUCCESS`.
    Override only for ablations; do not pin per-call inline.
    """
    return {
        "pLDDT": (
            pLDDT if pLDDT is not None else STRICT_SUCCESS["pLDDT"][0],
            "increase",
        ),
        "iPAE": (
            iPAE if iPAE is not None else STRICT_SUCCESS["iPAE"][0],
            "decrease",
        ),
        "binder_scRMSD": (
            binder_scRMSD if binder_scRMSD is not None else STRICT_SUCCESS["binder_scRMSD"][0],
            "decrease",
        ),
    }
