"""Deterministic candidate selection and allocation realization.

Use eligible Supervisor rankings subject to feasibility and capacity constraints. The
default fallback carries fractional allocation credit across cycles and spends it on
confirmed starts. Alternative quota policies support explicit comparisons.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .dedup_trust import near_miss_dedup_trusted_from_evidence as _near_miss_dedup_trusted
from .refilter_roles import is_canonical_score_conversion
from .selection import (
    CandidateAdmissionRequest,
    ModeQuotaRequest,
    PlannedEvidenceSnapshot,
    QuotaCandidateContext,
    RankingCandidateContext,
    SelectionEmissionRequest,
    SelectionPolicyRequest,
    SelectionRankingRequest,
    admit_candidates,
    emit_selection_decisions,
    rank_candidates,
    realize_mode_quotas,
    resolve_batch_diversity_policy,
    resolve_selection_policy,
)
from .selection.quota import (
    _clean_mode_credit,
    _fractional_carry_quotas,
    _mode_window_debug,
    _realize_quotas,
    _stochastic_quotas,
    _windowed_quotas,
    effective_mode_window_k,
)
from .selection.ranking import _enforce_batch_diversity
from .schemas import (
    ActionCandidate,
    CandidateDecision,
    EvidenceSummary,
    LaunchDecision,
    StateLabel,
    SupervisorOutput,
)


@dataclass(frozen=True)
class SelectorConfig:
    available_slots: int = 3
    diversity_max_per_family: int = 2  # avoid spamming one family in a single batch
    top_n_rejected_per_mode_to_log: int = 3
    # K-round window for fractional-share mode realisation. 10 (not the
    # doc's 6) because a window of length K can only resolve shares down to
    # ~1/K: at K=6 the Cat-A productive explore floor (0.05) and an equal
    # explore/rescue split (0.05/0.05) round away entirely; K=10 realises
    # them while staying responsive to mixture changes (deficit reacts to the
    # new target immediately; K only smooths small-share realisation).
    mode_window_k: int = 10
    # Evidence-adaptive inertia. The LLM owns the target E/R/X distribution; the
    # selector owns the deterministic realization window when the K-window
    # ablation is enabled. Keep K=10 in ordinary operation, but shorten it under
    # urgent evidence so a collapse diagnosis is not delayed by stale modes.
    adaptive_mode_window_k: bool = True
    mode_window_k_urgent: int = 4
    mode_window_k_stalled: int = 6
    mode_window_k_duplicate: int = 8
    # By default use the Supervisor scalar mixture. The optional rank-derived mixture
    # changes allocation shares; candidate ordering still follows the rankings.
    derive_mixture_from_ranking: bool = False
    # Convert allocation shares to worker starts using the configured strategy.
    # fractional_carry accumulates small shares without a fixed history window;
    # credit is corrected using confirmed starts.
    # fractional_carry: accumulated credit corrected for worker starts (default)
    # largest_remainder: per-tick proportional rounding
    # deterministic_deficit: largest-deficit fill over a K-start window
    # stochastic: seeded multinomial sampling
    quota_realization: str = "fractional_carry"
    # Fresh valid Supervisor decisions carry an explicit cross-mode global rank.
    # Use it for the immediate worker order; keep quota realization as the
    # deterministic fallback/evidence-skip path and as an audit target.
    realize_global_priority: bool = True
    mode_credit_cap: float = 3.0
    duplicate_pressure_fraction: float = 0.75
    family_share_cap_when_duplicate: float = 0.70
    dispatch_family_window_k: int = 10
    # Demote sufficiently observed, costly, unproductive families when feasible
    # alternatives exist. Preserve opportunities for under-observed or recently
    # productive families.
    cost_aware_family_tiebreak: bool = True
    cost_aware_waste_gpu_h: float = 3.0
    cost_aware_low_yield_gpu_h: float = 8.0
    cost_aware_min_best_su_per_gpu_h: float = 0.50
    # A family may be the comparator for "dominated" only after enough route
    # compute and enough direct/chained SU. This prevents one lucky low-cost hit
    # from prematurely suppressing mature alternatives.
    cost_aware_best_min_gpu_h: float = 2.0
    cost_aware_best_min_su: int = 2
    cost_aware_best_rate_prior_gpu_h: float = 2.0
    cost_aware_dominated_fraction: float = 0.35
    cost_aware_min_su_per_gpu_h: float = 0.25
    cost_aware_defer_penalty: int = 2
    # High-cost dispatch concentration guard. BindCraft workers have delayed
    # feedback and can occupy a worker for hours; keep them from filling all
    # worker slots. Two concurrent workers is still enough to exploit a good
    # BindCraft signal, while leaving one worker for score conversion or a
    # cheaper counterfactual route on the 3-worker production pool.
    # Set this tuple/env to empty only for an explicit no-cap ablation.
    high_cost_pending_families: tuple[str, ...] = ("bindcraft",)
    high_cost_pending_cap: int = 1
    high_cost_pending_promoted_cap: int = 2
    high_cost_pending_deep_stall_cap: int = 1
    high_cost_pending_dry_pivot_cap: int = 2
    high_cost_pending_strong_cap: int = 2
    high_cost_dry_pivot_min_gpu_h: float = 0.75
    high_cost_dry_pivot_min_completed_children: int = 32
    high_cost_dry_pivot_max_best_recent_su_per_gpu_h: float = 0.25
    high_cost_strong_min_su: int = 2
    high_cost_strong_min_gpu_h: float = 1.0
    high_cost_strong_min_recent_su_per_gpu_h: float = 0.50
    high_cost_strong_best_fraction: float = 0.80
    high_cost_stale_recent_gpu_h: float = 1.0
    # Repeated-support execution repair: if the Supervisor keeps proposing a
    # high-cost family but dispatch records show it rarely starts, allow one
    # extra bounded probe before interpreting the lane as scientifically failed.
    # This is disabled by clear negative completed evidence and by high-SU
    # productive_duplicate regimes, so easy targets are not taxed by hard probes.
    high_cost_repeated_support_min_proposed: int = 6
    high_cost_repeated_support_max_started_fraction: float = 0.35
    high_cost_repeated_support_cap: int = 2
    # One-slot realization repair: when the top exploit candidate is an exact
    # replay of a duplicate-heavy route whose GPU-normalized marginal new-SU has
    # decayed, allow a feasible rescue/explore material-diversity candidate to
    # take the slot. This is NOT a duplicate hard rule; positive recent SU/GPU-h
    # suppresses the override.
    # Opt-in ablation. Production leaves scientific mode choice to the LLM;
    # route-value penalties and collapse-aware mixture clamps already expose and
    # safeguard this condition without a second post-quota policy.
    marginal_diversity_override: bool = False
    # One-slot near-miss repair: when the run is stalled/rescue-rich with no
    # recent SU and a concrete near-miss parent has an untried LOW-cost rescue
    # candidate, give that rescue one slot over a higher-cost explore/exploit
    # pick. This is deliberately narrow: it does not fire on productive/easy
    # targets, does not score arbitrary backlog, and does not repeat an observed
    # parent-bound rescue route.
    # Opt-in ablation. Near-miss evidence is already explicit in the Planner and
    # Supervisor contracts, so production does not independently rewrite it.
    low_cost_near_miss_rescue_floor: bool = False
    high_cost_repeated_support_max_negative_gpu_h: float = 6.0
    high_cost_repeated_support_max_timeouts: int = 1
    # Opt-in ablation. Production fractional carry is reconstructed from actual
    # worker starts, which already repays an under-started LLM-selected mode.
    realize_repeated_support_probe: bool = False
    repeated_support_probe_rank_max: int = 1
    # Realization repair for Builder's dry/deep-stall cross-family escape
    # candidates. These candidates are created only from EvidenceSummary
    # stagnation signals; if Supervisor forgets to rank them, do not let a stale
    # same-root replay fill every slot and silently erase the escape floor.
    realize_cross_family_escape_floor: bool = True
    # Narrow TNF-style guard: a route-deferred diagnostic generator should not be
    # revived as a fresh cross-family generation when the same family already has
    # a large unscored diagnostic backlog, enough completed score-conversion
    # probes to show 0 SU, and no pending promising artifacts. In that case the
    # scientific bottleneck is conversion/another family, not more raw artifacts.
    backlog_saturated_generator_families: tuple[str, ...] = ("bindcraft", "boltzgen", "proteinmpnn_redesign")
    backlog_saturated_min_unscored: int = 32
    backlog_saturated_min_completed_refilters: int = 4
    # Live-throughput repair: fractional carry and safety floors must not round
    # away every exploit slot while the run is actively buying new official SU.
    # This does not force a family; it only preserves at least one launchable
    # exploit candidate when recent/HWM evidence says exploitation is still
    # paying. Strong productive ticks can keep two exploit slots on a 3-worker
    # batch, leaving one slot for rescue/explore.
    productive_wall_momentum_guard: bool = True
    productive_wall_momentum_max_dry_gpu_h: float = 6.0
    productive_wall_momentum_strong_max_dry_gpu_h: float = 3.0
    productive_wall_momentum_strong_min_recent_su_per_gpu_h: float = 0.50
    # Fractional carry is the durable realization of the Supervisor's requested
    # mixture. Do not use the momentum floor to spend another exploit slot when
    # prior worker starts have already overspent exploit relative to that mixture.
    # A one-slot tolerance preserves immediate throughput on a strong incumbent;
    # beyond that bounded overspend, fractional carry must repay the LLM's other
    # modes instead of letting single-slot momentum become permanent all-exploit.
    productive_wall_momentum_min_exploit_credit: float = -1.0
    # The strong 2-slot momentum repair is only for cheap/current exact routes.
    # High-cost delayed-feedback routes are governed by high_cost_* caps below;
    # otherwise one early BindCraft hit can consume the whole 3-worker batch.
    productive_wall_momentum_route_min_su_per_gpu_h: float = 0.50
    productive_wall_momentum_safe_cost_classes: tuple[str, ...] = ("low", "diagnostic", "standard")


def _mh_value(h, k, d=0):
    v = h.get(k, d) if isinstance(h, dict) else getattr(h, k, d)
    return d if v is None else v


def _repeated_support_probe_cap(
    evidence: EvidenceSummary,
    fam: str,
    *,
    min_proposed: int,
    max_started_fraction: float,
    cap: int,
    max_negative_gpu_h: float,
    max_timeouts: int,
) -> int | None:
    er = getattr(evidence, "execution_realization", {}) or {}
    by_family = er.get("by_family", {}) if isinstance(er, dict) else {}
    row = by_family.get(fam, {}) if isinstance(by_family, dict) else {}
    if not isinstance(row, dict):
        return None
    try:
        proposed = int(row.get("proposed", 0) or 0)
        started = int(row.get("started", 0) or 0)
        deferred = int(row.get("dispatch_deferred", 0) or 0)
        selected_not_started = int(row.get("selected_not_started", 0) or 0)
    except (TypeError, ValueError):
        return None
    if proposed < min_proposed:
        return None
    # Repeated Supervisor support alone must not relax a high-cost cap. The
    # Selector may have intentionally ranked the family below better routes.
    # Require evidence that selected work was actually delayed at dispatch.
    if deferred <= 0 and selected_not_started <= 0:
        return None
    if started / max(1, proposed) > max_started_fraction:
        return None

    state = str(getattr(evidence, "state_label", "") or "")
    run_su = int(getattr(evidence, "run_su_count", 0) or 0)
    try:
        dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        dry_gpu_h = 0.0
    productive_duplicate_dry_collapse = (
        state == "productive_duplicate"
        and bool(getattr(evidence, "strict_duplicate_collapse_signal", False))
        and dry_gpu_h >= 1.0
    )
    low_sample_or_stalled = (
        state in {"low_evidence", "stalled", "deep_stall", "strict_duplicate_collapse"}
        or run_su < 4
        or productive_duplicate_dry_collapse
    )
    if not low_sample_or_stalled:
        return None

    h = (getattr(evidence, "method_health", {}) or {}).get(fam, {})
    try:
        gpu_h = float(_mh_value(h, "cumulative_gpu_h", 0.0) or 0.0)
    except (TypeError, ValueError):
        gpu_h = 0.0
    try:
        timeouts = int(_mh_value(h, "timeout_count", _mh_value(h, "timeouts", 0)) or 0)
    except (TypeError, ValueError):
        timeouts = 0
    strict = int(_mh_value(h, "strict_yield_su", 0) or 0)
    chained = int(_mh_value(h, "chained_strict_yield_su", 0) or 0)
    near_trusted = _near_miss_dedup_trusted(evidence)
    near = int(_mh_value(h, "near_miss_yield", 0) or 0) if near_trusted else 0
    recent_near = int(_mh_value(h, "near_miss_yield_recent", 0) or 0) if near_trusted else 0
    clearly_negative = (
        gpu_h >= max_negative_gpu_h
        and timeouts > max_timeouts
        and strict == 0
        and chained == 0
        and near == 0
        and recent_near == 0
    )
    if clearly_negative:
        return None
    return max(1, int(cap))


def _rate(h: Any, *keys: str) -> float:
    vals: list[float] = []
    for k in keys:
        try:
            vals.append(float(_mh_value(h, k, 0.0) or 0.0))
        except (TypeError, ValueError):
            vals.append(0.0)
    return max(vals) if vals else 0.0


def _su_count(h: Any) -> int:
    vals: list[int] = []
    for k in (
        "strict_yield_su",
        "chained_strict_yield_su",
        "chained_strict_yield_su_recent",
    ):
        try:
            vals.append(int(_mh_value(h, k, 0) or 0))
        except (TypeError, ValueError):
            vals.append(0)
    return max(vals) if vals else 0


def _clear_negative_high_cost_evidence(
    h: Any,
    *,
    max_negative_gpu_h: float,
    max_timeouts: int,
    near_miss_trusted: bool = True,
) -> bool:
    try:
        gpu_h = float(_mh_value(h, "cumulative_gpu_h", 0.0) or 0.0)
    except (TypeError, ValueError):
        gpu_h = 0.0
    try:
        timeouts = int(_mh_value(h, "timeout_count", _mh_value(h, "timeouts", 0)) or 0)
    except (TypeError, ValueError):
        timeouts = 0
    strict = int(_mh_value(h, "strict_yield_su", 0) or 0)
    chained = int(_mh_value(h, "chained_strict_yield_su", 0) or 0)
    near = int(_mh_value(h, "near_miss_yield", 0) or 0) if near_miss_trusted else 0
    recent_near = int(_mh_value(h, "near_miss_yield_recent", 0) or 0) if near_miss_trusted else 0
    return (
        gpu_h >= max_negative_gpu_h
        and timeouts > max_timeouts
        and strict == 0
        and chained == 0
        and near == 0
        and recent_near == 0
    )


def _su_dedup_trusted(evidence: EvidenceSummary) -> bool:
    status = str(getattr(evidence, "foldseek_su_status", "ok") or "ok")
    coverage = getattr(evidence, "foldseek_su_coverage", None)
    if status == "ok":
        if coverage is None:
            # Synthetic tests and legacy evidence may omit coverage; production
            # live_tick writes it when strict structures exist. Treat missing+ok
            # as trusted for backward compatibility, but never trust explicit
            # degraded statuses below.
            return True
        try:
            return float(coverage) >= 0.999
        except (TypeError, ValueError):
            return False
    # No strict structures means there is no SU-rate evidence to inflate.
    return status in {"no_strict", "no_structures"}


def _best_recent_su_rate(evidence: EvidenceSummary) -> float:
    if not _su_dedup_trusted(evidence):
        return 0.0
    best = 0.0
    for fam, h in (getattr(evidence, "method_health", {}) or {}).items():
        if fam == "structure_refilter":
            continue
        best = max(best, _rate(h, "su_per_gpu_h_recent", "chained_su_per_gpu_h_recent"))
    return best


def _effective_gpu_h_for_rate_value(h: Any, rate: float) -> float:
    n = _su_count(h)
    if rate > 0.0 and n > 0:
        return max(0.0, n / rate)
    try:
        return float(_mh_value(h, "cumulative_gpu_h", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def high_cost_cap_for_evidence(
    evidence: EvidenceSummary,
    family: str,
    *,
    high_cost_pending_cap: int = 1,
    high_cost_pending_promoted_cap: int = 2,
    high_cost_pending_deep_stall_cap: int = 1,
    high_cost_pending_dry_pivot_cap: int = 2,
    high_cost_pending_strong_cap: int = 2,
    high_cost_dry_pivot_min_gpu_h: float = 0.75,
    high_cost_dry_pivot_min_completed_children: int = 32,
    high_cost_dry_pivot_max_best_recent_su_per_gpu_h: float = 0.25,
    high_cost_strong_min_su: int = 2,
    high_cost_strong_min_gpu_h: float = 1.0,
    high_cost_strong_min_recent_su_per_gpu_h: float = 0.50,
    high_cost_strong_best_fraction: float = 0.80,
    high_cost_stale_recent_gpu_h: float = 1.0,
    min_su_per_gpu_h: float = 0.25,
    high_cost_repeated_support_min_proposed: int = 6,
    high_cost_repeated_support_max_started_fraction: float = 0.35,
    high_cost_repeated_support_cap: int = 2,
    high_cost_repeated_support_max_negative_gpu_h: float = 6.0,
    high_cost_repeated_support_max_timeouts: int = 1,
) -> tuple[int, str]:
    """Evidence-adaptive concurrency cap for expensive delayed-feedback lanes.

    The selector hard cap is a running-worker count, not a lifetime budget. It stays
    conservative for unproven expensive lanes, opens when the lane has current
    signal, and can open for a dry pivot when cheap routes are not buying new
    SU/GPU-h. It intentionally does not use target identity.
    """
    method_health = getattr(evidence, "method_health", {}) or {}
    h = method_health.get(family, {})
    su_rates_trusted = _su_dedup_trusted(evidence)
    near_miss_trusted = _near_miss_dedup_trusted(evidence)
    recent_su_rate = (
        _rate(h, "su_per_gpu_h_recent", "chained_su_per_gpu_h_recent")
        if su_rates_trusted else 0.0
    )
    recent_signal = (
        recent_su_rate > 0.0
        or (near_miss_trusted and int(_mh_value(h, "near_miss_yield_recent", 0) or 0) > 0)
    )
    lifetime_rate = (
        _rate(h, "su_per_gpu_h", "chained_su_per_gpu_h")
        if su_rates_trusted else 0.0
    )
    lifetime_signal = (
        (su_rates_trusted and _su_count(h) > 0)
        or (near_miss_trusted and int(_mh_value(h, "near_miss_yield", 0) or 0) > 0)
    )
    state = str(getattr(evidence, "state_label", "") or "")

    if state == "deep_stall":
        cap = int(high_cost_pending_deep_stall_cap)
        source = "deep_stall"
    else:
        cap = int(high_cost_pending_cap)
        source = "default"

    run_su = int(getattr(evidence, "run_su_count", 0) or 0)
    completed = int(getattr(evidence, "completed_children", 0) or 0)
    try:
        dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        dry_gpu_h = 0.0
    try:
        worker_gpu_h = float(getattr(evidence, "worker_gpu_h_total", 0.0) or 0.0)
    except (TypeError, ValueError):
        worker_gpu_h = 0.0
    if run_su <= 0:
        dry_gpu_h = max(dry_gpu_h, worker_gpu_h)
    strict_collapse = bool(getattr(evidence, "strict_duplicate_collapse_signal", False))
    productive_duplicate = state == "productive_duplicate"
    easy_productive = productive_duplicate and run_su >= 4 and not strict_collapse
    enough_evidence = completed >= int(high_cost_dry_pivot_min_completed_children)
    dry_enough = dry_gpu_h >= float(high_cost_dry_pivot_min_gpu_h)
    hard_or_dry_state = (
        state in {"stalled", "deep_stall", "strict_duplicate_collapse"}
        or (state == "low_evidence" and enough_evidence)
        or (productive_duplicate and strict_collapse)
    )
    cheap_routes_not_buying_recent_su = (
        _best_recent_su_rate(evidence)
        < float(high_cost_dry_pivot_max_best_recent_su_per_gpu_h)
    )
    clearly_negative = _clear_negative_high_cost_evidence(
        h,
        max_negative_gpu_h=high_cost_repeated_support_max_negative_gpu_h,
        max_timeouts=high_cost_repeated_support_max_timeouts,
        near_miss_trusted=near_miss_trusted,
    )
    recent_rate = recent_su_rate
    best_recent = _best_recent_su_rate(evidence)

    def _rv_get(row: Any, key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)

    family_route_rates: list[tuple[float, float, int, bool, bool, float]] = []
    family_route_current_signal = False
    family_route_stale_recent_gpu_h = 0.0
    other_best_route_rate = 0.0
    route_su_rates_trusted = _su_dedup_trusted(evidence)
    for row in (getattr(evidence, "route_values", None) or []) if route_su_rates_trusted else []:
        fam = str(_rv_get(row, "action_family", "") or _rv_get(row, "family", "") or "")
        scope = str(_rv_get(row, "scope", "") or "")
        if scope not in {"route", "family"}:
            continue
        try:
            route_gpu_h = float(_rv_get(row, "route_gpu_h", 0.0) or 0.0)
        except (TypeError, ValueError):
            route_gpu_h = 0.0
        try:
            new_su = int(_rv_get(row, "new_su", 0) or 0)
        except (TypeError, ValueError):
            new_su = 0
        try:
            gpu_medium_recent_su = max(
                int(_rv_get(row, "new_su_recent_gpu", 0) or 0),
                _rv_medium_recent_su(row),
            )
        except (TypeError, ValueError):
            gpu_medium_recent_su = 0
        rank_rate_raw = _rv_current_rate(row)
        gpu_recent_rate_raw = _rv_get(row, "gpu_recent_new_su_per_route_gpu_h", None)
        safe_record_recent_rate_raw = None
        route_role = str(_rv_get(row, "route_role", "") or "")
        try:
            canonical_refilter_gpu_h = float(_rv_get(row, "canonical_refilter_gpu_h", 0.0) or 0.0)
        except (TypeError, ValueError):
            canonical_refilter_gpu_h = 0.0
        if canonical_refilter_gpu_h <= 0.0 and "score_conversion" not in route_role:
            safe_record_recent_rate_raw = _rv_record_recent_rate(row)
            record_recent_su_for_cap = _rv_record_recent_su(row)
        else:
            record_recent_su_for_cap = 0
        try:
            rate_f = float(rank_rate_raw) if rank_rate_raw is not None else 0.0
        except (TypeError, ValueError):
            rate_f = 0.0
        try:
            lifetime_route_rate_f = float(_rv_get(row, "new_su_per_route_gpu_h", 0.0) or 0.0)
        except (TypeError, ValueError):
            lifetime_route_rate_f = 0.0
        # _rv_current_rate returns an explicit 0.0 when the recent GPU-window is
        # dry. Keep the lifetime rate available only to detect stale cap-opening
        # risk; route_value_best below still requires current new-SU signal.
        rate_for_cap_bookkeeping = max(rate_f, lifetime_route_rate_f)
        recent_rate_raw = gpu_recent_rate_raw if gpu_recent_rate_raw is not None else safe_record_recent_rate_raw
        try:
            recent_rate_f = float(recent_rate_raw) if recent_rate_raw is not None else 0.0
        except (TypeError, ValueError):
            recent_rate_f = 0.0
        route_recent_su_for_cap = max(gpu_medium_recent_su, record_recent_su_for_cap)
        try:
            near_recent_for_cap = int(_rv_get(row, "near_miss_recent", 0) or 0) if near_miss_trusted else 0
        except (TypeError, ValueError):
            near_recent_for_cap = 0
        route_current_signal = (
            route_recent_su_for_cap > 0
            or recent_rate_f > 0.0
            or near_recent_for_cap > 0
        )
        route_current_su_signal = (
            route_recent_su_for_cap > 0
            or recent_rate_f > 0.0
        )
        try:
            recent_route_gpu_values = [
                float(_rv_get(row, "gpu_recent_route_gpu_h", 0.0) or 0.0),
                float(_rv_get(row, "medium_recent_route_gpu_h", 0.0) or 0.0),
            ]
            if _rv_safe_record_allowed(row):
                recent_route_gpu_values.extend([
                    float(_rv_get(row, "record_recent_route_gpu_h", 0.0) or 0.0),
                    float(_rv_get(row, "recent_route_gpu_h", 0.0) or 0.0),
                ])
            recent_route_gpu_h_for_stale = max(recent_route_gpu_values or [0.0])
        except (TypeError, ValueError):
            recent_route_gpu_h_for_stale = 0.0
        if new_su <= 0 or route_gpu_h <= 0.0 or rate_for_cap_bookkeeping <= 0.0:
            continue
        status = str(_rv_get(row, "status", "") or "")
        marginal_status = str(_rv_get(row, "marginal_status", "") or "")
        duplicate_or_dead = (
            status in {"defer", "collapse_risk"}
            or marginal_status in {"dry", "dry_duplicate", "dry_low_quality"}
        )
        # For opening a high-cost concurrency cap, compare against routes that are
        # still buying SU now. A stale duplicate-heavy route with a high lifetime
        # SU/GPU-h must not block the only currently useful high-cost route. The
        # duplicate/stale classification itself is EvidenceReducer-owned; keep
        # raw strict/SU thresholds out of Selector.
        competitive_for_cap = route_current_signal
        if fam == family:
            family_route_rates.append((
                rate_for_cap_bookkeeping,
                route_gpu_h,
                new_su,
                route_current_signal,
                route_current_su_signal,
                recent_route_gpu_h_for_stale,
            ))
            family_route_current_signal = family_route_current_signal or route_current_signal
            if new_su > 0 and not route_current_signal:
                family_route_stale_recent_gpu_h = max(
                    family_route_stale_recent_gpu_h,
                    recent_route_gpu_h_for_stale,
                )
        elif competitive_for_cap:
            other_best_route_rate = max(other_best_route_rate, rate_f)

    best_family_route_rate = max((r[0] for r in family_route_rates), default=0.0)
    best_family_rows = [r for r in family_route_rates if r[0] == best_family_route_rate]
    best_family_route_gpu_h = max((r[1] for r in best_family_rows), default=0.0)
    best_family_route_su = max((r[2] for r in best_family_rows), default=0)
    best_family_route_current_signal = any(r[3] for r in best_family_rows)
    best_family_route_current_su_signal = any(r[4] for r in best_family_rows)
    best_recent = max(best_recent, other_best_route_rate, best_family_route_rate)
    stale_lifetime_high_cost = (
        lifetime_signal
        and not recent_signal
        and not family_route_current_signal
        and family_route_stale_recent_gpu_h >= float(high_cost_stale_recent_gpu_h)
    )

    if (
        recent_signal
        or (
            lifetime_signal
            and lifetime_rate >= min_su_per_gpu_h
            and not stale_lifetime_high_cost
        )
    ) and int(high_cost_pending_promoted_cap) > cap:
        cap = int(high_cost_pending_promoted_cap)
        source = "promoted"

    strong_signal = (
        _su_count(h) >= int(high_cost_strong_min_su)
        and _effective_gpu_h_for_rate_value(h, recent_rate) >= float(high_cost_strong_min_gpu_h)
        and recent_rate >= float(high_cost_strong_min_recent_su_per_gpu_h)
        and recent_rate >= best_recent * float(high_cost_strong_best_fraction)
        and not clearly_negative
    )
    # Target-agnostic high-cost promotion: a high-cost family may be the only
    # lane that has just produced new SU. Single-SU current evidence opens a
    # cautious two-slot cap. The default strong cap remains two for the 3-worker
    # production pool; explicit ablations can raise it with config/env.
    single_su_high_value = (
        _su_count(h) >= 1
        and _effective_gpu_h_for_rate_value(h, recent_rate) >= float(high_cost_strong_min_gpu_h)
        and recent_rate >= float(high_cost_strong_min_recent_su_per_gpu_h)
        and recent_rate >= best_recent * float(high_cost_strong_best_fraction)
        and not strict_collapse
        and not clearly_negative
    )
    route_value_promoted = (
        best_family_route_su >= 1
        and best_family_route_gpu_h >= float(high_cost_strong_min_gpu_h)
        and best_family_route_current_signal
        and best_family_route_rate >= float(os.environ.get("TREX_HIGH_COST_ROUTE_VALUE_MIN_SU_PER_GPU_H", "0.10"))
        and best_family_route_rate >= other_best_route_rate * float(high_cost_strong_best_fraction)
        and not clearly_negative
    )
    route_value_best = (
        route_value_promoted
        and best_family_route_current_su_signal
        and (
            best_family_route_su >= 3
            or (
                best_family_route_su >= max(2, int(high_cost_strong_min_su))
                and best_family_route_rate >= float(high_cost_strong_min_recent_su_per_gpu_h)
            )
        )
    )
    if strong_signal and int(high_cost_pending_strong_cap) >= cap:
        cap = max(cap, int(high_cost_pending_strong_cap))
        source = "strong_evidence"
    elif route_value_best and int(high_cost_pending_strong_cap) >= cap:
        cap = max(cap, int(high_cost_pending_strong_cap))
        source = "route_value_best"
    elif route_value_promoted and int(high_cost_pending_promoted_cap) > cap:
        cap = int(high_cost_pending_promoted_cap)
        source = "route_value_promoted"
    elif single_su_high_value and int(high_cost_pending_promoted_cap) > cap:
        cap = int(high_cost_pending_promoted_cap)
        source = "single_su_high_value"
    elif (
        hard_or_dry_state
        and enough_evidence
        and dry_enough
        and cheap_routes_not_buying_recent_su
        and not easy_productive
        and not clearly_negative
        and int(high_cost_pending_dry_pivot_cap) > cap
    ):
        cap = int(high_cost_pending_dry_pivot_cap)
        source = "dry_pivot"

    repeat_cap = None if stale_lifetime_high_cost else _repeated_support_probe_cap(
        evidence,
        family,
        min_proposed=high_cost_repeated_support_min_proposed,
        max_started_fraction=high_cost_repeated_support_max_started_fraction,
        cap=high_cost_repeated_support_cap,
        max_negative_gpu_h=high_cost_repeated_support_max_negative_gpu_h,
        max_timeouts=high_cost_repeated_support_max_timeouts,
    )
    if repeat_cap is not None and repeat_cap > cap:
        cap = repeat_cap
        source = "repeated_support"

    return max(1, int(cap)), source

def _family_cost_penalties(
    evidence: EvidenceSummary,
    *,
    waste_gpu_h: float,
    low_yield_gpu_h: float = 8.0,
    min_best_su_per_gpu_h: float = 0.50,
    best_min_gpu_h: float = 2.0,
    best_min_su: int = 2,
    best_rate_prior_gpu_h: float = 2.0,
    dominated_fraction: float = 0.35,
    min_su_per_gpu_h: float = 0.25,
    high_cost_pending_families: tuple[str, ...] = (),
    high_cost_pending_cap: int = 1,
    high_cost_pending_promoted_cap: int = 2,
    high_cost_pending_deep_stall_cap: int = 1,
    high_cost_pending_dry_pivot_cap: int = 2,
    high_cost_pending_strong_cap: int = 2,
    high_cost_dry_pivot_min_gpu_h: float = 0.75,
    high_cost_dry_pivot_min_completed_children: int = 32,
    high_cost_dry_pivot_max_best_recent_su_per_gpu_h: float = 0.25,
    high_cost_strong_min_su: int = 2,
    high_cost_strong_min_gpu_h: float = 1.0,
    high_cost_strong_min_recent_su_per_gpu_h: float = 0.50,
    high_cost_strong_best_fraction: float = 0.80,
    high_cost_repeated_support_min_proposed: int = 6,
    high_cost_repeated_support_max_started_fraction: float = 0.35,
    high_cost_repeated_support_cap: int = 2,
    high_cost_repeated_support_max_negative_gpu_h: float = 6.0,
    high_cost_repeated_support_max_timeouts: int = 1,
) -> dict[str, int]:
    """Recent, cost-normalized family demotion signal.

    0 = healthy / under-sampled / recently productive.
    1 = stale or weak historical signal, but not bad enough to defer.
    2 = dead lane: enough GPU-h and no strict, chained strict, or near-miss.
        Also used for a high-cost lane that is strongly dominated by another
        observed family and has no recent SU/near-miss evidence.

    This keeps the T-REX near-miss lesson: a family with recent near-misses is not
    treated as dead, while an old one-off near-miss no longer protects an
    otherwise dry lane forever.
    """
    penalties: dict[str, int] = {}

    def _rate(h, *keys: str) -> float:
        vals: list[float] = []
        for k in keys:
            try:
                vals.append(float(_mh_value(h, k, 0.0) or 0.0))
            except (TypeError, ValueError):
                vals.append(0.0)
        return max(vals) if vals else 0.0

    def _su_count(h) -> int:
        vals: list[int] = []
        for k in (
            "strict_yield_su",
            "chained_strict_yield_su",
            "chained_strict_yield_su_recent",
        ):
            try:
                vals.append(int(_mh_value(h, k, 0) or 0))
            except (TypeError, ValueError):
                vals.append(0)
        return max(vals) if vals else 0

    def _effective_gpu_h_for_rate(h, rate: float) -> float:
        n = _su_count(h)
        if rate > 0.0 and n > 0:
            return max(0.0, n / rate)
        return float(_mh_value(h, "cumulative_gpu_h", 0.0) or 0.0)

    def _shrunk_rate(h, rate: float) -> float:
        g = _effective_gpu_h_for_rate(h, rate)
        if rate <= 0.0 or g <= 0.0:
            return 0.0
        return rate * (g / (g + max(0.0, best_rate_prior_gpu_h)))

    if not _su_dedup_trusted(evidence):
        return {}

    near_miss_trusted = _near_miss_dedup_trusted(evidence)
    method_health = getattr(evidence, "method_health", {}) or {}
    family_rate: dict[str, float] = {}
    for fam, h in method_health.items():
        # structure_refilter is score-conversion/evaluation plumbing; admission
        # control should act on the generator/refiner that created the artifact.
        if fam == "structure_refilter":
            continue
        lifetime = _rate(h, "su_per_gpu_h", "chained_su_per_gpu_h")
        recent = _rate(h, "su_per_gpu_h_recent", "chained_su_per_gpu_h_recent")
        raw_rate = max(lifetime, recent)
        eff_gpu_h = _effective_gpu_h_for_rate(h, raw_rate)
        if _su_count(h) >= best_min_su and eff_gpu_h >= best_min_gpu_h:
            family_rate[fam] = _shrunk_rate(h, raw_rate)
    best_rate = max(family_rate.values(), default=0.0)

    for fam, h in method_health.items():
        if fam == "structure_refilter":
            continue
        gpu_h = float(_mh_value(h, "cumulative_gpu_h", 0.0) or 0.0)
        if gpu_h < waste_gpu_h:
            continue
        strict = int(_mh_value(h, "strict_yield_su", 0) or 0)
        chained_strict = int(_mh_value(h, "chained_strict_yield_su", 0) or 0)
        near = int(_mh_value(h, "near_miss_yield", 0) or 0) if near_miss_trusted else 0

        recent_su = _rate(h, "su_per_gpu_h_recent")
        recent_chain = _rate(h, "chained_su_per_gpu_h_recent")
        recent_near_raw = _mh_value(h, "near_miss_yield_recent", None)
        # Use cumulative near-miss evidence when older records lack recent fields.
        recent_near = (
            near if recent_near_raw is None else int(recent_near_raw or 0)
        ) if near_miss_trusted else 0
        recent_signal = recent_su > 0.0 or recent_chain > 0.0 or recent_near > 0
        if recent_su > 0.0 or recent_chain > 0.0:
            continue

        if strict == 0 and chained_strict == 0 and near == 0:
            penalties[fam] = 2
            continue

        lifetime_rate = _rate(h, "su_per_gpu_h", "chained_su_per_gpu_h")
        dominated = (
            gpu_h >= low_yield_gpu_h
            and best_rate >= min_best_su_per_gpu_h
            and (
                lifetime_rate < best_rate * dominated_fraction
                or lifetime_rate < min_su_per_gpu_h
            )
        )
        if dominated:
            # Recent near-misses are useful rescue evidence, so keep them as a
            # soft demotion. Stale near-misses or low-rate SU after substantial
            # GPU burn should not keep occupying primary quota slots.
            penalties[fam] = 1 if recent_signal else 2
        elif gpu_h >= 2.0 * waste_gpu_h and not recent_signal:
            penalties[fam] = 1

    # Do not fold high-cost running/queued load into scientific family penalties.
    # That load is delayed-feedback execution pressure, not evidence that the
    # family/config is low value. Selector must preserve Supervisor's ranked
    # scientific choice; the dispatcher may temporarily defer start if all
    # same-family high-cost worker slots are already occupied. The pressure is
    # reported separately through the capacity_pressure_* diagnostic fields.
    return penalties


def family_cost_penalties_for_cfg(evidence: EvidenceSummary, cfg: SelectorConfig) -> dict[str, int]:
    """Return cost-admission penalties shared by selection and Supervisor guidance. Zero
    penalties are omitted; large penalties can defer a family from the primary pool.
    """
    if not cfg.cost_aware_family_tiebreak:
        return {}
    return _family_cost_penalties(
        evidence,
        waste_gpu_h=cfg.cost_aware_waste_gpu_h,
        low_yield_gpu_h=cfg.cost_aware_low_yield_gpu_h,
        min_best_su_per_gpu_h=cfg.cost_aware_min_best_su_per_gpu_h,
        best_min_gpu_h=cfg.cost_aware_best_min_gpu_h,
        best_min_su=cfg.cost_aware_best_min_su,
        best_rate_prior_gpu_h=cfg.cost_aware_best_rate_prior_gpu_h,
        dominated_fraction=cfg.cost_aware_dominated_fraction,
        min_su_per_gpu_h=cfg.cost_aware_min_su_per_gpu_h,
        high_cost_pending_families=cfg.high_cost_pending_families,
        high_cost_pending_cap=cfg.high_cost_pending_cap,
        high_cost_pending_promoted_cap=cfg.high_cost_pending_promoted_cap,
        high_cost_pending_deep_stall_cap=cfg.high_cost_pending_deep_stall_cap,
        high_cost_pending_dry_pivot_cap=cfg.high_cost_pending_dry_pivot_cap,
        high_cost_pending_strong_cap=cfg.high_cost_pending_strong_cap,
        high_cost_dry_pivot_min_gpu_h=cfg.high_cost_dry_pivot_min_gpu_h,
        high_cost_dry_pivot_min_completed_children=cfg.high_cost_dry_pivot_min_completed_children,
        high_cost_dry_pivot_max_best_recent_su_per_gpu_h=cfg.high_cost_dry_pivot_max_best_recent_su_per_gpu_h,
        high_cost_strong_min_su=cfg.high_cost_strong_min_su,
        high_cost_strong_min_gpu_h=cfg.high_cost_strong_min_gpu_h,
        high_cost_strong_min_recent_su_per_gpu_h=cfg.high_cost_strong_min_recent_su_per_gpu_h,
        high_cost_strong_best_fraction=cfg.high_cost_strong_best_fraction,
        high_cost_repeated_support_min_proposed=cfg.high_cost_repeated_support_min_proposed,
        high_cost_repeated_support_max_started_fraction=cfg.high_cost_repeated_support_max_started_fraction,
        high_cost_repeated_support_cap=cfg.high_cost_repeated_support_cap,
        high_cost_repeated_support_max_negative_gpu_h=cfg.high_cost_repeated_support_max_negative_gpu_h,
        high_cost_repeated_support_max_timeouts=cfg.high_cost_repeated_support_max_timeouts,
    )


def high_cost_capacity_block_reason_for_cfg(
    evidence: EvidenceSummary,
    cfg: SelectorConfig,
    family: str,
) -> str | None:
    """Return the concrete high-cost admission block reason for one family."""
    pending_load = getattr(evidence, "pending_family_load", {}) or {}
    pending_by_family = (
        pending_load.get("by_family", {})
        if isinstance(pending_load, dict) else {}
    )
    if not isinstance(pending_by_family, dict):
        return None
    row = pending_by_family.get(family)
    if not isinstance(row, dict):
        return None

    # The family allow-list is a config/env decision, not evidence telemetry.
    # Old archives can contain pending_family_load.high_cost_families from a
    # previous policy; treating that field as authoritative would silently
    # re-enable the cap during resume/replay after production disabled it.
    high_cost_families = set(cfg.high_cost_pending_families)
    if family not in high_cost_families:
        return None

    try:
        running = int(row.get("running", 0) or 0)
        queued = int(row.get("queued", 0) or 0)
        pending_total = int(row.get("pending_total", running + queued) or 0)
    except (TypeError, ValueError):
        return None
    if max(running, queued, pending_total) <= 0:
        return None

    cap, cap_source = high_cost_cap_for_evidence(
        evidence,
        family,
        high_cost_pending_cap=cfg.high_cost_pending_cap,
        high_cost_pending_promoted_cap=cfg.high_cost_pending_promoted_cap,
        high_cost_pending_deep_stall_cap=cfg.high_cost_pending_deep_stall_cap,
        high_cost_pending_dry_pivot_cap=cfg.high_cost_pending_dry_pivot_cap,
        high_cost_pending_strong_cap=cfg.high_cost_pending_strong_cap,
        high_cost_dry_pivot_min_gpu_h=cfg.high_cost_dry_pivot_min_gpu_h,
        high_cost_dry_pivot_min_completed_children=cfg.high_cost_dry_pivot_min_completed_children,
        high_cost_dry_pivot_max_best_recent_su_per_gpu_h=cfg.high_cost_dry_pivot_max_best_recent_su_per_gpu_h,
        high_cost_strong_min_su=cfg.high_cost_strong_min_su,
        high_cost_strong_min_gpu_h=cfg.high_cost_strong_min_gpu_h,
        high_cost_strong_min_recent_su_per_gpu_h=cfg.high_cost_strong_min_recent_su_per_gpu_h,
        high_cost_strong_best_fraction=cfg.high_cost_strong_best_fraction,
        high_cost_stale_recent_gpu_h=cfg.high_cost_stale_recent_gpu_h,
        min_su_per_gpu_h=cfg.cost_aware_min_su_per_gpu_h,
        high_cost_repeated_support_min_proposed=cfg.high_cost_repeated_support_min_proposed,
        high_cost_repeated_support_max_started_fraction=cfg.high_cost_repeated_support_max_started_fraction,
        high_cost_repeated_support_cap=cfg.high_cost_repeated_support_cap,
        high_cost_repeated_support_max_negative_gpu_h=cfg.high_cost_repeated_support_max_negative_gpu_h,
        high_cost_repeated_support_max_timeouts=cfg.high_cost_repeated_support_max_timeouts,
    )
    cap = max(0, int(cap))
    # Selection should not reject a fresh evidence-backed high-cost candidate
    # solely because an older same-family candidate is queued. Queued work is
    # already ordered by the controller's pending-priority repair and actual
    # worker concurrency is enforced at dispatch. Use running high-cost workers
    # as the hard selector pressure; keep queued/pending_total in the reason for
    # auditability.
    if running < cap:
        return None
    return (
        "capacity_pressure_high_cost_running_cap:"
        f"family={family}:running={running}:cap={cap}:"
        f"queued={queued}:pending_total={pending_total}:cap_source={cap_source}"
    )


def high_cost_capacity_blocked_families_for_cfg(
    evidence: EvidenceSummary,
    cfg: SelectorConfig,
) -> set[str]:
    """High-cost families whose running workers already fill admission cap."""
    pending_load = getattr(evidence, "pending_family_load", {}) or {}
    pending_by_family = (
        pending_load.get("by_family", {})
        if isinstance(pending_load, dict) else {}
    )
    if not isinstance(pending_by_family, dict):
        return set()

    blocked: set[str] = set()
    # See high_cost_capacity_block_reason_for_cfg: evidence records may carry
    # stale telemetry, but the cap is enabled only by SelectorConfig/env.
    high_cost_families = set(cfg.high_cost_pending_families)

    for fam, row in pending_by_family.items():
        if fam not in high_cost_families or not isinstance(row, dict):
            continue
        if high_cost_capacity_block_reason_for_cfg(evidence, cfg, str(fam)) is not None:
            blocked.add(str(fam))
    return blocked


def _high_cost_batch_hard_caps_for_cfg(
    evidence: EvidenceSummary,
    cfg: SelectorConfig,
) -> dict[str, int]:
    """Per-selection hard caps for delayed-feedback families.

    Diversity caps are soft elsewhere so the selector can keep GPUs occupied
    when only one family has work. BindCraft-style delayed-feedback lanes are
    different: filling every worker can hide score-conversion feedback for
    hours. Keep at least one non-high-cost lane when more than one worker is
    available, while preserving one-slot runs and explicit no-cap ablations.
    """
    if cfg.available_slots <= 0 or not cfg.high_cost_pending_families:
        return {}
    try:
        reserve = int(os.environ.get("TREX_HIGH_COST_RESERVE_NON_HIGH_COST_SLOTS", "1"))
    except ValueError:
        reserve = 1
    reserve = max(0, reserve)
    slot_cap = cfg.available_slots
    if cfg.available_slots > 1 and reserve > 0:
        slot_cap = max(1, cfg.available_slots - reserve)

    out: dict[str, int] = {}
    for fam in cfg.high_cost_pending_families:
        cap, _source = high_cost_cap_for_evidence(
            evidence,
            str(fam),
            high_cost_pending_cap=cfg.high_cost_pending_cap,
            high_cost_pending_promoted_cap=cfg.high_cost_pending_promoted_cap,
            high_cost_pending_deep_stall_cap=cfg.high_cost_pending_deep_stall_cap,
            high_cost_pending_dry_pivot_cap=cfg.high_cost_pending_dry_pivot_cap,
            high_cost_pending_strong_cap=cfg.high_cost_pending_strong_cap,
            high_cost_dry_pivot_min_gpu_h=cfg.high_cost_dry_pivot_min_gpu_h,
            high_cost_dry_pivot_min_completed_children=cfg.high_cost_dry_pivot_min_completed_children,
            high_cost_dry_pivot_max_best_recent_su_per_gpu_h=cfg.high_cost_dry_pivot_max_best_recent_su_per_gpu_h,
            high_cost_strong_min_su=cfg.high_cost_strong_min_su,
            high_cost_strong_min_gpu_h=cfg.high_cost_strong_min_gpu_h,
            high_cost_strong_min_recent_su_per_gpu_h=cfg.high_cost_strong_min_recent_su_per_gpu_h,
            high_cost_strong_best_fraction=cfg.high_cost_strong_best_fraction,
            high_cost_stale_recent_gpu_h=cfg.high_cost_stale_recent_gpu_h,
            min_su_per_gpu_h=cfg.cost_aware_min_su_per_gpu_h,
            high_cost_repeated_support_min_proposed=cfg.high_cost_repeated_support_min_proposed,
            high_cost_repeated_support_max_started_fraction=cfg.high_cost_repeated_support_max_started_fraction,
            high_cost_repeated_support_cap=cfg.high_cost_repeated_support_cap,
            high_cost_repeated_support_max_negative_gpu_h=cfg.high_cost_repeated_support_max_negative_gpu_h,
            high_cost_repeated_support_max_timeouts=cfg.high_cost_repeated_support_max_timeouts,
        )
        out[str(fam)] = max(1, min(int(cap), int(slot_cap)))
    return out


def _rv_get(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _rv_record_recent_su(row: Any) -> int:
    return int(_rv_get(row, "record_recent_new_su", _rv_get(row, "new_su_recent", 0)) or 0)


def _rv_record_recent_gpu_h(row: Any) -> float:
    return float(_rv_get(row, "record_recent_route_gpu_h", _rv_get(row, "recent_route_gpu_h", 0.0)) or 0.0)


def _rv_record_recent_rate(row: Any) -> float | None:
    rate = _rv_get(row, "record_recent_new_su_per_route_gpu_h", None)
    if rate is None:
        rate = _rv_get(row, "recent_new_su_per_route_gpu_h", None)
    return rate


def _rv_safe_record_allowed(row: Any) -> bool:
    route_role = str(_rv_get(row, "route_role", "") or "")
    try:
        canonical_refilter_gpu_h = float(_rv_get(row, "canonical_refilter_gpu_h", 0.0) or 0.0)
    except (TypeError, ValueError):
        canonical_refilter_gpu_h = 0.0
    return canonical_refilter_gpu_h <= 0.0 and "score_conversion" not in route_role


def _rv_current_rate(row: Any) -> float | None:
    """Decision-safe route value for selector admission.

    Prefer recent route value, but fall back to full-route lifetime value after
    excluding unsafe score-conversion record_recent rates. High-cost cap opening
    intentionally needs this lifetime route value for delayed-feedback lanes
    such as BindCraft; stale exact-replay suppression is handled separately by
    _is_dry_duplicate_exact_replay().
    """
    rate = _rv_get(row, "gpu_recent_new_su_per_route_gpu_h", None)
    if rate is not None:
        return rate
    rate = _rv_get(row, "medium_recent_new_su_per_route_gpu_h", None)
    if rate is not None:
        return rate
    record_rate = _rv_record_recent_rate(row)
    if record_rate is not None and _rv_safe_record_allowed(row):
        return record_rate
    return _rv_get(row, "new_su_per_route_gpu_h", None)


def _rv_medium_recent_su(row: Any) -> int:
    return int(_rv_get(row, "medium_recent_new_su", 0) or 0)


def _rv_any_recent_su(row: Any) -> float:
    values = [
        float(_rv_get(row, "new_su_recent_gpu", 0.0) or 0.0),
        float(_rv_medium_recent_su(row) or 0.0),
    ]
    if _rv_safe_record_allowed(row):
        values.append(float(_rv_record_recent_su(row) or 0.0))
    return max(values)


def _candidate_config_signature(c: ActionCandidate) -> str:
    from .evidence_reducer import canonical_config_signature
    return canonical_config_signature(c.config_delta)


def _route_strategy_keys_by_evidence_ref(evidence: EvidenceSummary) -> dict[str, set[str]]:
    """Map ResultRecord IDs seen in route rows to their exact route keys.

    Parent-bound actions need this to avoid treating `complexa->MPNN` evidence as
    evidence for `boltzgen->MPNN` simply because the action family/config match.
    """
    out: dict[str, set[str]] = {}
    for row in getattr(evidence, "route_values", None) or []:
        if _rv_get(row, "scope") != "route":
            continue
        key = str(_rv_get(row, "strategy_key", "") or "")
        if not key:
            continue
        for rid in (_rv_get(row, "evidence_refs", []) or []):
            if rid:
                out.setdefault(str(rid), set()).add(key)
    return out


def _candidate_parent_route_keys(c: ActionCandidate, evidence: EvidenceSummary) -> set[str]:
    if not c.parent_result_id:
        return set()
    return _route_strategy_keys_by_evidence_ref(evidence).get(str(c.parent_result_id), set())


def _matching_route_value_rows(
    c: ActionCandidate,
    evidence: EvidenceSummary,
    *,
    scope: str = "route",
) -> list[Any]:
    """Exact route/config rows matching this candidate.

    Family-level cost controls already operate through method_health. This helper
    is intentionally stricter: it only applies route-value pressure to the same
    action family/operator/validated config signature, and for parent-bound
    actions it also respects the parent/root route context. An unsuccessful
    beam_width=8 route must not poison beam_width=4, and a collapsed
    complexa->MPNN route must not poison boltzgen->MPNN.
    """
    if not _su_dedup_trusted(evidence):
        return []
    sig = _candidate_config_signature(c)
    parent_keys = _candidate_parent_route_keys(c, evidence)
    out: list[Any] = []
    for row in getattr(evidence, "route_values", None) or []:
        if _rv_get(row, "scope") != scope:
            continue
        action_family = _rv_get(row, "action_family") or _rv_get(row, "family")
        if action_family != c.method_family:
            continue
        op = _rv_get(row, "operator_id")
        if op and c.operator_id and op != c.operator_id:
            continue
        row_sig = _rv_get(row, "config_signature")
        if row_sig and row_sig not in {"family_rollup", sig}:
            continue
        row_parent = _rv_get(row, "parent_strategy_key")
        if row_parent:
            if not c.parent_result_id or str(row_parent) not in parent_keys:
                continue
        out.append(row)
    return out


def best_route_value_row(c: ActionCandidate, evidence: EvidenceSummary) -> Any | None:
    rows = _matching_route_value_rows(c, evidence, scope="route")
    if not rows:
        return None
    status_rank = {
        "promote": 0,
        "healthy": 1,
        "diversify": 2,
        "observed": 3,
        "defer": 4,
        "collapse_risk": 5,
    }

    def _row_key(r: Any) -> tuple[int, int, float, float, float, float]:
        rate = _rv_current_rate(r)
        try:
            rate_f = float(rate or 0.0)
        except (TypeError, ValueError):
            rate_f = 0.0
        recent_su = _rv_any_recent_su(r)
        current_value_rank = 0 if recent_su > 0.0 and rate_f > 0.0 else 1
        return (
            current_value_rank,
            status_rank.get(str(_rv_get(r, "status", "")), 9),
            -rate_f,
            -recent_su,
            -float(_rv_get(r, "new_su_per_route_gpu_h", 0.0) or 0.0),
            -float(_rv_get(r, "route_gpu_h", 0.0) or 0.0),
        )

    return sorted(rows, key=_row_key)[0]


# Compatibility alias for downstream callers.
_best_route_value_row = best_route_value_row


def _candidate_route_penalty(c: ActionCandidate, evidence: EvidenceSummary) -> int:
    row = best_route_value_row(c, evidence)
    if row is None:
        return 0
    marginal = str(_rv_get(row, "marginal_status", "") or "")
    # Duplicate/collapse features are warnings unless the route is dry on the
    # GPU-hour marginal window. If it is still buying new SU/GPU-h, respect the
    # Supervisor rank and let the LLM decide whether to keep exploiting it.
    if marginal in {"productive", "productive_but_duplicate", "delayed_productive", "delayed_productive_duplicate"}:
        return 0
    status = str(_rv_get(row, "status", ""))
    if status == "diversify":
        return 1
    if status in {"defer", "collapse_risk"} or marginal in {"dry", "dry_duplicate", "dry_low_quality"}:
        return 2
    return 0


def _candidate_route_rate(c: ActionCandidate, evidence: EvidenceSummary) -> float:
    row = best_route_value_row(c, evidence)
    if row is None:
        return 0.0
    try:
        return float(_rv_current_rate(row) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidate_marginal_status(c: ActionCandidate, evidence: EvidenceSummary) -> str:
    row = best_route_value_row(c, evidence)
    if row is None:
        return "unknown"
    return str(_rv_get(row, "marginal_status", "observed") or "observed")


def _diagnostic_backlog_family_row(evidence: EvidenceSummary, family: str) -> dict[str, Any]:
    backlog = getattr(evidence, "diagnostic_chain_backlog", {}) or {}
    if not isinstance(backlog, dict):
        return {}
    by_family = backlog.get("by_family", backlog)
    if not isinstance(by_family, dict):
        return {}
    row = by_family.get(family, {})
    return row if isinstance(row, dict) else {}


def _backlog_saturated_route_deferred_generator(
    c: ActionCandidate,
    evidence: EvidenceSummary,
    cfg: SelectorConfig,
    route_deferred_ids: set[str],
) -> bool:
    """True when a route-deferred diagnostic generator should not be revived.

    This is intentionally narrower than route deferral itself. It only applies
    after route evidence has already demoted the exact candidate, and only when
    the family has a large unscored diagnostic backlog with enough sampled
    score-conversions to show no SU/near-miss/promising-pending signal. Productive
    or still-promising diagnostic generators remain eligible.
    """
    if c.candidate_id not in route_deferred_ids:
        return False
    if not c.candidate_id.startswith("evidence_fallback_cross_family"):
        return False
    if c.method_family not in set(cfg.backlog_saturated_generator_families):
        return False
    row = _diagnostic_backlog_family_row(evidence, c.method_family)
    try:
        unscored = int(
            row.get("unscored_artifacts", row.get("pending_chain_candidates", 0)) or 0
        )
        completed = int(row.get("completed_refilters", 0) or 0)
        promising_pending = int(
            row.get(
                "proxy_promising_pending_refilter",
                row.get("pending_promising_score_conversion_count", 0),
            )
            or 0
        )
        native_pending = int(
            row.get(
                "native_or_proxy_pending_refilter",
                row.get("native_strict_like_pending_refilter", 0),
            )
            or 0
        )
    except (TypeError, ValueError):
        return False
    if unscored < cfg.backlog_saturated_min_unscored:
        return False
    if completed < cfg.backlog_saturated_min_completed_refilters:
        return False
    if promising_pending > 0 or native_pending > 0:
        return False

    h = (getattr(evidence, "method_health", None) or {}).get(c.method_family)
    if h is not None:
        su_fields = (
            "strict_yield_su",
            "chained_strict_yield_su",
            "strict_yield_su_recent",
            "chained_strict_yield_su_recent",
        )
        near_fields = ("near_miss_yield", "near_miss_yield_recent")
        try:
            if any(float(_mh_value(h, f, 0.0) or 0.0) > 0.0 for f in su_fields + near_fields):
                return False
        except (TypeError, ValueError):
            return False

    # Route rows can carry parent/config-specific evidence that is more current
    # than family rollups. Any SU or near-miss on the exact route keeps it
    # eligible despite backlog.
    for rv in _matching_route_value_rows(c, evidence, scope="route"):
        try:
            if float(_rv_get(rv, "new_su", 0.0) or 0.0) > 0.0:
                return False
            if float(_rv_get(rv, "near_miss_count", 0.0) or 0.0) > 0.0:
                return False
            if float(_rv_get(rv, "pending_promising_score_conversion_count", 0.0) or 0.0) > 0.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _is_dry_duplicate_exact_replay(c: ActionCandidate, evidence: EvidenceSummary) -> bool:
    if not c.candidate_id.startswith("route_replay"):
        return False
    row = best_route_value_row(c, evidence)
    if row is None:
        return False
    if int(_rv_get(row, "new_su_recent_gpu", 0) or 0) > 0:
        return False
    if _rv_medium_recent_su(row) > 0:
        return False
    if float(_rv_get(row, "gpu_recent_new_su_per_route_gpu_h", 0.0) or 0.0) > 0.0:
        return False

    marginal = str(_rv_get(row, "marginal_status", "") or "")
    if marginal == "dry_duplicate":
        return True

    # Route replay can also be stale when the exact route has only lifetime SU
    # memory and no current/medium/record-window signal. This catches the CD45
    # failure mode where a high lifetime SU/GPU-h Complexa route was launched as
    # exploit after a long dry plateau. Duplicate labels alone are not enough;
    # require a stale target-level dry signal as well.
    try:
        dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        dry_gpu_h = 0.0
    if dry_gpu_h < 12.0:
        return False
    try:
        gpu_or_medium_recent_gpu_h = max(
            float(_rv_get(row, "gpu_recent_route_gpu_h", 0.0) or 0.0),
            float(_rv_get(row, "medium_recent_route_gpu_h", 0.0) or 0.0),
        )
    except (TypeError, ValueError):
        gpu_or_medium_recent_gpu_h = 0.0
    has_gpu_or_medium_recent_su = (
        int(_rv_get(row, "new_su_recent_gpu", 0) or 0) > 0
        or _rv_medium_recent_su(row) > 0
    )
    has_gpu_or_medium_recent_rate = (
        _rv_get(row, "gpu_recent_new_su_per_route_gpu_h", None) is not None
        or _rv_get(row, "medium_recent_new_su_per_route_gpu_h", None) is not None
    )
    if gpu_or_medium_recent_gpu_h > 0.0 or has_gpu_or_medium_recent_su or has_gpu_or_medium_recent_rate:
        return False
    status = str(_rv_get(row, "status", "") or "")
    try:
        duplicate_fraction = float(_rv_get(row, "duplicate_bin_fraction", 0.0) or 0.0)
    except (TypeError, ValueError):
        duplicate_fraction = 0.0
    try:
        strict_per_su = float(_rv_get(row, "strict_per_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        strict_per_su = 0.0
    return status in {"diversify", "collapse_risk"} or duplicate_fraction >= 0.70 or strict_per_su >= 8.0


def _is_material_diversity_candidate(c: ActionCandidate, d: CandidateDecision | None) -> bool:
    if d is None or d.mode not in {"rescue", "explore"}:
        return False
    if c.candidate_id.startswith("route_replay"):
        return False
    if c.method_family in {"proteinmpnn_redesign", "bindcraft", "boltzgen", "complexa_mcts"}:
        return True
    if c.method_family in {"structure_refilter"}:
        return c.parent_result_id is not None and not is_canonical_score_conversion(c)
    if c.method_family.startswith("complexa_") and d.mode == "explore":
        return True
    return False


def _near_miss_parent_ids(evidence: EvidenceSummary) -> set[str]:
    out = {str(x) for x in (getattr(evidence, "production_near_miss_ids", None) or []) if x}
    for ex in getattr(evidence, "exemplars", None) or []:
        kind = ex.get("kind") if isinstance(ex, dict) else getattr(ex, "kind", None)
        rid = ex.get("result_id") if isinstance(ex, dict) else getattr(ex, "result_id", None)
        if kind == "near_miss" and rid:
            out.add(str(rid))
    for recipe in getattr(evidence, "recipes", None) or []:
        cls = recipe.get("recipe_class") if isinstance(recipe, dict) else getattr(recipe, "recipe_class", None)
        if cls != "near_miss":
            continue
        reps = recipe.get("representative_result_ids") if isinstance(recipe, dict) else getattr(recipe, "representative_result_ids", None)
        out.update(str(x) for x in (reps or []) if x)
    return out


def _has_positive_recent_su(evidence: EvidenceSummary) -> bool:
    if not _su_dedup_trusted(evidence):
        return False
    try:
        recent_rate = float(getattr(evidence, "su_per_gpu_h_recent", 0.0) or 0.0)
    except (TypeError, ValueError):
        recent_rate = 0.0
    return int(getattr(evidence, "run_su_count_delta", 0) or 0) > 0 or recent_rate > 0.0


def _candidate_has_current_su_value(c: ActionCandidate, evidence: EvidenceSummary) -> bool:
    if _candidate_route_rate(c, evidence) > 0.0 and not _is_dry_duplicate_exact_replay(c, evidence):
        return True
    h = (getattr(evidence, "method_health", None) or {}).get(c.method_family)
    if h is None:
        return False
    for field in (
        "su_per_gpu_h_recent",
        "chained_su_per_gpu_h_recent",
    ):
        try:
            if float(_mh_value(h, field, 0.0) or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _candidate_safe_for_strong_productive_momentum(
    c: ActionCandidate,
    evidence: EvidenceSummary,
    cfg: SelectorConfig,
) -> bool:
    """Whether this exact candidate may receive the 2-slot productive boost.

    The one-slot momentum guard can preserve any route with current SU signal.
    The stronger two-slot repair is intentionally narrower: it should recover
    fast CD45/PDL1-style cheap exploitation, but must not let a delayed high-cost
    family occupy all workers after a small early hit.
    """
    if c.method_family in set(cfg.high_cost_pending_families):
        return False
    if is_canonical_score_conversion(c):
        return False
    if c.estimated_cost_class not in set(cfg.productive_wall_momentum_safe_cost_classes):
        return False
    if _is_dry_duplicate_exact_replay(c, evidence):
        return False
    min_rate = cfg.productive_wall_momentum_route_min_su_per_gpu_h
    if _candidate_route_rate(c, evidence) >= min_rate:
        return True
    # Allow family-level recent productivity to support nearby configurations
    # without exact-route observations. High-cost workers and system evaluation
    # remain excluded.
    h = (getattr(evidence, "method_health", None) or {}).get(c.method_family)
    if h is None:
        return False
    for field in ("su_per_gpu_h_recent", "chained_su_per_gpu_h_recent"):
        try:
            if float(_mh_value(h, field, 0.0) or 0.0) >= min_rate:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _productive_wall_momentum_target_slots(
    evidence: EvidenceSummary,
    cfg: SelectorConfig,
    exploit_candidates: list[ActionCandidate],
    *,
    sup_decs: dict[str, CandidateDecision] | None = None,
    family_cost_penalties: dict[str, int] | None = None,
) -> int:
    if not cfg.productive_wall_momentum_guard or not exploit_candidates:
        return 0
    if evidence.state_label not in {"productive", "productive_duplicate", "rescue_rich"}:
        return 0
    if bool(getattr(evidence, "strict_duplicate_collapse_signal", False)):
        return 0
    try:
        dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        dry_gpu_h = 0.0
    if dry_gpu_h > cfg.productive_wall_momentum_max_dry_gpu_h:
        return 0
    valued = [c for c in exploit_candidates if _candidate_has_current_su_value(c, evidence)]
    if not valued:
        return 0
    try:
        recent_rate = float(getattr(evidence, "su_per_gpu_h_recent", 0.0) or 0.0)
    except (TypeError, ValueError):
        recent_rate = 0.0
    hwm_delta = int(getattr(evidence, "run_su_hwm_delta", None) or getattr(evidence, "run_su_count_delta", 0) or 0)
    target = 1
    if (
        cfg.available_slots >= 3
        and dry_gpu_h <= cfg.productive_wall_momentum_strong_max_dry_gpu_h
        and (recent_rate >= cfg.productive_wall_momentum_strong_min_recent_su_per_gpu_h or hwm_delta >= 2)
    ):
        ranked_exploit = sorted(
            exploit_candidates,
            key=lambda c: _tiebreak_key(
                c,
                sup_decs or {},
                evidence,
                overrepresented_families=set(),
                family_cost_penalties=family_cost_penalties,
            ),
        )
        top_two = ranked_exploit[:2]
        if (
            len(top_two) >= 2
            and all(_candidate_safe_for_strong_productive_momentum(c, evidence, cfg) for c in top_two)
        ):
            target = 2
    return min(target, len(valued), cfg.available_slots)


def _observed_parent_bound_route(c: ActionCandidate, evidence: EvidenceSummary) -> bool:
    """Whether this exact parent-bound rescue route already produced feedback.

    The near-miss floor is a one-shot realization repair, not a replay policy.
    Only parent-specific route rows count; family rollups must not suppress a
    different parent's first rescue attempt.
    """
    if not c.parent_result_id:
        return False
    parent = str(c.parent_result_id)
    for row in _matching_route_value_rows(c, evidence, scope="route"):
        refs = {str(x) for x in (_rv_get(row, "evidence_refs", []) or []) if x}
        parent_specific = bool(_rv_get(row, "parent_strategy_key", None)) or parent in refs
        if not parent_specific:
            continue
        if (
            float(_rv_get(row, "route_gpu_h", 0.0) or 0.0) > 0.0
            or int(_rv_get(row, "attempts", 0) or 0) > 0
            or int(_rv_get(row, "completions", 0) or 0) > 0
        ):
            return True
    return False


def _evidence_allows_low_cost_near_miss_rescue_floor(evidence: EvidenceSummary) -> bool:
    if evidence.state_label not in {"stalled", "rescue_rich", "deep_stall"}:
        return False
    if _has_positive_recent_su(evidence):
        return False
    if not _near_miss_dedup_trusted(evidence):
        return False
    return bool(_near_miss_parent_ids(evidence)) or int(getattr(evidence, "near_miss_count", 0) or 0) > 0


def _is_low_cost_near_miss_rescue_candidate(
    c: ActionCandidate,
    d: CandidateDecision | None,
    evidence: EvidenceSummary,
) -> bool:
    if d is None or d.mode != "rescue":
        return False
    if c.candidate_id.startswith("route_replay"):
        return False
    if c.estimated_cost_class != "low":
        return False
    if not c.parent_result_id:
        return False
    if str(c.parent_result_id) not in _near_miss_parent_ids(evidence):
        return False
    if c.method_family == "proteinmpnn_redesign":
        pass
    elif c.method_family == "structure_refilter":
        if is_canonical_score_conversion(c):
            return False
    else:
        return False
    return not _observed_parent_bound_route(c, evidence)


def _candidate_effective_cost_class(c: ActionCandidate, d: CandidateDecision | None) -> str:
    return str((d.resource_class if d else None) or c.estimated_cost_class or "standard")


def _candidate_cost_rank(c: ActionCandidate, d: CandidateDecision | None) -> int:
    return {"low": 0, "diagnostic": 1, "standard": 2, "extended": 3}.get(
        _candidate_effective_cost_class(c, d), 99
    )


def _wasteful_families(evidence: EvidenceSummary, *, waste_gpu_h: float) -> set[str]:
    """Back-compatible view of truly dead lanes used by older smoke tests."""
    return {
        fam for fam, penalty in _family_cost_penalties(
            evidence, waste_gpu_h=waste_gpu_h).items()
        if penalty >= 2
    }


def _tiebreak_key(
    c: ActionCandidate,
    sup_decs: dict[str, CandidateDecision],
    evidence: EvidenceSummary,
    overrepresented_families: set[str] | None = None,
    family_cost_penalties: dict[str, int] | None = None,
):
    d = sup_decs.get(c.candidate_id)
    rank = d.rank_in_mode if d else 10**6
    cost_order = {"low": 0, "diagnostic": 1, "standard": 2, "extended": 3}
    cost_class = d.resource_class if d else c.estimated_cost_class
    family_penalty = 1 if c.method_family in (overrepresented_families or set()) else 0
    cost_penalty = int((family_cost_penalties or {}).get(c.method_family, 0))
    route_penalty = _candidate_route_penalty(c, evidence)
    route_rate_key = -_candidate_route_rate(c, evidence)

    if d is None:
        # Fallback path: no Supervisor rank exists, so realized route/family value
        # is the scientific order within the feasible mode pool.
        rate_key = route_rate_key
        if rate_key == 0.0 and _su_dedup_trusted(evidence):
            h = (getattr(evidence, "method_health", {}) or {}).get(c.method_family)
            if h is not None:
                rate_key = -max(
                    float(_mh_value(h, "su_per_gpu_h_recent", 0.0) or 0.0),
                    float(_mh_value(h, "chained_su_per_gpu_h_recent", 0.0) or 0.0),
                )
        ranked_rate_key = rate_key
        post_rank_rate_key = 0.0
    else:
        # Ranked path: the Supervisor already interpreted the EvidenceSummary.
        # Keep hard/soft evidence penalties first, but do not let historical route
        # rate silently invert its rank-1 scientific choice. Route value is only a
        # tie-break among equal-ranked candidates in the same mode.
        ranked_rate_key = 0.0
        post_rank_rate_key = route_rate_key

    return (
        family_penalty,
        cost_penalty,
        route_penalty,
        ranked_rate_key,
        rank,
        post_rank_rate_key,
        cost_order.get(cost_class, 99),
        cost_order.get(c.estimated_cost_class, 99),
        c.candidate_id,
    )


def _build_quota_candidate_contexts(
    evidence: EvidenceSummary,
    candidates: list[ActionCandidate],
    supervisor_decisions: dict[str, CandidateDecision],
    candidate_modes: dict[str, str],
    family_cost_penalties: dict[str, int],
    cfg: SelectorConfig,
    *,
    repeated_support_probe_enabled: bool,
    near_miss_rescue_enabled: bool,
) -> list[QuotaCandidateContext]:
    """Freeze evidence-derived candidate facts consumed by quota repairs."""

    repairs_need_priority = bool(
        repeated_support_probe_enabled
        or near_miss_rescue_enabled
        or cfg.marginal_diversity_override
    )
    ordered_candidates = (
        sorted(
            candidates,
            key=lambda candidate: _tiebreak_key(
                candidate,
                supervisor_decisions,
                evidence,
                overrepresented_families=set(),
                family_cost_penalties=family_cost_penalties,
            ),
        )
        if repairs_need_priority
        else list(candidates)
    )
    contexts: list[QuotaCandidateContext] = []
    for priority, candidate in enumerate(ordered_candidates):
        decision = supervisor_decisions.get(candidate.candidate_id)
        repeated_support_eligible = bool(
            repeated_support_probe_enabled
            and decision is not None
            and decision.rank_in_mode <= cfg.repeated_support_probe_rank_max
            and decision.mode in ("exploit", "rescue", "explore")
            and _repeated_support_probe_cap(
                evidence,
                candidate.method_family,
                min_proposed=cfg.high_cost_repeated_support_min_proposed,
                max_started_fraction=(
                    cfg.high_cost_repeated_support_max_started_fraction
                ),
                cap=cfg.high_cost_repeated_support_cap,
                max_negative_gpu_h=(
                    cfg.high_cost_repeated_support_max_negative_gpu_h
                ),
                max_timeouts=cfg.high_cost_repeated_support_max_timeouts,
            ) is not None
        )
        contexts.append(QuotaCandidateContext(
            candidate_id=candidate.candidate_id,
            method_family=candidate.method_family,
            mode=candidate_modes.get(candidate.candidate_id),
            parent_result_id=candidate.parent_result_id,
            priority=priority,
            effective_cost_rank=_candidate_cost_rank(candidate, decision),
            repeated_support_probe_eligible=repeated_support_eligible,
            low_cost_near_miss_rescue_eligible=bool(
                near_miss_rescue_enabled
                and _is_low_cost_near_miss_rescue_candidate(
                    candidate,
                    decision,
                    evidence,
                )
            ),
            dry_duplicate_exact_replay=bool(
                cfg.marginal_diversity_override
                and _is_dry_duplicate_exact_replay(candidate, evidence)
            ),
            material_diversity_candidate=bool(
                cfg.marginal_diversity_override
                and _is_material_diversity_candidate(candidate, decision)
            ),
        ))
    return contexts


def _build_ranking_candidate_contexts(
    candidates: list[ActionCandidate],
    candidate_modes: dict[str, str],
    supervisor_decisions: dict[str, CandidateDecision],
    evidence: EvidenceSummary,
    overrepresented_families: set[str],
    family_cost_penalties: dict[str, int],
) -> dict[str, RankingCandidateContext]:
    """Freeze the existing Selector tiebreak into explicit priorities."""

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: _tiebreak_key(
            candidate,
            supervisor_decisions,
            evidence,
            overrepresented_families,
            family_cost_penalties,
        ),
    )
    contexts: dict[str, RankingCandidateContext] = {}
    for priority, candidate in enumerate(ordered_candidates):
        decision = supervisor_decisions.get(candidate.candidate_id)
        contexts[candidate.candidate_id] = RankingCandidateContext(
            candidate=candidate,
            mode=candidate_modes.get(candidate.candidate_id),
            priority=priority,
            rank_in_mode=(decision.rank_in_mode if decision else None),
            global_rank=(decision.global_rank if decision else None),
        )
    return contexts


def select_launches(
    evidence: EvidenceSummary,
    candidates: list[ActionCandidate],
    sup_out: SupervisorOutput,
    *,
    cfg: SelectorConfig | None = None,
    tick_id: str = "tick_xxx",
    fallback_reason: str | None = None,
    all_explore_backends_unhealthy: bool = False,
    route_backlog_saturated: bool = False,
    recent_modes: list[str] | None = None,
    mode_credit: dict[str, float] | None = None,
    candidate_mode_hint: dict[str, str] | None = None,
    panel_live: bool = False,
    recent_started_families: list[str] | None = None,
) -> tuple[list[LaunchDecision], dict[str, Any]]:
    """Returns (launches, debug_dict).

    `fallback_reason`, if set, indicates the supervisor output was a
    fallback and we should use DEFAULT_MIXTURES for the state instead
    of the supervisor's mode_mixture.

    `recent_modes`: realised modes of worker-started candidates (chronological,
    most-recent last), used by the deterministic_deficit ablation and debug.

    `mode_credit`: started-dispatch-corrected fractional credit reconstructed
    from prior SupervisorDecision mixtures and DispatchRecord(started) modes.
    Production fractional_carry uses this to realize small rescue/explore shares
    without a fixed K-window.

    `candidate_mode_hint`: candidate_id → mode from HypothesisCard
    mode_affinity, used to assign modes on the fallback path (no Supervisor
    per-candidate decisions) so explore-affinity families are not collapsed
    into one cheapest-first pool.
    """
    cfg = cfg or SelectorConfig()

    policy = resolve_selection_policy(
        SelectionPolicyRequest(
            evidence=evidence,
            candidates=candidates,
            supervisor_output=sup_out,
            fallback_reason=fallback_reason,
            candidate_mode_hints=candidate_mode_hint or {},
            derive_mixture_from_ranking=cfg.derive_mixture_from_ranking,
            realize_global_priority=cfg.realize_global_priority,
            all_explore_backends_unhealthy=all_explore_backends_unhealthy,
            route_backlog_saturated=route_backlog_saturated,
            panel_live=panel_live,
        )
    )
    sup_decs = policy.supervisor_decisions
    source = policy.source
    raw_mixture = policy.raw_mixture
    mixture = policy.clamped_mixture
    clamp_log = list(policy.clamp_log)
    cat_b_on = policy.category_b_enabled
    gate_reasons = list(policy.category_b_reasons)
    use_supervisor_ranks = policy.use_supervisor_ranks
    global_priority_used = policy.global_priority_used
    have_modes = policy.has_mode_information

    def _resolved_mode(candidate: ActionCandidate) -> str | None:
        return policy.mode_for(candidate)

    family_cost_penalties = family_cost_penalties_for_cfg(evidence, cfg)
    capacity_blocked_families = (
        high_cost_capacity_blocked_families_for_cfg(evidence, cfg)
    )
    capacity_block_reasons = {
        family: (
            high_cost_capacity_block_reason_for_cfg(evidence, cfg, family)
            or "capacity_pressure_high_cost_running_cap"
        )
        for family in capacity_blocked_families
    }
    candidate_route_penalties = {
        candidate.candidate_id: _candidate_route_penalty(candidate, evidence)
        for candidate in candidates
        if candidate.feasibility.all_ok()
    }
    admission = admit_candidates(
        CandidateAdmissionRequest(
            candidates=candidates,
            candidate_modes=policy.candidate_modes,
            has_mode_information=policy.has_mode_information,
            global_priority_used=global_priority_used,
            category_b_enabled=cat_b_on,
            family_cost_penalties=family_cost_penalties,
            capacity_block_reasons=capacity_block_reasons,
            candidate_route_penalties=candidate_route_penalties,
            defer_penalty=cfg.cost_aware_defer_penalty,
        )
    )
    feasible = list(admission.eligible_candidates)
    feasibility_by_mode = admission.feasibility_by_mode
    capacity_pressure_feasible = list(admission.capacity_pressure_candidates)
    capacity_deferred_feasible = list(admission.capacity_deferred_candidates)
    deferred_cost_families = set(admission.cost_deferred_families)
    deferred_feasible = list(admission.cost_deferred_candidates)
    route_deferred_feasible = list(admission.route_deferred_candidates)
    route_deferred_ids = set(admission.route_deferred_candidate_ids)
    route_cost_penalties = admission.route_cost_penalties
    effective_k = effective_mode_window_k(evidence, cfg)
    evidence_allows_near_miss_rescue = bool(
        cfg.low_cost_near_miss_rescue_floor
        and not global_priority_used
        and cfg.available_slots == 1
        and use_supervisor_ranks
        and _evidence_allows_low_cost_near_miss_rescue_floor(evidence)
    )
    quota_candidate_contexts = _build_quota_candidate_contexts(
        evidence,
        feasible,
        sup_decs,
        policy.candidate_modes,
        family_cost_penalties,
        cfg,
        repeated_support_probe_enabled=bool(
            cfg.realize_repeated_support_probe
            and use_supervisor_ranks
            and not global_priority_used
        ),
        near_miss_rescue_enabled=evidence_allows_near_miss_rescue,
    )

    exploit_candidates = [
        candidate
        for candidate in feasible
        if _resolved_mode(candidate) == "exploit"
    ]
    productive_wall_target_slots = (
        _productive_wall_momentum_target_slots(
            evidence,
            cfg,
            exploit_candidates,
            sup_decs=sup_decs,
            family_cost_penalties=family_cost_penalties,
        )
        if cfg.productive_wall_momentum_guard
        and have_modes
        and not global_priority_used
        else 0
    )
    quota_result = realize_mode_quotas(ModeQuotaRequest(
        mixture=mixture,
        available_slots=cfg.available_slots,
        feasibility_by_mode=feasibility_by_mode,
        candidates=quota_candidate_contexts,
        quota_realization=cfg.quota_realization,
        recent_modes=recent_modes,
        effective_window_k=effective_k,
        mode_credit=mode_credit,
        mode_credit_cap=cfg.mode_credit_cap,
        has_mode_information=have_modes,
        use_supervisor_ranks=use_supervisor_ranks,
        global_priority_used=global_priority_used,
        productive_wall_momentum_guard=cfg.productive_wall_momentum_guard,
        productive_wall_target_slots=productive_wall_target_slots,
        productive_wall_min_exploit_credit=(
            cfg.productive_wall_momentum_min_exploit_credit
        ),
        repeated_support_probe_enabled=cfg.realize_repeated_support_probe,
        low_cost_near_miss_rescue_floor=cfg.low_cost_near_miss_rescue_floor,
        evidence_allows_near_miss_rescue=evidence_allows_near_miss_rescue,
        marginal_diversity_override=cfg.marginal_diversity_override,
    ))
    quotas_raw = quota_result.raw_quotas
    quotas = dict(quota_result.final_quotas)
    redist_log = list(quota_result.redistribution_log)
    mode_credit_before = quota_result.mode_credit_before
    mode_credit_after_quota = quota_result.mode_credit_after_quota
    forced_probe = quota_result.forced_repeated_support_probe
    forced_near_miss_rescue = quota_result.forced_near_miss_rescue_floor

    diversity_policy = resolve_batch_diversity_policy(
        state_label=evidence.state_label,
        category_b_enabled=cat_b_on,
        available_slots=cfg.available_slots,
        default_max_per_family=cfg.diversity_max_per_family,
        duplicate_fraction=evidence.duplicate_fraction,
        recent_started_families=recent_started_families,
        duplicate_pressure_fraction=cfg.duplicate_pressure_fraction,
        family_share_cap=cfg.family_share_cap_when_duplicate,
        family_window_k=cfg.dispatch_family_window_k,
    )
    diversity_cap = diversity_policy.max_per_family
    overrepresented_families = set(
        diversity_policy.overrepresented_families
    )
    high_cost_batch_hard_caps = _high_cost_batch_hard_caps_for_cfg(
        evidence,
        cfg,
    )
    backlog_saturated_route_deferred_ids = {
        candidate.candidate_id
        for candidate in route_deferred_feasible
        if _backlog_saturated_route_deferred_generator(
            candidate,
            evidence,
            cfg,
            route_deferred_ids,
        )
    }
    route_backfill_candidates = [
        candidate
        for candidate in route_deferred_feasible
        if candidate.candidate_id
        not in backlog_saturated_route_deferred_ids
    ]
    cross_family_escape_candidates = [
        candidate
        for candidate in candidates
        if candidate.feasibility.all_ok()
        and candidate.candidate_id.startswith(
            "evidence_fallback_cross_family"
        )
        and not _backlog_saturated_route_deferred_generator(
            candidate,
            evidence,
            cfg,
            route_deferred_ids,
        )
    ]
    ranking_candidate_ids = {
        candidate.candidate_id
        for population in (
            feasible,
            deferred_feasible,
            route_backfill_candidates,
            cross_family_escape_candidates,
        )
        for candidate in population
    }
    ranking_context_by_id = _build_ranking_candidate_contexts(
        [
            candidate
            for candidate in candidates
            if candidate.candidate_id in ranking_candidate_ids
        ],
        policy.candidate_modes,
        sup_decs,
        evidence,
        overrepresented_families,
        family_cost_penalties,
    )

    def ranking_contexts_for(
        population: list[ActionCandidate],
    ) -> tuple[RankingCandidateContext, ...]:
        return tuple(
            ranking_context_by_id[candidate.candidate_id]
            for candidate in population
        )

    ranking_result = rank_candidates(SelectionRankingRequest(
        eligible_candidates=ranking_contexts_for(feasible),
        cost_deferred_candidates=ranking_contexts_for(deferred_feasible),
        route_deferred_candidates=ranking_contexts_for(
            route_backfill_candidates
        ),
        cross_family_escape_candidates=ranking_contexts_for(
            cross_family_escape_candidates
        ),
        quotas=quotas,
        available_slots=cfg.available_slots,
        diversity_policy=diversity_policy,
        high_cost_batch_hard_caps=high_cost_batch_hard_caps,
        has_mode_information=have_modes,
        global_priority_used=global_priority_used,
        top_n_rejected_per_mode=cfg.top_n_rejected_per_mode_to_log,
        cross_family_escape_floor_enabled=(
            cfg.realize_cross_family_escape_floor
        ),
    ))
    selections = list(ranking_result.selected_candidates)
    selection_mode = dict(ranking_result.selection_modes)
    global_priority_backfill_ids = list(
        ranking_result.global_priority_backfill_ids
    )
    forced_cross_family_escape = (
        ranking_result.forced_cross_family_escape_floor or {}
    )
    redist_log.extend(ranking_result.audit_log)

    emission_result = emit_selection_decisions(SelectionEmissionRequest(
        tick_id=tick_id,
        candidates=candidates,
        selected_candidates=selections,
        selection_modes=selection_mode,
        rejected_per_mode=ranking_result.rejected_per_mode,
        supervisor_decisions=sup_decs,
        quotas=quotas,
        source=source,
        fallback_used=fallback_reason is not None,
        global_priority_used=global_priority_used,
        evidence_snapshot=PlannedEvidenceSnapshot(
            completed_children=evidence.completed_children,
            run_su_count=evidence.run_su_count,
            state_label=evidence.state_label,
        ),
        capacity_blocked_families=frozenset(
            capacity_blocked_families
        ),
        capacity_block_reasons=capacity_block_reasons,
        high_cost_batch_hard_caps=high_cost_batch_hard_caps,
        route_deferred_candidate_ids=frozenset(route_deferred_ids),
        cost_deferred_families=frozenset(deferred_cost_families),
    ))
    launches = list(emission_result.launch_decisions)
    rank1_not_launched = list(emission_result.rank1_not_launched)
    global_priority_not_launched = list(
        emission_result.global_priority_not_launched
    )
    debug = {
        "source": "supervisor_global_priority" if global_priority_used else source,
        "quota_realization": (
            "global_priority" if global_priority_used else cfg.quota_realization
        ),
        "global_priority_used": global_priority_used,
        "global_priority_order": [
            d.candidate_id for d in sorted(
                sup_decs.values(),
                key=lambda d: (
                    int(d.global_rank or 10**6), d.candidate_id,
                ),
            )
        ] if global_priority_used else [],
        "global_priority_backfill_ids": global_priority_backfill_ids,
        "raw_mixture": raw_mixture,
        "clamped_mixture": mixture,
        "clamp_log": clamp_log,
        "category_b_enabled": cat_b_on,
        "category_b_reasons": gate_reasons,
        "diversity_cap_applied": diversity_cap,
        "high_cost_batch_hard_caps": dict(sorted(high_cost_batch_hard_caps.items())),
        "overrepresented_families": sorted(overrepresented_families),
        "family_cost_penalties": dict(sorted(family_cost_penalties.items())),
        # Legacy keys stay empty so downstream readers do not interpret high-cost
        # slot pressure as a hard Selector rejection. The live signal is the
        # capacity_pressure_* pair below.
        "capacity_blocked_families": [],
        "capacity_block_reasons": {},
        "capacity_pressure_families": sorted(capacity_blocked_families),
        "capacity_pressure_reasons": dict(sorted(capacity_block_reasons.items())),
        "n_capacity_blocked_candidates": 0,
        "n_capacity_pressure_candidates": len(capacity_pressure_feasible),
        "capacity_deferred_candidate_ids": [c.candidate_id for c in capacity_deferred_feasible],
        "rank1_not_launched": rank1_not_launched,
        "global_priority_not_launched": global_priority_not_launched,
        "cost_deferred_families": (
            [] if global_priority_used else sorted(deferred_cost_families)
        ),
        "cost_advisory_families": (
            sorted(deferred_cost_families) if global_priority_used else []
        ),
        "n_cost_deferred_candidates": len(deferred_feasible),
        "route_cost_penalties": dict(sorted(route_cost_penalties.items())),
        "route_deferred_candidate_ids": (
            [] if global_priority_used else sorted(route_deferred_ids)
        ),
        "route_advisory_candidate_ids": (
            sorted(route_deferred_ids) if global_priority_used else []
        ),
        "n_route_deferred_candidates": len(route_deferred_feasible),
        "backlog_saturated_route_deferred_candidate_ids": sorted(backlog_saturated_route_deferred_ids),
        "forced_repeated_support_probe": forced_probe or {},
        "forced_near_miss_rescue_floor": forced_near_miss_rescue or {},
        "forced_cross_family_escape_floor": forced_cross_family_escape or {},
        "mode_window_k_configured": cfg.mode_window_k,
        "mode_window_k_effective": effective_k,
        "mode_window": quota_result.mode_window,
        "mode_credit_before": mode_credit_before,
        "mode_credit_after_quota": mode_credit_after_quota or {},
        "mode_credit_cap": cfg.mode_credit_cap,
        "quotas_raw": quotas_raw,
        "quotas_final": quotas,
        "redistribute_log": redist_log,
        "n_feasible": sum(1 for c in candidates if c.feasibility.all_ok()),
        "n_total": len(candidates),
        "n_selected": len(selections),
        "selected_candidate_ids": [c.candidate_id for c in selections],
        "launch_modes": emission_result.launch_modes,
    }
    return launches, debug
