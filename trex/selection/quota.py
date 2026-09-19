"""Deterministic mode-quota realization and evidence-backed repairs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from ..fallback import largest_remainder, redistribute_empty_modes
from ..schemas import EvidenceSummary


MODES = ("exploit", "rescue", "explore")


class ModeWindowSettings(Protocol):
    """Configuration fields used to adapt the deterministic mode window."""

    mode_window_k: int
    adaptive_mode_window_k: bool
    mode_window_k_urgent: int
    mode_window_k_stalled: int
    mode_window_k_duplicate: int


@dataclass(frozen=True)
class QuotaCandidateContext:
    """Quota-repair facts for one already-admitted candidate.

    ``priority`` is the candidate's zero-based position under the Selector's
    deterministic tiebreak. The booleans record evidence-derived eligibility;
    this module only applies quota changes and records their audit trail.
    """

    candidate_id: str
    method_family: str
    mode: str | None
    parent_result_id: str | None
    priority: int
    effective_cost_rank: int
    repeated_support_probe_eligible: bool
    low_cost_near_miss_rescue_eligible: bool
    dry_duplicate_exact_replay: bool
    material_diversity_candidate: bool


@dataclass(frozen=True)
class ModeQuotaRequest:
    """Explicit inputs for quota realization and ordered quota repairs."""

    mixture: Mapping[str, float]
    available_slots: int
    feasibility_by_mode: Mapping[str, bool]
    candidates: Sequence[QuotaCandidateContext]
    quota_realization: str
    recent_modes: Sequence[str] | None
    effective_window_k: int
    mode_credit: dict[str, float] | None
    mode_credit_cap: float
    has_mode_information: bool
    use_supervisor_ranks: bool
    global_priority_used: bool
    productive_wall_momentum_guard: bool
    productive_wall_target_slots: int
    productive_wall_min_exploit_credit: float
    repeated_support_probe_enabled: bool
    low_cost_near_miss_rescue_floor: bool
    evidence_allows_near_miss_rescue: bool
    marginal_diversity_override: bool


@dataclass(frozen=True)
class ModeQuotaResult:
    """Realized quotas, persistent credit, and repair audit details."""

    raw_quotas: dict[str, int]
    final_quotas: dict[str, int]
    redistribution_log: tuple[str, ...]
    mode_credit_before: dict[str, float]
    mode_credit_after_quota: dict[str, float] | None
    forced_repeated_support_probe: dict[str, Any] | None
    forced_near_miss_rescue_floor: dict[str, Any] | None
    mode_window: dict[str, Any]


def _windowed_quotas(
    mixture: dict[str, float],
    n_slots: int,
    recent_modes: list[str],
    k: int,
    feasible_modes: set[str],
) -> dict[str, int]:
    """Allocate slots by largest deficit over a recent realized-mode window."""

    quotas = {mode: 0 for mode in MODES}
    if n_slots <= 0:
        return quotas
    pickable = [mode for mode in MODES if mode in feasible_modes] or list(MODES)
    window = list(recent_modes)[-k:]

    def ticks_since_last(mode: str) -> int:
        for index in range(len(window) - 1, -1, -1):
            if window[index] == mode:
                return len(window) - index
        return len(window) + 1

    n_prior = len(recent_modes)
    for slot_index in range(n_slots):
        denominator = len(window) or 1
        actual = {
            mode: window.count(mode) / denominator
            for mode in MODES
        }
        deficit = {
            mode: mixture.get(mode, 0.0) - actual[mode]
            for mode in pickable
        }
        rotation = (n_prior + slot_index) % len(MODES)
        selected_mode = max(
            pickable,
            key=lambda mode: (
                round(deficit[mode], 6),
                ticks_since_last(mode),
                mixture.get(mode, 0.0),
                -((MODES.index(mode) + rotation) % len(MODES)),
            ),
        )
        quotas[selected_mode] += 1
        window.append(selected_mode)
    return quotas


def _stochastic_quotas(
    mixture: dict[str, float],
    n_slots: int,
    feasible_modes: set[str],
    seed: int,
) -> dict[str, int]:
    """Sample a seeded probability-matching allocation for ablations."""

    pickable = [mode for mode in MODES if mode in feasible_modes] or list(MODES)
    weights = [max(0.0, mixture.get(mode, 0.0)) for mode in pickable]
    if sum(weights) <= 0:
        weights = [1.0] * len(pickable)
    rng = random.Random(seed)
    quotas = {mode: 0 for mode in MODES}
    for _ in range(max(0, n_slots)):
        quotas[rng.choices(pickable, weights=weights, k=1)[0]] += 1
    return quotas


def _clean_mode_credit(
    mode_credit: dict[str, float] | None,
    cap: float,
) -> dict[str, float]:
    cleaned = {mode: 0.0 for mode in MODES}
    if isinstance(mode_credit, dict):
        for mode in cleaned:
            try:
                cleaned[mode] = float(mode_credit.get(mode, 0.0) or 0.0)
            except (TypeError, ValueError):
                cleaned[mode] = 0.0
    cap = max(0.0, float(cap or 0.0))
    if cap > 0.0:
        cleaned = {
            mode: max(-cap, min(cap, value))
            for mode, value in cleaned.items()
        }
    return cleaned


def _fractional_carry_quotas(
    mixture: dict[str, float],
    n_slots: int,
    feasible_modes: set[str],
    mode_credit: dict[str, float] | None = None,
    *,
    cap: float = 3.0,
) -> tuple[dict[str, int], dict[str, float]]:
    """Realize a mixture with persistent, started-dispatch-corrected credit."""

    quotas = {mode: 0 for mode in MODES}
    if n_slots <= 0:
        return quotas, _clean_mode_credit(mode_credit, cap)
    pickable = [mode for mode in MODES if mode in feasible_modes] or list(MODES)
    credit = _clean_mode_credit(mode_credit, cap)
    for mode in MODES:
        credit[mode] += max(0.0, float(mixture.get(mode, 0.0) or 0.0)) * n_slots
    credit = _clean_mode_credit(credit, cap)
    for _ in range(max(0, n_slots)):
        selected_mode = max(
            pickable,
            key=lambda mode: (
                round(credit.get(mode, 0.0), 6),
                mixture.get(mode, 0.0),
                -MODES.index(mode),
            ),
        )
        quotas[selected_mode] += 1
        credit[selected_mode] = credit.get(selected_mode, 0.0) - 1.0
    return quotas, _clean_mode_credit(credit, cap)


def effective_mode_window_k(
    evidence: EvidenceSummary,
    settings: ModeWindowSettings,
) -> int:
    """Return the evidence-adapted deterministic realization window."""

    base = max(1, int(settings.mode_window_k))
    if not settings.adaptive_mode_window_k:
        return base
    state = evidence.state_label
    if state in ("strict_duplicate_collapse", "deep_stall"):
        return max(1, min(base, int(settings.mode_window_k_urgent)))
    if state in ("stalled", "rescue_rich"):
        return max(1, min(base, int(settings.mode_window_k_stalled)))
    if state == "productive_duplicate":
        return max(1, min(base, int(settings.mode_window_k_duplicate)))
    return base


def _mode_window_debug(
    mixture: dict[str, float],
    recent_modes: list[str],
    k: int,
) -> dict[str, Any]:
    window = list(recent_modes)[-max(1, k):]
    denominator = len(window) or 1
    counts = {mode: window.count(mode) for mode in MODES}
    actual = {mode: counts[mode] / denominator for mode in counts}
    deficit = {
        mode: float(mixture.get(mode, 0.0) or 0.0) - actual[mode]
        for mode in counts
    }
    return {
        "window_k": max(1, k),
        "last_modes": window,
        "counts": counts,
        "actual_share": actual,
        "deficit": deficit,
    }


def _realize_quotas(
    method: str,
    mixture: dict[str, float],
    n_slots: int,
    recent_modes: list[str],
    k: int,
    feasible_modes: set[str],
) -> dict[str, int]:
    """Turn a mode distribution into discrete slot quotas."""

    if method == "stochastic":
        return _stochastic_quotas(
            mixture,
            n_slots,
            feasible_modes,
            seed=len(recent_modes),
        )
    if method == "largest_remainder":
        feasible_mixture = {
            mode: max(0.0, mixture.get(mode, 0.0))
            for mode in MODES
            if mode in feasible_modes
        }
        if not feasible_mixture:
            return _windowed_quotas(
                mixture,
                n_slots,
                recent_modes,
                k,
                feasible_modes,
            )
        total = sum(feasible_mixture.values())
        normalized = (
            {
                mode: value / total
                for mode, value in feasible_mixture.items()
            }
            if total > 0
            else {
                mode: 1.0 / len(feasible_mixture)
                for mode in feasible_mixture
            }
        )
        return largest_remainder(normalized, n_slots)
    return _windowed_quotas(
        mixture,
        n_slots,
        recent_modes,
        k,
        feasible_modes,
    )


def _quota_donor(
    quotas: Mapping[str, int],
    mixture: Mapping[str, float],
    *,
    excluded_mode: str,
) -> str | None:
    donors = [
        mode
        for mode, quota in quotas.items()
        if mode != excluded_mode and quota > 0
    ]
    if not donors:
        return None
    return sorted(
        donors,
        key=lambda mode: (
            -quotas.get(mode, 0),
            -mixture.get(mode, 0.0),
            mode,
        ),
    )[0]


def _realize_base_quotas(
    request: ModeQuotaRequest,
    mixture: dict[str, float],
    feasible_modes: set[str],
    mode_credit_before: dict[str, float],
) -> tuple[dict[str, int], dict[str, float] | None]:
    """Apply the configured distribution-to-slots realization strategy."""

    if request.quota_realization == "fractional_carry":
        return _fractional_carry_quotas(
            mixture,
            request.available_slots,
            feasible_modes,
            mode_credit_before,
            cap=request.mode_credit_cap,
        )
    if request.recent_modes is not None:
        return (
            _realize_quotas(
                request.quota_realization,
                mixture,
                request.available_slots,
                list(request.recent_modes),
                request.effective_window_k,
                feasible_modes,
            ),
            None,
        )
    return largest_remainder(mixture, request.available_slots), None


def _apply_productive_wall_momentum(
    request: ModeQuotaRequest,
    quotas: dict[str, int],
    mixture: Mapping[str, float],
    redistribution_log: list[str],
    mode_credit_before: Mapping[str, float],
) -> None:
    """Preserve productive exploit throughput before narrower repairs."""

    if (
        not request.productive_wall_momentum_guard
        or not request.has_mode_information
        or request.global_priority_used
    ):
        return

    target_slots = request.productive_wall_target_slots
    exploit_credit = float(mode_credit_before.get("exploit", 0.0) or 0.0)
    if (
        target_slots > 0
        and exploit_credit < request.productive_wall_min_exploit_credit
    ):
        redistribution_log.append(
            "productive_wall_momentum_guard_skipped:"
            f"exploit_credit={exploit_credit:.3f}<"
            f"{request.productive_wall_min_exploit_credit:.3f}"
        )
        target_slots = 0

    while target_slots > quotas.get("exploit", 0):
        donor_mode = _quota_donor(
            {
                mode: quotas.get(mode, 0)
                for mode in ("rescue", "explore")
            },
            mixture,
            excluded_mode="exploit",
        )
        if donor_mode is None:
            break
        quotas[donor_mode] = max(0, quotas.get(donor_mode, 0) - 1)
        quotas["exploit"] = quotas.get("exploit", 0) + 1
        redistribution_log.append(
            "productive_wall_momentum_guard:"
            f"{donor_mode}->exploit:target={target_slots}"
        )


def _apply_repeated_support_probe(
    request: ModeQuotaRequest,
    quotas: dict[str, int],
    mixture: Mapping[str, float],
    redistribution_log: list[str],
    ordered_candidates: Sequence[QuotaCandidateContext],
) -> dict[str, Any] | None:
    """Spend one quota on a repeatedly supported but under-started route."""

    if (
        not request.repeated_support_probe_enabled
        or not request.use_supervisor_ranks
        or request.global_priority_used
    ):
        return None
    probe_candidates = [
        candidate
        for candidate in ordered_candidates
        if candidate.repeated_support_probe_eligible
    ]
    if not probe_candidates:
        return None

    candidate = probe_candidates[0]
    candidate_mode = candidate.mode
    if candidate_mode not in quotas or quotas.get(candidate_mode, 0) > 0:
        return None
    donor_mode = _quota_donor(
        quotas,
        mixture,
        excluded_mode=str(candidate_mode),
    )
    if donor_mode is None:
        return None

    quotas[donor_mode] = max(0, quotas.get(donor_mode, 0) - 1)
    quotas[str(candidate_mode)] = quotas.get(str(candidate_mode), 0) + 1
    redistribution_log.append(
        "forced_repeated_support_probe:"
        f"{candidate.method_family}:{donor_mode}->{candidate_mode}"
    )
    return {
        "candidate_id": candidate.candidate_id,
        "family": candidate.method_family,
        "mode": candidate_mode,
        "donor_mode": donor_mode,
    }


def _apply_near_miss_rescue_floor(
    request: ModeQuotaRequest,
    quotas: dict[str, int],
    mixture: Mapping[str, float],
    redistribution_log: list[str],
    ordered_candidates: Sequence[QuotaCandidateContext],
) -> dict[str, Any] | None:
    """Promote one low-cost parent-bound rescue over a costlier donor."""

    if (
        not request.low_cost_near_miss_rescue_floor
        or request.global_priority_used
        or request.available_slots != 1
        or quotas.get("rescue", 0) > 0
        or not request.use_supervisor_ranks
        or not request.evidence_allows_near_miss_rescue
    ):
        return None

    rescue_candidates = [
        candidate
        for candidate in ordered_candidates
        if candidate.low_cost_near_miss_rescue_eligible
    ]
    donor_mode = _quota_donor(
        quotas,
        mixture,
        excluded_mode="rescue",
    )
    donor_candidates = [
        candidate
        for candidate in ordered_candidates
        if candidate.mode == donor_mode
    ]
    donor_top = donor_candidates[0] if donor_candidates else None
    if (
        not rescue_candidates
        or donor_mode is None
        or donor_top is None
        or donor_top.effective_cost_rank <= 0
    ):
        return None

    pivot = rescue_candidates[0]
    quotas[donor_mode] = max(0, quotas.get(donor_mode, 0) - 1)
    quotas["rescue"] = quotas.get("rescue", 0) + 1
    redistribution_log.append(
        "low_cost_near_miss_rescue_floor:"
        f"{donor_mode}->rescue:{pivot.method_family}:"
        f"parent={pivot.parent_result_id}"
    )
    return {
        "candidate_id": pivot.candidate_id,
        "family": pivot.method_family,
        "parent_result_id": pivot.parent_result_id,
        "donor_mode": donor_mode,
        "donor_candidate_id": donor_top.candidate_id,
    }


def _apply_marginal_diversity_override(
    request: ModeQuotaRequest,
    quotas: dict[str, int],
    redistribution_log: list[str],
    ordered_candidates: Sequence[QuotaCandidateContext],
) -> None:
    """Replace one dry exact replay with the best material-diversity route."""

    if (
        not request.marginal_diversity_override
        or request.global_priority_used
        or request.available_slots != 1
        or quotas.get("exploit", 0) <= 0
        or quotas.get("rescue", 0) > 0
        or quotas.get("explore", 0) > 0
        or not request.use_supervisor_ranks
    ):
        return

    exploit_candidates = [
        candidate
        for candidate in ordered_candidates
        if candidate.mode == "exploit"
    ]
    if (
        not exploit_candidates
        or not exploit_candidates[0].dry_duplicate_exact_replay
    ):
        return
    diversity_candidates = [
        candidate
        for candidate in ordered_candidates
        if candidate.mode in {"rescue", "explore"}
        and candidate.material_diversity_candidate
    ]
    if not diversity_candidates:
        return

    pivot = diversity_candidates[0]
    pivot_mode = pivot.mode or "rescue"
    quotas["exploit"] = max(0, quotas.get("exploit", 0) - 1)
    quotas[pivot_mode] = quotas.get(pivot_mode, 0) + 1
    redistribution_log.append(
        "marginal_diversity_override:"
        f"{exploit_candidates[0].method_family}->"
        f"{pivot.method_family}:{pivot_mode}"
    )


def realize_mode_quotas(request: ModeQuotaRequest) -> ModeQuotaResult:
    """Realize quotas, then apply repairs in their documented order."""

    mixture = dict(request.mixture)
    feasible_modes = {
        mode
        for mode, is_feasible in request.feasibility_by_mode.items()
        if is_feasible
    }
    mode_credit_before = _clean_mode_credit(
        request.mode_credit,
        request.mode_credit_cap,
    )
    raw_quotas, mode_credit_after = _realize_base_quotas(
        request,
        mixture,
        feasible_modes,
        mode_credit_before,
    )
    quotas, redistribution_log = redistribute_empty_modes(
        raw_quotas,
        dict(request.feasibility_by_mode),
        mixture,
    )
    ordered_candidates = sorted(
        request.candidates,
        key=lambda candidate: candidate.priority,
    )

    _apply_productive_wall_momentum(
        request,
        quotas,
        mixture,
        redistribution_log,
        mode_credit_before,
    )
    forced_probe = _apply_repeated_support_probe(
        request,
        quotas,
        mixture,
        redistribution_log,
        ordered_candidates,
    )
    forced_near_miss = _apply_near_miss_rescue_floor(
        request,
        quotas,
        mixture,
        redistribution_log,
        ordered_candidates,
    )
    _apply_marginal_diversity_override(
        request,
        quotas,
        redistribution_log,
        ordered_candidates,
    )

    return ModeQuotaResult(
        raw_quotas=raw_quotas,
        final_quotas=quotas,
        redistribution_log=tuple(redistribution_log),
        mode_credit_before=mode_credit_before,
        mode_credit_after_quota=mode_credit_after,
        forced_repeated_support_probe=forced_probe,
        forced_near_miss_rescue_floor=forced_near_miss,
        mode_window=_mode_window_debug(
            mixture,
            list(request.recent_modes or []),
            request.effective_window_k,
        ),
    )
