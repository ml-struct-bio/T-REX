"""Feasibility, capacity, and evidence-aware candidate admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..schemas import ActionCandidate


MODES = ("exploit", "rescue", "explore")


@dataclass(frozen=True)
class CandidateAdmissionRequest:
    """Explicit inputs to deterministic candidate admission."""

    candidates: Sequence[ActionCandidate]
    candidate_modes: Mapping[str, str]
    has_mode_information: bool
    global_priority_used: bool
    category_b_enabled: bool
    family_cost_penalties: Mapping[str, int]
    capacity_block_reasons: Mapping[str, str]
    candidate_route_penalties: Mapping[str, int]
    defer_penalty: int


@dataclass(frozen=True)
class CandidateAdmissionResult:
    """Primary and deferred populations plus mode feasibility."""

    eligible_candidates: tuple[ActionCandidate, ...]
    feasibility_by_mode: dict[str, bool]
    capacity_pressure_candidates: tuple[ActionCandidate, ...]
    capacity_deferred_candidates: tuple[ActionCandidate, ...]
    cost_deferred_families: frozenset[str]
    cost_deferred_candidates: tuple[ActionCandidate, ...]
    route_deferred_candidates: tuple[ActionCandidate, ...]
    route_deferred_candidate_ids: frozenset[str]
    route_cost_penalties: dict[str, int]


def admit_candidates(
    request: CandidateAdmissionRequest,
) -> CandidateAdmissionResult:
    """Apply hard feasibility and ordered capacity/cost/route deferrals."""

    eligible = [
        candidate
        for candidate in request.candidates
        if candidate.feasibility.all_ok()
    ]
    capacity_blocked_families = set(request.capacity_block_reasons)
    capacity_pressure_candidates = [
        candidate
        for candidate in eligible
        if candidate.method_family in capacity_blocked_families
    ]
    capacity_deferred_candidates: list[ActionCandidate] = []
    if capacity_blocked_families:
        primary = [
            candidate
            for candidate in eligible
            if candidate.method_family not in capacity_blocked_families
        ]
        capacity_deferred_candidates = [
            candidate
            for candidate in eligible
            if candidate.method_family in capacity_blocked_families
        ]
        if primary:
            eligible = primary
        else:
            capacity_deferred_candidates = []

    cost_deferred_families = {
        family
        for family, penalty in request.family_cost_penalties.items()
        if penalty >= request.defer_penalty
    }
    cost_deferred_candidates: list[ActionCandidate] = []
    if cost_deferred_families and not request.global_priority_used:
        primary = [
            candidate
            for candidate in eligible
            if candidate.method_family not in cost_deferred_families
        ]
        cost_deferred_candidates = [
            candidate
            for candidate in eligible
            if candidate.method_family in cost_deferred_families
        ]
        if request.category_b_enabled:
            retained_explore = [
                candidate
                for candidate in cost_deferred_candidates
                if request.candidate_modes.get(candidate.candidate_id)
                == "explore"
            ]
            if retained_explore:
                primary.extend(retained_explore)
                cost_deferred_candidates = [
                    candidate
                    for candidate in cost_deferred_candidates
                    if candidate not in retained_explore
                ]
        if primary:
            eligible = primary
        else:
            cost_deferred_candidates = []

    route_deferred_candidates: list[ActionCandidate] = []
    route_deferred_ids = {
        candidate.candidate_id
        for candidate in eligible
        if request.candidate_route_penalties.get(candidate.candidate_id, 0)
        >= request.defer_penalty
    }
    route_cost_penalties = {
        candidate_id: request.defer_penalty
        for candidate_id in route_deferred_ids
    }
    if route_deferred_ids and not request.global_priority_used:
        primary = []
        for candidate in eligible:
            if candidate.candidate_id not in route_deferred_ids:
                primary.append(candidate)
                continue
            mode = request.candidate_modes.get(candidate.candidate_id)
            mode_peers = [
                other
                for other in eligible
                if other is not candidate
                and other.candidate_id not in route_deferred_ids
                and (
                    not request.has_mode_information
                    or request.candidate_modes.get(other.candidate_id) == mode
                )
            ]
            if (
                request.category_b_enabled and mode == "explore"
            ) or not mode_peers:
                primary.append(candidate)
            else:
                route_deferred_candidates.append(candidate)
        if primary:
            eligible = primary
            route_deferred_ids = {
                candidate.candidate_id
                for candidate in route_deferred_candidates
            }
            route_cost_penalties = {
                candidate_id: request.defer_penalty
                for candidate_id in route_deferred_ids
            }
        else:
            route_deferred_candidates = []
            route_deferred_ids = set()
            route_cost_penalties = {}

    if request.has_mode_information:
        feasibility_by_mode = {
            mode: any(
                request.candidate_modes.get(candidate.candidate_id) == mode
                for candidate in eligible
            )
            for mode in MODES
        }
        if not any(feasibility_by_mode.values()) and eligible:
            feasibility_by_mode = {mode: True for mode in MODES}
    else:
        feasibility_by_mode = {
            mode: bool(eligible)
            for mode in MODES
        }

    return CandidateAdmissionResult(
        eligible_candidates=tuple(eligible),
        feasibility_by_mode=feasibility_by_mode,
        capacity_pressure_candidates=tuple(capacity_pressure_candidates),
        capacity_deferred_candidates=tuple(capacity_deferred_candidates),
        cost_deferred_families=frozenset(cost_deferred_families),
        cost_deferred_candidates=tuple(cost_deferred_candidates),
        route_deferred_candidates=tuple(route_deferred_candidates),
        route_deferred_candidate_ids=frozenset(route_deferred_ids),
        route_cost_penalties=route_cost_penalties,
    )
