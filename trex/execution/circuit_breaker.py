"""Evidence-driven backend-family circuit-breaker policy.

This module owns the run-level decision to disable an unproductive backend
family.  It is intentionally independent of controller loop mutation: callers
apply the returned family and pending-queue changes to their runtime state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..archive import Archive
from ..capability_registry import CapabilityRegistry
from ..schemas import ActionCandidate, ResultRecord


@dataclass(frozen=True)
class FamilyCircuitBreakerRequest:
    """Inputs required to evaluate all currently eligible backend families."""

    archive: Archive
    all_families: Sequence[str]
    unavailable_families: Sequence[str]
    already_blocked_families: frozenset[str]
    pending_candidate_ids: Sequence[str]
    backend_registry: CapabilityRegistry
    max_no_yield_timeouts: int
    minimum_gpu_hours: float
    excluded_families: frozenset[str] = frozenset({"structure_refilter"})


@dataclass(frozen=True)
class FamilyCircuitBreakerDependencies:
    """Injectable scientific policy and archive-index construction hooks."""

    index_actions_by_spawned_result: Callable[
        [list[ActionCandidate], list[ResultRecord]],
        dict[str, ActionCandidate],
    ]
    is_family_circuit_broken: Callable[..., bool]


@dataclass(frozen=True)
class FamilyCircuitBreakerResult:
    """Immutable state delta produced by one breaker evaluation."""

    newly_blocked_families: tuple[str, ...]
    blocked_families: frozenset[str]
    pending_candidate_ids: tuple[str, ...]


def family_circuit_broken(
    archive: Archive,
    family: str,
    *,
    max_timeouts: int,
    min_gpu_h: float,
    results: list[ResultRecord] | None = None,
    by_result_id: dict[str, ResultRecord] | None = None,
    spawning_actions: dict[str, ActionCandidate] | None = None,
) -> bool:
    """Return whether one family has enough evidence to be disabled.

    A family is disabled only after enough completed/timeout GPU-hours and
    repeated no-yield kills, with no strict-success or near-miss signal.
    Success is attributed through refilter lineage so productive diagnostic
    generators are not penalized for using a canonical scoring child.
    """

    from ..evidence_reducer import resolve_generating_family
    from ..success_criteria import is_near_miss, is_strict_success
    from ..tick.archive_snapshot import index_actions_by_spawned_result

    if results is None:
        results = list(archive.iter_records(ResultRecord))
    if by_result_id is None:
        by_result_id = {result.result_id: result for result in results}
    if spawning_actions is None:
        spawning_actions = index_actions_by_spawned_result(
            list(archive.iter_records(ActionCandidate)), results
        )

    timeout_count = 0
    strict_success_count = 0
    near_miss_count = 0
    gpu_hours = 0.0
    for result in results:
        if result.backend_family == family:
            gpu_hours += float(getattr(result, "gpu_h", 0.0) or 0.0)
            if result.exit_status == "timeout":
                timeout_count += 1

        generating_family = (result.bins or {}).get("refilter_source_family")
        if not generating_family:
            generating_family = resolve_generating_family(
                result,
                by_result_id=by_result_id,
                spawning_actions=spawning_actions,
            )
        if generating_family != family or result.exit_status != "ok":
            continue
        if is_strict_success(result.metrics):
            strict_success_count += 1
        elif is_near_miss(result.metrics):
            near_miss_count += 1

    return (
        gpu_hours >= min_gpu_h
        and strict_success_count == 0
        and near_miss_count == 0
        and timeout_count >= max_timeouts
    )


def evaluate_family_circuit_breakers(
    request: FamilyCircuitBreakerRequest,
    dependencies: FamilyCircuitBreakerDependencies,
) -> FamilyCircuitBreakerResult:
    """Evaluate eligible families once and return a controller state delta.

    Archive result and lineage indexes are built once per evaluation and shared
    across families.  A family is never disabled when doing so would leave no
    route-light generator available at the start of the evaluation.
    """

    breakable_families = sorted(
        set(request.all_families)
        - set(request.unavailable_families)
        - set(request.already_blocked_families)
        - set(request.excluded_families)
    )
    if not breakable_families:
        return FamilyCircuitBreakerResult(
            newly_blocked_families=(),
            blocked_families=request.already_blocked_families,
            pending_candidate_ids=tuple(request.pending_candidate_ids),
        )

    results = list(request.archive.iter_records(ResultRecord))
    results_by_id = {result.result_id: result for result in results}
    spawning_actions = dependencies.index_actions_by_spawned_result(
        list(request.archive.iter_records(ActionCandidate)), results
    )

    newly_blocked: list[str] = []
    for family in breakable_families:
        alternative_route_light_generators = {
            other_family
            for other_family in breakable_families
            if other_family != family
            and (capability := request.backend_registry.get(other_family)) is not None
            and getattr(capability, "role", "generator") == "generator"
            and not getattr(capability, "requires_parent_pdb", False)
        }
        if not alternative_route_light_generators:
            continue
        if dependencies.is_family_circuit_broken(
            request.archive,
            family,
            max_timeouts=request.max_no_yield_timeouts,
            min_gpu_h=request.minimum_gpu_hours,
            results=results,
            by_result_id=results_by_id,
            spawning_actions=spawning_actions,
        ):
            newly_blocked.append(family)

    blocked_families = request.already_blocked_families | frozenset(newly_blocked)
    if not newly_blocked or not request.pending_candidate_ids:
        pending_candidate_ids = tuple(request.pending_candidate_ids)
    else:
        family_by_candidate_id = {
            candidate.candidate_id: candidate.method_family
            for candidate in request.archive.iter_records(ActionCandidate)
        }
        pending_candidate_ids = tuple(
            candidate_id
            for candidate_id in request.pending_candidate_ids
            if family_by_candidate_id.get(candidate_id) not in blocked_families
        )

    return FamilyCircuitBreakerResult(
        newly_blocked_families=tuple(newly_blocked),
        blocked_families=blocked_families,
        pending_candidate_ids=pending_candidate_ids,
    )


__all__ = [
    "FamilyCircuitBreakerDependencies",
    "FamilyCircuitBreakerRequest",
    "FamilyCircuitBreakerResult",
    "evaluate_family_circuit_breakers",
    "family_circuit_broken",
]
