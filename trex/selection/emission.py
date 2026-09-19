"""Launch-decision emission and Supervisor non-launch diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..refilter_roles import infer_refilter_role, is_canonical_score_conversion
from ..schemas import ActionCandidate, CandidateDecision, LaunchDecision


@dataclass(frozen=True)
class PlannedEvidenceSnapshot:
    """Evidence fields recorded on every selected launch."""

    completed_children: int
    run_su_count: int
    state_label: str


@dataclass(frozen=True)
class SelectionEmissionRequest:
    """Explicit inputs for launch emission and non-launch explanations."""

    tick_id: str
    candidates: Sequence[ActionCandidate]
    selected_candidates: Sequence[ActionCandidate]
    selection_modes: Mapping[str, str]
    rejected_per_mode: Mapping[str, Sequence[ActionCandidate]]
    supervisor_decisions: Mapping[str, CandidateDecision]
    quotas: Mapping[str, int]
    source: str
    fallback_used: bool
    global_priority_used: bool
    evidence_snapshot: PlannedEvidenceSnapshot
    capacity_blocked_families: frozenset[str]
    capacity_block_reasons: Mapping[str, str]
    high_cost_batch_hard_caps: Mapping[str, int]
    route_deferred_candidate_ids: frozenset[str]
    cost_deferred_families: frozenset[str]


@dataclass(frozen=True)
class SelectionEmissionResult:
    """Launch/rejection records plus structured non-launch diagnostics."""

    launch_decisions: tuple[LaunchDecision, ...]
    rank1_not_launched: tuple[dict[str, Any], ...]
    global_priority_not_launched: tuple[dict[str, Any], ...]
    launch_modes: dict[str, str]


def _realized_mode(
    candidate: ActionCandidate,
    selection_modes: Mapping[str, str],
) -> str:
    if is_canonical_score_conversion(candidate):
        return "chain_refilter"
    return selection_modes.get(candidate.candidate_id, "exploit")


def _emit_selected_launches(
    request: SelectionEmissionRequest,
) -> list[LaunchDecision]:
    decisions: list[LaunchDecision] = []
    for index, candidate in enumerate(request.selected_candidates):
        supervisor_decision = request.supervisor_decisions.get(
            candidate.candidate_id
        )
        resource_class = (
            supervisor_decision.resource_class
            if supervisor_decision is not None
            else candidate.estimated_cost_class
        )
        realized_mode = _realized_mode(candidate, request.selection_modes)
        resource_details: dict[str, Any] = {
            "class": resource_class,
            "source": (
                "supervisor_global_priority"
                if request.global_priority_used
                else request.source
            ),
            "planned_completed_children": (
                request.evidence_snapshot.completed_children
            ),
            "planned_run_su_count": request.evidence_snapshot.run_su_count,
            "planned_state_label": request.evidence_snapshot.state_label,
            "mode": realized_mode,
        }
        refilter_role = infer_refilter_role(candidate)
        if refilter_role:
            resource_details["refilter_role"] = refilter_role
        decisions.append(LaunchDecision(
            launch_id=f"launch_{request.tick_id}_{index:03d}",
            tick_id=request.tick_id,
            candidate_id=candidate.candidate_id,
            status="launched",
            resource_class_concrete=resource_details,
            why=(
                supervisor_decision.what
                if supervisor_decision is not None
                else "fallback_selected_by_tiebreak"
            ),
            fallback=request.fallback_used,
        ))
    return decisions


def _emit_rejected_launches(
    request: SelectionEmissionRequest,
    selected_candidate_ids: set[str],
) -> list[LaunchDecision]:
    decisions: list[LaunchDecision] = []
    rejected_candidate_ids: set[str] = set()
    rejected_index = 0
    for mode, candidates in request.rejected_per_mode.items():
        for candidate in candidates:
            candidate_id = candidate.candidate_id
            if (
                candidate_id in selected_candidate_ids
                or candidate_id in rejected_candidate_ids
            ):
                continue
            resource_details: dict[str, Any] = {
                "class": candidate.estimated_cost_class,
                "source": request.source,
                "mode": mode,
            }
            refilter_role = infer_refilter_role(candidate)
            if refilter_role:
                resource_details["refilter_role"] = refilter_role
            decisions.append(LaunchDecision(
                launch_id=(
                    f"launch_{request.tick_id}_rej_{rejected_index:03d}"
                ),
                tick_id=request.tick_id,
                candidate_id=candidate_id,
                status="rejected",
                resource_class_concrete=resource_details,
                why=(
                    "not_selected_after_global_priority"
                    if request.global_priority_used
                    else (
                        f"not_selected_in_{mode}_quota"
                        f"({request.quotas.get(mode, 0)})"
                    )
                ),
                fallback=request.fallback_used,
            ))
            rejected_candidate_ids.add(candidate_id)
            rejected_index += 1
    return decisions


def _selected_family_count(
    request: SelectionEmissionRequest,
    family: str,
) -> int:
    return sum(
        1
        for candidate in request.selected_candidates
        if candidate.method_family == family
    )


def _hard_cap_reason(
    request: SelectionEmissionRequest,
    candidate: ActionCandidate,
) -> str | None:
    family = candidate.method_family
    if family not in request.high_cost_batch_hard_caps:
        return None
    cap = request.high_cost_batch_hard_caps[family]
    if _selected_family_count(request, family) < cap:
        return None
    return f"high_cost_batch_hard_cap:family={family}:cap={cap}"


def _rank1_non_launch_diagnostics(
    request: SelectionEmissionRequest,
    launched_ids: set[str],
    rejected_by_id: Mapping[str, LaunchDecision],
) -> list[dict[str, Any]]:
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in request.candidates
    }
    diagnostics: list[dict[str, Any]] = []
    for decision in request.supervisor_decisions.values():
        candidate_id = decision.candidate_id
        if decision.rank_in_mode != 1 or candidate_id in launched_ids:
            continue
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            reason = "missing_candidate"
            family = "unknown"
        elif candidate_id in rejected_by_id:
            reason = rejected_by_id[candidate_id].why or "rejected"
            family = candidate.method_family
        elif not candidate.feasibility.all_ok():
            reason = "infeasible:" + ",".join(
                candidate.feasibility.reasons or []
            )
            family = candidate.method_family
        elif candidate.method_family in request.capacity_blocked_families:
            reason = request.capacity_block_reasons.get(
                candidate.method_family,
                "capacity_pressure_high_cost_running_cap",
            )
            family = candidate.method_family
        elif (hard_cap_reason := _hard_cap_reason(request, candidate)):
            reason = hard_cap_reason
            family = candidate.method_family
        elif (
            not request.global_priority_used
            and candidate_id in request.route_deferred_candidate_ids
        ):
            reason = "route_deferred"
            family = candidate.method_family
        elif (
            not request.global_priority_used
            and candidate.method_family in request.cost_deferred_families
        ):
            reason = "cost_deferred"
            family = candidate.method_family
        else:
            reason = "not_selected_after_quota_or_tiebreak"
            family = candidate.method_family
        diagnostics.append({
            "candidate_id": candidate_id,
            "family": family,
            "mode": decision.mode,
            "reason": reason,
        })
    return diagnostics


def _global_priority_non_launch_diagnostics(
    request: SelectionEmissionRequest,
    launched_ids: set[str],
    rejected_by_id: Mapping[str, LaunchDecision],
) -> list[dict[str, Any]]:
    if not request.global_priority_used:
        return []
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in request.candidates
    }
    diagnostics: list[dict[str, Any]] = []
    ordered_decisions = sorted(
        request.supervisor_decisions.values(),
        key=lambda decision: (
            int(decision.global_rank or 10**6),
            decision.candidate_id,
        ),
    )
    for decision in ordered_decisions:
        candidate_id = decision.candidate_id
        if candidate_id in launched_ids:
            continue
        candidate = candidate_by_id.get(candidate_id)
        if candidate is None:
            reason = "missing_candidate"
            family = "unknown"
        elif not candidate.feasibility.all_ok():
            reason = "infeasible:" + ",".join(
                candidate.feasibility.reasons or []
            )
            family = candidate.method_family
        elif candidate.method_family in request.capacity_blocked_families:
            reason = request.capacity_block_reasons.get(
                candidate.method_family,
                "capacity_pressure_high_cost_running_cap",
            )
            family = candidate.method_family
        elif (hard_cap_reason := _hard_cap_reason(request, candidate)):
            reason = hard_cap_reason
            family = candidate.method_family
        elif candidate_id in rejected_by_id:
            reason = rejected_by_id[candidate_id].why or "rejected"
            family = candidate.method_family
        else:
            reason = "not_selected_after_global_priority"
            family = candidate.method_family
        diagnostics.append({
            "candidate_id": candidate_id,
            "family": family,
            "mode": decision.mode,
            "global_rank": int(decision.global_rank or 0),
            "reason": reason,
        })
    return diagnostics


def emit_selection_decisions(
    request: SelectionEmissionRequest,
) -> SelectionEmissionResult:
    """Emit stable launch records and explain supported candidates not run."""

    selected_ids = {
        candidate.candidate_id
        for candidate in request.selected_candidates
    }
    launch_decisions = _emit_selected_launches(request)
    launch_decisions.extend(_emit_rejected_launches(request, selected_ids))
    launched_ids = {
        decision.candidate_id
        for decision in launch_decisions
        if decision.status == "launched"
    }
    rejected_by_id = {
        decision.candidate_id: decision
        for decision in launch_decisions
        if decision.status == "rejected"
    }
    return SelectionEmissionResult(
        launch_decisions=tuple(launch_decisions),
        rank1_not_launched=tuple(_rank1_non_launch_diagnostics(
            request,
            launched_ids,
            rejected_by_id,
        )),
        global_priority_not_launched=tuple(
            _global_priority_non_launch_diagnostics(
                request,
                launched_ids,
                rejected_by_id,
            )
        ),
        launch_modes={
            candidate.candidate_id: _realized_mode(
                candidate,
                request.selection_modes,
            )
            for candidate in request.selected_candidates
        },
    )
