"""Post-selection hypothesis lifecycle evaluation phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..lifecycle import (
    LifecycleConfig,
    default_axis_thresholds,
    update_hypothesis,
)
from ..schemas import ActionCandidate, HypothesisCard, ResultRecord


@dataclass(frozen=True)
class LifecyclePhaseRequest:
    """Archive-derived lineage needed for hypothesis lifecycle updates."""

    hypotheses: Sequence[HypothesisCard]
    actions: Sequence[ActionCandidate]
    results: Sequence[ResultRecord]
    spawning_actions: Mapping[str, ActionCandidate]
    current_tick: int
    lifecycle_config: LifecycleConfig


@dataclass(frozen=True)
class LifecyclePhaseResult:
    """Hypothesis records to append and compact user-facing update summaries."""

    updated_hypotheses: tuple[HypothesisCard, ...]
    update_summaries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PrePlannerLifecycleRequest:
    """Inputs for retiring stale cards before constructing the Planner prompt."""

    active_hypotheses: Sequence[HypothesisCard]
    all_hypotheses: Sequence[HypothesisCard]
    actions: Sequence[ActionCandidate]
    results: Sequence[ResultRecord]
    current_tick: int
    lifecycle_config: LifecycleConfig
    evidence_only: bool = False


@dataclass(frozen=True)
class PrePlannerLifecycleResult:
    """Updated in-memory hypothesis views and records for the archive owner."""

    active_hypotheses: tuple[HypothesisCard, ...]
    all_hypotheses: tuple[HypothesisCard, ...]
    retired_hypotheses: tuple[HypothesisCard, ...]


def retire_expired_hypotheses(
    request: PrePlannerLifecycleRequest,
) -> PrePlannerLifecycleResult:
    """Retire descendant-free expired cards without writing to the archive."""

    if not request.active_hypotheses or request.evidence_only:
        return PrePlannerLifecycleResult(
            active_hypotheses=tuple(request.active_hypotheses),
            all_hypotheses=tuple(request.all_hypotheses),
            retired_hypotheses=(),
        )

    candidate_hypotheses = {
        action.candidate_id: list(action.hypothesis_ids)
        for action in request.actions
        if action.hypothesis_ids
    }
    hypotheses_with_descendants: set[str] = set()
    for result in request.results:
        for parent_id in result.parent_ids or []:
            hypotheses_with_descendants.update(
                candidate_hypotheses.get(parent_id, [])
            )

    axis_thresholds = default_axis_thresholds()
    retained_hypotheses: list[HypothesisCard] = []
    retired_hypotheses: list[HypothesisCard] = []
    for hypothesis in request.active_hypotheses:
        expired = (
            request.current_tick - hypothesis.tick_created
            >= hypothesis.ttl_ticks
        )
        if expired and hypothesis.hypothesis_id not in hypotheses_with_descendants:
            updated = update_hypothesis(
                hypothesis,
                healthy_descendants=[],
                current_tick=request.current_tick,
                axis_thresholds=axis_thresholds,
                cfg=request.lifecycle_config,
            )
            if updated.status == "retired" and updated.status != hypothesis.status:
                retired_hypotheses.append(updated)
                continue
        retained_hypotheses.append(hypothesis)

    if not retired_hypotheses:
        return PrePlannerLifecycleResult(
            active_hypotheses=tuple(retained_hypotheses),
            all_hypotheses=tuple(request.all_hypotheses),
            retired_hypotheses=(),
        )

    latest_hypotheses = {
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in request.all_hypotheses
    }
    latest_hypotheses.update({
        hypothesis.hypothesis_id: hypothesis
        for hypothesis in retired_hypotheses
    })
    return PrePlannerLifecycleResult(
        active_hypotheses=tuple(retained_hypotheses),
        all_hypotheses=tuple(latest_hypotheses.values()),
        retired_hypotheses=tuple(retired_hypotheses),
    )


def update_lifecycle_phase(
    request: LifecyclePhaseRequest,
) -> LifecyclePhaseResult:
    """Evaluate descendants against each hypothesis's concrete baseline."""

    axis_thresholds = default_axis_thresholds()
    actions_by_candidate_id = {
        action.candidate_id: action for action in request.actions
    }
    results_by_id = {result.result_id: result for result in request.results}

    descendants_by_hypothesis_id: dict[
        str, list[tuple[ResultRecord, ResultRecord]]
    ] = {}
    for result in request.results:
        for parent_id in result.parent_ids:
            candidate = actions_by_candidate_id.get(parent_id)
            if candidate is None:
                continue
            baseline_result_id = candidate.baseline_result_id
            if (
                not baseline_result_id
                and candidate.candidate_id.startswith("chain_")
                and candidate.parent_result_id
            ):
                upstream_action = request.spawning_actions.get(
                    candidate.parent_result_id
                )
                baseline_result_id = (
                    getattr(upstream_action, "baseline_result_id", None)
                    or getattr(upstream_action, "parent_result_id", None)
                )
            if not baseline_result_id:
                baseline_result_id = candidate.parent_result_id
            if not baseline_result_id:
                continue
            baseline = results_by_id.get(baseline_result_id)
            if baseline is None:
                continue
            for hypothesis_id in candidate.hypothesis_ids:
                descendants_by_hypothesis_id.setdefault(
                    hypothesis_id, []
                ).append((result, baseline))

    updated_hypotheses: list[HypothesisCard] = []
    update_summaries: list[dict[str, Any]] = []
    for hypothesis in request.hypotheses:
        descendants = descendants_by_hypothesis_id.get(
            hypothesis.hypothesis_id, []
        )
        if (
            not descendants
            and request.current_tick - hypothesis.tick_created
            < hypothesis.ttl_ticks
        ):
            continue
        updated_hypothesis = update_hypothesis(
            hypothesis,
            healthy_descendants=descendants,
            current_tick=request.current_tick,
            axis_thresholds=axis_thresholds,
            cfg=request.lifecycle_config,
        )
        lifecycle_changed = (
            updated_hypothesis.status != hypothesis.status
            or updated_hypothesis.support_points != hypothesis.support_points
            or updated_hypothesis.contradiction_points
            != hypothesis.contradiction_points
            or updated_hypothesis.descendants_evaluated
            != hypothesis.descendants_evaluated
            or updated_hypothesis.last_evaluated_tick
            != hypothesis.last_evaluated_tick
        )
        if not lifecycle_changed:
            continue
        updated_hypotheses.append(updated_hypothesis)
        update_summaries.append({
            "hypothesis_id": hypothesis.hypothesis_id,
            "old_status": hypothesis.status,
            "new_status": updated_hypothesis.status,
            "support_points": updated_hypothesis.support_points,
            "contradiction_points": updated_hypothesis.contradiction_points,
            "n_descendants": updated_hypothesis.descendants_evaluated,
        })

    return LifecyclePhaseResult(
        updated_hypotheses=tuple(updated_hypotheses),
        update_summaries=tuple(update_summaries),
    )


__all__ = [
    "LifecyclePhaseRequest",
    "LifecyclePhaseResult",
    "PrePlannerLifecycleRequest",
    "PrePlannerLifecycleResult",
    "retire_expired_hypotheses",
    "update_lifecycle_phase",
]
