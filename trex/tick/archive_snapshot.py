"""Read-only archive inputs and bounded history for one live tick.

This module keeps archive traversal out of the scientific-tick composition
root. It does not append records or make policy decisions: callers receive a
typed snapshot and explicitly decide when any derived records are persisted.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, TypedDict

from ..archive import Archive
from ..schemas import (
    ActionCandidate,
    DispatchRecord,
    EvidenceSummary,
    HypothesisCard,
    LLMCallRecord,
    LLMHealthSummary,
    LaunchDecision,
    ResultRecord,
    SupervisorDecision,
)


class TrajectoryLaunch(TypedDict):
    """Compact launch configuration included in one trajectory entry."""

    family: str
    key_knobs: list[str]
    config_delta: dict[str, Any]


class TickTrajectoryEntry(TypedDict):
    """Bounded prior-tick context shown to the Planner."""

    tick_id: str
    state_label: str
    strict_count: int
    run_su_count: int
    run_su_count_delta: int
    near_miss_count: int
    n_launches: int
    dominant_family: str | None
    launches: list[TrajectoryLaunch]


@dataclass(frozen=True)
class TickArchiveSnapshot:
    """Append-order-preserving records needed to compose one live tick."""

    target_results: tuple[ResultRecord, ...]
    actions: tuple[ActionCandidate, ...]
    latest_target_hypotheses: tuple[HypothesisCard, ...]
    lifecycle_hypotheses: tuple[HypothesisCard, ...]
    active_hypotheses: tuple[HypothesisCard, ...]
    evidence_records: tuple[EvidenceSummary, ...]
    prior_target_evidence: tuple[EvidenceSummary, ...]
    launched_decisions: tuple[LaunchDecision, ...]
    dispatch_records: tuple[DispatchRecord, ...]
    supervisor_decisions: tuple[SupervisorDecision, ...]
    llm_call_records: tuple[LLMCallRecord, ...]
    spawning_action_by_result_id: Mapping[str, ActionCandidate]


@dataclass(frozen=True)
class WarmstartCoverage:
    """Observed and still-required deterministic cold-start families."""

    completed_families: frozenset[str]
    missing_families: frozenset[str]
    seen: bool


def index_actions_by_spawned_result(
    actions: Sequence[ActionCandidate],
    results: Sequence[ResultRecord],
) -> dict[str, ActionCandidate]:
    """Map result IDs to their immediate archived candidate parent.

    Archives without ActionCandidate records produce an empty mapping, allowing
    the reducer to use family-level recipe signatures.
    """

    action_by_candidate_id = {
        action.candidate_id: action for action in actions
    }
    spawning_action_by_result_id: dict[str, ActionCandidate] = {}
    for result in results:
        for parent_id in result.parent_ids:
            spawning_action = action_by_candidate_id.get(parent_id)
            if spawning_action is not None:
                spawning_action_by_result_id[result.result_id] = spawning_action
                break
    return spawning_action_by_result_id


def _target_hypothesis_views(
    hypotheses: Sequence[HypothesisCard],
    *,
    target_id: str,
) -> tuple[
    tuple[HypothesisCard, ...],
    tuple[HypothesisCard, ...],
    tuple[HypothesisCard, ...],
]:
    """Reproduce the archive's latest-row and active-memory projections."""

    latest_by_id: dict[str, tuple[int, HypothesisCard]] = {}
    for archive_index, hypothesis in enumerate(hypotheses):
        if hypothesis.target_id == target_id:
            latest_by_id[hypothesis.hypothesis_id] = (
                archive_index,
                hypothesis,
            )

    latest_target_hypotheses = tuple(
        hypothesis for _, hypothesis in latest_by_id.values()
    )

    def _ordered(allowed_statuses: set[str]) -> tuple[HypothesisCard, ...]:
        selected = [
            hypothesis
            for _, hypothesis in latest_by_id.values()
            if hypothesis.status in allowed_statuses
        ]
        selected.sort(
            key=lambda hypothesis: (
                hypothesis.status == "active",
                latest_by_id[hypothesis.hypothesis_id][0],
                hypothesis.tick_created,
            ),
            reverse=True,
        )
        return tuple(selected)

    lifecycle_hypotheses = _ordered({"active"})
    active_hypotheses = _ordered(
        {"active", "supported", "contradicted"}
    )[:8]
    return (
        latest_target_hypotheses,
        lifecycle_hypotheses,
        active_hypotheses,
    )


def load_tick_archive_snapshot(
    archive: Archive,
    target_id: str,
) -> TickArchiveSnapshot:
    """Load every pre-tick archive stream needed by ``run_live_tick`` once."""

    target_results = tuple(
        result
        for result in archive.iter_records(ResultRecord)
        if result.target_id == target_id
    )
    actions = tuple(archive.iter_records(ActionCandidate))
    hypotheses = tuple(archive.iter_records(HypothesisCard))
    (
        latest_target_hypotheses,
        lifecycle_hypotheses,
        active_hypotheses,
    ) = _target_hypothesis_views(hypotheses, target_id=target_id)
    launched_decisions = tuple(
        decision
        for decision in archive.iter_records(LaunchDecision)
        if decision.status == "launched"
    )
    evidence_records = tuple(archive.iter_records(EvidenceSummary))

    return TickArchiveSnapshot(
        target_results=target_results,
        actions=actions,
        latest_target_hypotheses=latest_target_hypotheses,
        lifecycle_hypotheses=lifecycle_hypotheses,
        active_hypotheses=active_hypotheses,
        evidence_records=evidence_records,
        prior_target_evidence=tuple(
            evidence
            for evidence in evidence_records
            if evidence.target_id == target_id
        ),
        launched_decisions=launched_decisions,
        dispatch_records=tuple(archive.iter_records(DispatchRecord)),
        supervisor_decisions=tuple(
            archive.iter_records(SupervisorDecision)
        ),
        llm_call_records=tuple(archive.iter_records(LLMCallRecord)),
        spawning_action_by_result_id=index_actions_by_spawned_result(
            actions,
            target_results,
        ),
    )


def resolve_warmstart_coverage(
    *,
    actions: Sequence[ActionCandidate],
    launched_decisions: Sequence[LaunchDecision],
    required_families: Iterable[str],
    unavailable_families: Iterable[str] = (),
) -> WarmstartCoverage:
    """Return deterministic cold-start family coverage from archived launches."""

    unavailable = set(unavailable_families)
    required = {
        family for family in required_families if family not in unavailable
    }
    action_family_by_id = {
        action.candidate_id: action.method_family for action in actions
    }
    completed = {
        action_family_by_id.get(decision.candidate_id, "")
        for decision in launched_decisions
        if decision.candidate_id.startswith("warmstart_")
    }
    completed.discard("")
    missing = required - completed
    return WarmstartCoverage(
        completed_families=frozenset(completed),
        missing_families=frozenset(missing),
        seen=not missing,
    )


def build_recent_tick_trajectory(
    *,
    prior_evidence: Sequence[EvidenceSummary],
    actions: Sequence[ActionCandidate],
    dispatch_records: Sequence[DispatchRecord],
    max_ticks: int = 20,
) -> tuple[TickTrajectoryEntry, ...]:
    """Build bounded state/family/config history from realized dispatches."""

    action_by_candidate_id = {
        action.candidate_id: action for action in actions
    }
    started_actions_by_tick: dict[str, list[ActionCandidate]] = {}
    for dispatch in dispatch_records:
        if dispatch.status != "started":
            continue
        action = action_by_candidate_id.get(dispatch.candidate_id)
        if action is not None:
            started_actions_by_tick.setdefault(dispatch.tick_id, []).append(
                action
            )

    trajectory: list[TickTrajectoryEntry] = []
    for evidence in prior_evidence[-max_ticks:]:
        launches: list[TrajectoryLaunch] = []
        for action in started_actions_by_tick.get(evidence.tick_id, []):
            config_delta = action.config_delta or {}
            key_knobs = sorted(config_delta)[:5]
            launches.append({
                "family": action.method_family,
                "key_knobs": key_knobs,
                "config_delta": {
                    key: config_delta[key] for key in key_knobs
                },
            })

        family_counts: dict[str, int] = {}
        for launch in launches:
            family = launch["family"]
            family_counts[family] = family_counts.get(family, 0) + 1
        dominant_family = (
            max(family_counts, key=family_counts.__getitem__)
            if family_counts
            else None
        )
        trajectory.append({
            "tick_id": evidence.tick_id,
            "state_label": evidence.state_label,
            "strict_count": evidence.strict_count,
            "run_su_count": evidence.run_su_count,
            "run_su_count_delta": evidence.run_su_count_delta,
            "near_miss_count": evidence.near_miss_count,
            "n_launches": len(launches),
            "dominant_family": dominant_family,
            "launches": launches,
        })
    return tuple(trajectory)


def recent_fallback_rate(
    llm_call_records: Sequence[LLMCallRecord],
    *,
    window: int = 4,
) -> float:
    """Return the recent fraction of fallback or non-OK LLM calls."""

    if not llm_call_records:
        return 0.0
    recent = llm_call_records[-window:]
    fallback_count = sum(
        1
        for call in recent
        if call.fallback_triggered or call.parse_status != "ok"
    )
    return fallback_count / max(1, len(recent))


def summarize_llm_health(
    llm_call_records: Sequence[LLMCallRecord],
    model: str,
    *,
    window: int = 12,
) -> LLMHealthSummary:
    """Summarize recent archived LLM calls for evidence telemetry."""

    if not llm_call_records:
        return LLMHealthSummary(
            model=model,
            last_calls_window=[],
            parse_fail_rate=0.0,
            schema_fail_rate=0.0,
            timeout_count=0,
            median_latency_s=0.0,
        )

    recent = llm_call_records[-max(1, int(window)):]
    status_window: list[str] = []
    for call in recent:
        status = str(call.parse_status or "parse_fail")
        if status == "ok" and call.abstain:
            status = "abstain"
        elif (
            status == "ok"
            and call.confidence is not None
            and call.confidence < 0.35
        ):
            status = "low_conf"
        status_window.append(status)
    latencies = [
        float(call.latency_s)
        for call in recent
        if (
            isinstance(call.latency_s, (int, float))
            and call.latency_s >= 0.0
        )
    ]
    sample_count = max(1, len(recent))
    return LLMHealthSummary(
        model=model,
        last_calls_window=status_window,  # type: ignore[arg-type]
        parse_fail_rate=round(
            sum(
                1 for call in recent if call.parse_status == "parse_fail"
            ) / sample_count,
            4,
        ),
        schema_fail_rate=round(
            sum(
                1 for call in recent if call.parse_status == "schema_fail"
            ) / sample_count,
            4,
        ),
        timeout_count=sum(
            1 for call in recent if call.parse_status == "timeout"
        ),
        median_latency_s=(
            round(statistics.median(latencies), 3) if latencies else 0.0
        ),
    )


__all__ = [
    "TickArchiveSnapshot",
    "TickTrajectoryEntry",
    "TrajectoryLaunch",
    "WarmstartCoverage",
    "build_recent_tick_trajectory",
    "index_actions_by_spawned_result",
    "load_tick_archive_snapshot",
    "recent_fallback_rate",
    "resolve_warmstart_coverage",
    "summarize_llm_health",
]
