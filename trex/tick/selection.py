"""Final live-tick selection inputs and deterministic launch decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..schemas import (
    ActionCandidate,
    EvidenceSummary,
    HypothesisCard,
    LaunchDecision,
    RouteHealthSummary,
    SupervisorOutput,
)
from ..selector import SelectorConfig, select_launches


@dataclass(frozen=True)
class SelectionPhaseRequest:
    """Complete, reviewable input to the deterministic Selector."""

    evidence: EvidenceSummary
    candidates: Sequence[ActionCandidate]
    supervisor_output: SupervisorOutput
    selector_config: SelectorConfig
    tick_id: str
    fallback_reason: str | None
    hypotheses: Sequence[HypothesisCard]
    recent_realized_modes: Sequence[str]
    mode_credit: Mapping[str, float]
    recent_started_families: Sequence[str]
    route_health: RouteHealthSummary | None


@dataclass(frozen=True)
class SelectionPhaseResult:
    """Launch decisions plus the exact derived inputs used by the Selector."""

    launches: tuple[LaunchDecision, ...]
    selector_debug: dict[str, Any]
    candidate_mode_hints: dict[str, str]
    route_backlog_saturated: bool


def _candidate_mode_hints(
    candidates: Sequence[ActionCandidate],
    hypotheses: Sequence[HypothesisCard],
) -> dict[str, str]:
    affinity_by_hypothesis_id = {
        hypothesis.hypothesis_id: hypothesis.mode_affinity
        for hypothesis in hypotheses
        if hypothesis.mode_affinity
    }
    candidate_mode_hints: dict[str, str] = {}
    for candidate in candidates:
        if candidate.candidate_id.startswith("evidence_fallback"):
            candidate_mode_hints[candidate.candidate_id] = "explore"
            continue

        best_mode: str | None = None
        best_affinity = float("-inf")
        for hypothesis_id in candidate.hypothesis_ids:
            mode_affinity = affinity_by_hypothesis_id.get(hypothesis_id)
            if not mode_affinity:
                continue
            candidate_best_mode = max(mode_affinity, key=mode_affinity.get)
            if mode_affinity[candidate_best_mode] > best_affinity:
                best_mode = candidate_best_mode
                best_affinity = mode_affinity[candidate_best_mode]
        if best_mode is not None:
            candidate_mode_hints[candidate.candidate_id] = best_mode
    return candidate_mode_hints


def select_live_tick_launches(
    request: SelectionPhaseRequest,
) -> SelectionPhaseResult:
    """Derive selector context and return this tick's launch decisions."""

    candidate_mode_hints = _candidate_mode_hints(
        request.candidates, request.hypotheses
    )
    route_backlog_saturated = bool(
        request.route_health is not None
        and request.route_health.backlog_used
        / max(1, request.route_health.backlog_cap)
        >= 0.95
    )
    launches, selector_debug = select_launches(
        request.evidence,
        list(request.candidates),
        request.supervisor_output,
        cfg=request.selector_config,
        tick_id=request.tick_id,
        fallback_reason=request.fallback_reason,
        recent_modes=list(request.recent_realized_modes),
        mode_credit=dict(request.mode_credit),
        candidate_mode_hint=candidate_mode_hints,
        route_backlog_saturated=route_backlog_saturated,
        recent_started_families=list(request.recent_started_families),
    )
    return SelectionPhaseResult(
        launches=tuple(launches),
        selector_debug=selector_debug,
        candidate_mode_hints=candidate_mode_hints,
        route_backlog_saturated=route_backlog_saturated,
    )


__all__ = [
    "SelectionPhaseRequest",
    "SelectionPhaseResult",
    "select_live_tick_launches",
]
