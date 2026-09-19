"""Supervisor/fallback policy resolution for deterministic selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..fallback import (
    DEFAULT_MIXTURES,
    clamp_mixture,
    should_apply_category_b_clamps,
)
from ..schemas import (
    ActionCandidate,
    CandidateDecision,
    EvidenceSummary,
    SupervisorOutput,
)


@dataclass(frozen=True)
class SelectionPolicyRequest:
    """Inputs that determine mixture source, clamps, ranks, and candidate modes."""

    evidence: EvidenceSummary
    candidates: Sequence[ActionCandidate]
    supervisor_output: SupervisorOutput
    fallback_reason: str | None
    candidate_mode_hints: Mapping[str, str]
    derive_mixture_from_ranking: bool
    realize_global_priority: bool
    all_explore_backends_unhealthy: bool
    route_backlog_saturated: bool
    panel_live: bool


@dataclass(frozen=True)
class SelectionPolicyResult:
    """Resolved read-only policy consumed by admission, quota, and ranking."""

    supervisor_decisions: dict[str, CandidateDecision]
    supervisor_used: bool
    use_supervisor_ranks: bool
    global_priority_used: bool
    source: str
    raw_mixture: dict[str, float]
    clamped_mixture: dict[str, float]
    clamp_log: tuple[str, ...]
    category_b_enabled: bool
    category_b_reasons: tuple[str, ...]
    candidate_modes: dict[str, str]
    has_mode_information: bool

    def mode_for(self, candidate: ActionCandidate) -> str | None:
        """Return the resolved E/R/E mode for one candidate."""

        return self.candidate_modes.get(candidate.candidate_id)


def _mixture_from_decisions(
    decisions: Mapping[str, CandidateDecision],
) -> dict[str, float] | None:
    weights = {"exploit": 0.0, "rescue": 0.0, "explore": 0.0}
    for decision in decisions.values():
        if decision.mode in weights:
            weights[decision.mode] += 1.0 / max(1, decision.rank_in_mode)
    total = sum(weights.values())
    if total <= 0:
        return None
    return {
        mode: weight / total
        for mode, weight in weights.items()
    }


def resolve_selection_policy(
    request: SelectionPolicyRequest,
) -> SelectionPolicyResult:
    """Resolve Supervisor intent without filtering or ranking candidates."""

    supervisor_output = request.supervisor_output
    decisions = {
        decision.candidate_id: decision
        for decision in supervisor_output.candidate_decisions
    }
    supervisor_used = (
        request.fallback_reason is None
        and supervisor_output.valid
        and (bool(supervisor_output.mode_mixture) or bool(decisions))
    )
    derived_mixture = (
        _mixture_from_decisions(decisions)
        if supervisor_used and request.derive_mixture_from_ranking
        else None
    )
    if derived_mixture is not None:
        source = "supervisor_ranking_derived"
        raw_mixture = derived_mixture
    elif supervisor_used and supervisor_output.mode_mixture:
        source = "supervisor_llm_scalar"
        raw_mixture = supervisor_output.mode_mixture
    else:
        source = (
            f"fallback({request.fallback_reason or 'invalid_supervisor'})"
        )
        raw_mixture = DEFAULT_MIXTURES.get(
            request.evidence.state_label,
            DEFAULT_MIXTURES["low_evidence"],
        )

    category_b_enabled, category_b_reasons = should_apply_category_b_clamps(
        request.evidence.state_label,
        top_bin_share=request.evidence.top_bin_share,
        strict_su_top_bin_share=getattr(
            request.evidence,
            "strict_su_top_bin_share",
            None,
        ),
        panel_ready_bins_covered=request.evidence.panel_ready_bins_covered,
        supervisor_confidence=supervisor_output.confidence,
        recent_fallback_high=bool(request.evidence.recent_fallback_high),
        supervisor_used=supervisor_used,
        panel_live=request.panel_live,
    )
    clamped_mixture, clamp_log = clamp_mixture(
        raw_mixture,
        request.evidence.state_label,
        all_explore_backends_unhealthy=(
            request.all_explore_backends_unhealthy
        ),
        route_backlog_saturated=request.route_backlog_saturated,
        top_bin_share=request.evidence.top_bin_share,
        strict_su_top_bin_share=getattr(
            request.evidence,
            "strict_su_top_bin_share",
            None,
        ),
        panel_ready_bins_covered=request.evidence.panel_ready_bins_covered,
        category_b_enabled=category_b_enabled,
        panel_live=request.panel_live,
    )
    clamp_tag = (
        "clamp:cat_b_on" if category_b_enabled else "clamp:cat_b_off"
    )
    clamp_log = [
        clamp_tag + "(" + ",".join(category_b_reasons) + ")",
        *clamp_log,
    ]

    use_supervisor_ranks = (
        request.fallback_reason is None
        and supervisor_output.valid
        and bool(decisions)
    )
    global_priority_used = bool(
        request.realize_global_priority
        and use_supervisor_ranks
        and decisions
        and all(
            isinstance(getattr(decision, "global_rank", None), int)
            and int(getattr(decision, "global_rank", 0) or 0) >= 1
            for decision in decisions.values()
        )
        and len({
            int(decision.global_rank)
            for decision in decisions.values()
        })
        == len(decisions)
    )

    has_mode_hints = bool(request.candidate_mode_hints)
    candidate_modes: dict[str, str] = {}
    for candidate in request.candidates:
        mode: str | None = None
        if use_supervisor_ranks:
            decision = decisions.get(candidate.candidate_id)
            if decision is not None:
                mode = decision.mode
        if mode is None and has_mode_hints:
            mode = request.candidate_mode_hints.get(candidate.candidate_id)
        if mode is not None:
            candidate_modes[candidate.candidate_id] = mode

    return SelectionPolicyResult(
        supervisor_decisions=decisions,
        supervisor_used=supervisor_used,
        use_supervisor_ranks=use_supervisor_ranks,
        global_priority_used=global_priority_used,
        source=source,
        raw_mixture=raw_mixture,
        clamped_mixture=clamped_mixture,
        clamp_log=tuple(clamp_log),
        category_b_enabled=category_b_enabled,
        category_b_reasons=tuple(category_b_reasons),
        candidate_modes=candidate_modes,
        has_mode_information=use_supervisor_ranks or has_mode_hints,
    )
