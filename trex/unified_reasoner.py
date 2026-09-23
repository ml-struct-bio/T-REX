"""Optional deterministic Supervisor output derived from Planner cards.

This experimental alternative avoids a separate Supervisor LLM call; it is not the
default campaign path.
"""
from __future__ import annotations

from .schemas import (
    ActionCandidate,
    CandidateDecision,
    HypothesisCard,
    PlannerOutput,
    ReasoningTrace,
    SupervisorOutput,
)

_MODES = ("exploit", "rescue", "explore")


def _card_mode(card: HypothesisCard | None) -> str:
    """Mode a card prefers = argmax of its mode_affinity (explore default)."""
    aff = (card.mode_affinity if card is not None else None) or {}
    if not aff:
        return "explore"
    return max(_MODES, key=lambda m: aff.get(m, 0.0))


def build_unified_supervisor_output(
    planner_out: PlannerOutput,
    candidates: list[ActionCandidate],
    *,
    cards: list[HypothesisCard] | None = None,
) -> SupervisorOutput:
    """Derive candidate modes, ordering, and allocation from Planner cards.

    The optional cards argument supplies the hypotheses used to construct
    candidates, including hypotheses reused after a Planner fallback.
    """
    source_cards = cards if cards is not None else (planner_out.cards or [])
    cards_by_id = {c.hypothesis_id: c for c in source_cards}

    per_mode: dict[str, list[ActionCandidate]] = {m: [] for m in _MODES}
    cand_card: dict[str, HypothesisCard | None] = {}
    for cand in candidates:
        card = next(
            (cards_by_id[h] for h in (cand.hypothesis_ids or []) if h in cards_by_id),
            None,
        )
        cand_card[cand.candidate_id] = card
        per_mode[_card_mode(card)].append(cand)

    decisions: list[CandidateDecision] = []
    global_rank_by_id = {
        cand.candidate_id: rank for rank, cand in enumerate(candidates, start=1)
    }
    for mode in _MODES:
        for rank, cand in enumerate(per_mode[mode], start=1):
            card = cand_card[cand.candidate_id]
            trace = (card.reasoning_trace if card is not None else None) or ReasoningTrace()
            decisions.append(
                CandidateDecision(
                    candidate_id=cand.candidate_id,
                    mode=mode,                       # type: ignore[arg-type]
                    rank_in_mode=rank,
                    global_rank=global_rank_by_id[cand.candidate_id],
                    resource_class=cand.estimated_cost_class,
                    what=((card.claim if card is not None else cand.expected_signal) or "")[:200],
                    why=(trace.inference or "")[:200],
                    evidence_refs=list(cand.evidence_refs or []),
                    expected_signal=cand.expected_signal,
                    stop_or_downgrade_if="",
                    reasoning_trace=trace,
                )
            )

    agg = {m: 0.0 for m in _MODES}
    for card in source_cards:
        aff = card.mode_affinity or {}
        for m in _MODES:
            agg[m] += aff.get(m, 0.0)
    total = sum(agg.values())
    mode_mixture = (
        {m: agg[m] / total for m in _MODES} if total > 0
        else {m: 1.0 / len(_MODES) for m in _MODES}
    )

    return SupervisorOutput(
        valid=bool(decisions),
        abstain=not decisions,
        confidence=float(getattr(planner_out, "confidence", 0.6) or 0.6),
        fail_reason=None if decisions else "unified_no_candidates",
        mode_mixture=mode_mixture,
        candidate_decisions=decisions,
        rationale="[unified-reasoner] ranking derived from Planner cards (no 2nd LLM call)",
        raw_text="",
        usage={},
    )
