"""Planner proposal and deterministic critic phase for one live tick."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..archive import Archive
from ..capability_registry import default_registry
from ..critic import CriticCallConfig
from ..critic_guard import critic_guard_record, deterministic_critic
from ..planner import PlannerCallConfig, build_user_prompt, planner_system_prompt
from ..schemas import (
    EvidenceSummary,
    HypothesisCard,
    LLMCallRecord,
    PlannerOutput,
    to_jsonable,
)


@dataclass(frozen=True)
class ProposalPhaseRequest:
    """Evidence, policy, and history consumed by the Planner phase."""

    archive: Archive
    evidence: EvidenceSummary
    active_hypotheses: Sequence[HypothesisCard]
    warmstart_missing_families: frozenset[str]
    warmstart_seen: bool
    unavailable_families: frozenset[str]
    planner_config: PlannerCallConfig
    critic_config: CriticCallConfig
    tick_id: str
    tick_number: int
    reuse_prior_plan: bool
    bootstrap_seed_action_families: Sequence[str]
    schema_version: str


@dataclass(frozen=True)
class ProposalPhaseDependencies:
    """Live-tick callbacks kept injectable for compatibility and testing."""

    call_planner: Callable[..., PlannerOutput]
    build_llm_call_record: Callable[..., LLMCallRecord]
    build_prompt_audit_snapshot: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ProposalPhaseResult:
    """Planner decision, selected cards, audit records, and fallback state."""

    planner_output: PlannerOutput
    planner_latency_seconds: float
    hypotheses: tuple[HypothesisCard, ...]
    fallback_reason: str | None
    critic_flags: tuple[str, ...]
    records_to_append: tuple[LLMCallRecord | HypothesisCard, ...]
    available_families: tuple[str, ...]
    seed_action_families: tuple[str, ...]


def run_proposal_phase(
    request: ProposalPhaseRequest,
    dependencies: ProposalPhaseDependencies,
    *,
    clock: Callable[[], float] = time.time,
) -> ProposalPhaseResult:
    """Run or reuse the Planner, then apply the deterministic critic guard."""

    registry = default_registry()
    available_families = tuple(
        family
        for family in registry.feasible_families()
        if family not in request.unavailable_families
    )

    recent_critic_flags: list[str] = []
    for record in request.archive.iter_records(LLMCallRecord):
        if record.role == "critic":
            recent_critic_flags = list(record.critic_flags or [])
    if recent_critic_flags:
        print(
            "  [planner] last-tick critic flags surfaced: "
            f"{recent_critic_flags[:5]}",
            flush=True,
        )

    should_seed_warmstart = (
        request.evidence.state_label == "low_evidence"
        and not request.evidence.recipes
        and not request.warmstart_seen
    )
    seed_action_families = (
        tuple(
            family
            for family in request.bootstrap_seed_action_families
            if family in request.warmstart_missing_families
        )
        if should_seed_warmstart
        else ()
    )
    active_hypothesis_payloads = [
        to_jsonable(hypothesis) for hypothesis in request.active_hypotheses
    ]
    planner_user_payload = build_user_prompt(
        request.evidence,
        active_hypothesis_payloads,
        list(seed_action_families),
        available_families=list(available_families),
        recent_critic_flags=recent_critic_flags,
    )
    planner_prompt_text = planner_system_prompt() + "\n" + planner_user_payload

    planner_started_at = clock()
    if request.reuse_prior_plan:
        planner_output = PlannerOutput(
            valid=True,
            abstain=False,
            confidence=1.0,
            fail_reason=None,
            cards=list(request.active_hypotheses),
            rationale="[evidence-skip] reused prior cards (signature unchanged)",
            raw_text="[evidence-skip]",
            usage={},
        )
        planner_latency_seconds = 0.0
    else:
        planner_output = dependencies.call_planner(
            request.evidence,
            active_hypotheses=active_hypothesis_payloads,
            seed_action_families=list(seed_action_families),
            tick_id_int=request.tick_number,
            cfg=request.planner_config,
            available_families=list(available_families),
            recent_critic_flags=recent_critic_flags,
        )
        planner_latency_seconds = clock() - planner_started_at

    records_to_append: list[LLMCallRecord | HypothesisCard] = [
        dependencies.build_llm_call_record(
            "planner",
            tick_id=request.tick_id,
            cfg_model=request.planner_config.model,
            prompt_text=planner_prompt_text,
            raw_text=planner_output.raw_text,
            valid=planner_output.valid,
            fail_reason=planner_output.fail_reason,
            confidence=planner_output.confidence,
            abstain=planner_output.abstain,
            fallback=not planner_output.valid or planner_output.abstain,
            latency_s=planner_latency_seconds,
            usage=planner_output.usage,
            prompt_audit=dependencies.build_prompt_audit_snapshot(
                request.evidence,
                role="planner",
                available_families=list(available_families),
                seed_action_families=list(seed_action_families),
                active_hypothesis_count=len(request.active_hypotheses),
                recent_critic_flags=recent_critic_flags,
            ),
        )
    ]
    if not request.reuse_prior_plan:
        records_to_append.extend(planner_output.cards)

    critic_flags: list[str] = []
    if (
        request.critic_config.enabled
        and planner_output.valid
        and not planner_output.abstain
        and planner_output.cards
    ):
        guard_output = deterministic_critic(
            request.evidence,
            planner_output.cards,
            current_tick=request.tick_number,
        )
        critic_flags = guard_output.flags
        records_to_append.append(critic_guard_record(
            tick_id=request.tick_id,
            model=request.critic_config.model,
            schema_version=request.schema_version,
            out=guard_output,
        ))

    planner_should_fallback = (
        not planner_output.valid
        or planner_output.abstain
        or not planner_output.cards
    )
    hypotheses = tuple(
        request.active_hypotheses
        if planner_should_fallback
        else planner_output.cards
    )
    if not planner_output.valid:
        fallback_reason = planner_output.fail_reason or "planner_invalid"
    elif planner_output.abstain:
        fallback_reason = "planner_abstain"
    elif not planner_output.cards:
        fallback_reason = "planner_empty_cards"
    else:
        fallback_reason = planner_output.fail_reason

    return ProposalPhaseResult(
        planner_output=planner_output,
        planner_latency_seconds=planner_latency_seconds,
        hypotheses=hypotheses,
        fallback_reason=fallback_reason,
        critic_flags=tuple(critic_flags),
        records_to_append=tuple(records_to_append),
        available_families=available_families,
        seed_action_families=seed_action_families,
    )


__all__ = [
    "ProposalPhaseDependencies",
    "ProposalPhaseRequest",
    "ProposalPhaseResult",
    "run_proposal_phase",
]
