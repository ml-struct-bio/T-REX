"""Supervisor context, decision, and fallback phase for one live tick."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..schemas import (
    ActionCandidate,
    EvidenceSummary,
    HypothesisCard,
    LLMCallRecord,
    PlannerOutput,
    SupervisorOutput,
)
from ..selector import (
    SelectorConfig,
    effective_mode_window_k,
    family_cost_penalties_for_cfg,
    high_cost_capacity_blocked_families_for_cfg,
    high_cost_cap_for_evidence,
)
from ..supervisor import (
    supervisor_system_prompt,
    SupervisorCallConfig,
    build_user_prompt,
)


@dataclass(frozen=True)
class SupervisionPhaseRequest:
    """Scientific and execution state shown to the Supervisor."""

    evidence: EvidenceSummary
    hypotheses: Sequence[HypothesisCard]
    candidates: Sequence[ActionCandidate]
    planner_output: PlannerOutput
    planner_fallback_reason: str | None
    selector_config: SelectorConfig
    supervisor_config: SupervisorCallConfig
    recent_realized_modes: Sequence[str]
    mode_credit: Mapping[str, float]
    diagnostic_chain_backlog: Mapping[str, Any] | None
    pending_family_load: Mapping[str, Any] | None
    execution_realization: Mapping[str, Any] | None
    tick_id: str
    reuse_prior_plan: bool
    reuse_mode_mixture: Mapping[str, float] | None
    reuse_confidence: float
    use_unified_reasoner: bool


@dataclass(frozen=True)
class SupervisionPhaseDependencies:
    """Injectable calls and audit constructors used by supervision."""

    call_supervisor: Callable[..., SupervisorOutput]
    build_llm_call_record: Callable[..., LLMCallRecord]
    build_prompt_audit_snapshot: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class SupervisionPhaseResult:
    """Supervisor output and exact Selector-facing fallback context."""

    supervisor_output: SupervisorOutput
    supervisor_latency_seconds: float
    fallback_reason: str | None
    selector_context: dict[str, Any]
    llm_call_record: LLMCallRecord


def _recent_mode_context(request: SupervisionPhaseRequest) -> dict[str, Any]:
    effective_window = effective_mode_window_k(
        request.evidence, request.selector_config
    )
    uses_windowed_quota = (
        request.selector_config.quota_realization == "deterministic_deficit"
    )
    recent_mode_window = (
        list(request.recent_realized_modes[-effective_window:])
        if uses_windowed_quota
        else []
    )
    if uses_windowed_quota:
        return {
            "configured_window_k": request.selector_config.mode_window_k,
            "effective_window_k": effective_window,
            "window_disabled": False,
            "last_modes": recent_mode_window,
            "counts": {
                mode: recent_mode_window.count(mode)
                for mode in ("exploit", "rescue", "explore")
            },
            "actual_share": {
                mode: recent_mode_window.count(mode)
                / max(1, len(recent_mode_window))
                for mode in ("exploit", "rescue", "explore")
            },
            "mode_credit": {},
        }
    return {
        "configured_window_k": 0,
        "effective_window_k": 0,
        "window_disabled": True,
        "realization_memory": request.selector_config.quota_realization,
        "last_modes": [],
        "counts": {mode: 0 for mode in ("exploit", "rescue", "explore")},
        "actual_share": {
            mode: 0.0 for mode in ("exploit", "rescue", "explore")
        },
        "mode_credit": (
            dict(request.mode_credit)
            if request.selector_config.quota_realization == "fractional_carry"
            else {}
        ),
    }


def build_supervisor_selector_context(
    request: SupervisionPhaseRequest,
) -> dict[str, Any]:
    """Build the execution and cost context shown to the Supervisor."""

    selector_config = request.selector_config
    cost_penalties = family_cost_penalties_for_cfg(
        request.evidence, selector_config
    )
    capacity_pressure_families = (
        high_cost_capacity_blocked_families_for_cfg(
            request.evidence, selector_config
        )
    )
    evidence_pending_load = (
        getattr(request.evidence, "pending_family_load", {}) or {}
    )
    pending_by_family = (
        evidence_pending_load.get("by_family", {})
        if isinstance(evidence_pending_load, dict)
        else {}
    )

    high_cost_caps: dict[str, Any] = {}
    high_cost_families = sorted(
        set(selector_config.high_cost_pending_families)
        | set(pending_by_family.keys())
    )
    for family in high_cost_families:
        if family not in set(selector_config.high_cost_pending_families):
            continue
        cap, cap_source = high_cost_cap_for_evidence(
            request.evidence,
            family,
            high_cost_pending_cap=selector_config.high_cost_pending_cap,
            high_cost_pending_promoted_cap=(
                selector_config.high_cost_pending_promoted_cap
            ),
            high_cost_pending_deep_stall_cap=(
                selector_config.high_cost_pending_deep_stall_cap
            ),
            high_cost_pending_dry_pivot_cap=(
                selector_config.high_cost_pending_dry_pivot_cap
            ),
            high_cost_pending_strong_cap=(
                selector_config.high_cost_pending_strong_cap
            ),
            high_cost_dry_pivot_min_gpu_h=(
                selector_config.high_cost_dry_pivot_min_gpu_h
            ),
            high_cost_dry_pivot_min_completed_children=(
                selector_config.high_cost_dry_pivot_min_completed_children
            ),
            high_cost_dry_pivot_max_best_recent_su_per_gpu_h=(
                selector_config.high_cost_dry_pivot_max_best_recent_su_per_gpu_h
            ),
            high_cost_strong_min_su=selector_config.high_cost_strong_min_su,
            high_cost_strong_min_gpu_h=(
                selector_config.high_cost_strong_min_gpu_h
            ),
            high_cost_strong_min_recent_su_per_gpu_h=(
                selector_config.high_cost_strong_min_recent_su_per_gpu_h
            ),
            high_cost_strong_best_fraction=(
                selector_config.high_cost_strong_best_fraction
            ),
            min_su_per_gpu_h=selector_config.cost_aware_min_su_per_gpu_h,
            high_cost_repeated_support_min_proposed=(
                selector_config.high_cost_repeated_support_min_proposed
            ),
            high_cost_repeated_support_max_started_fraction=(
                selector_config.high_cost_repeated_support_max_started_fraction
            ),
            high_cost_repeated_support_cap=(
                selector_config.high_cost_repeated_support_cap
            ),
            high_cost_repeated_support_max_negative_gpu_h=(
                selector_config.high_cost_repeated_support_max_negative_gpu_h
            ),
            high_cost_repeated_support_max_timeouts=(
                selector_config.high_cost_repeated_support_max_timeouts
            ),
        )
        family_load = (
            pending_by_family.get(family, {})
            if isinstance(pending_by_family, dict)
            else {}
        )
        high_cost_caps[family] = {
            "cap_running_workers": cap,
            "cap_source": cap_source,
            "running": (
                int((family_load or {}).get("running", 0) or 0)
                if isinstance(family_load, dict)
                else 0
            ),
            "queued": (
                int((family_load or {}).get("queued", 0) or 0)
                if isinstance(family_load, dict)
                else 0
            ),
            "pending_total": (
                int((family_load or {}).get("pending_total", 0) or 0)
                if isinstance(family_load, dict)
                else 0
            ),
        }

    diagnostic_backlog = request.diagnostic_chain_backlog or {}
    return {
        "available_slots_now": int(selector_config.available_slots),
        "quota_realization": selector_config.quota_realization,
        "recent_realized_modes": _recent_mode_context(request),
        "cost_admission": {
            "deferred_families": sorted(
                family
                for family, penalty in cost_penalties.items()
                if penalty >= selector_config.cost_aware_defer_penalty
            ),
            "demoted_families": sorted(
                family
                for family, penalty in cost_penalties.items()
                if 0 < penalty < selector_config.cost_aware_defer_penalty
            ),
            "capacity_blocked_families": [],
            "capacity_pressure_families": sorted(
                capacity_pressure_families
            ),
            "yield_deferred_families": sorted(
                family
                for family, penalty in cost_penalties.items()
                if penalty >= selector_config.cost_aware_defer_penalty
            ),
            "high_cost_caps": high_cost_caps,
        },
        "score_conversion_backlog": {
            "total_unscored_diagnostic_artifacts": diagnostic_backlog.get(
                "total_unscored_diagnostic_artifacts", 0
            ),
            "pending_chain_candidates": diagnostic_backlog.get(
                "pending_chain_candidates", 0
            ),
            "queued_chain_candidates": diagnostic_backlog.get(
                "queued_chain_candidates", 0
            ),
            "dispatched_chain_candidates": diagnostic_backlog.get(
                "dispatched_chain_candidates", 0
            ),
            "native_or_proxy_pending_refilter": diagnostic_backlog.get(
                "native_or_proxy_pending_refilter", 0
            ),
        },
        "pending_family_load": dict(request.pending_family_load or {}),
        "execution_realization": dict(request.execution_realization or {}),
    }


def run_supervision_phase(
    request: SupervisionPhaseRequest,
    dependencies: SupervisionPhaseDependencies,
    *,
    clock: Callable[[], float] = time.time,
) -> SupervisionPhaseResult:
    """Run, reuse, or deterministically derive one Supervisor decision."""

    selector_context = build_supervisor_selector_context(request)
    hypotheses = list(request.hypotheses)
    candidates = list(request.candidates)
    supervisor_user_payload = (
        build_user_prompt(
            request.evidence,
            hypotheses,
            candidates,
            selector_context=selector_context,
        )
        if hypotheses and candidates
        else ""
    )
    supervisor_prompt_text = supervisor_system_prompt() + "\n" + supervisor_user_payload

    supervisor_started_at = clock()
    if not hypotheses or not candidates:
        supervisor_output = SupervisorOutput(
            valid=False,
            abstain=True,
            confidence=0.0,
            fail_reason="no_hypotheses_or_candidates",
            mode_mixture={},
            candidate_decisions=[],
            rationale="",
            raw_text="",
            usage={},
        )
        supervisor_latency_seconds = 0.0
    elif request.reuse_prior_plan and request.reuse_mode_mixture:
        supervisor_output = SupervisorOutput(
            valid=True,
            abstain=False,
            confidence=request.reuse_confidence,
            fail_reason=None,
            mode_mixture=dict(request.reuse_mode_mixture),
            candidate_decisions=[],
            rationale=(
                "[evidence-skip] reused prior mode_mixture "
                "(signature unchanged)"
            ),
            raw_text="[evidence-skip]",
            usage={},
        )
        supervisor_latency_seconds = 0.0
    elif request.use_unified_reasoner:
        from ..unified_reasoner import build_unified_supervisor_output

        supervisor_output = build_unified_supervisor_output(
            request.planner_output,
            candidates,
            cards=hypotheses,
        )
        supervisor_latency_seconds = clock() - supervisor_started_at
    else:
        supervisor_output = dependencies.call_supervisor(
            request.evidence,
            hypotheses,
            candidates,
            cfg=request.supervisor_config,
            selector_context=selector_context,
        )
        supervisor_latency_seconds = clock() - supervisor_started_at

    llm_call_record = dependencies.build_llm_call_record(
        "supervisor",
        tick_id=request.tick_id,
        cfg_model=request.supervisor_config.model,
        prompt_text=supervisor_prompt_text,
        raw_text=supervisor_output.raw_text,
        valid=supervisor_output.valid,
        fail_reason=supervisor_output.fail_reason,
        confidence=supervisor_output.confidence,
        abstain=supervisor_output.abstain,
        fallback=not supervisor_output.valid or supervisor_output.abstain,
        latency_s=supervisor_latency_seconds,
        usage=supervisor_output.usage,
        prompt_audit=dependencies.build_prompt_audit_snapshot(
            request.evidence,
            role="supervisor",
            selector_context=selector_context,
            hypothesis_count=len(hypotheses),
            candidate_count=len(candidates),
        ),
    )

    if not supervisor_output.valid or supervisor_output.abstain:
        fallback_reason = (
            supervisor_output.fail_reason or "supervisor_invalid"
        )
    elif request.planner_fallback_reason in {
        "planner_invalid",
        "planner_abstain",
        "planner_empty_cards",
    }:
        fallback_reason = request.planner_fallback_reason
    else:
        fallback_reason = None

    return SupervisionPhaseResult(
        supervisor_output=supervisor_output,
        supervisor_latency_seconds=supervisor_latency_seconds,
        fallback_reason=fallback_reason,
        selector_context=selector_context,
        llm_call_record=llm_call_record,
    )


__all__ = [
    "SupervisionPhaseDependencies",
    "SupervisionPhaseRequest",
    "SupervisionPhaseResult",
    "build_supervisor_selector_context",
    "run_supervision_phase",
]
