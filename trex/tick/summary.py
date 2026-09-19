"""Stable, JSON-serializable live-tick output projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..schemas import (
    ActionCandidate,
    EvidenceSummary,
    LaunchDecision,
    PlannerOutput,
    SupervisorOutput,
)


def build_evidence_only_summary(
    *,
    tick_id: str,
    target_id: str,
    evidence: EvidenceSummary,
    all_result_count: int,
    window_result_count: int,
    strict_total: int,
    strict_window: int,
) -> dict[str, Any]:
    """Return the monitoring payload used by evidence-only controller probes."""

    return {
        "tick_id": tick_id,
        "target_id": target_id,
        "evidence": {
            "state_label": evidence.state_label,
            "n_all_results": all_result_count,
            "n_window_results": window_result_count,
            "strict_total": strict_total,
            "strict_window": strict_window,
            "elapsed_wall_h": evidence.elapsed_wall_h,
            "remaining_wall_h": evidence.remaining_wall_h,
            "worker_wall_gpu_h_total": evidence.worker_wall_gpu_h_total,
            "run_su_per_worker_wall_gpu_h_total": (
                evidence.run_su_per_worker_wall_gpu_h_total
            ),
            "run_su_count": evidence.run_su_count,
            "run_su_count_delta": evidence.run_su_count_delta,
            "gpu_h_since_last_su": evidence.gpu_h_since_last_su,
            "top_bin_share": evidence.top_bin_share,
            "duplicate_fraction": evidence.duplicate_fraction,
            "near_miss_count": evidence.near_miss_count,
            "production_panel_status": evidence.production_panel_status,
            "production_panel_n": len(evidence.production_panel_selected_ids),
            "production_panel_value": evidence.production_panel_value,
            "strict_su_tm08_recent_count": evidence.strict_su_tm08_recent_count,
            "strict_su_live_recent_count": evidence.strict_su_live_recent_count,
            "strict_su_tm05_recent_count": evidence.strict_su_tm05_recent_count,
            "strict_su_tm08_live_split_ratio": (
                evidence.strict_su_tm08_live_split_ratio
            ),
            "strict_su_tm08_split_ratio": evidence.strict_su_tm08_split_ratio,
        },
        "evidence_only": True,
    }


@dataclass(frozen=True)
class LiveTickSummaryRequest:
    """All values exposed in the stable user-facing full-tick summary."""

    tick_id: str
    target_id: str
    evidence: EvidenceSummary
    all_result_count: int
    window_result_count: int
    strict_total: int
    strict_window: int
    critic_enabled: bool
    critic_flags: Sequence[str]
    planner_output: PlannerOutput
    planner_latency_seconds: float
    candidates: Sequence[ActionCandidate]
    supervisor_output: SupervisorOutput
    supervisor_latency_seconds: float
    selector_debug: dict[str, Any]
    launches: Sequence[LaunchDecision]
    lifecycle_updates: Sequence[dict[str, Any]]
    fallback_reason: str | None


def build_live_tick_summary(request: LiveTickSummaryRequest) -> dict[str, Any]:
    """Return the stable full-tick CLI/controller payload."""

    evidence = request.evidence
    candidates = request.candidates
    supervisor_output = request.supervisor_output
    planner_output = request.planner_output
    selector_debug = request.selector_debug
    launches = request.launches
    return {
        "tick_id": request.tick_id,
        "target_id": request.target_id,
        "evidence": {
            "state_label": evidence.state_label,
            "n_all_results": request.all_result_count,
            "n_window_results": request.window_result_count,
            "strict_total": request.strict_total,
            "strict_window": request.strict_window,
            "n_recipes": len(evidence.recipes),
            "recipes_by_class": {
                recipe_class: sum(
                    1
                    for recipe in evidence.recipes
                    if recipe.recipe_class == recipe_class
                )
                for recipe_class in (
                    "strict_success",
                    "panel_ready",
                    "near_miss",
                    "joint_fail",
                )
            },
            "recent_fallback_high": evidence.recent_fallback_high,
            "axis_pass_counts": {
                axis: stats.pass_count
                for axis, stats in evidence.axis_stats.items()
            },
            "production_panel_status": evidence.production_panel_status,
            "production_panel_n": len(evidence.production_panel_selected_ids),
            "production_panel_value": evidence.production_panel_value,
        },
        "critic": {
            "enabled": request.critic_enabled,
            "n_flags": len(request.critic_flags),
            "flags": list(request.critic_flags[:5]),
        },
        "planner": {
            "valid": planner_output.valid,
            "confidence": planner_output.confidence,
            "abstain": planner_output.abstain,
            "fail_reason": planner_output.fail_reason,
            "n_cards": len(planner_output.cards),
            "latency_s": round(request.planner_latency_seconds, 2),
        },
        "candidates": {
            "n_total": len(candidates),
            "n_feasible": sum(
                1 for candidate in candidates if candidate.feasibility.all_ok()
            ),
            "by_family": {
                family: sum(
                    1
                    for candidate in candidates
                    if candidate.method_family == family
                )
                for family in {
                    candidate.method_family for candidate in candidates
                }
            },
        },
        "supervisor": {
            "valid": supervisor_output.valid,
            "confidence": supervisor_output.confidence,
            "mode_mixture": supervisor_output.mode_mixture,
            "n_decisions": len(supervisor_output.candidate_decisions),
            "latency_s": round(request.supervisor_latency_seconds, 2),
        },
        "selector": {
            "source": selector_debug["source"],
            "raw_mixture": selector_debug["raw_mixture"],
            "clamped_mixture": selector_debug["clamped_mixture"],
            "clamp_log": selector_debug["clamp_log"],
            "quotas": selector_debug["quotas_final"],
            "n_launched": sum(
                1 for launch in launches if launch.status == "launched"
            ),
            "n_rejected": sum(
                1 for launch in launches if launch.status == "rejected"
            ),
        },
        "lifecycle_updates": list(request.lifecycle_updates),
        "fallback_used": request.fallback_reason is not None,
        "fallback_reason": request.fallback_reason,
    }


__all__ = [
    "LiveTickSummaryRequest",
    "build_evidence_only_summary",
    "build_live_tick_summary",
]
