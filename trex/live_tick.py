"""Run a planning cycle and append its evidence, hypotheses, candidates, and decisions.

Also update hypothesis status from accumulated outcomes. Worker dispatch is handled
separately by the controller.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .archive import Archive
from .output_identity import prepare_archive_results
from .cross_campaign_memory import cross_campaign_memory_prompt_audit
from .candidate_builder import WARMSTART_FAMILIES
from .capability_registry import CapabilityRegistry, default_registry
from .evidence_reducer import ReducerConfig
from .planner import call_planner, diagnostic_driver_tldr, evidence_tldr
from .provenance import model_digest_from_env
from .refilter_roles import CANONICAL_SCORE_CONVERSION
from .score_conversion import (
    is_score_convertible_diagnostic_record,
    native_strict_like_pending_score_conversion,
    proxy_promising_pending_score_conversion,
)
from .schemas import (
    ActionCandidate,
    DispatchRecord,
    EvidenceSummary,
    HypothesisCard,
    LLMCallRecord,
    LLMHealthSummary,
    LaunchDecision,
    ResultRecord,
    RouteHealthSummary,
    SupervisorDecision,
    SupervisorOutput,
    TargetConstraint,
    to_jsonable,
)
from .success_criteria import (
    NEAR_PASS_MARGINS,
    STRICT_SUCCESS,
    is_near_miss,
    is_strict_success,
)
from .supervisor import call_supervisor
from .tick.config import (
    FoldseekConfig,
    LiveTickConfig,
    SequenceDedupConfig,
    SkipConfig,
)
from .tick import (
    CandidatePhaseRequest,
    EvidenceAssemblyRequest,
    LifecyclePhaseRequest,
    LiveTickSummaryRequest,
    PrePlannerLifecycleRequest,
    ProposalPhaseDependencies,
    ProposalPhaseRequest,
    ResultClusteringRequest,
    SelectionPhaseRequest,
    SupervisionPhaseDependencies,
    SupervisionPhaseRequest,
    assemble_tick_evidence,
    build_candidate_phase,
    build_evidence_only_summary,
    build_live_tick_summary,
    build_recent_tick_trajectory,
    cluster_evidence_results,
    index_actions_by_spawned_result as _index_actions_by_spawned_result,
    load_tick_archive_snapshot,
    recent_fallback_rate,
    retire_expired_hypotheses,
    resolve_warmstart_coverage,
    run_proposal_phase,
    run_supervision_phase,
    select_live_tick_launches,
    summarize_llm_health,
    update_lifecycle_phase,
)


from . import SCHEMA_VERSION

BOOTSTRAP_SEED_ACTION_FAMILIES = [
    # Keep this family-level prompt payload synchronized with
    # candidate_builder.WARMSTART_FAMILIES. These are scheduling seeds only:
    # they tell the Planner which deterministic cold-start arms were injected,
    # not which family should win after evidence arrives.
    "complexa_beam",
    "boltzgen",
    "bindcraft",
]


def _charged_gpu_count_from_env() -> tuple[float | None, str]:
    """Reserved GPU count retained as raw run-level audit metadata."""
    raw = os.environ.get("TREX_CHARGED_GPUS")
    if raw:
        try:
            n = float(raw)
            if n > 0:
                return n, "env:TREX_CHARGED_GPUS"
        except ValueError:
            pass
    raw = os.environ.get("SLURM_GPUS_ON_NODE")
    if raw:
        try:
            n = float(raw.split(",", 1)[0])
            if n > 0:
                return n, "env:SLURM_GPUS_ON_NODE"
        except ValueError:
            pass
    raw = os.environ.get("SLURM_JOB_GPUS")
    if raw:
        toks = [t for t in raw.replace(",", " ").split() if t]
        if toks:
            return float(len(toks)), "env:SLURM_JOB_GPUS"
    return None, "unavailable"


def _compact_prompt_audit_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "<truncated>"
    value = to_jsonable(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, key in enumerate(sorted(value, key=str)):
            if idx >= 24:
                out["_truncated_keys"] = len(value) - idx
                break
            out[str(key)] = _compact_prompt_audit_value(value[key], depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_compact_prompt_audit_value(v, depth=depth + 1) for v in value[:24]]
    if isinstance(value, str) and len(value) > 5000:
        return value[:5000] + "<truncated>"
    return value


def _compact_prompt_audit_for_archive(prompt_audit: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(prompt_audit, dict):
        return {}
    return _compact_prompt_audit_value(prompt_audit)


def _prompt_audit_snapshot(
    evidence: EvidenceSummary,
    *,
    role: str,
    selector_context: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Bounded snapshot of the prompt headers that drive decisions.

    Full prompt archiving would add tens of thousands of tokens per tick to the
    JSONL archive. This compact audit preserves the evidence digest, diagnostic
    TL;DR, objective values, and optional selector-context summary needed for
    later forensics without turning prompt logging into a controller bottleneck.
    """
    try:
        target_dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        target_dry_gpu_h = 0.0

    def _route_value_prompt_rate(row: Any) -> tuple[Any, str, Any]:
        lifetime = getattr(row, "new_su_per_route_gpu_h", None)
        value = getattr(row, "gpu_recent_new_su_per_route_gpu_h", None)
        if value is not None:
            return value, "gpu_recent", lifetime
        value = getattr(row, "medium_recent_new_su_per_route_gpu_h", None)
        if value is not None:
            return value, "medium_recent", lifetime
        role = str(getattr(row, "route_role", "") or "")
        try:
            canonical_refilter_gpu_h = float(getattr(row, "canonical_refilter_gpu_h", 0.0) or 0.0)
        except (TypeError, ValueError):
            canonical_refilter_gpu_h = 0.0
        record_rate = getattr(row, "record_recent_new_su_per_route_gpu_h", None)
        if record_rate is None:
            record_rate = getattr(row, "recent_new_su_per_route_gpu_h", None)
        if (
            record_rate is not None
            and canonical_refilter_gpu_h <= 0.0
            and "score_conversion" not in role
            and target_dry_gpu_h < 12.0
        ):
            return record_rate, "record_recent", lifetime
        return None, "lifetime_memory", lifetime

    route_diag = []
    for row in getattr(evidence, "route_values", None) or []:
        score = float(getattr(row, "diagnostic_improvement_score", 0.0) or 0.0)
        if score <= 0.0:
            continue
        value_rate, value_rate_source, lifetime_rate = _route_value_prompt_rate(row)
        route_diag.append({
            "strategy_key": str(getattr(row, "strategy_key", ""))[:120],
            "family": getattr(row, "family", None),
            "scope": getattr(row, "scope", None),
            "new_su": getattr(row, "new_su", 0),
            "value_rate": value_rate,
            "value_rate_source": value_rate_source,
            "lifetime_su_per_route_gpu_h": lifetime_rate,
            "diagnostic_improvement_score": round(score, 3),
            "diagnostic_improvement_axes": list(getattr(row, "diagnostic_improvement_axes", []) or [])[:3],
        })
    route_diag.sort(key=lambda x: (-float(x.get("diagnostic_improvement_score") or 0.0), str(x.get("strategy_key") or "")))

    audit: dict[str, Any] = {
        "role": role,
        "state_label": str(evidence.state_label),
        "evidence_tldr": evidence_tldr(evidence),
        "diagnostic_driver_tldr": getattr(evidence, "diagnostic_driver_tldr", "none") or "none",
        "route_diagnostic_support": route_diag[:6],
        "run_su_count": evidence.run_su_count,
        "run_su_count_delta": evidence.run_su_count_delta,
        "run_su_hwm": getattr(evidence, "run_su_hwm", None),
        "run_su_hwm_delta": getattr(evidence, "run_su_hwm_delta", None),
        "su_per_gpu_h_recent": evidence.su_per_gpu_h_recent,
        "worker_gpu_h_total": evidence.worker_gpu_h_total,
        "worker_wall_gpu_h_total": getattr(evidence, "worker_wall_gpu_h_total", None),
        "run_su_per_worker_wall_gpu_h_total": getattr(evidence, "run_su_per_worker_wall_gpu_h_total", None),
        "gpu_h_since_last_su": getattr(evidence, "gpu_h_since_last_su", None),
        "near_miss_count": evidence.near_miss_count,
        "duplicate_fraction": evidence.duplicate_fraction,
        "top_bin_share": evidence.top_bin_share,
        "strict_duplicate_collapse_signal": getattr(evidence, "strict_duplicate_collapse_signal", False),
    }
    if selector_context is not None:
        audit["selector_context"] = {
            k: selector_context.get(k)
            for k in (
                "available_slots_now",
                "quota_realization",
                "recent_realized_modes",
                "cost_admission",
                "score_conversion_backlog",
                "execution_realization",
            )
            if k in selector_context
        }
    memory_audit = cross_campaign_memory_prompt_audit()
    if memory_audit is not None:
        audit["cross_campaign_memory"] = memory_audit
    audit.update(extra)
    return _compact_prompt_audit_for_archive(audit)

def _llm_call_record(
    role: str,
    *,
    tick_id: str,
    cfg_model: str,
    prompt_text: str,
    raw_text: str,
    valid: bool,
    fail_reason: str | None,
    confidence: float | None,
    abstain: bool,
    fallback: bool,
    latency_s: float,
    usage: dict,
    prompt_audit: dict[str, Any] | None = None,
) -> LLMCallRecord:
    # Hash the sent system and user prompts for reproducibility.
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
    # The T-REX LLM adapter normalizes provider usage as input_tokens/output_tokens
    # (prefill/generation), while some test fakes and OpenAI raw payloads use
    # prompt_tokens/completion_tokens. Accept both so appendix token accounting
    # does not silently zero out on vLLM/OpenAI-compatible clients.
    tokens_in = int(
        usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    )
    tokens_out = int(
        usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    )
    return LLMCallRecord(
        call_id=f"llm_{tick_id}_{role}_{int(time.time())}",
        tick_id=tick_id,
        role=role,  # type: ignore[arg-type]
        model=cfg_model,
        model_digest=model_digest_from_env(),
        prompt_hash=prompt_hash,
        schema_version=SCHEMA_VERSION,
        latency_s=round(latency_s, 3),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        parse_status="ok" if valid else (fail_reason.split(":")[0] if fail_reason else "parse_fail"),  # type: ignore[arg-type]
        confidence=confidence,
        abstain=abstain,
        fallback_triggered=fallback,
        fail_reason=fail_reason,
        prompt_audit=_compact_prompt_audit_for_archive(prompt_audit),
    )


def _recent_fallback_rate(archive: Archive, window: int = 4) -> float:
    """Compatibility wrapper for callers that still pass an ``Archive``."""

    return recent_fallback_rate(
        tuple(archive.iter_records(LLMCallRecord)),
        window=window,
    )


def _archive_llm_health(
    archive: Archive,
    model: str,
    *,
    window: int = 12,
) -> LLMHealthSummary:
    """Compatibility wrapper for callers that still pass an ``Archive``."""

    return summarize_llm_health(
        tuple(archive.iter_records(LLMCallRecord)),
        model,
        window=window,
    )


def _compact_selector_debug_for_archive(sel_debug: dict[str, Any] | None) -> dict[str, Any]:
    # Persist only the launch-alignment fields needed for forensics and LLM feedback.
    if not isinstance(sel_debug, dict):
        return {}
    keys = (
        "source", "quota_realization", "raw_mixture", "clamped_mixture",
        "global_priority_used", "global_priority_order",
        "global_priority_backfill_ids", "global_priority_not_launched",
        "quotas_raw", "quotas_final", "mode_credit_before",
        "mode_credit_after_quota", "mode_credit_cap",
        "redistribute_log", "clamp_log",
        "forced_repeated_support_probe", "forced_near_miss_rescue_floor",
        "forced_cross_family_escape_floor",
        "effective_mode_window_k", "rank1_not_launched",
        "capacity_blocked_families", "capacity_block_reasons",
        "capacity_pressure_families", "capacity_pressure_reasons",
        "n_capacity_pressure_candidates", "capacity_deferred_candidate_ids",
        "cost_deferred_families", "n_cost_deferred_candidates",
        "route_deferred_candidate_ids", "n_route_deferred_candidates",
        "selected_candidate_ids", "launch_modes",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k not in sel_debug:
            continue
        v = sel_debug[k]
        if k in {
            "rank1_not_launched", "global_priority_order",
            "global_priority_backfill_ids", "global_priority_not_launched",
            "route_deferred_candidate_ids",
            "selected_candidate_ids", "redistribute_log", "clamp_log",
        } and isinstance(v, list):
            out[k] = v[:12]
        else:
            out[k] = v
    return out


def _compact_selector_context_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "<truncated>"
    value = to_jsonable(value)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, key in enumerate(sorted(value, key=str)):
            if idx >= 24:
                out["_truncated_keys"] = len(value) - idx
                break
            out[str(key)] = _compact_selector_context_value(value[key], depth=depth + 1)
        return out
    if isinstance(value, list):
        return [_compact_selector_context_value(v, depth=depth + 1) for v in value[:24]]
    return value


def _compact_selector_context_for_archive(selector_context: dict[str, Any] | None) -> dict[str, Any]:
    """Persist the read-only Selector context shown to the Supervisor.

    This is a bounded forensic copy, not another decision input. It lets later
    audits reproduce whether the LLM saw capacity pressure, score-conversion
    backlog, recent realized modes, and proposed->selected->started gaps.
    """
    if not isinstance(selector_context, dict):
        return {}
    keys = (
        "available_slots_now",
        "quota_realization",
        "recent_realized_modes",
        "cost_admission",
        "score_conversion_backlog",
        "pending_family_load",
        "execution_realization",
    )
    return {
        k: _compact_selector_context_value(selector_context[k])
        for k in keys
        if k in selector_context
    }


def _supervisor_decision_record(
    sup_out: SupervisorOutput, *, tick_id: str, fallback_used: bool, clamps: list[str],
    selector_debug: dict[str, Any] | None = None,
    selector_context: dict[str, Any] | None = None,
) -> SupervisorDecision:
    mode_mixture = dict(sup_out.mode_mixture)
    if not mode_mixture and isinstance(selector_debug, dict):
        raw_mixture = selector_debug.get("raw_mixture")
        if isinstance(raw_mixture, dict):
            mode_mixture = {
                str(k): float(v)
                for k, v in raw_mixture.items()
                if k in {"exploit", "rescue", "explore"} and isinstance(v, (int, float))
            }
    return SupervisorDecision(
        tick_id=tick_id,
        mode_mixture=mode_mixture,
        candidate_decisions=list(sup_out.candidate_decisions),
        clamps_applied=clamps,
        fallback_used=fallback_used,
        rationale=sup_out.rationale[:1000],
        confidence=sup_out.confidence,
        selector_debug=_compact_selector_debug_for_archive(selector_debug),
        selector_context=_compact_selector_context_for_archive(selector_context),
    )


def _decision_signature(ev: EvidenceSummary) -> tuple:
    """Fingerprint decision-relevant evidence for optional LLM-call reuse.

    Include recent productivity, near misses, duplication, and remaining-time bands.
    Bucket noisy values while excluding irrelevant monotonic counters.
    """
    mh = ev.method_health or {}
    fam = tuple(sorted(
        (
            k,
            mh[k].strict_yield_su,
            mh[k].chained_strict_yield_su,
            round((mh[k].su_per_gpu_h or 0.0), 3),
            round((mh[k].chained_su_per_gpu_h or 0.0), 3),
            # Recent route evidence must invalidate reuse when productivity changes.
            round((mh[k].su_per_gpu_h_recent or 0.0), 3),
            mh[k].near_miss_yield_recent,
            round((mh[k].chained_su_per_gpu_h_recent or 0.0), 3),
            mh[k].chained_strict_yield_su_recent,
        )
        for k in mh
    ))

    def _b1(x: float | None) -> float:
        return round(x, 1) if x is not None else -1.0

    def _n(x: int | None) -> int:
        return x if x is not None else -1

    backlog = getattr(ev, "diagnostic_chain_backlog", None) or {}
    backlog_by_family = tuple(sorted(
        (
            fam,
            int(v.get("unscored_artifacts", 0)),
            int(v.get("pending_chain_candidates", 0)),
            int(v.get("queued_chain_candidates", 0)),
            int(v.get("dispatched_chain_candidates", 0)),
            int(v.get("native_strict_like_pending_refilter", 0)),
        )
        for fam, v in (backlog.get("by_family") or {}).items()
        if isinstance(v, dict)
    ))
    refilter_roles = tuple(sorted(
        (
            role,
            int(v.get("attempts", 0)),
            int(v.get("strict_yield_su", 0)),
            round(float(v.get("gpu_h", 0.0)), 3),
            tuple(sorted((v.get("source_families") or {}).items())),
        )
        for role, v in (getattr(ev, "refilter_role_health", None) or {}).items()
        if isinstance(v, dict)
    ))

    def _rv(row, key, default=None):
        return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)

    try:
        target_dry_gpu_h = float(getattr(ev, "gpu_h_since_last_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        target_dry_gpu_h = 0.0

    def _rv_prompt_rate(row) -> tuple[float, str]:
        gpu_rate = _rv(row, "gpu_recent_new_su_per_route_gpu_h", None)
        if gpu_rate is not None:
            return float(gpu_rate or 0.0), "gpu_recent"
        medium_rate = _rv(row, "medium_recent_new_su_per_route_gpu_h", None)
        if medium_rate is not None:
            return float(medium_rate or 0.0), "medium_recent"
        role = str(_rv(row, "route_role", "") or "")
        try:
            canonical_refilter_gpu_h = float(_rv(row, "canonical_refilter_gpu_h", 0.0) or 0.0)
        except (TypeError, ValueError):
            canonical_refilter_gpu_h = 0.0
        record_rate = _rv(row, "record_recent_new_su_per_route_gpu_h", _rv(row, "recent_new_su_per_route_gpu_h", None))
        if (
            record_rate is not None
            and canonical_refilter_gpu_h <= 0.0
            and "score_conversion" not in role
            and target_dry_gpu_h < 12.0
        ):
            return float(record_rate or 0.0), "record_recent"
        return 0.0, "lifetime_memory"

    route_values = tuple(sorted(
        (
            str(_rv(r, "strategy_key", ""))[:120],
            str(_rv(r, "status", "")),
            str(_rv(r, "route_role", "") or ""),
            str(_rv(r, "refilter_role", "") or ""),
            int(_rv(r, "record_recent_new_su", _rv(r, "new_su_recent", 0)) or 0),
            round(_rv_prompt_rate(r)[0], 3),
            _rv_prompt_rate(r)[1],
            round(float(_rv(r, "new_su_per_route_gpu_h", 0.0) or 0.0), 3),
            round(float(_rv(r, "route_gpu_h", 0.0) or 0.0), 2),
            round(float(_rv(r, "strict_per_su", 0.0) or 0.0), 2),
            str(_rv(r, "marginal_status", "") or ""),
            int(_rv(r, "pending_score_conversion_count", 0) or 0),
            round(float(_rv(r, "diagnostic_improvement_score", 0.0) or 0.0), 2),
            tuple(str(x)[:80] for x in (_rv(r, "diagnostic_improvement_axes", []) or [])[:3]),
        )
        for r in (getattr(ev, "route_values", None) or [])
        if _rv(r, "scope") == "route"
        and (
            _rv(r, "status") in {"promote", "awaiting_score_conversion", "defer", "collapse_risk"}
            or float(_rv(r, "diagnostic_improvement_score", 0.0) or 0.0) > 0.0
        )
    ))
    family_diagnostic_support = tuple(sorted(
        (
            str(_rv(r, "family", ""))[:80],
            str(_rv(r, "marginal_status", "") or ""),
            round(float(_rv(r, "diagnostic_improvement_score", 0.0) or 0.0), 2),
            tuple(str(x)[:80] for x in (_rv(r, "diagnostic_improvement_axes", []) or [])[:3]),
        )
        for r in (getattr(ev, "route_values", None) or [])
        if _rv(r, "scope") == "family"
        and float(_rv(r, "diagnostic_improvement_score", 0.0) or 0.0) > 0.0
    ))
    diagnostic_stats = tuple(sorted(
        (
            str(axis),
            int(getattr(stat, "n", 0) or 0),
            int(getattr(stat, "fail_count", 0) or 0),
            int(getattr(stat, "near_pass_count", 0) or 0),
            round(float(getattr(stat, "median_raw", 0.0) or 0.0), 3),
            round(float(getattr(stat, "median_deficit", 0.0) or 0.0), 3),
        )
        for axis, stat in (getattr(ev, "diagnostic_axis_stats", None) or {}).items()
    ))
    diagnostic_tldr_sig = str(
        getattr(ev, "diagnostic_driver_tldr", None) or diagnostic_driver_tldr(ev)
    )[:1000]
    wall_band = int(max(0.0, ev.remaining_wall_h) // 3.0)
    return (
        ev.state_label,
        ev.run_su_count,
        # Include new-SU and recent-rate signals in the reuse fingerprint.
        ev.run_su_count_delta,
        _b1(ev.su_per_gpu_h_recent),
        ev.global_new_strict,
        ev.near_miss_count,
        _b1(ev.top_bin_share),
        _b1(ev.duplicate_fraction),
        _n(getattr(ev, "strict_su_tm08_recent_count", None)),
        _n(getattr(ev, "strict_su_tm05_recent_count", None)),
        _b1(getattr(ev, "strict_su_tm08_split_ratio", None)),
        ev.sequence_dedup_status,
        _b1(ev.sequence_dedup_coverage),
        _n(ev.seq_unique_strict_count),
        _n(ev.joint_struct_seq_unique_count),
        _b1(ev.seq_duplicate_fraction),
        _b1(ev.top_seq_bin_share),
        ev.production_panel_status,
        len(ev.production_panel_selected_ids),
        tuple(ev.production_panel_selected_ids[:8]),
        _b1(ev.production_panel_value),
        int(backlog.get("total_unscored_diagnostic_artifacts", 0) or 0),
        int(backlog.get("pending_chain_candidates", 0) or 0),
        int(backlog.get("queued_chain_candidates", 0) or 0),
        int(backlog.get("dispatched_chain_candidates", 0) or 0),
        backlog_by_family,
        refilter_roles,
        route_values,
        family_diagnostic_support,
        diagnostic_tldr_sig,
        diagnostic_stats,
        wall_band,
        fam,
    )

_ERE_MODES = {"exploit", "rescue", "explore"}


def _recent_realized_modes(
    prior_launches: list[LaunchDecision],
    prior_dispatches: list[DispatchRecord] | None = None,
) -> list[str]:
    """Realized non-system E/R/E modes for quota realization.

    LaunchDecision is selector intent. In the event-driven controller a selected
    candidate can be stale-dropped before any worker starts, so K-window inertia
    must be based on DispatchRecord(status="started"). Fall back to selected
    LaunchDecision rows only for legacy archives with no started dispatch audit.
    """
    by_launch_id = {L.launch_id: L for L in prior_launches}
    latest_by_candidate: dict[str, LaunchDecision] = {}
    for L in prior_launches:
        latest_by_candidate[L.candidate_id] = L

    if prior_dispatches is not None:
        out: list[str] = []
        saw_started = False
        for d in prior_dispatches:
            if d.status != "started":
                continue
            saw_started = True
            launch = by_launch_id.get(d.launch_id or "") or latest_by_candidate.get(d.candidate_id)
            if launch is None:
                continue
            mode = (launch.resource_class_concrete or {}).get("mode")
            if d.candidate_id.startswith("chain_") or mode == "chain_refilter":
                continue
            if mode in _ERE_MODES:
                out.append(mode)
        if saw_started:
            return out

    return [
        m
        for L in prior_launches
        if not L.candidate_id.startswith("chain_")
        and (m := (L.resource_class_concrete or {}).get("mode"))
        and m in _ERE_MODES
    ]


def _mode_credit_from_history(
    prior_supervisors: list[SupervisorDecision],
    prior_launches: list[LaunchDecision],
    prior_dispatches: list[DispatchRecord],
    *,
    cap: float = 3.0,
) -> dict[str, float]:
    """Reconstruct fractional E/R/E credit from durable archive records.

    Each SupervisorDecision contributes its clamped mixture times the number of
    non-system candidates selected on that tick. Only actual worker-started
    DispatchRecords spend credit. Therefore a candidate that was selected but
    later stale-dropped or never dispatched does not falsely satisfy the
    Supervisor mixture.
    """
    modes = ("exploit", "rescue", "explore")
    credit = {m: 0.0 for m in modes}
    try:
        cap_f = max(0.0, float(cap or 0.0))
    except (TypeError, ValueError):
        cap_f = 3.0

    def _bounded() -> None:
        if cap_f <= 0.0:
            return
        for m in modes:
            credit[m] = max(-cap_f, min(cap_f, credit[m]))

    by_launch_id = {L.launch_id: L for L in prior_launches}
    latest_by_candidate: dict[str, LaunchDecision] = {}
    selected_by_tick: dict[str, list[str]] = {}
    for L in prior_launches:
        latest_by_candidate[L.candidate_id] = L
        if L.status != "launched" or L.candidate_id.startswith("chain_"):
            continue
        mode = (L.resource_class_concrete or {}).get("mode")
        if mode in _ERE_MODES:
            selected_by_tick.setdefault(L.tick_id, []).append(str(mode))

    started_by_tick: dict[str, list[str]] = {}
    seen_dispatch_ids: set[str] = set()
    for d in prior_dispatches:
        if d.status != "started":
            continue
        dispatch_key = d.dispatch_id or f"{d.tick_id}:{d.candidate_id}:{len(seen_dispatch_ids)}"
        if dispatch_key in seen_dispatch_ids:
            continue
        seen_dispatch_ids.add(dispatch_key)
        launch = by_launch_id.get(d.launch_id or "") or latest_by_candidate.get(d.candidate_id)
        if launch is None or d.candidate_id.startswith("chain_"):
            continue
        mode = (launch.resource_class_concrete or {}).get("mode")
        if mode in _ERE_MODES:
            started_by_tick.setdefault(launch.tick_id, []).append(str(mode))

    for sd in prior_supervisors:
        dbg = sd.selector_debug or {}
        mix = dbg.get("clamped_mixture") if isinstance(dbg, dict) else None
        if not isinstance(mix, dict) or not mix:
            mix = sd.mode_mixture or {}
        n_slots = max(
            len(selected_by_tick.get(sd.tick_id, [])),
            len(started_by_tick.get(sd.tick_id, [])),
        )
        if n_slots <= 0:
            continue
        for m in modes:
            try:
                credit[m] += max(0.0, float(mix.get(m, 0.0) or 0.0)) * n_slots
            except (TypeError, ValueError):
                pass
        for m in started_by_tick.get(sd.tick_id, []):
            credit[m] -= 1.0
        _bounded()
    return {m: round(credit[m], 6) for m in modes}


def _candidate_family(cand: ActionCandidate | None, dispatch: DispatchRecord | None = None) -> str:
    """Best available family label for realization/provenance summaries."""
    if cand is not None and cand.method_family:
        return cand.method_family
    fam = getattr(dispatch, "method_family", None) if dispatch is not None else None
    return str(fam or "unknown")


def _recent_started_families(
    actions: list[ActionCandidate],
    dispatches: list[DispatchRecord],
) -> list[str]:
    """Actual worker-started scientific families, chronological."""
    by_id = {c.candidate_id: c for c in actions}
    out: list[str] = []
    for d in dispatches:
        if d.status != "started" or d.candidate_id.startswith("chain_"):
            continue
        fam = _candidate_family(by_id.get(d.candidate_id), d)
        if fam != "unknown":
            out.append(fam)
    return out


def _dispatch_realization_summary(
    actions: list[ActionCandidate],
    launches: list[LaunchDecision],
    dispatches: list[DispatchRecord],
    *,
    recent_started_window_k: int,
) -> dict[str, Any]:
    """Compact selected-vs-started-vs-stale audit for E/R/E realization.

    `recent_started_window_k` controls only how many recent started modes/families
    are shown in this provenance block. It is not the selector E/R/E quota
    realization window, which may be disabled or use fractional carry.
    """
    recent_started_window_k = max(1, int(recent_started_window_k))
    by_action = {c.candidate_id: c for c in actions}
    by_launch_id = {L.launch_id: L for L in launches}
    latest_launch_by_cid: dict[str, LaunchDecision] = {}
    for L in launches:
        latest_launch_by_cid[L.candidate_id] = L

    def _bump(table: dict[str, int], key: str | None) -> None:
        if key:
            table[key] = table.get(key, 0) + 1

    selected_by_mode = {m: 0 for m in sorted(_ERE_MODES)}
    selected_by_family: dict[str, int] = {}
    for L in launches:
        if L.status != "launched" or L.candidate_id.startswith("chain_"):
            continue
        mode = (L.resource_class_concrete or {}).get("mode")
        if mode in _ERE_MODES:
            selected_by_mode[mode] += 1
            cand = by_action.get(L.candidate_id)
            _bump(selected_by_family, _candidate_family(cand))

    started_by_mode = {m: 0 for m in sorted(_ERE_MODES)}
    started_by_family: dict[str, int] = {}
    stale_by_mode = {m: 0 for m in sorted(_ERE_MODES)}
    stale_by_family: dict[str, int] = {}
    dispatch_deferred_by_mode = {m: 0 for m in sorted(_ERE_MODES)}
    dispatch_deferred_by_family: dict[str, int] = {}
    dispatch_defer_reason_counts: dict[str, int] = {}
    started_ids: set[str] = set()
    recent_started_modes: list[str] = []
    recent_started_families: list[str] = []
    for d in dispatches:
        launch = by_launch_id.get(d.launch_id or "") or latest_launch_by_cid.get(d.candidate_id)
        mode = (launch.resource_class_concrete or {}).get("mode") if launch else None
        cand = by_action.get(d.candidate_id)
        family = _candidate_family(cand, d)
        is_system = d.candidate_id.startswith("chain_") or mode == "chain_refilter"
        if d.status == "started" and not is_system and mode in _ERE_MODES:
            started_ids.add(d.candidate_id)
            started_by_mode[mode] += 1
            _bump(started_by_family, family)
            recent_started_modes.append(mode)
            recent_started_families.append(family)
        elif d.status == "dispatch_failed" and not is_system and mode in _ERE_MODES:
            why = d.why or "dispatch_failed"
            reason = why.split(":", 1)[0]
            _bump(dispatch_defer_reason_counts, reason)
            if "stale scientific prefetch" in why:
                stale_by_mode[mode] += 1
                _bump(stale_by_family, family)
            elif why.startswith("high_cost_inflight_cap"):
                dispatch_deferred_by_mode[mode] += 1
                _bump(dispatch_deferred_by_family, family)

    selected_not_started_by_mode = {m: 0 for m in sorted(_ERE_MODES)}
    selected_not_started_by_family: dict[str, int] = {}
    for L in launches:
        if L.status != "launched" or L.candidate_id.startswith("chain_"):
            continue
        mode = (L.resource_class_concrete or {}).get("mode")
        if mode not in _ERE_MODES or L.candidate_id in started_ids:
            continue
        selected_not_started_by_mode[mode] += 1
        cand = by_action.get(L.candidate_id)
        _bump(selected_not_started_by_family, _candidate_family(cand))

    return {
        "recent_started_window_k": recent_started_window_k,
        "selected_by_mode": selected_by_mode,
        "started_by_mode": started_by_mode,
        "stale_dropped_by_mode": stale_by_mode,
        "dispatch_deferred_by_mode": dispatch_deferred_by_mode,
        "selected_not_started_by_mode": selected_not_started_by_mode,
        "selected_by_family": dict(sorted(selected_by_family.items())),
        "started_by_family": dict(sorted(started_by_family.items())),
        "stale_dropped_by_family": dict(sorted(stale_by_family.items())),
        "dispatch_deferred_by_family": dict(sorted(dispatch_deferred_by_family.items())),
        "selected_not_started_by_family": dict(sorted(selected_not_started_by_family.items())),
        "dispatch_defer_reason_counts": dict(sorted(dispatch_defer_reason_counts.items())),
        "recent_started_modes": recent_started_modes[-recent_started_window_k:],
        "recent_started_families": recent_started_families[-recent_started_window_k:],
    }


def _execution_realization_summary(
    actions: list[ActionCandidate],
    launches: list[LaunchDecision],
    dispatches: list[DispatchRecord],
    supervisors: list[SupervisorDecision],
) -> dict[str, Any]:
    """Compact proposal, selection, and worker-start audit by family."""
    by_action = {c.candidate_id: c for c in actions}
    by_launch_id = {L.launch_id: L for L in launches}
    latest_launch_by_cid: dict[str, LaunchDecision] = {}
    for L in launches:
        latest_launch_by_cid[L.candidate_id] = L
    rows: dict[str, dict[str, Any]] = {}
    def _row(fam: str) -> dict[str, Any]:
        return rows.setdefault(str(fam or "unknown"), {"proposed": 0, "selected": 0, "started": 0, "dispatch_deferred": 0, "selected_not_started": 0, "defer_reasons": {}})
    for S in supervisors:
        for d in S.candidate_decisions or []:
            cand = by_action.get(d.candidate_id)
            _row(_candidate_family(cand))["proposed"] += 1
    started_ids: set[str] = set()
    for L in launches:
        if (not L.status == "launched") or L.candidate_id.startswith("chain_"):
            continue
        cand = by_action.get(L.candidate_id)
        _row(_candidate_family(cand))["selected"] += 1
    for L in launches:
        if L.status != "rejected" or L.candidate_id.startswith("chain_"):
            continue
        mode = (L.resource_class_concrete or {}).get("mode")
        if mode == "chain_refilter":
            continue
        cand = by_action.get(L.candidate_id)
        fam = _candidate_family(cand)
        reason = (L.why or L.status or "rejected").split(":", 1)[0]
        r = _row(fam)
        reasons = r.setdefault("defer_reasons", {})
        reasons[reason] = reasons.get(reason, 0) + 1
    for d in dispatches:
        launch = by_launch_id.get(d.launch_id or "") or latest_launch_by_cid.get(d.candidate_id)
        mode = (launch.resource_class_concrete or {}).get("mode") if launch else None
        if d.candidate_id.startswith("chain_") or mode == "chain_refilter":
            continue
        cand = by_action.get(d.candidate_id)
        fam = _candidate_family(cand, d)
        if d.status == "started":
            started_ids.add(d.candidate_id)
            _row(fam)["started"] += 1
        elif d.status == "dispatch_failed" and (d.why or "").startswith("high_cost_inflight_cap"):
            r = _row(fam)
            r["dispatch_deferred"] += 1
            reason = (d.why or "dispatch_failed").split(":", 1)[0]
            reasons = r.setdefault("defer_reasons", {})
            reasons[reason] = reasons.get(reason, 0) + 1
    for L in launches:
        if (not L.status == "launched") or L.candidate_id.startswith("chain_") or L.candidate_id in started_ids:
            continue
        cand = by_action.get(L.candidate_id)
        _row(_candidate_family(cand))["selected_not_started"] += 1
    under_started: list[str] = []
    for fam, r in rows.items():
        proposed = int(r.get("proposed", 0) or 0)
        started = int(r.get("started", 0) or 0)
        selected = int(r.get("selected", 0) or 0)
        gap = max(0, proposed - started)
        selection_gap = max(0, selected - started)
        r["proposal_to_start_gap"] = gap
        r["selection_to_start_gap"] = selection_gap
        r["started_fraction_of_proposals"] = started / max(1, proposed)
        r["started_fraction_of_selected"] = started / max(1, selected)
        # A proposal that the Selector did not choose is a ranking/allocation
        # outcome, not an execution failure.  Only selected work that still has
        # not reached a worker belongs in the under-started audit.
        if selected >= 2 and selection_gap >= 2 and (started / max(1, selected)) <= 0.50:
            under_started.append(fam)
    return {"by_family": dict(sorted(rows.items())), "under_started_families": sorted(under_started)}


def _registry_diagnostic_only_families() -> set[str]:
    """Families whose native/advisory outputs require canonical AF2 scoring.

    This includes diagnostic generators (BindCraft/BoltzGen) and sequence
    redesign (ProteinMPNN); their native outputs are not the official strict/SU gate.
    """
    reg = default_registry()
    return {
        fam
        for fam, cap in reg.capabilities.items()
        if cap.outputs_diagnostic_only
        and cap.role in {"generator", "seq_redesign", "refilter"}
    }


_DIAGNOSTIC_ONLY_FAMILIES = _registry_diagnostic_only_families()
_ROUTE_SCORING_FAMILIES = {"structure_refilter"}
# Route backlog is the score-conversion/refilter queue only. Fresh diagnostic
# generators such as BindCraft/BoltzGen must remain feasible as cross-paradigm
# escapes even when earlier score conversions are backed up.
_ROUTE_HEAVY_FAMILIES = set(_ROUTE_SCORING_FAMILIES)

def _native_strict_like_pending_score_conversion(r: ResultRecord) -> bool:
    """Native/reward-gate near-official pass that still needs canonical scoring."""
    return native_strict_like_pending_score_conversion(r)


def _proxy_promising_pending_score_conversion(r: ResultRecord) -> bool:
    """Weaker native-model signal used for queue ordering, not strict credit."""
    return proxy_promising_pending_score_conversion(r)


def _has_usable_structure_artifact(r: ResultRecord) -> bool:
    art = r.artifacts or {}
    p = art.get("pdb_path")
    if p and Path(p).exists():
        return True
    d = art.get("pdb_dir")
    if d and Path(d).exists():
        d_path = Path(d)
        return any(d_path.glob("*.pdb")) or any(d_path.glob("*.cif")) or any(
            d_path.glob("*.mmcif")
        )
    return False


def _record_has_canonical_strict_axes(r: ResultRecord) -> bool:
    m = r.metrics or {}
    return all(isinstance(m.get(k), (int, float)) for k in ("pLDDT", "iPAE", "binder_scRMSD"))


def _record_needs_canonical_score_conversion(r: ResultRecord) -> bool:
    """Whether this result is a generated artifact still lacking official strict axes."""
    if r.exit_status != "ok" or not _has_usable_structure_artifact(r):
        return False
    if not is_score_convertible_diagnostic_record(r):
        return False
    fam = str(r.backend_family or "")
    if fam in _DIAGNOSTIC_ONLY_FAMILIES:
        return True
    if fam.startswith("complexa_") and not _record_has_canonical_strict_axes(r):
        return True
    return False


def _canonical_score_conversion_sources(results: list[ResultRecord]) -> set[str]:
    """Parent result ids whose canonical score-conversion child has completed."""
    out: set[str] = set()
    for r in results:
        bins = r.bins or {}
        if r.backend_family != "structure_refilter":
            continue
        if bins.get("refilter_role") != CANONICAL_SCORE_CONVERSION:
            continue
        src = bins.get("refilter_source")
        if src:
            out.add(str(src))
    return out


def _diagnostic_chain_backlog(
    actions: list[ActionCandidate],
    results: list[ResultRecord],
    launches: list[LaunchDecision],
    dispatches: list[DispatchRecord] | None = None,
) -> dict[str, Any]:
    """Summarize diagnostic artifacts waiting on canonical AF2 refilter.

    Diagnostic-only families intentionally do not emit the strict success keys.
    Their SU appears only after a child structure_refilter record. This snapshot
    makes the coverage gap visible to the Planner and to offline reviewers.
    """
    dispatches = dispatches or []
    by_result_id = {r.result_id: r for r in results}
    selected_ids = {L.candidate_id for L in launches if L.status == "launched"}
    dispatched_ids = {
        d.candidate_id for d in dispatches if d.status == "started"
    }
    dispatch_failed_ids = {
        d.candidate_id for d in dispatches
        if d.status in {"dispatch_failed", "parse_failed"}
    }
    refilter_sources = _canonical_score_conversion_sources(results)
    diag_records = [
        r for r in results
        if _record_needs_canonical_score_conversion(r)
    ]
    by_family: dict[str, dict[str, Any]] = {}

    def entry(fam: str) -> dict[str, Any]:
        return by_family.setdefault(
            fam,
            {
                "accepted_artifacts": 0,
                "unscored_artifacts": 0,
                "completed_refilters": 0,
                "chain_candidates": 0,
                "pending_chain_candidates": 0,
                "queued_chain_candidates": 0,
                "dispatched_chain_candidates": 0,
                "launched_chain_candidates": 0,
                "dispatch_failed_chain_candidates": 0,
                "native_strict_like_total": 0,
                "native_strict_like_pending_refilter": 0,
                "proxy_promising_total": 0,
                "proxy_promising_pending_refilter": 0,
                "native_or_proxy_pending_refilter": 0,
            },
        )

    for r in diag_records:
        e = entry(r.backend_family)
        e["accepted_artifacts"] += 1
        scored = r.result_id in refilter_sources
        if scored:
            e["completed_refilters"] += 1
        else:
            e["unscored_artifacts"] += 1
        native_like = _native_strict_like_pending_score_conversion(r)
        proxy_promising = _proxy_promising_pending_score_conversion(r)
        if native_like:
            e["native_strict_like_total"] += 1
            if not scored:
                e["native_strict_like_pending_refilter"] += 1
        if proxy_promising:
            e["proxy_promising_total"] += 1
            if not scored:
                e["proxy_promising_pending_refilter"] += 1
        if (native_like or proxy_promising) and not scored:
            e["native_or_proxy_pending_refilter"] += 1

    total_chain = pending_chain = queued_chain = dispatched_chain = failed_chain = 0
    for c in actions:
        if not (
            c.candidate_id.startswith("chain_")
            and c.method_family == "structure_refilter"
        ):
            continue
        parent = by_result_id.get(c.parent_result_id or "")
        fam = parent.backend_family if parent is not None else "unknown"
        if parent is None or not _record_needs_canonical_score_conversion(parent):
            continue
        total_chain += 1
        e = entry(fam)
        e["chain_candidates"] += 1
        if c.candidate_id in dispatch_failed_ids:
            failed_chain += 1
            e["dispatch_failed_chain_candidates"] += 1
        elif c.candidate_id in dispatched_ids:
            dispatched_chain += 1
            e["dispatched_chain_candidates"] += 1
            # The compatibility key "launched" counts confirmed worker dispatches.
            e["launched_chain_candidates"] += 1
        elif c.candidate_id in selected_ids:
            queued_chain += 1
            e["queued_chain_candidates"] += 1
        else:
            pending_chain += 1
            e["pending_chain_candidates"] += 1

    return {
        "diagnostic_families": sorted(by_family),
        "total_diagnostic_artifacts": len(diag_records),
        "total_unscored_diagnostic_artifacts": sum(
            int(v["unscored_artifacts"]) for v in by_family.values()
        ),
        "total_chain_candidates": total_chain,
        "pending_chain_candidates": pending_chain,
        "queued_chain_candidates": queued_chain,
        "dispatched_chain_candidates": dispatched_chain,
        "launched_chain_candidates": dispatched_chain,
        "dispatch_failed_chain_candidates": failed_chain,
        "by_family": by_family,
        "proxy_promising_pending_refilter": sum(
            int(v["proxy_promising_pending_refilter"]) for v in by_family.values()
        ),
        "native_or_proxy_pending_refilter": sum(
            int(v["native_or_proxy_pending_refilter"]) for v in by_family.values()
        ),
    }


def _route_health_from_archive(
    actions: list[ActionCandidate],
    results: list[ResultRecord],
    launches: list[LaunchDecision],
    dispatches: list[DispatchRecord],
    *,
    backlog_cap: int,
) -> RouteHealthSummary:
    """Live route-backlog signal for feasibility checks.

    The old route_health block was a constant stub, while candidate_builder
    still used it to gate route-heavy families. This derives a conservative
    backlog from selected/dispatched route-heavy candidates that have not yet
    produced any ResultRecord. It is intentionally coarse but no longer
    communicates "capacity is always empty".
    """
    route_action_ids = {
        c.candidate_id for c in actions
        if c.method_family in _ROUTE_HEAVY_FAMILIES
    }
    route_action_ids.update(
        d.candidate_id for d in dispatches
        if getattr(d, "method_family", None) in _ROUTE_HEAVY_FAMILIES
    )
    selected_ids = {
        L.candidate_id for L in launches
        if L.status == "launched" and L.candidate_id in route_action_ids
    }
    started_ids = {
        d.candidate_id for d in dispatches
        if d.status == "started" and d.candidate_id in route_action_ids
    }
    failed_ids = {
        d.candidate_id for d in dispatches
        if d.status in {"dispatch_failed", "parse_failed"}
        and not str(getattr(d, "why", "") or "").startswith("high_cost_inflight_cap")
        and d.candidate_id in route_action_ids
    }
    completed_ids: set[str] = set()
    for r in results:
        for pid in r.parent_ids or []:
            if pid in route_action_ids:
                completed_ids.add(pid)

    active_ids = (selected_ids | started_ids) - completed_ids - failed_ids
    route_results = [
        r for r in results
        if r.backend_family in _ROUTE_SCORING_FAMILIES
        and any(pid in route_action_ids for pid in (r.parent_ids or []))
    ]
    completed = len(route_results)
    near = sum(1 for r in route_results if r.exit_status == "ok" and is_near_miss(r.metrics))
    strict = sum(1 for r in route_results if r.exit_status == "ok" and is_strict_success(r.metrics))
    panel_ready = sum(1 for r in route_results if r.panel_ready)
    denom = max(1, completed)
    return RouteHealthSummary(
        raw_routed=len(selected_ids | started_ids),
        score_files_completed=completed,
        backlog_used=len(active_ids),
        backlog_cap=max(1, backlog_cap),
        near_miss_conversion=(near / denom) if completed else None,
        strict_conversion=(strict / denom) if completed else None,
        panel_ready_conversion=(panel_ready / denom) if completed else None,
    )


def _charged_gpu_h_recent_for_window(
    prior_evs: list,
    window_start_count: int,
    charged_gpu_h_total: float,
) -> float:
    """Compute charged GPU-hours over the SU window using archived tick boundaries.

    Use the latest available preceding boundary and clamp the difference to zero. The
    tick-aligned span can exceed the exact record window.
    """
    baseline_charged: float | None = None
    for e in prior_evs:  # chronological (oldest→newest)
        cc = getattr(e, "completed_children", None)
        if cc is None:
            continue
        if cc <= window_start_count:
            ch = getattr(e, "charged_gpu_h_total", None)
            if ch is not None:
                baseline_charged = ch
        else:
            break
    return (
        max(0.0, charged_gpu_h_total - baseline_charged)
        if baseline_charged is not None else charged_gpu_h_total
    )


def run_live_tick(
    archive: Archive,
    target: TargetConstraint,
    *,
    tick_id: str,
    tick_id_int: int,
    elapsed_wall_h: float,
    remaining_wall_h: float,
    pending_children: int = 0,
    cfg: LiveTickConfig | None = None,
    available_slots_override: int | None = None,
    inflight_gpu_h: float = 0.0,
    pending_family_load: dict[str, Any] | None = None,
    evidence_only: bool = False,
) -> dict[str, Any]:
    """Run one full T-REX tick on a real archive.

    Returns a JSON-serializable summary. Side effect: appends records to
    the archive. Does NOT actually launch SLURM jobs.

    `available_slots_override`: when set (>=1), used as the Selector's
    `available_slots` and the LiveTickConfig's `available_slots` for
    this single call only. Allows the event-driven controller to ask
    for exactly N launches when N slots are free, instead of always
    planning for the fixed batch size.

    `evidence_only`: compute the current EvidenceSummary and return before
    appending evidence or calling Planner/Supervisor. The event-driven
    controller uses this as a cheap state refresh before deterministic
    chain-refilter reserves, so a just-entered deep_stall state throttles those
    reserves without a one-tick lag.
    """
    cfg = cfg or LiveTickConfig()
    if available_slots_override is not None and available_slots_override >= 1:
        from dataclasses import replace as _dc_replace
        cfg = _dc_replace(
            cfg,
            available_slots=available_slots_override,
            selector=_dc_replace(
                cfg.selector, available_slots=available_slots_override,
            ),
        )

    # ----- 1. Load one read-only pre-tick archive snapshot ----------------
    archive_snapshot = load_tick_archive_snapshot(archive, target.target_id)
    all_results = prepare_archive_results(archive_snapshot.target_results, archive.root)
    all_actions = list(archive_snapshot.actions)
    all_hypotheses = list(archive_snapshot.latest_target_hypotheses)
    spawning = dict(archive_snapshot.spawning_action_by_result_id)
    prior_evidence = list(archive_snapshot.prior_target_evidence)
    previous_evidence = prior_evidence[-1] if prior_evidence else None

    # Track cold-start coverage by family. A one-slot tick may select only one of
    # the three seed paradigms; the next tick must emit the missing families rather
    # than treating the whole warm-start as complete.
    warmstart_coverage = resolve_warmstart_coverage(
        actions=archive_snapshot.actions,
        launched_decisions=archive_snapshot.launched_decisions,
        required_families=(family for family, _ in WARMSTART_FAMILIES),
        unavailable_families=(
            cfg.builder.unavailable_backends_override or ()
        ),
    )
    warmstart_completed_families = set(
        warmstart_coverage.completed_families
    )
    warmstart_missing_families = set(warmstart_coverage.missing_families)
    warmstart_seen = warmstart_coverage.seen

    clustering_phase = cluster_evidence_results(
        ResultClusteringRequest(
            results=all_results,
            target=target,
            spawning_actions=spawning,
            previous_evidence=previous_evidence,
            tick_id_int=tick_id_int,
            window_size=cfg.window_size,
            foldseek_config=cfg.foldseek,
            sequence_config=cfg.sequence_dedup,
        )
    )

    window_results = all_results[-cfg.window_size :]
    lifecycle_hyps = list(archive_snapshot.lifecycle_hypotheses)
    active_hyps = list(archive_snapshot.active_hypotheses)

    preplanner_lifecycle = retire_expired_hypotheses(
        PrePlannerLifecycleRequest(
            active_hypotheses=lifecycle_hyps,
            all_hypotheses=all_hypotheses,
            actions=all_actions,
            results=all_results,
            current_tick=tick_id_int,
            lifecycle_config=cfg.lifecycle,
            evidence_only=evidence_only,
        )
    )
    lifecycle_hyps = list(preplanner_lifecycle.active_hypotheses)
    all_hypotheses = list(preplanner_lifecycle.all_hypotheses)
    for retired_hypothesis in preplanner_lifecycle.retired_hypotheses:
        archive.append(retired_hypothesis)
    if preplanner_lifecycle.retired_hypotheses:
        print(
            f"  [lifecycle] retired "
            f"{len(preplanner_lifecycle.retired_hypotheses)} TTL-expired "
            f"descendant-free hypothesis card(s) before Planner",
            flush=True,
        )
        active_hyps = archive.retrieve_active_hypotheses(
            target_id=target.target_id,
            limit=8,
            include_terminal=True,
        )

    # Build archive-derived execution context before assembling current evidence.
    charged_gpu_count, charged_gpu_h_scope = _charged_gpu_count_from_env()
    charged_gpu_h_total: float | None = None
    charged_gpu_h_recent: float | None = None
    if charged_gpu_count is not None:
        charged_gpu_h_total = max(0.0, elapsed_wall_h * charged_gpu_count)
        # Keep raw reserved-allocation exposure tick-aligned for audit metadata.
        # It is not converted into a SU/GPU-h objective or surfaced to the LLM.
        window_start_count = max(0, len(all_results) - cfg.window_size)
        charged_gpu_h_recent = _charged_gpu_h_recent_for_window(
            prior_evidence,
            window_start_count,
            charged_gpu_h_total,
        )
    prior_launches = list(archive_snapshot.launched_decisions)
    prior_dispatches = list(archive_snapshot.dispatch_records)
    recent_modes = _recent_realized_modes(prior_launches, prior_dispatches)
    recent_started_families = _recent_started_families(all_actions, prior_dispatches)
    dispatch_realization = _dispatch_realization_summary(
        all_actions, prior_launches, prior_dispatches,
        recent_started_window_k=cfg.selector.mode_window_k,
    )
    prior_supervisors = list(archive_snapshot.supervisor_decisions)
    mode_credit = _mode_credit_from_history(
        prior_supervisors, prior_launches, prior_dispatches,
        cap=cfg.selector.mode_credit_cap,
    )
    execution_realization = _execution_realization_summary(
        all_actions, prior_launches, prior_dispatches, prior_supervisors,
    )
    diagnostic_chain_backlog = _diagnostic_chain_backlog(
        all_actions, all_results, prior_launches, prior_dispatches
    )
    route_health_summary = _route_health_from_archive(
        all_actions,
        all_results,
        prior_launches,
        prior_dispatches,
        backlog_cap=max(4, cfg.available_slots * 4),
    )
    # Twenty compact entries preserve long-loop state/family/SU evolution while
    # bounding Planner context to roughly 2-4K tokens.
    recent_tick_history = build_recent_tick_trajectory(
        prior_evidence=prior_evidence,
        actions=all_actions,
        dispatch_records=prior_dispatches,
        max_ticks=20,
    )

    evidence_phase = assemble_tick_evidence(
        EvidenceAssemblyRequest(
            target=target,
            tick_id=tick_id,
            tick_id_int=tick_id_int,
            elapsed_wall_h=elapsed_wall_h,
            remaining_wall_h=remaining_wall_h,
            pending_children=pending_children,
            inflight_gpu_h=inflight_gpu_h,
            results=all_results,
            hypotheses=all_hypotheses,
            spawning_actions=spawning,
            prior_evidence=prior_evidence,
            clustering=clustering_phase,
            window_size=cfg.window_size,
            worker_wall_gpu_count=cfg.worker_wall_gpu_count,
            planner_model=cfg.planner.model,
            enable_exemplars=cfg.enable_exemplars,
            route_health_summary=route_health_summary,
            recent_fallback_rate=recent_fallback_rate(
                archive_snapshot.llm_call_records
            ),
            charged_gpu_count=charged_gpu_count,
            charged_gpu_h_total=charged_gpu_h_total,
            charged_gpu_h_recent=charged_gpu_h_recent,
            charged_gpu_h_scope=charged_gpu_h_scope,
            recent_ticks_history=recent_tick_history,
            diagnostic_chain_backlog=diagnostic_chain_backlog,
            llm_health=summarize_llm_health(
                archive_snapshot.llm_call_records,
                cfg.planner.model,
            ),
            pending_family_load=pending_family_load or {},
            dispatch_realization=dispatch_realization,
            execution_realization=execution_realization,
        )
    )
    evidence = evidence_phase.evidence
    production_panel_record = evidence_phase.production_panel_record
    naive_strict_total = evidence_phase.strict_total
    naive_strict_window = evidence_phase.strict_window

    if evidence_only:
        return build_evidence_only_summary(
            tick_id=tick_id,
            target_id=target.target_id,
            evidence=evidence,
            all_result_count=len(all_results),
            window_result_count=len(window_results),
            strict_total=naive_strict_total,
            strict_window=naive_strict_window,
        )

    # ----- Evidence-skip gate (over-calling reduction) -------------------
    # If the decision-relevant signature is unchanged from the previous tick
    # AND we have prior cards + a prior valid mode_mixture, reuse them and skip
    # the Planner + Supervisor LLM calls. The deterministic builder + selector
    # still run, so free slots still get launches. Computed BEFORE appending
    # this evidence so the "previous" lookups don't include the current tick.
    skip_llm = False
    reuse_mixture: dict[str, float] | None = None
    reuse_confidence: float = 1.0
    if cfg.skip.enabled and active_hyps:
        prev_evs = archive_snapshot.evidence_records
        prev_sup = [
            s for s in archive_snapshot.supervisor_decisions
            if s.mode_mixture and not s.fallback_used
        ]
        if prev_evs and prev_sup:
            if _decision_signature(prev_evs[-1]) == _decision_signature(evidence):
                skip_llm = True
                reuse_mixture = dict(prev_sup[-1].mode_mixture)
                reuse_confidence = getattr(prev_sup[-1], "confidence", 1.0)
                print(
                    f"  [evidence-skip] signature unchanged → reuse prior plan "
                    f"(mixture={reuse_mixture}); skipping Planner+Supervisor LLM",
                    flush=True,
                )

    archive.append(evidence)
    if production_panel_record is not None:
        archive.append(production_panel_record)

    # ----- 2. Planner proposal and critic validation ---------------------
    proposal_phase = run_proposal_phase(
        ProposalPhaseRequest(
            archive=archive,
            evidence=evidence,
            active_hypotheses=active_hyps,
            warmstart_missing_families=frozenset(
                warmstart_missing_families
            ),
            warmstart_seen=warmstart_seen,
            unavailable_families=frozenset(
                cfg.builder.unavailable_backends_override or ()
            ),
            planner_config=cfg.planner,
            critic_config=cfg.critic,
            tick_id=tick_id,
            tick_number=tick_id_int,
            reuse_prior_plan=skip_llm,
            bootstrap_seed_action_families=(
                BOOTSTRAP_SEED_ACTION_FAMILIES
            ),
            schema_version=SCHEMA_VERSION,
        ),
        ProposalPhaseDependencies(
            call_planner=call_planner,
            build_llm_call_record=_llm_call_record,
            build_prompt_audit_snapshot=_prompt_audit_snapshot,
        ),
    )
    for proposal_record in proposal_phase.records_to_append:
        archive.append(proposal_record)
    planner_out = proposal_phase.planner_output
    planner_lat = proposal_phase.planner_latency_seconds
    hyps = list(proposal_phase.hypotheses)
    planner_fb_reason = proposal_phase.fallback_reason
    critic_flags_summary = list(proposal_phase.critic_flags)

    # ----- 3. Candidate validation ---------------------------------------
    candidate_phase = build_candidate_phase(
        CandidatePhaseRequest(
            hypotheses=hyps,
            evidence=evidence,
            builder_config=cfg.builder,
            include_warmstart=not warmstart_seen,
            warmstart_completed_families=frozenset(
                warmstart_completed_families
            ),
            results=all_results,
            existing_actions=all_actions,
            prior_launches=prior_launches,
            spawning_actions=spawning,
        )
    )
    candidates = list(candidate_phase.candidates)
    selector_pool = list(candidate_phase.feasible_candidates)
    pending_chain = list(candidate_phase.pending_chain_candidates)
    for candidate in candidates:
        archive.append(candidate)

    if candidate_phase.infeasible_candidate_count:
        print(
            "  [candidate_filter] "
            f"{candidate_phase.infeasible_candidate_count} infeasible "
            "candidate(s) archived but hidden from Supervisor/Selector",
            flush=True,
        )
    if pending_chain:
        print(
            f"  [auto_chain] {len(pending_chain)} pending chain candidate(s) "
            "handled by bounded reserve/backfill lane",
            flush=True,
        )

    # ----- 4. Supervisor --------------------------------------------------
    supervision_phase = run_supervision_phase(
        SupervisionPhaseRequest(
            evidence=evidence,
            hypotheses=hyps,
            candidates=selector_pool,
            planner_output=planner_out,
            planner_fallback_reason=planner_fb_reason,
            selector_config=cfg.selector,
            supervisor_config=cfg.supervisor,
            recent_realized_modes=recent_modes,
            mode_credit=mode_credit,
            diagnostic_chain_backlog=diagnostic_chain_backlog,
            pending_family_load=pending_family_load,
            execution_realization=execution_realization,
            tick_id=tick_id,
            reuse_prior_plan=skip_llm,
            reuse_mode_mixture=reuse_mixture,
            reuse_confidence=reuse_confidence,
            use_unified_reasoner=cfg.unified_reasoner,
        ),
        SupervisionPhaseDependencies(
            call_supervisor=call_supervisor,
            build_llm_call_record=_llm_call_record,
            build_prompt_audit_snapshot=_prompt_audit_snapshot,
        ),
    )
    archive.append(supervision_phase.llm_call_record)
    sup_out = supervision_phase.supervisor_output
    sup_lat = supervision_phase.supervisor_latency_seconds
    sup_fb_reason = supervision_phase.fallback_reason
    selector_context = supervision_phase.selector_context

    # ----- 5. Selection ---------------------------------------------------
    selection_phase = select_live_tick_launches(
        SelectionPhaseRequest(
            evidence=evidence,
            candidates=selector_pool,
            supervisor_output=sup_out,
            selector_config=cfg.selector,
            tick_id=tick_id,
            fallback_reason=sup_fb_reason,
            hypotheses=hyps,
            recent_realized_modes=recent_modes,
            mode_credit=mode_credit,
            recent_started_families=recent_started_families,
            route_health=route_health_summary,
        )
    )
    launches = list(selection_phase.launches)
    sel_debug = selection_phase.selector_debug
    for launch in launches:
        archive.append(launch)

    archive.append(
        _supervisor_decision_record(
            sup_out,
            tick_id=tick_id,
            fallback_used=sup_fb_reason is not None,
            clamps=list(sel_debug.get("clamp_log", [])),
            selector_debug=sel_debug,
            selector_context=selector_context,
        )
    )

    # ----- 6. Hypothesis lifecycle update --------------------------------
    lifecycle_phase = update_lifecycle_phase(
        LifecyclePhaseRequest(
            hypotheses=lifecycle_hyps,
            actions=all_actions,
            results=all_results,
            spawning_actions=spawning,
            current_tick=tick_id_int,
            lifecycle_config=cfg.lifecycle,
        )
    )
    for updated_hypothesis in lifecycle_phase.updated_hypotheses:
        archive.append(updated_hypothesis)
    updated_lifecycle = list(lifecycle_phase.update_summaries)

    # ----- 7. User-facing summary ----------------------------------------
    return build_live_tick_summary(
        LiveTickSummaryRequest(
            tick_id=tick_id,
            target_id=target.target_id,
            evidence=evidence,
            all_result_count=len(all_results),
            window_result_count=len(window_results),
            strict_total=naive_strict_total,
            strict_window=naive_strict_window,
            critic_enabled=cfg.critic.enabled,
            critic_flags=critic_flags_summary,
            planner_output=planner_out,
            planner_latency_seconds=planner_lat,
            candidates=candidates,
            supervisor_output=sup_out,
            supervisor_latency_seconds=sup_lat,
            selector_debug=sel_debug,
            launches=launches,
            lifecycle_updates=updated_lifecycle,
            fallback_reason=sup_fb_reason,
        )
    )


# ---------------------------------------------------------------------------
# CLI (observe-mode runbook entry point)
# ---------------------------------------------------------------------------

def main() -> None:
    """Run one planning cycle on an existing archive and record decisions without
    dispatching workers.
    """
    import argparse

    p = argparse.ArgumentParser(
        description="Run one live T-REX tick on an existing archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--archive-root", type=Path, required=True,
                    help="Archive directory (contains *.jsonl files)")
    p.add_argument("--target-constraint", type=Path, required=True,
                    help="JSON file with TargetConstraint fields (target_id, target_class, hotspots, etc.)")
    p.add_argument("--tick-id", type=str, default=None,
                    help="Round/tick identifier; defaults to timestamp")
    p.add_argument("--elapsed-wall-h", type=float, default=0.0,
                    help="Elapsed wall-clock since campaign start")
    p.add_argument("--remaining-wall-h", type=float, default=48.0,
                    help="Remaining budget")
    p.add_argument("--pending-children", type=int, default=0)
    p.add_argument("--once", action="store_true",
                    help="Run a single tick and exit (default behavior).")
    args = p.parse_args()

    # Load TargetConstraint from JSON
    tgt_data = json.loads(Path(args.target_constraint).read_text())
    target = TargetConstraint(**tgt_data)

    # Open archive
    archive = Archive(args.archive_root)

    # Default tick_id from timestamp if not provided
    tick_id = args.tick_id or time.strftime("t%Y%m%d_%H%M%S")
    # Derive integer tick by counting prior EvidenceSummary records
    tick_id_int = sum(1 for _ in archive.iter_records(EvidenceSummary)) + 1

    summary = run_live_tick(
        archive, target,
        tick_id=tick_id,
        tick_id_int=tick_id_int,
        elapsed_wall_h=args.elapsed_wall_h,
        remaining_wall_h=args.remaining_wall_h,
        pending_children=args.pending_children,
    )

    # Pretty-print summary to stdout
    print(json.dumps(summary, default=str, indent=2))


if __name__ == "__main__":
    main()
