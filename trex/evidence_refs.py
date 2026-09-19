"""Shared evidence-reference citation aliases for Planner and Supervisor.

The LLM prompts expose both structured EvidenceSummary keys and compact TL;DR
header aliases. Validators in planner.py and supervisor.py must accept the same
namespace so a valid scientific citation cannot pass one stage and fail another.
"""

from __future__ import annotations


EVIDENCE_REF_PREFIXES = {
    "axis_stats",
    "diagnostic_alt_model_scores",
    "diagnostic_axis_stats",
    "diagnostic_chain_backlog",
    "refilter_role_health",
    "examples",
    "exemplars",
    "exemplar",
    "example",
    "execution_realization",
    "joint_patterns",
    "method_health",
    "pending_family_load",
    "production_near_miss_ids",
    "production_panel",
    "production_panel_selected_ids",
    "recent_ticks_history",
    "recipes",
    "refold_probe_outcomes",
    "remediation_outcomes",
    "diagnosis_outcomes",
    "state_label",
    "strategy_feedback",
    "route_values",
    "route_replay",
    "route_value_replay",
    "objective_summary",
    "diversity_summary",
    "selector_context",
    "mode_mixture_ranges",
    # Evidence TL;DR aliases used in model citations. These are header-only
    # summaries from planner.evidence_tldr(), so they may not appear as
    # top-level keys in build_evidence_for_prompt() but must still be valid
    # citations.
    "run_SU",
    "raw_strict",
    "worker_gpu_h",
    "SU_per_worker_gpu_h",
    "SU/worker-GPUh_recent",
    "SU/worker-GPUh_total",
    "SU/worker-wall-GPUh_total",
    "SU_HWM/worker-wall-GPUh_total",
    "route_feedback_SU/completed-worker-GPUh_recent",
    "route_feedback_SU/completed-worker-GPUh_total",
    "strict_per_SU_total",
    "strict_duplicate_collapse",
    "top_bin_share",
    "top_bin_share(live,all)",
    "strict_su_top_bin_share",
    "strict_su_top_bin_share(live,winners)",
    "strict_su_top_bin_share(live,strict_records)",
    "strict_record_bin_share(live)",
    "duplicate_fraction",
    "basin",
    "dry_since_last_SU",
    "untried_cross_family_candidates",
    "execution_gap_under-tested",
    "top_diagnostic_blocker",
    "diagnostic_driver_tldr",
    "diagnostic_improvement_score",
    "diagnostic_improvement_axes",
    "negative_evidence",
    "hard_target_signal",
    "best_exemplar",
    "nearest_miss",
    "remaining_wall_h",
    "fine_diversity_aux",
    "recent_critic_flags_last_tick",
    # Natural aliases produced by LLM citations during cold start / active-card
    # reasoning. These are not always literal EvidenceSummary keys, but rejecting
    # them causes harmless scientific citations to trigger schema fallback. Do
    # not include broad ID prefixes such as "result"/"result_id" here; concrete
    # result citations must resolve against allowed_refs.
    "EvidenceSummary",
    "cross_family_escape",
    "hyp_cross_family_escape",
    "diagnostic_i4_mcts",
    "warmstart",
    "hypotheses",
    "hypothesis_ids",
}


# Compact namespace citations are summary references, not concrete result IDs.
# Keep this set explicit so tightening concrete route/result validation does not
# accidentally reject valid high-level citations such as route_values::bindcraft.

# Source-scoped diagnostic-axis citations are produced by diagnostic_driver_tldr,
# e.g. design_to_target_iptm[src=boltzgen:32]. They are summary citations, not
# concrete result IDs. Keep the set explicit so arbitrary bracketed strings do
# not bypass evidence validation.
SOURCE_SCOPED_DIAGNOSTIC_AXIS_REFS = {
    "avg_ipsae",
    "binder_pLDDT_avg",
    "binder_pTM_avg",
    "buried_sasa",
    "design_iiptm",
    "design_ptm",
    "design_to_target_iptm",
    "hotspot_rmsd",
    "interface_dG",
    "interface_hbonds",
    "interface_unsat_hbonds",
    "ipTM",
    "max_ipsae",
    "min_design_to_target_pae",
    "min_ipae",
    "shape_complementarity",
}

COMPACT_NAMESPACE_PREFIXES = {
    "axis_stats",
    "diagnostic_alt_model_scores",
    "diagnostic_axis_stats",
    "diagnostic_chain_backlog",
    "diversity_summary",
    "execution_realization",
    "hypotheses",
    "hypothesis_ids",
    "joint_patterns",
    "method_health",
    "objective_summary",
    "pending_family_load",
    "production_panel",
    "recent_ticks_history",
    "refilter_role_health",
    "refold_probe_outcomes",
    "remediation_outcomes",
    "diagnosis_outcomes",
    "route_value_replay",
    "route_values",
    "selector_context",
    "mode_mixture_ranges",
    "strategy_feedback",
}

COMPACT_UNDERSCORE_PREFIXES = {
    "route_replay",
    "route_value_replay",
}


def evidence_ref_allowed(ref: object, allowed_refs: set[str]) -> bool:
    """Shared citation validator for Planner and Supervisor.

    Broad namespace aliases are allowed only as literal field names or structured
    field citations. Concrete result/route IDs must either be exact allowed refs
    or use a suffix that is already present in allowed_refs. This prevents
    hallucinated IDs such as ``result_fake`` or ``route::fake`` from passing just
    because their prefix is familiar.
    """
    if not isinstance(ref, str) or not ref.strip():
        return False
    ref = ref.strip()
    if ref in allowed_refs or ref in EVIDENCE_REF_PREFIXES:
        return True
    if ref.startswith("state="):
        return True
    if "[src=" in ref:
        axis = ref.split("[src=", 1)[0].strip()
        if axis in SOURCE_SCOPED_DIAGNOSTIC_AXIS_REFS:
            return True
    if "=" in ref:
        key = ref.split("=", 1)[0].strip()
        if key in allowed_refs or key in EVIDENCE_REF_PREFIXES:
            return True

    for prefix in ("result_id", "result"):
        for sep in ("::", ":"):
            marker = prefix + sep
            if ref.startswith(marker):
                return ref[len(marker):] in allowed_refs

    if ref.startswith("route::"):
        return ref in allowed_refs

    for prefix in ("exemplar", "exemplars", "example", "examples", "recipe"):
        if ref.startswith(prefix + "::") or ref.startswith(prefix + ":"):
            return True

    namespace_prefixes = COMPACT_NAMESPACE_PREFIXES & (set(EVIDENCE_REF_PREFIXES) | set(allowed_refs))
    if any(ref.startswith(prefix + "::") or ref.startswith(prefix + ":") for prefix in namespace_prefixes):
        return True

    underscore_prefixes = COMPACT_UNDERSCORE_PREFIXES & (set(EVIDENCE_REF_PREFIXES) | set(allowed_refs))
    if any(ref.startswith(prefix + "_") for prefix in underscore_prefixes):
        return True

    prefixes = set(EVIDENCE_REF_PREFIXES) | set(allowed_refs)
    return any(
        ref.startswith(prefix + ".")
        or ref.startswith(prefix + "[")
        for prefix in prefixes
    )
