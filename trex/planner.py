"""Planner calls, structured-output validation, and hypothesis materialization."""

from __future__ import annotations

import json
import hashlib
import math
import time
from dataclasses import dataclass, replace
from typing import Any

from .dedup_trust import near_miss_dedup_trusted
from .evidence_refs import EVIDENCE_REF_PREFIXES, evidence_ref_allowed
from .cross_campaign_memory import (
    cross_campaign_memory_evidence_refs,
    cross_campaign_memory_for_prompt,
    cross_campaign_memory_system_appendix,
)
from trex.llm import create_client

from .schemas import (
    HypothesisCard,
    PlannerOutput,
    PredictedChange,
    PreserveConstraint,
    ReasoningTrace,
    EvidenceSummary,
    to_jsonable,
)
from .success_criteria import STRICT_SUCCESS


# ---------------------------------------------------------------------------
# Prompt + schema
# ---------------------------------------------------------------------------


MAX_PLANNER_CARDS = 4
PLANNER_MODES = ("exploit", "rescue", "explore")


_PLANNER_CORE_CONTRACT = """You are the Planner for T-REX, a budgeted protein-binder design controller.

Task: from the latest EvidenceSummary, propose 1-4 testable HypothesisCards that
increase structurally-unique strict successes per live WORKER-WALL GPU-hour. Do not use
target identity, target class, or unsupported family priors. If a
cross_campaign_memory block is provided, treat it only as advisory historical
experience conditioned on current evidence; infer the runnable strategy from
EvidenceSummary, route value, available candidates, and constraints.

Return JSON only. Exact top-level shape; fail_reason may be null or omitted:
{
  "abstain": false,
  "confidence": 0.6,
  "fail_reason": null,
  "rationale": "brief evidence-grounded rationale",
  "cards": [
    {
      "claim": "one-sentence testable claim",
      "mode_affinity": {"exploit": 0.1, "rescue": 0.7, "explore": 0.2},
      "evidence_refs": ["EvidenceSummary", "result_or_route_id"],
      "predicted_metric_changes": [
        {"axis": "iPAE", "direction": "decrease", "baseline_refs": ["result_id"],
         "min_relative_deficit_reduction": 0.15, "min_absolute_delta": null}
      ],
      "preserve_constraints": [{"axis": "pLDDT", "max_relative_deficit_increase": 0.05}],
      "recommended_action_families": ["complexa_beam"],
      "config_delta_suggestions": {"complexa_beam": {"beam_width": 8}},
      "reasoning_trace": {
        "observed_signal": "evidence signal",
        "inference": "scientific interpretation",
        "action_implication": "why this family/config follows"
      }
    }
  ]
}

Hard schema rules:
- Top-level required keys are abstain, confidence, rationale, cards; fail_reason is optional.
- Each card needs claim, mode_affinity, evidence_refs, predicted_metric_changes,
  preserve_constraints, recommended_action_families, and reasoning_trace.
- predicted_metric_changes axes are ONLY pLDDT, iPAE, binder_scRMSD.
- Axis direction is fixed: pLDDT increases; iPAE and binder_scRMSD decrease.
- All predicted_metric_changes in one card must use the same baseline_refs in
  the same order because lifecycle evaluation has one comparison baseline.
- preserve_constraints axes are ONLY pLDDT, iPAE, binder_scRMSD. Diversity is
  expressed through claim/action choice and route/panel evidence, not lifecycle
  preserve constraints. Diagnostic axes are evidence and lever cues, not
  PredictedChange axes.
- min_relative_deficit_reduction must be finite and in [0,1]; use 0.0 for
  absolute-delta-only, with a positive min_absolute_delta.
- recommended_action_families must come from available_families_this_cluster.
- config_delta_suggestions may use only allowed_params_per_family and must obey
  eval_budget_per_family product caps. Unknown keys are dropped/audited.
- Parent-required families need concrete baseline_refs/result_ids; aggregate
  labels such as method_health, top_bin_share, strategy_feedback, or family names
  are evidence_refs only, not parent refs.
- Cite evidence IDs shown in the prompt; no hallucinated IDs or prose-only refs.
- reasoning_trace is an audit trace, not hidden chain-of-thought; keep each field
  one short sentence.
"""


_PLANNER_EVIDENCE_CONTRACT = """Evidence reading order:
1. objective_summary: live run objective is Foldseek SU@TM0.6 per
   worker-wall GPU-h. Use live_su_tm06_hwm_delta/dry_since_last_su to judge
   recent progress. Reserved controller/LLM GPU allocation is not a decision
   objective.
2. route_values: primary route/config evidence. Prefer exact route rows over
   family averages. Rank by gpu_recent_new_su_per_route_gpu_h, then
   medium_recent_* as delayed-feedback guard, then safe direct-scored
   record_recent_*. Lifetime rate is memory/context, not current exploit rank.
3. method_health: family fallback only. For outputs_diagnostic_only families,
   read chained_su_per_gpu_h_recent/chained_strict_yield_su_recent; their own
   su_per_gpu_h_recent may be None by design.
4. exemplars + axis_stats: concrete strict/near-miss binders and the three strict
   axes. Official strict gate is pLDDT >= 90, iPAE <= 7/31, binder_scRMSD < 1.5.
5. diagnostic_driver_tldr + diagnostic_axis_stats + diagnostic_alt_model_scores:
   scientific lever evidence. These do not mint SU; use them to choose knobs,
   rescue parents, or justify continued probing when SU is not yet present.
6. diversity_summary + dedup_trust + recent_ticks_history + execution_realization:
   collapse, trust, trajectory, and selected->started feedback.

Interpretation rules:
- Raw strict_count is not the objective. Many strict hits in one Foldseek bin are
  one SU. If strict grows but new SU/GPU-h is flat, treat it as duplicate/SU
  bottleneck rather than progress.
- route_values.status: promote = competitive new-SU/GPU-h;
  awaiting_score_conversion = accepted artifacts still need canonical scoring;
  defer = enough compute but weak/no recent SU; diversify/collapse_risk = old or
  raw-strict success is re-mining too few official SU bins.
- Read recent before lifetime. Old SU should not keep exploit weight unless there
  is current near-miss, unscored diagnostic backlog, diagnostic improvement, or a
  material config/parent change worth testing.
- Do not generalize one failed route/config into family failure. Down-rank a
  family only after multiple distinct configs exceed lane warmup with no SU,
  no recent near-miss, and no diagnostic/chained progress.
- Treat a first official SU on a sparse/hard run as a rare positive signal. It
  justifies a bounded confirmation hypothesis even when its absolute rate is
  low, but it is not mature evidence for concentrating every worker.
- Scientific objectives are lexicographic. New-SU/GPU-h is primary. When exact
  routes have approximately comparable recent value (about 15%, accounting for
  sparse counts/exposure), prefer higher strict_quality_p25/median and inspect
  strict_quality_axis_margins for the limiting canonical axis.
- Backend-specific diagnostics (ipTM, ipsae, pTM, interface_dG, buried_sasa,
  contacts, clashes, and verifier scores) are source-scoped Pareto evidence, not
  interchangeable global rewards. Use diagnostic_axis_stats and
  diagnosis_outcomes to improve one supported blocker while preserving the
  three canonical strict axes. Never penalize another family for not emitting a
  metric, and do not average differently calibrated sources.
- Lane warmups are cost/feedback-delay, not target priors: cheap direct-scored
  Complexa lanes ~0.5 GPU-h; delayed/high-cost lanes such as bindcraft,
  boltzgen, proteinmpnn_redesign, complexa_mcts ~1.5-2 GPU-h.
- diversity_summary fields are live-TM0.6 structural collapse plus optional TM0.8/sequence panel
  signals. The headline live objective is TM0.6 SU; report TM0.5/0.6/0.8 post-hoc.
- If dedup_trust is degraded, treat SU/diversity as uncertain and cite it.
"""


_PLANNER_ACTION_GUIDE = """Action policy:
Action evidence hierarchy is a ranking prior, NOT a feasibility ban. Use the
lowest sufficient level supported by evidence, but allow another level when route
value, near-miss, stuck-lineage, or diagnostics justify it:
  L0 low-evidence root-family coverage.
  L1 same-family material search perturbation: seed/noise, beam_width/n_branch,
     FK temperature, sc_scale_noise, MCTS exploration knobs.
  L2 blocker-matched rescue around a concrete near-miss/stuck lineage:
    parent-bound proteinmpnn_redesign when a usable parent exists, OR same-family
    material perturbation when no concrete parent
     exists or the root route needs re-search.
  L3 scalar reward reweighting (reward_i_pae/reward_plddt/min_ipae/etc.) after
     enough axis/diagnostic evidence, or when paired with material search.
  L4 extended/high-cost expansion when recent route value or repeated near-miss
     evidence supports the cost.

Mode choice:
- Exploit a route/config that is still buying new SU/GPU-h, or has active
  near-miss/chained progress without collapse/defer.
- Rescue a concrete blocker. binder_scRMSD -> sequence redesign/hallucination;
  iPAE -> interface/backbone re-dock; pLDDT -> regenerate/backbone search or
  pLDDT-aware MCTS/hallucination when iPAE is pass/near. Prefer L1/L2 material
  perturbations before reward-only retries.
- Explore when routes are dry past warmup, duplicate-collapsed, unsupported by
  recent evidence, or the dominant root family stopped buying new SU/GPU-h.

Special safeguards:
- Appendix-I.4-style behavior is target-agnostic: use complexa_mcts plus
  pLDDT/iPAE reward knobs/sequence_hallucination when evidence shows low pLDDT
  with plausible interface/iPAE. It is an option, not a forced launch.
- Greedy knobs matter only with refinement_algorithm="sequence_hallucination".
- canonical_score_conversion is fixed official AF2 scoring plumbing; do not vary
  it to chase SU. parent_model_refold is advisory only and writes refold_*.
- Do not propose bindcraft/boltzgen/proteinmpnn_redesign merely to score, clear,
  or convert existing diagnostic backlog; scoring backlog is system-managed.
- proteinmpnn_redesign broadens a concrete parent; structure_refilter alone is not
  a de-novo SU-diversifying generator.
- If remaining_wall_h < ~3h, avoid fresh high-cost probes unlikely to finish.
"""


def _render_family_role_appendix() -> str:
    """Registry-derived compact role appendix for the Planner system prompt.

    The per-tick user payload still carries the authoritative available subset,
    allowed_params_per_family, and eval_budget_per_family. This appendix explains
    how to interpret the registry roles without maintaining a long static table.
    """
    from .capability_registry import default_registry

    def scoring_label(fam: str, role: str, outputs_diag: bool) -> str:
        if fam.startswith("complexa_"):
            return "direct official strict/SU when canonical af2folding metrics are present; fallback chain only if missing"
        if fam == "structure_refilter":
            return "AF2 refilter; canonical_score_conversion mints SU upstream, parent_model_refold is advisory"
        if role == "refilter":
            return "diagnostic/advisory refilter unless separately canonically scored"
        if outputs_diag:
            return "diagnostic-only output; chain through canonical structure_refilter for official strict/SU"
        return "direct-scored"

    reg = default_registry()
    lines = [
        "FAMILY ROLE APPENDIX (generated from capability_registry; per-tick family_role_table is authoritative for availability):"
    ]
    for fam in sorted(reg.capabilities):
        cap = reg.capabilities[fam]
        parent = "parent_required" if cap.requires_parent_pdb else "de_novo_or_target_conditioned"
        diag = "diagnostic_only" if cap.outputs_diagnostic_only else "strict_capable"
        note = f"; notes={cap.notes}" if cap.notes else ""
        lines.append(
            f"- {fam}: role={cap.role}; cost={cap.default_cost_class}; {parent}; {diag}; "
            f"scoring={scoring_label(fam, cap.role, cap.outputs_diagnostic_only)}{note}"
        )
    lines.extend([
        "Chain patterns:",
        "- bindcraft/boltzgen/proteinmpnn_redesign -> canonical structure_refilter for official AF2 strict/SU.",
        "- proteinmpnn_redesign rescues a concrete parent sequence/backbone; root_family preserves the original generator.",
        "- For NEW diagnostic generation/redesign, list the first stage in recommended_action_families; the controller schedules canonical scoring of its new artifacts.",
        "- For EXISTING artifacts already in diagnostic_chain_backlog, do not list the generator to trigger scoring; chain_refilter/canonical_score_conversion is system-managed.",
    ])
    return "\n".join(lines)


_PLANNER_OUTPUT_CONSTRAINTS = """Additional constraints:
- diagnostic_axis_stats are two-tier. quality_threshold is the good-interface
  band; pass_threshold/below_accept_count are tool accept floors. Strict
  axis_stats use only the three success axes.
- Use diagnostic_driver_tldr as the compact scientific driver: top 3 levered axes
  plus top 1 corroboration-only axis, severity margin-normalized. Keep
  PredictedChange axes strict-only.
- Diagnostic axes are source-scoped. If an axis appears as axis[src=family],
  use that source family and the remediation-lever map; do not silently
  translate a BoltzGen-only axis into a BindCraft/Complexa knob.
- In productive states, diagnostics are monitoring/tie-break/knob-tweak evidence
  unless recent new SU/GPU-h decays. In rescue_rich/stalled/deep_stall/
  duplicate-collapse states, diagnostics are primary evidence for rescue/explore
  lever selection.
- If dry >=6 worker GPU-h and diagnostics show low pLDDT/scaffold confidence with
  plausible iPAE/interface, explicitly consider an I.4-style complexa_mcts card.
- If deep_stall, or stalled/strict_duplicate_collapse with dry >=6 worker GPU-h
  and the dominant root family has no recent new SU/GPU-h, include at least two
  distinct untried/under-tested non-dominant root-family probes when available.
  Keep productive cheap routes when they still buy recent new SU/GPU-h.
- If sustained deep_stall recovers with only 1-2 fresh SU on a low-yield run,
  keep that recovered route in exploit/rescue but leave at least one
  non-dominant root-family probe open until total/recent SU/GPU-h is healthy.
- If a stuck_lineage_root or contradicted/retired hypothesis is cited, do not
  repeat the same parent/config unless evidence materially changed.
- If no available family fits, pivot to the closest available registry family;
  do not emit missing_candidate_requests.
- Set abstain=true only when no grounded hypothesis can be formed. Self-report
  confidence >0.6 only when direct evidence supports the cards.
"""


def _build_planner_system() -> str:
    return "\n\n".join([
        _PLANNER_CORE_CONTRACT,
        _PLANNER_EVIDENCE_CONTRACT,
        _PLANNER_ACTION_GUIDE,
        _render_family_role_appendix(),
        _PLANNER_OUTPUT_CONSTRAINTS,
    ])


PLANNER_SYSTEM = _build_planner_system()

def _render_diagnostic_lever_map() -> str:
    """Render diagnostic remediation guidance from
    evidence_reducer.DIAGNOSTIC_AXIS_REMEDIATION.
    """
    from .evidence_reducer import DIAGNOSTIC_AXIS_REMEDIATION
    levered = [
        f"    {a} -> {lev}"
        for a, lev in DIAGNOSTIC_AXIS_REMEDIATION.items() if lev
    ]
    corrob = [a for a, lev in DIAGNOSTIC_AXIS_REMEDIATION.items() if lev is None]
    return (
        "\nDIAGNOSTIC REMEDIATION LEVERS (how to ACT on a diagnostic axis):\n"
        "- Use the named (family.param) when either the compact diagnostic_driver_tldr\n"
        "  flags that axis as a run-level blocker OR an exemplar carries it as\n"
        "  `diagnostic_blocking_axis`. Strict axes still define success, but diagnostic\n"
        "  axes decide which scientific lever/family should be tested next. Treat\n"
        "  axis[src=family] provenance as binding unless explicitly proposing an\n"
        "  orthogonal cross-family probe; never cite one family's diagnostic axis\n"
        "  as direct evidence for another family's config_delta.\n"
        + "\n".join(levered)
        + "\n- CORROBORATION-ONLY diagnostic axes (NO lever in any family — cite to\n"
        "  explain WHY a design fails, NEVER as a config_delta target): "
        + ", ".join(corrob) + "\n"
        "- `diagnosis_outcomes` (evidence) reports whether remediating each blocked\n"
        "  axis HAS WORKED so far: {axis: {lever, attempts, improved, improve_rate,\n"
        "  strict, strict_rate, unique_su, unique_su_rate, provisional}}. PREFER levers with high unique_su_rate/unique_su\n"
        "  as secondary signal; use improve_rate only when provisional=false, and\n"
        "  down-weight a lever tried often with low unique_su_rate and low improve_rate.\n"
    )


PLANNER_SYSTEM = PLANNER_SYSTEM + _render_diagnostic_lever_map()


def planner_system_prompt() -> str:
    """Return the system prompt for the current, explicitly applied run env."""

    return PLANNER_SYSTEM + cross_campaign_memory_system_appendix("Planner")


PLANNER_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["abstain", "confidence", "cards", "rationale"],
    "properties": {
        "abstain": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "fail_reason": {"type": ["string", "null"]},
        "rationale": {"type": "string"},
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "claim",
                    "mode_affinity",
                    "evidence_refs",
                    "predicted_metric_changes",
                    "preserve_constraints",
                    "recommended_action_families",
                    "reasoning_trace",
                ],
                "properties": {
                    "claim": {"type": "string", "minLength": 8},
                    "mode_affinity": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["exploit", "rescue", "explore"],
                        "properties": {
                            "exploit": {"type": "number", "minimum": 0},
                            "rescue": {"type": "number", "minimum": 0},
                            "explore": {"type": "number", "minimum": 0},
                        },
                    },
                    "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                    "predicted_metric_changes": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "axis",
                                "direction",
                                "min_relative_deficit_reduction",
                            ],
                            "properties": {
                                "axis": {
                                    "type": "string",
                                    "enum": ["pLDDT", "iPAE", "binder_scRMSD"],
                                },
                                "direction": {"type": "string", "enum": ["increase", "decrease"]},
                                "baseline_refs": {"type": "array", "items": {"type": "string"}},
                                "min_relative_deficit_reduction": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                                "min_absolute_delta": {
                                    "type": ["number", "null"],
                                    "exclusiveMinimum": 0.0,
                                },
                            },
                        },
                    },
                    "preserve_constraints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["axis", "max_relative_deficit_increase"],
                            "properties": {
                                "axis": {
                                    "type": "string",
                                    "enum": ["pLDDT", "iPAE", "binder_scRMSD"],
                                },
                                "max_relative_deficit_increase": {
                                    "type": "number",
                                    "minimum": 0.0,
                                },
                            },
                        },
                    },
                    "recommended_action_families": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "config_delta_suggestions": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    },
                    "reasoning_trace": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "observed_signal",
                            "inference",
                            "action_implication",
                        ],
                        "properties": {
                            "observed_signal": {"type": "string"},
                            "inference": {"type": "string"},
                            "action_implication": {"type": "string"},
                        },
                    },
                    "ttl_ticks": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            },
        },
    },
}


VALID_ACTION_FAMILIES = {
    "complexa_beam",
    "complexa_best_of_n",
    "complexa_fk_steering",
    "complexa_mcts",
    "proteinmpnn_redesign",
    "structure_refilter",
    "bindcraft",
    "boltzgen",
}


PARENT_BOUND_ACTION_FAMILIES = {
    "proteinmpnn_redesign",
    "structure_refilter",
}


_TARGET_IDENTITY_KEYS = {"target_id", "target_class"}


def _strip_target_identity_for_prompt(obj: Any) -> Any:
    """Remove target identity/class fields from LLM-facing payloads.

    The controller still keeps target_id internally for materialization and
    archive lookup, but the Planner/Supervisor should allocate from diagnostics
    and observed outcomes rather than target-name priors.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_target_identity_for_prompt(v)
            for k, v in obj.items()
            if k not in _TARGET_IDENTITY_KEYS
        }
    if isinstance(obj, list):
        return [_strip_target_identity_for_prompt(v) for v in obj]
    return obj


def build_evidence_for_prompt(evidence: EvidenceSummary) -> dict[str, Any]:
    """Build a compact EvidenceSummary view while retaining empty evidence collections.
    Empty collections distinguish unobserved families from omitted context.
    """
    d = _strip_target_identity_for_prompt(to_jsonable(evidence))
    KEEP_ALWAYS = {
        "method_health", "axis_stats", "joint_patterns", "recipes",
        "diagnostic_axis_stats", "diagnostic_alt_model_scores",
        "diagnosis_outcomes",
        "examples", "exemplars", "strategy_feedback", "route_values",
        "metric_availability", "recent_ticks_history",
        "diagnostic_chain_backlog", "refilter_role_health",
        "production_panel_selected_ids", "production_panel_diversity_bins",
        "production_panel_gap_reasons", "production_near_miss_ids",
        "stuck_lineage_roots",
    }
    # Keep populated route evidence but omit the all-open placeholder.
    EXCLUDE_WHILE_STUB = {"llm_health"}

    def _is_route_stub(value: object) -> bool:
        return (
            isinstance(value, dict)
            and value.get("raw_routed") == 0
            and value.get("score_files_completed") == 0
            and value.get("backlog_used") == 0
            and value.get("backlog_cap") == 96
        )

    view = {
        k: v for k, v in d.items()
        if k not in EXCLUDE_WHILE_STUB
        and not (k == "route_health" and _is_route_stub(v))
        and (v not in (None, "") or k in KEEP_ALWAYS)
    }
    # #10: drop the empty {} refold-probe block (the v-not-in-None rule keeps
    # empty dicts) so it is surfaced ONLY on ticks where refold probes ran.
    if not view.get("refold_probe_outcomes"):
        view.pop("refold_probe_outcomes", None)
    # Expose the actual compute-window meaning in the prompt while preserving archive
    # field names. Round displayed floats to four decimal places.
    if "worker_gpu_h_last_3_ticks" in view:
        view["worker_gpu_h_window"] = view.pop("worker_gpu_h_last_3_ticks")
    view["diagnostic_driver_tldr"] = diagnostic_driver_tldr_for_prompt(evidence)
    return _compact_prompt_view(
        _round_floats_for_prompt(
            _annotate_parent_bound_method_health(
                _trim_axis_noise(
                    _collapse_dedup_provenance(
                        _collapse_diversity_summary(
                            _collapse_objective_summary(view)
                        )
                    )
                )
            )
        )
    )


_DEDUP_PROVENANCE_KEYS = (
    "foldseek_su_status", "foldseek_su_coverage", "foldseek_archive_status",
    "foldseek_archive_coverage", "foldseek_archive_result_scope",
    "structure_dedup_scope", "structure_dedup_fallback_count",
    "whole_archive_structure_dedup_scope", "whole_archive_structure_dedup_fallback_count",
    "near_miss_dedup_status", "near_miss_dedup_coverage",
    "sequence_dedup_status", "sequence_dedup_coverage", "charged_gpu_h_scope",
)


_OBJECTIVE_SUMMARY_KEYS = (
    "run_su_count", "run_su_count_delta", "run_su_hwm", "run_su_hwm_delta",
    "run_su_per_worker_wall_gpu_h_total",
    "run_su_hwm_per_worker_wall_gpu_h_total",
    "worker_wall_gpu_count", "worker_wall_gpu_h_total",
    "worker_gpu_h_total", "worker_gpu_h_window",
    "run_su_per_worker_gpu_h_total", "su_per_gpu_h_recent",
    "gpu_h_since_last_su", "ticks_since_last_su",
    "strict_count", "global_new_strict",
    "charged_gpu_count", "charged_gpu_h_total", "charged_gpu_h_recent",
    "charged_gpu_h_scope", "run_su_per_charged_gpu_h_total",
    "run_su_per_charged_gpu_h_recent",
)


def _collapse_objective_summary(view: dict[str, Any]) -> dict[str, Any]:
    """Expose one objective block instead of many denominator-like scalars.

    The raw EvidenceSummary keeps all fields for archive/replay compatibility.
    The prompt view should not ask the LLM to choose between similarly named
    totals. Live allocation optimizes SU@TM0.6 per worker-wall GPU-h;
    completed worker-GPU-h is route-attribution feedback; charged GPU-h is
    audit-only reservation overhead.
    """
    vals = {k: view.pop(k) for k in _OBJECTIVE_SUMMARY_KEYS if k in view}
    if not vals:
        return view

    summary: dict[str, Any] = {
        "headline_objective": "live_SU_TM0.6_per_worker_wall_GPUh",
        "live_su_tm06": vals.get("run_su_count"),
        "live_su_tm06_recent_delta": vals.get("run_su_count_delta"),
        "live_su_tm06_hwm": vals.get("run_su_hwm"),
        "live_su_tm06_hwm_delta": vals.get("run_su_hwm_delta"),
        "su_per_worker_wall_gpu_h_total": vals.get("run_su_per_worker_wall_gpu_h_total"),
        "su_hwm_per_worker_wall_gpu_h_total": vals.get("run_su_hwm_per_worker_wall_gpu_h_total"),
        "worker_wall_gpu_count": vals.get("worker_wall_gpu_count"),
        "worker_wall_gpu_h_total": vals.get("worker_wall_gpu_h_total"),
        "completed_worker_gpu_h_total": vals.get("worker_gpu_h_total"),
        "completed_worker_gpu_h_window": vals.get("worker_gpu_h_window"),
        "route_learning_su_per_completed_worker_gpu_h_total": vals.get("run_su_per_worker_gpu_h_total"),
        "recent_su_per_completed_worker_gpu_h": vals.get("su_per_gpu_h_recent"),
        "dry_since_last_su_worker_gpu_h": vals.get("gpu_h_since_last_su"),
        "ticks_since_last_su": vals.get("ticks_since_last_su"),
        "raw_strict_count": vals.get("strict_count"),
    }
    charged = {
        "charged_gpu_count": vals.get("charged_gpu_count"),
        "charged_gpu_h_total": vals.get("charged_gpu_h_total"),
        "charged_gpu_h_recent": vals.get("charged_gpu_h_recent"),
        "charged_gpu_h_scope": vals.get("charged_gpu_h_scope"),
        "su_per_charged_gpu_h_total": vals.get("run_su_per_charged_gpu_h_total"),
        "su_per_charged_gpu_h_recent": vals.get("run_su_per_charged_gpu_h_recent"),
        "role": "audit_only_not_decision_objective",
    }
    charged = {k: v for k, v in charged.items() if v not in (None, "", [], {})}
    if len(charged) > 1:
        summary["charged_gpu_audit"] = charged
    view["objective_summary"] = {
        k: v for k, v in summary.items() if v not in (None, "", [], {})
    }
    return view


_DIVERSITY_SUMMARY_KEYS = (
    "duplicate_fraction", "top_bin_share", "strict_per_su_total",
    "strict_duplicate_collapse_signal", "strict_su_top_bin_share",
    "strict_su_tm08_status", "strict_su_tm08_coverage",
    "strict_su_tm08_recent_count", "strict_su_live_recent_count",
    "strict_su_tm08_delta_vs_live", "strict_su_tm08_live_split_ratio",
    "strict_su_tm05_recent_count",
    "strict_su_tm08_delta_vs_tm05", "strict_su_tm08_split_ratio",
    "strict_su_tm08_result_scope", "seq_unique_strict_count",
    "seq_unique_strict_delta", "joint_struct_seq_unique_count",
    "seq_duplicate_fraction", "top_seq_bin_share",
)


def _collapse_diversity_summary(view: dict[str, Any]) -> dict[str, Any]:
    """Collect diversity/collapse auxiliaries into one prompt block.

    TM0.6 Foldseek SU is the live objective. TM0.8 and sequence metrics are
    auxiliary diagnostics for basin collapse and panel diversity; grouping them
    prevents the prompt from looking like several independent objectives.
    """
    vals = {k: view.pop(k) for k in _DIVERSITY_SUMMARY_KEYS if k in view}
    if not vals:
        return view
    summary: dict[str, Any] = {
        "live_structural_novelty": "Foldseek_SU_TM0.6",
        "raw_strict_per_live_su": vals.get("strict_per_su_total"),
        "strict_duplicate_collapse_signal": vals.get("strict_duplicate_collapse_signal"),
        "recent_all_scored_live_top_bin_share": vals.get("top_bin_share"),
        "recent_all_scored_duplicate_fraction": vals.get("duplicate_fraction"),
        "recent_strict_records_live_su_bin_share": vals.get("strict_su_top_bin_share"),
        "tm08_recent_strict_bins": vals.get("strict_su_tm08_recent_count"),
        "live_recent_strict_bins": vals.get("strict_su_live_recent_count", vals.get("strict_su_tm05_recent_count")),
        "tm08_minus_live_recent_bins": vals.get("strict_su_tm08_delta_vs_live", vals.get("strict_su_tm08_delta_vs_tm05")),
        "tm08_live_recent_split_ratio": vals.get("strict_su_tm08_live_split_ratio", vals.get("strict_su_tm08_split_ratio")),
        "tm08_status": vals.get("strict_su_tm08_status"),
        "tm08_coverage": vals.get("strict_su_tm08_coverage"),
        "tm08_scope": vals.get("strict_su_tm08_result_scope"),
        "seq_unique_strict_count": vals.get("seq_unique_strict_count"),
        "seq_unique_strict_delta": vals.get("seq_unique_strict_delta"),
        "joint_struct_seq_unique_count": vals.get("joint_struct_seq_unique_count"),
        "seq_duplicate_fraction": vals.get("seq_duplicate_fraction"),
        "top_seq_bin_share": vals.get("top_seq_bin_share"),
        "interpretation": "TM0.6 collapse values are live-control signals; optional TM0.8/sequence values are auxiliary panel/ablation signals; live objective is TM0.6 SU; report TM0.5/0.6/0.8 post-hoc",
    }
    view["diversity_summary"] = {
        k: v for k, v in summary.items() if v not in (None, "", [], {})
    }
    return view


def _annotate_parent_bound_method_health(view: dict[str, Any]) -> dict[str, Any]:
    """Mark parent-required action-family rollups before prompt compaction.

    `method_health[proteinmpnn_redesign].chained_strict_yield_su=1` is useful
    action-level credit, but it is not a standalone de-novo generator result:
    the exact upstream route is `root_family -> proteinmpnn_redesign` in
    route_values. Without this marker, the LLM can read the rollup as
    "ProteinMPNN alone produced SU" and lose the parent/backbone provenance.
    """
    mh = view.get("method_health")
    if not isinstance(mh, dict):
        return view
    for fam, row in mh.items():
        if fam not in PARENT_BOUND_ACTION_FAMILIES or not isinstance(row, dict):
            continue
        row.setdefault("parent_bound_rollup", True)
        row.setdefault("standalone_generator", False)
        row.setdefault("interpretation", "action-family rollup only; use exact route_values root_family->action_family rows for upstream provenance and route SU/GPU-h")
        if row.get("chained_strict_yield_su"):
            row.setdefault("provenance_warning", "chained SU is credited through parent-bound route lineage, not this action family alone")
    return view


def _collapse_dedup_provenance(view: dict[str, Any]) -> dict[str, Any]:
    """Summarize deduplication provenance under a single dedup_trust key.

    Use "ok" when trusted; otherwise include the degraded fields needed to interpret
    SU evidence.
    """
    prov = {k: view.pop(k) for k in _DEDUP_PROVENANCE_KEYS if k in view}
    if not prov:
        return view
    fsu = prov.get("foldseek_su_status")
    fallbacks = sum(
        int(prov.get(k, 0) or 0)
        for k in ("structure_dedup_fallback_count",
                  "whole_archive_structure_dedup_fallback_count")
    )
    cov = prov.get("foldseek_su_coverage")
    near_status = prov.get("near_miss_dedup_status")
    near_cov = prov.get("near_miss_dedup_coverage")
    seq_status = prov.get("sequence_dedup_status")
    seq_cov = prov.get("sequence_dedup_coverage")
    strict_like_count = int(view.get("strict_count") or view.get("run_su_count") or 0)
    near_like_count = int(view.get("near_miss_count") or 0)
    near_trusted = near_miss_dedup_trusted(
        near_status,
        near_cov,
        near_miss_count=near_like_count,
    )
    seq_trusted = (
        seq_status in (None, "ok", "cached_ok", "no_sequences", "no_strict", "disabled", "legacy_or_unknown")
        and (seq_cov is None or seq_cov >= 0.999 or seq_status in ("no_sequences", "no_strict", "disabled"))
    )
    healthy = (
        (fsu in ("ok", "no_strict") or (fsu == "disabled" and strict_like_count == 0))
        and fallbacks == 0
        # MUST match the controller's su_dedup_trusted gate: status=='ok' with
        # coverage<0.999 means official SU evidence is incomplete/untrusted, not
        # a fallback count. Near-miss and sequence coverage are advisory, but if
        # they degrade, keep their provenance visible so the LLM does not
        # over-trust raw rescue/diversity counts.
        and (cov is None or cov >= 0.999)
        and near_trusted
        and seq_trusted
    )
    if healthy:
        view["dedup_trust"] = "ok"
    else:
        # degraded: surface the full provenance (rare) so coverage (incl. 0.0)
        # and statuses reach the LLM. Drop only None/"" — never numeric 0.0.
        view["dedup_trust"] = {
            k: v for k, v in prov.items() if v not in (None, "")
        } or "degraded"
    return view


def _trim_axis_noise(view: dict[str, Any]) -> dict[str, Any]:
    """Remove redundant or unset axis fields from the LLM view.

    Omit median_calibrated when it duplicates median_raw under provisional
    calibration, and omit unused acceptance/quality fields on strict axes.
    """
    for block in ("axis_stats", "diagnostic_axis_stats"):
        stats = view.get(block)
        if not isinstance(stats, dict):
            continue
        for s in stats.values():
            if not isinstance(s, dict):
                continue
            if s.get("median_calibrated") == s.get("median_raw"):
                s.pop("median_calibrated", None)
            if block == "axis_stats":  # strict axes: two-tier fields are inert
                for k in ("pass_threshold", "quality_threshold", "below_accept_count"):
                    s.pop(k, None)
    return view


def _round_floats_for_prompt(obj: Any, ndigits: int = 4) -> Any:
    """Recursively round floats to drop false-precision noise from the prompt."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats_for_prompt(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats_for_prompt(v, ndigits) for v in obj]
    return obj


def _short_route_id(strategy_key: Any) -> str | None:
    if not strategy_key:
        return None
    raw = str(strategy_key)
    return "route::" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def route_evidence_refs_for_validation(evidence: EvidenceSummary) -> set[str]:
    """Evidence refs exposed through compact route/strategy prompt rows.

    `build_evidence_for_prompt()` replaces verbose route strategy keys with
    short `route::<hash>` IDs to keep prompts under context limits. The schema
    validator must accept exactly those prompt-visible IDs; otherwise a
    correctly grounded LLM citation is rejected as `unknown_evidence_refs`,
    forcing a fallback Selector path.
    """
    refs: set[str] = set()

    def _get(row: Any, key: str, default: Any = None) -> Any:
        if isinstance(row, dict):
            return row.get(key, default)
        return getattr(row, key, default)

    def _add_strategy_key(value: Any) -> None:
        if not value:
            return
        raw = str(value)
        refs.add(raw)
        suffix = raw.rsplit("#", 1)[-1] if "#" in raw else ""
        if suffix and suffix.replace("_", "").replace("-", "").isalnum():
            refs.add("route::" + suffix)
        short = _short_route_id(raw)
        if short:
            refs.add(short)

    for row in getattr(evidence, "route_values", None) or []:
        _add_strategy_key(_get(row, "strategy_key"))
        _add_strategy_key(_get(row, "parent_strategy_key"))
        route_id = _get(row, "route_id")
        if route_id:
            refs.add(str(route_id))
        for rid in (_get(row, "evidence_refs", []) or []):
            if rid:
                refs.add(str(rid))

    for row in getattr(evidence, "strategy_feedback", None) or []:
        _add_strategy_key(_get(row, "strategy_key"))
        strategy_id = _get(row, "strategy_id")
        if strategy_id:
            refs.add(str(strategy_id))
        for key in ("representative_failure_ids", "representative_near_miss_ids"):
            for rid in (_get(row, key, []) or []):
                if rid:
                    refs.add(str(rid))

    return refs


def _compact_config_delta(config: Any, *, max_items: int = 8) -> dict[str, Any]:
    if not isinstance(config, dict) or not config:
        return {}
    out: dict[str, Any] = {}
    for k in sorted(config)[:max_items]:
        v = config[k]
        if isinstance(v, float):
            v = round(v, 4)
        out[str(k)] = v
    if len(config) > max_items:
        out["_truncated_keys"] = len(config) - max_items
    return out


def _route_value_safe_record_allowed(row: dict[str, Any]) -> bool:
    route_role = str(row.get("route_role") or "")
    try:
        canonical_refilter_gpu_h = float(row.get("canonical_refilter_gpu_h") or 0.0)
    except (TypeError, ValueError):
        canonical_refilter_gpu_h = 0.0
    return canonical_refilter_gpu_h <= 0.0 and "score_conversion" not in route_role


def _route_value_rate_signal(row: dict[str, Any], *, target_dry_gpu_h: float = 0.0) -> tuple[Any, str, Any]:
    """Prompt ranking rate, its source, and lifetime memory.

    Lifetime route SU/GPU-h is useful memory, but it is not current exploit
    evidence. Keep it separate so prompt tables cannot present old value as a
    recent route winner.
    """
    lifetime = row.get("new_su_per_route_gpu_h")
    value = row.get("gpu_recent_new_su_per_route_gpu_h")
    if value is not None:
        return value, "gpu_recent", lifetime
    value = row.get("medium_recent_new_su_per_route_gpu_h")
    if value is not None:
        return value, "medium_recent", lifetime
    record_recent_rate = row.get(
        "record_recent_new_su_per_route_gpu_h",
        row.get("recent_new_su_per_route_gpu_h"),
    )
    if (
        record_recent_rate is not None
        and _route_value_safe_record_allowed(row)
        and target_dry_gpu_h < 12.0
    ):
        return record_recent_rate, "record_recent", lifetime
    return None, "lifetime_memory", lifetime


def _route_value_decision_rate(row: dict[str, Any], *, target_dry_gpu_h: float = 0.0) -> Any:
    return _route_value_rate_signal(row, target_dry_gpu_h=target_dry_gpu_h)[0]


def _route_value_recent_su_for_ranking(row: dict[str, Any], *, target_dry_gpu_h: float = 0.0) -> float:
    values = [
        float(row.get("new_su_recent_gpu") or 0.0),
        float(row.get("medium_recent_new_su") or 0.0),
    ]
    if _route_value_safe_record_allowed(row) and target_dry_gpu_h < 12.0:
        values.append(float(row.get("record_recent_new_su", row.get("new_su_recent")) or 0.0))
    return max(values)


def compact_route_value_for_prompt(row: Any, *, target_dry_gpu_h: float = 0.0) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    record_recent_new_su = row.get("record_recent_new_su", row.get("new_su_recent"))
    record_recent_route_gpu_h = row.get("record_recent_route_gpu_h", row.get("recent_route_gpu_h"))
    record_recent_rate = row.get(
        "record_recent_new_su_per_route_gpu_h",
        row.get("recent_new_su_per_route_gpu_h"),
    )
    rate, rate_source, lifetime_rate = _route_value_rate_signal(row, target_dry_gpu_h=target_dry_gpu_h)
    route_id = _short_route_id(row.get("strategy_key"))
    scope = row.get("scope")
    action_family = row.get("action_family") or row.get("family")
    root_family = row.get("root_family")
    compact = {
        "route_id": route_id,
        "scope": scope,
        "status": row.get("status"),
        "marginal_status": row.get("marginal_status"),
        "family": row.get("family"),
        "action_family": action_family,
        "root_family": root_family,
        "operator_id": row.get("operator_id"),
        "route_role": row.get("route_role"),
        "refilter_role": row.get("refilter_role"),
        "config_delta": _compact_config_delta(row.get("config_delta")),
        "attempts": row.get("attempts"),
        "completions": row.get("completions"),
        "new_su": row.get("new_su"),
        "record_recent_new_su": record_recent_new_su,
        "new_su_recent_gpu": row.get("new_su_recent_gpu"),
        "pending_score_conversion_count": row.get("pending_score_conversion_count"),
        "strict_count": row.get("strict_count"),
        "near_miss_count": row.get("near_miss_count"),
        "near_miss_recent": row.get("near_miss_recent"),
        "route_gpu_h": row.get("route_gpu_h"),
        "record_recent_route_gpu_h": record_recent_route_gpu_h,
        "gpu_recent_route_gpu_h": row.get("gpu_recent_route_gpu_h"),
        "medium_recent_new_su": row.get("medium_recent_new_su"),
        "medium_recent_route_gpu_h": row.get("medium_recent_route_gpu_h"),
        "new_su_per_route_gpu_h": row.get("new_su_per_route_gpu_h"),
        "record_recent_new_su_per_route_gpu_h": record_recent_rate,
        "gpu_recent_new_su_per_route_gpu_h": row.get("gpu_recent_new_su_per_route_gpu_h"),
        "medium_recent_new_su_per_route_gpu_h": row.get("medium_recent_new_su_per_route_gpu_h"),
        "value_rate_for_ranking": rate,
        "value_rate_source": rate_source,
        "lifetime_su_per_route_gpu_h": lifetime_rate,
        "strict_per_su": row.get("strict_per_su"),
        "duplicate_bin_fraction": row.get("duplicate_bin_fraction"),
        "strict_quality_n_unique_bins": row.get("strict_quality_n_unique_bins"),
        "strict_quality_median": row.get("strict_quality_median"),
        "strict_quality_p25": row.get("strict_quality_p25"),
        "strict_quality_axis_margins": row.get("strict_quality_axis_margins"),
        "diagnostic_improvement_score": row.get("diagnostic_improvement_score"),
        "diagnostic_improvement_axes": list(row.get("diagnostic_improvement_axes") or [])[:3],
        "diagnostic_improvement_n": row.get("diagnostic_improvement_n"),
    }
    if scope == "family" and action_family in PARENT_BOUND_ACTION_FAMILIES:
        compact["parent_bound_rollup"] = True
        compact["standalone_generator"] = False
        compact["interpretation"] = "rollup only; exact route rows carry root_family->action_family provenance"
    if scope == "route" and root_family and action_family and root_family != action_family:
        compact["route_lineage"] = f"{root_family}->{action_family}"
    return {k: v for k, v in compact.items() if v not in (None, [], {}, "")}


# Compatibility alias for downstream callers.
_compact_route_value_row = compact_route_value_for_prompt


def _route_value_priority(row: dict[str, Any], *, target_dry_gpu_h: float = 0.0) -> tuple[int, float, float, float]:
    status_rank = {
        "promote": 0,
        "awaiting_score_conversion": 1,
        "healthy": 2,
        "diversify": 3,
        "collapse_risk": 4,
        "defer": 5,
        "observed": 6,
        "untried": 7,
        "advisory": 8,
    }
    status = str(row.get("status", ""))
    rate, rate_source, lifetime_rate = _route_value_rate_signal(row, target_dry_gpu_h=target_dry_gpu_h)
    recent_su = _route_value_recent_su_for_ranking(row, target_dry_gpu_h=target_dry_gpu_h)
    source_rank = {
        "gpu_recent": 0,
        "medium_recent": 1,
        "record_recent": 2,
        "lifetime_memory": 3,
    }.get(rate_source, 4)
    current_value_rank = 0 if recent_su > 0 and float(rate or 0.0) > 0.0 else 1
    return (
        current_value_rank,
        source_rank,
        status_rank.get(status, 9),
        -float(rate or 0.0),
        -recent_su,
        -float(row.get("near_miss_recent") or 0.0),
        -float(lifetime_rate or 0.0),
    )


def _compact_strategy_feedback_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    compact = {
        "strategy_id": _short_route_id(row.get("strategy_key")),
        "family": row.get("family"),
        "root_family": row.get("root_family"),
        "parent_strategy_id": _short_route_id(row.get("parent_strategy_key")),
        "feedback_scope": row.get("feedback_scope"),
        "operator_id": row.get("operator_id"),
        "config_delta": _compact_config_delta(row.get("config_delta")),
        "attempts": row.get("attempts"),
        "result_count": row.get("result_count"),
        "gpu_h": row.get("gpu_h"),
        "near_miss_count": row.get("near_miss_count"),
        "strict_success_count": row.get("strict_success_count"),
        "panel_ready_count": row.get("panel_ready_count"),
        "dominant_failure_axes": row.get("dominant_failure_axes"),
        "last_tick": row.get("last_tick"),
        "repeat_policy": row.get("repeat_policy"),
        "representative_failure_ids": list(row.get("representative_failure_ids") or [])[:2],
        "representative_near_miss_ids": list(row.get("representative_near_miss_ids") or [])[:2],
    }
    return {k: v for k, v in compact.items() if v not in (None, [], {}, "")}


def _compact_recipe_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    compact = dict(row)
    compact["config_delta"] = _compact_config_delta(row.get("config_delta"))
    if isinstance(compact.get("representative_result_ids"), list):
        compact["representative_result_ids"] = compact["representative_result_ids"][:2]
    return compact


def _compact_exemplar_row(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict):
        return row
    compact = dict(row)
    compact["config_delta"] = _compact_config_delta(row.get("config_delta"))
    return compact


def _compact_prompt_view(view: dict[str, Any]) -> dict[str, Any]:
    """Bound LLM prompt growth without changing archived EvidenceSummary.

    The archive can accumulate many route outcomes. For planning, keep compact
    per-route economics and short IDs; drop long per-result evidence-ref lists
    and full route signatures that caused 65k-context failures on productive
    targets.
    """
    parent_artifact_ids = view.pop("parent_artifact_result_ids", None)
    if isinstance(parent_artifact_ids, list):
        # The full concrete-ID list is archived for Builder/dispatch feasibility.
        # The LLM only needs to know whether parent-bound rescue is possible;
        # dumping hundreds of IDs into the prompt wastes context and attention.
        view["parent_artifact_result_count"] = len(parent_artifact_ids)
        view["parent_artifacts_available"] = bool(parent_artifact_ids)
    route_values = view.get("route_values")
    if isinstance(route_values, list):
        # Exact route rows must lead family rollups. Parent-bound actions such as
        # BindCraft->ProteinMPNN are scientifically different from a context-free
        # "ProteinMPNN worked" rollup, and the LLM should see the route lineage
        # before the action-family aggregate. Family rows remain visible after
        # exact routes so untried launchable families are still represented.
        objective_summary = view.get("objective_summary") if isinstance(view.get("objective_summary"), dict) else {}
        dry_value = view.get("gpu_h_since_last_su")
        if dry_value is None:
            dry_value = objective_summary.get("dry_since_last_su_worker_gpu_h")
        try:
            target_dry_gpu_h = float(dry_value or 0.0)
        except (TypeError, ValueError):
            target_dry_gpu_h = 0.0
        family_rows = [r for r in route_values if isinstance(r, dict) and r.get("scope") == "family"]
        route_rows = [r for r in route_values if not (isinstance(r, dict) and r.get("scope") == "family")]
        route_rows = sorted(
            route_rows,
            key=lambda r: _route_value_priority(
                r if isinstance(r, dict) else {},
                target_dry_gpu_h=target_dry_gpu_h,
            ),
        )[:18]
        view["route_values"] = [
            compact_route_value_for_prompt(r, target_dry_gpu_h=target_dry_gpu_h)
            for r in (route_rows + family_rows)
        ]
    if isinstance(view.get("strategy_feedback"), list):
        view["strategy_feedback"] = [
            _compact_strategy_feedback_row(r) for r in view["strategy_feedback"][:8]
        ]
    if isinstance(view.get("recipes"), list):
        view["recipes"] = [_compact_recipe_row(r) for r in view["recipes"][:10]]
    if isinstance(view.get("exemplars"), list):
        view["exemplars"] = [_compact_exemplar_row(r) for r in view["exemplars"][:8]]
    if isinstance(view.get("examples"), list):
        view["examples"] = [_compact_exemplar_row(r) for r in view["examples"][:4]]
    if isinstance(view.get("recent_ticks_history"), list):
        view["recent_ticks_history"] = view["recent_ticks_history"][-10:]
    return view


def diagnostic_driver_tldr(ev: EvidenceSummary) -> str:
    """Structured diagnostic-driver digest used in both JSON evidence and TL;DR."""
    diag_stats = getattr(ev, "diagnostic_axis_stats", {}) or {}
    if not diag_stats:
        return "none"
    try:
        from .evidence_reducer import DIAGNOSTIC_AXIS_REMEDIATION, DIAGNOSTIC_AXIS_THRESHOLDS

        def _fmt_diag(v: object) -> str:
            return f"{float(v):.3g}" if isinstance(v, (int, float)) else "NA"

        def _source_hint(v: object) -> str:
            fam_counts = getattr(v, "source_families", None)
            if not isinstance(fam_counts, dict) or not fam_counts:
                return ""

            def _root(fam: str) -> str:
                return "complexa" if fam.startswith("complexa_") else fam

            collapsed: dict[str, int] = {}
            for fam, count in fam_counts.items():
                try:
                    n = int(count)
                except (TypeError, ValueError):
                    continue
                if n <= 0:
                    continue
                root = _root(str(fam))
                collapsed[root] = collapsed.get(root, 0) + n
            if not collapsed:
                return ""
            top = sorted(collapsed.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
            return "[src=" + ",".join(f"{fam}:{n}" for fam, n in top) + "]"

        levered_drivers = []
        corroborating_drivers = []
        for dax, dst in diag_stats.items():
            n_axis = int(getattr(dst, "n", 0) or 0)
            if n_axis < 3:
                continue
            fail = int(getattr(dst, "fail_count", 0) or 0)
            near = int(getattr(dst, "near_pass_count", 0) or 0)
            deficit = getattr(dst, "median_deficit", None)
            if fail <= 0 and not (isinstance(deficit, (int, float)) and deficit > 0):
                continue
            _pass_thr, _quality_thr, direction, _margin = DIAGNOSTIC_AXIS_THRESHOLDS.get(
                dax, (None, None, "unknown", 0.0)
            )
            raw_deficit = float(deficit) if isinstance(deficit, (int, float)) else 0.0
            norm_base = float(_margin or 0.0) or max(abs(float(_quality_thr or 0.0)), 1.0)
            norm_deficit = raw_deficit / norm_base
            clipped_norm_deficit = max(0.0, min(norm_deficit, 5.0))
            driver_rate = (fail + 0.5 * near) / max(1, n_axis)
            role = "levered" if DIAGNOSTIC_AXIS_REMEDIATION.get(dax) else "corroborates"
            severity = driver_rate * (1.0 + clipped_norm_deficit)
            med = getattr(dst, "median_raw", None)
            text = (
                f"{dax}{_source_hint(dst)}:{direction},fail={fail}/{n_axis},near={near},"
                f"median={_fmt_diag(med)},def={_fmt_diag(deficit)},"
                f"norm_def={_fmt_diag(norm_deficit)},{role}"
            )
            (levered_drivers if role == "levered" else corroborating_drivers).append((severity, text))
        selected = [x for _score, x in sorted(levered_drivers, reverse=True)[:3]]
        selected += [x for _score, x in sorted(corroborating_drivers, reverse=True)[:1]]
        if len(selected) < 4:
            used = set(selected)
            leftovers = sorted(levered_drivers + corroborating_drivers, reverse=True)
            selected += [x for _score, x in leftovers if x not in used][: 4 - len(selected)]
        return ";".join(selected[:4]) if selected else "none"
    except Exception as exc:  # noqa: BLE001
        return f"error:{type(exc).__name__}"


def diagnostic_driver_tldr_for_prompt(ev: EvidenceSummary) -> str:
    """Return the archived diagnostic digest when available, else compute it.

    Live ticks persist this field in EvidenceSummary for auditability. Direct
    reducer/test callers and legacy archives may not have it, so preserve the
    previous deterministic prompt behavior as a fallback.
    """
    stored = getattr(ev, "diagnostic_driver_tldr", None)
    if isinstance(stored, str) and stored.strip() and stored.strip() != "none":
        return stored.strip()
    return diagnostic_driver_tldr(ev)


def evidence_tldr(ev: EvidenceSummary) -> str:
    """Summarize decision-relevant evidence with units at the start of the prompt."""
    from collections import Counter
    p: list[str] = [f"state={ev.state_label}"]
    p.append(f"run_SU={ev.run_su_count}(+{ev.run_su_count_delta} recent)")
    p.append(f"raw_strict={ev.strict_count}")
    strict_per_su = getattr(ev, "strict_per_su_total", None)
    if strict_per_su is not None:
        p.append(f"strict_per_SU_total={strict_per_su:.3g}")
    if getattr(ev, "strict_duplicate_collapse_signal", False):
        p.append("strict_duplicate_collapse=YES: raw strict is not unique SU; diversify roots/families/configs")
    # Surface the basin-collapse read the diversify clamp uses.
    # strict_su_top_bin_share is recent raw strict records concentrated into
    # live SU bins — stronger for the SU objective than top_bin_share
    # (live TM0.6 over all scored structures). Trigger = max(...) >= 0.5 (mirrors the
    # deterministic clamp). Absence => insufficient strict records/bins, not
    # "diverse".
    _tbs = getattr(ev, "top_bin_share", None)
    _sstbs = getattr(ev, "strict_su_top_bin_share", None)
    if _tbs is not None or _sstbs is not None:
        _bp: list[str] = []
        if _tbs is not None:
            _bp.append(f"top_bin_share(live,all)={_tbs:.2f}")
        if _sstbs is not None:
            _bp.append(f"strict_su_top_bin_share(live,strict_records)={_sstbs:.2f}")
        _trig = max([x for x in (_tbs, _sstbs) if x is not None], default=0.0)
        p.append("basin=" + " ".join(_bp) + (" COLLAPSE>=0.5:diversify" if _trig >= 0.5 else ""))
    # Headline = the LIVE OBJECTIVE: SU per worker-wall GPU-h. Completed
    # worker-GPU windows remain route-attribution feedback; charged is audit-only
    # and intentionally not surfaced in the TL;DR.
    wall_obj = getattr(ev, "run_su_per_worker_wall_gpu_h_total", None)
    if wall_obj is not None:
        p.append(f"SU/worker-wall-GPUh_total={wall_obj:.3g}")
    hwm_obj = getattr(ev, "run_su_hwm_per_worker_wall_gpu_h_total", None)
    hwm_delta = getattr(ev, "run_su_hwm_delta", None)
    if hwm_obj is not None:
        p.append(f"SU_HWM/worker-wall-GPUh_total={hwm_obj:.3g}(+{hwm_delta or 0} HWM recent)")
    if ev.su_per_gpu_h_recent is not None:
        p.append(f"route_feedback_SU/completed-worker-GPUh_recent={ev.su_per_gpu_h_recent:.3g}")
    obj_total = getattr(ev, "run_su_per_worker_gpu_h_total", None)
    if obj_total is not None:
        p.append(f"route_feedback_SU/completed-worker-GPUh_total={obj_total:.3g}")
    gph_dry = getattr(ev, "gpu_h_since_last_su", None)
    if gph_dry is not None and gph_dry > 0:
        t_dry = getattr(ev, "ticks_since_last_su", None)
        p.append(
            f"dry_since_last_SU={gph_dry:.3g}worker-GPUh"
            + (f"/{t_dry}ticks" if t_dry else "")
        )
    er = getattr(ev, "execution_realization", {}) or {}
    er_by_family = er.get("by_family", {}) if isinstance(er, dict) else {}
    if isinstance(er_by_family, dict):
        gaps: list[str] = []
        for fam, row in er_by_family.items():
            if not isinstance(row, dict):
                continue
            try:
                selected = int(row.get("selected", 0) or 0)
                started = int(row.get("started", 0) or 0)
                deferred = int(row.get("dispatch_deferred", 0) or 0)
                pending = int(row.get("selected_not_started", 0) or 0)
            except (TypeError, ValueError):
                continue
            if selected >= 2 and selected - started >= 2 and started / max(1, selected) <= 0.50:
                extra = f",cap_deferred={deferred}" if deferred else ""
                gaps.append(
                    f"{fam}:selected={selected},started={started},pending={pending}{extra}"
                )
        if gaps:
            p.append("execution_gap_selected_not_started=" + ";".join(sorted(gaps)[:3]))
    blockers = Counter(
        e.diagnostic_blocking_axis for e in (ev.exemplars or [])
        if getattr(e, "diagnostic_blocking_axis", None)
    )
    if blockers:
        ax, n = blockers.most_common(1)[0]
        p.append(f"top_diagnostic_blocker={ax}(x{n} exemplars)")
    p.append("diagnostic_driver_tldr=" + diagnostic_driver_tldr_for_prompt(ev))
    # Appendix-I.4-style trigger, target-agnostic: interface placement is good
    # enough (iPAE passes) but fold confidence fails (pLDDT fails). Surface this
    # early so the Planner can choose pLDDT-aware MCTS/hallucination without
    # relying on target identity.
    hard_i4_n = 0
    for jp in ev.joint_patterns or []:
        if jp.axes == ("pLDDT", "iPAE") and jp.pattern == "A_fail_B_pass":
            hard_i4_n += int(jp.count)
    if hard_i4_n > 0:
        p.append(f"hard_target_signal=pLDDT_fail/iPAE_pass(x{hard_i4_n})")
    best = next((e for e in (ev.exemplars or []) if e.kind == "best"), None)
    near = next((e for e in (ev.exemplars or []) if e.kind == "near_miss"), None)
    if best is not None:
        p.append(f"best_exemplar={best.result_id}")
    if near is not None:
        nb = f"/blocked_on:{near.dominant_deficit_axis}" if near.dominant_deficit_axis else ""
        p.append(f"nearest_miss={near.result_id}{nb}")
    if ev.remaining_wall_h is not None:
        p.append(f"remaining_wall_h={ev.remaining_wall_h:.1f}")
    tm08_ratio = getattr(ev, "strict_su_tm08_live_split_ratio", getattr(ev, "strict_su_tm08_split_ratio", None))
    tm08 = getattr(ev, "strict_su_tm08_recent_count", None)
    live_bins = getattr(ev, "strict_su_live_recent_count", getattr(ev, "strict_su_tm05_recent_count", None))
    if tm08_ratio is not None and tm08 is not None and live_bins is not None:
        p.append(
            f"fine_diversity_aux=TM0.8_recent/live_recent {tm08}/{live_bins} "
            f"ratio={tm08_ratio:.2f}; live objective is TM0.6"
        )
    head = "TLDR (units: pLDDT 0-100; iPAE/ipTM/ipsae/min_ipae 0-1; binder_pLDDT_avg 0-1; " \
           "interface_dG REU; buried_sasa A^2; hotspot_rmsd A; boltzgen pae raw): " \
           + " | ".join(p)
    # Expose observed outcomes of attempts to improve blocking measurements.
    outcomes = getattr(ev, "diagnosis_outcomes", None)
    if outcomes:
        from .diagnosis_outcome import format_diagnosis_outcomes_tldr
        line = format_diagnosis_outcomes_tldr(outcomes)
        if line:
            head += "\n" + line
    # #10: surface the refold-probe answer so a structure-limited parent gets
    # REGENERATED/redesigned (which can mint SU) rather than refolded again.
    rp = getattr(ev, "refold_probe_outcomes", None)
    if rp and rp.get("probed"):
        sl = rp.get("structure_limited", 0)
        cl = rp.get("confirmed_limited", 0)
        ex = rp.get("example_parent_ids") or []
        seg = (
            f"refold_probe: {rp['probed']} probed | structure_limited={sl} "
            f"(sequence OK, backbone limited -> REGENERATE/proteinmpnn_redesign "
            f"these parents to mint SU; refold cannot) | confirmed_limited={cl} "
            f"(design itself bad -> pivot family/seed)"
        )
        if ex:
            seg += " | structure_limited_parents=" + ",".join(ex)
        head += "\n" + seg
    # Surface nonproductive routes and exhausted lineages alongside the detailed
    # evidence.
    def _mh(h, k, d=0):
        v = h.get(k, d) if isinstance(h, dict) else getattr(h, k, d)
        return d if v is None else v
    dead: list[str] = []
    stale: list[str] = []
    active_near: list[str] = []
    for fam, h in (getattr(ev, "method_health", {}) or {}).items():
        gpu_h = float(_mh(h, "cumulative_gpu_h", 0.0) or 0.0)
        strict = int(_mh(h, "strict_yield_su", 0) or 0)
        chained = int(_mh(h, "chained_strict_yield_su", 0) or 0)
        near = int(_mh(h, "near_miss_yield", 0) or 0)
        recent_near = int(_mh(h, "near_miss_yield_recent", 0) or 0)
        recent_su = float(_mh(h, "su_per_gpu_h_recent", 0.0) or 0.0)
        recent_chain = float(_mh(h, "chained_su_per_gpu_h_recent", 0.0) or 0.0)
        timeouts = int(_mh(h, "timeouts", 0) or 0)
        recent_signal = recent_su > 0.0 or recent_chain > 0.0 or recent_near > 0
        if gpu_h >= 1.5 and strict == 0 and chained == 0 and near == 0 and not recent_signal:
            dead.append(f"{fam}={gpu_h:.1f}GPU-h/0SU/0near/{timeouts}timeouts")
        elif gpu_h >= 3.0 and strict == 0 and chained == 0 and near > 0:
            msg = f"{fam}={gpu_h:.1f}GPU-h/0SU/{near}near(recent={recent_near})"
            if recent_signal:
                active_near.append(msg)
            else:
                stale.append(msg)
    stuck = getattr(ev, "stuck_lineage_roots", None) or []
    for s in stuck[:3]:
        dead.append(
            f"parent:{s.get('root_result_id')}({s.get('attempts')}x,{s.get('reason')})"
        )
    route_rows = [r for r in (getattr(ev, "route_values", None) or [])
                  if getattr(r, "scope", None) == "route"
                  and getattr(r, "status", None) in {"promote", "awaiting_score_conversion", "healthy", "diversify", "defer", "collapse_risk"}]
    if route_rows:
        def _rv_rate(r):
            row = r if isinstance(r, dict) else getattr(r, "__dict__", {})
            if isinstance(row, dict):
                return _route_value_decision_rate(row, target_dry_gpu_h=float(getattr(ev, "gpu_h_since_last_su", 0.0) or 0.0)) or 0.0
            return 0.0

        def _rv_recent_su(r):
            row = r if isinstance(r, dict) else getattr(r, "__dict__", {})
            if isinstance(row, dict):
                return _route_value_recent_su_for_ranking(row, target_dry_gpu_h=float(getattr(ev, "gpu_h_since_last_su", 0.0) or 0.0))
            return 0.0

        def _rv_priority(r):
            rate = float(_rv_rate(r) or 0.0)
            recent_su = _rv_recent_su(r)
            return (
                0 if recent_su > 0.0 and rate > 0.0 else 1,
                {"promote": 0, "awaiting_score_conversion": 1, "healthy": 2, "diversify": 3, "defer": 4, "collapse_risk": 5}.get(getattr(r, "status", ""), 9),
                -rate,
                -recent_su,
                -float(getattr(r, "route_gpu_h", 0.0) or 0.0),
            )

        bits = []
        for r in sorted(route_rows, key=_rv_priority)[:6]:
            bits.append(
                f"{getattr(r, 'status', '?')}/{getattr(r, 'marginal_status', 'observed')}:{getattr(r, 'strategy_key', '?')}"
                f" role={getattr(r, 'route_role', None) or getattr(r, 'refilter_role', None) or 'route'}"
                f" SU={getattr(r, 'new_su', 0)}(+{_rv_recent_su(r):.3g})"
                f" rate={_rv_rate(r):.3g}"
                f" gpu={float(getattr(r, 'route_gpu_h', 0.0) or 0.0):.2g}"
                f" pending_score={getattr(r, 'pending_score_conversion_count', 0)}"
                f" strict/SU={getattr(r, 'strict_per_su', None)}"
            )
        if bits:
            head += "\nROUTE VALUE / COST-NORMALIZED FEEDBACK: " + " | ".join(bits)


    cost_bits: list[str] = []
    if dead:
        cost_bits.append(
            "dead lanes; do NOT repeat without material config/parent/family change: "
            + " | ".join(dead)
        )
    if stale:
        cost_bits.append(
            "stale near-miss lanes; do not keep sampling unchanged, rescue the named blocker or pivot: "
            + " | ".join(stale)
        )
    if active_near:
        cost_bits.append(
            "active near-miss lanes; not strict success yet, use blocker-matched rescue/redesign: "
            + " | ".join(active_near)
        )
    if cost_bits:
        head += "\nNEGATIVE EVIDENCE / COST FEEDBACK: " + " || ".join(cost_bits)
    return head


def _hypothesis_outcomes_tldr(active_hypotheses: list[dict[str, Any]]) -> str:
    """Summarize hypothesis status and accumulated feedback for subsequent planning."""
    rows = []
    for h in active_hypotheses or []:
        st = h.get("status")
        if st in ("contradicted", "retired", "supported"):
            rows.append(
                f"{h.get('hypothesis_id', '?')}={st}"
                f"(sup{h.get('support_points', 0)}/con{h.get('contradiction_points', 0)}"
                f"/n{h.get('descendants_evaluated', 0)})"
            )
    if not rows:
        return ""
    return ("HYPOTHESIS OUTCOMES (do NOT re-propose a contradicted/retired "
            "hypothesis's config unless evidence materially changed; build on "
            "supported ones): " + " | ".join(rows[:8]))


def build_user_prompt(
    evidence: EvidenceSummary,
    active_hypotheses: list[dict[str, Any]],
    seed_action_families: list[str],
    *,
    available_families: list[str] | None = None,
    recent_critic_flags: list[str] | None = None,
) -> str:
    # Supply allowed settings only for currently available families.
    from .capability_registry import (
        default_registry,
        family_eval_budget_schema_for_prompt,
        family_params_schema_for_prompt,
    )
    if available_families is None:
        available_families = default_registry().feasible_families()
    schema = family_params_schema_for_prompt(default_registry(), families=available_families)
    budget_schema = family_eval_budget_schema_for_prompt(
        default_registry(), families=available_families
    )
    # Render family roles from the registry so prompt metadata matches feasibility
    # checks.
    reg_now = default_registry()
    family_role_table: dict[str, dict[str, object]] = {}
    for fam in sorted(available_families):
        cap = reg_now.get(fam)
        if cap is None:
            continue
        family_role_table[fam] = {
            "role": cap.role,
            "requires_parent_pdb": cap.requires_parent_pdb,
            "outputs_diagnostic_only": cap.outputs_diagnostic_only,
        }
    payload: dict[str, Any] = {
        "evidence": build_evidence_for_prompt(evidence),
        "active_hypotheses": _strip_target_identity_for_prompt(active_hypotheses),
        "seed_action_families": seed_action_families,
        "available_families_this_cluster": sorted(available_families),
        "family_role_table": family_role_table,
        "allowed_params_per_family": schema,
        "eval_budget_per_family": budget_schema,
        "instructions": (
            "Propose 1-4 HypothesisCards JSON. Each card must cite evidence "
            "from the EvidenceSummary axes/joint_patterns/method_health, with "
            "concrete predicted_metric_changes. Each card must include a short "
            "reasoning_trace: observed_signal (what evidence matters), inference "
            "(what that implies), and action_implication (why the proposed "
            "family/config follows). "
            "When the evidence carries `recent_ticks_history`, examine how the "
            "state has evolved across recent ticks — did a particular family "
            "produce strict successes or near-misses, did a rescue attempt "
            "improve a specific axis, did exploration keep cycling through "
            "the same configurations? Use that trajectory to choose between "
            "exploit (refine what's working), rescue (raise the dominant "
            "blocker axis), or explore (pivot family/paradigm) for the new "
            "hypotheses. "
            "`recommended_action_families` MUST be chosen from "
            "`available_families_this_cluster` only — other capabilities "
            "are not staged on this cluster and proposing them wastes a "
            "hypothesis slot. When proposing config_delta_suggestions, "
            "ONLY use parameter names that appear in "
            "`allowed_params_per_family` for that family — keys not in "
            "the schema will be dropped and audited. Values must fall in the "
            "listed [lo, hi] range (or enum). Also obey "
            "`eval_budget_per_family`: each config's formula must be <= cap. "
            "The individual ranges are deliberately broad; use the product cap "
            "to trade off width/depth/samples instead of assuming all maxima "
            "can be combined. Return JSON matching the schema."
        ),
    }
    cross_memory = cross_campaign_memory_for_prompt()
    if cross_memory is not None:
        payload["cross_campaign_memory"] = cross_memory
        payload["instructions"] = payload["instructions"] + (
            " The optional `cross_campaign_memory` block is historical advisory "
            "experience only. Use it only when its match_features are observable "
            "in the current EvidenceSummary and current action eligibility. It "
            "cannot override available_families_this_cluster, parent availability, "
            "evaluation requirements, deterministic safeguards, or current-target "
            "evidence. Do not use hidden target identity or target-class priors."
        )
    if seed_action_families:
        payload["instructions"] = payload["instructions"] + (
            " `seed_action_families` are deterministic bootstrap seeds injected "
            "by the controller to avoid an empty first tick. Treat them as a "
            "transparent scheduling prior, not as evidence that any family should "
            "win this target."
        )
    # Expose previous advisory flags for consideration without blocking proposals.
    if recent_critic_flags:
        payload["recent_critic_flags_last_tick"] = list(recent_critic_flags)
        payload["instructions"] = payload["instructions"] + (
            " The Critic LLM raised the listed flags about your previous "
            "hypotheses. These are advisory observations, not directives. "
            "Consider each flag against the current evidence; if a flag is "
            "well-supported by evidence, reflect it in your reasoning. If "
            "the evidence has shifted or the flag was speculative, your new "
            "hypotheses may diverge from it without justification."
        )
    head = evidence_tldr(evidence)
    _hyp_line = _hypothesis_outcomes_tldr(active_hypotheses)
    if _hyp_line:
        head += "\n" + _hyp_line
    return head + "\n" + json.dumps(payload, indent=None, sort_keys=True)


# ---------------------------------------------------------------------------
# Validation + parsing
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    """Find the first JSON object in the text.

    LLMs often wrap JSON in prose or code fences. Use JSONDecoder.raw_decode
    from each object-like start so braces inside string values do not confuse the
    fallback extractor.
    """
    text = text.strip()
    if not text:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _coerce_legacy_shape(obj: dict) -> dict:
    """Normalize common LLM drift before strict validation.

    Accepts:
      - "hypotheses" as alias for "cards"
      - per-card "title"+"abstract" concatenated into "claim"
      - predicted_metric_changes as dict {axis: {...}} (converted to list)
      - missing top-level "abstain"/"confidence"/"rationale" defaults
    """
    if "cards" not in obj and "hypotheses" in obj:
        obj["cards"] = obj.pop("hypotheses")
    obj.setdefault("abstain", False)
    obj.setdefault("confidence", 0.5)
    obj.setdefault("rationale", "")
    # Discard unsupported fields retained in older LLM responses.
    obj.pop("missing_candidate_requests", None)
    if isinstance(obj.get("cards"), list):
        for card in obj["cards"]:
            if not isinstance(card, dict):
                continue
            if "claim" not in card:
                parts = []
                if card.get("title"):
                    parts.append(str(card["title"]))
                if card.get("abstract"):
                    parts.append(str(card["abstract"]))
                if parts:
                    card["claim"] = " — ".join(parts)
            pmc = card.get("predicted_metric_changes")
            if isinstance(pmc, dict):
                # convert {axis: {direction, ...}} → [{axis, direction, ...}]
                lst = []
                for axis, body in pmc.items():
                    if not isinstance(body, dict):
                        continue
                    entry = dict(body)
                    entry["axis"] = axis
                    if entry.get("min_relative_deficit_reduction") is None:
                        entry["min_relative_deficit_reduction"] = (
                            0.10 if entry.get("min_absolute_delta") is None else 0.0
                        )
                    lst.append(entry)
                card["predicted_metric_changes"] = lst
            card.setdefault("preserve_constraints", [])
            card.setdefault("evidence_refs", [])
            card.setdefault("recommended_action_families", [])
    return obj


def _allowed_evidence_refs(
    evidence: EvidenceSummary,
    active_hypotheses: list[dict[str, Any]],
) -> set[str]:
    refs = {evidence.tick_id, "EvidenceSummary", "evidence", "summary"}
    refs.add(f"tick_{evidence.tick_id}")
    refs.update(EVIDENCE_REF_PREFIXES)
    refs.update(cross_campaign_memory_evidence_refs())
    refs.update(route_evidence_refs_for_validation(evidence))
    # Keep the validator synchronized with the exact top-level evidence payload
    # shown to the LLM. This includes scalar fields such as duplicate_fraction,
    # top_bin_share, su_per_gpu_h_recent, worker_gpu_h_window, and dedup_trust
    # after build_evidence_for_prompt's compaction/renaming.
    refs.update(str(k) for k in build_evidence_for_prompt(evidence).keys())
    for h in active_hypotheses or []:
        hid = h.get("hypothesis_id")
        if hid:
            refs.add(str(hid))
        refs.update(str(x) for x in (h.get("evidence_refs") or []) if str(x))
    for e in evidence.examples or []:
        refs.add(e.result_id)
        parent_id = getattr(e, "parent_result_id", None)
        if parent_id:
            refs.add(str(parent_id))
    for e in evidence.exemplars or []:
        refs.add(e.result_id)
        if e.parent_result_id:
            refs.add(e.parent_result_id)
    for r in evidence.recipes or []:
        refs.add(r.recipe_hash)
        refs.add(f"recipe_{r.recipe_hash}")
        refs.update(str(x) for x in (r.representative_result_ids or []) if str(x))
    refs.update(str(x) for x in (evidence.production_panel_selected_ids or []) if str(x))
    refs.update(str(x) for x in (evidence.production_near_miss_ids or []) if str(x))
    for item in evidence.stuck_lineage_roots or []:
        if isinstance(item, dict) and item.get("root_result_id"):
            refs.add(str(item["root_result_id"]))
    for item in evidence.recent_ticks_history or []:
        if isinstance(item, dict) and item.get("tick_id"):
            refs.add(str(item["tick_id"]))
    return refs


def _evidence_ref_allowed(ref: Any, allowed_refs: set[str]) -> bool:
    return evidence_ref_allowed(ref, allowed_refs)


def _normalize_mode_affinity(
    raw: Any, *, require_sum_to_one: bool
) -> tuple[dict[str, float] | None, str]:
    """Validate and normalize the Planner's E/R/X card affinity.

    This keeps validator, repair, and materialization aligned: malformed mode
    affinities no longer silently become equal-thirds fallback hints.
    """
    if not isinstance(raw, dict):
        return None, "mode_affinity_not_object"
    vals: dict[str, float] = {}
    for mode in PLANNER_MODES:
        if mode not in raw:
            return None, "mode_affinity_missing_modes"
        try:
            value = float(raw[mode])
        except (TypeError, ValueError):
            return None, f"mode_affinity_bad_value:{mode}"
        if not math.isfinite(value):
            return None, f"mode_affinity_nonfinite:{mode}"
        if value < 0.0:
            return None, f"mode_affinity_negative:{mode}"
        vals[mode] = value
    total = sum(vals.values())
    if total <= 0.0:
        return None, "mode_affinity_zero_sum"
    if require_sum_to_one and abs(total - 1.0) > 1e-3:
        return None, "mode_affinity_sum_not_one"
    return {mode: vals[mode] / total for mode in PLANNER_MODES}, ""


def _validate_schema(
    obj: dict,
    *,
    allowed_evidence_refs: set[str] | None = None,
) -> tuple[bool, str]:
    """Validate the Planner response without a runtime jsonschema dependency."""
    obj = _coerce_legacy_shape(obj)
    for k in ("abstain", "confidence", "cards", "rationale"):
        if k not in obj:
            return False, f"missing_top:{k}"
    if not isinstance(obj["abstain"], bool):
        return False, "abstain_not_bool"
    if isinstance(obj["confidence"], bool) or not isinstance(obj["confidence"], (int, float)):
        return False, "confidence_not_number"
    if not math.isfinite(float(obj["confidence"])) or not (0.0 <= float(obj["confidence"]) <= 1.0):
        return False, "confidence_out_of_range"
    if not isinstance(obj["cards"], list):
        return False, "cards_not_list"
    if not isinstance(obj["rationale"], str):
        return False, "rationale_not_string"
    if len(obj["cards"]) > MAX_PLANNER_CARDS:
        return False, f"too_many_cards:{len(obj['cards'])}"

    for i, card in enumerate(obj["cards"]):
        if not isinstance(card, dict):
            return False, f"card[{i}]:not_object"
        for k in (
            "claim",
            "mode_affinity",
            "evidence_refs",
            "predicted_metric_changes",
            "preserve_constraints",
            "recommended_action_families",
            "reasoning_trace",
        ):
            if k not in card:
                return False, f"card[{i}]:missing:{k}"
        if not isinstance(card["claim"], str) or len(card["claim"].strip()) < 8:
            return False, f"card[{i}]:claim_not_string"
        ma = card["mode_affinity"]
        _, ma_reason = _normalize_mode_affinity(ma, require_sum_to_one=True)
        if ma_reason:
            return False, f"card[{i}]:{ma_reason}"
        if not isinstance(card["evidence_refs"], list) or not card["evidence_refs"]:
            return False, f"card[{i}]:evidence_refs_empty"
        if any(not isinstance(ref, str) or not ref.strip() for ref in card["evidence_refs"]):
            return False, f"card[{i}]:bad_evidence_ref_type"
        if allowed_evidence_refs is not None:
            bad_refs = [
                str(ref) for ref in card["evidence_refs"]
                if not _evidence_ref_allowed(ref, allowed_evidence_refs)
            ]
            if bad_refs:
                return False, f"card[{i}]:unknown_evidence_refs:{bad_refs[:3]}"
        if not isinstance(card["predicted_metric_changes"], list) or not card["predicted_metric_changes"]:
            return False, f"card[{i}]:predicted_changes_empty"
        baseline_contract: tuple[str, ...] | None = None
        for j, pc in enumerate(card["predicted_metric_changes"]):
            if not isinstance(pc, dict):
                return False, f"card[{i}].predicted[{j}]:not_object"
            for kk in ("axis", "direction", "min_relative_deficit_reduction"):
                if kk not in pc:
                    return False, f"card[{i}].predicted[{j}]:missing:{kk}"
            # Predicted changes must name a qualification measurement.
            if pc["axis"] not in ("pLDDT", "iPAE", "binder_scRMSD"):
                return False, f"card[{i}].predicted[{j}]:bad_axis"
            if pc["direction"] not in ("increase", "decrease"):
                return False, f"card[{i}].predicted[{j}]:bad_direction"
            expected_direction = STRICT_SUCCESS[pc["axis"]][1]
            if pc["direction"] != expected_direction:
                return False, f"card[{i}].predicted[{j}]:direction_mismatch"
            baseline_refs = pc.get("baseline_refs", [])
            if not isinstance(baseline_refs, list) or any(
                not isinstance(ref, str) or not ref.strip() for ref in baseline_refs
            ):
                return False, f"card[{i}].predicted[{j}]:bad_baseline_refs"
            current_baseline = tuple(baseline_refs)
            if baseline_contract is None:
                baseline_contract = current_baseline
            elif current_baseline != baseline_contract:
                return False, f"card[{i}]:inconsistent_baseline_refs"
            if pc["min_relative_deficit_reduction"] is None:
                if pc.get("min_absolute_delta") is None:
                    return False, f"card[{i}].predicted[{j}]:missing_threshold"
                pc["min_relative_deficit_reduction"] = 0.0
            rdr = pc["min_relative_deficit_reduction"]
            if isinstance(rdr, bool) or not isinstance(rdr, (int, float)):
                return False, f"card[{i}].predicted[{j}]:bad_rdr_type"
            if not math.isfinite(float(rdr)) or not 0.0 <= float(rdr) <= 1.0:
                return False, f"card[{i}].predicted[{j}]:bad_rdr_range"
            abs_delta = pc.get("min_absolute_delta")
            if abs_delta is not None:
                if isinstance(abs_delta, bool) or not isinstance(abs_delta, (int, float)):
                    return False, f"card[{i}].predicted[{j}]:bad_abs_delta_type"
                if not math.isfinite(float(abs_delta)) or float(abs_delta) <= 0.0:
                    return False, f"card[{i}].predicted[{j}]:bad_abs_delta"
        rt = card["reasoning_trace"]
        if not isinstance(rt, dict):
            return False, f"card[{i}]:reasoning_trace_not_object"
        for kk in ("observed_signal", "inference", "action_implication"):
            if kk not in rt:
                return False, f"card[{i}].reasoning_trace:missing:{kk}"
            if not isinstance(rt[kk], str) or not rt[kk].strip():
                return False, f"card[{i}].reasoning_trace:empty:{kk}"
        if not isinstance(card["preserve_constraints"], list):
            return False, f"card[{i}]:preserve_constraints_not_list"
        for j, pc in enumerate(card["preserve_constraints"]):
            if not isinstance(pc, dict):
                return False, f"card[{i}].preserve[{j}]:not_object"
            for kk in ("axis", "max_relative_deficit_increase"):
                if kk not in pc:
                    return False, f"card[{i}].preserve[{j}]:missing:{kk}"
            if pc["axis"] not in ("pLDDT", "iPAE", "binder_scRMSD"):
                return False, f"card[{i}].preserve[{j}]:bad_axis"
            if isinstance(pc["max_relative_deficit_increase"], bool) or not isinstance(pc["max_relative_deficit_increase"], (int, float)):
                return False, f"card[{i}].preserve[{j}]:bad_max_increase_type"
            if not math.isfinite(float(pc["max_relative_deficit_increase"])) or float(pc["max_relative_deficit_increase"]) < 0.0:
                return False, f"card[{i}].preserve[{j}]:bad_max_increase"
        if not isinstance(card["recommended_action_families"], list) or not card[
            "recommended_action_families"
        ]:
            return False, f"card[{i}]:action_families_empty"
        for af in card["recommended_action_families"]:
            if not isinstance(af, str):
                return False, f"card[{i}]:bad_family_type"
            if af not in VALID_ACTION_FAMILIES:
                return False, f"card[{i}]:bad_family:{af}"
        if "ttl_ticks" in card:
            ttl = card["ttl_ticks"]
            if isinstance(ttl, bool) or not isinstance(ttl, int):
                return False, f"card[{i}]:bad_ttl_type"
            if not 1 <= ttl <= 20:
                return False, f"card[{i}]:bad_ttl_range"
    return True, ""


_AXIS_ALIASES = {
    "plddt": "pLDDT",
    "binder_plddt": "pLDDT",
    "pae": "iPAE",
    "ipae": "iPAE",
    "i_pae": "iPAE",
    "interface_pae": "iPAE",
    "binder_scrmsd": "binder_scRMSD",
    "scrmsd": "binder_scRMSD",
    "sc_rmsd": "binder_scRMSD",
    "binder_rmsd": "binder_scRMSD",
    "rmsd": "binder_scRMSD",
}


def _repair_planner_schema_drift(
    obj: dict,
    *,
    allowed_evidence_refs: set[str] | None,
) -> tuple[dict, list[str]]:
    """Conservative format repair for otherwise usable Planner JSON.

    This does NOT accept unsupported families or unsupported scientific axes.
    It only repairs common LLM shape drift that is already unambiguous from the
    prompt contract: missing E/R/E keys, common metric spelling variants,
    missing default thresholds, and evidence refs that need to fall back to the
    current EvidenceSummary when the card is otherwise grounded.
    """
    obj = _coerce_legacy_shape(obj)
    repairs: list[str] = []
    cards = obj.get("cards")
    if not isinstance(cards, list):
        return obj, repairs

    default_ref = "evidence"
    if allowed_evidence_refs:
        for preferred in ("EvidenceSummary", "evidence", "summary"):
            if preferred in allowed_evidence_refs:
                default_ref = preferred
                break

    if len(cards) > MAX_PLANNER_CARDS:
        repairs.append(f"cards_truncated_to_{MAX_PLANNER_CARDS}:from_{len(cards)}")
        cards = cards[:MAX_PLANNER_CARDS]

    repaired_cards: list[dict] = []
    for i, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        drop_card = False

        ma = card.get("mode_affinity")
        if isinstance(ma, dict):
            changed = False
            for m in PLANNER_MODES:
                if m not in ma:
                    ma[m] = 0.0
                    changed = True
            if changed:
                repairs.append(f"card[{i}]:filled_mode_affinity")
            norm_ma, ma_reason = _normalize_mode_affinity(ma, require_sum_to_one=False)
            if norm_ma is None:
                repairs.append(f"card[{i}]:dropped_bad_mode_affinity:{ma_reason}")
                drop_card = True
            else:
                if changed or any(abs(float(ma[m]) - norm_ma[m]) > 1e-3 for m in PLANNER_MODES):
                    repairs.append(f"card[{i}]:normalized_mode_affinity")
                card["mode_affinity"] = norm_ma
        else:
            repairs.append(f"card[{i}]:dropped_bad_mode_affinity:mode_affinity_not_object")
            drop_card = True

        refs = card.get("evidence_refs")
        if isinstance(refs, list):
            if allowed_evidence_refs is not None:
                good = [str(r) for r in refs if _evidence_ref_allowed(r, allowed_evidence_refs)]
            else:
                good = [str(r) for r in refs if isinstance(r, str) and r.strip()]
            if not good:
                if any(isinstance(r, str) and r.strip() for r in refs):
                    # Do not silently launder hallucinated references into a
                    # generic EvidenceSummary citation. Leave them for the
                    # validator so LLM reliability issues stay visible.
                    repairs.append(f"card[{i}]:kept_unknown_evidence_refs")
                    good = [str(r) for r in refs if isinstance(r, str) and r.strip()]
                else:
                    good = [default_ref]
                    repairs.append(f"card[{i}]:defaulted_evidence_ref")
            elif len(good) != len(refs):
                repairs.append(f"card[{i}]:filtered_unknown_evidence_refs")
            card["evidence_refs"] = good

        pmc = card.get("predicted_metric_changes")
        if isinstance(pmc, list):
            repaired_pmc = []
            for j, pc in enumerate(pmc):
                if not isinstance(pc, dict):
                    continue
                entry = dict(pc)
                axis = entry.get("axis")
                if isinstance(axis, str):
                    key = axis.strip().replace("-", "_").replace(" ", "_").lower()
                    mapped = _AXIS_ALIASES.get(key)
                    if mapped is not None and mapped != axis:
                        entry["axis"] = mapped
                        repairs.append(f"card[{i}].predicted[{j}]:axis_alias")
                if "min_relative_deficit_reduction" not in entry or entry.get("min_relative_deficit_reduction") is None:
                    if entry.get("min_absolute_delta") is None:
                        entry["min_relative_deficit_reduction"] = 0.10
                        repairs.append(f"card[{i}].predicted[{j}]:defaulted_threshold")
                    else:
                        entry["min_relative_deficit_reduction"] = 0.0
                        repairs.append(f"card[{i}].predicted[{j}]:absolute_threshold")
                abs_delta = entry.get("min_absolute_delta")
                if abs_delta is not None:
                    fixed_abs_delta: float | None = None
                    if not isinstance(abs_delta, bool):
                        try:
                            fixed_abs_delta = float(abs_delta)
                        except (TypeError, ValueError):
                            fixed_abs_delta = None
                    if fixed_abs_delta is None or not math.isfinite(fixed_abs_delta):
                        entry["min_absolute_delta"] = None
                        repairs.append(f"card[{i}].predicted[{j}]:dropped_bad_abs_delta")
                    elif fixed_abs_delta < 0.0:
                        entry["min_absolute_delta"] = abs(fixed_abs_delta)
                        repairs.append(f"card[{i}].predicted[{j}]:abs_delta_magnitude")
                    elif fixed_abs_delta == 0.0:
                        entry["min_absolute_delta"] = None
                        repairs.append(f"card[{i}].predicted[{j}]:dropped_zero_abs_delta")
                    elif fixed_abs_delta != abs_delta:
                        entry["min_absolute_delta"] = fixed_abs_delta
                        repairs.append(f"card[{i}].predicted[{j}]:coerced_abs_delta")
                    if entry.get("min_absolute_delta") is None:
                        rdr = entry.get("min_relative_deficit_reduction")
                        if (
                            isinstance(rdr, bool)
                            or not isinstance(rdr, (int, float))
                            or not math.isfinite(float(rdr))
                            or float(rdr) <= 0.0
                        ):
                            entry["min_relative_deficit_reduction"] = 0.10
                            repairs.append(f"card[{i}].predicted[{j}]:defaulted_threshold_after_abs_delta_drop")
                repaired_pmc.append(entry)
            card["predicted_metric_changes"] = repaired_pmc

        preserve = card.get("preserve_constraints")
        if isinstance(preserve, list):
            had_preserve_object = any(isinstance(pc, dict) for pc in preserve)
            repaired_preserve = []
            for j, pc in enumerate(preserve):
                if not isinstance(pc, dict):
                    repairs.append(f"card[{i}].preserve[{j}]:dropped_non_object")
                    continue
                entry = dict(pc)
                axis = entry.get("axis")
                if isinstance(axis, str):
                    key = axis.strip().replace("-", "_").replace(" ", "_").lower()
                    mapped = _AXIS_ALIASES.get(key)
                    if mapped is not None and mapped != axis:
                        entry["axis"] = mapped
                        repairs.append(f"card[{i}].preserve[{j}]:axis_alias")
                if entry.get("axis") in ("pLDDT", "iPAE", "binder_scRMSD"):
                    repaired_preserve.append(entry)
                else:
                    repairs.append(f"card[{i}].preserve[{j}]:dropped_bad_axis:{entry.get('axis')}")
            if had_preserve_object and not repaired_preserve:
                repairs.append(f"card[{i}]:dropped_all_invalid_preserve_constraints")
                drop_card = True
            card["preserve_constraints"] = repaired_preserve

        fams = card.get("recommended_action_families")
        if isinstance(fams, list):
            good_fams = [f for f in fams if f in VALID_ACTION_FAMILIES]
            if not good_fams and isinstance(card.get("config_delta_suggestions"), dict):
                good_fams = [
                    f for f in card["config_delta_suggestions"].keys()
                    if f in VALID_ACTION_FAMILIES
                ]
            if good_fams != fams:
                repairs.append(f"card[{i}]:filtered_bad_families")
                card["recommended_action_families"] = good_fams

        if drop_card:
            continue
        repaired_cards.append(card)

    obj["cards"] = repaired_cards
    return obj, repairs


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Coerce a value to float; return default when it is None or not numeric."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    return default


def _short_text(v: Any, limit: int = 280) -> str:
    if not isinstance(v, str):
        return ""
    text = " ".join(v.strip().split())
    return text[:limit]


def _extract_reasoning_trace(card: dict) -> ReasoningTrace:
    raw = card.get("reasoning_trace")
    if not isinstance(raw, dict):
        return ReasoningTrace()
    return ReasoningTrace(
        observed_signal=_short_text(raw.get("observed_signal")),
        inference=_short_text(raw.get("inference")),
        action_implication=_short_text(raw.get("action_implication")),
    )


def _extract_config_delta_suggestions(card: dict) -> dict[str, dict[str, object]]:
    """Read optional per-card config_delta_suggestions map.

    Defensively coerces malformed values to an empty dict so the rest of
    the materializer can proceed.
    """
    raw = card.get("config_delta_suggestions")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for family, params in raw.items():
        if isinstance(family, str) and isinstance(params, dict):
            out[family] = {k: v for k, v in params.items() if isinstance(k, str)}
    return out


def _materialize_cards(
    obj: dict, *, target_id: str, tick_id: int
) -> list[HypothesisCard]:
    out: list[HypothesisCard] = []
    for i, c in enumerate(obj.get("cards", [])):
        predicted = []
        for p in c.get("predicted_metric_changes", []):
            # Defensive: if BOTH thresholds are None, skip the change rather than crash.
            rdr = _safe_float(p.get("min_relative_deficit_reduction"), 0.0)
            abs_delta = p.get("min_absolute_delta")
            if abs_delta is not None:
                abs_delta = _safe_float(abs_delta, 0.0)
            if rdr <= 0.0 and abs_delta is None:
                # Both missing — assume a conservative 10% RDR rather than skip.
                rdr = 0.10
            predicted.append(
                PredictedChange(
                    axis=p.get("axis", "iPAE"),
                    direction=p.get("direction", "decrease"),
                    baseline_refs=p.get("baseline_refs", []) or [],
                    min_relative_deficit_reduction=rdr,
                    min_absolute_delta=abs_delta,
                )
            )
        preserve = [
            PreserveConstraint(
                axis=p["axis"],
                max_relative_deficit_increase=_safe_float(p.get("max_relative_deficit_increase"), 0.05),
            )
            for p in c.get("preserve_constraints", [])
            if isinstance(p, dict) and p.get("axis") in ("pLDDT", "iPAE", "binder_scRMSD")
        ]
        # Validator/repair should already provide a normalized vector; keep a
        # final defensive fallback only for direct test/helper callers.
        norm, ma_reason = _normalize_mode_affinity(
            c.get("mode_affinity") or {},
            require_sum_to_one=False,
        )
        if norm is None:
            norm = {"exploit": 1 / 3, "rescue": 1 / 3, "explore": 1 / 3}
        cds = _extract_config_delta_suggestions(c)
        out.append(
            HypothesisCard(
                hypothesis_id=f"hyp_{tick_id:04d}_{i:02d}",
                target_id=target_id,
                tick_created=tick_id,
                claim=c["claim"],
                mode_affinity=norm,
                evidence_refs=list(c["evidence_refs"]),
                predicted_metric_changes=predicted,
                preserve_constraints=preserve,
                recommended_action_families=list(c["recommended_action_families"]),
                # Coerce lifetime defensively and clamp to the supported range.
                ttl_ticks=max(1, min(20, int(_safe_float(c.get("ttl_ticks"), 10)))),
                config_delta_suggestions=cds,
                reasoning_trace=_extract_reasoning_trace(c),
            )
        )
    return out


def _filter_cards_to_available(
    cards: list[HypothesisCard], available_families: set[str],
) -> list[HypothesisCard]:
    """Filter card families when a nonempty availability set is supplied.

    Drop cards with no remaining family; an empty availability set leaves cards unchanged.
    """
    if not available_families:
        return cards
    kept: list[HypothesisCard] = []
    for c in cards:
        fams = [f for f in c.recommended_action_families if f in available_families]
        if not fams:
            continue  # no available family → drop card, recover the slot
        if fams != list(c.recommended_action_families):
            c = replace(c, recommended_action_families=fams)
        kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Top-level call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerCallConfig:
    model: str = "vllm/Qwen/Qwen3.6-27B-FP8"
    base_url: str = "http://127.0.0.1:8500/v1"
    # Bound output length while allowing complete structured responses.
    max_tokens: int = 3072
    enable_thinking: bool = False  # Disable model thinking output for the structured-response call.
    confidence_threshold: float = 0.55
    timeout_s: float = 90.0
    prompt_variant: str = "default"  # see trex.prompts.VARIANTS
    # Pin sampling instead of inheriting Qwen/vLLM generation_config defaults
    # (temperature=1.0/top-p sampling). Planner keeps a little diversity for
    # hypothesis generation; Supervisor is pinned lower in supervisor.py.
    temperature: float | None = 0.2


def call_planner(
    evidence: EvidenceSummary,
    active_hypotheses: list[dict[str, Any]] | None,
    seed_action_families: list[str] | None,
    *,
    tick_id_int: int,
    cfg: PlannerCallConfig | None = None,
    available_families: list[str] | None = None,
    recent_critic_flags: list[str] | None = None,
) -> PlannerOutput:
    cfg = cfg or PlannerCallConfig()
    active_hypotheses = active_hypotheses or []
    seed_action_families = seed_action_families or []
    allowed_refs = _allowed_evidence_refs(evidence, active_hypotheses)
    # Bound LLM latency so model failures return control to worker supervision and
    # fallback.
    client = create_client(cfg.model, base_url=cfg.base_url,
                           enable_thinking=cfg.enable_thinking,
                           timeout=cfg.timeout_s, max_retries=1)
    user_payload = build_user_prompt(
        evidence,
        active_hypotheses,
        seed_action_families,
        available_families=available_families,
        recent_critic_flags=recent_critic_flags,
    )

    # Resolve the system prompt for the configured variant.
    base_system_prompt = planner_system_prompt()
    if cfg.prompt_variant == "default":
        system_prompt = base_system_prompt
    else:
        from .prompts import get_variant  # lazy import to keep top of file light
        system_prompt = (
            base_system_prompt
            + "\n\nVARIANT APPENDIX (additive; core JSON/evidence contract above remains binding):\n"
            + get_variant(cfg.prompt_variant)
        )

    started = time.time()
    try:
        resp = client.chat(
            [{"role": "user", "content": user_payload}],
            system=system_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
    except Exception as exc:  # noqa: BLE001
        return PlannerOutput(
            valid=False,
            abstain=False,
            confidence=0.0,
            fail_reason=f"call_error:{type(exc).__name__}:{exc}",
            cards=[],
            rationale="",
            raw_text="",
            usage={},
        )

    raw = resp.text or ""
    obj = _extract_json(raw)
    if obj is None:
        return PlannerOutput(
            valid=False,
            abstain=False,
            confidence=0.0,
            fail_reason="parse_fail",
            cards=[],
            rationale="",
            raw_text=raw,
            usage=resp.usage or {},
        )

    ok, why = _validate_schema(obj, allowed_evidence_refs=allowed_refs)
    if not ok:
        repaired_obj, repairs = _repair_planner_schema_drift(
            obj,
            allowed_evidence_refs=allowed_refs,
        )
        repaired_ok, repaired_why = _validate_schema(
            repaired_obj,
            allowed_evidence_refs=allowed_refs,
        )
        if not repaired_ok:
            return PlannerOutput(
                valid=False,
                abstain=False,
                confidence=float(obj.get("confidence") or 0.0),
                fail_reason=f"schema_fail:{why}; repair_failed:{repaired_why}",
                cards=[],
                rationale=str(obj.get("rationale", "")),
                raw_text=raw,
                usage=resp.usage or {},
            )
        obj = repaired_obj
        repair_reason = "schema_repaired:" + why
        if repairs:
            repair_reason += "; " + ",".join(repairs[:8])
    else:
        repair_reason = None

    try:
        cards = _materialize_cards(obj, target_id=evidence.target_id, tick_id=tick_id_int)
    except Exception as exc:  # noqa: BLE001
    # Malformed card fields must activate fallback rather than abort the planning cycle.
        return PlannerOutput(
            valid=False,
            abstain=False,
            confidence=float(obj.get("confidence") or 0.0),
            fail_reason=f"materialize_fail:{type(exc).__name__}:{exc}",
            cards=[],
            rationale=str(obj.get("rationale", "")),
            raw_text=raw,
            usage=resp.usage or {},
        )
    confidence = float(obj.get("confidence") or 0.0)
    abstain = bool(obj.get("abstain"))

    # The static schema does not establish current backend availability.
    from .capability_registry import default_registry as _default_registry
    cards = _filter_cards_to_available(
        cards, set(available_families or _default_registry().feasible_families()))

    fail_reason = repair_reason
    if abstain:
        fail_reason = "abstain"
    elif confidence < cfg.confidence_threshold:
        fail_reason = f"low_confidence({confidence:.2f}<{cfg.confidence_threshold})"
    elif not cards:
        fail_reason = "empty_cards"

    return PlannerOutput(
        valid=True,
        abstain=abstain,
        confidence=confidence,
        fail_reason=fail_reason,
        cards=cards,
        rationale=str(obj.get("rationale", "")),
        raw_text=raw,
        usage=resp.usage or {"elapsed_s": time.time() - started},
    )
