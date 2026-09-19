"""Supervisor LLM via vLLM-served Qwen3.6-27B-FP8 (V5/V6.3 production model).

Inputs: EvidenceSummary + active HypothesisCards + validated ActionCandidates.
Output: mode_mixture + globally and within-mode ordered candidate_decisions.

See plan §12.1. Single call per tick. JSON-only output.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

from trex.llm import create_client

from .cross_campaign_memory import (
    cross_campaign_memory_evidence_refs,
    cross_campaign_memory_for_prompt,
    cross_campaign_memory_system_appendix,
)
from .evidence_refs import EVIDENCE_REF_PREFIXES, evidence_ref_allowed
from .fallback import CONFIDENT_MIXTURE_THRESHOLD
from .schemas import (
    ActionCandidate,
    CandidateDecision,
    EvidenceSummary,
    HypothesisCard,
    ReasoningTrace,
    SupervisorOutput,
)


SUPERVISOR_CONFIDENCE_GUIDANCE = f"{CONFIDENT_MIXTURE_THRESHOLD:.2f}-0.75"

SUPERVISOR_SYSTEM = """You are the Supervisor for T-ReX, a budgeted protein-binder design controller.

Given decomposed evidence, active hypotheses, and validated ActionCandidates,
choose a mode_mixture and rank launchable candidates.

Decision order:
1. Objective: maximize live Foldseek SU@TM0.6 per live WORKER-WALL GPU-hour
   from objective_summary. Use completed worker-GPU route values only for exact
   route/config learning.
2. Route value: prefer exact route_values rows over family averages. Rank current
   exploit by gpu_recent_new_su_per_route_gpu_h, then medium_recent_* as a
   delayed-feedback guard, then safe direct-scored record_recent_*. Lifetime value
   is memory/context, not current exploit rank.
3. Strict/near-miss evidence: raw strict is progress only when it creates new
   Foldseek SU@TM0.6. Duplicate-heavy routes can still be good if they continue
   buying new SU/GPU-h.
   A first official SU on a sparse/hard run is rare positive evidence: keep a
   bounded confirmation launchable, but do not infer mature all-worker value.
   Among exact routes with reliable, approximately comparable recent SU/GPU-h
   (the lower rate is at least 85% of the higher rate; treat sparse counts or
   exposure cautiously), put them in the same primary-efficiency tier and MUST
   rank higher strict_quality_p25, then strict_quality_median, first. Do not rank
   the lower-quality route first solely for a marginal rate advantage inside
   this tier. Quality cannot compensate outside the comparable-efficiency tier.
   Each candidate may include matched_exact_route_value. This is the exact
   family/operator/config (and parent-lineage, when applicable) join for that
   candidate. Use this joined row for candidate-specific efficiency and quality;
   do not transfer its values to another candidate.
4. Diagnostics: always inspect diagnostic_driver_tldr, diagnostic_axis_stats, and
   route_values[*].diagnostic_improvement_score/axes. Diagnostics do not mint SU;
   use them to choose rescue/explore levers. In productive states they are
   monitoring/tie-break evidence unless recent new SU/GPU-h decays; in stalled,
   deep_stall, rescue_rich, or duplicate-collapse states they are primary
   scientific evidence for rescue/explore ranking.
   Treat ipTM, ipsae, pTM, interface energetics, contacts, buried surface area,
   clashes, and verifier agreement as source-scoped Pareto objectives. Prefer a
   supported lever that improves the current blocker while preserving canonical
   pLDDT/iPAE/scRMSD. Missing or differently calibrated backend metrics are not
   negative evidence and must not be compared as if they shared one scale.
   Diagnostic axes may be source-scoped as axis[src=family]. Do not rank a
   candidate as if a BoltzGen-only diagnostic failure directly supports a
   BindCraft/Complexa knob unless the candidate explicitly frames it as an
   orthogonal cross-family probe rather than same-axis remediation.
5. Diversity/trust/realization: read diversity_summary as auxiliary collapse and
   panel evidence, dedup_trust as SU/diversity trust, and execution_realization as
   selected->started feedback. Selected-but-not-started work is pending execution,
   not scientific failure and not a reason to submit duplicate probes. A large
   proposed->started gap with selected approximately equal to started is a ranking
   outcome, not evidence that the family was blocked or under-tested.

Action hierarchy is a ranking prior, not a ban: L1 material search knobs include
FK temperature/sc_scale_noise/beam breadth; L2 rescue may be parent-bound
redesign/refilter on a concrete parent or same-family material perturbation
around a near-miss/stuck route. Scalar reward-only changes are later levers unless
backed by axis evidence or paired with material search.

Mode mixture:
- Emit finite nonnegative exploit/rescue/explore values that SUM TO 1.0.
- Keep mode_mixture coherent with candidate_decisions. If a rank-1 candidate is
  intended to launch within the next few worker starts, assign nontrivial mass
  to its mode; do not rely on the Selector to repair a contradictory scalar.
- mode_mixture_ranges are advisory target ranges. Minimal safety floors are hard;
  tighter ranges bind mainly under low confidence or collapse.
- If state is strict_duplicate_collapse or diversity_summary shows collapse, push
  explore upward. If a cheap route still buys new SU/GPU-h, keep exploit weight.
- If dry_since_last_SU >=6 worker-GPU-h, shift some budget to diversifying
  rescue/explore. >=12 without new SU should not be protected as near-miss
  progress. If sustained deep_stall recovers with only 1-2 SU on a low-yield run,
  keep the recovered route but rank at least one non-dominant root-family probe
  high enough to launch.

Immediate launch priority:
- Give every candidate_decision a unique global_rank in 1..N. global_rank=1 is
  the candidate you want on the NEXT available worker across ALL modes; it is
  authoritative whenever feasible. rank_in_mode remains the ordering within
  exploit/rescue/explore, while mode_mixture remains the longer-horizon target.
- Rank by expected NEW trusted Foldseek SU per FULL route worker-GPU-hour,
  including generation plus canonical score conversion. This is not a
  shortest-runtime contest: probability of strict success, structural novelty,
  strict quality, diagnostic evidence, and information value on stalled targets
  all matter.
- When a low-cost Complexa or ProteinMPNN route is currently buying comparable or
  better new SU/GPU-h with adequate quality, rank it ahead of a slower high-cost
  route. When BindCraft is the only route with replicated current SU or strong
  canonical-scored signal, it may correctly receive the top global ranks.
- BindCraft Accepted/Rejected/native metrics are diagnostic evidence only. Use
  chained canonical AF2 strict/SU outcomes for route value; do not mistake native
  acceptance for official success or tiny AF2 child cost for total route cost.

Candidate ranking:
- Rank only candidate_ids present in the input pool.
- Do not use target identity, target class, or unsupported family priors. If a cross_campaign_memory block is provided, treat it only as advisory historical experience conditioned on current evidence. No family is the expected winner for any target.
- Direct-scored families can be ranked by direct recent SU/GPU-h. Diagnostic-only
  families require chained credit or accepted-artifact/diagnostic evidence; do not
  let recent AF2 score-conversion children make the generator look artificially
  cheap.
- canonical_score_conversion refilters are system scoring lanes, not LLM rescue/
  explore budget. parent_model_refold is advisory only: it writes refold_* and
  does not mint SU.
- Do not describe de-novo candidates as parent-scaffold actions when
  parent_result_id is null. Parent-bound rescue requires a concrete parent family
  such as structure_refilter, proteinmpnn_redesign.
- resource_class is an audit/tie-break label, not a backend timeout promise.
  extended is allowed only when the candidate itself is extended-cost or the
  hypothesis is supported by at least one routed/refiltered descendant.

Output contract:
- candidate_decisions must include candidate_id, mode, rank_in_mode, global_rank,
  resource_class, what, why, evidence_refs, expected_signal, stop_or_downgrade_if,
  and reasoning_trace.
- reasoning_trace must include observed_signal, inference, action_implication.
  Keep each field short; this is an audit trace, not hidden chain-of-thought.
- Cite provided evidence/hypothesis IDs only.
- CRITICAL duplicate-candidate rule: each candidate_id must appear in AT MOST ONE mode across the entire candidate_decisions list. If a candidate fits two modes, choose the dominant role. Duplicate IDs
  across modes cause schema_fail.
- If allocation cannot be grounded, set abstain=true with fail_reason.
- Confidence: 0.45-0.55 for sparse/noisy evidence; use
  __SUPERVISOR_CONFIDENCE_GUIDANCE__ when direct evidence supports the mixture;
  >0.75 only when multiple direct signals agree.
- Output JSON only. No prose outside the JSON object.
""".replace("__SUPERVISOR_CONFIDENCE_GUIDANCE__", SUPERVISOR_CONFIDENCE_GUIDANCE)
def supervisor_system_prompt() -> str:
    """Return the system prompt for the current, explicitly applied run env."""

    return SUPERVISOR_SYSTEM + cross_campaign_memory_system_appendix("Supervisor")


def _matched_route_value_for_prompt(
    c: ActionCandidate,
    evidence: EvidenceSummary,
) -> dict[str, Any] | None:
    """Return the exact route row already used by deterministic route safeguards.

    Route values and candidates used to be separate prompt sections, leaving the
    LLM to reconstruct an error-prone family/operator/config/parent join. Reuse
    the Selector's strict matcher so the LLM sees the same candidate-specific
    evidence without adding a new ranking or allocation policy.
    """
    from .planner import compact_route_value_for_prompt
    from .schemas import to_jsonable
    from .selector import best_route_value_row

    row = best_route_value_row(c, evidence)
    if row is None:
        return None
    compact = compact_route_value_for_prompt(
        to_jsonable(row),
        target_dry_gpu_h=float(evidence.gpu_h_since_last_su or 0.0),
    )
    keep = {
        "route_id",
        "route_lineage",
        "status",
        "marginal_status",
        "attempts",
        "completions",
        "new_su",
        "record_recent_new_su",
        "new_su_recent_gpu",
        "record_recent_route_gpu_h",
        "gpu_recent_route_gpu_h",
        "medium_recent_new_su",
        "medium_recent_route_gpu_h",
        "value_rate_for_ranking",
        "value_rate_source",
        "lifetime_su_per_route_gpu_h",
        "strict_quality_n_unique_bins",
        "strict_quality_p25",
        "strict_quality_median",
        "strict_quality_axis_margins",
        "diagnostic_improvement_score",
        "diagnostic_improvement_axes",
        "diagnostic_improvement_n",
    }
    return {k: v for k, v in compact.items() if k in keep}


def _candidate_for_prompt(
    c: ActionCandidate,
    *,
    evidence: EvidenceSummary | None = None,
) -> dict[str, Any]:
    try:
        from .capability_registry import default_registry
        cap = default_registry().get(c.method_family)
    except Exception:  # noqa: BLE001
        cap = None

    outputs_diagnostic_only = bool(getattr(cap, "outputs_diagnostic_only", False))
    if c.method_family == "structure_refilter" and c.refilter_role == "canonical_score_conversion":
        score_credit_basis = "official_score_conversion"
    elif c.method_family == "structure_refilter" and c.refilter_role == "parent_model_refold":
        score_credit_basis = "advisory_refold_only"
    elif outputs_diagnostic_only:
        score_credit_basis = "requires_canonical_score_conversion"
    else:
        score_credit_basis = "direct_official_strict_metrics"

    prompt_row = {
        "candidate_id": c.candidate_id,
        "hypothesis_ids": list(c.hypothesis_ids),
        "method_family": c.method_family,
        "operator_id": c.operator_id,
        "config_delta": dict(c.config_delta or {}),
        "refilter_role": c.refilter_role,
        "downstream_route_plan": list(c.downstream_route_plan or []),
        "parent_result_id": c.parent_result_id,
        "baseline_result_id": c.baseline_result_id,
        "estimated_cost_class": c.estimated_cost_class,
        "expected_signal": c.expected_signal,
        "evidence_refs": list(c.evidence_refs),
        "feasibility_ok": c.feasibility.all_ok(),
        "feasibility_reasons": list(c.feasibility.reasons or []),
        "capability_role": getattr(cap, "role", None),
        "requires_parent_pdb": bool(getattr(cap, "requires_parent_pdb", False)),
        "outputs_diagnostic_only": outputs_diagnostic_only,
        "score_credit_basis": score_credit_basis,
    }
    if evidence is not None:
        matched = _matched_route_value_for_prompt(c, evidence)
        if matched is not None:
            prompt_row["matched_exact_route_value"] = matched
    return prompt_row


def _hypothesis_for_prompt(h: HypothesisCard) -> dict[str, Any]:
    return {
        "hypothesis_id": h.hypothesis_id,
        "claim": h.claim,
        "mode_affinity": h.mode_affinity,
        "status": h.status,
        "support_points": h.support_points,
        "contradiction_points": h.contradiction_points,
        "recommended_action_families": list(h.recommended_action_families),
        "predicted_axes": [pc.axis for pc in h.predicted_metric_changes],
        "reasoning_trace": {
            "observed_signal": h.reasoning_trace.observed_signal,
            "inference": h.reasoning_trace.inference,
            "action_implication": h.reasoning_trace.action_implication,
        },
    }


def build_user_prompt(
    evidence: EvidenceSummary,
    hypotheses: list[HypothesisCard],
    candidates: list[ActionCandidate],
    *,
    selector_context: dict[str, Any] | None = None,
) -> str:
    from .fallback import describe_clamp_for_prompt
    # Supervisor stub-leak fix (2026-05-30): use the SAME curated view as the
    # Planner (build_evidence_for_prompt) so the Supervisor — the LLM that emits
    # mode_mixture/resource_class — does NOT receive the route_health/llm_health
    # stubs (constant "route wide open / LLM perfectly healthy" dicts) that would
    # anchor it toward over-allocating capacity. This also keeps both LLMs'
    # evidence views identical, preventing future divergence.
    from .planner import build_evidence_for_prompt
    mixture_ranges = describe_clamp_for_prompt(evidence.state_label)
    payload = {
        "evidence": build_evidence_for_prompt(evidence),
        "hypotheses": [_hypothesis_for_prompt(h) for h in hypotheses],
        "candidates": [_candidate_for_prompt(c, evidence=evidence) for c in candidates],
        "selector_context": selector_context or {},
        "mode_mixture_ranges": (
            f"state={evidence.state_label}: recommended {mixture_ranges}. "
            "Emit a mode_mixture summing to 1.0 that reflects the evidence; "
            "these ranges are guidance, while downstream safety clamps enforce "
            "hard floors/ceilings. NOTE: under mode-collapse (top_bin_share >= 0.50) "
            "the explore ceiling is automatically RAISED beyond the figure above, "
            "so REQUEST MORE explore in that case to break the collapse — the clamp will permit it."
        ),
        "instructions": (
            "Return JSON with mode_mixture (summing to 1.0; use "
            "mode_mixture_ranges as evidence guidance, not a hard cap) and "
            "candidate_decisions. Rank enough worthwhile candidates to cover "
            "selector_context.available_slots_now when feasible. Give every "
            "decision a unique contiguous "
            "global_rank in 1..N; global_rank=1 is the immediate cross-mode "
            "next-worker choice. Rank within "
            "each mode in 1..N where N is the count of decisions you give that mode. "
            "Each candidate_decision must include a short reasoning_trace "
            "(observed_signal, inference, action_implication). Skip candidates "
            "not worth launching this tick. selector_context is read-only: use "
            "recent_realized_modes as observational launch-history evidence. "
            "recent_realized_modes and mode_credit are observational history, "
            "not an instruction to repay stale quota before global_rank=1. Emit "
            "the desired evidence-driven mode_mixture as the longer-horizon "
            "policy; global_rank controls fresh immediate launches. "
            "selector_context.cost_admission lists "
            "families the controller has yield-DEFERRED and high-cost dispatch "
            "pressure hints. Treat yield_deferred_families as weak scientific routes "
            "unless fresh evidence says otherwise. Use selected_not_started and "
            "dispatch_deferred as execution pressure, and avoid duplicate proposals "
            "while equivalent work is pending. Do not reinterpret proposals that the "
            "Selector did not choose as blocked execution or scientific under-testing. "
            "If method_health is not clearly negative, one bounded ranked probe may be "
            "reasonable; if another route is already producing high new SU/GPU-h and "
            "diversity is healthy, prefer that efficient route instead."
        ),
    }
    cross_memory = cross_campaign_memory_for_prompt()
    if cross_memory is not None:
        payload["cross_campaign_memory"] = cross_memory
        payload["instructions"] = payload["instructions"] + (
            " The optional `cross_campaign_memory` block is historical advisory "
            "experience only. Use it only when its match_features are observable "
            "in the current EvidenceSummary, active hypotheses, candidate pool, "
            "and selector context. It cannot override candidate feasibility, "
            "parent availability, evaluation requirements, deterministic safeguards, "
            "or current-target evidence."
        )
    # v7_3: prepend the TL;DR digest and mixture guidance so the evidence
    # summary and clamp context are the FIRST things read, instead of being
    # buried after the full evidence dump.
    from .planner import evidence_tldr
    header = (
        evidence_tldr(evidence)
        + "\nMIXTURE GUIDANCE: " + payload["mode_mixture_ranges"]
    )
    return header + "\n" + json.dumps(payload, sort_keys=True)


# ---------------------------------------------------------------------------
# Parsing + validation
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
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


VALID_MODES = {"exploit", "rescue", "explore"}
VALID_RESOURCES = {"low", "diagnostic", "standard", "extended"}
_RESOURCE_RANK = {"low": 0, "diagnostic": 1, "standard": 2, "extended": 3}


def _resource_rank(resource_class: object) -> int:
    return _RESOURCE_RANK.get(str(resource_class or "standard"), 2)


def _candidate_resource_class(cand: ActionCandidate | None) -> str:
    if cand is None:
        return "standard"
    cls = str(getattr(cand, "estimated_cost_class", "") or "standard")
    return cls if cls in VALID_RESOURCES else "standard"


def _extended_resource_allowed(
    decision: dict,
    *,
    candidate_by_id: dict[str, ActionCandidate] | None,
    hypothesis_by_id: dict[str, HypothesisCard] | None,
) -> bool:
    if decision.get("resource_class") != "extended":
        return True
    if candidate_by_id is None:
        return True
    cand = candidate_by_id.get(str(decision.get("candidate_id")))
    if cand is None:
        return False
    if str(getattr(cand, "estimated_cost_class", "") or "") == "extended":
        return True
    for hid in getattr(cand, "hypothesis_ids", None) or []:
        hyp = (hypothesis_by_id or {}).get(str(hid))
        if hyp is None:
            continue
        if getattr(hyp, "status", "") == "supported":
            return True
        if float(getattr(hyp, "support_points", 0.0) or 0.0) >= 1.0:
            return True
    # Extended-cost escalation must be evidence-enforced. Do not infer it from
    # LLM/candidate evidence_ref strings such as "repeated_near_pass"; those are
    # citations, not lifecycle support. Repeated near-pass can still become
    # eligible once lifecycle converts descendants into support_points.
    return False


def _allowed_evidence_refs(
    evidence: EvidenceSummary,
    hypotheses: list[HypothesisCard],
    candidates: list[ActionCandidate],
) -> set[str]:
    from .planner import build_evidence_for_prompt, route_evidence_refs_for_validation
    refs = {evidence.tick_id, "EvidenceSummary", "evidence", "summary"}
    refs.add(f"tick_{evidence.tick_id}")
    # Candidate rows may contain this Supervisor-only exact route join. The
    # prompt explicitly tells the model to use it, so citing either the block
    # or a dotted field within it must not trigger schema fallback.
    refs.add("matched_exact_route_value")
    refs.update(EVIDENCE_REF_PREFIXES)
    refs.update(cross_campaign_memory_evidence_refs())
    refs.update(route_evidence_refs_for_validation(evidence))
    # Keep this validator synchronized with the exact curated evidence payload
    # sent in build_user_prompt (including compacted/renamed scalars such as
    # worker_gpu_h_window and dedup_trust).
    refs.update(str(k) for k in build_evidence_for_prompt(evidence).keys())
    for h in hypotheses:
        refs.add(h.hypothesis_id)
        refs.update(str(x) for x in (h.evidence_refs or []) if str(x))
    for c in candidates:
        refs.add(c.candidate_id)
        refs.add(f"candidate:{c.candidate_id}")
        refs.add(f"candidate::{c.candidate_id}")
        refs.update(str(x) for x in (c.evidence_refs or []) if str(x))
        refs.update(str(x) for x in (c.hypothesis_ids or []) if str(x))
        if c.parent_result_id:
            refs.add(c.parent_result_id)
        if c.baseline_result_id:
            refs.add(c.baseline_result_id)
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


def _validate_schema(
    obj: dict,
    known_candidate_ids: set[str],
    *,
    allowed_evidence_refs: set[str] | None = None,
    candidate_by_id: dict[str, ActionCandidate] | None = None,
    hypothesis_by_id: dict[str, HypothesisCard] | None = None,
) -> tuple[bool, str]:
    # `rationale` is auxiliary; default to "" if missing.
    # `abstain` defaults to False, `confidence` to 0.5, both nudged.
    obj.setdefault("rationale", "")
    obj.setdefault("abstain", False)
    obj.setdefault("confidence", 0.5)
    obj.setdefault("candidate_decisions", obj.get("decisions", []))
    for k in ("abstain", "confidence", "mode_mixture", "candidate_decisions"):
        if k not in obj:
            return False, f"missing_top:{k}"
    if not isinstance(obj["abstain"], bool):
        return False, "abstain_not_bool"
    if (
        not isinstance(obj["confidence"], (int, float))
        or not math.isfinite(float(obj["confidence"]))
        or not (0.0 <= float(obj["confidence"]) <= 1.0)
    ):
        return False, "confidence_invalid"

    mm = obj["mode_mixture"]
    if not isinstance(mm, dict) or not all(k in mm for k in VALID_MODES):
        return False, "mode_mixture_modes_missing"
    if any(
        not isinstance(mm[k], (int, float))
        or not math.isfinite(float(mm[k]))
        or float(mm[k]) < 0
        for k in VALID_MODES
    ):
        return False, "mode_mixture_negative_or_non_numeric"
    mixture_sum = sum(float(mm[k]) for k in VALID_MODES)
    if mixture_sum <= 0:
        return False, "mode_mixture_zero_sum"
    if abs(mixture_sum - 1.0) > 1e-3:
        return False, f"mode_mixture_not_normalized:{mixture_sum:.4f}"

    decs = obj["candidate_decisions"]
    if not isinstance(decs, list):
        return False, "candidate_decisions_not_list"
    for i, d in enumerate(decs):
        for k in (
            "candidate_id",
            "mode",
            "rank_in_mode",
            "global_rank",
            "resource_class",
            "what",
            "why",
            "evidence_refs",
            "expected_signal",
            "stop_or_downgrade_if",
        ):
            if k not in d:
                return False, f"dec[{i}]:missing:{k}"
        if not isinstance(d["evidence_refs"], list) or not d["evidence_refs"]:
            return False, f"dec[{i}]:evidence_refs_empty"
        _repair_reasoning_trace(d)
        rt = d.get("reasoning_trace")
        if not isinstance(rt, dict):
            return False, f"dec[{i}]:reasoning_trace_not_object"
        for kk in _REASONING_TRACE_KEYS:
            if kk not in rt:
                return False, f"dec[{i}].reasoning_trace:missing:{kk}"
            if not isinstance(rt[kk], str) or not rt[kk].strip():
                return False, f"dec[{i}].reasoning_trace:empty:{kk}"
        if d["mode"] not in VALID_MODES:
            return False, f"dec[{i}]:bad_mode"
        if d["resource_class"] not in VALID_RESOURCES:
            return False, f"dec[{i}]:bad_resource_class"
        cand = (candidate_by_id or {}).get(str(d.get("candidate_id")))
        if (
            cand is not None
            and d["resource_class"] == "extended"
            and not _extended_resource_allowed(
                d, candidate_by_id=candidate_by_id, hypothesis_by_id=hypothesis_by_id,
            )
        ):
            # Overstating a non-extended candidate as `extended` is not a safety
            # issue; it only degrades ordering and used to trigger fallback.
            # Downgrade to the registry-estimated concrete cost while preserving
            # the evidence-enforced guard for true extended-cost candidates.
            d["resource_class"] = _candidate_resource_class(cand)
        if cand is not None and _resource_rank(d["resource_class"]) < _resource_rank(getattr(cand, "estimated_cost_class", None)):
            return False, f"dec[{i}]:resource_class_understates_candidate_cost"
        if not _extended_resource_allowed(
            d, candidate_by_id=candidate_by_id, hypothesis_by_id=hypothesis_by_id,
        ):
            return False, f"dec[{i}]:extended_without_supported_or_near_pass"
        if not isinstance(d["rank_in_mode"], int) or d["rank_in_mode"] < 1:
            return False, f"dec[{i}]:bad_rank"
        if not isinstance(d["global_rank"], int) or d["global_rank"] < 1:
            return False, f"dec[{i}]:bad_global_rank"
        if d["candidate_id"] not in known_candidate_ids:
            return False, f"dec[{i}]:unknown_candidate_id:{d['candidate_id']}"
        if allowed_evidence_refs is not None:
            bad_refs = [
                str(ref) for ref in d["evidence_refs"]
                if not _evidence_ref_allowed(ref, allowed_evidence_refs)
            ]
            if bad_refs:
                return False, f"dec[{i}]:unknown_evidence_refs:{bad_refs[:3]}"

    # Per-mode unique ranks check
    by_mode: dict[str, list[int]] = {}
    for d in decs:
        by_mode.setdefault(d["mode"], []).append(d["rank_in_mode"])
    for m, ranks in by_mode.items():
        if len(set(ranks)) != len(ranks):
            return False, f"dup_rank_in_mode:{m}"

    # Each candidate may appear in AT MOST ONE mode. The LLM sometimes
    # double-counts a candidate across exploit+explore which causes
    # silent rank ambiguity downstream (selector dict-collapse keeps
    # only the last decision). Catch that explicitly here.
    seen_candidates: dict[str, str] = {}
    for d in decs:
        cid = d["candidate_id"]
        if cid in seen_candidates:
            return False, (
                f"duplicate_candidate_across_modes:{cid}"
                f"({seen_candidates[cid]} and {d['mode']})"
            )
        seen_candidates[cid] = d["mode"]

    global_ranks = [d["global_rank"] for d in decs]
    if len(set(global_ranks)) != len(global_ranks):
        return False, "dup_global_rank"
    if sorted(global_ranks) != list(range(1, len(global_ranks) + 1)):
        return False, "noncontiguous_global_rank"
    return True, ""


def _short_text(v: Any, limit: int = 280) -> str:
    if not isinstance(v, str):
        return ""
    text = " ".join(v.strip().split())
    return text[:limit]


_REASONING_TRACE_KEYS = ("observed_signal", "inference", "action_implication")


def _repair_reasoning_trace(decision: dict) -> None:
    """Preserve valid Supervisor audit traces; repair missing/blank traces.

    This is not chain-of-thought reconstruction. It deterministically derives a
    compact audit trail from already-validated decision fields so every accepted
    Supervisor choice has evidence, inference, and action implication text.
    """
    raw = decision.get("reasoning_trace")
    trace = raw if isinstance(raw, dict) else {}
    refs = [str(r) for r in (decision.get("evidence_refs") or []) if str(r)]
    observed = _short_text(trace.get("observed_signal"))
    inference = _short_text(trace.get("inference"))
    implication = _short_text(trace.get("action_implication"))
    if not observed:
        observed = _short_text("cites " + ", ".join(refs[:4])) if refs else ""
    if not inference:
        inference = _short_text(decision.get("why"))
    if not implication:
        what = _short_text(decision.get("what"), limit=160)
        expected = _short_text(decision.get("expected_signal"), limit=120)
        implication = _short_text(
            f"{what}; expected signal: {expected}" if expected else what
        )
    decision["reasoning_trace"] = {
        "observed_signal": observed,
        "inference": inference,
        "action_implication": implication,
    }


def _extract_reasoning_trace(decision: dict) -> ReasoningTrace:
    raw = decision.get("reasoning_trace")
    if not isinstance(raw, dict):
        return ReasoningTrace()
    return ReasoningTrace(
        observed_signal=_short_text(raw.get("observed_signal")),
        inference=_short_text(raw.get("inference")),
        action_implication=_short_text(raw.get("action_implication")),
    )


def _materialize_decisions(obj: dict) -> list[CandidateDecision]:
    out: list[CandidateDecision] = []
    for d in obj["candidate_decisions"]:
        out.append(
            CandidateDecision(
                candidate_id=d["candidate_id"],
                mode=d["mode"],
                rank_in_mode=int(d["rank_in_mode"]),
                global_rank=int(d["global_rank"]),
                resource_class=d["resource_class"],
                what=d["what"],
                why=d["why"],
                evidence_refs=list(d["evidence_refs"]),
                expected_signal=d["expected_signal"],
                stop_or_downgrade_if=d["stop_or_downgrade_if"],
                reasoning_trace=_extract_reasoning_trace(d),
            )
        )
    return out


def _renumber_candidate_decisions(decs: list[dict]) -> None:
    mode_counts = {mode: 0 for mode in VALID_MODES}
    for global_rank, decision in enumerate(decs, start=1):
        mode = decision.get("mode")
        if mode in mode_counts:
            mode_counts[mode] += 1
            decision["rank_in_mode"] = mode_counts[mode]
        decision["global_rank"] = global_rank


def _repair_supervisor_schema_drift(
    obj: dict,
    *,
    known_candidate_ids: set[str],
) -> tuple[dict, list[str]]:
    """Repair avoidable ranking drift without accepting unsafe decisions.

    The Supervisor sometimes includes a stale or hallucinated candidate_id while
    the rest of the ranking is valid. Dropping the whole tick makes the selector
    fall back exactly when evidence-rich dry/stall states need an LLM pivot. This
    repair only removes decisions for unknown candidates and duplicate IDs, then
    renumbers ranks and lets the normal validator re-check all remaining fields.
    """
    repaired = dict(obj)
    raw_decs = repaired.get("candidate_decisions", repaired.get("decisions", []))
    repairs: list[str] = []
    if not isinstance(raw_decs, list):
        return repaired, repairs

    kept: list[dict] = []
    seen: set[str] = set()
    for i, raw_decision in enumerate(raw_decs):
        if not isinstance(raw_decision, dict):
            repairs.append(f"dec[{i}]:dropped_non_object")
            continue
        decision = dict(raw_decision)
        cid = str(decision.get("candidate_id", ""))
        if cid not in known_candidate_ids:
            repairs.append(f"dec[{i}]:dropped_unknown_candidate_id:{cid}")
            continue
        if cid in seen:
            repairs.append(f"dec[{i}]:dropped_duplicate_candidate_id:{cid}")
            continue
        seen.add(cid)
        kept.append(decision)

    if len(kept) != len(raw_decs):
        _renumber_candidate_decisions(kept)
        repaired["candidate_decisions"] = kept
    return repaired, repairs


# ---------------------------------------------------------------------------
# Top-level call
# ---------------------------------------------------------------------------


def build_ranking_repair_prompt(
    validation_error: str,
    candidate_ids: set[str],
) -> str:
    """Build the bounded one-shot schema-repair message."""

    ordered_ids = sorted(candidate_ids)
    candidate_hint = ", ".join(ordered_ids[:40])
    if len(ordered_ids) > 40:
        candidate_hint += ", ..."
    return (
        "Your previous output rejected: " + validation_error + ". "
        "Re-emit the JSON. Ensure each candidate_id appears in exactly ONE "
        "mode and give every candidate_decision one unique contiguous "
        "global_rank in 1..N, where 1 is the immediate next-worker choice. "
        "Use only these candidate_id values: " + candidate_hint + "."
    )


@dataclass(frozen=True)
class SupervisorCallConfig:
    # See PlannerCallConfig: 27B is pinned for T-ReX MVP (V5/V6.3 history +
    # smoke 8667509 8/8 PASS). 35B-A3B passes 8/8 (job 8692376) at ~2.7x
    # latency win; optional sensitivity, not default.
    model: str = "vllm/Qwen/Qwen3.6-27B-FP8"
    base_url: str = "http://127.0.0.1:8500/v1"
    # Live traces hit tokens_out=2048 on schema failures; give the JSON emitter
    # enough room while keeping the call bounded.
    max_tokens: int = 3072
    enable_thinking: bool = False
    confidence_threshold: float = 0.55
    timeout_s: float = 90.0
    # Pin Supervisor near-deterministic so E/R/E distributions are stable and
    # do not inherit Qwen/vLLM generation_config sampling defaults.
    temperature: float | None = 0.0


def call_supervisor(
    evidence: EvidenceSummary,
    hypotheses: list[HypothesisCard],
    candidates: list[ActionCandidate],
    *,
    cfg: SupervisorCallConfig | None = None,
    selector_context: dict[str, Any] | None = None,
) -> SupervisorOutput:
    cfg = cfg or SupervisorCallConfig()
    known_ids = {c.candidate_id for c in candidates}
    candidate_by_id = {c.candidate_id: c for c in candidates}
    hypothesis_by_id = {h.hypothesis_id: h for h in hypotheses}
    allowed_refs = _allowed_evidence_refs(evidence, hypotheses, candidates)
    user_payload = build_user_prompt(
        evidence, hypotheses, candidates, selector_context=selector_context,
    )
    # Bug A fix (2026-05-30): thread the per-call timeout (see planner.py) so a
    # hung vLLM request cannot block the synchronous controller reap loop; on
    # timeout the except below falls back to the deterministic mode mixture.
    client = create_client(
        cfg.model, base_url=cfg.base_url, enable_thinking=cfg.enable_thinking,
        timeout=cfg.timeout_s, max_retries=1,
    )

    system_prompt = supervisor_system_prompt()
    started = time.time()
    try:
        resp = client.chat(
            [{"role": "user", "content": user_payload}],
            system=system_prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
    except Exception as exc:  # noqa: BLE001
        return SupervisorOutput(
            valid=False,
            abstain=False,
            confidence=0.0,
            fail_reason=f"call_error:{type(exc).__name__}:{exc}",
            mode_mixture={},
            candidate_decisions=[],
            rationale="",
            raw_text="",
            usage={},
        )

    raw = resp.text or ""
    obj = _extract_json(raw)
    if obj is None:
        return SupervisorOutput(
            valid=False,
            abstain=False,
            confidence=0.0,
            fail_reason="parse_fail",
            mode_mixture={},
            candidate_decisions=[],
            rationale="",
            raw_text=raw,
            usage=resp.usage or {},
        )

    ok, why = _validate_schema(
        obj, known_ids, allowed_evidence_refs=allowed_refs,
        candidate_by_id=candidate_by_id, hypothesis_by_id=hypothesis_by_id,
    )
    # Scoped one-shot retry for structural ranking errors. These are cheap to
    # repair and otherwise turn an evidence-grounded scientific decision into a
    # deterministic fallback for an avoidable formatting omission.
    # This specific schema_fail recurs on stalled-state Supervisor calls
    # (3 separate smokes: 8721974, 8723110, 8724321). The model appears to
    # "hedge" by listing the same candidate in two modes. A targeted retry
    # with an explicit corrective hint is cheap and scoped.
    ranking_schema_error = (
        "duplicate_candidate_across_modes" in why
        or "global_rank" in why
        or "unknown_candidate_id" in why
    )
    if not ok and ranking_schema_error:
        repair_hint = build_ranking_repair_prompt(why, known_ids)
        try:
            resp2 = client.chat(
                [
                    {"role": "user", "content": user_payload},
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": repair_hint},
                ],
                system=system_prompt,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            raw2 = resp2.text or ""
            obj2 = _extract_json(raw2)
            if obj2 is not None:
                ok2, why2 = _validate_schema(
                    obj2, known_ids, allowed_evidence_refs=allowed_refs,
                    candidate_by_id=candidate_by_id, hypothesis_by_id=hypothesis_by_id,
                )
                if ok2:
                    obj, raw, ok, why = obj2, raw2, ok2, why2
                    # merge usage so the caller still sees a fair tokens count
                    try:
                        for k, v in (resp2.usage or {}).items():
                            resp.usage[k] = resp.usage.get(k, 0) + v
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001 — retry is best-effort
            pass

    schema_repair_reason = None
    if not ok and "unknown_candidate_id" in why:
        repaired_obj, repairs = _repair_supervisor_schema_drift(
            obj,
            known_candidate_ids=known_ids,
        )
        if repairs:
            repaired_ok, repaired_why = _validate_schema(
                repaired_obj, known_ids, allowed_evidence_refs=allowed_refs,
                candidate_by_id=candidate_by_id, hypothesis_by_id=hypothesis_by_id,
            )
            if repaired_ok and repaired_obj.get("candidate_decisions"):
                obj, ok, why = repaired_obj, True, repaired_why
                schema_repair_reason = "schema_repaired:" + ",".join(repairs[:8])

    if not ok:
        return SupervisorOutput(
            valid=False,
            abstain=False,
            confidence=float(obj.get("confidence") or 0.0),
            fail_reason=f"schema_fail:{why}",
            mode_mixture={},
            candidate_decisions=[],
            rationale=str(obj.get("rationale", "")),
            raw_text=raw,
            usage=resp.usage or {},
        )

    decs = _materialize_decisions(obj)
    confidence = float(obj.get("confidence") or 0.0)
    abstain = bool(obj.get("abstain"))

    fail_reason = schema_repair_reason
    if abstain:
        fail_reason = "abstain"
    elif confidence < cfg.confidence_threshold:
        low_conf = f"low_confidence({confidence:.2f}<{cfg.confidence_threshold})"
        fail_reason = f"{fail_reason}; {low_conf}" if fail_reason else low_conf

    return SupervisorOutput(
        valid=True,
        abstain=abstain,
        confidence=confidence,
        fail_reason=fail_reason,
        mode_mixture={k: float(obj["mode_mixture"][k]) for k in VALID_MODES},
        candidate_decisions=decs,
        rationale=str(obj.get("rationale", "")),
        raw_text=raw,
        usage=resp.usage or {"elapsed_s": time.time() - started},
    )
