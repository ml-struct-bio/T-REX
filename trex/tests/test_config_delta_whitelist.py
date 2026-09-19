"""Gap A: capability whitelist for LLM-proposed config_delta values."""

from __future__ import annotations

import pytest

from trex.candidate_builder import build_candidates
from trex.capability_registry import (
    Capability,
    CapabilityRegistry,
    default_registry,
    validate_config_delta,
    validate_config_delta_partial,
)
from trex.schemas import (
    AxisStat,
    EvidenceSummary,
    HypothesisCard,
    LLMHealthSummary,
    PredictedChange,
    PreserveConstraint,
    RouteHealthSummary,
)


def _evidence(state: str = "productive", *, axis_stats=None, diagnostic_axis_stats=None) -> EvidenceSummary:
    return EvidenceSummary(
        tick_id="t1", target_id="t", target_class="c", schema_version="v",
        elapsed_wall_h=1, remaining_wall_h=10, completed_children=0, pending_children=0,
        worker_gpu_h_total=1, worker_gpu_h_last_3_ticks=1,
        strict_count=0, global_new_strict=0, run_su_count=0, run_su_count_delta=0,
        su_per_gpu_h_recent=None, duplicate_fraction=None, top_bin_share=None,
        axis_stats=axis_stats or {}, diagnostic_axis_stats=diagnostic_axis_stats or {}, joint_patterns=[], near_miss_count=0,
        panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 96, None, None, None),
        llm_health=LLMHealthSummary("m", [], 0.0, 0.0, 0, 0.0),
        state_label=state,  # type: ignore[arg-type]
        examples=[], metric_availability={},
    )


def _hyp(
    suggestions: dict | None = None,
    *,
    family: str = "complexa_beam",
) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id="h1", target_id="t", tick_created=1,
        claim="x", mode_affinity={"exploit": 0.5, "rescue": 0.5, "explore": 0.0},
        evidence_refs=["e1"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["b"], 0.20, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", 0.05)],
        recommended_action_families=[family],
        config_delta_suggestions=suggestions or {},
    )


# ---- validate_config_delta direct -----------------------------------------


def test_validate_accepts_in_range_numeric():
    reg = default_registry()
    cap = reg.get("complexa_beam")
    ok, reasons = validate_config_delta(cap, {"beam_width": 8, "n_branch": 4})
    assert ok, reasons


def test_validate_rejects_out_of_range():
    reg = default_registry()
    cap = reg.get("complexa_beam")
    ok, reasons = validate_config_delta(cap, {"beam_width": 100})
    assert not ok
    assert any("out_of_range" in r for r in reasons)


def test_validate_rejects_unknown_param():
    reg = default_registry()
    cap = reg.get("complexa_beam")
    ok, reasons = validate_config_delta(cap, {"not_a_param": 0.5})
    assert not ok
    assert any("unknown_param" in r for r in reasons)


def test_validate_empty_passes():
    reg = default_registry()
    cap = reg.get("complexa_beam")
    ok, reasons = validate_config_delta(cap, {})
    assert ok and not reasons


# ---- CandidateBuilder integration -----------------------------------------


def test_builder_accepts_valid_suggestion():
    h = _hyp({"complexa_beam": {"beam_width": 8, "n_branch": 4}})
    cands = build_candidates([h], _evidence())
    assert cands
    assert cands[0].config_delta == {"beam_width": 8, "n_branch": 4}


def test_builder_clamps_numeric_overshoot_instead_of_dropping_family_choice():
    h = _hyp({"complexa_beam": {"beam_width": 999}})  # out of range
    cands = build_candidates([h], _evidence())
    assert cands
    # Numeric overshoots should be clamped and budget-repaired instead of
    # silently converting the intended family/config to a default no-op.
    assert cands[0].config_delta
    assert cands[0].config_delta["beam_width"] <= 16
    assert cands[0].feasibility.all_ok()



def test_partial_validator_clamps_numeric_to_nearest_bound():
    cap = default_registry().get("complexa_best_of_n")
    kept, dropped = validate_config_delta_partial(cap, {"replicas": 999})
    assert kept["replicas"] == 8
    assert any(r.startswith("clamped:replicas=999->8") for r in dropped)


def test_builder_chunks_large_replica_best_of_n_without_reducing_search_breadth():
    h = _hyp(
        {"complexa_best_of_n": {"replicas": 8, "nsamples": 4, "nsteps": 400}},
        family="complexa_best_of_n",
    )
    cands = build_candidates([h], _evidence())

    assert len(cands) == 1
    assert cands[0].config_delta["replicas"] == 8
    assert cands[0].config_delta["batch_size"] == 8
    assert any(
        "memory_safe_best_of_n_batch_size=16->8" in reason
        for reason in cands[0].feasibility.reasons
    )


def test_builder_preserves_explicit_smaller_best_of_n_batch():
    h = _hyp(
        {"complexa_best_of_n": {"replicas": 8, "batch_size": 4}},
        family="complexa_best_of_n",
    )
    cands = build_candidates([h], _evidence())

    assert len(cands) == 1
    assert cands[0].config_delta["batch_size"] == 4


def test_builder_drops_unknown_family_suggestions():
    h = _hyp({"some_other_family": {"x": 1}})
    cands = build_candidates([h], _evidence())
    assert cands
    # the unknown family's key must not leak onto complexa_beam (the original
    # contract); rec 3 may add an axis-matched soft-default since the LLM gave
    # complexa_beam no config of its own.
    assert "x" not in cands[0].config_delta


def test_builder_no_suggestions_still_works():
    # rec 3 (2026-05-30): with NO LLM config on an iPAE-diagnosed complexa_beam
    # hypothesis, the builder seeds the axis-matched soft-default (interface
    # blocker -> raise backbone noise). Still produces exactly one candidate.
    h = _hyp(None)
    cands = build_candidates([h], _evidence())
    assert cands
    assert cands[0].config_delta == {"sc_scale_noise": 0.2975}


def _axis_stat(n: int, *, fail: int | None = None) -> AxisStat:
    fail = n if fail is None else fail
    return AxisStat(
        pass_count=max(0, n - fail),
        near_pass_count=0,
        fail_count=fail,
        median_raw=None,
        median_calibrated=None,
        median_deficit=1.0 if fail else 0.0,
        calibration_status="provisional",
        n=n,
    )


def test_v73_sparse_complexa_reward_is_deferred_to_material_search():
    h = _hyp({"complexa_beam": {"reward_i_pae_weight": -1.5}})
    cands = build_candidates([h], _evidence())
    assert cands
    cfg = cands[0].config_delta
    assert "reward_i_pae_weight" not in cfg
    assert cfg.get("beam_width") == 8
    assert cfg.get("n_branch") == 4
    assert any("reward_deferred:evidence_sparse" in r for r in cands[0].feasibility.reasons)


def test_v73_evidence_rich_complexa_reward_keeps_reward_and_adds_material_search():
    h = _hyp({"complexa_beam": {"reward_i_pae_weight": -1.5}})
    e = _evidence(axis_stats={"iPAE": _axis_stat(32)})
    cands = build_candidates([h], e)
    assert cands
    cfg = cands[0].config_delta
    assert cfg.get("reward_i_pae_weight") == -1.5
    assert cfg.get("beam_width") == 8
    assert cfg.get("n_branch") == 4
    assert any("axis_matched_material_search_for_reward_retry" in r for r in cands[0].feasibility.reasons)


def test_v73_complexa_reward_with_existing_material_knob_is_not_overpaired():
    h = _hyp({"complexa_beam": {"reward_i_pae_weight": -1.5, "beam_width": 8}})
    e = _evidence(axis_stats={"iPAE": _axis_stat(32)})
    cands = build_candidates([h], e)
    assert cands
    cfg = cands[0].config_delta
    assert cfg.get("reward_i_pae_weight") == -1.5
    assert cfg.get("beam_width") == 8
    assert "n_branch" not in cfg
    assert not any("axis_matched_material_search_for_reward_retry" in r for r in cands[0].feasibility.reasons)


def test_v73_sparse_complexa_reward_with_material_knob_defers_reward():
    h = _hyp({"complexa_beam": {"reward_i_pae_weight": -1.5, "beam_width": 8}})
    cands = build_candidates([h], _evidence())
    assert cands
    cfg = cands[0].config_delta
    assert "reward_i_pae_weight" not in cfg
    assert cfg.get("beam_width") == 8
    assert any("reward_deferred:evidence_sparse" in r for r in cands[0].feasibility.reasons)
    assert "n_branch" not in cfg


# ---- llm-002 (2026-06-18): budget repair shrinks WIDTH before DEPTH ----


def test_repair_shrinks_width_not_depth_and_never_injects_omitted_nsteps():
    """An over-budget complexa_beam config that over-asks WIDTH (beam_width) but
    omits nsteps must be repaired by cutting beam_width — NOT by injecting/cutting
    nsteps (which the LLM never set). Depth is preserved."""
    from trex.capability_registry import (
        repair_config_delta_to_budget, compute_eval_budget)
    cap = default_registry().get("complexa_beam")
    cd = {"beam_width": 16, "n_branch": 8}   # 400*4*16*8 = 204800 >> 32768 cap
    repaired, notes = repair_config_delta_to_budget(cap, cd)
    assert repaired is not None, notes
    assert compute_eval_budget("complexa_beam", repaired) <= 65_536
    assert "nsteps" not in repaired                 # omitted knob never injected
    # at least one explicitly-set width knob was reduced
    assert repaired["beam_width"] <= 16 and (
        repaired["beam_width"] < 16 or repaired.get("n_branch", 8) < 8)


def test_repair_keeps_explicit_nsteps_at_or_above_default():
    """When the LLM explicitly over-asks BOTH depth and width, width is cut first and
    nsteps is never driven below its family default (400)."""
    from trex.capability_registry import (
        repair_config_delta_to_budget, compute_eval_budget)
    cap = default_registry().get("complexa_beam")
    cd = {"nsteps": 500, "nsamples": 8, "beam_width": 8}  # 500*8*8*4 = 128000 > cap
    repaired, notes = repair_config_delta_to_budget(cap, cd)
    assert repaired is not None, notes
    assert compute_eval_budget("complexa_beam", repaired) <= 65_536
    # nsteps, if present, is never below the family default 400 (depth preserved)
    assert int(repaired.get("nsteps", 400)) >= 400


def test_default_complexa_budget_matches_executor_400_steps():
    """nsteps default aligned to the executor (400), so the prompt's default_budget
    matches what actually runs: 400*4*4*4 = 25600."""
    from trex.capability_registry import compute_eval_budget
    assert compute_eval_budget("complexa_beam", {}) == 25_600


# ---- llm-004 (2026-06-18): wall gate scales by config_delta budget ----


def test_scaled_wall_reason_flags_heavy_config_near_end_of_run():
    """A heavy boltzgen config (num_designs=128,budget=16 → 32x the default budget)
    scales the expected runtime far past the flat FAMILY_RUNTIME_H, so near end-of-run
    it is flagged infeasible; the default config and ample-wall cases are not."""
    from types import SimpleNamespace
    from trex.candidate_builder import _scaled_wall_reason
    heavy = {"num_designs": 128, "budget": 16}
    assert _scaled_wall_reason("boltzgen", heavy, SimpleNamespace(remaining_wall_h=2.5)) is not None
    assert _scaled_wall_reason("boltzgen", heavy, SimpleNamespace(remaining_wall_h=100.0)) is None
    # default/empty config → base gate covers it, helper is a no-op
    assert _scaled_wall_reason("boltzgen", {}, SimpleNamespace(remaining_wall_h=2.5)) is None
    assert _scaled_wall_reason("boltzgen", None, SimpleNamespace(remaining_wall_h=2.5)) is None
    # unknown family / no remaining wall → None
    assert _scaled_wall_reason("nonexistent", heavy, SimpleNamespace(remaining_wall_h=2.5)) is None
    assert _scaled_wall_reason("boltzgen", heavy, SimpleNamespace(remaining_wall_h=0.0)) is None
