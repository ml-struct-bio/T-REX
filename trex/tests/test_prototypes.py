"""Tests for the two architecture prototypes:
  #2 unified_reasoner — derive the Supervisor ranking from Planner cards (no 2nd LLM call).
  #3 diagnosis_outcome — close the diagnosis→outcome loop over the archive lineage.
"""
from __future__ import annotations

from trex.diagnosis_outcome import (
    compute_diagnosis_outcomes,
    format_diagnosis_outcomes_tldr,
)
from trex.schemas import (
    ActionCandidate,
    FeasibilityCheck,
    HypothesisCard,
    PredictedChange,
    PlannerOutput,
    ReasoningTrace,
    ResultRecord,
)
from trex.unified_reasoner import build_unified_supervisor_output


def _card(hid: str, mode: str) -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hid, target_id="t", tick_created=0, claim=f"claim {hid}",
        mode_affinity={"exploit": 0.0, "rescue": 0.0, "explore": 0.0, mode: 1.0},
        evidence_refs=[], predicted_metric_changes=[], preserve_constraints=[],
        recommended_action_families=["complexa_beam"],
        reasoning_trace=ReasoningTrace(inference=f"why {hid}"),
    )


def _cand(cid: str, hid: str, cost: str = "standard") -> ActionCandidate:
    feas = FeasibilityCheck(True, "rb", True, True, True, True)
    return ActionCandidate(
        candidate_id=cid, hypothesis_ids=[hid], parent_result_id=None,
        method_family="complexa_beam", operator_id="beam", lane_id="l",
        config_delta={}, downstream_route_plan=[], estimated_cost_class=cost,
        expected_signal="sig", evidence_refs=[], feasibility=feas,
    )


def _action(
    cid: str,
    family: str,
    *,
    parent_result_id: str | None = None,
    baseline_result_id: str | None = None,
    expected_signal: str = "test",
) -> ActionCandidate:
    feas = FeasibilityCheck(True, "rb", True, True, True, True)
    return ActionCandidate(
        candidate_id=cid,
        hypothesis_ids=["h1"],
        parent_result_id=parent_result_id,
        method_family=family,
        operator_id=f"{family}_default",
        lane_id=family,
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",
        expected_signal=expected_signal,
        evidence_refs=[],
        feasibility=feas,
        baseline_result_id=baseline_result_id,
    )


def _planner(cards) -> PlannerOutput:
    return PlannerOutput(
        valid=True, abstain=False, confidence=0.7, fail_reason=None,
        cards=cards, rationale="r", raw_text="{}", usage={},
    )


# ---- Prototype #2: unified reasoner -----------------------------------------

def test_unified_reasoner_derives_modes_and_ranks_from_cards():
    cards = [_card("h_exploit", "exploit"), _card("h_explore", "explore")]
    cands = [
        _cand("c1", "h_exploit"), _cand("c2", "h_exploit"),  # 2 exploit
        _cand("c3", "h_explore"),                            # 1 explore
    ]
    out = build_unified_supervisor_output(_planner(cards), cands)
    assert out.valid and not out.abstain
    modes = {d.candidate_id: d.mode for d in out.candidate_decisions}
    assert modes == {"c1": "exploit", "c2": "exploit", "c3": "explore"}
    # ranks are 1..N within each mode
    ranks = {d.candidate_id: d.rank_in_mode for d in out.candidate_decisions}
    assert {ranks["c1"], ranks["c2"]} == {1, 2} and ranks["c3"] == 1
    # the DECISIONS (what the hybrid derives the operative budget from) are the
    # candidate-level 2-exploit : 1-explore split
    n_exploit = sum(1 for d in out.candidate_decisions if d.mode == "exploit")
    n_explore = sum(1 for d in out.candidate_decisions if d.mode == "explore")
    assert (n_exploit, n_explore) == (2, 1)
    # the scalar mode_mixture is the normalized CARD-affinity aggregate (1:1 here,
    # the fallback only; the hybrid uses the decisions above)
    assert abs(sum(out.mode_mixture.values()) - 1.0) < 1e-9
    # carries the why from the card reasoning_trace
    assert any(d.why for d in out.candidate_decisions)


def test_unified_reasoner_empty_cards_abstains():
    out = build_unified_supervisor_output(_planner([]), [])
    assert not out.valid and out.abstain


def test_unified_degenerate_ranking_is_clamped_by_safeguard():
    # The unified path has NO Supervisor LLM, so the deterministic clamp is the
    # safeguard. All-exploit cards → unified derives an exploit-only mixture; the
    # ALWAYS-ON Category-A floor must still inject the explore diversity insurance.
    from trex.fallback import clamp_mixture
    cards = [_card("h1", "exploit"), _card("h2", "exploit")]
    cands = [_cand("c1", "h1"), _cand("c2", "h2")]
    sup = build_unified_supervisor_output(_planner(cards), cands)
    assert sup.mode_mixture.get("explore", 0.0) == 0.0          # degenerate derived
    for state in ("productive", "low_evidence", "stalled"):
        clamped, _ = clamp_mixture(sup.mode_mixture, state, category_b_enabled=False)
        assert clamped["explore"] >= 0.05, (state, clamped)     # safeguard floors it
        assert abs(sum(clamped.values()) - 1.0) < 1e-9


def test_unified_invalid_planner_is_treated_as_fallback():
    # invalid/empty planner → unified sup_out must be INVALID so the Selector
    # deterministically falls back to DEFAULT_MIXTURES + candidate_mode_hint.
    sup = build_unified_supervisor_output(_planner([]), [])
    assert not sup.valid and sup.abstain


def test_unified_reasoner_candidate_without_card_defaults_explore():
    # a chain/auto-merged candidate whose card is absent → explore default
    out = build_unified_supervisor_output(_planner([_card("h1", "exploit")]),
                                          [_cand("orphan", "missing_card")])
    assert out.candidate_decisions[0].mode == "explore"


# ---- Prototype #3: diagnosis → outcome loop ---------------------------------

def _rec(rid, parents, **metrics):
    return ResultRecord(
        result_id=rid, parent_ids=parents, target_id="t",
        backend_family="complexa_beam", runtime_bucket_id="rb",
        metrics=metrics, metrics_calibrated={}, route_lineage=[], gpu_h=0.1,
        exit_status="ok", bins={},
    )


def test_diagnosis_outcome_tracks_remediation_improvement_and_su():
    # parent is ipTM-blocked (0.65 < 0.75 quality, levered); child improves ipTM
    # to 0.85 AND is a strict success.
    parent = _rec("p", [], pLDDT=92.0, iPAE=0.20, binder_scRMSD=1.2, ipTM=0.65)
    child = _rec("c", ["p"], pLDDT=93.0, iPAE=0.18, binder_scRMSD=1.1, ipTM=0.85)
    out = compute_diagnosis_outcomes([parent, child])
    assert "ipTM" in out
    o = out["ipTM"]
    assert o["attempts"] == 1 and o["improved"] == 1 and o["strict"] == 1
    assert o["improve_rate"] == 1.0 and o["strict_rate"] == 1.0
    assert o["unique_su"] == 1 and o["unique_su_rate"] == 1.0
    assert o["provisional"] is True and o["prefer_key"] == "unique_su_rate"
    assert "weights_iptm" in (o["lever"] or "")
    tldr = format_diagnosis_outcomes_tldr(out)
    assert "ipTM" in tldr and "strict" in tldr and "1 SU" in tldr


def test_diagnosis_outcome_credits_generator_via_auto_refilter_lineage():
    # BindCraft/BoltzGen/MPNN attempts are diagnostic-only. Their strict outcome
    # lives on the downstream chain_* structure_refilter record, but the attempt
    # baseline is the generator ActionCandidate's baseline_result_id.
    parent = _rec("p", [], pLDDT=92.0, iPAE=0.20, binder_scRMSD=1.2, ipTM=0.65)
    gen = ResultRecord(
        result_id="g", parent_ids=["bc_cand"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=1.0,
        exit_status="ok", bins={}, artifacts={"pdb_path": "/tmp/g.pdb"},
    )
    scored = ResultRecord(
        result_id="r", parent_ids=["chain_cand", "g"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 94.0, "iPAE": 0.18, "binder_scRMSD": 1.0, "ipTM": 0.85},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
        exit_status="ok", bins={"refilter_source": "g"},
    )
    spawning = {
        "g": _action(
            "bc_cand", "bindcraft", baseline_result_id="p",
            expected_signal="bindcraft rescue"),
        "r": _action(
            "chain_cand", "structure_refilter", parent_result_id="g",
            expected_signal="auto_chain:bindcraft->structure_refilter parent=g"),
    }
    out = compute_diagnosis_outcomes([parent, gen, scored], spawning_actions=spawning)
    assert out["ipTM"]["attempts"] == 1
    assert out["ipTM"]["improved"] == 1
    assert out["ipTM"]["strict"] == 1
    assert out["ipTM"]["unique_su"] == 1


def test_diagnosis_outcome_separates_raw_strict_from_unique_su():
    parent = _rec("p", [], pLDDT=92.0, iPAE=0.20, binder_scRMSD=1.2, ipTM=0.65)
    c1 = _rec("c1", ["p"], pLDDT=93.0, iPAE=0.18, binder_scRMSD=1.1, ipTM=0.85)
    c2 = _rec("c2", ["p"], pLDDT=94.0, iPAE=0.17, binder_scRMSD=1.0, ipTM=0.86)
    c1.bins["foldseek_su"] = "cluster_A"
    c2.bins["foldseek_su"] = "cluster_A"
    out = compute_diagnosis_outcomes([parent, c1, c2])
    o = out["ipTM"]
    assert o["attempts"] == 2
    assert o["strict"] == 2
    assert o["unique_su"] == 1
    assert o["strict_rate"] == 1.0
    assert o["unique_su_rate"] == 0.5


def test_diagnosis_outcome_tracks_strict_axis_prediction_via_auto_refilter():
    parent = _rec("p", [], pLDDT=92.0, iPAE=0.30, binder_scRMSD=1.2, ipTM=0.90)
    hyp = HypothesisCard(
        hypothesis_id="h1", target_id="t", tick_created=0,
        claim="lower iPAE on the near miss",
        mode_affinity={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        evidence_refs=["p"],
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["p"], 0.2, None),
        ],
        preserve_constraints=[],
        recommended_action_families=["bindcraft"],
    )
    gen = ResultRecord(
        result_id="g", parent_ids=["bc_cand"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=1.0,
        exit_status="ok", bins={}, artifacts={"pdb_path": "/tmp/g.pdb"},
    )
    scored = ResultRecord(
        result_id="r", parent_ids=["chain_cand", "g"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 94.0, "iPAE": 0.18, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
        exit_status="ok", bins={"refilter_source": "g"},
    )
    spawning = {
        "g": _action("bc_cand", "bindcraft", baseline_result_id="p"),
        "r": _action(
            "chain_cand", "structure_refilter", parent_result_id="g",
            expected_signal="auto_chain:bindcraft->structure_refilter parent=g"),
    }
    out = compute_diagnosis_outcomes(
        [parent, gen, scored], spawning_actions=spawning, hypotheses=[hyp])
    assert out["iPAE"]["attempts"] == 1
    assert out["iPAE"]["improved"] == 1
    assert out["iPAE"]["strict"] == 1


def test_diagnosis_outcome_counts_non_improvement():
    # child does NOT improve ipTM (0.60 < parent 0.65 for an increase-axis) and is not strict
    parent = _rec("p", [], pLDDT=92.0, iPAE=0.20, binder_scRMSD=1.2, ipTM=0.65)
    child = _rec("c", ["p"], pLDDT=70.0, iPAE=0.40, binder_scRMSD=2.0, ipTM=0.60)
    out = compute_diagnosis_outcomes([parent, child])
    o = out["ipTM"]
    assert o["attempts"] == 1 and o["improved"] == 0 and o["strict"] == 0



def test_diagnosis_outcome_ignores_sub_margin_jitter():
    parent = _rec("p", [], pLDDT=92.0, iPAE=0.20, binder_scRMSD=1.2, ipTM=0.65)
    child = _rec("c", ["p"], pLDDT=70.0, iPAE=0.40, binder_scRMSD=2.0, ipTM=0.651)
    out = compute_diagnosis_outcomes([parent, child])
    o = out["ipTM"]
    assert o["attempts"] == 1 and o["improved"] == 0
    assert o["improve_rate"] == 0.0

def test_diagnosis_outcome_ignores_unblocked_parents():
    # parent has a clean ipTM (0.90 passes) → no actionable blocker → no tracking
    parent = _rec("p", [], pLDDT=92.0, iPAE=0.20, binder_scRMSD=1.2, ipTM=0.90)
    child = _rec("c", ["p"], pLDDT=93.0, iPAE=0.18, binder_scRMSD=1.1, ipTM=0.95)
    assert compute_diagnosis_outcomes([parent, child]) == {}
