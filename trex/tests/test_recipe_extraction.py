"""Gap A': Recipe extraction from past results."""

from __future__ import annotations

import pytest

from trex.evidence_reducer import (
    ReducerConfig,
    _classify_result,
    _recipe_signature,
    extract_recipes,
)
from trex.schemas import ActionCandidate, FeasibilityCheck, ResultRecord


def _r(rid: str, *, family: str = "complexa_beam", panel_ready: bool = False,
       pLDDT: float | None = None, iPAE: float | None = None, scRMSD: float | None = None,
       exit_status: str = "ok") -> ResultRecord:
    m: dict[str, float] = {}
    if pLDDT is not None: m["pLDDT"] = pLDDT
    if iPAE is not None: m["iPAE"] = iPAE
    if scRMSD is not None: m["binder_scRMSD"] = scRMSD
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t", backend_family=family,
        runtime_bucket_id="rb1", metrics=m, metrics_calibrated=dict(m),
        route_lineage=[], gpu_h=1.0, exit_status=exit_status,  # type: ignore[arg-type]
        panel_ready=panel_ready,
    )


def _ac(cand_id: str, *, op: str = "complexa_beam_default", family: str = "complexa_beam",
        config: dict | None = None, parent_result_id: str | None = None) -> ActionCandidate:
    feas = FeasibilityCheck(True, "rb1", True, True, True, True)
    return ActionCandidate(
        candidate_id=cand_id, hypothesis_ids=["h1"], parent_result_id=parent_result_id,
        method_family=family, operator_id=op, lane_id=family,
        config_delta=config or {}, downstream_route_plan=[],
        estimated_cost_class="standard", expected_signal="x",
        evidence_refs=[], feasibility=feas,
    )


def test_classify_strict_success():
    r = _r("r1", pLDDT=92.0, iPAE=0.20, scRMSD=1.3)
    assert _classify_result(r, ReducerConfig()) == "strict_success"


def test_classify_panel_ready_overrides_strict():
    r = _r("r1", pLDDT=92.0, iPAE=0.20, scRMSD=1.3, panel_ready=True)
    assert _classify_result(r, ReducerConfig()) == "panel_ready"


def test_classify_near_miss_single_axis():
    # iPAE fails, pLDDT passes, scRMSD passes
    r = _r("r1", pLDDT=92.0, iPAE=0.55, scRMSD=1.3)
    assert _classify_result(r, ReducerConfig()) == "near_miss"


def test_classify_joint_fail():
    r = _r("r1", pLDDT=70.0, iPAE=0.55, scRMSD=1.3)
    assert _classify_result(r, ReducerConfig()) == "joint_fail"


def test_classify_skips_failed_exit():
    r = _r("r1", pLDDT=92.0, iPAE=0.20, scRMSD=1.3, exit_status="timeout")
    assert _classify_result(r, ReducerConfig()) is None


def test_signature_stable_across_dict_order():
    s1 = _recipe_signature("op", {"a": 1, "b": 2})
    s2 = _recipe_signature("op", {"b": 2, "a": 1})
    assert s1 == s2


def test_signature_differs_on_value_change():
    s1 = _recipe_signature("op", {"beam_width": 4})
    s2 = _recipe_signature("op", {"beam_width": 8})
    assert s1 != s2


def test_extract_recipes_aggregates_strict_successes():
    """Three strict successes sharing the same (operator, config) should
    collapse to one Recipe with descendant_count=3."""
    results = [
        _r(f"r{i}", pLDDT=92, iPAE=0.20, scRMSD=1.3) for i in range(3)
    ]
    cand = _ac("c1", config={"beam_width": 4, "n_branch": 8})
    spawning = {r.result_id: cand for r in results}
    recipes = extract_recipes(results, spawning_action=spawning)
    strict_recipes = [r for r in recipes if r.recipe_class == "strict_success"]
    assert len(strict_recipes) == 1
    assert strict_recipes[0].descendant_count == 3
    assert strict_recipes[0].config_delta == {"beam_width": 4, "n_branch": 8}


def test_extract_recipes_separates_classes():
    """Strict + near-miss + joint-fail should each produce their own recipe."""
    rs = [
        _r("r1", pLDDT=92, iPAE=0.20, scRMSD=1.3),       # strict
        _r("r2", pLDDT=92, iPAE=0.55, scRMSD=1.3),       # near-miss (iPAE only)
        _r("r3", pLDDT=70, iPAE=0.55, scRMSD=1.3),       # joint fail
    ]
    recipes = extract_recipes(rs)
    classes = {r.recipe_class for r in recipes}
    assert "strict_success" in classes
    assert "near_miss" in classes
    assert "joint_fail" in classes


def test_extract_recipes_top_k_limit():
    """Successful recipes retain distinct quality, support, and recency examples."""
    cands = [
        _ac(f"c{i}", config={"beam_width": w})
        for i, w in enumerate([4, 8, 16])
    ]
    results: list[ResultRecord] = []
    spawning = {}
    # recipe A: 3 strict descendants
    for i in range(3):
        r = _r(f"a{i}", pLDDT=92, iPAE=0.20, scRMSD=1.3)
        results.append(r); spawning[r.result_id] = cands[0]
    # recipe B: 2 strict descendants
    for i in range(2):
        r = _r(f"b{i}", pLDDT=92, iPAE=0.20, scRMSD=1.3)
        results.append(r); spawning[r.result_id] = cands[1]
    # recipe C: 1 strict descendant
    r = _r("c0", pLDDT=92, iPAE=0.20, scRMSD=1.3)
    results.append(r); spawning[r.result_id] = cands[2]

    recipes = extract_recipes(results, spawning_action=spawning, top_strict=2)
    strict = [r for r in recipes if r.recipe_class == "strict_success"]
    # The strict-example limit is applied separately from the near-miss limit.
    assert len(strict) == 3
    counts = sorted([r.descendant_count for r in strict], reverse=True)
    assert counts == [3, 2, 1]


def test_extract_recipes_fallback_when_no_spawning():
    """Without ActionCandidate info, fall back to (family, {}) signature."""
    results = [_r(f"r{i}", pLDDT=92, iPAE=0.20, scRMSD=1.3) for i in range(3)]
    recipes = extract_recipes(results, spawning_action={})
    strict = [r for r in recipes if r.recipe_class == "strict_success"]
    assert len(strict) == 1
    assert strict[0].operator_id == "complexa_beam_default"
    assert strict[0].descendant_count == 3


def test_stratified_strict_success_preserves_early_high_quality_winner():
    """Keep an early high-quality recipe despite newer, more frequent results."""
    # Recipe A: early (tick 5), high quality (pLDDT=95, iPAE=0.15, scRMSD=0.5),
    #           only 1 descendant.
    # Recipes B-E: late (tick 80-83), mediocre quality (pLDDT=91, iPAE=0.22,
    #              scRMSD=1.4), 8 descendants each.
    cands = {
        "A": _ac("cA", family="bindcraft", config={"max_traj": 16}),
    }
    for i in range(4):
        cands[f"L{i}"] = _ac(f"cL{i}", family="complexa_beam", config={"beam": 8, "n_branch": i})

    results = []
    spawning = {}
    # Recipe A: 1 early high-quality strict success (tick 5)
    r = ResultRecord(
        result_id="rA", parent_ids=[], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.5},
        metrics_calibrated={}, route_lineage=[], gpu_h=1.0,
        exit_status="ok", panel_ready=False, tick_id="v7r005",
    )
    results.append(r)
    spawning[r.result_id] = cands["A"]

    # Recipes B-E: 8 mediocre strict_success each, all at recent ticks 80-83
    for ri, fam_key in enumerate(("L0", "L1", "L2", "L3")):
        for j in range(8):
            r = ResultRecord(
                result_id=f"r{fam_key}_{j}", parent_ids=[], target_id="t",
                backend_family="complexa_beam", runtime_bucket_id="rb1",
                metrics={"pLDDT": 91.0, "iPAE": 0.22, "binder_scRMSD": 1.4},
                metrics_calibrated={}, route_lineage=[], gpu_h=1.0,
                exit_status="ok", panel_ready=False,
                tick_id=f"v7r{80+ri:03d}",
            )
            results.append(r)
            spawning[r.result_id] = cands[fam_key]

    recipes = extract_recipes(results, spawning_action=spawning, current_tick=100)
    strict = [r for r in recipes if r.recipe_class == "strict_success"]

    # The early high-quality bindcraft recipe MUST be present in the output
    # (it's #1 by quality even though dead-last by count/recency).
    bindcraft_strict = [r for r in strict if r.method_family == "bindcraft"]
    assert len(bindcraft_strict) == 1, (
        f"Early high-quality bindcraft winner pruned! got recipes: "
        f"{[(r.method_family, r.descendant_count, r.recency_tick) for r in strict]}"
    )
    # Quality score: pLDDT margin (95-90)/5=1.0, iPAE margin (7/31-0.15)/0.05=1.52,
    # scRMSD margin (1.5-0.5)/0.3=3.33 → total ~5.85. Mediocre ones ~0.
    # The bindcraft entry should be a strict_success recipe.
    assert bindcraft_strict[0].descendant_count == 1
    assert bindcraft_strict[0].recency_tick == 5


def test_stratified_dedup_when_same_recipe_top_in_multiple_buckets():
    """A recipe that is BOTH most-quality AND most-recent shouldn't appear
    twice in the output (dedup by recipe_hash)."""
    cands = {"A": _ac("cA", family="bindcraft", config={"x": 1})}
    results = []
    spawning = {}
    # Single high-quality + recent + few-descendants recipe
    for i in range(3):
        r = ResultRecord(
            result_id=f"r{i}", parent_ids=[], target_id="t",
            backend_family="bindcraft", runtime_bucket_id="rb1",
            metrics={"pLDDT": 96.0, "iPAE": 0.12, "binder_scRMSD": 0.4},
            metrics_calibrated={}, route_lineage=[], gpu_h=1.0,
            exit_status="ok", panel_ready=False, tick_id="v7r050",
        )
        results.append(r)
        spawning[r.result_id] = cands["A"]
    recipes = extract_recipes(results, spawning_action=spawning, current_tick=51)
    strict = [r for r in recipes if r.recipe_class == "strict_success"]
    assert len(strict) == 1  # de-duped, not 3 copies (quality/count/recency)


def test_method_health_strict_yield_su_dedup():
    """Distinguish repeated qualified records from structural diversity."""
    from trex.evidence_reducer import method_health
    rs = []
    # Family A: 5 strict passes but all in ONE foldseek cluster
    for i in range(5):
        rs.append(ResultRecord(
            result_id=f"rA{i}", parent_ids=[], target_id="t",
            backend_family="bindcraft", runtime_bucket_id="rb1",
            metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.5},
            metrics_calibrated={}, route_lineage=[], gpu_h=0.2,
            exit_status="ok", panel_ready=False,
            bins={"foldseek": "foldseek:repA", "foldseek_su": "foldseek:repA"},  # same cluster
        ))
    # Family B: 3 strict passes in 3 distinct clusters
    for i in range(3):
        rs.append(ResultRecord(
            result_id=f"rB{i}", parent_ids=[], target_id="t",
            backend_family="complexa_beam", runtime_bucket_id="rb1",
            metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.5},
            metrics_calibrated={}, route_lineage=[], gpu_h=0.2,
            exit_status="ok", panel_ready=False,
            bins={"foldseek": f"foldseek:repB_{i}", "foldseek_su": f"foldseek:repB_{i}"},  # distinct clusters
        ))
    mh = method_health(rs)
    # bindcraft: 5 raw strict, 1 SU (all same cluster)
    assert mh["bindcraft"].strict_yield == 5
    assert mh["bindcraft"].strict_yield_su == 1, (
        f"expected SU=1, got {mh['bindcraft'].strict_yield_su}"
    )
    # complexa_beam: 3 raw strict, 3 SU (all distinct)
    assert mh["complexa_beam"].strict_yield == 3
    assert mh["complexa_beam"].strict_yield_su == 3


def _strict_rec(rid, family, su_bin, exit_status="ok"):
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t", backend_family=family,
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.5},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.5,
        exit_status=exit_status, panel_ready=False,  # type: ignore[arg-type]
        bins={"foldseek_su": su_bin},
    )


def test_method_health_per_family_su_no_cross_family_double_count():
    """Assign a shared strict cluster one family owner."""
    from trex.evidence_reducer import method_health
    rs = [
        _strict_rec("a", "complexa_beam", "shared"),
        _strict_rec("b", "complexa_fk_steering", "shared"),  # same structure, other family
        _strict_rec("c", "complexa_beam", "other"),
    ]
    mh = method_health(rs)
    total = sum(m.strict_yield_su for m in mh.values())
    assert total == 2, f"shared cluster counts once across families; got sum={total}"


def test_strict_yield_requires_exit_ok():
    """Strict and structure-unique counts share the successful-exit condition."""
    from trex.evidence_reducer import method_health
    mh = method_health([_strict_rec("x", "complexa_beam", "c", exit_status="timeout")])
    assert mh["complexa_beam"].strict_yield == 0
    assert mh["complexa_beam"].strict_yield_su == 0


def test_method_health_strict_yield_su_requires_official_foldseek_bin():
    """When foldseek_su is missing, strict records remain strict_yield evidence
    but must not mint official SU credit."""
    from trex.evidence_reducer import method_health
    rs = []
    for i in range(3):
        rs.append(ResultRecord(
            result_id=f"r{i}", parent_ids=[], target_id="t",
            backend_family="bindcraft", runtime_bucket_id="rb1",
            metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.5},
            metrics_calibrated={}, route_lineage=[], gpu_h=0.2,
            exit_status="ok", panel_ready=False,
            bins={"design": f"d{i}"},  # NO foldseek key
        ))
    mh = method_health(rs)
    assert mh["bindcraft"].strict_yield == 3
    assert mh["bindcraft"].strict_yield_su == 0


def test_chained_refilter_success_credits_upstream_generator():
    """Credit evaluated designs to their generating family."""
    from trex.evidence_reducer import method_health
    parent = ResultRecord(
        result_id="bc_parent", parent_ids=["bc_cand"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=2.0,
        exit_status="ok", panel_ready=False,
    )
    child = ResultRecord(
        result_id="refilter_child", parent_ids=["chain_cand", "bc_parent"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.8},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
        exit_status="ok", panel_ready=False, bins={"foldseek_su": "FS_new"},
    )
    spawning = {
        "bc_parent": _ac(
            "bc_cand", family="bindcraft", op="bindcraft_default",
            config={"max_trajectories": 16},
        ),
        "refilter_child": _ac(
            "chain_cand", family="structure_refilter", op="af2_multimer",
            parent_result_id="bc_parent",
        ),
    }
    mh = method_health([parent, child], spawning)
    # The generating family owns the SU; the evaluator does not receive a second credit.
    assert mh["structure_refilter"].strict_yield_su == 0
    assert mh["structure_refilter"].su_per_gpu_h is None
    assert mh["bindcraft"].strict_yield_su == 1
    # Both yield fields retain generator attribution for chained evaluation outcomes.
    assert mh["bindcraft"].chained_strict_yield_su == 1
    # Include upstream generation and downstream evaluation in the route denominator.
    assert mh["bindcraft"].chained_su_per_gpu_h == 1.0 / 2.05


def test_chained_refilter_recipe_uses_upstream_config():
    parent = ResultRecord(
        result_id="bc_parent", parent_ids=["bc_cand"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb1",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=2.0,
        exit_status="ok", panel_ready=False,
    )
    child = ResultRecord(
        result_id="refilter_child", parent_ids=["chain_cand", "bc_parent"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb1",
        metrics={"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.8},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05,
        exit_status="ok", panel_ready=False, bins={"foldseek_su": "FS_new"},
    )
    spawning = {
        "bc_parent": _ac(
            "bc_cand", family="bindcraft", op="bindcraft_default",
            config={"max_trajectories": 16},
        ),
        "refilter_child": _ac(
            "chain_cand", family="structure_refilter", op="af2_multimer",
            parent_result_id="bc_parent",
        ),
    }
    recipes = extract_recipes([parent, child], spawning_action=spawning)
    strict = [r for r in recipes if r.recipe_class == "strict_success"]
    assert len(strict) == 1
    assert strict[0].method_family == "bindcraft"
    assert strict[0].operator_id == "bindcraft_default"
    assert strict[0].config_delta == {"max_trajectories": 16}
    assert strict[0].su_per_gpu_h is not None
