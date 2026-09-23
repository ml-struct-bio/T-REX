"""Test one-owner structural credit attributed through evaluation lineage."""
from __future__ import annotations

from trex.evidence_reducer import (
    build_route_values,
    method_health,
    StateClassifierConfig,
    refilter_role_health,
    resolve_generating_family,
)
from trex.panel import strict_margin_quality
from trex.refilter_roles import CANONICAL_SCORE_CONVERSION, PARENT_MODEL_REFOLD
from trex.schemas import ActionCandidate, FeasibilityCheck, ResultRecord


def _gen(rid, fam, *, strict=False, metrics=None):
    m = metrics if metrics is not None else (
        {"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.0} if strict else {})
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t", backend_family=fam,
        runtime_bucket_id="rb", metrics=m, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.1, exit_status="ok", bins={},
    )


def _refilter(rid, source_id, *, su_bin, parents=None):
    return ResultRecord(
        result_id=rid, parent_ids=parents or ["chain_x", source_id],
        target_id="t", backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={"refilter_source": source_id, "foldseek_su": su_bin},
    )


def test_raw_strict_records_without_foldseek_su_do_not_mint_su_credit():
    from trex.evidence_reducer import (
        ReducerConfig,
        build_strategy_feedback,
        extract_recipes,
    )

    direct = _gen("cx_raw", "complexa_beam", strict=True)
    generator = _gen("mp_raw", "proteinmpnn_redesign")
    refilter_no_su = ResultRecord(
        result_id="rf_raw", parent_ids=["chain_x", "mp_raw"],
        target_id="t", backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 93.0, "iPAE": 0.18, "binder_scRMSD": 0.9},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={"refilter_source": "mp_raw"},
    )
    recs = [direct, generator, refilter_no_su]

    mh = method_health(recs, spawning_actions={})
    assert mh["complexa_beam"].strict_yield == 1
    assert mh["complexa_beam"].strict_yield_su == 0
    assert mh["proteinmpnn_redesign"].chained_strict_yield_su == 0

    routes = build_route_values(recs, [direct, refilter_no_su], {}, cfg=StateClassifierConfig())
    assert sum(row.strict_count for row in routes if row.scope == "route") == 2
    assert all(row.new_su == 0 for row in routes)
    assert all(row.new_su_per_route_gpu_h is None for row in routes)

    recipes = extract_recipes(recs, spawning_action={}, cfg=ReducerConfig())
    strict_recipes = [r for r in recipes if r.recipe_class == "strict_success"]
    assert strict_recipes
    assert all(r.su_per_gpu_h is None for r in strict_recipes)

    feedback = build_strategy_feedback(recs, {}, ReducerConfig())
    assert any(row["strict_count"] > 0 for row in feedback)
    assert all(row["strict_su"] == 0 for row in feedback)
    assert all(row["su_per_gpu_h"] is None for row in feedback)


def test_parent_model_refold_strict_metrics_are_advisory_not_su():
    parent = _gen("cx1", "complexa_beam")
    refold = ResultRecord(
        result_id="rf_parent_model", parent_ids=["manual_refold", "cx1"],
        target_id="t", backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 94.0, "iPAE": 0.15, "binder_scRMSD": 0.8},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={
            "refilter_source": "cx1",
            "foldseek_su": "cA",
            "refilter_role": PARENT_MODEL_REFOLD,
        },
    )

    mh = method_health([parent, refold], spawning_actions={})
    assert mh["structure_refilter"].strict_yield_su == 0
    assert sum(row.strict_yield_su for row in mh.values()) == 0

    role_health = refilter_role_health([parent, refold], spawning_actions={})
    assert role_health[PARENT_MODEL_REFOLD]["strict_yield"] == 1
    assert role_health[PARENT_MODEL_REFOLD]["strict_yield_su"] == 0

def test_chained_refilter_su_credited_to_generator():
    bc = _gen("bc1", "bindcraft")                 # diagnostic generator (metrics={})
    rf = _refilter("rf1", "bc1", su_bin="cA")     # strict re-score of bc1
    mh = method_health([bc, rf], spawning_actions={})
    assert mh["bindcraft"].strict_yield_su == 1               # generator gets the SU
    assert mh["structure_refilter"].strict_yield_su == 0      # scorer gets 0


def test_refilter_of_refilter_recurses_to_generator():
    bc = _gen("bc1", "bindcraft")
    rf1 = _refilter("rf1", "bc1", su_bin="cA")
    rf2 = _refilter("rf2", "rf1", su_bin="cA")    # refilter of a refilter, same cluster
    fam = resolve_generating_family(
        rf2, by_result_id={"bc1": bc, "rf1": rf1, "rf2": rf2}, spawning_actions={})
    assert fam == "bindcraft"


def test_proteinmpnn_one_hop_credits_mpnn_not_backbone():
    # one-hop: the redesigned sequence is what passed → credit proteinmpnn
    mpnn = _gen("mp1", "proteinmpnn_redesign")
    rf = _refilter("rf1", "mp1", su_bin="cA")
    fam = resolve_generating_family(
        rf, by_result_id={"mp1": mpnn, "rf1": rf}, spawning_actions={})
    assert fam == "proteinmpnn_redesign"


def test_ssot_invariant_sum_equals_su_clusters():
    # two generators, each rescored to a distinct SU cluster → sum == 2,
    # structure_refilter owns 0
    recs = [
        _gen("bc1", "bindcraft"), _refilter("rf1", "bc1", su_bin="cA"),
        _gen("bg1", "boltzgen"), _refilter("rf2", "bg1", su_bin="cB"),
    ]
    mh = method_health(recs, spawning_actions={})
    total = sum(s.strict_yield_su for s in mh.values())
    assert total == 2
    assert mh["bindcraft"].strict_yield_su == 1
    assert mh["boltzgen"].strict_yield_su == 1
    assert mh["structure_refilter"].strict_yield_su == 0


def test_effective_provenance_credits_generator_for_all_planner_blocks():
    # Planner evidence credits the generating family through evaluation lineage.
    from trex.evidence_reducer import (
        _effective_provenance, resolve_generating_record, resolve_generating_family,
    )
    cx = _gen("cx1", "complexa_beam")              # NON-diagnostic generator
    rf = _refilter("rf1", "cx1", su_bin="cA")
    by_id = {"cx1": cx, "rf1": rf}
    assert resolve_generating_record(rf, by_result_id=by_id, spawning_actions={}).result_id == "cx1"
    assert resolve_generating_family(rf, by_result_id=by_id, spawning_actions={}) == "complexa_beam"
    fam, _op, _cd, _p = _effective_provenance(rf, by_result_id=by_id, spawning_actions={})
    assert fam == "complexa_beam"                  # NOT structure_refilter


def test_unresolvable_source_falls_back_to_refilter():
    # a refilter whose source id is missing from the archive → keep refilter
    rf = _refilter("rf1", "ghost", su_bin="cA")
    fam = resolve_generating_family(rf, by_result_id={"rf1": rf}, spawning_actions={})
    assert fam == "structure_refilter"


def _action(cid, family, *, parent=None, role=None):
    return ActionCandidate(
        candidate_id=cid, hypothesis_ids=["h"], parent_result_id=parent,
        method_family=family, operator_id="op", lane_id=family, config_delta={},
        downstream_route_plan=[], estimated_cost_class="low", expected_signal="x",
        evidence_refs=[],
        feasibility=FeasibilityCheck(
            backend_healthy=True, runtime_bucket_id="rb", compiler_ok=True,
            verifier_ok=True, route_cap_ok=True, cost_ok=True,
        ),
        refilter_role=role,
    )


def test_refilter_role_health_splits_canonical_from_manual_refold():
    bc = _gen("bc1", "bindcraft")
    auto = _refilter("rf_auto", "bc1", su_bin="cA", parents=["chain_auto", "bc1"])
    auto = ResultRecord(**{
        **auto.__dict__,
        "bins": {
            **auto.bins,
            "refilter_role": CANONICAL_SCORE_CONVERSION,
            "refilter_source_family": "bindcraft",
        },
    })
    # A parent_model_refold is ADVISORY (option b): the parser writes refold_* not
    # the canonical strict keys, so it mints NO SU and is_strict_success is False.
    manual = _refilter("rf_manual", "bc1", su_bin="cB", parents=["cand_manual", "bc1"])
    manual = ResultRecord(**{
        **manual.__dict__,
        "metrics": {"refold_pLDDT": 92.0, "refold_iPAE": 0.20,
                    "refold_binder_scRMSD": 1.0},
        "bins": {
            **manual.bins,
            "refilter_role": PARENT_MODEL_REFOLD,
            "refilter_source_family": "bindcraft",
            "af2_strict_basis": "advisory_refold",
        },
    })
    spawning = {
        "rf_auto": _action(
            "chain_auto", "structure_refilter", parent="bc1",
            role=CANONICAL_SCORE_CONVERSION,
        ),
        "rf_manual": _action(
            "cand_manual", "structure_refilter", parent="bc1",
            role=PARENT_MODEL_REFOLD,
        ),
    }

    mh = method_health([bc, auto, manual], spawning_actions=spawning)
    # canonical conversion mints the SU, credited to the generator
    assert mh["bindcraft"].strict_yield_su == 1
    assert mh["bindcraft"].chained_strict_yield_su == 1
    # the parent_model_refold is advisory → mints NO canonical SU
    assert mh["structure_refilter"].strict_yield_su == 0

    roles = refilter_role_health([bc, auto, manual], spawning)
    assert roles[CANONICAL_SCORE_CONVERSION]["attempts"] == 1
    assert roles[CANONICAL_SCORE_CONVERSION]["strict_yield_su"] == 1
    assert roles[CANONICAL_SCORE_CONVERSION]["source_families"] == {"bindcraft": 1}
    # the refold is still logged as an attempt, but yields 0 strict SU
    assert roles[PARENT_MODEL_REFOLD]["attempts"] == 1
    assert roles[PARENT_MODEL_REFOLD]["strict_yield_su"] == 0
    assert roles[PARENT_MODEL_REFOLD]["credit_policy"] == "intentional_refold_action_advisory_role_credit"


def test_parent_model_refold_is_advisory_does_not_mint_su():
    bc = _gen("bc1", "bindcraft")
    # advisory refold record as the parser produces it (no canonical strict keys)
    manual = _refilter("rf_manual", "bc1", su_bin="cB", parents=["cand_manual", "bc1"])
    manual = ResultRecord(**{
        **manual.__dict__,
        "metrics": {"refold_pLDDT": 92.0, "refold_iPAE": 0.20,
                    "refold_binder_scRMSD": 1.0},
        "bins": {
            **manual.bins,
            "refilter_role": PARENT_MODEL_REFOLD,
            "refilter_source_family": "bindcraft",
            "af2_strict_basis": "advisory_refold",
        },
    })
    spawning = {
        "rf_manual": _action(
            "cand_manual", "structure_refilter", parent="bc1",
            role=PARENT_MODEL_REFOLD,
        ),
    }
    by_id = {"bc1": bc, "rf_manual": manual}

    # strategy attribution still resolves the refold to structure_refilter (for the
    # advisory role ledger), but the refold mints NO canonical SU.
    assert resolve_generating_family(
        manual, by_result_id=by_id, spawning_actions=spawning,
    ) == "structure_refilter"
    mh = method_health([bc, manual], spawning_actions=spawning)
    assert mh["structure_refilter"].strict_yield_su == 0
    assert mh["bindcraft"].strict_yield_su == 0


def _roled_refilter(rid, source_id, *, su_bin, role, parents):
    return ResultRecord(
        result_id=rid, parent_ids=parents, target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={"refilter_source": source_id, "foldseek_su": su_bin,
              "refilter_role": role},
    )


def test_shared_cluster_owner_is_generator_regardless_of_order():
    """Cluster ownership must not depend on record order."""
    bc = _gen("bc1", "bindcraft")
    canon = _roled_refilter("rf_auto", "bc1", su_bin="cShared",
                            role=CANONICAL_SCORE_CONVERSION, parents=["chain_a", "bc1"])
    refold = _roled_refilter("rf_manual", "bc1", su_bin="cShared",
                             role=PARENT_MODEL_REFOLD, parents=["cand_m", "bc1"])
    for order in ([bc, canon, refold], [bc, refold, canon], [refold, canon, bc]):
        mh = method_health(order, spawning_actions={})
        assert mh["bindcraft"].strict_yield_su == 1, order
        assert mh["structure_refilter"].strict_yield_su == 0, order
        assert sum(s.strict_yield_su for s in mh.values()) == 1, order  # one cluster


def test_exemplar_diagnostic_blocking_axis_uses_generator_record():
    """On diagnostic-only-generator targets the strict SU (best exemplar) is a
    structure_refilter record carrying only pLDDT/iPAE/binder_scRMSD/ipTM. The
    advisory per-design diagnostic_blocking_axis must be resolved over the
    GENERATING record so the generator's interface blocker (here min_ipae) survives
    instead of reading ipTM/None off the refilter."""
    from trex.evidence_reducer import ReducerConfig, build_exemplars

    gen = ResultRecord(
        result_id="cx1", parent_ids=[], target_id="t", backend_family="complexa_beam",
        runtime_bucket_id="rb", metrics={"min_ipae": 0.30, "ipTM": 0.85},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status="ok", bins={},
    )
    refilter = ResultRecord(
        result_id="rf1", parent_ids=["chain_x", "cx1"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.0, "ipTM": 0.85},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.05, exit_status="ok",
        bins={"refilter_source": "cx1", "foldseek_su": "cA"},
    )
    best = [e for e in build_exemplars([gen, refilter], {}, ReducerConfig())
            if e.kind == "best"]
    assert len(best) == 1
    # The failing generator diagnostic remains visible even when the evaluation ipTM passes.
    assert best[0].diagnostic_blocking_axis == "min_ipae"
    assert best[0].family == "complexa_beam"  # provenance credit unchanged


# ---- route-value ledger: exact config + lineage-aware route economics ----


def _configured_action(cid, family, *, op="op", config=None, parent=None, role=None):
    a = _action(cid, family, parent=parent, role=role)
    return ActionCandidate(**{
        **a.__dict__,
        "operator_id": op,
        "config_delta": dict(config or {}),
    })


def test_route_values_split_exact_complexa_configs_and_charge_refilter_upstream():
    from trex.evidence_reducer import build_route_values, canonical_config_signature

    cx4 = _gen("cx4", "complexa_beam")
    cx8 = _gen("cx8", "complexa_beam")
    rf4 = _refilter("rf4", "cx4", su_bin="cA")
    rf8 = _refilter("rf8", "cx8", su_bin="cB")
    spawning = {
        "cx4": _configured_action(
            "cand_cx4", "complexa_beam", op="complexa_beam",
            config={"beam_width": 4, "beam_depth": 4},
        ),
        "cx8": _configured_action(
            "cand_cx8", "complexa_beam", op="complexa_beam",
            config={"beam_width": 8, "beam_depth": 4},
        ),
    }

    rows = build_route_values([cx4, cx8, rf4, rf8], [rf4, rf8], spawning)
    exact = [r for r in rows if r.scope == "route" and r.action_family == "complexa_beam"]
    sig4 = canonical_config_signature({"beam_width": 4, "beam_depth": 4})
    sig8 = canonical_config_signature({"beam_width": 8, "beam_depth": 4})
    by_sig = {r.config_signature: r for r in exact}

    assert {sig4, sig8} <= set(by_sig)
    assert by_sig[sig4].new_su == 1
    assert by_sig[sig8].new_su == 1
    assert by_sig[sig4].canonical_refilter_gpu_h == rf4.gpu_h
    assert by_sig[sig4].route_gpu_h == cx4.gpu_h + rf4.gpu_h


def test_route_quality_is_canonical_and_deduplicated_by_foldseek_bin():
    from dataclasses import replace

    cx = _gen("cx", "complexa_beam")
    low_a = replace(
        _refilter("rf_a1", "cx", su_bin="A"),
        metrics={"pLDDT": 91.0, "iPAE": 0.22, "binder_scRMSD": 1.4},
    )
    duplicate_high_a = replace(
        _refilter("rf_a2", "cx", su_bin="A"),
        metrics={"pLDDT": 99.0, "iPAE": 0.05, "binder_scRMSD": 0.2},
    )
    high_b = replace(
        _refilter("rf_b", "cx", su_bin="B"),
        metrics={"pLDDT": 96.0, "iPAE": 0.10, "binder_scRMSD": 0.5},
    )
    spawning = {
        "cx": _configured_action(
            "cand_cx", "complexa_beam", op="complexa_beam",
            config={"beam_width": 4},
        ),
    }

    rows = build_route_values(
        [cx, low_a, duplicate_high_a, high_b],
        [low_a, duplicate_high_a, high_b],
        spawning,
    )
    route = next(
        r for r in rows
        if r.scope == "route" and r.action_family == "complexa_beam"
    )

    assert route.strict_count == 3
    assert route.new_su == 2
    assert route.strict_quality_n_unique_bins == 2
    expected_median = (
        strict_margin_quality(duplicate_high_a) + strict_margin_quality(high_b)
    ) / 2
    assert route.strict_quality_median == expected_median
    assert route.strict_quality_p25 is not None
    assert set(route.strict_quality_axis_margins) == {
        "pLDDT", "iPAE", "binder_scRMSD",
    }


def test_route_values_preserve_proteinmpnn_root_family_context():
    from trex.evidence_reducer import build_route_values

    cx = _gen("cx1", "complexa_beam")
    mp = _gen("mp1", "proteinmpnn_redesign")
    rf = _refilter("rf1", "mp1", su_bin="cA")
    spawning = {
        "cx1": _configured_action(
            "cand_cx", "complexa_beam", op="complexa_beam",
            config={"beam_width": 4},
        ),
        "mp1": _configured_action(
            "cand_mp", "proteinmpnn_redesign", op="proteinmpnn_redesign",
            config={"temperature": 0.1}, parent="cx1",
        ),
    }

    rows = build_route_values([cx, mp, rf], [rf], spawning)
    mp_rows = [r for r in rows if r.scope == "route" and r.action_family == "proteinmpnn_redesign"]
    assert len(mp_rows) == 1
    row = mp_rows[0]
    assert row.root_family == "complexa_beam"
    assert row.action_family == "proteinmpnn_redesign"
    assert "complexa_beam" in row.strategy_key
    assert "->proteinmpnn_redesign" in row.strategy_key
    assert row.new_su == 1
    assert row.parent_strategy_key is not None
    assert "complexa_beam" in row.parent_strategy_key


def test_proteinmpnn_route_value_charges_root_parent_generator_gpu_h():
    from trex.evidence_reducer import (
        ReducerConfig,
        extract_recipes,
        build_route_values,
        build_strategy_feedback,
        reduce_evidence,
    )

    def with_gpu(record, gpu_h):
        return ResultRecord(**{**record.__dict__, "gpu_h": gpu_h})

    cx = with_gpu(_gen("cx1", "complexa_beam"), 2.0)
    mp = with_gpu(_gen("mp1", "proteinmpnn_redesign"), 0.20)
    rf = with_gpu(_refilter("rf1", "mp1", su_bin="cA"), 0.05)
    spawning = {
        "cx1": _configured_action(
            "cand_cx", "complexa_beam", op="complexa_beam",
            config={"beam_width": 4},
        ),
        "mp1": _configured_action(
            "cand_mp", "proteinmpnn_redesign", op="proteinmpnn_redesign",
            config={"temperature": 0.1}, parent="cx1",
        ),
        "rf1": _configured_action(
            "chain_mp", "structure_refilter", op="af2_consensus",
            parent="mp1", role=CANONICAL_SCORE_CONVERSION,
        ),
    }

    rows = build_route_values(
        [cx, mp, rf], [rf], spawning,
        cfg=StateClassifierConfig(route_gpu_recent_window_h=0.01),
    )
    route = next(
        r for r in rows
        if r.scope == "route" and r.action_family == "proteinmpnn_redesign"
    )
    expected = 2.0 + 0.20 + 0.05
    assert route.root_family == "complexa_beam"
    assert route.new_su == 1
    assert abs(route.route_gpu_h - expected) < 1e-9
    assert abs(route.generator_gpu_h - 2.20) < 1e-9
    assert abs(route.canonical_refilter_gpu_h - 0.05) < 1e-9
    assert abs(route.record_recent_route_gpu_h - expected) < 1e-9
    assert abs(route.gpu_recent_route_gpu_h - expected) < 1e-9
    assert abs((route.gpu_recent_new_su_per_route_gpu_h or 0.0) - (1.0 / expected)) < 1e-9

    mh = method_health([cx, mp, rf], spawning_actions=spawning)["proteinmpnn_redesign"]
    assert mh.chained_strict_yield_su == 1
    assert abs((mh.chained_su_per_gpu_h or 0.0) - (1.0 / expected)) < 1e-9

    ev = reduce_evidence(
        tick_id="t1", target_id="t", target_class="c", elapsed_wall_h=1.0,
        remaining_wall_h=47.0, pending_children=0, worker_gpu_h_total=expected,
        all_results=[cx, mp, rf], window_results=[rf],
        run_su_count=1, run_su_count_delta=1, duplicate_fraction=None,
        near_miss_count=0, top_bin_share=None, panel_ready_count=0,
        panel_ready_bins_covered=0, llm_model="test", spawning_actions=spawning,
    )
    recent = ev.method_health["proteinmpnn_redesign"].chained_su_per_gpu_h_recent
    assert abs((recent or 0.0) - (1.0 / expected)) < 1e-9

    recipes = extract_recipes(
        [cx, mp, rf], spawning_action=spawning, cfg=ReducerConfig(),
    )
    recipe = next(
        r for r in recipes
        if r.method_family == "proteinmpnn_redesign" and r.recipe_class == "strict_success"
    )
    assert abs((recipe.su_per_gpu_h or 0.0) - (1.0 / expected)) < 1e-9

    feedback = build_strategy_feedback([cx, mp, rf], spawning, ReducerConfig())
    strategy = next(r for r in feedback if r["family"] == "proteinmpnn_redesign")
    assert strategy["root_family"] == "complexa_beam"
    assert strategy["feedback_scope"] == "exact_route_operator_config"
    assert "complexa_beam" in strategy["strategy_key"]
    assert "->proteinmpnn_redesign" in strategy["strategy_key"]
    assert strategy["strict_su"] == 1
    assert abs(float(strategy["gpu_h"]) - expected) < 1e-9
    assert strategy["su_per_gpu_h"] is None


def test_route_values_label_refilter_route_roles_for_prompt_clarity():
    from dataclasses import replace
    from trex.evidence_reducer import build_route_values

    cx = _gen("cx1", "complexa_beam")
    cx_af2 = replace(
        _refilter("cx_af2", "cx1", su_bin="cx_su"),
        bins={
            "refilter_source": "cx1",
            "foldseek_su": "cx_su",
            "refilter_role": CANONICAL_SCORE_CONVERSION,
        },
    )
    manual_af2 = replace(
        _refilter("manual_af2", "cx1", su_bin="manual_su"),
        bins={
            "refilter_source": "cx1",
            "foldseek_su": "manual_su",
            "refilter_role": PARENT_MODEL_REFOLD,
        },
    )
    spawning = {
        "cx1": _configured_action(
            "cand_cx", "complexa_beam", op="complexa_beam",
            config={"beam_width": 4},
        ),
        "cx_af2": _configured_action(
            "chain_cx", "structure_refilter", op="af2_consensus",
            parent="cx1", role=CANONICAL_SCORE_CONVERSION,
        ),
        "manual_af2": _configured_action(
            "cand_manual_af2", "structure_refilter", op="af2_consensus",
            config={"num_recycles": 6}, parent="cx1", role=PARENT_MODEL_REFOLD,
        ),
    }

    rows = build_route_values(
        [cx, cx_af2, manual_af2],
        [cx_af2, manual_af2],
        spawning,
    )
    route_rows = [r for r in rows if r.scope == "route"]
    cx_row = next(r for r in route_rows if r.action_family == "complexa_beam")
    manual_row = next(
        r for r in route_rows
        if r.action_family == "structure_refilter"
        and r.refilter_role == PARENT_MODEL_REFOLD
    )

    assert cx_row.route_role == "generator_with_af2_score_conversion"
    assert cx_row.refilter_role == CANONICAL_SCORE_CONVERSION
    assert manual_row.route_role == "af2_parent_model_refold"
    assert manual_row.status == "advisory"


def test_route_values_mark_promising_unscored_diagnostic_route_as_awaiting_score_conversion(tmp_path):
    from trex.evidence_reducer import build_route_values

    pdb = tmp_path / "bc_unscored.pdb"
    pdb.write_text("ATOM\n")
    bc = ResultRecord(
        result_id="bc_unscored", parent_ids=["cand_bc"], target_id="t",
        backend_family="bindcraft", runtime_bucket_id="rb",
        metrics={"bindcraft_native_pLDDT": 92.0, "bindcraft_native_iPAE": 0.18, "bindcraft_native_binder_RMSD": 0.9},
        metrics_calibrated={}, route_lineage=[], gpu_h=4.0, exit_status="ok",
        artifacts={"pdb_path": str(pdb)}, bins={},
    )
    spawning = {
        "bc_unscored": _configured_action(
            "cand_bc", "bindcraft", op="bindcraft", config={"soft_iterations": 50}
        ),
    }

    rows = build_route_values(
        [bc], [bc], spawning,
        cfg=StateClassifierConfig(route_zero_su_defer_gpu_h=0.5),
    )
    route = next(r for r in rows if r.scope == "route" and r.action_family == "bindcraft")
    family = next(r for r in rows if r.scope == "family" and r.family == "bindcraft")
    assert route.new_su == 0
    assert route.pending_score_conversion_count == 1
    assert route.pending_promising_score_conversion_count == 1
    assert route.status == "awaiting_score_conversion"
    assert route.marginal_status == "awaiting_score_conversion"
    assert family.pending_score_conversion_count == 1
    assert family.pending_promising_score_conversion_count == 1
    assert family.status == "awaiting_score_conversion"


def test_route_values_do_not_protect_low_value_bulk_score_conversion_backlog(tmp_path):
    from trex.evidence_reducer import build_route_values

    cif = tmp_path / "rank01_design_01.cif"
    cif.write_text("data_design\n")
    bg = ResultRecord(
        result_id="bg_low_proxy", parent_ids=["cand_bg"], target_id="t",
        backend_family="boltzgen", runtime_bucket_id="rb",
        metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=1.0,
        exit_status="ok", artifacts={"cif_path": str(cif)},
        bins={
            "boltzgen_design_id": "design_01",
            "boltzgen_final_rank": "1",
            "boltzgen_design_iptm": "0.20",
            "boltzgen_design_to_target_iptm": "0.25",
            "boltzgen_design_iiptm": "0.25",
            "boltzgen_min_design_to_target_pae": "20.0",
        },
    )
    spawning = {
        "bg_low_proxy": _configured_action(
            "cand_bg", "boltzgen", op="boltzgen", config={"num_designs": 16}
        ),
    }

    rows = build_route_values(
        [bg], [bg], spawning,
        cfg=StateClassifierConfig(route_zero_su_defer_gpu_h=0.5),
    )
    route = next(r for r in rows if r.scope == "route" and r.action_family == "boltzgen")
    family = next(r for r in rows if r.scope == "family" and r.family == "boltzgen")
    assert route.pending_score_conversion_count == 1
    assert route.pending_promising_score_conversion_count == 0
    assert route.status == "defer"
    assert route.marginal_status == "dry_low_quality"
    assert family.pending_score_conversion_count == 1
    assert family.pending_promising_score_conversion_count == 0
    assert family.status == "defer"


def test_route_values_keep_structure_refilter_as_plumbing_zero_su():
    from trex.evidence_reducer import build_route_values

    bg = _gen("bg1", "boltzgen")
    rf = _refilter("rf1", "bg1", su_bin="cA")
    spawning = {
        "bg1": _configured_action("cand_bg", "boltzgen", op="boltzgen", config={"num_diffusion_samples": 8}),
    }

    rows = build_route_values([bg, rf], [rf], spawning)
    sf = next(r for r in rows if r.scope == "family" and r.family == "structure_refilter")
    bg_route = next(r for r in rows if r.scope == "route" and r.action_family == "boltzgen")
    assert sf.new_su == 0
    assert sf.status == "plumbing"
    assert bg_route.new_su == 1
    assert bg_route.canonical_refilter_gpu_h == rf.gpu_h


def test_route_values_mark_productive_duplicate_route_as_diversify():
    gen = _gen("cx1", "complexa_fk_steering")
    spawning = {"cx1": _configured_action("cand_cx", "complexa_fk_steering", op="fk", config={})}
    refilters = [
        _refilter(f"rf{i}", "cx1", su_bin="same_cluster", parents=[f"chain{i}", "cx1"])
        for i in range(5)
    ]
    for i, rf in enumerate(refilters):
        spawning[rf.result_id] = _configured_action(
            f"chain{i}", "structure_refilter", op="af2_consensus",
            parent="cx1", role=CANONICAL_SCORE_CONVERSION,
        )
    rows = build_route_values([gen, *refilters], refilters, spawning)
    route = next(r for r in rows if r.scope == "route" and r.action_family == "complexa_fk_steering" and r.new_su == 1)
    assert route.strict_count == 5
    assert route.strict_per_su == 5.0
    assert route.status == "diversify"


def test_route_values_track_gpu_hour_recent_suffix_separately_from_record_window():
    bc = _gen("bc1", "bindcraft")

    def with_gpu(record, gpu_h):
        return ResultRecord(**{**record.__dict__, "gpu_h": gpu_h})

    rf_old = with_gpu(_refilter("rf_old", "bc1", su_bin="old"), 1.0)
    rf_mid = with_gpu(_refilter("rf_mid", "bc1", su_bin="mid"), 1.0)
    rf_recent_a = with_gpu(_refilter("rf_recent_a", "bc1", su_bin="recent_a"), 0.05)
    rf_recent_b = with_gpu(_refilter("rf_recent_b", "bc1", su_bin="recent_b"), 0.05)
    recs = [bc, rf_old, rf_mid, rf_recent_a, rf_recent_b]

    rows = build_route_values(
        recs,
        window_results=[rf_recent_b],
        spawning_actions={},
        cfg=StateClassifierConfig(route_gpu_recent_window_h=0.09),
    )
    route = next(r for r in rows if r.scope == "route" and r.action_family == "bindcraft")

    assert route.new_su == 4
    assert route.new_su_recent == 1
    assert route.new_su_recent_gpu == 2
    # The recent windows contain only score-conversion refilters, but the route
    # also paid the diagnostic generator once. Recent route value must include
    # that parent GPU-h instead of reporting a refilter-only 20 SU/GPU-h.
    assert abs(route.recent_route_gpu_h - 0.15) < 1e-9
    assert abs(route.gpu_recent_route_gpu_h - 0.20) < 1e-9
    assert abs(route.recent_new_su_per_route_gpu_h - (1.0 / 0.15)) < 1e-9
    assert abs(route.gpu_recent_new_su_per_route_gpu_h - 10.0) < 1e-9


def test_route_values_charge_generator_when_recent_refilters_only_pd_l1_like():
    bc = ResultRecord(
        result_id="bc_parent", parent_ids=[], target_id="t", backend_family="bindcraft",
        runtime_bucket_id="rb", metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.638, exit_status="ok", bins={},
    )
    rf1 = ResultRecord(
        result_id="rf_a", parent_ids=["chain_a", "bc_parent"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.0},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.012, exit_status="ok",
        bins={"refilter_source": "bc_parent", "foldseek_su": "su_a"},
    )
    rf2 = ResultRecord(
        result_id="rf_b", parent_ids=["chain_b", "bc_parent"], target_id="t",
        backend_family="structure_refilter", runtime_bucket_id="rb",
        metrics={"pLDDT": 93.0, "iPAE": 0.19, "binder_scRMSD": 0.9},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.012, exit_status="ok",
        bins={"refilter_source": "bc_parent", "foldseek_su": "su_b"},
    )

    rows = build_route_values(
        [bc, rf1, rf2],
        window_results=[rf1, rf2],
        spawning_actions={},
        cfg=StateClassifierConfig(route_gpu_recent_window_h=0.02),
    )
    route = next(r for r in rows if r.scope == "route" and r.action_family == "bindcraft")

    expected_gpu = 0.638 + 0.012 + 0.012
    assert route.new_su_recent == 2
    assert abs(route.recent_route_gpu_h - expected_gpu) < 1e-9
    assert abs(route.recent_new_su_per_route_gpu_h - (2.0 / expected_gpu)) < 1e-9
    assert route.recent_new_su_per_route_gpu_h < 4.0


def test_method_health_recent_chained_rate_charges_parent_generator_when_window_is_refilter_only():
    from trex.evidence_reducer import reduce_evidence

    bc = ResultRecord(
        result_id="bc_parent", parent_ids=[], target_id="t", backend_family="bindcraft",
        runtime_bucket_id="rb", metrics={}, metrics_calibrated={}, route_lineage=[],
        gpu_h=0.638, exit_status="ok", bins={},
    )
    rf1 = _refilter("rf_a", "bc_parent", su_bin="su_a")
    rf2 = _refilter("rf_b", "bc_parent", su_bin="su_b")
    rf1 = ResultRecord(**{**rf1.__dict__, "gpu_h": 0.012})
    rf2 = ResultRecord(**{**rf2.__dict__, "gpu_h": 0.012})

    spawning = {
        "rf_a": _configured_action(
            "chain_a", "structure_refilter", parent="bc_parent",
            role=CANONICAL_SCORE_CONVERSION,
        ),
        "rf_b": _configured_action(
            "chain_b", "structure_refilter", parent="bc_parent",
            role=CANONICAL_SCORE_CONVERSION,
        ),
    }
    ev = reduce_evidence(
        tick_id="t1", target_id="t", target_class="c", elapsed_wall_h=1.0,
        remaining_wall_h=47.0, pending_children=0, worker_gpu_h_total=0.662,
        all_results=[bc, rf1, rf2], window_results=[rf1, rf2],
        run_su_count=2, run_su_count_delta=2, duplicate_fraction=None,
        near_miss_count=0, top_bin_share=None, panel_ready_count=0,
        panel_ready_bins_covered=0, llm_model="test", spawning_actions=spawning,
    )

    mh = ev.method_health["bindcraft"]
    expected_gpu = 0.638 + 0.012 + 0.012
    assert mh.chained_strict_yield_su_recent == 2
    assert abs((mh.chained_su_per_gpu_h_recent or 0.0) - (2.0 / expected_gpu)) < 1e-9
    assert (mh.chained_su_per_gpu_h_recent or 0.0) < 4.0


def test_route_values_retention_prefers_medium_recent_over_stale_lifetime():
    stale = ResultRecord(**{
        **_gen("cx_stale", "complexa_beam", strict=True).__dict__,
        "gpu_h": 0.02,
        "bins": {"foldseek_su": "stale_su"},
    })
    bc = _gen("bc1", "bindcraft")

    def with_gpu(record, gpu_h):
        return ResultRecord(**{**record.__dict__, "gpu_h": gpu_h})

    refilters = [
        with_gpu(_refilter(f"rf_med_{i}", "bc1", su_bin="med_su"), 0.10)
        for i in range(10)
    ]
    tail = with_gpu(_gen("bc_tail", "bindcraft"), 0.20)
    rows = build_route_values(
        [stale, bc, *refilters, tail],
        window_results=[tail],
        spawning_actions={},
        cfg=StateClassifierConfig(
            route_gpu_recent_window_h=0.15,
            route_gpu_medium_window_h=1.20,
            deep_stall_gpu_h=12.0,
        ),
        gpu_h_since_last_su=13.0,
    )
    route_rows = [r for r in rows if r.scope == "route"]

    assert route_rows[0].action_family == "bindcraft"
    assert route_rows[0].medium_recent_new_su == 1
    assert route_rows[1].action_family == "complexa_beam"
    assert route_rows[1].new_su_per_route_gpu_h > route_rows[0].new_su_per_route_gpu_h


def test_route_values_medium_gpu_window_protects_delayed_productive_route():
    bc = _gen("bc1", "bindcraft")

    def with_gpu(record, gpu_h):
        return ResultRecord(**{**record.__dict__, "gpu_h": gpu_h})

    # Ten strict records in one new Foldseek cluster landed just outside the
    # short 3-GPU-h-style suffix, while a tiny non-strict tail is the only short
    # evidence. This should read as delayed productive, not dry duplicate.
    refilters = [
        with_gpu(_refilter(f"rf_med_{i}", "bc1", su_bin="med"), 0.10)
        for i in range(10)
    ]
    tail = with_gpu(_gen("bc_tail", "bindcraft"), 0.20)
    rows = build_route_values(
        [bc, *refilters, tail],
        window_results=[tail],
        spawning_actions={},
        cfg=StateClassifierConfig(
            route_gpu_recent_window_h=0.15,
            route_gpu_medium_window_h=1.20,
        ),
    )
    route = next(r for r in rows if r.scope == "route" and r.action_family == "bindcraft")

    assert route.new_su == 1
    assert route.new_su_recent_gpu == 0
    assert route.medium_recent_new_su == 1
    assert abs(route.medium_recent_route_gpu_h - 1.30) < 1e-9
    assert route.marginal_status == "delayed_productive_duplicate"
    assert route.status != "collapse_risk"
