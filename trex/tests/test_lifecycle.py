"""Unit tests for lifecycle arithmetic (pure functions, deterministic)."""

from __future__ import annotations

import pytest

from trex.lifecycle import (
    LifecycleConfig,
    contradicts_descendant,
    contradicts_hyp,
    deficit,
    default_axis_thresholds,
    rdr,
    supports_descendant,
    supports_hyp,
    update_hypothesis,
    violates_preserve,
)
from trex.schemas import (
    HypothesisCard,
    PredictedChange,
    PreserveConstraint,
    ResultRecord,
)


def _r(rid: str, *, pLDDT: float | None = None, iPAE: float | None = None, scRMSD: float | None = None, panel_ready: bool = False, family: str = "complexa_beam") -> ResultRecord:
    metrics: dict[str, float] = {}
    if pLDDT is not None:
        metrics["pLDDT"] = pLDDT
    if iPAE is not None:
        metrics["iPAE"] = iPAE
    if scRMSD is not None:
        metrics["binder_scRMSD"] = scRMSD
    return ResultRecord(
        result_id=rid,
        parent_ids=[],
        target_id="t",
        backend_family=family,
        runtime_bucket_id="rb1",
        metrics=metrics,
        metrics_calibrated=dict(metrics),
        route_lineage=[],
        gpu_h=1.0,
        exit_status="ok",
        panel_ready=panel_ready,
    )


# ---- deficit + rdr -------------------------------------------------------


def test_deficit_higher_is_better():
    # pLDDT threshold 80, value 70 → deficit 10
    assert deficit(70.0, 80.0, "increase") == 10.0
    assert deficit(85.0, 80.0, "increase") == 0.0


def test_deficit_lower_is_better():
    # iPAE threshold 0.4, value 0.6 → deficit 0.2
    assert deficit(0.6, 0.4, "decrease") == pytest.approx(0.2)
    assert deficit(0.3, 0.4, "decrease") == 0.0


def test_rdr_returns_none_when_baseline_passing():
    assert rdr(0.0, 0.0) is None
    assert rdr(0.005, 0.005, eps=0.01) is None


def test_rdr_relative_improvement():
    # baseline deficit 10, descendant deficit 7 → RDR 0.3
    assert rdr(10.0, 7.0) == pytest.approx(0.3)


# ---- supports / contradicts ----------------------------------------------


def test_supports_descendant_pLDDT():
    baseline = _r("b", pLDDT=70.0)  # deficit 10
    descendant = _r("d", pLDDT=78.0)  # deficit 2, RDR = 0.8
    pc = PredictedChange(axis="pLDDT", direction="increase", baseline_refs=["b"], min_relative_deficit_reduction=0.10, min_absolute_delta=None)
    ok, why = supports_descendant(descendant, baseline, pc, default_axis_thresholds(), LifecycleConfig())
    assert ok, why


def test_supports_descendant_iPAE():
    baseline = _r("b", iPAE=0.7)  # deficit 0.3
    descendant = _r("d", iPAE=0.5)  # deficit 0.1, RDR ≈ 0.67
    pc = PredictedChange(axis="iPAE", direction="decrease", baseline_refs=["b"], min_relative_deficit_reduction=0.20, min_absolute_delta=None)
    ok, _ = supports_descendant(descendant, baseline, pc, default_axis_thresholds(), LifecycleConfig())
    assert ok


def test_wrong_direction_can_neither_support_nor_contradict():
    baseline = _r("b", iPAE=0.7)
    descendant = _r("d", iPAE=0.5)
    pc = PredictedChange(
        axis="iPAE", direction="increase", baseline_refs=["b"],
        min_relative_deficit_reduction=0.20, min_absolute_delta=None,
    )

    ok, why = supports_descendant(
        descendant, baseline, pc, default_axis_thresholds(), LifecycleConfig()
    )

    assert not ok
    assert why == "direction_mismatch:increase!=decrease"
    assert not contradicts_descendant(
        descendant, baseline, pc, default_axis_thresholds(), LifecycleConfig()
    )


def test_supports_descendant_baseline_already_passing_uses_abs_delta():
    # Complexa threshold pLDDT >= 90 → baseline at 92 is passing (deficit 0)
    baseline = _r("b", pLDDT=92.0)  # deficit 0
    descendant = _r("d", pLDDT=95.0)
    # baseline passing → RDR is None → falls through to absolute delta
    pc = PredictedChange(axis="pLDDT", direction="increase", baseline_refs=["b"], min_relative_deficit_reduction=0.10, min_absolute_delta=2.0)
    ok, _ = supports_descendant(descendant, baseline, pc, default_axis_thresholds(), LifecycleConfig())
    assert ok
    # below absolute delta should fail
    pc2 = PredictedChange(axis="pLDDT", direction="increase", baseline_refs=["b"], min_relative_deficit_reduction=0.10, min_absolute_delta=5.0)
    ok2, _ = supports_descendant(descendant, baseline, pc2, default_axis_thresholds(), LifecycleConfig())
    assert not ok2


def test_contradicts_descendant_when_rdr_small():
    baseline = _r("b", iPAE=0.7)
    descendant = _r("d", iPAE=0.69)  # RDR ≈ 0.033, below 0.05
    pc = PredictedChange(axis="iPAE", direction="decrease", baseline_refs=["b"], min_relative_deficit_reduction=0.20, min_absolute_delta=None)
    assert contradicts_descendant(descendant, baseline, pc, default_axis_thresholds(), LifecycleConfig())


def test_contradicts_descendant_false_when_improvement_meaningful():
    baseline = _r("b", iPAE=0.7)
    descendant = _r("d", iPAE=0.5)  # RDR ≈ 0.67
    pc = PredictedChange(axis="iPAE", direction="decrease", baseline_refs=["b"], min_relative_deficit_reduction=0.20, min_absolute_delta=None)
    assert not contradicts_descendant(descendant, baseline, pc, default_axis_thresholds(), LifecycleConfig())


# ---- preserve constraint -------------------------------------------------


def test_violates_preserve_passing_to_failing():
    baseline = _r("b", pLDDT=85.0, iPAE=0.7)
    descendant = _r("d", pLDDT=75.0, iPAE=0.5)  # pLDDT was passing, now failing
    hyp = HypothesisCard(
        hypothesis_id="h1",
        target_id="t",
        tick_created=0,
        claim="...",
        mode_affinity={"exploit": 0.0, "rescue": 1.0, "explore": 0.0},
        evidence_refs=["b"],
        predicted_metric_changes=[PredictedChange("iPAE", "decrease", ["b"], 0.2, None)],
        preserve_constraints=[PreserveConstraint("pLDDT", max_relative_deficit_increase=0.05)],
        recommended_action_families=["proteinmpnn_redesign"],
    )
    assert violates_preserve(descendant, baseline, hyp, default_axis_thresholds(), LifecycleConfig())




def test_chained_strict_refilter_supports_metric_missing_generator_baseline():
    baseline = _r("bc_parent", family="bindcraft")
    descendant = _r(
        "af2_child", pLDDT=94.0, iPAE=0.12, scRMSD=0.8,
        family="structure_refilter",
    )
    pc = PredictedChange("iPAE", "decrease", ["bc_parent"], 0.20, None)

    ok, why = supports_descendant(
        descendant, baseline, pc, default_axis_thresholds(), LifecycleConfig()
    )

    assert ok
    assert why == ""

# ---- update_hypothesis lifecycle -----------------------------------------


def _hyp(hid: str = "h1") -> HypothesisCard:
    return HypothesisCard(
        hypothesis_id=hid,
        target_id="t",
        tick_created=10,
        claim="iPAE improves under MPNN redesign",
        mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
        evidence_refs=["b"],
        predicted_metric_changes=[
            PredictedChange("iPAE", "decrease", ["b"], 0.20, None),
        ],
        preserve_constraints=[
            PreserveConstraint("pLDDT", max_relative_deficit_increase=0.05),
        ],
        recommended_action_families=["proteinmpnn_redesign"],
    )


def test_update_to_supported_after_two_healthy_descendants():
    # Under Complexa thresholds (pLDDT>=90, iPAE<=0.226): keep pLDDT passing
    # in baseline + descendants so preserve_constraint on pLDDT holds.
    baseline = _r("b", pLDDT=92.0, iPAE=0.7)
    d1 = _r("d1", pLDDT=92.0, iPAE=0.5)   # iPAE RDR (0.474->0.274)=0.42 >= 0.20; pLDDT preserved
    d2 = _r("d2", pLDDT=91.0, iPAE=0.45)  # iPAE RDR (0.474->0.224)=0.53 >= 0.20; pLDDT preserved
    new = update_hypothesis(
        _hyp(),
        healthy_descendants=[(d1, baseline), (d2, baseline)],
        current_tick=11,
        axis_thresholds=default_axis_thresholds(),
    )
    assert new.status == "supported"
    assert new.support_points >= 2
    assert new.last_evaluated_tick == 11


def test_update_to_contradicted_after_three_failing_descendants():
    baseline = _r("b", iPAE=0.7)
    bad1 = _r("bad1", pLDDT=82.0, iPAE=0.695)  # tiny RDR → contradicts
    bad2 = _r("bad2", pLDDT=82.0, iPAE=0.69)
    bad3 = _r("bad3", pLDDT=82.0, iPAE=0.691)
    new = update_hypothesis(
        _hyp(),
        healthy_descendants=[(bad1, baseline), (bad2, baseline), (bad3, baseline)],
        current_tick=11,
        axis_thresholds=default_axis_thresholds(),
    )
    assert new.status == "contradicted"
    assert new.last_evaluated_tick == 11


def test_ttl_retires_without_evidence():
    baseline = _r("b", iPAE=0.7)
    new = update_hypothesis(
        _hyp(),
        healthy_descendants=[],
        current_tick=21,  # 11 ticks after creation (tick_created=10), TTL=10 (Q30)
        axis_thresholds=default_axis_thresholds(),
    )
    assert new.status == "retired"


def test_panel_ready_bonus_can_promote_to_supported():
    baseline = _r("b", iPAE=0.7)
    d1 = _r("d1", pLDDT=85.0, iPAE=0.5, panel_ready=True)
    # One scientific support + one panel-ready bonus reaches the threshold.
    new = update_hypothesis(
        _hyp(),
        healthy_descendants=[(d1, baseline)],
        current_tick=11,
        axis_thresholds=default_axis_thresholds(),
    )
    assert new.status == "supported"


def test_panel_ready_bonus_requires_scientific_support():
    baseline = _r("b", pLDDT=92.0, iPAE=0.7)
    d1 = _r("d1", pLDDT=92.0, iPAE=0.69, panel_ready=True)
    d2 = _r("d2", pLDDT=92.0, iPAE=0.68, panel_ready=True)

    new = update_hypothesis(
        _hyp(),
        healthy_descendants=[(d1, baseline), (d2, baseline)],
        current_tick=11,
        axis_thresholds=default_axis_thresholds(),
    )

    assert new.status == "active"
    assert new.support_points == 0


def test_supported_is_terminal():
    baseline = _r("b", iPAE=0.7)
    hyp = _hyp().__class__(  # construct a supported hyp
        hypothesis_id="h1",
        target_id="t",
        tick_created=10,
        claim="...",
        mode_affinity={"exploit": 0.1, "rescue": 0.8, "explore": 0.1},
        evidence_refs=["b"],
        predicted_metric_changes=_hyp().predicted_metric_changes,
        preserve_constraints=_hyp().preserve_constraints,
        recommended_action_families=["proteinmpnn_redesign"],
        ttl_ticks=3,
        status="supported",
        support_points=3.0,
        contradiction_points=0,
        descendants_evaluated=2,
    )
    # Future TTL pressure should not move it
    new = update_hypothesis(
        hyp,
        healthy_descendants=[(_r("bad", iPAE=0.695), baseline)],
        current_tick=99,
        axis_thresholds=default_axis_thresholds(),
    )
    assert new.status == "supported"
