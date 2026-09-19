"""Evidence quality: duplicate/near-miss/top-bin/panel-bins/SU-dedup
must be derived from real data instead of left as None/0.

Feedback issue #1+2: live_tick previously passed None/0 placeholders
to reduce_evidence, causing state classifier to misclassify productive
vs stalled targets on real campaigns.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from trex.archive import Archive
from trex.foldseek_clusterer import ClusteringResult
from trex.live_tick import LiveTickConfig, run_live_tick
from trex.schemas import (
    PanelSelection, PlannerOutput, ResultRecord, SupervisorOutput, TargetConstraint,
)


def _r(
    rid, *, fb="FS_A", pLDDT=92.0, iPAE=0.20, scRMSD=1.3,
    panel_ready=False, exit_status="ok", target="t1",
    extra_bins: dict[str, str] | None = None,
) -> ResultRecord:
    # Defaults match Complexa strict_success (pLDDT>=90, iPAE<=0.226, scRMSD<1.5)
    m = {"pLDDT": pLDDT, "iPAE": iPAE, "binder_scRMSD": scRMSD}
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id=target, backend_family="complexa_beam",
        runtime_bucket_id="rb1", metrics=m, metrics_calibrated=dict(m),
        route_lineage=[], gpu_h=0.4, exit_status=exit_status,  # type: ignore[arg-type]
        # foldseek = whole-archive bin (duplicate_fraction); foldseek_su =
        # strict-only bin (SU). Production sets both on strict records; here
        # they're equal since the test pre-clusters directly.
        bins={"foldseek": fb, "foldseek_su": fb, **(extra_bins or {})},
        panel_ready=panel_ready,
    )


def _stub_planner():
    return PlannerOutput(
        valid=True, abstain=False, confidence=0.7, fail_reason=None,
        cards=[], rationale="x",
        raw_text='{"cards": []}', usage={},
    )


def _stub_sup():
    return SupervisorOutput(
        valid=True, abstain=False, confidence=0.7, fail_reason=None,
        mode_mixture={"exploit": 0.5, "rescue": 0.4, "explore": 0.1},
        candidate_decisions=[], rationale="y", raw_text='{}', usage={},
    )


def test_duplicate_fraction_computed_from_foldseek_bins(tmp_path: Path):
    """5 results, 3 in same bin → high duplicate_fraction."""
    arc = Archive(tmp_path / "a")
    for i in range(3):
        arc.append(_r(f"r{i}", fb="FS_A"))
    arc.append(_r("r3", fb="FS_B"))
    arc.append(_r("r4", fb="FS_C"))
    target = TargetConstraint(target_id="t1", target_class="c")
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        summary = run_live_tick(
            arc, target, tick_id="t1", tick_id_int=1,
            elapsed_wall_h=0, remaining_wall_h=48,
            cfg=LiveTickConfig(),
        )
    # Verify by re-reading the persisted EvidenceSummary
    from trex.schemas import EvidenceSummary
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    # duplicate_fraction is 1 - unique_bins/scored = 1 - 3/5 = 2/5.
    # top_bin_share below is the single-bin collapse signal.
    assert ev.duplicate_fraction == 2 / 5
    # top_bin_share: FS_A 3/5
    assert ev.top_bin_share == 3 / 5


def test_near_miss_count_computed(tmp_path: Path):
    """near-miss = exactly 1 strict-axis fails. Complexa thresholds:
    pLDDT>=90 (near margin 5), iPAE<=0.226 (near margin 0.05), scRMSD<1.5 (near margin 0.3).
    'fail' label requires deficit > margin."""
    arc = Archive(tmp_path / "a2")
    # review #5: near_miss_count is now DISTINCT structural basins (foldseek bin),
    # so give the two near-misses distinct bins to count as 2.
    arc.append(_r("r1", iPAE=0.45, fb="NM_A"))   # iPAE fail (deficit 0.22 > margin 0.05)
    arc.append(_r("r2", iPAE=0.50, fb="NM_B"))   # iPAE fail (deficit 0.27 > margin 0.05)
    arc.append(_r("r3", pLDDT=92, iPAE=0.20, scRMSD=1.3))  # strict pass (all axes)
    arc.append(_r("r4", pLDDT=65, iPAE=0.80, scRMSD=1.3))  # pLDDT + iPAE both fail
    target = TargetConstraint(target_id="t1", target_class="c")
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(arc, target, tick_id="t1", tick_id_int=1,
                      elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
    from trex.schemas import EvidenceSummary
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    # r1 and r2 are exactly 1-axis (iPAE) fail in DISTINCT basins → 2
    # r3 is strict pass → not near_miss; r4 is 2-axis fail → joint_fail
    assert ev.near_miss_count == 2


def test_near_miss_count_structurally_deduped(tmp_path: Path):
    """review #5 (2026-05-31): two near-misses in the SAME structural basin
    (same foldseek bin) count as ONE — so a single basin re-discovered cannot
    inflate near_miss_count → rescue_rich."""
    arc = Archive(tmp_path / "a2dedup")
    arc.append(_r("r1", iPAE=0.45, fb="SAME"))
    arc.append(_r("r2", iPAE=0.50, fb="SAME"))   # same basin
    arc.append(_r("r3", iPAE=0.47, fb="SAME"))   # same basin
    target = TargetConstraint(target_id="t1", target_class="c")
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(arc, target, tick_id="t1", tick_id_int=1,
                      elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
    from trex.schemas import EvidenceSummary
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.near_miss_count == 1, "3 near-misses in one basin = 1 distinct near-miss"


def test_panel_ready_bins_covered(tmp_path: Path):
    arc = Archive(tmp_path / "a3")
    # Need to also have results with metrics so reducer doesn't NaN
    for i in range(2):
        arc.append(_r(f"x{i}", fb=f"FS_X{i}"))
    arc.append(_r("p1", fb="FS_a", panel_ready=True, extra_bins={"sequence_su": "SEQ_a"}))
    arc.append(_r("p2", fb="FS_b", panel_ready=True))
    arc.append(_r("p3", fb="FS_a", panel_ready=True))  # same FS_a as p1
    target = TargetConstraint(target_id="t1", target_class="c")
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(arc, target, tick_id="t1", tick_id_int=1,
                      elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
    from trex.schemas import EvidenceSummary
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    # FS_a (1 unique) + FS_b (1 unique) → 2 distinct bins across panel_ready
    assert ev.panel_ready_bins_covered == 2


def test_su_dedup_uses_foldseek_bins(tmp_path: Path):
    """5 strict-pass results, 3 in same FS bin → SU=3 (3 distinct bins),
    not naive 5."""
    arc = Archive(tmp_path / "a4")
    arc.append(_r("s1", fb="FS_A"))  # strict
    arc.append(_r("s2", fb="FS_A"))  # strict, same bin
    arc.append(_r("s3", fb="FS_A"))  # strict, same bin
    arc.append(_r("s4", fb="FS_B"))  # strict, new bin
    arc.append(_r("s5", fb="FS_C"))  # strict, new bin
    target = TargetConstraint(target_id="t1", target_class="c")
    with patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        summary = run_live_tick(
            arc, target, tick_id="t1", tick_id_int=1,
            elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig(),
        )
    # run_su_count_delta (the window count) = 3 (FS_A + FS_B + FS_C, not naive 5):
    # SU dedup correctly uses the foldseek bins.
    from trex.schemas import EvidenceSummary
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    assert ev.run_su_count_delta == 3
    # State: the live re-cluster finds no PDBs here -> foldseek_su_status != "ok"
    # -> su_dedup_trusted=False. Review fix #5 (2026-06-13): an untrusted dedup is
    # treated as a plateau (stalled) so a collapsed run can escape, instead of the
    # old masquerade where an inflated/untrusted delta fell through to low_evidence.
    assert summary["evidence"]["state_label"] == "stalled"


def test_live_tick_writes_production_panel_snapshot(tmp_path: Path):
    arc = Archive(tmp_path / "panel_arc")
    pdb = tmp_path / "binder.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    arc.append(_r("p1", fb="FS_A", extra_bins={"sequence_su": "S_A"}))
    arc.append(_r("p2", fb="FS_B", extra_bins={"sequence_su": "S_B"}))
    # Re-append with structure artifacts because _r keeps artifact-less records
    # for older metric tests.
    recs = list(arc.iter_records(ResultRecord))
    arc = Archive(tmp_path / "panel_arc2")
    from dataclasses import replace
    for r in recs:
        arc.append(replace(r, artifacts={"pdb_path": str(pdb)}))
    target = TargetConstraint(target_id="t1", target_class="c", panel_size_K=2)
    def trusted_cluster(results, *, only_result_ids=None, **_kwargs):
        ids = sorted(only_result_ids or [r.result_id for r in results])
        bins = {rid: f"trusted_{rid}" for rid in ids}
        return ClusteringResult(
            cluster_by_result_id=bins,
            n_structures=len(ids),
            n_clusters=len(bins),
            status="ok",
            structure_scope="binder_chain",
        )

    with patch(
             "trex.clustering_cache.cluster_archive_pdbs_cached",
             side_effect=trusted_cluster,
         ), patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(arc, target, tick_id="t1", tick_id_int=1,
                      elapsed_wall_h=0, remaining_wall_h=48, cfg=LiveTickConfig())
    from trex.schemas import EvidenceSummary
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    panels = list(arc.iter_records(PanelSelection))
    assert ev.production_panel_status == "ok"
    assert set(ev.production_panel_selected_ids) == {"p1", "p2"}
    assert panels and panels[-1].selected_ids == ev.production_panel_selected_ids


def test_live_tick_does_not_publish_panel_with_untrusted_structure_dedup(tmp_path: Path):
    arc = Archive(tmp_path / "panel_untrusted")
    pdb = tmp_path / "binder_untrusted.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    from dataclasses import replace
    arc.append(replace(
        _r("p1", fb="legacy_bin", extra_bins={"sequence_su": "S_A"}),
        artifacts={"pdb_path": str(pdb)},
    ))
    untrusted = ClusteringResult(
        cluster_by_result_id={}, n_structures=1, n_clusters=0,
        status="no_binary", structure_scope="binder_chain",
    )

    with patch(
             "trex.clustering_cache.cluster_archive_pdbs_cached",
             return_value=untrusted,
         ), patch("trex.live_tick.call_planner", return_value=_stub_planner()), \
         patch("trex.live_tick.call_supervisor", return_value=_stub_sup()):
        run_live_tick(
            arc, TargetConstraint(target_id="t1", target_class="c"),
            tick_id="t1", tick_id_int=1, elapsed_wall_h=0,
            remaining_wall_h=48, cfg=LiveTickConfig(),
        )

    from trex.schemas import EvidenceSummary
    ev = list(arc.iter_records(EvidenceSummary))[-1]
    panel = list(arc.iter_records(PanelSelection))[-1]
    assert ev.production_panel_status == "degraded_untrusted_structure_dedup"
    assert ev.production_panel_selected_ids == []
    assert panel.selected_ids == []
    assert panel.hard_gate_failures[0].startswith("untrusted_structure_dedup:")
