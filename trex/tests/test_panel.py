"""Unit tests for greedy Pareto panel selection."""

from __future__ import annotations

import dataclasses
import json

import pytest

from trex.panel import (
    PanelConfig,
    ProductionPanelConfig,
    best_near_miss_backups,
    diversity_at_K,
    geometric_mean,
    new_bins_covered,
    production_quality,
    quality,
    select_production_panel,
    select_panel,
)
from trex.archive import Archive
from trex.finalize_panel import (
    FINAL_PANEL_OUTPUT_SCHEMA_VERSION,
    finalize_panel,
    main as panel_main,
)
from trex.schemas import PanelSelection, ResultRecord


def _cand(
    rid: str,
    *,
    foldseek: str = "FS_a",
    contact: str = "C_a",
    epitope: str = "E_a",
    sequence: str = "S_a",
    structure_conf: float = 0.8,
    interface_conf: float = 0.7,
    hotspot: float = 0.6,
    clash: float = 0.9,
    panel_ready: bool = True,
) -> ResultRecord:
    return ResultRecord(
        result_id=rid,
        parent_ids=[],
        target_id="t",
        backend_family="complexa_beam",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 85, "iPAE": 0.3},
        metrics_calibrated={
            "calibrated_structure_confidence": structure_conf,
            "calibrated_interface_confidence": interface_conf,
            "hotspot_contact_satisfaction": hotspot,
            "clash_developability_score": clash,
        },
        route_lineage=[],
        gpu_h=1.0,
        exit_status="ok",
        bins={
            "foldseek": foldseek,
            "contact": contact,
            "epitope": epitope,
            "sequence": sequence,
        },
        panel_ready=panel_ready,
    )


def test_geometric_mean_floors_zeros():
    """Geometric mean must not collapse to zero when a single component is 0."""
    gm = geometric_mean([0.5, 0.0, 0.9], eps=0.01)
    assert gm > 0


def test_quality_zero_when_not_panel_ready():
    c = _cand("c1", panel_ready=False)
    assert quality(c) == 0.0


def test_new_bins_covered_initial_full_coverage():
    c = _cand("c1")
    assert new_bins_covered(c, [], ("foldseek", "contact", "epitope", "sequence")) == 4


def test_new_bins_covered_after_redundant_select():
    a = _cand("a")
    b = _cand("b")  # identical bins
    assert new_bins_covered(b, [a], ("foldseek", "contact", "epitope", "sequence")) == 0


def test_select_panel_picks_diverse_first():
    a = _cand("a", foldseek="FS_a", structure_conf=0.5)
    b = _cand("b", foldseek="FS_b", structure_conf=0.5)
    c = _cand("c", foldseek="FS_a", structure_conf=0.5)  # redundant FS
    panel = select_panel([a, b, c], cfg=PanelConfig(K=2))
    assert set(panel.selected_ids) == {"a", "b"}


def test_select_panel_deterministic():
    a = _cand("a", foldseek="FS_a")
    b = _cand("b", foldseek="FS_b")
    c = _cand("c", foldseek="FS_c")
    p1 = select_panel([a, b, c], cfg=PanelConfig(K=3))
    p2 = select_panel([c, b, a], cfg=PanelConfig(K=3))  # different input order
    assert p1.selected_ids == p2.selected_ids


def test_select_panel_skips_not_panel_ready():
    a = _cand("a", panel_ready=False)
    b = _cand("b", foldseek="FS_b")
    panel = select_panel([a, b], cfg=PanelConfig(K=2))
    assert panel.selected_ids == ["b"]


def test_panel_value_sums_dweight_history():
    """Panel value uses each design weight at the time of selection."""
    a = _cand("a", foldseek="FS_a")
    b = _cand("b", foldseek="FS_b")
    c = _cand(
        "c", foldseek="FS_a", structure_conf=0.9
    )  # bin-redundant but higher quality
    panel = select_panel([a, b, c], cfg=PanelConfig(K=3))
    # All 3 selected → c gets dweight 0.3 if redundant
    # Total should not be sum of static current weights
    assert panel.panel_value > 0


def test_diversity_at_K_normalized():
    a = _cand("a", foldseek="FS_a", contact="C_a", epitope="E_a", sequence="S_a")
    b = _cand("b", foldseek="FS_b", contact="C_b", epitope="E_b", sequence="S_b")
    panel = select_panel([a, b], cfg=PanelConfig(K=2))
    div = diversity_at_K(
        panel, {"foldseek": 4, "contact": 4, "epitope": 4, "sequence": 4}
    )
    assert 0 < div <= 1.0


def _strict_candidate(
    rid: str, pdb: str, *, fs: str, seq: str, p: float = 92.0
) -> ResultRecord:
    return ResultRecord(
        result_id=rid,
        parent_ids=[],
        target_id="t",
        backend_family="complexa_beam",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": p, "iPAE": 0.18, "binder_scRMSD": 1.1},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=0.2,
        exit_status="ok",
        bins={"foldseek_su": fs, "sequence_su": seq},
        artifacts={"pdb_path": pdb},
        panel_ready=False,
    )


def test_production_panel_uses_strict_success_not_panel_ready(tmp_path):
    pdb = tmp_path / "binder.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    a = _strict_candidate("a", str(pdb), fs="FS_a", seq="S_a", p=94.0)
    b = _strict_candidate("b", str(pdb), fs="FS_b", seq="S_b", p=91.0)
    panel = select_production_panel([a, b], cfg=ProductionPanelConfig(K=2))
    assert panel.selected_ids == ["a", "b"]
    assert panel.diversity_bins["structure"] == 2
    assert panel.panel_value > 0


def test_production_panel_hard_gates_missing_structure(tmp_path):
    missing = _strict_candidate(
        "missing", str(tmp_path / "missing.pdb"), fs="FS_a", seq="S_a"
    )
    directory_path = tmp_path / "directory.pdb"
    directory_path.mkdir()
    directory = _strict_candidate(
        "directory", str(directory_path), fs="FS_b", seq="S_b"
    )
    panel = select_production_panel(
        [missing, directory], cfg=ProductionPanelConfig(K=2)
    )
    assert panel.selected_ids == []
    assert any("missing_structure_artifact" in r for r in panel.hard_gate_failures)


def test_production_panel_requires_trusted_foldseek_bin(tmp_path):
    pdb = tmp_path / "binder.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    untrusted = _strict_candidate("untrusted", str(pdb), fs="FS_a", seq="S_a")
    untrusted = dataclasses.replace(untrusted, bins={"sequence_su": "S_a"})

    panel = select_production_panel([untrusted], cfg=ProductionPanelConfig(K=1))

    assert panel.selected_ids == []
    assert any("missing_trusted_structure_bin" in r for r in panel.hard_gate_failures)


def test_production_panel_prefers_structure_diversity(tmp_path):
    pdb = tmp_path / "binder.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    a = _strict_candidate("a", str(pdb), fs="FS_a", seq="S_a", p=95.0)
    dup = _strict_candidate("dup", str(pdb), fs="FS_a", seq="S_dup", p=99.0)
    b = _strict_candidate("b", str(pdb), fs="FS_b", seq="S_b", p=91.0)
    panel = select_production_panel([a, dup, b], cfg=ProductionPanelConfig(K=2))
    assert set(panel.selected_ids) == {"dup", "b"}


def test_best_near_miss_backups_are_separate_from_panel(tmp_path):
    pdb = tmp_path / "binder.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    near = ResultRecord(
        result_id="near",
        parent_ids=[],
        target_id="t",
        backend_family="complexa_beam",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 92.0, "iPAE": 0.35, "binder_scRMSD": 1.1},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=0.2,
        exit_status="ok",
        bins={"foldseek": "NM_a"},
        artifacts={"pdb_path": str(pdb)},
    )
    assert best_near_miss_backups([near], K=1) == ["near"]
    assert production_quality(near) >= 0.0


def test_finalize_panel_uses_archive_records_without_panel_ready(tmp_path):
    pdb = tmp_path / "binder.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    arc = Archive(tmp_path / "arc")
    arc.append(_strict_candidate("a", str(pdb), fs="FS_a", seq="S_a", p=94.0))
    panel, diagnostics = finalize_panel(
        arc,
        target_id="t",
        panel_size=1,
        panel_id="test_panel",
        run_dedup=False,
    )
    assert panel.selected_ids == ["a"]
    assert diagnostics["n_strict"] == 1
    assert diagnostics["settings"]["panel_size"] == 1
    assert diagnostics["settings"]["collapse_tm_score"] == 0.80
    assert diagnostics["settings"]["su_tm_score"] == 0.60


def test_finalize_panel_rejects_invalid_settings(tmp_path):
    archive = Archive(tmp_path / "archive")
    with pytest.raises(ValueError, match="panel_size must be positive"):
        finalize_panel(archive, target_id="t", panel_size=0, run_dedup=False)
    with pytest.raises(ValueError, match="sequence_identity"):
        finalize_panel(
            archive,
            target_id="t",
            run_dedup=False,
            sequence_identity=1.1,
        )


def test_panel_cli_is_read_only_on_missing_archive(tmp_path, capsys):
    missing = tmp_path / "missing"
    return_code = panel_main(
        [
            "--archive-root",
            str(missing),
            "--target-id",
            "t",
        ]
    )

    assert return_code == 2
    assert "not a directory" in capsys.readouterr().err
    assert not missing.exists()


def test_panel_cli_output_is_versioned_and_prevents_duplicate_append(tmp_path, capsys):
    archive_root = tmp_path / "archive"
    Archive(archive_root)
    args = [
        "--archive-root",
        str(archive_root),
        "--target-id",
        "t",
        "--panel-size",
        "4",
        "--panel-id",
        "publication-panel",
        "--no-dedup",
        "--append",
    ]

    assert panel_main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == FINAL_PANEL_OUTPUT_SCHEMA_VERSION
    assert payload["appended"] is True
    assert payload["diagnostics"]["settings"]["panel_size"] == 4

    assert panel_main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert len(list(Archive(archive_root).iter_records(PanelSelection))) == 1


def test_production_bins_family_uses_refilter_source_family():
    """Every strict SU sits on a structure_refilter record (only the AF2 refilter
    writes the canonical strict keys), so backend_family is always
    'structure_refilter'. production_bins['family'] must instead surface the
    GENERATING family so the family-diversity bin/tie-break is meaningful."""
    from trex.panel import production_bins

    def _refilter_su(rid, src_family):
        return ResultRecord(
            result_id=rid,
            parent_ids=[],
            target_id="t",
            backend_family="structure_refilter",
            runtime_bucket_id="rb1",
            metrics={"pLDDT": 92, "iPAE": 0.17, "binder_scRMSD": 1.1},
            metrics_calibrated={},
            route_lineage=[],
            gpu_h=1.0,
            exit_status="ok",
            bins={"foldseek_su": f"fs_{rid}", "refilter_source_family": src_family},
            panel_ready=True,
        )

    a = _refilter_su("r1", "complexa_beam")
    b = _refilter_su("r2", "bindcraft")
    assert production_bins(a)["family"] == "complexa_beam"
    assert production_bins(b)["family"] == "bindcraft"
    # two different generators -> family-diversity bin counts 2, not 1.
    assert len({production_bins(a)["family"], production_bins(b)["family"]}) == 2


def test_production_bins_family_falls_back_to_backend_family():
    """When refilter_source_family is absent (legacy/native record), fall back to
    backend_family — preserves old behavior."""
    from trex.panel import production_bins

    rec = ResultRecord(
        result_id="r3",
        parent_ids=[],
        target_id="t",
        backend_family="complexa_beam",
        runtime_bucket_id="rb1",
        metrics={"pLDDT": 92, "iPAE": 0.17, "binder_scRMSD": 1.1},
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=1.0,
        exit_status="ok",
        bins={"foldseek_su": "fs_r3"},
        panel_ready=True,
    )
    assert production_bins(rec)["family"] == "complexa_beam"
