from __future__ import annotations

from pathlib import Path

from trex.tests.fixtures.trajectory_analysis import (
    _path_aliases,
    _select_probe_manifest,
    _summarize_descendant_scores,
)


def test_trajectory_probe_selection_is_stratified_per_target():
    rows = [
        {"target_key": "t1", "worker_dir": "w1", "trajectory_stem": "a", "selection_bucket": "no_final_descendant"},
        {"target_key": "t1", "worker_dir": "w1", "trajectory_stem": "b", "selection_bucket": "accepted_descendant"},
        {"target_key": "t1", "worker_dir": "w1", "trajectory_stem": "c", "selection_bucket": "rejected_only_descendant"},
        {"target_key": "t1", "worker_dir": "w1", "trajectory_stem": "d", "selection_bucket": "no_final_descendant"},
        {"target_key": "t2", "worker_dir": "w1", "trajectory_stem": "e", "selection_bucket": "accepted_descendant"},
    ]

    selected = _select_probe_manifest(rows, max_per_target=2)

    assert [(r["target_key"], r["trajectory_stem"]) for r in selected] == [
        ("t1", "a"),
        ("t1", "b"),
        ("t2", "e"),
    ]


def test_descendant_summary_counts_strict_but_not_unpersisted_su(tmp_path: Path):
    pdb = tmp_path / "design.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1      0.0   0.0   0.0  1.00 10.00           C\n")
    parent = {"result_id": "bc1", "backend_family": "bindcraft", "artifacts": {"pdb_path": str(pdb)}}
    child = {
        "result_id": "af2_1",
        "backend_family": "structure_refilter",
        "exit_status": "ok",
        "bins": {"refilter_source": "bc1", "refilter_source_family": "bindcraft"},
        "metrics": {"pLDDT": 95.0, "iPAE": 0.2, "binder_scRMSD": 0.9},
    }

    summary = _summarize_descendant_scores(
        [{"status": "accepted", "pdb_path": str(pdb), "stem": "design_mpnn1_model1"}],
        bc_by_path={str(pdb): parent},
        refilter_by_source={"bc1": [child]},
    )

    assert summary["descendant_score_count"] == 1
    assert summary["descendant_strict_count"] == 1
    assert summary["descendant_su_count"] == 0


def test_path_aliases_normalize_tigress_projects_mounts():
    aliases = _path_aliases("/tigress/project/user/trex/x.pdb")

    assert "/tigress/project/user/trex/x.pdb" in aliases
    assert "/projects/project/user/trex/x.pdb" in aliases
