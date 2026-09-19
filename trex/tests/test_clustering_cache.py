from __future__ import annotations

from pathlib import Path

from trex import clustering_cache as cc
from trex.foldseek_clusterer import ClusteringResult
from trex.schemas import ResultRecord
from trex.sequence_clusterer import SequenceClusteringResult
from trex.success_criteria import STRICT_SUCCESS


_PASS_METRICS = {
    "pLDDT": STRICT_SUCCESS["pLDDT"][0] + 2.0,
    "iPAE": STRICT_SUCCESS["iPAE"][0] - 0.05,
    "binder_scRMSD": STRICT_SUCCESS["binder_scRMSD"][0] - 0.3,
}


def _pdb(path: Path, x: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"ATOM      1  CA  GLY B   1       {x:6.3f}   0.000   0.000"
        "  1.00  0.00           C\nEND\n"
    )


def _result(rid: str, pdb_path: Path, *, seq: str = "GS") -> ResultRecord:
    return ResultRecord(
        result_id=rid,
        parent_ids=[],
        target_id="t1",
        backend_family="complexa_beam",
        runtime_bucket_id="rb",
        metrics=dict(_PASS_METRICS),
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=0.1,
        exit_status="ok",
        bins={},
        artifacts={"pdb_path": str(pdb_path), "binder_sequence": seq},
        panel_ready=False,
    )


def test_foldseek_cache_reuses_identical_effective_input(monkeypatch, tmp_path: Path):
    cc.clear_clustering_caches()
    p = tmp_path / "r1.pdb"
    _pdb(p, x=0.0)
    recs = [_result("r1", p)]
    calls = {"n": 0}

    def fake_cluster(*args, **kwargs):
        calls["n"] += 1
        return ClusteringResult(
            cluster_by_result_id={"r1": "foldseek:r1"},
            n_structures=1,
            n_clusters=1,
            status="ok",
        )

    monkeypatch.setattr(cc, "cluster_archive_pdbs", fake_cluster)

    first = cc.cluster_archive_pdbs_cached(recs, target_id="t1")
    second = cc.cluster_archive_pdbs_cached(recs, target_id="t1")

    assert calls["n"] == 1
    assert first.cluster_by_result_id == second.cluster_by_result_id


def test_foldseek_cache_invalidates_when_alignment_type_changes(monkeypatch, tmp_path: Path):
    cc.clear_clustering_caches()
    p = tmp_path / "r1.pdb"
    _pdb(p, x=0.0)
    recs = [_result("r1", p)]
    calls = {"n": 0}

    def fake_cluster(*args, **kwargs):
        calls["n"] += 1
        return ClusteringResult(
            cluster_by_result_id={"r1": f"foldseek:r1:{kwargs.get('alignment_type')}"},
            n_structures=1,
            n_clusters=1,
            status="ok",
        )

    monkeypatch.setattr(cc, "cluster_archive_pdbs", fake_cluster)

    cc.cluster_archive_pdbs_cached(recs, target_id="t1", alignment_type=1)
    cc.cluster_archive_pdbs_cached(recs, target_id="t1", alignment_type=1)
    cc.cluster_archive_pdbs_cached(recs, target_id="t1", alignment_type=None)

    assert calls["n"] == 2


def test_foldseek_cache_invalidates_when_artifact_changes(monkeypatch, tmp_path: Path):
    cc.clear_clustering_caches()
    p = tmp_path / "r1.pdb"
    _pdb(p, x=0.0)
    recs = [_result("r1", p)]
    calls = {"n": 0}

    def fake_cluster(*args, **kwargs):
        calls["n"] += 1
        return ClusteringResult(
            cluster_by_result_id={"r1": f"foldseek:r1:{calls['n']}"},
            n_structures=1,
            n_clusters=1,
            status="ok",
        )

    monkeypatch.setattr(cc, "cluster_archive_pdbs", fake_cluster)

    cc.cluster_archive_pdbs_cached(recs, target_id="t1")
    p.write_text(p.read_text() + "REMARK changed\n")
    second = cc.cluster_archive_pdbs_cached(recs, target_id="t1")

    assert calls["n"] == 2
    assert second.cluster_by_result_id["r1"] == "foldseek:r1:2"


def test_mmseqs_cache_reuses_identical_sequence_input(monkeypatch, tmp_path: Path):
    cc.clear_clustering_caches()
    p = tmp_path / "r1.pdb"
    _pdb(p)
    recs = [_result("r1", p, seq="GSGS")]
    calls = {"n": 0}

    def fake_cluster(*args, **kwargs):
        calls["n"] += 1
        return SequenceClusteringResult(
            cluster_by_result_id={"r1": "seq90:r1"},
            n_sequences=1,
            n_clusters=1,
            status="ok",
        )

    monkeypatch.setattr(cc, "cluster_archive_sequences", fake_cluster)

    cc.cluster_archive_sequences_cached(recs, target_id="t1")
    cc.cluster_archive_sequences_cached(recs, target_id="t1")

    assert calls["n"] == 1
