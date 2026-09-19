"""R1/R2/R4 regression tests: clustering-cache exactness + near-miss dedup key.

Stable + fast: NO real Foldseek/mmseqs (monkeypatched fake clusterer). The real
~5k-structure timing check lives in a manual perf script, not CI.
"""

from __future__ import annotations

import trex.clustering_cache as cc
from trex.clustering_cache import (
    cluster_archive_pdbs_cached,
    clear_clustering_caches,
)
from trex.foldseek_clusterer import ClusteringResult, _scored_pdb_path
from trex.schemas import ResultRecord


def _rec(rid, *, exit_status="ok", metrics=None, pdb=None):
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t", backend_family="complexa_beam",
        runtime_bucket_id="rb",
        metrics=({"pLDDT": 95.0, "iPAE": 0.15, "binder_scRMSD": 0.8} if metrics is None else metrics),
        metrics_calibrated={}, route_lineage=[], gpu_h=0.1, exit_status=exit_status,
        artifacts=({"pdb_path": str(pdb)} if pdb else {}), panel_ready=False,
    )


def _mk_pdb(p):
    p.write_text("ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    return p


# --- R4b(i): the SHARED scored-record predicate (guards SU exactness vs drift) ---
def test_scored_pdb_path_predicate(tmp_path):
    pdb = _mk_pdb(tmp_path / "a.pdb")
    assert _scored_pdb_path(_rec("ok", pdb=pdb)) is not None                       # valid scored
    assert _scored_pdb_path(_rec("bad_exit", exit_status="timeout", pdb=pdb)) is None
    assert _scored_pdb_path(_rec("no_axis", metrics={"ipTM": 0.5}, pdb=pdb)) is None
    assert _scored_pdb_path(_rec("no_pdb", pdb=None)) is None


# --- R4b(iii): cache behavior — identical input → hit (no recompute); a changed
# artifact fingerprint → miss; non-cacheable status → not cached. ---
def test_foldseek_cache_hit_miss_behavior(tmp_path, monkeypatch):
    clear_clustering_caches()
    pdb = _mk_pdb(tmp_path / "x.pdb")
    recs = [_rec("r1", pdb=pdb)]

    calls = {"n": 0}

    def fake_cluster(results, **kw):
        calls["n"] += 1
        rids = {r.result_id for r in results
                if (kw.get("only_result_ids") is None or r.result_id in kw["only_result_ids"])}
        return ClusteringResult(
            cluster_by_result_id={rid: "c0" for rid in rids},
            n_structures=len(rids), n_clusters=1, status="ok",
            structure_scope="binder_chain",
        )

    monkeypatch.setattr(cc, "cluster_archive_pdbs", fake_cluster)

    cluster_archive_pdbs_cached(recs, target_id="t")
    cluster_archive_pdbs_cached(recs, target_id="t")
    assert calls["n"] == 1, "identical input must hit cache (no 2nd real cluster call)"

    # change the artifact fingerprint (content+size) → cache MISS
    pdb.write_text(pdb.read_text() + "ATOM      2  CA  GLY B   2       1.0 1.0 1.0  1.00 0.00     C\n")
    cluster_archive_pdbs_cached(recs, target_id="t")
    assert calls["n"] == 2, "changed PDB fingerprint must recompute"


def test_failed_status_not_cached(tmp_path, monkeypatch):
    clear_clustering_caches()
    pdb = _mk_pdb(tmp_path / "y.pdb")
    recs = [_rec("r1", pdb=pdb)]
    calls = {"n": 0}

    def fake_no_binary(results, **kw):
        calls["n"] += 1
        return ClusteringResult(
            cluster_by_result_id={}, n_structures=0, n_clusters=0,
            status="no_binary", structure_scope="binder_chain",
        )

    monkeypatch.setattr(cc, "cluster_archive_pdbs", fake_no_binary)
    cluster_archive_pdbs_cached(recs, target_id="t")
    cluster_archive_pdbs_cached(recs, target_id="t")
    assert calls["n"] == 2, "no_binary (transient failure) must NOT be cached → retried"


# --- R2: strategy_feedback near-miss dedup must use foldseek_near_miss ---
def test_strategy_feedback_near_miss_dedups_on_foldseek_near_miss():
    from trex.evidence_reducer import ReducerConfig, build_strategy_feedback
    from trex.tests.test_recipe_extraction import _ac

    # Two near-misses (binder_scRMSD just over 1.5, within near-pass margin),
    # same (family,operator,config), DIFFERENT foldseek_near_miss bins but the
    # SAME stale whole-archive foldseek bin. Correct dedup (foldseek_near_miss)
    # → 2 distinct near-misses; the old (foldseek) key would collapse to 1.
    def nm(rid, nm_bin):
        return ResultRecord(
            result_id=rid, parent_ids=[], target_id="t", backend_family="complexa_beam",
            runtime_bucket_id="rb",
            metrics={"pLDDT": 95.0, "iPAE": 0.18, "binder_scRMSD": 1.7},
            metrics_calibrated={}, route_lineage=[], gpu_h=0.2, exit_status="ok",
            bins={"foldseek_near_miss": nm_bin, "foldseek": "STALE_SAME"}, panel_ready=False,
        )

    recs = [nm("n1", "NM_a"), nm("n2", "NM_b")]
    spawning = {
        "n1": _ac("c1", family="complexa_beam", op="complexa_beam_default", config={"beam_width": 8}),
        "n2": _ac("c2", family="complexa_beam", op="complexa_beam_default", config={"beam_width": 8}),
    }
    sf = build_strategy_feedback(recs, spawning, ReducerConfig())
    rows = [e for e in sf if e["family"] == "complexa_beam" and e.get("near_miss_count", 0) > 0]
    assert rows, f"expected a near-miss strategy row, got {sf}"
    assert rows[0]["near_miss_count"] == 2, (
        "near-miss dedup must use foldseek_near_miss (2 distinct), not the shared "
        f"whole-archive foldseek bin (would be 1): {rows[0]}"
    )
