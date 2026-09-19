"""Unit tests for `foldseek_clusterer` — the 2026-05-27 Foldseek wire-up
that closes the plan §-0.5 SU dedup design-debt gap.

The actual `foldseek easy-cluster` binary is not assumed installed in
CI. Tests cover:
  - graceful degradation when binary missing
  - filtering (target_id mismatch, exit_status, missing PDB)
  - single-structure short-circuit
  - cluster TSV parsing (mocked binary)
  - apply_clusters_to_bins mutation behavior
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from trex.foldseek_clusterer import (
    ClusteringResult,
    _parse_cluster_tsv,
    _pdb_for_result,
    _resolve_stem,
    _safe_stem,
    _write_chain_only_pdb,
    apply_clusters_to_bins,
    cluster_archive_pdbs,
)
from trex.schemas import ResultRecord
from trex.success_criteria import STRICT_SUCCESS


def test_controller_preflight_requires_foldseek_for_official_su():
    from trex.controller import _require_foldseek_available

    with patch("trex.controller.shutil.which", return_value=None):
        with pytest.raises(SystemExit, match="Official SU requires Foldseek"):
            _require_foldseek_available("foldseek")


def test_controller_preflight_returns_resolved_foldseek_path():
    from trex.controller import _require_foldseek_available

    with patch("trex.controller.shutil.which", return_value="/fake/foldseek"):
        assert _require_foldseek_available("foldseek") == "/fake/foldseek"


def test_resolve_stem_handles_multimodel_and_chain_suffixes():
    """2026-05-31 fairness audit: foldseek appends chain AND multi-MODEL tokens
    to input stems (af2_refilter saves all AF2 models). The mapping must strip
    ALL trailing _<token> segments back to the result_id, else most strict
    records get NO foldseek_su bin. Historically that split the fallback
    dedup keyspace and over-counted SU; with official no-fallback it would lose
    valid SU credit."""
    n2r = {"abc123def456": "abc123def456"}   # result_id = hex (no underscore)
    assert _resolve_stem("abc123def456", n2r) == "abc123def456"          # exact
    assert _resolve_stem("abc123def456_A", n2r) == "abc123def456"        # chain suffix
    assert _resolve_stem("abc123def456_MODEL_3_A", n2r) == "abc123def456"  # multi-MODEL + chain
    assert _resolve_stem("abc123def456_MODEL_1_B", n2r) == "abc123def456"
    assert _resolve_stem("unknown_stem_xyz", n2r) is None                # no match


# Strict-success thresholds + a tiny margin so the fixture clearly passes.
# Foldseek clusterer never reads `metrics`, so the only purpose here is to
# (a) satisfy ResultRecord's required field, (b) avoid hardcoded magic
# numbers that diverge from the §2.5 SSOT.
_PASS_METRICS = {
    "pLDDT":         STRICT_SUCCESS["pLDDT"][0] + 2.0,         # 92.0
    "iPAE":          STRICT_SUCCESS["iPAE"][0] - 0.05,         # ~0.176
    "binder_scRMSD": STRICT_SUCCESS["binder_scRMSD"][0] - 0.3,  # 1.2
}


def _make_pdb(path: Path, chain: str = "B") -> None:
    """Write a minimal valid PDB with a single CA atom on `chain`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"ATOM      1  CA  ALA {chain}   1       0.000   0.000   0.000"
        "  1.00  0.00           C\n"
    )


def _make_two_chain_pdb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
        "  1.00  0.00           C\n"
        "TER\n"
        "ATOM      2  CA  GLY B   1       5.000   0.000   0.000"
        "  1.00  0.00           C\n"
        "ATOM      3  CA  SER B   2       6.000   0.000   0.000"
        "  1.00  0.00           C\n"
        "TER\n"
        "END\n"
    )


def _result(
    rid: str, *, target: str = "t1", exit_status: str = "ok",
    pdb_path: str | None = None, pdb_dir: str | None = None,
    bins: dict[str, str] | None = None,
) -> ResultRecord:
    artifacts = {}
    if pdb_path is not None:
        artifacts["pdb_path"] = pdb_path
    if pdb_dir is not None:
        artifacts["pdb_dir"] = pdb_dir
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id=target,
        backend_family="bindcraft", runtime_bucket_id="rb_v7",
        metrics=dict(_PASS_METRICS),  # SSOT-derived strict-pass fixture
        metrics_calibrated={},
        route_lineage=[], gpu_h=1.0, exit_status=exit_status,  # type: ignore[arg-type]
        bins=bins if bins is not None else {"design": "d1"},
        artifacts=artifacts, panel_ready=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_safe_stem_strips_special_chars():
    assert _safe_stem("a/b c.d") == "a_b_c_d"
    assert _safe_stem("__abc__") == "abc"
    assert _safe_stem("a" * 200) == "a" * 80
    assert _safe_stem("...") == ""


def test_pdb_for_result_uses_pdb_path(tmp_path: Path):
    pdb = tmp_path / "binder.pdb"
    _make_pdb(pdb)
    r = _result("r1", pdb_path=str(pdb))
    assert _pdb_for_result(r) == pdb


def test_pdb_for_result_falls_back_to_pdb_dir(tmp_path: Path):
    d = tmp_path / "binder_dir"
    d.mkdir()
    p1 = d / "design_001.pdb"
    _make_pdb(p1)
    r = _result("r1", pdb_dir=str(d))
    out = _pdb_for_result(r)
    assert out is not None
    assert out.suffix == ".pdb"


def test_pdb_for_result_none_when_missing(tmp_path: Path):
    r = _result("r1")  # no artifacts
    assert _pdb_for_result(r) is None
    r2 = _result("r1", pdb_path=str(tmp_path / "nope.pdb"))
    assert _pdb_for_result(r2) is None  # path doesn't exist


def test_pdb_for_result_accepts_cif(tmp_path: Path):
    cif = tmp_path / "binder.cif"
    cif.write_text("# minimal cif\n")
    r = _result("r1", pdb_path=str(cif))
    assert _pdb_for_result(r) == cif


def test_write_chain_only_pdb_prefers_binder_chain_b(tmp_path: Path):
    src = tmp_path / "complex.pdb"
    dest = tmp_path / "binder_only.pdb"
    _make_two_chain_pdb(src)

    ok, chain = _write_chain_only_pdb(src, dest, preferred_chain="B")

    assert ok is True
    assert chain == "B"
    atoms = [line for line in dest.read_text().splitlines() if line.startswith("ATOM")]
    assert atoms
    assert {line[21] for line in atoms} == {"B"}


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_cluster_skips_foreign_target(tmp_path: Path):
    pdb = tmp_path / "x.pdb"
    _make_pdb(pdb)
    results = [
        _result("r_self", target="t1", pdb_path=str(pdb)),
        _result("r_foreign", target="t2", pdb_path=str(pdb)),
    ]
    with patch("trex.foldseek_clusterer.shutil.which", return_value=None):
        cl = cluster_archive_pdbs(results, target_id="t1")
    # n_structures counts only target-matched results, even when binary missing.
    # No Foldseek binary means no trusted singleton fallback.
    assert cl.status == "no_binary"
    assert cl.n_structures == 1
    assert cl.cluster_by_result_id == {}


def test_cluster_skips_nonok_exit_status(tmp_path: Path):
    pdb = tmp_path / "x.pdb"
    _make_pdb(pdb)
    results = [
        _result("r_ok", exit_status="ok", pdb_path=str(pdb)),
        _result("r_timeout", exit_status="timeout", pdb_path=str(pdb)),
        _result("r_fail", exit_status="parse_fail", pdb_path=str(pdb)),
    ]
    with patch("trex.foldseek_clusterer.shutil.which", return_value=None):
        cl = cluster_archive_pdbs(results, target_id="t1")
    assert cl.status == "no_binary"
    assert cl.n_structures == 1
    assert cl.cluster_by_result_id == {}


def test_cluster_no_structures_status():
    cl = cluster_archive_pdbs([], target_id="t1")
    assert cl.status == "no_structures"
    assert cl.n_structures == 0
    assert cl.cluster_by_result_id == {}


def test_cluster_single_structure_short_circuits_when_binary_available(tmp_path: Path):
    pdb = tmp_path / "x.pdb"
    _make_pdb(pdb)
    r = _result("r1", pdb_path=str(pdb))
    with patch("trex.foldseek_clusterer.shutil.which", return_value="/fake/foldseek"):
        cl = cluster_archive_pdbs([r], target_id="t1")
    assert cl.status == "ok"
    assert cl.n_structures == 1
    assert cl.n_clusters == 1
    assert cl.cluster_by_result_id == {"r1": "foldseek:r1"}


def test_cluster_single_structure_no_binary_is_untrusted(tmp_path: Path):
    pdb = tmp_path / "x.pdb"
    _make_pdb(pdb)
    r = _result("r1", pdb_path=str(pdb))
    with patch("trex.foldseek_clusterer.shutil.which", return_value=None):
        cl = cluster_archive_pdbs([r], target_id="t1")
    assert cl.status == "no_binary"
    assert cl.n_structures == 1
    assert cl.n_clusters == 0
    assert cl.cluster_by_result_id == {}


@pytest.mark.parametrize(
    ("suffix", "content"),
    [
        (".pdb", ""),
        (".cif", "# no parseable binder atoms\n"),
    ],
)
def test_cluster_single_structure_requires_valid_binder_scope(
    tmp_path: Path, suffix: str, content: str,
):
    """A singleton only earns a trusted SU after binder-scope preparation.

    The old pre-processing shortcut minted a Foldseek bin for an empty PDB or
    unsupported CIF even though no structure could have reached Foldseek.
    """
    structure = tmp_path / f"invalid{suffix}"
    structure.write_text(content)
    r = _result("r_invalid", pdb_path=str(structure))

    with patch("trex.foldseek_clusterer.shutil.which", return_value="/fake/foldseek"):
        cl = cluster_archive_pdbs([r], target_id="t1")

    assert cl.status == "no_structures"
    assert cl.n_structures == 1
    assert cl.n_clusters == 0
    assert cl.n_scope_fallback == 1
    assert cl.cluster_by_result_id == {}


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


def test_cluster_no_binary_returns_empty_map(tmp_path: Path):
    """Foldseek absent -> status="no_binary", empty map.

    Official SU has no per-result fallback; callers may still report strict_count,
    but these records must not mint trusted Foldseek-deduped SU.
    """
    pdbs = [tmp_path / f"r{i}.pdb" for i in range(3)]
    for p in pdbs:
        _make_pdb(p)
    results = [
        _result(f"r{i}", pdb_path=str(pdbs[i])) for i in range(3)
    ]
    with patch("trex.foldseek_clusterer.shutil.which", return_value=None):
        cl = cluster_archive_pdbs(results, target_id="t1")
    assert cl.status == "no_binary"
    assert cl.cluster_by_result_id == {}
    assert cl.n_structures == 3
    assert cl.n_clusters == 0


def test_cluster_subprocess_failure_returns_empty_map(tmp_path: Path):
    """foldseek returncode != 0 → status="failed", stderr captured."""
    pdbs = [tmp_path / f"r{i}.pdb" for i in range(2)]
    for p in pdbs:
        _make_pdb(p)
    results = [
        _result(f"r{i}", pdb_path=str(pdbs[i])) for i in range(2)
    ]
    fake = MagicMock(returncode=1, stderr="fake error", stdout="")
    with patch("trex.foldseek_clusterer.shutil.which",
                 return_value="/fake/foldseek"), \
         patch("trex.foldseek_clusterer.subprocess.run",
                 return_value=fake):
        cl = cluster_archive_pdbs(results, target_id="t1")
    assert cl.status == "failed"
    assert "fake error" in cl.stderr_tail
    assert cl.cluster_by_result_id == {}


# ---------------------------------------------------------------------------
# TSV parsing
# ---------------------------------------------------------------------------


def test_parse_cluster_tsv_simple(tmp_path: Path):
    """Two clusters: r1, r2 in cluster_A (rep=r1); r3 singleton in cluster_B."""
    cluster_file = tmp_path / "out_cluster.tsv"
    cluster_file.write_text("r1\tr1\nr1\tr2\nr3\tr3\n")
    name_map = {"r1": "rid_001", "r2": "rid_002", "r3": "rid_003"}
    out = _parse_cluster_tsv(tmp_path, name_map)
    assert out == {
        "rid_001": "foldseek:rid_001",
        "rid_002": "foldseek:rid_001",
        "rid_003": "foldseek:rid_003",
    }


def test_parse_cluster_tsv_handles_chain_suffix(tmp_path: Path):
    """Foldseek sometimes appends '_B' to member stems for chain selection."""
    cluster_file = tmp_path / "out_cluster.tsv"
    cluster_file.write_text("r1_B\tr2_B\n")
    name_map = {"r1": "rid_001", "r2": "rid_002"}
    out = _parse_cluster_tsv(tmp_path, name_map)
    assert out == {"rid_002": "foldseek:rid_001"}


def test_parse_cluster_tsv_empty_when_no_file(tmp_path: Path):
    """No *_cluster.tsv → empty map."""
    assert _parse_cluster_tsv(tmp_path, {}) == {}


# ---------------------------------------------------------------------------
# apply_clusters_to_bins
# ---------------------------------------------------------------------------


def test_apply_clusters_updates_bins(tmp_path: Path):
    r1 = _result("r1", bins={"design": "d1"})
    r2 = _result("r2", bins={"design": "d2"})
    r3 = _result("r3", bins={"design": "d3"})
    cluster_map = {"r1": "foldseek:c1", "r2": "foldseek:c1"}
    n = apply_clusters_to_bins([r1, r2, r3], cluster_map)
    assert n == 2
    assert r1.bins["foldseek"] == "foldseek:c1"
    assert r2.bins["foldseek"] == "foldseek:c1"
    assert "foldseek" not in r3.bins  # r3 not in cluster_map


def test_apply_clusters_empty_map_is_noop(tmp_path: Path):
    r1 = _result("r1", bins={"design": "d1"})
    n = apply_clusters_to_bins([r1], {})
    assert n == 0
    assert "foldseek" not in r1.bins


# ---------------------------------------------------------------------------
# NEW-001 (2026-06-18): scope-fallback must NOT inflate the SU numerator
# ---------------------------------------------------------------------------


def _fake_singleton_cluster(cmd, **_kwargs):
    """Stand-in for `foldseek easy-cluster`: writes a *_cluster.tsv that makes
    every file actually present in the input dir its own singleton cluster.
    cmd = [binary, "easy-cluster", input_dir, out_prefix, work_dir, ...]."""
    input_dir = Path(cmd[2])
    out_prefix = Path(cmd[3])
    stems = sorted(p.stem for p in input_dir.iterdir() if p.is_file())
    (out_prefix.parent / f"{out_prefix.name}_cluster.tsv").write_text(
        "".join(f"{s}\t{s}\n" for s in stems)
    )
    return MagicMock(returncode=0, stderr="", stdout="")


def test_binder_chain_extraction_failure_is_excluded_not_minted(tmp_path: Path):
    """NEW-001: a strict structure whose binder chain cannot be extracted (here a
    .cif, which `_write_chain_only_pdb` rejects) must NOT be copied as a FULL
    complex into the binder-chain easy-cluster run — that would form a bogus
    singleton and INFLATE SU while coverage still read 1.0. It is skipped (counted
    in n_scope_fallback, gets no foldseek_su bin → refilter_source fallback)."""
    good1 = tmp_path / "g1.pdb"; _make_two_chain_pdb(good1)
    good2 = tmp_path / "g2.pdb"; _make_two_chain_pdb(good2)
    bad = tmp_path / "b.cif"; bad.write_text("# minimal cif: no extractable binder chain\n")
    results = [
        _result("ridgood1", pdb_path=str(good1)),
        _result("ridgood2", pdb_path=str(good2)),
        _result("ridbadcif", pdb_path=str(bad)),
    ]
    with patch("trex.foldseek_clusterer.shutil.which",
               return_value="/fake/foldseek"), \
         patch("trex.foldseek_clusterer.subprocess.run",
               side_effect=_fake_singleton_cluster):
        cl = cluster_archive_pdbs(
            results, target_id="t1", structure_scope="binder_chain",
        )
    assert cl.status == "ok"
    assert cl.n_scope_fallback == 1
    # the extraction-failure record must be ABSENT from the cluster map (no bogus
    # singleton → no SU inflation); the two extractable binders ARE clustered.
    assert "ridbadcif" not in cl.cluster_by_result_id
    assert set(cl.cluster_by_result_id) == {"ridgood1", "ridgood2"}
    # n_structures still reports all scored inputs (coverage = 2/3 < 1.0 downstream
    # → su_dedup_trusted=False, the degraded-Foldseek contract).
    assert cl.n_structures == 3
