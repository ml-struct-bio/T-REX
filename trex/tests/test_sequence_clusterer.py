from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from trex.schemas import ResultRecord
from trex.sequence_clusterer import (
    _binder_sequence_from_pdb,
    apply_sequence_clusters_to_bins,
    cluster_archive_sequences,
)
from trex.success_criteria import STRICT_SUCCESS


_PASS_METRICS = {
    "pLDDT": STRICT_SUCCESS["pLDDT"][0] + 2.0,
    "iPAE": STRICT_SUCCESS["iPAE"][0] - 0.05,
    "binder_scRMSD": STRICT_SUCCESS["binder_scRMSD"][0] - 0.3,
}


def _make_complex_pdb(path: Path, *, chain_b_res: tuple[str, ...] = ("GLY", "SER")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C",
        "TER",
    ]
    serial = 2
    for i, res in enumerate(chain_b_res, start=1):
        lines.append(
            f"ATOM  {serial:5d}  CA  {res} B{i:4d}       {float(i):6.3f}   1.000   0.000"
            "  1.00  0.00           C"
        )
        serial += 1
    lines.extend(["TER", "END"])
    path.write_text("\n".join(lines) + "\n")


def _result(rid: str, pdb: Path, *, target: str = "t1") -> ResultRecord:
    return ResultRecord(
        result_id=rid,
        parent_ids=[],
        target_id=target,
        backend_family="complexa",
        runtime_bucket_id="rb_v7",
        metrics=dict(_PASS_METRICS),
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=0.1,
        exit_status="ok",
        bins={},
        artifacts={"pdb_path": str(pdb)},
        panel_ready=False,
    )


def test_binder_sequence_from_pdb_prefers_chain_b(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    _make_complex_pdb(pdb, chain_b_res=("GLY", "SER", "TYR"))

    seq, chain = _binder_sequence_from_pdb(pdb, preferred_chain="B")

    assert chain == "B"
    assert seq == "GSY"


def test_sequence_cluster_no_binary_is_visible_degrade(tmp_path: Path):
    pdb1 = tmp_path / "r1.pdb"
    pdb2 = tmp_path / "r2.pdb"
    _make_complex_pdb(pdb1, chain_b_res=("GLY", "SER"))
    _make_complex_pdb(pdb2, chain_b_res=("GLY", "TYR"))
    results = [_result("r1", pdb1), _result("r2", pdb2)]

    with patch("trex.sequence_clusterer.shutil.which", return_value=None):
        cl = cluster_archive_sequences(results, target_id="t1")

    assert cl.status == "no_binary"
    assert cl.n_sequences == 2
    assert cl.cluster_by_result_id == {}


def test_sequence_cluster_parses_mmseqs_easy_cluster_output(tmp_path: Path):
    pdb1 = tmp_path / "r1.pdb"
    pdb2 = tmp_path / "r2.pdb"
    _make_complex_pdb(pdb1, chain_b_res=("GLY", "SER"))
    _make_complex_pdb(pdb2, chain_b_res=("GLY", "SER"))
    results = [_result("r1", pdb1), _result("r2", pdb2)]

    def _fake_run(cmd, **_kwargs):
        out_prefix = Path(cmd[3])
        (out_prefix.parent / f"{out_prefix.name}_cluster.tsv").write_text(
            "s_00000\ts_00000\ns_00000\ts_00001\n"
        )
        return MagicMock(returncode=0, stderr="", stdout="")

    with patch("trex.sequence_clusterer.shutil.which", return_value="/fake/mmseqs"), \
         patch("trex.sequence_clusterer.subprocess.run", side_effect=_fake_run):
        cl = cluster_archive_sequences(results, target_id="t1")

    assert cl.status == "ok"
    assert cl.n_sequences == 2
    assert cl.n_clusters == 1
    assert cl.cluster_by_result_id == {
        "r1": "seq90:r1",
        "r2": "seq90:r1",
    }
    assert apply_sequence_clusters_to_bins(results, cl.cluster_by_result_id) == 2
    assert results[0].bins["sequence_su"] == "seq90:r1"
    assert results[1].bins["sequence_su"] == "seq90:r1"
