from __future__ import annotations

from pathlib import Path

from trex import controller as controller
from trex.foldseek_clusterer import _record_binder_chain, _write_chain_only_pdb
from trex.output_parsers.proteinmpnn import _binder_sequence_from_mpnn
from trex.schemas import ResultRecord
from trex.sequence_clusterer import _sequence_for_result


def _write_ca_pdb(path: Path, chain_lengths: dict[str, int]) -> None:
    lines: list[str] = []
    serial = 1
    for chain, n_res in chain_lengths.items():
        for resi in range(1, n_res + 1):
            lines.append(
                f"ATOM  {serial:5d}  CA  ALA {chain}{resi:4d}    "
                f"{float(resi):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def test_resolve_pdb_chains_preserves_multichain_target(tmp_path: Path):
    parent = tmp_path / "multichain_parent.pdb"
    _write_ca_pdb(parent, {"A": 5, "B": 5, "C": 5, "D": 3})

    target_chains, binder_chain = controller._resolve_pdb_chains(
        parent,
        target_res_count=5,
        target_chain_ids=("A", "B", "C"),
    )

    assert target_chains == "A,B,C"
    assert binder_chain == "D"


def test_first_free_chain_avoids_multichain_target_chains():
    assert controller._first_free_chain_id(("A", "B", "C"), default="B") == "D"


def test_boltzgen_yaml_includes_all_multichain_target_chains(tmp_path: Path):
    target_pdb = tmp_path / "multichain_target.pdb"
    _write_ca_pdb(target_pdb, {"A": 2, "B": 2, "C": 2})

    spec = controller._write_boltzgen_yaml(
        tmp_path,
        "TEST_MULTICHAIN",
        str(target_pdb),
        hotspots=["A1", "C2"],
        chain_ids=["A", "B", "C"],
        binder_chain="D",
        length_range=(50, 120),
    )

    text = spec.read_text()
    assert "id: D" in text
    assert "id: A" in text
    assert "id: B" in text
    assert "id: C" in text
    assert 'binding: "1"' in text
    assert 'binding: "2"' in text


def test_proteinmpnn_extracts_d_chain_binder_sequence(tmp_path: Path):
    parent = tmp_path / "multichain_parent.pdb"
    _write_ca_pdb(parent, {"A": 4, "B": 4, "C": 4, "D": 3})
    seq = "AAAA" + ("X" * 20) + "BBBB" + ("X" * 20) + "CCCC" + ("X" * 20) + "DEF"

    assert _binder_sequence_from_mpnn(seq, parent, "D") == "DEF"


def _result_with_pdb(path: Path, binder_chain: str) -> ResultRecord:
    return ResultRecord(
        result_id="r1",
        parent_ids=[],
        target_id="TEST_MULTICHAIN",
        backend_family="structure_refilter",
        runtime_bucket_id="rb",
        metrics={"pLDDT": 91.0, "iPAE": 0.2, "binder_scRMSD": 1.0},
        metrics_calibrated={},
        route_lineage=["structure_refilter"],
        gpu_h=0.1,
        exit_status="ok",
        bins={"binder_chain": binder_chain, "target_chains": "A,B,C"},
        artifacts={"pdb_path": str(path), "binder_chain": binder_chain, "target_chains": "A,B,C"},
    )


def test_foldseek_chain_extraction_uses_record_binder_chain_d(tmp_path: Path):
    parent = tmp_path / "multichain_scored.pdb"
    _write_ca_pdb(parent, {"A": 4, "B": 4, "C": 4, "D": 3})
    rec = _result_with_pdb(parent, "D")
    out = tmp_path / "binder_only.pdb"

    ok, chain = _write_chain_only_pdb(
        parent, out, preferred_chain=_record_binder_chain(rec, "B")
    )

    assert ok
    assert chain == "D"
    chains = {line[21] for line in out.read_text().splitlines() if line.startswith("ATOM")}
    assert chains == {"D"}


def test_sequence_dedup_uses_record_binder_chain_d(tmp_path: Path):
    parent = tmp_path / "multichain_scored.pdb"
    _write_ca_pdb(parent, {"A": 4, "B": 4, "C": 4, "D": 3})
    rec = _result_with_pdb(parent, "D")

    seq, source = _sequence_for_result(rec, binder_chain_id="B")

    assert seq == "AAA"
    assert source == "D"
