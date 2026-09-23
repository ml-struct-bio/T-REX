"""Output chain identities survive input relabelling across the live pipeline."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from trex.af2_chain_identity import ChainIdentityError
from trex.archive import Archive
from trex.controller import _resolve_pdb_chains
from trex.foldseek_clusterer import (
    ClusteringResult,
    _record_binder_chain,
    _write_chain_only_pdb,
)
from trex.live_tick import LiveTickConfig, run_live_tick
from trex.output_identity import (
    MAP_KEY,
    annotate_output_identity,
    archive_target_reference,
    chain_sequences,
    prepare_archive_results,
    resolve_output_chains,
)
from trex.schemas import ResultRecord, TargetConstraint
from trex.sequence_clusterer import _sequence_for_result
from trex.tick.config import FoldseekConfig


def _pdb(path, chains):
    lines = []
    serial = 0
    for chain, residues in chains.items():
        for i, residue in enumerate(residues, 1):
            serial += 1
            lines.append(
                f"ATOM  {serial:5d}  CA  {residue} {chain}{i:4d}    "
                f"{float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 90.00           C"
            )
    path.write_text("\n".join(lines) + "\nEND\n")
    return path


def _record(path, rid="r1", family="complexa_beam"):
    return ResultRecord(
        result_id=rid,
        parent_ids=[],
        target_id="test",
        backend_family=family,
        runtime_bucket_id="rb",
        metrics={"pLDDT": 95.0, "iPAE": 0.1, "binder_scRMSD": 0.5},
        metrics_calibrated={},
        route_lineage=[family],
        gpu_h=0.25,
        exit_status="ok",
        artifacts={"pdb_path": str(path), "target_chains": "B"},
    )


@pytest.mark.parametrize("input_chain", ["A", "B", "C", "X"])
@pytest.mark.parametrize(
    "family",
    [
        "complexa_beam",
        "complexa_best_of_n",
        "complexa_fk_steering",
        "complexa_mcts",
        "bindcraft",
        "boltzgen",
        "proteinmpnn_redesign",
    ],
)
def test_reference_sequence_identifies_relabelled_equal_length_chains(
    tmp_path, input_chain, family
):
    target = _pdb(tmp_path / "target.pdb", {input_chain: ["ALA", "GLY"]})
    output = _pdb(tmp_path / "output.pdb", {"Q": ["SER", "THR"], "Y": ["ALA", "GLY"]})
    raw = _record(output, family=family)
    result = annotate_output_identity(
        raw, target_pdb=target, target_chains=[input_chain]
    )
    assert _record_binder_chain(result, input_chain) == "Q"
    assert result.artifacts["target_chains"] == "Y"
    assert _sequence_for_result(result) == ("ST", "Q")
    assert result.metrics == raw.metrics and result.gpu_h == raw.gpu_h
    assert result.result_id == raw.result_id and raw.artifacts["target_chains"] == "B"
    assert _resolve_pdb_chains(
        output, target_pdb=target, target_chain_ids=[input_chain]
    ) == ("Y", "Q")


def test_multimer_target_and_ambiguous_binder(tmp_path):
    target = _pdb(tmp_path / "target.pdb", {c: ["ALA", "GLY"] for c in "ABC"})
    output = _pdb(
        tmp_path / "output.pdb",
        {**{c: ["ALA", "GLY"] for c in "XYZ"}, "D": ["SER", "THR"]},
    )
    assert resolve_output_chains(output, target, "ABC")["binder_chain"] == "D"
    _pdb(output, {c: ["ALA", "GLY"] for c in "XYZD"})
    with pytest.raises(ChainIdentityError, match="unique binder"):
        resolve_output_chains(output, target, "ABC")


def test_missing_complexa_metadata_cannot_choose_input_target_fallback(tmp_path):
    output = _pdb(tmp_path / "output.pdb", {"A": ["ALA"], "B": ["SER"]})
    assert _record_binder_chain(_record(output), "A") == ""
    assert _write_chain_only_pdb(
        output, tmp_path / "binder.pdb", preferred_chain="Q"
    ) == (False, None)


def test_portable_map_checks_output_and_prevents_stale_sequence_use(tmp_path):
    target = _pdb(tmp_path / "target.pdb", {"B": ["ALA", "GLY"]})
    output = _pdb(tmp_path / "output.pdb", {"A": ["ALA", "GLY"], "B": ["SER", "THR"]})
    result = annotate_output_identity(
        _record(output), target_pdb=target, target_chains=["B"]
    )
    target.unlink()
    result.artifacts["sequence"] = "AG"
    assert _sequence_for_result(result) == ("ST", "B")
    assert annotate_output_identity(result).bins["output_chain_identity"] == "verified"
    _pdb(output, {"A": ["ALA", "GLY"], "B": ["SER", "GLY"]})
    assert _record_binder_chain(result) == ""
    assert _sequence_for_result(result) == (None, None)


def _provenance(archive, target, config):
    path = archive.root / "run_provenance.json"
    path.write_text(
        json.dumps(
            {
                "target": {
                    "pdb_path": str(target),
                    "pdb_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "config_path": str(config),
                    "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                }
            }
        )
    )


def test_archive_reference_cache_rechecks_changed_assets(tmp_path):
    archive = Archive(tmp_path / "archive")
    target = _pdb(tmp_path / "target.pdb", {"B": ["ALA", "GLY"]})
    config = tmp_path / "target.json"
    config.write_text(json.dumps({"chain_ids": ["B"]}))
    _provenance(archive, target, config)
    assert archive_target_reference(archive.root) == (target, ("B",))
    _pdb(target, {"B": ["ALA", "SER"]})
    with pytest.raises(ChainIdentityError, match="hash mismatch"):
        archive_target_reference(archive.root)


def test_real_live_tick_recovers_legacy_roles_before_clustering(tmp_path, monkeypatch):
    """Regression: the actual evidence caller must not cluster two copied targets."""
    archive = Archive(tmp_path / "archive")
    target = _pdb(tmp_path / "target.pdb", {"B": ["ALA", "GLY"]})
    config = tmp_path / "target.json"
    config.write_text(json.dumps({"chain_ids": ["B"]}))
    _provenance(archive, target, config)
    for rid, binder in [("r1", ["SER", "THR"]), ("r2", ["VAL", "LEU"])]:
        output = _pdb(tmp_path / (rid + ".pdb"), {"A": ["ALA", "GLY"], "B": binder})
        archive.append(_record(output, rid))
    before = (archive.root / "result_records.jsonl").read_bytes()
    selected = []

    def cluster(rows, **kwargs):
        groups = {}
        for row in rows:
            if row.result_id not in kwargs["only_result_ids"]:
                continue
            chain = _record_binder_chain(row, kwargs["binder_chain_id"])
            selected.append(chain)
            assert chain == "B"
            sequence = chain_sequences(Path(row.artifacts["pdb_path"]))[chain]
            groups[row.result_id] = "foldseek:" + "-".join(sequence)
        return ClusteringResult(
            groups,
            len(groups),
            len(set(groups.values())),
            "ok",
            structure_scope="binder_chain",
        )

    monkeypatch.setattr("trex.clustering_cache.cluster_archive_pdbs_cached", cluster)
    result = run_live_tick(
        archive,
        TargetConstraint("test", "protein", chain_ids=["B"]),
        tick_id="tick1",
        tick_id_int=1,
        elapsed_wall_h=1.0,
        remaining_wall_h=47.0,
        cfg=LiveTickConfig(foldseek=FoldseekConfig(binary="fixture")),
        evidence_only=True,
    )
    assert selected and result["evidence"]["run_su_count"] == 2
    assert (archive.root / "result_records.jsonl").read_bytes() == before
    assert all(
        MAP_KEY in row.artifacts
        for row in prepare_archive_results(
            archive.iter_records(ResultRecord), archive.root
        )
    )


def test_ingestion_records_output_roles_and_retains_input_provenance(tmp_path):
    from trex.execution.result_processing import (
        OutputParserRegistry,
        WorkerOutputRequest,
        process_worker_output,
    )
    from trex.tests.test_result_processing import _candidate

    target = _pdb(tmp_path / "target.pdb", {"B": ["ALA", "GLY"]})
    output = _pdb(tmp_path / "output.pdb", {"A": ["ALA", "GLY"], "B": ["SER", "THR"]})
    raw = _record(output)
    processed = process_worker_output(
        WorkerOutputRequest(
            candidate=_candidate("complexa_beam"),
            requested_output_directory=tmp_path,
            target=TargetConstraint("test", "protein", chain_ids=["B"]),
            tick_id="t1",
            target_pdb_path=str(target),
            target_chains_csv="B",
            elapsed_gpu_hours=0.5,
        ),
        parser_registry=OutputParserRegistry(complexa=lambda *_: [raw]),
    )
    record = processed.result_records[0]
    assert record.artifacts["input_target_chains"] == "B"
    assert record.artifacts["target_chains"] == "A"
    assert record.artifacts["binder_chain"] == "B"
    assert record.bins["output_chain_identity"] == "verified"
    assert record.result_id == raw.result_id and record.metrics == raw.metrics
    assert record.gpu_h == 0.5


def test_unresolved_identity_clears_stale_credit_and_cannot_enter_export(tmp_path):
    from trex.export_best_n import export_best_n
    from trex.panel import ProductionPanelConfig, _hard_gate_failure

    archive = Archive(tmp_path / "archive")
    target = _pdb(tmp_path / "target.pdb", {"B": ["ALA", "GLY"]})
    output = _pdb(tmp_path / "output.pdb", {"A": ["ALA", "GLY"], "B": ["ALA", "GLY"]})
    record = replace(
        _record(output), bins={"foldseek_su": "stale", "sequence_su": "stale"}
    )
    record = annotate_output_identity(record, target_pdb=target, target_chains=["B"])
    assert record.bins["output_chain_identity"] == "unresolved"
    assert "foldseek_su" not in record.bins and "sequence_su" not in record.bins
    assert _record_binder_chain(record) == ""
    assert (
        _hard_gate_failure(record, ProductionPanelConfig())
        == "unresolved_output_chain_identity"
    )
    archive.append(record)
    result = export_best_n(
        archive,
        target_id="test",
        n=1,
        out_dir=tmp_path / "export",
        annotate_dedup=False,
    )
    assert result["n_exported"] == 0


@pytest.mark.parametrize("formatting", ["standard", "blank_lines", "no_hash_separators"])
def test_mmcif_output_roles_and_sequence(tmp_path, formatting):
    from Bio.PDB import MMCIFIO, PDBParser

    target = _pdb(tmp_path / "target.pdb", {"C": ["ALA", "GLY"]})
    output = _pdb(tmp_path / "output.pdb", {"A": ["ALA", "GLY"], "D": ["SER", "THR"]})
    writer = MMCIFIO()
    writer.set_structure(PDBParser(QUIET=True).get_structure("output", output))
    cif = tmp_path / "output.cif"
    writer.save(str(cif))
    if formatting == "blank_lines":
        cif.write_text(cif.read_text().replace("ATOM ", "\nATOM "))
    elif formatting == "no_hash_separators":
        cif.write_text(cif.read_text().replace("#\n", "\n") + "_audit.creation_method fixture\n")
    record = annotate_output_identity(
        _record(cif), target_pdb=target, target_chains=["C"]
    )
    assert _record_binder_chain(record) == "D"
    assert _sequence_for_result(record) == ("ST", "D")


def test_live_tick_does_not_trust_partial_chain_identity(tmp_path, monkeypatch):
    import trex.evidence_reducer as reducer

    archive = Archive(tmp_path / "archive")
    target = _pdb(tmp_path / "target.pdb", {"B": ["ALA", "GLY"]})
    config = tmp_path / "target.json"
    config.write_text(json.dumps({"chain_ids": ["B"]}))
    _provenance(archive, target, config)
    valid = _pdb(tmp_path / "valid.pdb", {"A": ["ALA", "GLY"], "B": ["SER", "THR"]})
    ambiguous = _pdb(
        tmp_path / "ambiguous.pdb", {"A": ["ALA", "GLY"], "B": ["ALA", "GLY"]}
    )
    archive.append(_record(valid, "valid"))
    archive.append(
        replace(_record(ambiguous, "ambiguous"), bins={"foldseek_su": "stale"})
    )
    trust = []
    classify = reducer.classify_state

    def observe(**kwargs):
        trust.append(kwargs["su_dedup_trusted"])
        return classify(**kwargs)

    monkeypatch.setattr(reducer, "classify_state", observe)
    # The one resolved binder takes the real clusterer's singleton shortcut.
    # The ambiguous second structure must be rejected before that shortcut.
    result = run_live_tick(
        archive,
        TargetConstraint("test", "protein", chain_ids=["B"]),
        tick_id="tick1",
        tick_id_int=1,
        elapsed_wall_h=1.0,
        remaining_wall_h=47.0,
        cfg=LiveTickConfig(foldseek=FoldseekConfig(binary="true")),
        evidence_only=True,
    )
    evidence = result["evidence"]
    assert evidence["run_su_count"] == 1
    assert trust == [False]
