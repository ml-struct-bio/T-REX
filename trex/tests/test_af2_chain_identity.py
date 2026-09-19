"""Synthetic chain-identity regressions; no model or design backend is run."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from trex.af2_chain_identity import resolve_prediction_chains
from trex.clustering_cache import _cached_sequence_for_result
from trex.foldseek_clusterer import _record_binder_chain, _write_chain_only_pdb
from trex.output_parsers.af2_refilter import parse_af2_refilter_output
from trex.output_parsers.types import ParserContext
from trex.schemas import ResultRecord
from trex.sequence_clusterer import _sequence_for_result
from trex.success_criteria import is_strict_success


def write_pdb(path, chains):
    lines = []
    serial = 0
    for chain, residues in chains.items():
        for pos, residue in enumerate(residues, 1):
            serial += 1
            lines.append(f"ATOM  {serial:5d}  CA  {residue:3s} {chain}{pos:4d}    "
                         f"{float(pos):8.3f}{0.:8.3f}{0.:8.3f}  1.00  0.00           C")
    path.write_text("\n".join(lines) + "\nEND\n")


@pytest.fixture
def pair(tmp_path):
    source = tmp_path / "input.pdb"
    prediction = tmp_path / "af2_refilter_prediction.pdb"
    write_pdb(source, {"A": ["SER", "THR"], "B": ["ALA", "GLY", "ALA"]})
    write_pdb(prediction, {"A": ["ALA", "GLY", "ALA"], "B": ["SER", "THR"]})
    report = {
        "input_pdb": str(source), "binder_chain": "A", "target_chain": "B",
        "metrics": {"plddt": .95, "i_pae": .1, "binder_scrmsd_ca": 1.,
                    "predicted_pdb": str(prediction), "model_names": ["one"]},
    }
    (tmp_path / "af2_refilter_result.json").write_text(json.dumps(report))
    return source, prediction, report


def parse(tmp_path):
    return parse_af2_refilter_output(tmp_path, ParserContext(
        target_id="synthetic", runtime_bucket_id="test", candidate_id="test",
        parent_ids=[], method_family="structure_refilter",
        binder_chain="A", target_chains_csv="B",
    ))[0]


def legacy_record(prediction):
    return ResultRecord(
        result_id="legacy", parent_ids=[], target_id="synthetic",
        backend_family="structure_refilter", runtime_bucket_id="test",
        metrics={"pLDDT": 95., "iPAE": .1, "binder_scRMSD": 1.},
        metrics_calibrated={}, route_lineage=[], gpu_h=.01, exit_status="ok",
        bins={"af2_strict_basis": "canonical", "binder_chain": "A"},
        artifacts={"pdb_path": str(prediction), "binder_chain": "A"},
    )


def test_parser_records_prediction_identity_and_preserves_measurements(tmp_path, pair):
    rec = parse(tmp_path)
    assert rec.artifacts["binder_chain"] == "B"
    assert rec.artifacts["target_chains"] == "A"
    assert rec.bins["af2_chain_identity"] == "verified"
    assert rec.metrics == {"pLDDT": 95., "iPAE": .1, "binder_scRMSD": 1.}
    assert is_strict_success(rec.metrics)


def test_legacy_archive_is_resolved_without_mutation(tmp_path, pair):
    rec = legacy_record(pair[1])
    before = copy.deepcopy(rec)
    original_report = (tmp_path / "af2_refilter_result.json").read_bytes()
    chain = _record_binder_chain(rec)
    assert chain == "B"
    extracted = tmp_path / "binder.pdb"
    assert _write_chain_only_pdb(pair[1], extracted, preferred_chain=chain) == (True, "B")
    assert len([line for line in extracted.read_text().splitlines() if line.startswith("ATOM")]) == 2
    assert _sequence_for_result(rec) == ("ST", "B")
    assert rec == before
    assert (tmp_path / "af2_refilter_result.json").read_bytes() == original_report


def test_verified_record_is_portable_without_original_input_or_report(tmp_path, pair):
    rec = parse(tmp_path)
    pair[0].unlink()
    (tmp_path / "af2_refilter_result.json").unlink()
    assert _record_binder_chain(rec) == "B"
    assert _cached_sequence_for_result(rec, "B") == ("ST", "B")
    # A replaced structure at the same path must not reuse a cached sequence.
    write_pdb(pair[1], {"A": ["ALA", "GLY", "ALA"], "B": ["GLY", "THR"]})
    assert _record_binder_chain(rec) == ""
    assert _cached_sequence_for_result(rec, "B") == (None, None)


@pytest.mark.parametrize("problem", ["missing_input", "missing_report", "malformed_report", "ambiguous_binder", "target_mismatch"])
def test_unresolved_identity_never_falls_back_to_target(tmp_path, pair, problem):
    source, prediction, report = pair
    if problem == "missing_input":
        source.unlink()
    elif problem == "missing_report":
        (tmp_path / "af2_refilter_result.json").unlink()
    elif problem == "malformed_report":
        (tmp_path / "af2_refilter_result.json").write_text("[]")
    elif problem == "ambiguous_binder":
        write_pdb(prediction, {"A": ["SER", "THR"], "B": ["SER", "THR"]})
    else:
        write_pdb(prediction, {"A": ["GLY", "GLY", "ALA"], "B": ["SER", "THR"]})
    rec = legacy_record(prediction)
    assert _record_binder_chain(rec) == ""
    assert _sequence_for_result(rec) == (None, None)
    assert _write_chain_only_pdb(prediction, tmp_path / "out.pdb", preferred_chain="") == (False, None)


def test_identical_target_subunits_are_supported(tmp_path):
    source, prediction = tmp_path / "in.pdb", tmp_path / "out.pdb"
    target = ["ALA", "GLY", "ALA"]
    write_pdb(source, {"X": target, "Y": target, "Z": target, "Q": ["SER", "THR"]})
    write_pdb(prediction, {"A": target, "B": target, "C": target, "D": ["SER", "THR"]})
    mapping = resolve_prediction_chains(
        {"input_pdb": "in.pdb", "target_chain": "X,Y,Z", "binder_chain": "Q"},
        prediction, report_dir=tmp_path,
    )
    assert mapping["binder_chain"] == "D"
    assert mapping["target_chains"] == ["A", "B", "C"]


def test_unresolved_parser_retains_quality_evidence(tmp_path, pair):
    pair[0].unlink()
    rec = parse(tmp_path)
    assert rec.bins["af2_chain_identity"] == "unresolved"
    assert "binder_chain" not in rec.artifacts
    assert is_strict_success(rec.metrics)
    assert _record_binder_chain(rec) == ""


def test_runner_saves_separate_prediction_map(tmp_path, pair, monkeypatch):
    import sys
    from trex import af2_refilter_runner as runner
    source, prediction, _ = pair
    contents = prediction.read_text()
    class FakeModel:
        _model_names = ["one"]
        def prep_inputs(self, *args, **kwargs):
            pass
        def set_seq(self, **kwargs):
            pass
        def predict(self, **kwargs):
            return {"losses": {"plddt": .05, "i_pae": .1, "rmsd": 1.}}
        def save_pdb(self, path, **kwargs):
            Path(path).write_text(contents)
    monkeypatch.setitem(sys.modules, "colabdesign", SimpleNamespace(mk_af_model=lambda **kw: FakeModel()))
    monkeypatch.setattr(runner, "_args", lambda: SimpleNamespace(
        input_pdb=source, out_dir=tmp_path, community_root=tmp_path,
        af2_data_dir=tmp_path, target_chain="B", binder_chain="A", model_names="one",
        use_initial_guess=1, use_multimer=True, num_recycles=3, seed=0,
    ))
    runner.main()
    report = json.loads((tmp_path / "af2_refilter_result.json").read_text())
    assert report["binder_chain"] == "A"  # input provenance
    assert report["prediction_chains"]["binder_chain"] == "B"
    assert report["metrics"]["binder_scrmsd_ca"] == 1.


def test_unresolved_singleton_receives_no_diversity_credit(tmp_path, pair, monkeypatch):
    from trex import foldseek_clusterer as structures
    from trex.sequence_clusterer import cluster_archive_sequences
    pair[0].unlink()
    rec = legacy_record(pair[1])
    monkeypatch.setattr(structures.shutil, "which", lambda binary: "/fake/foldseek")
    def unexpected_run(*args, **kwargs):
        raise AssertionError("Unresolved structures must be excluded before clustering")
    monkeypatch.setattr(structures.subprocess, "run", unexpected_run)
    result = structures.cluster_archive_pdbs([rec], target_id="synthetic")
    assert result.cluster_by_result_id == {}
    assert result.n_scope_fallback == 1
    assert result.status == "no_structures"
    sequences = cluster_archive_sequences([rec], target_id="synthetic")
    assert sequences.cluster_by_result_id == {}
    assert sequences.n_sequences == 0
