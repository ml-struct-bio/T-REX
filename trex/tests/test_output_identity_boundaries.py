"""Synthetic regressions for identity changes at parser and cache boundaries."""

import json
from dataclasses import replace

import pytest

from trex.clustering_cache import (
    _cached_sequence_for_result,
    clear_clustering_caches,
)
from trex.foldseek_clusterer import _write_chain_only_pdb
from trex.output_identity import MAP_KEY, annotate_output_identity
from trex.sequence_clusterer import _sequence_for_result
from trex.tests.test_output_identity import _pdb, _record


@pytest.mark.parametrize("second_model_offset", [0, 10])
def test_extraction_uses_the_model_that_was_identity_verified(tmp_path, second_model_offset):
    target = _pdb(tmp_path / "target.pdb", {"A": ["ALA", "GLY"]})
    first = _pdb(
        tmp_path / "first.pdb", {"A": ["ALA", "GLY"], "B": ["SER", "THR"]}
    )
    second = _pdb(
        tmp_path / "second.pdb", {"A": ["SER", "THR"], "B": ["ALA", "GLY"]}
    )
    second_lines = [
        line[:22] + f"{int(line[22:26]) + second_model_offset:4d}" + line[26:]
        if line.startswith("ATOM  ") else line
        for line in second.read_text().splitlines()
    ]
    second.write_text("\n".join(second_lines) + "\n")
    output = tmp_path / "models.pdb"
    output.write_text(
        "MODEL        1\n" + first.read_text().removesuffix("END\n")
        + "ENDMDL\nMODEL        2\n"
        + second.read_text().removesuffix("END\n") + "ENDMDL\nEND\n"
    )
    record = annotate_output_identity(
        _record(output), target_pdb=target, target_chains=["A"]
    )
    assert record.bins["output_chain_identity"] == "verified"
    selected = tmp_path / "selected.pdb"
    assert _write_chain_only_pdb(output, selected, preferred_chain="B") == (True, "B")
    residue_names = [
        line[17:20].strip()
        for line in selected.read_text().splitlines()
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA"
    ]
    assert residue_names == ["SER", "THR"]
    legacy = _record(output, family="bindcraft")
    legacy.artifacts["binder_chain"] = "B"
    assert _sequence_for_result(legacy) == ("ST", "B")


def test_sequence_cache_invalidates_when_legacy_roles_become_verified(tmp_path):
    clear_clustering_caches()
    target = _pdb(tmp_path / "target.pdb", {"A": ["ALA", "GLY"]})
    output = _pdb(
        tmp_path / "output.pdb", {"A": ["ALA", "GLY"], "B": ["SER", "THR"]}
    )
    legacy = _record(output, family="bindcraft")
    legacy.artifacts.update(binder_chain="B", sequence="AG")
    assert _cached_sequence_for_result(legacy, "B") == ("AG", "artifact")
    verified = annotate_output_identity(
        legacy, target_pdb=target, target_chains=["A"]
    )
    assert _sequence_for_result(verified) == ("ST", "B")
    assert _cached_sequence_for_result(verified, "B") == ("ST", "B")


@pytest.mark.parametrize("sequence_key", ["binder_sequence", "sequence", "seq"])
def test_sequence_cache_invalidates_changed_artifact_sequence(tmp_path, sequence_key):
    clear_clustering_caches()
    record = _record(tmp_path / "absent.pdb", family="bindcraft")
    record = replace(record, artifacts={"binder_chain": "B", sequence_key: "AG"})
    assert _cached_sequence_for_result(record, "B") == ("AG", "artifact")
    changed = replace(record, artifacts={"binder_chain": "B", sequence_key: "ST"})
    assert _cached_sequence_for_result(changed, "B") == ("ST", "artifact")


@pytest.mark.parametrize("source", [None, ""])
def test_lost_output_path_clears_verified_status_and_cluster_credit(tmp_path, source):
    target = _pdb(tmp_path / "target.pdb", {"A": ["ALA", "GLY"]})
    output = _pdb(
        tmp_path / "output.pdb", {"A": ["ALA", "GLY"], "B": ["SER", "THR"]}
    )
    record = annotate_output_identity(
        _record(output), target_pdb=target, target_chains=["A"]
    )
    assert json.loads(record.artifacts[MAP_KEY])["binder_chain"] == "B"
    record.bins.update(foldseek_su="stale", sequence_su="stale")
    if source is None:
        record.artifacts.pop("pdb_path")
    else:
        record.artifacts["pdb_path"] = source
    checked = annotate_output_identity(record)
    assert checked.bins["output_chain_identity"] == "unresolved"
    assert "foldseek_su" not in checked.bins
    assert "sequence_su" not in checked.bins
    assert "binder_chain" not in checked.artifacts


def test_score_only_record_needs_no_structure_identity(tmp_path):
    record = replace(_record(tmp_path / "absent.pdb"), artifacts={}, bins={})
    assert annotate_output_identity(record) is record



def test_verified_recovery_invalidates_the_cached_clustering_result(tmp_path, monkeypatch):
    import trex.clustering_cache as cache
    from trex.sequence_clusterer import SequenceClusteringResult

    clear_clustering_caches()
    target = _pdb(tmp_path / "target.pdb", {"A": ["ALA", "GLY"]})
    legacy = []
    for rid, residues in [("one", ["SER", "THR"]), ("two", ["VAL", "LEU"])]:
        output = _pdb(tmp_path / (rid + ".pdb"), {"A": ["ALA", "GLY"], "B": residues})
        record = _record(output, rid=rid, family="bindcraft")
        record.artifacts.update(binder_chain="B", sequence="AG")
        legacy.append(record)
    calls = []

    def cluster(rows, **kwargs):
        groups = {row.result_id: _sequence_for_result(row)[0] for row in rows}
        calls.append(groups)
        return SequenceClusteringResult(groups, len(rows), len(set(groups.values())), "ok")

    monkeypatch.setattr(cache, "cluster_archive_sequences", cluster)
    before = cache.cluster_archive_sequences_cached(legacy, target_id="test")
    assert before.n_clusters == 1
    verified = [
        annotate_output_identity(row, target_pdb=target, target_chains=["A"])
        for row in legacy
    ]
    after = cache.cluster_archive_sequences_cached(verified, target_id="test")
    assert after.n_clusters == 2
    assert len(calls) == 2


def test_live_tick_drops_credit_when_a_verified_output_loses_its_path(tmp_path, monkeypatch):
    import trex.evidence_reducer as reducer
    from trex.archive import Archive
    from trex.live_tick import LiveTickConfig, run_live_tick
    from trex.schemas import TargetConstraint
    from trex.tests.test_output_identity import _provenance
    from trex.tick.config import FoldseekConfig

    archive = Archive(tmp_path / "archive")
    target = _pdb(tmp_path / "target.pdb", {"A": ["ALA", "GLY"]})
    config = tmp_path / "target.json"
    config.write_text(json.dumps({"chain_ids": ["A"]}))
    _provenance(archive, target, config)
    output = _pdb(
        tmp_path / "output.pdb", {"A": ["ALA", "GLY"], "B": ["SER", "THR"]}
    )
    record = annotate_output_identity(
        _record(output), target_pdb=target, target_chains=["A"]
    )
    record.artifacts.pop("pdb_path")
    record.bins["foldseek_su"] = "stale"
    archive.append(record)
    before = (archive.root / "result_records.jsonl").read_bytes()
    trust = []
    classify = reducer.classify_state

    def observe(**kwargs):
        trust.append(kwargs["su_dedup_trusted"])
        return classify(**kwargs)

    monkeypatch.setattr(reducer, "classify_state", observe)
    result = run_live_tick(
        archive,
        TargetConstraint("test", "protein", chain_ids=["A"]),
        tick_id="tick1",
        tick_id_int=1,
        elapsed_wall_h=1.0,
        remaining_wall_h=1.0,
        cfg=LiveTickConfig(foldseek=FoldseekConfig(binary="true")),
        evidence_only=True,
    )
    assert result["evidence"]["run_su_count"] == 0
    assert trust == [False]
    assert (archive.root / "result_records.jsonl").read_bytes() == before
