from __future__ import annotations

import csv
from pathlib import Path

from trex.capability_registry import default_registry
from trex.output_parsers.complexa import parse_complexa_output
from trex.output_parsers.types import ParserContext


def _write_complexa_rewards(out_dir: Path, pdb_name: str = "model.pdb") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / pdb_name).write_text(
        "ATOM      1  CA  ALA B   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n"
    )
    with (out_dir / "rewards_test.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pdb_path",
                "sample_type",
                "af2folding_plddt_log",
                "af2folding_i_pae",
                "af2folding_rmsd",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pdb_path": str(out_dir / pdb_name),
                "sample_type": "final",
                "af2folding_plddt_log": "0.93",
                "af2folding_i_pae": "0.2",
                "af2folding_rmsd": "1.1",
            }
        )


def _ctx(*, tick_id: str = "") -> ParserContext:
    return ParserContext(
        target_id="t1",
        runtime_bucket_id="rb1",
        candidate_id="warmstart_00_complexa_beam",
        parent_ids=["warmstart_00_complexa_beam"],
        method_family="complexa_beam",
        tick_id=tick_id,
    )


def test_complexa_result_id_is_stable_for_same_launch(tmp_path: Path):
    out_dir = tmp_path / "search_binder_local_pipeline_t1_v7_r001_warmstart_00"
    _write_complexa_rewards(out_dir)

    rid1 = parse_complexa_output(out_dir, _ctx(tick_id="v7r001"))[0].result_id
    rid2 = parse_complexa_output(out_dir, _ctx(tick_id="v7r001"))[0].result_id

    assert rid1 == rid2


def test_complexa_result_id_separates_repeated_warmstart_across_ticks(tmp_path: Path):
    out_dir = tmp_path / "search_binder_local_pipeline_t1_warmstart_00"
    _write_complexa_rewards(out_dir)

    rid1 = parse_complexa_output(out_dir, _ctx(tick_id="v7r001"))[0].result_id
    rid2 = parse_complexa_output(out_dir, _ctx(tick_id="v7r002"))[0].result_id

    assert rid1 != rid2


def test_complexa_result_id_separates_launch_dirs_when_tick_missing(tmp_path: Path):
    out1 = tmp_path / "launch_a"
    out2 = tmp_path / "launch_b"
    _write_complexa_rewards(out1, pdb_name="same_model.pdb")
    _write_complexa_rewards(out2, pdb_name="same_model.pdb")

    rid1 = parse_complexa_output(out1, _ctx())[0].result_id
    rid2 = parse_complexa_output(out2, _ctx())[0].result_id

    assert rid1 != rid2


def test_complexa_parser_emits_canonical_strict_metrics_directly(tmp_path: Path):
    out_dir = tmp_path / "launch_official_gate"
    _write_complexa_rewards(out_dir)

    rec = parse_complexa_output(out_dir, _ctx(tick_id="v7r001"))[0]

    assert rec.metrics["pLDDT"] == 93.0
    assert rec.metrics["iPAE"] == 0.2
    assert rec.metrics["binder_scRMSD"] == 1.1
    assert rec.metrics["complexa_native_pLDDT"] == 93.0
    assert rec.metrics["complexa_native_iPAE"] == 0.2
    assert rec.metrics["complexa_native_binder_scRMSD"] == 1.1
    assert rec.bins["strict_score_source"] == "complexa_af2folding_canonical"


def test_complexa_registry_is_direct_scored_not_diagnostic_only():
    reg = default_registry()
    for fam in ("complexa_beam", "complexa_best_of_n", "complexa_fk_steering", "complexa_mcts"):
        cap = reg.get(fam)
        assert cap is not None
        assert cap.outputs_diagnostic_only is False
    for fam in ("bindcraft", "boltzgen", "proteinmpnn_redesign"):
        cap = reg.get(fam)
        assert cap is not None
        assert cap.outputs_diagnostic_only is True


def test_complexa_parser_missing_core_metrics_is_diagnostic_fallback(tmp_path: Path):
    out_dir = tmp_path / "cx_missing"
    out_dir.mkdir()
    (out_dir / "candidate.pdb").write_text(
        "ATOM      1  CA  ALA B   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n"
    )
    with (out_dir / "rewards_missing.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdb_path", "pdb_index", "aatype", "total_reward", "sample_type", "metadata_tag"],
        )
        writer.writeheader()
        writer.writerow({
            "pdb_path": str(out_dir / "candidate.pdb"),
            "pdb_index": "0",
            "aatype": "ACDE",
            "total_reward": "0.0",
            "sample_type": "final",
            "metadata_tag": "af2_failed",
        })

    rec = parse_complexa_output(out_dir, _ctx(tick_id="v7r001"))[0]
    assert rec.metrics == {}
    for key in ("pLDDT", "iPAE", "binder_scRMSD"):
        assert key not in rec.metrics
    assert "strict_score_source" not in rec.bins
    assert rec.bins["complexa_af2folding_status"] == "missing_core_metrics"
    assert rec.bins["needs_canonical_score_conversion"] == "1"
    assert rec.artifacts["pdb_path"].endswith("candidate.pdb")


def test_complexa_parser_normalizes_relative_structure_path(tmp_path: Path):
    out_dir = tmp_path / "cx_relative"
    out_dir.mkdir()
    (out_dir / "candidate.pdb").write_text(
        "ATOM      1  CA  ALA B   1       0.0   0.0   0.0  1.00 90.00           C\nEND\n"
    )
    with (out_dir / "rewards_relative.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdb_path", "sample_type", "af2folding_plddt_log",
                "af2folding_i_pae", "af2folding_rmsd",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "pdb_path": "candidate.pdb",
            "sample_type": "final",
            "af2folding_plddt_log": "0.93",
            "af2folding_i_pae": "0.2",
            "af2folding_rmsd": "1.1",
        })

    rec = parse_complexa_output(out_dir, _ctx(tick_id="v7r001"))[0]
    assert rec.artifacts["pdb_path"] == str((out_dir / "candidate.pdb").resolve())
    assert rec.bins["strict_score_source"] == "complexa_af2folding_canonical"


def test_complexa_missing_core_with_missing_structure_does_not_trigger_chain_noop(tmp_path: Path):
    out_dir = tmp_path / "cx_missing_artifact"
    out_dir.mkdir()
    with (out_dir / "rewards_missing_artifact.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdb_path", "sample_type", "metadata_tag"],
        )
        writer.writeheader()
        writer.writerow({
            "pdb_path": "gone.pdb",
            "sample_type": "final",
            "metadata_tag": "af2_failed",
        })

    rec = parse_complexa_output(out_dir, _ctx(tick_id="v7r001"))[0]
    assert rec.metrics == {}
    assert rec.artifacts == {}
    assert rec.bins["complexa_af2folding_status"] == "missing_core_metrics"
    assert rec.bins["score_conversion_status"] == "missing_structure_artifact"
    assert rec.bins["artifact_status"] == "missing_or_unusable_pdb_path"
    assert "needs_canonical_score_conversion" not in rec.bins


def test_complexa_core_metrics_without_structure_do_not_emit_strict_keys(tmp_path: Path):
    out_dir = tmp_path / "cx_core_missing_artifact"
    out_dir.mkdir()
    with (out_dir / "rewards_core_missing_artifact.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pdb_path", "sample_type", "af2folding_plddt_log",
                "af2folding_i_pae", "af2folding_rmsd",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "pdb_path": "gone.pdb",
            "sample_type": "final",
            "af2folding_plddt_log": "0.96",
            "af2folding_i_pae": "0.1",
            "af2folding_rmsd": "0.7",
        })

    rec = parse_complexa_output(out_dir, _ctx(tick_id="v7r001"))[0]
    for key in ("pLDDT", "iPAE", "binder_scRMSD"):
        assert key not in rec.metrics
    assert rec.metrics["complexa_native_pLDDT"] == 96.0
    assert rec.metrics["complexa_native_iPAE"] == 0.1
    assert rec.metrics["complexa_native_binder_scRMSD"] == 0.7
    assert rec.artifacts == {}
    assert "strict_score_source" not in rec.bins
    assert rec.bins["complexa_af2folding_status"] == "missing_structure_artifact"
    assert rec.bins["artifact_status"] == "missing_or_unusable_pdb_path"
