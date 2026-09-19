"""Tests for the best-N-by-score wet-lab export.

The export is intentionally DIFFERENT from select_production_panel: it ranks
strict successes by score and keeps structural near-duplicates (you may want
several variants of a strong basin), whereas the panel deduplicates to one per
structure bin. These tests lock in: (1) hard gates (strict + exit ok + a real
PDB on disk), (2) score-descending order, (3) structural duplicates are KEPT,
(4) PDBs are physically copied with a stable manifest.
"""

from __future__ import annotations

import csv
import json

import pytest

from trex.archive import Archive
from trex.export_best_n import (
    BEST_N_MANIFEST_SCHEMA_VERSION,
    BEST_N_OUTPUT_SCHEMA_VERSION,
    export_best_n,
    main as export_main,
)
from trex.panel import production_quality
from trex.schemas import ResultRecord


def _rec(
    rid,
    pdb,
    *,
    fam="complexa_beam",
    plddt=94.0,
    ipae=0.15,
    rmsd=1.0,
    ok=True,
    fs="FS_a",
) -> ResultRecord:
    return ResultRecord(
        result_id=rid,
        parent_ids=[],
        target_id="02_PDL1",
        backend_family=fam,
        runtime_bucket_id="rb",
        metrics={"pLDDT": plddt, "iPAE": ipae, "binder_scRMSD": rmsd},
        metrics_calibrated={},
        route_lineage=[fam],
        gpu_h=0.1,
        exit_status=("ok" if ok else "timeout"),
        bins={"foldseek_su": fs},
        artifacts={"pdb_path": str(pdb)} if pdb else {},
        panel_ready=False,
    )


def _mk_pdb(p):
    p.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    return p


def test_export_ranks_by_score_and_copies_pdbs(tmp_path):
    arch = Archive(tmp_path / "arch")
    pa, pb, pc = (_mk_pdb(tmp_path / f"{n}.pdb") for n in ("a", "b", "c"))
    arch.append(_rec("a", pa, plddt=96.0, ipae=0.10, rmsd=0.6))  # strongest
    arch.append(_rec("b", pb, plddt=93.0, ipae=0.18, rmsd=1.0))  # mid
    arch.append(
        _rec("c", pc, plddt=90.5, ipae=0.224, rmsd=1.45)
    )  # weakest (just passes)

    out = tmp_path / "panel"
    diag = export_best_n(
        arch,
        target_id="02_PDL1",
        n=100,
        out_dir=out,
        copy_pdbs=True,
        annotate_dedup=False,
    )

    man = json.loads((out / "manifest.json").read_text())
    assert man["schema_version"] == BEST_N_MANIFEST_SCHEMA_VERSION
    assert man["diagnostics"]["settings"]["su_tm_score"] == 0.60
    assert [d["result_id"] for d in man["designs"]] == ["a", "b", "c"]
    # ranking strictly follows production_quality
    qs = [d["production_quality"] for d in man["designs"]]
    assert qs == sorted(qs, reverse=True)
    assert diag["n_pdbs_copied"] == 3
    copied = sorted(p.name for p in (out / "pdbs").glob("*.pdb"))
    assert copied == [
        "rank001_complexa_beam_a.pdb",
        "rank002_complexa_beam_b.pdb",
        "rank003_complexa_beam_c.pdb",
    ]
    # CSV manifest is written with the strict axes as columns
    with (out / "manifest.csv").open() as fh:
        header = next(csv.reader(fh))
    for axis in ("pLDDT", "iPAE", "binder_scRMSD", "production_quality"):
        assert axis in header


def test_export_hard_gates(tmp_path):
    arch = Archive(tmp_path / "arch")
    good = _mk_pdb(tmp_path / "good.pdb")
    arch.append(_rec("strict_ok", good))  # kept
    arch.append(
        _rec("not_strict", good, plddt=80.0, ipae=0.4, rmsd=3.0)
    )  # fails strict
    arch.append(_rec("bad_exit", good, ok=False))  # exit != ok
    arch.append(_rec("no_pdb", None))  # no structure
    directory_named_pdb = tmp_path / "directory.pdb"
    directory_named_pdb.mkdir()
    arch.append(_rec("directory_pdb", directory_named_pdb))
    arch.append(_rec("missing_pdb", tmp_path / "does_not_exist.pdb"))  # path absent

    diag = export_best_n(
        arch,
        target_id="02_PDL1",
        n=100,
        out_dir=tmp_path / "panel",
        copy_pdbs=True,
        annotate_dedup=False,
    )
    man = json.loads((tmp_path / "panel" / "manifest.json").read_text())
    assert [d["result_id"] for d in man["designs"]] == ["strict_ok"]
    assert diag["n_strict_with_structure"] == 1


def test_export_keeps_structural_duplicates(tmp_path):
    """Unlike the diversity panel, best-N keeps same-structure-bin duplicates."""
    arch = Archive(tmp_path / "arch")
    p1, p2 = _mk_pdb(tmp_path / "1.pdb"), _mk_pdb(tmp_path / "2.pdb")
    # both share foldseek_su bin FS_x → the production panel would keep only one
    arch.append(_rec("dup1", p1, plddt=96.0, ipae=0.10, rmsd=0.6, fs="FS_x"))
    arch.append(_rec("dup2", p2, plddt=95.0, ipae=0.11, rmsd=0.7, fs="FS_x"))
    diag = export_best_n(
        arch,
        target_id="02_PDL1",
        n=100,
        out_dir=tmp_path / "panel",
        copy_pdbs=True,
        annotate_dedup=False,
    )
    man = json.loads((tmp_path / "panel" / "manifest.json").read_text())
    assert [d["result_id"] for d in man["designs"]] == ["dup1", "dup2"]  # BOTH kept
    assert diag["n_exported"] == 2
    assert diag["distinct_structure_bins"] == 1  # but flagged as 1 unique structure


def test_export_respects_n_cap(tmp_path):
    arch = Archive(tmp_path / "arch")
    for i in range(5):
        arch.append(
            _rec(
                f"r{i}",
                _mk_pdb(tmp_path / f"r{i}.pdb"),
                plddt=96.0 - i,
                ipae=0.10 + 0.01 * i,
                rmsd=0.6,
            )
        )
    diag = export_best_n(
        arch,
        target_id="02_PDL1",
        n=3,
        out_dir=tmp_path / "panel",
        copy_pdbs=False,
        annotate_dedup=False,
    )
    assert diag["n_exported"] == 3
    man = json.loads((tmp_path / "panel" / "manifest.json").read_text())
    assert [d["result_id"] for d in man["designs"]] == ["r0", "r1", "r2"]


@pytest.mark.parametrize(
    ("argument", "value"),
    (("n", 0), ("su_tm_score", 0.0), ("sequence_identity", 1.1)),
)
def test_export_rejects_invalid_settings(tmp_path, argument, value):
    archive = Archive(tmp_path / "archive")
    settings = {argument: value}
    with pytest.raises(ValueError):
        export_best_n(
            archive,
            target_id="02_PDL1",
            out_dir=tmp_path / "panel",
            annotate_dedup=False,
            **settings,
        )


def test_export_cli_rejects_missing_archive_without_creating_it(tmp_path, capsys):
    missing = tmp_path / "missing"
    return_code = export_main(
        [
            "--archive-root",
            str(missing),
            "--target-id",
            "02_PDL1",
        ]
    )

    assert return_code == 2
    assert "not a directory" in capsys.readouterr().err
    assert not missing.exists()


def test_export_cli_output_is_versioned_and_points_to_manifest(tmp_path, capsys):
    archive_root = tmp_path / "archive"
    Archive(archive_root)
    out_dir = tmp_path / "export"
    arguments = [
        "--archive-root",
        str(archive_root),
        "--target-id",
        "02_PDL1",
        "--n",
        "2",
        "--out-dir",
        str(out_dir),
        "--no-copy",
        "--no-dedup",
    ]
    return_code = export_main(arguments)

    assert return_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == BEST_N_OUTPUT_SCHEMA_VERSION
    assert payload["manifest_json"] == str(out_dir / "manifest.json")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == BEST_N_MANIFEST_SCHEMA_VERSION
    assert payload["manifest_csv"] == str(out_dir / "manifest.csv")
    assert (out_dir / "manifest.csv").read_text().startswith("rank,result_id,")

    assert export_main(arguments) == 2
    assert "output directory is not empty" in capsys.readouterr().err

    assert export_main([*arguments, "--overwrite"]) == 0
    replacement = json.loads(capsys.readouterr().out)
    assert replacement["diagnostics"]["settings"]["overwrite"] is True
    assert (out_dir / "manifest.csv").is_file()
    assert (out_dir / "manifest.json").is_file()


def test_export_overwrite_never_removes_unrecognized_files(tmp_path):
    archive = Archive(tmp_path / "archive")
    out_dir = tmp_path / "export"
    out_dir.mkdir()
    note = out_dir / "notes.txt"
    note.write_text("keep me")

    with pytest.raises(ValueError, match="unrecognized entries"):
        export_best_n(
            archive,
            target_id="02_PDL1",
            out_dir=out_dir,
            copy_pdbs=False,
            annotate_dedup=False,
            overwrite=True,
        )

    assert note.read_text() == "keep me"


def test_failed_overwrite_preserves_the_previous_complete_export(tmp_path, monkeypatch):
    archive = Archive(tmp_path / "archive")
    structure = _mk_pdb(tmp_path / "binder.pdb")
    archive.append(_rec("binder", structure))
    out_dir = tmp_path / "export"
    export_best_n(
        archive,
        target_id="02_PDL1",
        out_dir=out_dir,
        copy_pdbs=False,
        annotate_dedup=False,
    )
    original_manifest = (out_dir / "manifest.json").read_bytes()

    def fail_copy(*_args, **_kwargs):
        raise OSError("simulated copy failure")

    monkeypatch.setattr("trex.export_best_n.shutil.copy2", fail_copy)
    with pytest.raises(OSError, match="simulated copy failure"):
        export_best_n(
            archive,
            target_id="02_PDL1",
            out_dir=out_dir,
            annotate_dedup=False,
            overwrite=True,
        )

    assert (out_dir / "manifest.json").read_bytes() == original_manifest
    assert not list(tmp_path.glob(".export.tmp-*"))
    assert not list(tmp_path.glob(".export.previous-*"))
