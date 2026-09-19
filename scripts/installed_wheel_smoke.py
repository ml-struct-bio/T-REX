#!/usr/bin/env python3
"""Exercise public T-ReX workflows from an installed wheel, outside the source tree."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


TARGET_CONFIG = {
    "target_id": "smoke_target",
    "target_class": "protein",
    "hotspots": ["A1"],
    "forbidden_surfaces": [],
    "chain_ids": ["A"],
    "assay_geometry": None,
    "developability_filters": {},
    "panel_size_K": 8,
}
TARGET_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.121  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   9.121  1.00 20.00           C
ATOM      3  C   ALA A   1      13.000  14.620   9.500  1.00 20.00           C
TER
END
"""
EXPECTED_MANIFEST_FIELDS = [
    "rank",
    "result_id",
    "target_id",
    "backend_family",
    "production_quality",
    "pLDDT",
    "iPAE",
    "binder_scRMSD",
    "ipTM",
    "min_ipae",
    "avg_ipsae",
    "interface_contact_density",
    "structure_bin",
    "sequence_bin",
    "src_pdb",
    "dst_pdb",
]


def _run(
    bin_dir: Path,
    work_dir: Path,
    command: str,
    *arguments: object,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(bin_dir / command), *(str(value) for value in arguments)],
        cwd=work_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[:2_000]
        raise RuntimeError(f"{command} exited with {completed.returncode}: {detail}")
    return completed


def _run_json(
    bin_dir: Path,
    work_dir: Path,
    command: str,
    *arguments: object,
) -> dict[str, Any]:
    completed = _run(bin_dir, work_dir, command, *arguments)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{command} did not emit one JSON document: {completed.stdout[:2_000]}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{command} JSON output must be an object")
    return payload


def _expect_schema(payload: dict[str, Any], expected: str) -> None:
    observed = payload.get("schema_version")
    if observed != expected:
        raise RuntimeError(f"expected schema {expected!r}, observed {observed!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bin_dir = args.bin_dir.expanduser().resolve()
    expected_prefix = args.expected_prefix.expanduser().resolve()

    import trex

    imported_package = Path(trex.__file__).resolve()
    if not imported_package.is_relative_to(expected_prefix):
        raise RuntimeError(
            "wheel smoke imported T-ReX outside the isolated environment: "
            f"{imported_package}"
        )

    commands = (
        "trex",
        "trex-analyze",
        "trex-backend",
        "trex-controller",
        "trex-export",
        "trex-panel",
        "trex-provenance",
        "trex-target",
        "trex-validate",
    )
    with tempfile.TemporaryDirectory(prefix="trex-installed-wheel-") as temp:
        work_dir = Path(temp)
        for command in commands:
            _run(bin_dir, work_dir, command, "--help")

        target_config = work_dir / "target.json"
        target_config.write_text(
            json.dumps(TARGET_CONFIG, indent=2) + "\n", encoding="utf-8"
        )
        target_pdb = work_dir / "target.pdb"
        target_pdb.write_text(TARGET_PDB, encoding="utf-8")
        archive_root = work_dir / "archive"
        archive_root.mkdir()

        campaign_path = work_dir / "campaign.yaml"
        campaign_archive = work_dir / "campaign-archive"
        _run(
            bin_dir,
            work_dir,
            "trex",
            "campaign",
            "init",
            campaign_path,
            "--target",
            "custom",
            "--archive-root",
            campaign_archive,
            "--target-constraint",
            target_config,
            "--target-pdb",
            target_pdb,
        )
        resolved = _run_json(
            bin_dir, work_dir, "trex", "campaign", "show", campaign_path, "--json"
        )
        _expect_schema(resolved, "trex.resolved-campaign.v1")
        preflight = _run_json(
            bin_dir,
            work_dir,
            "trex",
            "campaign",
            "preflight",
            campaign_path,
            "--json",
        )
        _expect_schema(preflight, "trex.campaign-preflight.v1")
        _expect_schema(preflight["preflight"], "trex.preflight-report.v1")
        _run(bin_dir, work_dir, "trex", "campaign", "run", campaign_path, "--dry-run")
        if campaign_archive.exists():
            raise RuntimeError("campaign dry-run created its archive")
        status = _run_json(
            bin_dir, work_dir, "trex", "campaign", "status", archive_root, "--json"
        )
        _expect_schema(status, "trex.campaign-status.v2")

        target_list = _run_json(bin_dir, work_dir, "trex-target", "list", "--json")
        _expect_schema(target_list, "trex.target-list.v1")
        resolved_target = _run_json(
            bin_dir,
            work_dir,
            "trex-target",
            "resolve",
            "custom",
            "--target-config",
            target_config,
            "--target-pdb",
            target_pdb,
            "--format",
            "json",
        )
        _expect_schema(resolved_target, "trex.resolved-target.v1")
        backend_list = _run_json(bin_dir, work_dir, "trex-backend", "list", "--json")
        _expect_schema(backend_list, "trex.backend-list.v1")
        validation = _run_json(
            bin_dir,
            work_dir,
            "trex-validate",
            "--target",
            "custom",
            "--target-config",
            target_config,
            "--target-pdb",
            target_pdb,
            "--enabled-families",
            "",
            "--json",
        )
        _expect_schema(validation, "trex.installation-validation.v1")
        if not validation.get("ok"):
            raise RuntimeError("custom-target installation validation failed")

        analysis_commands = (
            ("summary", "trex.analysis-summary.v1"),
            ("validate", "trex.archive-validation.v1"),
            ("trace", "trex.decision-trace.v1"),
        )
        for subcommand, schema in analysis_commands:
            payload = _run_json(
                bin_dir,
                work_dir,
                "trex-analyze",
                subcommand,
                "--archive-root",
                archive_root,
                "--json",
            )
            _expect_schema(payload, schema)
        layout = _run_json(bin_dir, work_dir, "trex-analyze", "schema", "--json")
        _expect_schema(layout, "trex.archive-layout.v1")
        if len(layout.get("streams", [])) != 13:
            raise RuntimeError("archive layout does not expose all 13 record streams")

        panel = _run_json(
            bin_dir,
            work_dir,
            "trex-panel",
            "--archive-root",
            archive_root,
            "--target-id",
            "custom",
            "--panel-size",
            5,
            "--no-dedup",
        )
        _expect_schema(panel, "trex.final-panel.v1")
        export_dir = work_dir / "export"
        exported = _run_json(
            bin_dir,
            work_dir,
            "trex-export",
            "--archive-root",
            archive_root,
            "--target-id",
            "custom",
            "--n",
            5,
            "--out-dir",
            export_dir,
            "--no-copy",
            "--no-dedup",
        )
        _expect_schema(exported, "trex.best-n-export.v1")
        manifest = json.loads(
            (export_dir / "manifest.json").read_text(encoding="utf-8")
        )
        _expect_schema(manifest, "trex.best-n-manifest.v1")
        with (export_dir / "manifest.csv").open(encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        if header != EXPECTED_MANIFEST_FIELDS:
            raise RuntimeError(f"unexpected manifest.csv fields: {header}")

        example_script = Path(__file__).resolve().parents[1] / "examples/analysis_demo/run_demo.py"
        example = _run_json(
            bin_dir, work_dir, "python", example_script,
            "--out", work_dir / "analysis-demo",
        )
        if not example.get("ok") or example.get("exported_result_ids") != ["demo_qualified"]:
            raise RuntimeError("installed-wheel analysis example failed")

    print(
        json.dumps(
            {
                "schema_version": "trex.installed-wheel-smoke.v1",
                "ok": True,
                "imported_package": str(imported_package),
                "commands_checked": list(commands),
                "analysis_example_checked": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
