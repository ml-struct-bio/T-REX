"""Public CLI commands preserve the campaign contract and launcher inputs."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import yaml

from trex.campaign.config import load_campaign
from trex.cli import main
from trex.public_cli import _submission_environment
from trex.targets import resolve_target


_ONE_RESIDUE_PDB = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.121  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.207   9.121  1.00 20.00           C
ATOM      3  C   ALA A   1      13.000  14.620   9.500  1.00 20.00           C
TER
END
"""


def _custom_campaign(tmp_path: Path) -> Path:
    constraint = tmp_path / "target.json"
    pdb = tmp_path / "target.pdb"
    constraint.write_text(
        json.dumps(
            {
                "target_id": "custom_001",
                "target_class": "protein",
                "chain_ids": ["A"],
                "hotspots": ["A1"],
                "panel_size_K": 8,
            }
        )
    )
    pdb.write_text(_ONE_RESIDUE_PDB)
    source = tmp_path / "campaign.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": "trex.campaign.v1",
                "name": "public-interface-test",
                "target": {
                    "name": "custom",
                    "constraint": str(constraint),
                    "pdb": str(pdb),
                },
                "run": {
                    "archive_root": str(tmp_path / "archive"),
                    "max_wall_hours": 2.5,
                    "seed": 4,
                    "worker_gpus": [1, 2],
                    "enabled_families": ["complexa_beam"],
                },
            },
            sort_keys=False,
        )
    )
    return source


def test_init_records_target_gpu_time_and_output(tmp_path: Path) -> None:
    config = tmp_path / "campaign.yaml"

    rc = main(
        [
            "init",
            str(config),
            "--target",
            "cd45",
            "--gpus",
            "8",
            "--hours",
            "12.5",
            "--output",
            str(tmp_path / "runs"),
        ]
    )

    assert rc == 0
    payload = yaml.safe_load(config.read_text())
    assert payload["target"]["name"] == "cd45"
    assert payload["run"]["worker_gpus"] == [str(i) for i in range(1, 8)]
    assert payload["run"]["max_wall_hours"] == 12.5
    assert payload["run"]["archive_root"] == str(tmp_path / "runs")


def test_init_rejects_invalid_resource_values(tmp_path: Path) -> None:
    assert (
        main(
            [
                "init",
                str(tmp_path / "bad.yaml"),
                "--target",
                "cd45",
                "--gpus",
                "1",
                "--output",
                str(tmp_path / "runs"),
            ]
        )
        == 2
    )


def test_design_alias_runs_the_existing_read_only_path(tmp_path: Path) -> None:
    archive = tmp_path / "archive"

    assert main(["design", str(_custom_campaign(tmp_path)), "--dry-run"]) == 0
    assert not archive.exists()


def test_submission_environment_is_derived_from_yaml(tmp_path: Path) -> None:
    campaign = load_campaign(_custom_campaign(tmp_path))

    environment = _submission_environment(
        campaign,
        {
            "TREX_CONTROLLER_PYTHON": "/installed/controller/python",
            "TARGET": "stale-target",
            "TREX_NUM_GPUS": "99",
        },
    )

    assert environment["TARGET"] == "custom"
    assert environment["TREX_NUM_GPUS"] == "3"
    assert environment["TREX_MAX_WALL_H"] == "2.5"
    assert environment["TREX_ARCHIVE_BASE"] == str(tmp_path / "archive")
    assert environment["TREX_WORKER_GPUS"] == "1,2"
    assert environment["TREX_CONTROLLER_PYTHON"] == "/installed/controller/python"
    assert base64.b64decode(environment["TREX_CAMPAIGN_SOURCE_B64"]) == (
        campaign.source_content
    )
    resolved = json.loads(base64.b64decode(environment["TREX_CAMPAIGN_RESOLVED_B64"]))
    assert resolved["campaign"]["source_sha256"] == campaign.source_sha256


def test_registered_target_accepts_bundle_or_targets_root(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    config_dir = repo / "config" / "targets"
    config_dir.mkdir(parents=True)
    (config_dir / "registry.json").write_text(
        json.dumps(
            {
                "schema_version": "trex_target_registry.v1",
                "targets": {
                    "demo": {
                        "config": "demo.json",
                        "pdb": "bindcraft_targets/demo.pdb",
                        "target_id": "demo_001",
                    }
                },
            }
        )
    )
    (config_dir / "demo.json").write_text(json.dumps({"target_id": "demo_001"}))
    bundle = tmp_path / "T-REX-assets"
    pdb = bundle / "targets" / "bindcraft_targets" / "demo.pdb"
    pdb.parent.mkdir(parents=True)
    pdb.write_text(_ONE_RESIDUE_PDB)

    from_bundle = resolve_target("demo", repo_root=repo, asset_root=bundle)
    from_targets = resolve_target("demo", repo_root=repo, asset_root=bundle / "targets")

    assert from_bundle.pdb_path == pdb.resolve()
    assert from_targets.pdb_path == pdb.resolve()


def test_backend_defaults_match_documented_environment_layout(tmp_path: Path) -> None:
    campaign = load_campaign(_custom_campaign(tmp_path))

    assert campaign.config.backends.bindcraft_env == (
        campaign.config.backends.bindcraft_repo / ".venv"
    )
    assert campaign.config.backends.boltzgen_binary == (
        campaign.config.backends.boltzgen_repo / ".venv" / "bin" / "boltzgen"
    )


def test_export_infers_target_from_latest_evidence(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from trex import public_cli

    archive = tmp_path / "archive"
    archive.mkdir()
    captured: list[str] = []
    monkeypatch.setattr(
        public_cli,
        "collect_campaign_status",
        lambda _: SimpleNamespace(
            campaign_input=None,
            latest_evidence={"target_id": "target_from_evidence"},
        ),
    )
    monkeypatch.setattr(
        public_cli,
        "export_main",
        lambda argv: captured.extend(argv) or 0,
    )
    args = SimpleNamespace(
        archive_root=archive,
        target_id=None,
        n=10,
        out_dir=None,
        foldseek_binary="foldseek",
        mmseqs_binary="mmseqs",
        no_copy=False,
        no_dedup=False,
        allow_missing_structure=False,
        overwrite=False,
    )

    assert public_cli._export(args) == 0
    assert captured[captured.index("--target-id") + 1] == "target_from_evidence"


def test_setup_command_delegates_one_shot_install(tmp_path: Path, monkeypatch) -> None:
    from trex import public_cli

    captured = {}
    monkeypatch.setattr(public_cli, "_repository_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(
        public_cli,
        "setup_installation",
        lambda **kwargs: captured.update(kwargs),
    )

    assert (
        main(
            [
                "setup",
                "--asset-root",
                str(tmp_path / "assets"),
                "--asset-zip",
                str(tmp_path / "assets.zip"),
                "--jobs",
                "3",
            ]
        )
        == 0
    )
    assert captured["root"] == tmp_path / "repo"
    assert captured["asset_root"] == tmp_path / "assets"
    assert captured["asset_zip"] == tmp_path / "assets.zip"
    assert captured["jobs"] == 3
