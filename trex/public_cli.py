"""Concise public commands layered over the reviewed campaign interfaces."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
from pathlib import Path

from .campaign import cli as campaign_cli
from .campaign.config import ConfigError, load_campaign
from .campaign.environment import applied_environment
from .campaign.status import collect_campaign_status
from .campaign.template import write_campaign_template
from .export_best_n import main as export_main
from .setup_installation import PYROSETTA_INDEX, setup_installation


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_profile(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        type=Path,
        help="installation profile; default: .env in the T-REX checkout when present",
    )


def _add_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", type=Path, help="campaign YAML to create")
    parser.add_argument("--target", required=True, help="registered target label")
    parser.add_argument(
        "--gpus",
        type=int,
        default=4,
        help="total GPUs on one node, including one for the local LLM (default: 4)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=48.0,
        help="campaign duration in hours (default: 48)",
    )
    parser.add_argument(
        "--output",
        "--archive-root",
        dest="archive_root",
        required=True,
        help="output directory (submission creates a timestamped run below it)",
    )
    parser.add_argument("--name")
    parser.add_argument(
        "--asset-root",
        help="T-REX-assets bundle root or its targets/ directory; omit to use .env",
    )
    parser.add_argument("--target-constraint")
    parser.add_argument("--target-pdb")
    parser.add_argument("--force", action="store_true")


def add_public_parsers(subparsers: argparse._SubParsersAction) -> None:
    setup = subparsers.add_parser(
        "setup",
        help="install pinned backends, environments, tools, checkpoints and profiles",
    )
    setup.add_argument(
        "--asset-root",
        type=Path,
        default=Path("../T-REX-assets"),
        help="checkpoint/target directory (default: ../T-REX-assets)",
    )
    setup.add_argument(
        "--asset-zip",
        type=Path,
        help="import a previously downloaded T-REX-assets.zip instead of Google Drive",
    )
    setup.add_argument(
        "--drive-url", help="override the published Google Drive asset URL"
    )
    setup.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 1, 16),
        help="parallel compile jobs for Foldseek/MMseqs2 (default: up to 16)",
    )
    setup.add_argument(
        "--pyrosetta-index",
        default=PYROSETTA_INDEX,
        help="licensed PyRosetta wheel index",
    )
    setup.set_defaults(command_handler=_setup)

    init = subparsers.add_parser(
        "init",
        help="create one campaign YAML from target, GPU, time, and output settings",
    )
    _add_init_arguments(init)
    init.set_defaults(command_handler=_init)

    check = subparsers.add_parser(
        "check",
        help="validate a campaign and its installed backends without running it",
    )
    check.add_argument("config", type=Path)
    _add_profile(check)
    check.add_argument(
        "--verify-checkpoint-content",
        action="store_true",
        help="rehash all declared checkpoint files (slow on shared storage)",
    )
    check.add_argument("--json", action="store_true", dest="as_json")
    check.set_defaults(command_handler=_check)

    design = subparsers.add_parser(
        "design", help="run a campaign inside an existing GPU allocation"
    )
    design.add_argument("config", type=Path)
    _add_profile(design)
    design.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and preflight without writing an archive or starting workers",
    )
    design.add_argument("--verify-backend-revisions", action="store_true")
    design.add_argument("--verify-checkpoint-content", action="store_true")
    design.set_defaults(command_handler=_design)

    submit = subparsers.add_parser(
        "submit", help="validate a campaign and submit it to Slurm"
    )
    submit.add_argument("config", type=Path)
    _add_profile(submit)
    submit.add_argument("--account", help="Slurm project/account")
    submit.add_argument("--partition", help="Slurm partition")
    submit.add_argument("--qos", help="Slurm quality of service")
    submit.add_argument(
        "--resume",
        type=Path,
        help="append to this interrupted campaign archive",
    )
    submit.add_argument(
        "--sbatch-option",
        action="append",
        default=[],
        metavar="OPTION",
        help="additional sbatch option; repeat as needed",
    )
    submit.set_defaults(command_handler=_submit)

    status = subparsers.add_parser(
        "status", help="summarize progress and archive integrity"
    )
    status.add_argument("archive_root", type=Path)
    status.add_argument("--json", action="store_true", dest="as_json")
    status.set_defaults(command_handler=campaign_cli._status)

    export = subparsers.add_parser(
        "export", help="write a ranked manifest and copy selected structures"
    )
    export.add_argument("archive_root", type=Path)
    export.add_argument("--target-id")
    export.add_argument("--n", type=int, default=100)
    export.add_argument("--out-dir", type=Path)
    export.add_argument("--no-copy", action="store_true")
    export.add_argument("--no-dedup", action="store_true")
    export.add_argument("--foldseek-binary", default="foldseek")
    export.add_argument("--mmseqs-binary", default="mmseqs")
    export.add_argument("--allow-missing-structure", action="store_true")
    export.add_argument("--overwrite", action="store_true")
    export.set_defaults(command_handler=_export)


def _setup(args: argparse.Namespace) -> int:
    setup_installation(
        root=_repository_root(),
        asset_root=args.asset_root,
        drive_url=args.drive_url,
        asset_zip=args.asset_zip,
        jobs=args.jobs,
        pyrosetta_index=args.pyrosetta_index,
    )
    return 0


def _init(args: argparse.Namespace) -> int:
    path = write_campaign_template(
        args.config,
        name=args.name or args.target,
        target=args.target,
        archive_root=args.archive_root,
        asset_root=args.asset_root,
        target_constraint=args.target_constraint,
        target_pdb=args.target_pdb,
        max_wall_hours=args.hours,
        total_gpus=args.gpus,
        force=args.force,
    )
    print(path)
    print(f"Next: trex check {shlex.quote(str(path))}")
    return 0


def _profile_environment(
    profile: Path | None, *, required: bool = False
) -> dict[str, str]:
    explicit = profile is not None
    selected = (profile or (_repository_root() / ".env")).expanduser().resolve()
    if not selected.is_file():
        if explicit or required:
            raise ConfigError(
                f"installation profile not found: {selected}; copy .env.example to .env"
            )
        return {}
    script = 'set -a\nsource "$1"\nset +a\nexec env -0'
    result = subprocess.run(
        ["bash", "-c", script, "trex-profile", str(selected)],
        env=os.environ.copy(),
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise ConfigError(f"could not load installation profile {selected}: {detail}")
    environment: dict[str, str] = {}
    for item in result.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        name, value = item.split(b"=", 1)
        environment[os.fsdecode(name)] = os.fsdecode(value)
    return environment


def _check(args: argparse.Namespace) -> int:
    profile = _profile_environment(args.profile)
    forwarded = argparse.Namespace(
        config=args.config,
        strict=True,
        require_model=True,
        skip_asset_hash=False,
        verify_backend_revisions=True,
        verify_checkpoint_content=args.verify_checkpoint_content,
        as_json=args.as_json,
    )
    with applied_environment(profile):
        return campaign_cli._preflight(forwarded)


def _design(args: argparse.Namespace) -> int:
    profile = _profile_environment(args.profile)
    forwarded = argparse.Namespace(
        config=args.config,
        dry_run=args.dry_run,
        require_model=not args.dry_run,
        verify_backend_revisions=args.verify_backend_revisions,
        verify_checkpoint_content=args.verify_checkpoint_content,
    )
    with applied_environment(profile):
        return campaign_cli._run(forwarded)


def _submission_environment(campaign, profile: dict[str, str]) -> dict[str, str]:
    workers = campaign.config.run.worker_gpus
    expected = tuple(str(index) for index in range(1, len(workers) + 1))
    if workers != expected:
        raise ConfigError(
            "trex submit reserves GPU 0 for the LLM and requires consecutive "
            f"worker_gpus {list(expected)}; got {list(workers)}"
        )
    policy = campaign.config.policy
    environment = os.environ.copy()
    environment.update(profile)
    environment.update(campaign.environment())
    environment.update(
        {
            "TREX_SKIP_ENV_FILE": "1",
            "TREX_CAMPAIGN_CONFIG": str(campaign.source_path),
            "TREX_CAMPAIGN_SOURCE_SHA256": campaign.source_sha256,
            "TREX_CAMPAIGN_SOURCE_B64": base64.b64encode(
                campaign.source_content
            ).decode("ascii"),
            "TREX_CAMPAIGN_RESOLVED_B64": base64.b64encode(
                (
                    json.dumps(campaign.as_dict(), indent=2, sort_keys=True) + "\n"
                ).encode()
            ).decode("ascii"),
            "TARGET": campaign.config.target.name,
            "TREX_TARGET_CONFIG": str(campaign.target.config_path),
            "TREX_TARGET_PDB": str(campaign.target.pdb_path),
            "TREX_NUM_GPUS": str(len(workers) + 1),
            "TREX_MAX_WALL_H": str(campaign.config.run.max_wall_hours),
            "TREX_ARCHIVE_BASE": str(campaign.config.run.archive_root),
            "TREX_ENABLED_FAMILIES": ",".join(campaign.effective_families),
            "TREX_SEED": str(campaign.config.run.seed),
            "TREX_LLM_MODEL": campaign.config.llm.model,
            "TREX_ENABLE_CRITIC": str(int(policy.critic)),
            "TREX_ENABLE_EVIDENCE_SKIP": str(int(policy.evidence_skip)),
            "TREX_ENABLE_EXEMPLARS": str(int(policy.exemplars)),
            "TREX_FOLDSEEK_SU_TM_SCORE": str(policy.foldseek_su_tm_score),
            "TREX_FOLDSEEK_COLLAPSE_TM_SCORE": str(policy.foldseek_collapse_tm_score),
            "TREX_SELECTOR_QUOTA_REALIZATION": policy.selector_quota_realization,
            "TREX_SELECTOR_MODE_WINDOW_K": str(policy.selector_mode_window_k),
            "TREX_SELECTOR_ADAPTIVE_MODE_WINDOW_K": str(
                int(policy.selector_adaptive_mode_window_k)
            ),
        }
    )
    return environment


def _submit(args: argparse.Namespace) -> int:
    profile = _profile_environment(args.profile, required=True)
    with applied_environment(profile):
        campaign = load_campaign(args.config)
    command = ["bash", str(_repository_root() / "scripts" / "submit.sh")]
    for name in ("account", "partition", "qos"):
        value = getattr(args, name)
        if value:
            command.append(f"--{name}={value}")
    command.extend(args.sbatch_option)
    environment = _submission_environment(campaign, profile)
    if args.resume is not None:
        archive = args.resume.expanduser().resolve()
        if not archive.is_dir():
            raise ConfigError(f"resume archive is not a directory: {archive}")
        environment["TREX_RESUME_ARCHIVE"] = str(archive)
    else:
        environment.pop("TREX_RESUME_ARCHIVE", None)
    result = subprocess.run(
        command,
        env=environment,
        cwd=_repository_root(),
    )
    return int(result.returncode)


def _export(args: argparse.Namespace) -> int:
    archive = args.archive_root.expanduser().resolve()
    target_id = args.target_id
    if not target_id:
        status = collect_campaign_status(archive)
        if status.campaign_input:
            target_id = status.campaign_input.get("target_id")
        if not target_id and status.latest_evidence:
            target_id = status.latest_evidence.get("target_id")
    if not target_id:
        raise ConfigError(
            "could not infer target ID from the archive; provide --target-id"
        )
    out_dir = (args.out_dir or archive / "export").expanduser().resolve()
    forwarded = [
        "--archive-root",
        str(archive),
        "--target-id",
        str(target_id),
        "--n",
        str(args.n),
        "--out-dir",
        str(out_dir),
        "--foldseek-binary",
        args.foldseek_binary,
        "--mmseqs-binary",
        args.mmseqs_binary,
    ]
    for enabled, flag in (
        (args.no_copy, "--no-copy"),
        (args.no_dedup, "--no-dedup"),
        (args.allow_missing_structure, "--allow-missing-structure"),
        (args.overwrite, "--overwrite"),
    ):
        if enabled:
            forwarded.append(flag)
    return int(export_main(forwarded))
