"""Command handlers for the user-facing ``trex campaign`` interface."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .artifacts import write_campaign_artifacts
from .config import load_campaign
from .environment import applied_environment
from .preflight import preflight_campaign
from .provenance import write_campaign_provenance
from .run_lock import campaign_run_lock
from .presentation import render_campaign, render_preflight, render_status
from .status import collect_campaign_status
from .template import write_campaign_template


PREFLIGHT_OUTPUT_SCHEMA_VERSION = "trex.campaign-preflight.v1"


def add_campaign_parser(subparsers: argparse._SubParsersAction) -> None:
    campaign = subparsers.add_parser(
        "campaign",
        help="create, inspect, validate, run, and monitor a binder-design campaign",
    )
    commands = campaign.add_subparsers(dest="campaign_command", required=True)

    init = commands.add_parser("init", help="write an explicit campaign YAML template")
    init.add_argument("config", type=Path)
    init.add_argument("--target", required=True)
    init.add_argument("--archive-root", required=True)
    init.add_argument("--name")
    init.add_argument("--asset-root")
    init.add_argument("--target-constraint")
    init.add_argument("--target-pdb")
    init.add_argument("--force", action="store_true")
    init.set_defaults(command_handler=_init)

    show = commands.add_parser(
        "show", help="show absolute paths and the effective controller input"
    )
    show.add_argument("config", type=Path)
    show.add_argument("--json", action="store_true", dest="as_json")
    show.add_argument(
        "--controller-command",
        action="store_true",
        help="also print the equivalent legacy trex-controller command",
    )
    show.set_defaults(command_handler=_show)

    preflight = commands.add_parser(
        "preflight", help="validate the resolved input without launching work"
    )
    preflight.add_argument("config", type=Path)
    preflight.add_argument(
        "--strict", action="store_true", help="require all enabled backends"
    )
    preflight.add_argument("--require-model", action="store_true")
    preflight.add_argument("--skip-asset-hash", action="store_true")
    preflight.add_argument("--verify-backend-revisions", action="store_true")
    preflight.add_argument("--verify-checkpoint-content", action="store_true")
    preflight.add_argument("--json", action="store_true", dest="as_json")
    preflight.set_defaults(command_handler=_preflight)

    run = commands.add_parser(
        "run", help="preflight and start the asynchronous controller"
    )
    run.add_argument("config", type=Path)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and preflight without writing an archive or starting workers",
    )
    run.add_argument(
        "--require-model",
        action="store_true",
        help="require the local model during dry-run (always required for a real run)",
    )
    run.add_argument("--verify-backend-revisions", action="store_true")
    run.add_argument("--verify-checkpoint-content", action="store_true")
    run.set_defaults(command_handler=_run)

    status = commands.add_parser(
        "status", help="summarize records, execution, evidence, and archive integrity"
    )
    status.add_argument("archive_root", type=Path)
    status.add_argument("--json", action="store_true", dest="as_json")
    status.set_defaults(command_handler=_status)


def _init(args: argparse.Namespace) -> int:
    path = write_campaign_template(
        args.config,
        name=args.name or args.target,
        target=args.target,
        archive_root=args.archive_root,
        asset_root=args.asset_root,
        target_constraint=args.target_constraint,
        target_pdb=args.target_pdb,
        force=args.force,
    )
    print(path)
    print(f"Next: trex campaign show {shlex.quote(str(path))}")
    return 0


def _show(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    if args.as_json:
        print(json.dumps(campaign.as_dict(), indent=2, sort_keys=True))
    else:
        print(render_campaign(campaign))
        if args.controller_command:
            print("\nEquivalent controller command")
            print("  " + shlex.join(["trex-controller", *campaign.controller_argv()]))
    return 0


def _preflight(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    report = preflight_campaign(
        campaign,
        require_backends=args.strict,
        require_model=args.require_model,
        verify_asset_hash=not args.skip_asset_hash,
        verify_backend_revisions=args.verify_backend_revisions,
        verify_checkpoint_content=args.verify_checkpoint_content,
    )
    if args.as_json:
        print(
            json.dumps(
                {
                    "schema_version": PREFLIGHT_OUTPUT_SCHEMA_VERSION,
                    "resolved_campaign": campaign.as_dict(),
                    "preflight": report.as_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_campaign(campaign))
        print("\nPreflight checks")
        print(render_preflight(report))
    return 0 if report.ok else 1


def _run(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    report = preflight_campaign(
        campaign,
        require_backends=not args.dry_run,
        require_model=args.require_model or not args.dry_run,
        verify_backend_revisions=args.verify_backend_revisions,
        verify_checkpoint_content=args.verify_checkpoint_content,
    )
    print(render_campaign(campaign))
    print("\nPreflight checks")
    print(render_preflight(report))
    if not report.ok:
        return 1
    if args.dry_run:
        print("\nDry run complete; no archive files or workers were created.")
        return 0

    with campaign_run_lock(campaign.config.run.archive_root):
        artifacts = write_campaign_artifacts(campaign)
        print("\nRecorded campaign input")
        print(f"  input     {artifacts.input_path}")
        print(f"  resolved  {artifacts.resolved_path}")

        with applied_environment(campaign.environment()):
            provenance = write_campaign_provenance(campaign, artifacts)
            print("\nRecorded run provenance")
            print(f"  provenance  {provenance.path}")
            print(f"  source SHA  {provenance.source_tree_sha256}")
            print(f"  model SHA   {provenance.model_content_sha256}")

            # Keep the environment for extension/legacy compatibility, while
            # the built-in controller receives the reviewed typed path snapshot.
            # The critic records the verified model digest on every LLM call.
            with applied_environment(
                {"TREX_MODEL_DIGEST": provenance.model_content_sha256}
            ):
                from .. import controller

                result = controller.main(
                    campaign.controller_argv(),
                    runtime_paths=campaign.runtime_paths(),
                )
    return int(result or 0)


def _status(args: argparse.Namespace) -> int:
    status = collect_campaign_status(args.archive_root)
    if args.as_json:
        print(json.dumps(status.as_dict(), indent=2, sort_keys=True))
    else:
        print(render_status(status))
    return 0
