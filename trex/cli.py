"""Top-level T-REX command-line interface."""

from __future__ import annotations

import argparse
import sys

from .campaign.cli import add_campaign_parser
from .campaign.config import ConfigError
from .public_cli import add_public_parsers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trex",
        description="Configure, run, and inspect T-REX binder-design campaigns.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_public_parsers(subparsers)
    add_campaign_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.command_handler(args))
    except (ConfigError, OSError, ValueError) as exc:
        print(f"trex: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
