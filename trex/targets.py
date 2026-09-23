"""Resolve, inspect, and scaffold T-REX target inputs.

The target registry is the single source of truth for bundled target labels.
Custom targets use the same resolver by supplying an explicit constraint JSON
and PDB, so launchers do not need a second hard-coded target table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .resource_paths import publication_data_path


TARGET_REGISTRY_SCHEMA = "trex_target_registry.v1"
RESOLVED_TARGET_SCHEMA_VERSION = "trex.resolved-target.v1"
TARGET_LIST_SCHEMA_VERSION = "trex.target-list.v1"
_HOTSPOT_RE = re.compile(r".(-?\d+)[A-Za-z]?")


def _default_repo_root() -> Path:
    return (
        Path(os.environ.get("TREX_REPO_ROOT", Path(__file__).resolve().parents[1]))
        .expanduser()
        .resolve()
    )


@dataclass(frozen=True)
class ResolvedTarget:
    """Concrete, launch-ready target paths and identity."""

    name: str
    target_id: str
    config_path: Path
    pdb_path: Path
    registered: bool

    def as_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = RESOLVED_TARGET_SCHEMA_VERSION
        payload["config_path"] = str(self.config_path)
        payload["pdb_path"] = str(self.pdb_path)
        return payload


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def load_target_registry(repo_root: Path | None = None) -> dict[str, dict[str, str]]:
    """Load and minimally validate the checked-in target registry."""

    root = (repo_root or _default_repo_root()).expanduser().resolve()
    path = publication_data_path(root, "config/targets/registry.json")
    payload = _load_json_object(path, label="target registry")
    if payload.get("schema_version") != TARGET_REGISTRY_SCHEMA:
        raise ValueError(
            f"unsupported target registry schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, dict):
        raise ValueError(f"target registry has no targets object: {path}")

    targets: dict[str, dict[str, str]] = {}
    for name, raw in raw_targets.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError(f"malformed target registry entry: {name!r}")
        entry = {
            key: str(raw.get(key, "")).strip() for key in ("config", "pdb", "target_id")
        }
        if not all(entry.values()):
            raise ValueError(f"incomplete target registry entry: {name}")
        for key in ("config", "pdb"):
            candidate = Path(entry[key])
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"unsafe {key} path for target {name}: {entry[key]!r}")
        targets[name] = entry
    return targets


def resolve_target(
    name: str,
    *,
    repo_root: Path | None = None,
    asset_root: Path | None = None,
    target_config: Path | None = None,
    target_pdb: Path | None = None,
) -> ResolvedTarget:
    """Resolve one registered or explicitly supplied custom target.

    Registered labels obtain their constraint and target ID from
    ``config/targets/registry.json``. A caller may pass explicit paths for a
    registered label, but normal validation will still enforce its registered
    identity and hash. Unknown labels require both explicit paths.
    """

    if not name.strip() or "\n" in name or "\r" in name:
        raise ValueError("target name must be a non-empty single line")
    root = (repo_root or _default_repo_root()).expanduser().resolve()
    registry_path = publication_data_path(root, "config/targets/registry.json")
    registry = load_target_registry(root) if registry_path.is_file() else {}
    entry = registry.get(name)
    registered = entry is not None

    if registered:
        assert entry is not None
        config = target_config or publication_data_path(
            root, Path("config/targets") / entry["config"]
        )
        if target_pdb is not None:
            pdb = target_pdb
        elif asset_root is not None:
            # Accept both the portable bundle root (T-REX-assets/) and the
            # target-only root written to .env.assets (T-REX-assets/targets/).
            direct = asset_root / entry["pdb"]
            nested = asset_root / "targets" / entry["pdb"]
            pdb = direct if direct.is_file() or not nested.is_file() else nested
        else:
            raise ValueError(
                f"registered target {name!r} needs --asset-root or --target-pdb"
            )
        expected_id = entry["target_id"]
    else:
        if target_config is None or target_pdb is None:
            raise ValueError(
                f"custom target {name!r} needs both --target-config and --target-pdb"
            )
        config = target_config
        pdb = target_pdb
        expected_id = None

    config = config.expanduser().resolve()
    pdb = pdb.expanduser().resolve()
    if any("\n" in str(path) or "\r" in str(path) for path in (config, pdb)):
        raise ValueError("target paths must not contain newlines")
    if not config.is_file():
        raise ValueError(f"target constraint not found: {config}")
    if not pdb.is_file():
        raise ValueError(f"target PDB not found: {pdb}")

    constraint = _load_json_object(config, label="target constraint")
    observed_id = str(constraint.get("target_id", "")).strip()
    if not observed_id:
        raise ValueError(f"target constraint has no target_id: {config}")
    if expected_id is not None and observed_id != expected_id:
        raise ValueError(
            f"registered target ID mismatch for {name}: "
            f"expected={expected_id} observed={observed_id}"
        )
    return ResolvedTarget(
        name=name,
        target_id=observed_id,
        config_path=config,
        pdb_path=pdb,
        registered=registered,
    )


def _write_target_template(args: argparse.Namespace) -> Path:
    out = Path(args.out).expanduser().resolve()
    if out.exists() and not args.force:
        raise ValueError(f"refusing to overwrite existing file: {out} (use --force)")
    chains = list(dict.fromkeys(args.chain))
    if not chains or any(len(chain) != 1 for chain in chains):
        raise ValueError("provide at least one one-character --chain")
    hotspots = list(dict.fromkeys(args.hotspot or []))
    for hotspot in hotspots:
        if not _HOTSPOT_RE.fullmatch(hotspot) or hotspot[0] not in chains:
            raise ValueError(
                f"hotspot {hotspot!r} must use a configured chain and PDB residue number"
            )
    payload = {
        "target_id": args.target_id,
        "target_class": args.target_class,
        "hotspots": hotspots,
        "forbidden_surfaces": [],
        "chain_ids": chains,
        "assay_geometry": None,
        "developability_filters": {},
        "panel_size_K": args.panel_size,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="list registered target labels")
    list_cmd.add_argument("--json", action="store_true", dest="as_json")

    resolve = sub.add_parser(
        "resolve", help="resolve a target to concrete launch inputs"
    )
    resolve.add_argument("target")
    resolve.add_argument("--asset-root", type=Path)
    resolve.add_argument("--target-config", type=Path)
    resolve.add_argument("--target-pdb", type=Path)
    resolve.add_argument("--format", choices=("json", "lines"), default="json")

    init = sub.add_parser("init", help="write a custom TargetConstraint template")
    init.add_argument("--out", required=True, type=Path)
    init.add_argument("--target-id", required=True)
    init.add_argument("--target-class", default="protein")
    init.add_argument("--chain", action="append", required=True)
    init.add_argument("--hotspot", action="append", default=[])
    init.add_argument("--panel-size", type=int, default=8)
    init.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "list":
            targets = load_target_registry(args.repo_root)
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "schema_version": TARGET_LIST_SCHEMA_VERSION,
                            "targets": targets,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                for name in sorted(targets):
                    print(f"{name}\t{targets[name]['target_id']}")
            return 0
        if args.command == "resolve":
            resolved = resolve_target(
                args.target,
                repo_root=args.repo_root,
                asset_root=args.asset_root,
                target_config=args.target_config,
                target_pdb=args.target_pdb,
            )
            if args.format == "lines":
                print(resolved.target_id)
                print(resolved.config_path)
                print(resolved.pdb_path)
                print("registered" if resolved.registered else "custom")
            else:
                print(json.dumps(resolved.as_json(), indent=2, sort_keys=True))
            return 0
        if args.command == "init":
            if args.panel_size < 1:
                raise ValueError("--panel-size must be positive")
            if not str(args.target_id).strip():
                raise ValueError("--target-id must be non-empty")
            out = _write_target_template(args)
            print(out)
            return 0
    except ValueError as exc:
        print(f"trex-target: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
