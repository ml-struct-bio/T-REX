#!/usr/bin/env python3
"""Zero-dependency structural and publication-artifact checks."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
LOWER_LAYER_PACKAGES = ("backends", "evidence", "execution", "selection", "tick")
FORBIDDEN_DEPENDENCIES = {"trex.controller", "trex.live_tick"}
PUBLICATION_DATA_DIRECTORIES = (
    "config/targets",
    "config/trex",
    "config/reproducibility",
    "external/patches",
)


def _absolute_import(module_path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    relative = module_path.relative_to(REPOSITORY_ROOT).with_suffix("")
    package_parts = list(relative.parts[:-1])
    keep = max(0, len(package_parts) - (node.level - 1))
    prefix = package_parts[:keep]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def dependency_violations() -> list[str]:
    violations: list[str] = []
    for package in LOWER_LAYER_PACKAGES:
        for path in sorted((REPOSITORY_ROOT / "trex" / package).rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            imported: list[tuple[int, str]] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend((node.lineno, alias.name) for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append((node.lineno, _absolute_import(path, node)))
            for line_number, module_name in imported:
                if any(
                    module_name == forbidden or module_name.startswith(forbidden + ".")
                    for forbidden in FORBIDDEN_DEPENDENCIES
                ):
                    relative = path.relative_to(REPOSITORY_ROOT)
                    violations.append(f"{relative}:{line_number}: imports {module_name}")
    return violations


def temporary_artifacts() -> list[str]:
    return sorted(
        str(path.relative_to(REPOSITORY_ROOT))
        for suffix in ("*.orig", "*.rej")
        for path in REPOSITORY_ROOT.rglob(suffix)
    )


def publication_data_violations() -> list[str]:
    violations: list[str] = []
    packaged_root = REPOSITORY_ROOT / "trex" / "data"
    for relative_directory in PUBLICATION_DATA_DIRECTORIES:
        source_directory = REPOSITORY_ROOT / relative_directory
        packaged_directory = packaged_root / relative_directory
        for source_path in sorted(source_directory.glob("*")):
            if not source_path.is_file():
                continue
            packaged_path = packaged_directory / source_path.name
            if not packaged_path.is_file():
                violations.append(f"missing packaged publication data: {relative_directory}/{source_path.name}")
            elif packaged_path.read_bytes() != source_path.read_bytes():
                violations.append(f"stale packaged publication data: {relative_directory}/{source_path.name}")
    source_snapshot = REPOSITORY_ROOT / "docs" / "all_prompts_snapshot.txt"
    packaged_snapshot = packaged_root / "docs" / "all_prompts_snapshot.txt"
    if not packaged_snapshot.is_file():
        violations.append("missing packaged publication data: docs/all_prompts_snapshot.txt")
    elif packaged_snapshot.read_bytes() != source_snapshot.read_bytes():
        violations.append("stale packaged publication data: docs/all_prompts_snapshot.txt")
    return violations


def main() -> int:
    errors = dependency_violations()
    errors.extend(publication_data_violations())
    errors.extend(f"temporary patch artifact: {path}" for path in temporary_artifacts())

    os.environ.pop("TREX_CROSS_CAMPAIGN_MEMORY", None)
    from trex.prompt_catalog import render_prompt_catalog

    prompt_snapshot = REPOSITORY_ROOT / "docs" / "all_prompts_snapshot.txt"
    expected_prompt_snapshot = render_prompt_catalog(
        ("planner", "supervisor", "critic")
    )
    if not prompt_snapshot.is_file():
        errors.append(f"missing prompt snapshot: {prompt_snapshot}")
    elif prompt_snapshot.read_text() != expected_prompt_snapshot:
        errors.append(
            "docs/all_prompts_snapshot.txt is stale; regenerate with "
            "python docs/all_prompts.py"
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(
        "Architecture checks passed: dependency direction, publication data, "
        "prompt snapshot, and patch-artifact hygiene."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
