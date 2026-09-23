#!/usr/bin/env python3
"""Reject private runtime files and external checkouts in built distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def unexpected_members(names: list[str], *, wheel: bool) -> list[str]:
    unexpected = []
    for name in names:
        path = PurePosixPath(name)
        parts = path.parts
        private = (
            path.is_absolute()
            or ".." in parts
            or any(part in {".git", "__pycache__", "slurm_logs", "internal", "audit", "reviews"}
                   or part.startswith(".venv") for part in parts)
            or (path.name.startswith(".env") and path.name != ".env.example")
            or path.suffix in {".pyc", ".pyo"}
        )
        external_checkout = (
            bool(parts) and parts[0] == "external"
            and name != "external/README.md"
            and not (len(parts) == 3 and parts[1] == "patches" and path.suffix == ".patch")
        )
        if private or external_checkout or (wheel and name.startswith("trex/tests/")):
            unexpected.append(name)
    return unexpected


def check_distribution(path: Path) -> dict[str, object]:
    wheel = path.suffix == ".whl"
    if wheel:
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"Corrupt wheel member: {corrupt}")
            names = [info.filename for info in archive.infolist() if not info.is_dir()]
    else:
        with tarfile.open(path) as archive:
            names = [member.name.partition("/")[2] for member in archive.getmembers()
                     if member.isfile() or member.issym() or member.islnk()]
    unexpected = unexpected_members(names, wheel=wheel)
    if unexpected:
        raise ValueError(f"{path.name}: unexpected distribution files: {unexpected}")
    return {"file": path.name, "members": len(names), "bytes": path.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    args = parser.parse_args()
    wheels = list(args.dist_dir.glob("*.whl"))
    sources = list(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        parser.error("expected exactly one wheel and one source distribution")
    reports = [check_distribution(path) for path in wheels + sources]
    print(json.dumps({"ok": True, "distributions": reports}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
