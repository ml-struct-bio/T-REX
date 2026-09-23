#!/usr/bin/env python3
"""Check the captured controller/LLM environment, including its known exceptions."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def verify_environment(lock_bytes, profile, distributions, environment):
    """Compare installed metadata to the freeze without resolving new versions."""
    failures = []
    if hashlib.sha256(lock_bytes).hexdigest() != profile["lock_sha256"]:
        failures.append("dependency inventory SHA256 differs from the recorded file")
    for field, key in (
        ("python_full_version", "python"),
        ("sys_platform", "platform"),
        ("platform_machine", "machine"),
    ):
        if environment[field] != profile[key]:
            failures.append(
                f"{field}: expected {profile[key]}, got {environment[field]}"
            )
    installed = {canonicalize_name(d.metadata["Name"]): d for d in distributions}
    pins = [
        Requirement(line)
        for line in lock_bytes.decode().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for pin in pins:
        dist = installed.get(canonicalize_name(pin.name))
        if dist is None:
            failures.append(f"missing package: {pin.name}")
            continue
        if pin.url:
            url, revision = pin.url.removeprefix("git+").rsplit("@", 1)
            origin = json.loads(dist.read_text("direct_url.json") or "{}")
            vcs = origin.get("vcs_info", {})
            if (
                origin.get("url") != url
                or vcs.get("vcs") != "git"
                or vcs.get("commit_id") != revision
            ):
                failures.append(
                    f"{pin.name}: installed source does not match {pin.url}"
                )
        elif not pin.specifier.contains(dist.version, prereleases=True):
            failures.append(f"{pin.name}: expected {pin.specifier}, got {dist.version}")

    observed_exceptions = []
    for name, dist in sorted(installed.items()):
        for raw in dist.requires or []:
            required = Requirement(raw)
            if required.marker and not required.marker.evaluate(
                {**environment, "extra": ""}
            ):
                continue
            dependency = installed.get(canonicalize_name(required.name))
            if dependency is None:
                failures.append(f"{name}: missing required dependency {required}")
            elif required.specifier and not required.specifier.contains(
                dependency.version, prereleases=True
            ):
                observed_exceptions.append(
                    {
                        "package": name,
                        "requirement": str(required),
                        "installed_version": dependency.version,
                    }
                )

    expected = profile["recorded_dependency_exceptions"]
    for item in observed_exceptions:
        if item not in expected:
            failures.append(f"unexpected dependency conflict: {item}")
    for item in expected:
        if item not in observed_exceptions:
            failures.append(f"recorded dependency exception is absent: {item}")
    return {
        "matches_study_inventory": not failures,
        "python": environment["python_full_version"],
        "locked_packages": len(pins),
        "recorded_dependency_exceptions": observed_exceptions,
        "failures": failures,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args(argv)
    profile = json.loads(
        (
            args.repo_root / "config/reproducibility/controller_environment.json"
        ).read_text()
    )
    report = verify_environment(
        (args.repo_root / profile["lock"]).read_bytes(),
        profile,
        metadata.distributions(),
        default_environment(),
    )
    report["executable"] = sys.executable
    report["machine"] = platform.machine()
    print(json.dumps(report, indent=2))
    return 0 if report["matches_study_inventory"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
