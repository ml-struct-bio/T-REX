#!/usr/bin/env python3
"""Install the verified Complexa dependency profile in a separate environment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/reproducibility/complexa_environment.json"
MARKER = ".trex-complexa-environment.json"


def run(command, **kwargs):
    print("+ " + shlex.join([str(x) for x in command]), flush=True)
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def checked_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Profile file SHA256 mismatch: {path}")
    return data


def check_destination(environment, repository):
    """Never sync over an unrelated existing environment."""
    if environment.is_symlink():
        raise ValueError("Environment directory must not be a symlink")
    if environment.exists() and any(environment.iterdir()):
        marker = environment / MARKER
        if not marker.is_file():
            raise ValueError(
                "Choose a new environment directory; existing unmarked environments are preserved"
            )
        previous = json.loads(marker.read_text())
        if previous.get("repository") != str(repository):
            raise ValueError(
                "Existing environment belongs to a different source checkout"
            )


def prepare_source(repository, profile):
    head = run(
        ["git", "-C", repository, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if head != profile["git_commit"]:
        raise ValueError(
            f"Expected Complexa revision {profile['git_commit']}; got {head}"
        )
    # LFS datasets/images are outside the dependency check; code and config
    # remain checked. Compute nodes need no git-lfs for this source check.
    changed = run(
        [
            "git",
            "-C",
            repository,
            "diff",
            "HEAD",
            "--name-only",
            "--",
            ".",
            ":(exclude,attr:filter=lfs)",
        ],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if set(changed) - {"pyproject.toml"}:
        raise ValueError(
            "Tracked backend changes outside dependency metadata; use a clean checkout"
        )
    project = repository / "pyproject.toml"
    current = project.read_bytes()
    if hashlib.sha256(current).hexdigest() == profile["patched_pyproject_sha256"]:
        return
    original = run(
        ["git", "-C", repository, "show", "HEAD:pyproject.toml"], capture_output=True
    ).stdout
    if current != original:
        raise ValueError(
            "Unrecognized pyproject.toml edits; refusing to overwrite them"
        )
    patch = ROOT / profile["dependency_patch"]
    checked_bytes(patch, profile["dependency_patch_sha256"])
    run(["git", "-C", repository, "apply", "--check", patch])
    run(["git", "-C", repository, "apply", patch])
    checked_bytes(project, profile["patched_pyproject_sha256"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, required=True, help="Pinned public Complexa checkout"
    )
    parser.add_argument(
        "--env",
        type=Path,
        required=True,
        help="New environment, or one created by this installer",
    )
    parser.add_argument(
        "--python",
        default="3.12.13",
        help="Python version or interpreter on compute-node-accessible storage",
    )
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args(argv)
    try:
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise ValueError("This verified profile requires Linux x86_64")
        profile = json.loads(PROFILE.read_text())
        lock = ROOT / profile["lock"]
        checked_bytes(lock, profile["lock_sha256"])
        checked_bytes(
            ROOT / profile["dependency_patch"], profile["dependency_patch_sha256"]
        )
        uv = shutil.which(args.uv)
        if not uv:
            raise ValueError(f"Install uv=={profile['uv']} first")
        version = run([uv, "--version"], capture_output=True, text=True).stdout.split()[
            1
        ]
        if version != profile["uv"]:
            raise ValueError(f"Expected uv {profile['uv']}; got {version}")
        repository = args.repo.resolve()
        environment = args.env.absolute()
        check_destination(environment, repository)
        prepare_source(repository, profile)
        marker = environment / MARKER
        record = {
            "repository": str(repository),
            "lock_sha256": profile["lock_sha256"],
            "status": "installing",
        }
        if not marker.exists():
            run([uv, "venv", "--python", args.python, environment])
            marker.write_text(json.dumps(record, indent=2) + "\n")
        python = environment / "bin/python"
        observed = run(
            [python, "-I", "-c", "import platform; print(platform.python_version())"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed != profile["python"]:
            raise ValueError(f"Expected Python {profile['python']}; got {observed}")
        run(
            [
                uv,
                "pip",
                "sync",
                "--python",
                python,
                lock,
                "--index-strategy",
                "unsafe-best-match",
            ]
        )
        run(
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "-c",
                lock,
                "-e",
                repository,
                "-e",
                repository / "community_models/colabdesign",
                "--index-strategy",
                "unsafe-best-match",
            ]
        )
        run([uv, "pip", "check", "--python", python])
        run(
            [python, ROOT / "scripts/check_complexa_env.py", "--repo", repository],
            cwd=repository,
        )
        record["status"] = "dependency-and-import-checks-passed"
        marker.write_text(json.dumps(record, indent=2) + "\n")
        print(f"Installed and checked: {python}")
        print(
            "Run the documented GPU check on a compute node before using this environment."
        )
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Complexa environment setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
