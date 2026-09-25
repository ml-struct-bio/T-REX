"""Reproducible, resumable installation of the T-REX campaign stack."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path


PYTHON = "3.12.13"
BINDCRAFT_PYTHON = "3.10.20"
UV_VERSION = "0.11.1"
PYROSETTA_INDEX = (
    "https://west.rosettacommons.org/pyrosetta/quarterly/"
    "release.cxx11thread.serialization"
)
MANAGED_HEADER = "# Managed by `trex setup`; rerun that command to regenerate.\n"


class SetupError(ValueError):
    """An actionable installation error."""


def _run(command: list[object], *, cwd: Path | None = None, env=None, capture=False):
    argv = [str(part) for part in command]
    print("+ " + shlex.join(argv), flush=True)
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            capture_output=capture,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise SetupError(f"command failed ({shlex.join(argv)}){suffix}") from exc


def _capture(command: list[object], *, cwd: Path | None = None) -> str:
    return _run(command, cwd=cwd, capture=True).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_source_checkout(root: Path) -> None:
    required = (
        root / "pyproject.toml",
        root / "scripts/manage_assets.py",
        root / "config/reproducibility/production_stack.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SetupError(
            "trex setup must run from an editable T-REX source checkout; missing "
            + ", ".join(missing)
        )


def _require_host_tools(uv: str) -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SetupError("the verified installation profile requires Linux x86_64")
    missing = [
        name
        for name in ("git", "git-lfs", "cmake", "c++", "rustc")
        if not shutil.which(name)
    ]
    if missing:
        raise SetupError(
            "install these system build tools first: " + ", ".join(missing)
        )
    observed = _capture([uv, "--version"]).split()
    if len(observed) < 2 or observed[1] != UV_VERSION:
        got = observed[1] if len(observed) >= 2 else "unknown"
        raise SetupError(f"expected uv {UV_VERSION}; got {got}")


def _git_head(path: Path) -> str:
    return _capture(["git", "-C", path, "rev-parse", "HEAD"])


def _tracked_diff_sha(path: Path) -> str | None:
    diff = _capture(["git", "-C", path, "diff", "HEAD", "--binary", "--no-ext-diff"])
    return hashlib.sha256(diff.encode()).hexdigest() if diff else None


def _checkout(path: Path, repository: str, commit: str, *, recursive=False) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        command: list[object] = ["git", "clone"]
        if recursive:
            command.append("--recursive")
        command.extend([repository, path])
        environment = os.environ.copy()
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
        _run(command, env=environment)
    if not (path / ".git").exists():
        raise SetupError(f"existing backend path is not a Git checkout: {path}")
    head = _git_head(path)
    if head != commit:
        changed = _capture(
            ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"]
        )
        if changed:
            raise SetupError(
                f"{path} has tracked changes at {head}; preserve them or use a clean path"
            )
        _run(["git", "-C", path, "fetch", "origin", commit])
        _run(["git", "-C", path, "checkout", "--detach", commit])
    if recursive:
        _run(["git", "-C", path, "submodule", "update", "--init", "--recursive"])


def _prepare_sources(root: Path, stack: dict) -> dict[str, Path]:
    components = stack["components"]
    paths = {
        "proteina_complexa": root / "external/Proteina-Complexa",
        "bindcraft": root / "external/BindCraft",
        "boltzgen": root / "external/BoltzGen",
        "foldseek": root / "external/Foldseek",
        "mmseqs2": root / "external/MMseqs2",
    }
    for name, path in paths.items():
        component = components[name]
        _checkout(
            path,
            component["repository"],
            component.get("git_commit") or component["version"],
            recursive=name in {"foldseek", "mmseqs2"},
        )

    bindcraft = paths["bindcraft"]
    expected = components["bindcraft"]["production_tracked_diff_sha256"]
    observed = _tracked_diff_sha(bindcraft)
    if observed != expected:
        if observed is not None:
            raise SetupError(
                "BindCraft has an unrecognized tracked diff; preserve it or use a clean checkout"
            )
        patch = root / components["bindcraft"]["patch"]
        if _sha256(patch) != components["bindcraft"]["patch_file_sha256"]:
            raise SetupError(f"BindCraft patch SHA256 mismatch: {patch}")
        _run(["git", "-C", bindcraft, "apply", "--check", patch])
        _run(["git", "-C", bindcraft, "apply", patch])
        if _tracked_diff_sha(bindcraft) != expected:
            raise SetupError(
                "BindCraft production patch did not produce the recorded diff"
            )
    return paths


def _python_version(python: Path) -> str:
    return _capture(
        [python, "-I", "-c", "import platform; print(platform.python_version())"]
    )


def _ensure_venv(uv: str, path: Path, version: str) -> Path:
    python = path / "bin/python"
    if not python.is_file():
        if path.exists() and any(path.iterdir()):
            raise SetupError(f"refusing to replace a non-environment directory: {path}")
        _run([uv, "venv", "--python", version, path])
    observed = _python_version(python)
    if observed != version:
        raise SetupError(f"{path} uses Python {observed}; expected {version}")
    return python


def _install_serving(root: Path, uv: str) -> Path:
    python = _ensure_venv(uv, root / ".venv-serving", PYTHON)
    lock = root / "config/trex/serving_requirements.lock.txt"
    _run([uv, "pip", "sync", "--python", python, lock])
    _run([uv, "pip", "install", "--python", python, "--no-deps", "-e", root])
    _run([uv, "pip", "check", "--python", python])
    return python


def _install_complexa(root: Path, uv: str, repository: Path) -> Path:
    script = root / "scripts/setup_complexa_env.py"
    environment = repository / ".venv"
    _run(
        [
            os.sys.executable,
            script,
            "--repo",
            repository,
            "--env",
            environment,
            "--python",
            PYTHON,
            "--uv",
            uv,
        ],
        cwd=root,
    )
    return environment / "bin/python"


def _install_bindcraft(root: Path, uv: str, repository: Path, index: str) -> Path:
    python = _ensure_venv(uv, repository / ".venv", BINDCRAFT_PYTHON)
    lock = root / "config/trex/bindcraft_requirements.lock.txt"
    _run(
        [
            uv,
            "pip",
            "sync",
            "--python",
            python,
            lock,
            "--find-links",
            index,
        ]
    )
    _run([uv, "pip", "check", "--python", python])
    _run(
        [
            os.sys.executable,
            root / "scripts/manage_assets.py",
            "verify-bindcraft",
            "--python",
            python,
        ],
        cwd=root,
    )
    return python


def _install_boltzgen(root: Path, uv: str, repository: Path) -> Path:
    python = _ensure_venv(uv, repository / ".venv", PYTHON)
    lock = root / "config/trex/boltzgen_requirements.lock.txt"
    _run([uv, "pip", "sync", "--python", python, lock])
    _run([uv, "pip", "install", "--python", python, "--no-deps", "-e", repository])
    _run([uv, "pip", "check", "--python", python])
    executable = repository / ".venv/bin/boltzgen"
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise SetupError(f"BoltzGen executable was not installed: {executable}")
    return python


def _tool_version(executable: Path) -> str:
    if not executable.is_file() or not os.access(executable, os.X_OK):
        return ""
    try:
        return _capture([executable, "version"])
    except SetupError:
        return ""


def _install_tool(
    source: Path,
    prefix: Path,
    executable_name: str,
    expected_version: str,
    jobs: int,
) -> Path:
    executable = prefix / "bin" / executable_name
    if expected_version in _tool_version(executable):
        print(f"Verified {executable_name} {expected_version}", flush=True)
        return executable
    build = source / "build-trex"
    _run(
        [
            "cmake",
            "-S",
            source,
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            "-DHAVE_MPI=0",
            "-DHAVE_TESTS=0",
        ]
    )
    _run(["cmake", "--build", build, "--parallel", jobs])
    _run(["cmake", "--install", build])
    observed = _tool_version(executable)
    if expected_version not in observed:
        raise SetupError(
            f"{executable_name} version mismatch: expected {expected_version}; got {observed}"
        )
    return executable


def _assets_complete(root: Path, manifest: dict) -> bool:
    for component in manifest["components"].values():
        for entry in component["files"]:
            path = root / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["size"]:
                return False
    return True


def _prepare_assets(
    root: Path,
    asset_root: Path,
    source_paths: dict[str, Path],
    *,
    drive_url: str,
    asset_zip: Path | None,
) -> None:
    script = root / "scripts/manage_assets.py"
    manifest_path = root / "config/reproducibility/assets_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not _assets_complete(asset_root, manifest):
        if asset_zip is not None:
            _run(
                [
                    os.sys.executable,
                    script,
                    "import-zip",
                    "--root",
                    asset_root,
                    "--archive",
                    asset_zip,
                ],
                cwd=root,
            )
        else:
            _run(
                [
                    os.sys.executable,
                    script,
                    "fetch",
                    "--root",
                    asset_root,
                    "--drive-url",
                    drive_url,
                ],
                cwd=root,
            )
    _run(
        [
            os.sys.executable,
            script,
            "configure",
            "--root",
            asset_root,
            "--complexa-repo",
            source_paths["proteina_complexa"],
            "--community-repo",
            source_paths["proteina_complexa"],
            "--bindcraft-repo",
            source_paths["bindcraft"],
            "--env-out",
            root / ".env.assets",
        ],
        cwd=root,
    )


def _managed_write(path: Path, content: str) -> None:
    if path.exists():
        existing = path.read_text()
        if existing == content:
            return
        if not existing.startswith(MANAGED_HEADER):
            raise SetupError(
                f"refusing to overwrite existing configuration: {path}; preserve or rename it"
            )
    path.write_text(content)


def _complexa_environment(
    repository: Path,
    asset_root: Path,
    foldseek: Path,
    mmseqs: Path,
    bindcraft: Path,
) -> str:
    values = {
        "LOCAL_CODE_PATH": repository,
        "LOCAL_DATA_PATH": asset_root / "targets",
        "LOCAL_CACHE_DIR": repository / ".cache",
        "LOCAL_CHECKPOINT_PATH": repository / "ckpts",
        "COMMUNITY_MODELS_PATH": repository / "community_models",
        "ESM_DIR": repository / "community_models/ckpts/ESM2",
        "AF2_DIR": repository / "community_models/ckpts/AF2",
        "UV_VENV": repository / ".venv",
        "UV_FOLDSEEK_EXEC": foldseek,
        "UV_MMSEQS_EXEC": mmseqs,
        "UV_DSSP_EXEC": bindcraft / "functions/dssp",
        "UV_SC_EXEC": bindcraft / "functions/DAlphaBall.gcc",
        "FOLDSEEK_EXEC": foldseek,
        "MMSEQS_EXEC": mmseqs,
        "DSSP_EXEC": bindcraft / "functions/dssp",
        "SC_EXEC": bindcraft / "functions/DAlphaBall.gcc",
        "DATA_PATH": asset_root / "targets",
        "CKPT_PATH": repository / "ckpts",
    }
    lines = [MANAGED_HEADER.rstrip(), "USE_V2_COMPLEXA_ARCH=False"]
    lines.extend(f"{key}={shlex.quote(str(value))}" for key, value in values.items())
    return "\n".join(lines) + "\n"


def _configure_runtime(
    root: Path,
    asset_root: Path,
    paths: dict[str, Path],
    foldseek: Path,
    mmseqs: Path,
) -> None:
    complexa = paths["proteina_complexa"]
    backend_env = _complexa_environment(
        complexa, asset_root, foldseek, mmseqs, paths["bindcraft"]
    )
    _managed_write(complexa / ".env", backend_env)
    _run(
        [complexa / ".venv/bin/complexa", "init", "uv", "--force"],
        cwd=complexa,
    )
    profile = root / ".env"
    example = root / ".env.example"
    if not profile.exists():
        shutil.copy2(example, profile)
        print(f"Created {profile}", flush=True)
    if not profile.is_file():
        raise SetupError(f"installation profile is not a regular file: {profile}")


def setup_installation(
    *,
    root: Path,
    asset_root: Path,
    drive_url: str | None,
    asset_zip: Path | None,
    jobs: int,
    pyrosetta_index: str,
) -> None:
    """Install and configure every component needed by the default campaign."""
    root = root.expanduser().resolve()
    asset_root = asset_root.expanduser().resolve()
    asset_zip = asset_zip.expanduser().resolve() if asset_zip else None
    _require_source_checkout(root)
    uv = shutil.which("uv")
    if not uv:
        raise SetupError(f"install uv=={UV_VERSION} and place it on PATH")
    _require_host_tools(uv)
    if jobs < 1:
        raise SetupError("--jobs must be at least 1")

    stack = json.loads(
        (root / "config/reproducibility/production_stack.json").read_text()
    )
    release = json.loads(
        (root / "config/reproducibility/assets_release.json").read_text()
    )
    selected_drive_url = drive_url or release["google_drive_url"]

    print("\n[1/7] Preparing pinned backend sources", flush=True)
    paths = _prepare_sources(root, stack)
    print("\n[2/7] Fetching, verifying and linking assets", flush=True)
    _prepare_assets(
        root,
        asset_root,
        paths,
        drive_url=selected_drive_url,
        asset_zip=asset_zip,
    )
    print("\n[3/7] Installing local LLM/controller environment", flush=True)
    _install_serving(root, uv)
    print("\n[4/7] Installing Complexa/AF2/ProteinMPNN environment", flush=True)
    _install_complexa(root, uv, paths["proteina_complexa"])
    print("\n[5/7] Building the recorded Foldseek and MMseqs2 revisions", flush=True)
    prefix = paths["proteina_complexa"] / ".venv"
    components = stack["components"]
    foldseek = _install_tool(
        paths["foldseek"],
        prefix,
        "foldseek",
        components["foldseek"]["version"],
        jobs,
    )
    mmseqs = _install_tool(
        paths["mmseqs2"],
        prefix,
        "mmseqs",
        components["mmseqs2"]["version"],
        jobs,
    )
    print("\n[6/7] Installing BindCraft and BoltzGen environments", flush=True)
    _install_bindcraft(root, uv, paths["bindcraft"], pyrosetta_index)
    _install_boltzgen(root, uv, paths["boltzgen"])
    print("\n[7/7] Writing runtime configuration", flush=True)
    _configure_runtime(root, asset_root, paths, foldseek, mmseqs)
    print(
        "\nT-REX setup completed. Create a campaign with `trex init`, then run "
        "`trex check` before submission.",
        flush=True,
    )
