"""Shared subprocess lifecycle and environment management for backends."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkerLaunch:
    """A fully resolved, inspectable command for one backend worker."""

    argv: tuple[str, ...]
    output_dir: Path
    environment: dict[str, str]
    cwd: Path | None = None


def popen_kwargs() -> dict[str, Any]:
    """Launch every worker in an isolated process group."""

    return {"start_new_session": True}


def terminate_process_group(proc: Any) -> None:
    """Terminate a worker tree, escalating to SIGKILL after a grace period."""

    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def isolated_subprocess_environment(env_root: Path) -> dict[str, str]:
    """Build an environment that protects backend-bundled CUDA libraries.

    Cluster modules can inject a different CUDA runtime through
    ``LD_LIBRARY_PATH``. BindCraft, BoltzGen, and ProteinMPNN ship their own
    runtime libraries, so backend processes receive only their environment's
    library directory while retaining all unrelated campaign variables.
    """

    env = os.environ.copy()
    env["PATH"] = f"{env_root}/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{env_root}/lib"
    if not env.get("MPLCONFIGDIR"):
        temporary_root = Path(env.get("TMPDIR", "/tmp"))
        env["MPLCONFIGDIR"] = str(
            temporary_root / f"v7_mplconfig_{os.getuid()}"
        )
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


__all__ = [
    "WorkerLaunch",
    "isolated_subprocess_environment",
    "popen_kwargs",
    "terminate_process_group",
]
