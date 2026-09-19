"""Single-writer lease for campaign artifact creation and controller startup."""

from __future__ import annotations

import fcntl
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


CAMPAIGN_RUN_LOCK_NAME = ".campaign_run.lock"


@contextmanager
def campaign_run_lock(archive_root: Path) -> Iterator[Path]:
    """Prevent concurrent launchers from racing provenance and resume files."""

    archive_root.mkdir(parents=True, exist_ok=True)
    lock_path = archive_root / CAMPAIGN_RUN_LOCK_NAME
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OSError(
                f"campaign archive already has an active launcher: {archive_root}"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (
                f"pid={os.getpid()} host={socket.gethostname()} "
                f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}\n"
            ).encode("utf-8"),
        )
        yield lock_path
    finally:
        os.close(descriptor)
