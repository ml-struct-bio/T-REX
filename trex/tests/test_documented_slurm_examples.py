"""Check that documented optional submissions preserve the caller's settings."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SLURM_SUBMISSIONS = [
    block
    for block in re.findall(r"```bash\n(.*?)```", (ROOT / "docs/slurm.md").read_text(), re.S)
    if "sbatch " in block
]


def test_documented_slurm_submissions_exist():
    assert SLURM_SUBMISSIONS


@pytest.mark.parametrize("block", SLURM_SUBMISSIONS)
def test_submission_examples_keep_overrides_local(tmp_path, block):
    # No scheduler is contacted. The shell function absorbs every submission.
    environment = {
        **os.environ,
        "TARGET": "fixture",
        "TREX_MAX_WALL_H": "48.0",
        "TREX_WORKER_GPUS": "1,2,3",
        "TREX_REQUIRE_THREE_WORKERS": "1",
        "TREX_CHARGED_GPUS": "4",
        "TREX_RESUME_ARCHIVE": "",
    }
    script = (
        "set -eu\n"
        "sbatch() { :; }\n"
        + block
        + '\ntest "$TARGET" = fixture\n'
        + 'test "$TREX_MAX_WALL_H" = 48.0\n'
        + 'test "$TREX_WORKER_GPUS" = 1,2,3\n'
        + 'test "$TREX_REQUIRE_THREE_WORKERS" = 1\n'
        + 'test "$TREX_CHARGED_GPUS" = 4\n'
        + 'test -z "$TREX_RESUME_ARCHIVE"\n'
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
