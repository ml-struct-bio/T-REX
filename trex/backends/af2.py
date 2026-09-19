"""AF2 refilter launch construction.

The controller owns scheduling and process lifecycle. This module owns the
scientific command contract so it can be reviewed and tested independently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .process_management import WorkerLaunch, isolated_subprocess_environment


def _launch_seed(candidate_id: str, campaign_seed: int) -> int:
    seed_key = f"{candidate_id}|af2_refilter|s{campaign_seed}".encode()
    return int(hashlib.sha256(seed_key).hexdigest()[:8], 16) % 1_000_000


def prepare_af2_refilter_launch(
    *,
    output_dir: Path,
    parent_pdb: Path,
    target_chain: str,
    binder_chain: str,
    gpu_id: str,
    python: Path,
    community_root: Path,
    af2_data_dir: Path,
    trex_repo_root: Path,
    candidate_id: str,
    campaign_seed: int,
    config_delta: Mapping[str, object] | None,
) -> WorkerLaunch:
    """Resolve one reproducible AF2-multimer refilter command."""

    delta = config_delta or {}
    model_names = str(delta.get("model_names", "model_1_multimer_v3"))
    # AF2 multimer needs recycles to converge. The historical value of zero
    # yielded unusable folds, while Complexa's canonical AF2 route uses three.
    num_recycles = int(delta.get("num_recycles", 3))
    argv = (
        str(python),
        "-m",
        "trex.af2_refilter_runner",
        "--input-pdb",
        str(parent_pdb),
        "--out-dir",
        str(output_dir),
        "--community-root",
        str(community_root),
        "--af2-data-dir",
        str(af2_data_dir),
        "--target-chain",
        target_chain,
        "--binder-chain",
        binder_chain,
        "--model-names",
        model_names,
        "--num-recycles",
        str(num_recycles),
        "--use-initial-guess",
        str(int(delta.get("use_initial_guess", 1))),
        "--seed",
        str(_launch_seed(candidate_id, campaign_seed)),
        "--use-multimer",
    )
    environment = isolated_subprocess_environment(python.parent.parent)
    environment["PYTHONPATH"] = (
        f"{trex_repo_root}:" + environment.get("PYTHONPATH", "")
    )
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return WorkerLaunch(
        argv=argv,
        output_dir=output_dir,
        environment=environment,
    )


__all__ = ["prepare_af2_refilter_launch"]
