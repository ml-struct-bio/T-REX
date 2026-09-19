"""ProteinMPNN sequence-redesign launch construction."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .process_management import WorkerLaunch, isolated_subprocess_environment


def _launch_seed(candidate_id: str, campaign_seed: int) -> int:
    seed_key = f"{candidate_id}|mpnn|s{campaign_seed}".encode()
    return int(hashlib.sha256(seed_key).hexdigest()[:8], 16) % 1_000_000


def prepare_proteinmpnn_launch(
    *,
    output_dir: Path,
    parent_pdb: Path,
    binder_chain: str,
    gpu_id: str,
    python: Path,
    proteinmpnn_dir: Path,
    candidate_id: str,
    campaign_seed: int,
    config_delta: Mapping[str, object] | None,
) -> WorkerLaunch:
    """Resolve one deterministic ProteinMPNN redesign command."""

    delta = config_delta or {}
    argv = [
        str(python),
        str(proteinmpnn_dir / "protein_mpnn_run.py"),
        "--pdb_path",
        str(parent_pdb),
        "--pdb_path_chains",
        binder_chain or "B",
        "--out_folder",
        str(output_dir),
        "--num_seq_per_target",
        str(int(delta.get("num_seq_per_target", 8))),
        "--batch_size",
        "1",
        "--sampling_temp",
        str(delta.get("sampling_temp", 0.1)),
        "--seed",
        str(_launch_seed(candidate_id, campaign_seed)),
        "--model_name",
        str(delta.get("model_name", "v_48_020")),
        "--backbone_noise",
        str(float(delta.get("backbone_noise", 0.0))),
        "--path_to_model_weights",
        str(proteinmpnn_dir / "vanilla_model_weights"),
        "--suppress_print",
        "1",
    ]
    if "omit_AAs" in delta:
        argv.extend(("--omit_AAs", str(delta["omit_AAs"])))
    environment = isolated_subprocess_environment(python.parent.parent)
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return WorkerLaunch(
        argv=tuple(argv),
        output_dir=output_dir,
        environment=environment,
    )


__all__ = ["prepare_proteinmpnn_launch"]
