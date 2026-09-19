"""Deterministic command construction for built-in Complexa families."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path
from typing import Mapping


ALGORITHM_BY_FAMILY = {
    "complexa_beam": "beam-search",
    "complexa_best_of_n": "best-of-n",
    "complexa_fk_steering": "fk-steering",
    "complexa_mcts": "mcts",
    "complexa_single_pass": "single-pass",
}

NAMESPACE_BY_ALGORITHM = {
    "beam-search": "beam_search",
    "fk-steering": "fk_steering",
    "mcts": "mcts",
    "best-of-n": "best_of_n",
    "single-pass": None,
}


def refinement_overrides(config_delta: Mapping[str, object]) -> list[str]:
    """Return internally consistent sequence-refinement Hydra overrides."""

    overrides: list[str] = []
    greedy_requested = (
        int(config_delta.get("n_greedy_iters", 0) or 0) > 0
        or bool(config_delta.get("enable_greedy_optimization", False))
        or "greedy_percentage" in config_delta
    )
    refinement_algorithm = str(
        config_delta.get("refinement_algorithm", "") or ""
    )
    if greedy_requested and refinement_algorithm != "sequence_hallucination":
        refinement_algorithm = "sequence_hallucination"
        print(
            "  [complexa] enabling sequence_hallucination because greedy "
            "knobs were requested",
            flush=True,
        )
    if "refinement_algorithm" in config_delta or greedy_requested:
        overrides.append(
            "++generation.refinement.algorithm="
            f"{refinement_algorithm or 'null'}"
        )
    if refinement_algorithm == "sequence_hallucination":
        if "n_greedy_iters" in config_delta:
            overrides.append(
                "++generation.refinement.n_greedy_iters="
                f"{int(config_delta['n_greedy_iters'])}"
            )
        if "enable_greedy_optimization" in config_delta:
            enabled = bool(config_delta["enable_greedy_optimization"])
            overrides.append(
                "++generation.refinement.enable_greedy_optimization="
                f"{str(enabled).lower()}"
            )
        if "greedy_percentage" in config_delta:
            overrides.append(
                "++generation.refinement.greedy_percentage="
                f"{float(config_delta['greedy_percentage'])}"
            )
    return overrides


def reward_filter_overrides(config_delta: Mapping[str, object]) -> list[str]:
    """Return bounded hard-target reward and filtering overrides."""

    overrides: list[str] = []
    if "filter_samples_limit" in config_delta:
        overrides.append(
            "++generation.filter.filter_samples_limit="
            f"{int(config_delta['filter_samples_limit'])}"
        )
    reward_keys = {
        "reward_i_pae_weight": "i_pae",
        "reward_plddt_weight": "plddt",
        "reward_min_ipae_weight": "min_ipae",
        "reward_min_ipsae_weight": "min_ipsae",
        "reward_avg_ipsae_weight": "avg_ipsae",
        "reward_max_ipsae_weight": "max_ipsae",
        "reward_i_con_weight": "i_con",
        "reward_i_ptm_weight": "i_ptm",
    }
    for config_key, reward_key in reward_keys.items():
        if config_key in config_delta:
            overrides.append(
                "++generation.reward_model.reward_models.af2folding."
                f"reward_weights.{reward_key}={float(config_delta[config_key])}"
            )
    return overrides


def canonical_af2_overrides() -> list[str]:
    """Force Complexa's AF2 reward settings to match the canonical gate."""

    return [
        "++generation.reward_model.reward_models.af2folding.model_nums=0",
        "++generation.reward_model.reward_models.af2folding.num_recycles=3",
        "++generation.reward_model.reward_models.af2folding.use_initial_guess=True",
        "++generation.reward_model.reward_models.af2folding.use_multimer=True",
    ]


def _launch_seed(candidate_id: str, run_name: str, campaign_seed: int) -> int:
    material = f"{candidate_id}|{run_name}|s{campaign_seed}".encode()
    return int(hashlib.sha256(material).hexdigest()[:8], 16) % 1_000_000


def build_complexa_overrides(
    *,
    family: str,
    candidate_id: str,
    config_delta: Mapping[str, object] | None,
    target_id: str,
    run_name: str,
    repo: Path,
    campaign_seed: int,
) -> list[str]:
    """Compile one validated candidate into the complete Hydra override list."""

    algorithm = ALGORITHM_BY_FAMILY.get(family, "beam-search")
    namespace = NAMESPACE_BY_ALGORITHM.get(algorithm)
    seed = _launch_seed(candidate_id, run_name, campaign_seed)
    overrides = [
        f"++generation.task_name={target_id}",
        f"++ckpt_path={repo}/ckpts",
        "++ckpt_name=complexa.ckpt",
        f"++autoencoder_ckpt_path={repo}/ckpts/complexa_ae.ckpt",
        f"++run_name={run_name}",
        "++gen_njobs=1",
        "++job_id=0",
        f"++seed={seed}",
        "++base_config_name=search_binder_local_pipeline",
        f"++generation.search.algorithm={algorithm}",
    ]
    delta = config_delta or {}
    if "nsteps" in delta:
        nsteps = int(delta["nsteps"])
        checkpoints = sorted({
            0,
            nsteps // 4,
            nsteps // 2,
            3 * nsteps // 4,
            nsteps,
        })
        overrides.extend([
            f"++generation.args.nsteps={nsteps}",
            "++generation.search.step_checkpoints="
            f"[{','.join(str(value) for value in checkpoints)}]",
        ])
    if "nsamples" in delta:
        overrides.append(
            f"++generation.dataloader.dataset.nres.nsamples={int(delta['nsamples'])}"
        )
    if "batch_size" in delta:
        overrides.append(
            f"++generation.dataloader.batch_size={int(delta['batch_size'])}"
        )
    if "sc_scale_noise" in delta:
        overrides.append(
            "++generation.model.bb_ca.simulation_step_params.sc_scale_noise="
            f"{float(delta['sc_scale_noise'])}"
        )
    overrides.extend(refinement_overrides(delta))
    overrides.extend(canonical_af2_overrides())
    overrides.extend(reward_filter_overrides(delta))

    if namespace in {"beam_search", "fk_steering"}:
        prefix = f"generation.search.{namespace}"
        if "beam_width" in delta:
            overrides.append(f"++{prefix}.beam_width={int(delta['beam_width'])}")
        if "n_branch" in delta:
            overrides.append(f"++{prefix}.n_branch={int(delta['n_branch'])}")
        if "temperature" in delta and namespace == "fk_steering":
            overrides.append(
                f"++{prefix}.temperature={float(delta['temperature'])}"
            )
    elif namespace == "best_of_n" and "replicas" in delta:
        overrides.append(
            f"++generation.search.best_of_n.replicas={int(delta['replicas'])}"
        )
    elif namespace == "mcts":
        prefix = "generation.search.mcts"
        if "n_simulations" in delta:
            overrides.append(
                f"++{prefix}.n_simulations={int(delta['n_simulations'])}"
            )
        if "exploration_prob" in delta:
            overrides.append(
                f"++{prefix}.exploration_prob={float(delta['exploration_prob'])}"
            )
        if "exploration_constant" in delta:
            overrides.append(
                "++generation.search.mcts.exploration_constant="
                f"{float(delta['exploration_constant'])}"
            )
    return overrides


def build_complexa_shell_command(
    *, repo: Path, python: Path, overrides: list[str],
) -> str:
    """Build the reviewed shell boundary required to source Complexa env.sh."""

    override_text = " ".join(shlex.quote(value) for value in overrides)
    return (
        f"cd {shlex.quote(str(repo))} && "
        "source env.sh > /dev/null 2>&1 && "
        f"{shlex.quote(str(python))} -m proteinfoundation.generate "
        f"--config-path {shlex.quote(str(repo / 'configs'))} "
        "--config-name search_binder_local_pipeline "
        f"{override_text}"
    )


__all__ = [
    "ALGORITHM_BY_FAMILY",
    "NAMESPACE_BY_ALGORITHM",
    "build_complexa_overrides",
    "build_complexa_shell_command",
    "canonical_af2_overrides",
    "refinement_overrides",
    "reward_filter_overrides",
]
