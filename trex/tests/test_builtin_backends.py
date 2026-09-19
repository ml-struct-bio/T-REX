"""Contract tests for built-in backend command construction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trex.backends.af2 import prepare_af2_refilter_launch
from trex.backends.bindcraft import prepare_bindcraft_launch
from trex.backends.boltzgen import prepare_boltzgen_launch, write_boltzgen_yaml
from trex.backends.complexa import (
    build_complexa_overrides,
    build_complexa_shell_command,
    canonical_af2_overrides,
    refinement_overrides,
)
from trex.backends.proteinmpnn import prepare_proteinmpnn_launch


def _argument(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def _write_numbered_pdb(
    path: Path,
    residues: list[tuple[str, int, str]],
) -> None:
    lines = []
    for serial, (chain, residue_number, insertion_code) in enumerate(
        residues, start=1,
    ):
        lines.append(
            f"ATOM  {serial:5d}  CA  ALA {chain}{residue_number:4d}"
            f"{insertion_code:1s}   {float(serial):8.3f}{0.0:8.3f}{0.0:8.3f}"
            "  1.00 20.00           C"
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    "family,algorithm,specific_override",
    [
        (
            "complexa_beam",
            "beam-search",
            "++generation.search.beam_search.beam_width=4",
        ),
        (
            "complexa_fk_steering",
            "fk-steering",
            "++generation.search.fk_steering.temperature=0.2",
        ),
        (
            "complexa_best_of_n",
            "best-of-n",
            "++generation.search.best_of_n.replicas=3",
        ),
        (
            "complexa_mcts",
            "mcts",
            "++generation.search.mcts.n_simulations=8",
        ),
    ],
)
def test_complexa_family_uses_one_canonical_override_builder(
    family: str, algorithm: str, specific_override: str,
) -> None:
    overrides = build_complexa_overrides(
        family=family,
        candidate_id="candidate-1",
        config_delta={
            "nsteps": 300,
            "beam_width": 4,
            "temperature": 0.2,
            "replicas": 3,
            "n_simulations": 8,
        },
        target_id="target-1",
        run_name="run-1",
        repo=Path("/opt/complexa"),
        campaign_seed=11,
    )

    assert f"++generation.search.algorithm={algorithm}" in overrides
    assert specific_override in overrides
    assert "++generation.search.step_checkpoints=[0,75,150,225,300]" in overrides
    assert "++job_id=0" in overrides
    assert sum(item.startswith("++seed=") for item in overrides) == 1
    for canonical in canonical_af2_overrides():
        assert canonical in overrides


def test_complexa_launch_seed_is_deterministic_and_campaign_specific() -> None:
    common = {
        "family": "complexa_beam",
        "candidate_id": "candidate-1",
        "config_delta": {},
        "target_id": "target-1",
        "run_name": "run-1",
        "repo": Path("/opt/complexa"),
    }

    first = build_complexa_overrides(**common, campaign_seed=1)
    repeated = build_complexa_overrides(**common, campaign_seed=1)
    replicate = build_complexa_overrides(**common, campaign_seed=2)

    first_seed = next(item for item in first if item.startswith("++seed="))
    repeated_seed = next(item for item in repeated if item.startswith("++seed="))
    replicate_seed = next(item for item in replicate if item.startswith("++seed="))
    assert first_seed == repeated_seed
    assert first_seed != replicate_seed


def test_greedy_knobs_activate_the_required_refinement() -> None:
    overrides = refinement_overrides({
        "n_greedy_iters": 5,
        "enable_greedy_optimization": True,
    })

    assert "++generation.refinement.algorithm=sequence_hallucination" in overrides
    assert "++generation.refinement.n_greedy_iters=5" in overrides
    assert "++generation.refinement.enable_greedy_optimization=true" in overrides


def test_complexa_shell_boundary_quotes_paths_and_override_values() -> None:
    command = build_complexa_shell_command(
        repo=Path("/opt/Complexa checkout"),
        python=Path("/opt/python env/bin/python"),
        overrides=["++generation.task_name=target with space"],
    )

    assert "cd '/opt/Complexa checkout' &&" in command
    assert "'/opt/python env/bin/python' -m proteinfoundation.generate" in command
    assert "'++generation.task_name=target with space'" in command


def test_controller_keeps_legacy_private_builder_aliases() -> None:
    from trex import controller

    assert controller._complexa_canonical_af2_overrides is canonical_af2_overrides
    assert controller._complexa_refinement_overrides is refinement_overrides


def test_bindcraft_builder_writes_reviewable_inputs_and_direct_argv(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "BindCraft"
    advanced_dir = repo / "settings_advanced"
    advanced_dir.mkdir(parents=True)
    (repo / "settings_filters").mkdir()
    (advanced_dir / "default_4stage_multimer_mpnn_hardtarget.json").write_text(
        '{"soft_iterations": 50, "mpnn_fix_interface": false}'
    )
    output = tmp_path / "worker"

    launch = prepare_bindcraft_launch(
        output_dir=output,
        repo=repo,
        environment_root=tmp_path / "bindcraft-env",
        target_id="target-1",
        target_pdb="/inputs/target.pdb",
        hotspots="A10,A20",
        chains="A",
        default_lengths=(50, 100),
        config_delta={
            "target_length_min": 120,
            "target_length_max": 80,
            "max_trajectories": 6,
            "soft_iterations": 75.9,
            "mpnn_fix_interface": 1,
        },
    )

    settings = json.loads(launch.settings_path.read_text())
    advanced = json.loads(launch.advanced_path.read_text())
    assert settings["lengths"] == [80, 120]
    assert settings["number_of_final_designs"] == 6
    assert settings["target_hotspot_residues"] == "A10,A20"
    assert advanced["soft_iterations"] == 75
    assert advanced["mpnn_fix_interface"] is True
    assert launch.argv[0].endswith("bindcraft-env/bin/python")
    assert tuple(launch.argv[-2:]) == ("--advanced", str(launch.advanced_path))


def test_af2_refilter_builder_preserves_scientific_parameters_and_seed(
    tmp_path: Path,
) -> None:
    common = dict(
        output_dir=tmp_path / "af2",
        parent_pdb=tmp_path / "parent.pdb",
        target_chain="A,C",
        binder_chain="D",
        gpu_id="2",
        python=tmp_path / "af2-env/bin/python",
        community_root=tmp_path / "community_models",
        af2_data_dir=tmp_path / "community_models/ckpts/AF2",
        trex_repo_root=tmp_path / "T-ReX",
        candidate_id="candidate-7",
        config_delta={
            "model_names": "model_2_multimer_v3",
            "num_recycles": 5,
            "use_initial_guess": 0,
        },
    )

    first = prepare_af2_refilter_launch(**common, campaign_seed=41)
    repeated = prepare_af2_refilter_launch(**common, campaign_seed=41)
    replicate = prepare_af2_refilter_launch(**common, campaign_seed=42)

    assert _argument(first.argv, "--target-chain") == "A,C"
    assert _argument(first.argv, "--binder-chain") == "D"
    assert _argument(first.argv, "--model-names") == "model_2_multimer_v3"
    assert _argument(first.argv, "--num-recycles") == "5"
    assert _argument(first.argv, "--use-initial-guess") == "0"
    assert _argument(first.argv, "--seed") == _argument(repeated.argv, "--seed")
    assert _argument(first.argv, "--seed") != _argument(replicate.argv, "--seed")
    assert first.environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert first.environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert first.environment["PYTHONPATH"].startswith(f"{tmp_path / 'T-ReX'}:")


def test_boltzgen_spec_maps_raw_residues_to_chain_local_positions(
    tmp_path: Path,
) -> None:
    target_pdb = tmp_path / "target.pdb"
    _write_numbered_pdb(
        target_pdb,
        [("A", 10, ""), ("A", 20, ""), ("A", 20, "A"), ("A", 30, "")],
    )

    spec_path = write_boltzgen_yaml(
        tmp_path / "worker",
        "target-1",
        str(target_pdb),
        ["A10", "A30"],
        ["A"],
        "B",
        (60, 90),
    )

    spec = spec_path.read_text()
    assert "sequence: 60..90" in spec
    assert 'binding: "1,4"' in spec
    assert 'structure_groups: "all"' in spec


def test_boltzgen_builder_exposes_command_output_and_offline_cache(
    tmp_path: Path,
) -> None:
    target_pdb = tmp_path / "target.pdb"
    _write_numbered_pdb(target_pdb, [("A", 10, "")])
    cache = tmp_path / "cache"
    cache.mkdir()
    output = tmp_path / "worker"

    launch = prepare_boltzgen_launch(
        output_dir=output,
        target_id="target-1",
        target_pdb=str(target_pdb),
        hotspots=["A10"],
        chain_ids=["A"],
        binder_chain="B",
        length_range=(50, 100),
        gpu_id="3",
        binary=tmp_path / "boltzgen-env/bin/boltzgen",
        repo=tmp_path / "BoltzGen",
        cache=cache,
        config_delta={
            "protocol": "protein-anything",
            "num_designs": 24,
            "budget": 6,
            "diffusion_batch_size": 3,
            "step_scale": 1.25,
            "noise_scale": 0.8,
        },
    )

    assert _argument(launch.argv, "--num_designs") == "24"
    assert _argument(launch.argv, "--budget") == "6"
    assert _argument(launch.argv, "--diffusion_batch_size") == "3"
    assert _argument(launch.argv, "--step_scale") == "1.25"
    assert _argument(launch.argv, "--noise_scale") == "0.8"
    assert _argument(launch.argv, "--cache") == str(cache)
    assert launch.output_dir == output / "boltzgen"
    assert launch.cwd == tmp_path / "BoltzGen"
    assert launch.environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert launch.environment["HF_HUB_OFFLINE"] == "1"
    assert launch.environment["BOLTZGEN_CACHE"] == str(cache)


def test_proteinmpnn_builder_preserves_chain_parameters_and_seed(
    tmp_path: Path,
) -> None:
    common = dict(
        output_dir=tmp_path / "mpnn",
        parent_pdb=tmp_path / "parent.pdb",
        binder_chain="D",
        gpu_id="4",
        python=tmp_path / "complexa-env/bin/python",
        proteinmpnn_dir=tmp_path / "community_models/ProteinMPNN",
        candidate_id="candidate-8",
        config_delta={
            "num_seq_per_target": 12,
            "sampling_temp": 0.2,
            "model_name": "v_48_030",
            "backbone_noise": 0.1,
            "omit_AAs": "CX",
        },
    )

    first = prepare_proteinmpnn_launch(**common, campaign_seed=9)
    repeated = prepare_proteinmpnn_launch(**common, campaign_seed=9)
    replicate = prepare_proteinmpnn_launch(**common, campaign_seed=10)

    assert _argument(first.argv, "--pdb_path_chains") == "D"
    assert _argument(first.argv, "--num_seq_per_target") == "12"
    assert _argument(first.argv, "--sampling_temp") == "0.2"
    assert _argument(first.argv, "--model_name") == "v_48_030"
    assert _argument(first.argv, "--backbone_noise") == "0.1"
    assert _argument(first.argv, "--omit_AAs") == "CX"
    assert _argument(first.argv, "--seed") == _argument(repeated.argv, "--seed")
    assert _argument(first.argv, "--seed") != _argument(replicate.argv, "--seed")
    assert first.environment["CUDA_VISIBLE_DEVICES"] == "4"
