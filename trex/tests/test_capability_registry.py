"""Capability registry: availability, overrides, feasibility filter."""

from __future__ import annotations

import pytest

from trex.capability_registry import (
    Capability,
    CapabilityRegistry,
    default_registry,
    validate_config_delta_partial,
)
from trex.diagnosis_outcome import STRICT_AXIS_REMEDIATION
from trex.evidence_reducer import DIAGNOSTIC_AXIS_REMEDIATION


def test_default_registry_includes_local_families():
    reg = default_registry()
    # Current Complexa families map directly to implemented search algorithms;
    # unsupported legacy aliases are intentionally absent. The 4 exposed
    # Complexa families + 4 refilter / redesign helpers are what the planner
    # can pick.
    for fam in (
        "complexa_beam",
        "complexa_best_of_n",
        "complexa_fk_steering",
        "complexa_mcts",
        "proteinmpnn_redesign",
        "structure_refilter",
    ):
        assert reg.is_available(fam), f"{fam} should be available"


def test_default_registry_external_default_available():
    reg = default_registry()
    for fam in ("bindcraft", "boltzgen"):
        assert reg.is_available(fam)


def test_external_first_use_is_diagnostic_class():
    reg = default_registry()
    for fam in ("bindcraft", "boltzgen"):
        cap = reg.get(fam)
        assert cap is not None
        assert cap.default_cost_class == "diagnostic"


def test_complexa_families_emit_canonical_af2_scores_directly():
    reg = default_registry()
    for fam in (
        "complexa_beam",
        "complexa_best_of_n",
        "complexa_fk_steering",
        "complexa_mcts",
    ):
        cap = reg.get(fam)
        assert cap is not None
        assert cap.role == "generator"
        assert cap.outputs_diagnostic_only is False

def test_override_to_degrade():
    reg = default_registry()
    reg2 = reg.with_override("boltzgen", availability="degraded")
    assert reg.is_available("boltzgen")
    assert not reg2.is_available("boltzgen")
    assert reg2.get("boltzgen").availability == "degraded"


def test_feasible_families_filters_unavailable():
    # 2026-05-26: alphafold3 removed entirely (no weights). Test only verifies
    # the feasibility filter behavior on a still-available family.
    reg = default_registry()
    fams = reg.feasible_families()
    assert "complexa_beam" in fams


def test_unknown_family_returns_none():
    reg = default_registry()
    assert reg.get("not_a_family") is None
    assert not reg.is_available("not_a_family")


def test_override_unknown_raises():
    reg = default_registry()
    with pytest.raises(KeyError):
        reg.with_override("nope", availability="degraded")


def test_external_can_be_explicitly_unavailable():
    reg = default_registry(available_external=("bindcraft",))
    assert reg.is_available("bindcraft")
    assert reg.get("boltzgen").availability == "unavailable"


def test_diagnostic_remediation_levers_name_registry_params():
    import re

    reg = default_registry(available_external=("bindcraft", "boltzgen"))
    texts = list(x for x in DIAGNOSTIC_AXIS_REMEDIATION.values() if x) + list(STRICT_AXIS_REMEDIATION.values())
    forbidden = (
        "complexa_*",
        "target_length_min/max",
        "reward_i_pae_weight↓/",
        "sc_scale_noise/refinement_algorithm",
    )
    for text in texts:
        for bad in forbidden:
            assert bad not in text, text
        refs = re.findall(r"\b([a-z][a-z0-9_]+)\.([A-Za-z][A-Za-z0-9_]*)", text)
        assert refs, text
        for family, param in refs:
            cap = reg.get(family)
            assert cap is not None, (family, text)
            assert param in cap.allowed_params, (family, param, text)


def test_complexa_hard_target_knobs_are_exposed_and_validated():
    cap = default_registry().get("complexa_mcts")
    assert cap is not None
    for key in (
        "reward_plddt_weight",
        "reward_i_pae_weight",
        "reward_min_ipae_weight",
        "reward_min_ipsae_weight",
        "reward_avg_ipsae_weight",
        "reward_max_ipsae_weight",
        "reward_i_con_weight",
        "reward_i_ptm_weight",
        "filter_samples_limit",
        "greedy_percentage",
    ):
        assert key in cap.allowed_params
    kept, dropped = validate_config_delta_partial(cap, {
        "n_simulations": 20,
        "exploration_prob": 0.5,
        "exploration_constant": 1.0,
        "reward_i_pae_weight": -1.0,
        "reward_plddt_weight": 1.0,
        "reward_min_ipae_weight": -0.5,
        "reward_avg_ipsae_weight": 0.7,
        "reward_i_con_weight": -0.2,
        "reward_i_ptm_weight": 0.6,
        "filter_samples_limit": 100,
        "refinement_algorithm": "sequence_hallucination",
        "greedy_percentage": 5,
    })
    assert not dropped
    assert kept["reward_plddt_weight"] == 1.0
    assert kept["reward_avg_ipsae_weight"] == 0.7


def test_structure_refilter_ensemble_and_initial_guess_knobs_are_exposed():
    cap = default_registry().get("structure_refilter")
    assert cap is not None
    all_models = (
        "model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3,"
        "model_4_multimer_v3,model_5_multimer_v3"
    )
    kept, dropped = validate_config_delta_partial(cap, {
        "model_names": all_models,
        "num_recycles": 6,
        "use_initial_guess": 0,
    })
    assert not dropped
    assert kept["model_names"] == all_models
    assert kept["use_initial_guess"] == 0


def test_proteinmpnn_omit_aas_knob_is_exposed():
    cap = default_registry().get("proteinmpnn_redesign")
    assert cap is not None
    kept, dropped = validate_config_delta_partial(cap, {"omit_AAs": "CP"})
    assert not dropped
    assert kept["omit_AAs"] == "CP"
