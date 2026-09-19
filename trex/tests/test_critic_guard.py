"""Tests for the deterministic Critic guard (critic_guard.py).

Locks the subtle behaviors that broad replay (48 archives / 731 ticks)
established: (c) flags a pure-loser config re-proposal, (c) does NOT flag a
stochastic config that also yields strict successes (the false-positive trap),
(b) flags a deadline violation, and (a) is computed-but-not-emitted.
"""

from __future__ import annotations

from trex.critic_guard import CriticGuardConfig, deterministic_critic


def _recipe(family, cls, cfg, *, h, desc, tick=1):
    return {
        "method_family": family, "recipe_class": cls, "config_delta": cfg,
        "recipe_hash": h, "descendant_count": desc, "recency_tick": tick,
    }


def _card(family, cfg):
    return {
        "recommended_action_families": [family],
        "config_delta_suggestions": {family: cfg},
    }


def _evidence(recipes, remaining_wall_h=40.0):
    return {"recipes": recipes, "remaining_wall_h": remaining_wall_h}


CFG_A = {"nsamples": 16, "nsteps": 400, "sc_scale_noise": 0.5}


def test_c_flags_pure_loser_reproposal():
    ev = _evidence([_recipe("complexa_best_of_n", "joint_fail", CFG_A,
                            h="deadbeef", desc=70, tick=2)])
    out = deterministic_critic(ev, [_card("complexa_best_of_n", CFG_A)],
                               current_tick=10)
    assert any(f.startswith("(c)") for f in out.flags)


def test_c_does_not_flag_when_only_scaffolding_shared():
    """2026-06-13: the levered reward knob CHANGED (on a non-shared key); only
    compute-budget scaffolding is shared -> NOT a repeat (CD45 9630421 false +)."""
    joint = {"beam_width": 8, "n_branch": 4, "nsamples": 4, "nsteps": 200,
             "reward_avg_ipsae_weight": 1.0}
    proposed = {"beam_width": 8, "n_branch": 4, "nsamples": 4, "nsteps": 200,
                "reward_max_ipsae_weight": 1.5}
    ev = _evidence([_recipe("complexa_fk_steering", "joint_fail", joint,
                            h="cafef00d", desc=70, tick=2)])
    out = deterministic_critic(ev, [_card("complexa_fk_steering", proposed)],
                               current_tick=10)
    assert not any(f.startswith("(c)") for f in out.flags)


def test_c_still_flags_when_levered_key_repeats():
    """Same levered key + value repeated (plus shared scaffolding) -> real repeat."""
    cfg = {"beam_width": 8, "nsteps": 200, "reward_avg_ipsae_weight": 1.0}
    ev = _evidence([_recipe("complexa_fk_steering", "joint_fail", cfg,
                            h="beadfeed", desc=70, tick=2)])
    out = deterministic_critic(ev, [_card("complexa_fk_steering", dict(cfg))],
                               current_tick=10)
    assert any(f.startswith("(c)") for f in out.flags)


def test_c_does_not_flag_stochastic_config_that_also_succeeds():
    # Same recipe_hash appears as BOTH strict_success and joint_fail — the
    # evidence reducer splits one config's descendants by outcome. Re-proposing
    # it is correct, so the guard must NOT flag (the false-positive trap).
    h = "1bbe7d48"
    ev = _evidence([
        _recipe("complexa_beam", "strict_success", CFG_A, h=h, desc=2, tick=2),
        _recipe("complexa_beam", "joint_fail", CFG_A, h=h, desc=18, tick=4),
    ])
    out = deterministic_critic(ev, [_card("complexa_beam", CFG_A)],
                               current_tick=5)
    assert out.flags == []


def test_c_ignores_single_noisy_failure():
    ev = _evidence([_recipe("complexa_beam", "joint_fail", CFG_A,
                            h="cafe", desc=1, tick=2)])
    out = deterministic_critic(ev, [_card("complexa_beam", CFG_A)],
                               current_tick=3,
                               cfg=CriticGuardConfig(min_fail_descendants=2))
    assert out.flags == []


def test_b_flags_deadline_violation():
    # bindcraft expected ~2.5h; only 1h remains.
    ev = _evidence([], remaining_wall_h=1.0)
    out = deterministic_critic(ev, [_card("bindcraft", {"nsamples": 8})],
                               current_tick=1)
    assert any(f.startswith("(b)") for f in out.flags)


def test_a_computed_but_not_emitted():
    # recent strict success from a family the planner did NOT propose.
    ev = _evidence([_recipe("bindcraft", "strict_success", {"x": 1},
                            h="aa", desc=5, tick=9)])
    out = deterministic_critic(ev, [_card("complexa_beam", CFG_A)],
                               current_tick=10)
    assert out.flags == []                 # (a) never emitted
    assert any(j.startswith("(a)") for j in out.judgment_flags)
