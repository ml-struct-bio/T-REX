"""Registered action families, supported settings, resource classes, and availability.

Candidate construction and selection share this registry. Preflight is performed at
campaign setup rather than on every planning cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Tuple, Union


Availability = Literal[
    "available", "degraded", "unavailable", "legacy_archive_only"
]
FamilyRole = Literal["generator", "refilter", "seq_redesign"]


ParamRange = Union[Tuple[float, float], list]


@dataclass(frozen=True)
class Capability:
    family: str
    default_operator_id: str
    default_lane_id: str
    default_cost_class: Literal["low", "diagnostic", "standard", "extended"]
    runtime_bucket_id: str | None  # None if backend has not been preflighted
    availability: Availability
    notes: str = ""
    # Optional preflight callable; returns (ok: bool, message: str).
    # Not invoked at registration; operator runs it once per run.
    preflight: Callable[[], tuple[bool, str]] | None = None
    # Allowed numeric ranges and categorical settings for each family.
    allowed_params: dict[str, ParamRange] = field(default_factory=dict)
    # Registry metadata supplies prompt roles and deterministic feasibility checks.
    # Diagnostic-only outputs require standardized evaluation before qualification.
    role: FamilyRole = "generator"
    requires_parent_pdb: bool = False
    outputs_diagnostic_only: bool = False


def validate_config_delta(
    capability: "Capability", config_delta: dict[str, object]
) -> tuple[bool, list[str]]:
    """Validate parameter ranges, allowed values, and the per-job evaluation budget.

    Return (ok, reasons), with an empty reasons list when all checks pass.
    """
    reasons: list[str] = []
    for k, v in (config_delta or {}).items():
        rng = capability.allowed_params.get(k)
        if rng is None:
            reasons.append(f"unknown_param:{k}")
            continue
        if isinstance(rng, tuple) and len(rng) == 2:
            lo, hi = rng
            try:
                fv = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                reasons.append(f"non_numeric:{k}={v}")
                continue
            if not (lo <= fv <= hi):
                reasons.append(f"out_of_range:{k}={v} not in [{lo},{hi}]")
        elif isinstance(rng, list):
            if v not in rng:
                reasons.append(f"not_allowed_value:{k}={v} not in {rng}")
        else:
            reasons.append(f"bad_range_spec:{k}")

    # (2) Eval-budget guardrail
    budget = compute_eval_budget(capability.family, config_delta)
    cap = EVAL_BUDGET_CAP_PER_FAMILY.get(capability.family, DEFAULT_EVAL_BUDGET_CAP)
    if budget > cap:
        reasons.append(f"eval_budget_exceeded:family={capability.family} "
                        f"budget={budget} > cap={cap} "
                        f"(nsteps×nsamples×beam_width×n_branch or family equivalent)")

    return (len(reasons) == 0, reasons)


def validate_config_delta_partial(
    capability: "Capability", config_delta: dict[str, object]
) -> tuple[dict[str, object], list[str]]:
    """Return validated settings and reasons for rejected settings.

    Clamp numeric overshoots, drop unknown or invalid categorical settings, and repair
    the evaluation budget if possible. Reject unrepairable settings explicitly rather
    than silently substituting defaults.
    """
    kept: dict[str, object] = {}
    dropped: list[str] = []
    for k, v in (config_delta or {}).items():
        rng = capability.allowed_params.get(k)
        if rng is None:
            if capability.family == "proteinmpnn_redesign" and k == "designed_chains":
                dropped.append(f"derived_param:{k}=controller_resolved_binder_chain")
            else:
                dropped.append(f"unknown_param:{k}")
            continue
        if isinstance(rng, tuple) and len(rng) == 2:
            lo, hi = rng
            try:
                fv = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                dropped.append(f"non_numeric:{k}={v}")
                continue
            if fv < lo or fv > hi:
                clipped = lo if fv < lo else hi
                if isinstance(v, int) and not isinstance(v, bool) and float(clipped).is_integer():
                    kept[k] = int(clipped)
                else:
                    kept[k] = clipped
                dropped.append(f"clamped:{k}={v}->{kept[k]} range=[{lo},{hi}]")
                continue
        elif isinstance(rng, list):
            if v not in rng:
                dropped.append(f"not_allowed_value:{k}={v} not in {rng}")
                continue
        else:
            dropped.append(f"bad_range_spec:{k}")
            continue
        kept[k] = v

    if kept:
        budget = compute_eval_budget(capability.family, kept)
        cap = EVAL_BUDGET_CAP_PER_FAMILY.get(capability.family, DEFAULT_EVAL_BUDGET_CAP)
        if budget > cap:
            repaired, repair_reasons = repair_config_delta_to_budget(capability, kept)
            dropped.extend(repair_reasons)
            if repaired is None:
                return ({}, dropped)
            return (repaired, dropped)
    return (kept, dropped)


def family_params_schema_for_prompt(
    registry: "CapabilityRegistry", families: list[str] | None = None
) -> dict[str, dict[str, object]]:
    """Serialize allowed parameters as JSON-safe family-to-parameter mappings."""
    out: dict[str, dict[str, object]] = {}
    keys = families if families is not None else list(registry.capabilities.keys())
    for fam in keys:
        cap = registry.get(fam)
        if cap is None:
            continue
        schema: dict[str, object] = {}
        for k, rng in cap.allowed_params.items():
            if isinstance(rng, tuple) and len(rng) == 2:
                schema[k] = [rng[0], rng[1]]
            elif isinstance(rng, list):
                schema[k] = list(rng)
            else:
                schema[k] = str(rng)
        out[fam] = schema
    return out


# Per-job evaluation-budget limits.

# Per-launch evaluation caps bound search expansion.
DEFAULT_EVAL_BUDGET_CAP: int = 32_768

# Per-family limits use workload settings, not measured GPU time.
# BindCraft counts requested native accepted designs; BoltzGen uses designs × budget.
EVAL_BUDGET_CAP_PER_FAMILY: dict[str, int] = {
    # Evaluation budgets include search expansion and the executor default step count.
    "complexa_beam":            65_536,
    "complexa_best_of_n":       65_536,
    "complexa_fk_steering":     65_536,
    "complexa_mcts":            131_072,  # MCTS extended cost class — 2× cap
    # Sequence-only (low cost)
    "proteinmpnn_redesign":     2_048,   # num_seq_per_target
    # Refilters (low cost)
    "structure_refilter":       1_024,
    # External generators use their native workload settings.
    "bindcraft":                64,      # requested native accepted designs; not attempted trajectories
    "boltzgen":                 256,     # num_designs × budget cap
}


# Match the executor default when nsteps is omitted.
EVAL_BUDGET_DEFAULTS_PER_FAMILY: dict[str, dict[str, int]] = {
    "complexa_beam": {"nsteps": 400, "nsamples": 4, "beam_width": 4, "n_branch": 4},
    "complexa_fk_steering": {"nsteps": 400, "nsamples": 4, "beam_width": 4, "n_branch": 4},
    "complexa_best_of_n": {"nsteps": 400, "nsamples": 4, "replicas": 4},
    "complexa_mcts": {"nsteps": 400, "nsamples": 4, "n_simulations": 8},
    "proteinmpnn_redesign": {"num_seq_per_target": 8},
    "structure_refilter": {"num_recycles": 3},
    "bindcraft": {"max_trajectories": 4},
    "boltzgen": {"num_designs": 16, "budget": 4},
}

EVAL_BUDGET_FORMULA_PER_FAMILY: dict[str, str] = {
    "complexa_beam": "nsteps * nsamples * beam_width * n_branch",
    "complexa_fk_steering": "nsteps * nsamples * beam_width * n_branch",
    "complexa_best_of_n": "nsteps * nsamples * replicas",
    "complexa_mcts": "nsteps * nsamples * n_simulations",
    "proteinmpnn_redesign": "num_seq_per_target",
    "structure_refilter": "model_count(model_names) * num_recycles",
    "bindcraft": "max_trajectories",
    "boltzgen": "num_designs * budget",
}


def _eval_budget_defaults(family: str) -> dict[str, int]:
    if family in EVAL_BUDGET_DEFAULTS_PER_FAMILY:
        return EVAL_BUDGET_DEFAULTS_PER_FAMILY[family]
    if family.startswith("complexa_"):
        return EVAL_BUDGET_DEFAULTS_PER_FAMILY["complexa_beam"]
    return {}


def _int_param(config_delta: dict[str, object], key: str, default: int) -> int:
    value = config_delta.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


def _model_names_count(config_delta: dict[str, object], default: str) -> int:
    value = config_delta.get("model_names", default)
    if not isinstance(value, str):
        return 1
    names = [x.strip() for x in value.split(",") if x.strip()]
    return max(1, len(names))


def compute_eval_budget(family: str, config_delta: dict[str, object] | None) -> int:
    """Compute the per-launch eval budget for a given family + config_delta.

    For Complexa search families:
      budget = nsteps × nsamples × beam_width × n_branch
    For MCTS: budget = nsteps × nsamples × n_simulations
    For ProteinMPNN: budget = num_seq_per_target
    For refilters: budget = model/sample count × recycle/step count
    For BindCraft: budget = max_trajectories
    For Boltzgen: budget = num_designs × budget
    """
    cd = config_delta or {}
    defaults = _eval_budget_defaults(family)
    g = lambda k, default: _int_param(cd, k, defaults.get(k, default))  # noqa: E731
    if family.startswith("complexa_"):
        if family == "complexa_mcts":
            return g("nsteps", 400) * g("nsamples", 4) * g("n_simulations", 8)
        if family == "complexa_best_of_n":
            return g("nsteps", 400) * g("nsamples", 4) * g("replicas", 4)
        # beam-family default
        return (g("nsteps", 400) * g("nsamples", 4)
                * g("beam_width", 4) * g("n_branch", 4))
    if family == "proteinmpnn_redesign":
        return g("num_seq_per_target", 8)
    if family == "structure_refilter":
        return _model_names_count(
            cd, "model_1_multimer_v3") * g("num_recycles", 3)
    if family == "bindcraft":
        return g("max_trajectories", 4)
    if family == "boltzgen":
        return g("num_designs", 16) * g("budget", 4)
    # TREX_STANDALONE_EXTENSION_BEGIN: multiplicative work-unit contract.
    # Adapter metadata is installed before an extension family becomes
    # feasible. Built-in families return above, so their frozen accounting is
    # unchanged.
    if family in EVAL_BUDGET_DEFAULTS_PER_FAMILY:
        budget = 1
        for key, default in EVAL_BUDGET_DEFAULTS_PER_FAMILY[family].items():
            budget *= max(1, g(key, int(default)))
        return budget
    # TREX_STANDALONE_EXTENSION_END
    # Unknown family: be conservative, return cap so it gets rejected
    return DEFAULT_EVAL_BUDGET_CAP + 1


def repair_config_delta_to_budget(
    capability: "Capability", config_delta: dict[str, object]
) -> tuple[dict[str, object] | None, list[str]]:
    """Shrink budget-driving knobs until the config fits the family cap.

    Non-budget knobs such as reward weights, pLDDT rescue controls,
    refinement_algorithm, and structural noise are preserved. This prevents the
    critical hidden no-op where an over-budget intended config was silently
    converted to `{}` and launched as a default worker run.
    """
    family = capability.family
    cap = EVAL_BUDGET_CAP_PER_FAMILY.get(family, DEFAULT_EVAL_BUDGET_CAP)
    start_budget = compute_eval_budget(family, config_delta)
    if start_budget <= cap:
        return dict(config_delta), []

    defaults = _eval_budget_defaults(family)
    keys = list(defaults.keys())
    if not keys:
        return None, [
            f"eval_budget_exceeded_unrepairable:family={family} "
            f"budget={start_budget} > cap={cap}"
        ]

    repaired = dict(config_delta)
    changes: list[str] = []
    # Reduce explicitly supplied width settings before depth. Never inject an omitted
    # setting or lower nsteps below its family default.
    _DEPTH_KEYS = {"nsteps"}
    shrink_order = sorted(
        (k for k in keys if k in config_delta),  # only knobs the LLM explicitly set
        key=lambda k: (1 if k in _DEPTH_KEYS else 0, keys.index(k)),
    )
    for key in shrink_order:
        rng = capability.allowed_params.get(key)
        if isinstance(rng, tuple) and len(rng) == 2:
            min_value = max(1, int(rng[0]))
        else:
            min_value = max(1, int(defaults.get(key, 1)))
        if key in _DEPTH_KEYS:
            # depth floor = family default, not the allowed-range floor.
            min_value = max(min_value, int(defaults.get(key, min_value)))
        current = _int_param(repaired, key, defaults.get(key, 1))
        if current <= min_value:
            continue

        other_product = 1
        for other in keys:
            if other == key:
                continue
            other_product *= max(1, _int_param(repaired, other, defaults.get(other, 1)))
        max_for_key = max(min_value, cap // max(1, other_product))
        new_value = max(min_value, min(current, max_for_key))
        if new_value < current:
            repaired[key] = new_value
            changes.append(f"{key}:{current}->{new_value}")

        if compute_eval_budget(family, repaired) <= cap:
            repaired_budget = compute_eval_budget(family, repaired)
            return repaired, [
                f"eval_budget_repaired:family={family} "
                f"budget={start_budget}>{cap}; "
                f"changes={','.join(changes)}; repaired_budget={repaired_budget}"
            ]

    repaired_budget = compute_eval_budget(family, repaired)
    if repaired_budget <= cap:
        return repaired, [
            f"eval_budget_repaired:family={family} "
            f"budget={start_budget}>{cap}; "
            f"changes={','.join(changes)}; repaired_budget={repaired_budget}"
        ]
    return None, [
        f"eval_budget_exceeded_unrepairable:family={family} "
        f"budget={start_budget} > cap={cap}; "
        f"after_repair={repaired_budget}; changes={','.join(changes)}"
    ]


def family_eval_budget_schema_for_prompt(
    registry: "CapabilityRegistry", families: list[str] | None = None
) -> dict[str, dict[str, object]]:
    """Serialize per-family eval-budget formulas/caps for the Planner prompt."""
    out: dict[str, dict[str, object]] = {}
    keys = families if families is not None else list(registry.capabilities.keys())
    for fam in keys:
        cap = registry.get(fam)
        if cap is None:
            continue
        defaults = _eval_budget_defaults(fam)
        formula = EVAL_BUDGET_FORMULA_PER_FAMILY.get(fam)
        if not formula:
            continue
        budget_cap = EVAL_BUDGET_CAP_PER_FAMILY.get(fam, DEFAULT_EVAL_BUDGET_CAP)
        out[fam] = {
            "formula": formula,
            "cap": budget_cap,
            "defaults_when_key_omitted": defaults,
            "default_budget": compute_eval_budget(fam, {}),
            "rule": f"{formula} <= {budget_cap}",
        }
    return out


@dataclass(frozen=True)
class CapabilityRegistry:
    capabilities: dict[str, Capability]

    def get(self, family: str) -> Capability | None:
        return self.capabilities.get(family)

    def is_available(self, family: str) -> bool:
        c = self.capabilities.get(family)
        return c is not None and c.availability == "available"

    def feasible_families(self) -> list[str]:
        return [f for f, c in self.capabilities.items() if c.availability == "available"]

    def with_override(self, family: str, **overrides) -> "CapabilityRegistry":
        if family not in self.capabilities:
            raise KeyError(family)
        current = self.capabilities[family]
        from dataclasses import replace

        new_cap = replace(current, **overrides)
        new_map = dict(self.capabilities)
        new_map[family] = new_cap
        return CapabilityRegistry(capabilities=new_map)


def default_registry(
    *,
    runtime_bucket_id: str = "rb_v7_mvp_default",
    available_external: tuple[str, ...] = (
        "bindcraft",
        "boltzgen",
    ),
    unavailable: tuple[str, ...] = (),
) -> CapabilityRegistry:
    """Construct the capability registry; callers can mark backends unavailable after
    preflight.
    """
    # Allowed settings for the four supported Complexa search families.
    COMPLEXA_REWARD_PARAMS = {
        "reward_i_pae_weight": (-2.0, 0.0),
        "reward_plddt_weight": (0.0, 2.0),
        "reward_min_ipae_weight": (-2.0, 0.0),
        "reward_min_ipsae_weight": (0.0, 2.0),
        "reward_avg_ipsae_weight": (0.0, 2.0),
        "reward_max_ipsae_weight": (0.0, 2.0),
        "reward_i_con_weight": (-2.0, 2.0),
        "reward_i_ptm_weight": (0.0, 2.0),
    }

    LOCAL_PARAMS = {
        "complexa_beam": {
            "beam_width": (2.0, 16.0),
            "n_branch": (2.0, 8.0),
            "nsamples": (1.0, 8.0),
            "nsteps": (100.0, 500.0),
            "batch_size": (1.0, 16.0),
            # Expose refinement and noise settings within the same search family.
            "sc_scale_noise": (0.05, 0.50),
            "refinement_algorithm": ["", "sequence_hallucination"],
            "n_greedy_iters": (0.0, 30.0),
            "enable_greedy_optimization": [True, False],
            # Appendix I.4-style hard-target knobs, exposed target-agnostically.
            # They are evidence-triggered by pLDDT-low/iPAE-good failure modes,
            # not by target identity.
            "greedy_percentage": (1.0, 10.0),
            "filter_samples_limit": (1.0, 1000.0),
            **COMPLEXA_REWARD_PARAMS,
        },
        "complexa_best_of_n": {            # search.algorithm = best-of-n
            "replicas": (1.0, 8.0),
            "nsamples": (1.0, 16.0),
            "nsteps": (100.0, 500.0),
            "batch_size": (1.0, 16.0),
            "sc_scale_noise": (0.05, 0.50),
            "refinement_algorithm": ["", "sequence_hallucination"],
            "n_greedy_iters": (0.0, 30.0),
            "enable_greedy_optimization": [True, False],
            "greedy_percentage": (1.0, 10.0),
            "filter_samples_limit": (1.0, 1000.0),
            **COMPLEXA_REWARD_PARAMS,
        },
        "complexa_fk_steering": {          # search.algorithm = fk-steering
            "beam_width": (2.0, 16.0),
            "n_branch": (2.0, 8.0),
            "temperature": (0.05, 0.50),   # tighter band than stochastic beam
            "nsamples": (1.0, 8.0),
            "nsteps": (100.0, 500.0),
            "batch_size": (1.0, 16.0),
            "sc_scale_noise": (0.05, 0.50),
            "refinement_algorithm": ["", "sequence_hallucination"],
            "n_greedy_iters": (0.0, 30.0),
            "enable_greedy_optimization": [True, False],
            "greedy_percentage": (1.0, 10.0),
            "filter_samples_limit": (1.0, 1000.0),
            **COMPLEXA_REWARD_PARAMS,
        },
        "complexa_mcts": {                 # search.algorithm = mcts
            "n_simulations": (5.0, 80.0),
            "exploration_prob": (0.10, 0.80),
            "exploration_constant": (0.50, 2.0),
            "nsteps": (100.0, 500.0),
            "nsamples": (1.0, 8.0),
            "batch_size": (1.0, 16.0),
            "sc_scale_noise": (0.05, 0.50),
            "refinement_algorithm": ["", "sequence_hallucination"],
            "n_greedy_iters": (0.0, 30.0),
            "enable_greedy_optimization": [True, False],
            "greedy_percentage": (1.0, 10.0),
            "filter_samples_limit": (1.0, 1000.0),
            **COMPLEXA_REWARD_PARAMS,
        },
        # Registry keys must match settings consumed by the backend adapters.
        "proteinmpnn_redesign": {
            "num_seq_per_target": (1.0, 32.0),
            "sampling_temp": (0.05, 0.50),
            "model_name": ["v_48_002", "v_48_010", "v_48_020", "v_48_030"],
            "backbone_noise": (0.0, 0.30),
            "omit_AAs": ["X", "C", "CP", "CW", "FWY", "G", "P", "GP", "DE", "KR"],
            # designed_chains is intentionally not exposed. The controller resolves
            # the binder chain from target/parent geometry and passes it to the
            # ProteinMPNN executor and parser; making it LLM-tunable creates a
            # hidden no-op or an invalid chain on multichain targets.
        },
        "structure_refilter": {            # AF2 multimer refilter
            "num_recycles": (0.0, 6.0),
            "model_names": [
                "model_1_multimer_v3",
                "model_2_multimer_v3",
                "model_3_multimer_v3",
                "model_4_multimer_v3",
                "model_5_multimer_v3",
                "model_1_multimer_v3,model_2_multimer_v3",
                "model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3",
                "model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3,model_4_multimer_v3",
                "model_1_multimer_v3,model_2_multimer_v3,model_3_multimer_v3,model_4_multimer_v3,model_5_multimer_v3",
            ],
            "use_initial_guess": [0, 1],
        },
    }
    EXTERNAL_PARAMS = {
        "bindcraft": {
            # The upper trajectory bound matches the family evaluation-budget cap.
            "max_trajectories": (2.0, 64.0),
            "target_length_min": (40.0, 120.0),
            "target_length_max": (60.0, 200.0),
            # These settings override the backend advanced-settings file for each
            # launch.
            "weights_plddt":      (0.05, 0.40),
            "weights_pae_inter":  (0.05, 0.40),
            "weights_iptm":       (0.02, 0.20),
            "weights_helicity":   (-0.80, 0.20),
            "soft_iterations":    (20.0, 200.0),
            "hard_iterations":    (1.0,  20.0),
            "greedy_iterations":  (5.0,  40.0),
            "num_seqs":           (4.0,  64.0),
            "mpnn_fix_interface": [True, False],
        },
        "boltzgen": {
            # Expose only protocols whose required inputs the launcher implements.
            "protocol": ["protein-anything"],
            "num_designs": (8.0, 128.0),
            "budget": (1.0, 16.0),
            "diffusion_batch_size": (1.0, 16.0),
            "step_scale": (0.5, 2.0),
            "noise_scale": (0.0, 1.0),
        },
    }

    caps: dict[str, Capability] = {
        "complexa_beam": Capability(
            family="complexa_beam",
            default_operator_id="complexa_beam_default",
            default_lane_id="complexa_beam",
            default_cost_class="standard",
            runtime_bucket_id=runtime_bucket_id,
            availability="available",
            allowed_params=LOCAL_PARAMS["complexa_beam"],
        ),
        "complexa_best_of_n": Capability(
            family="complexa_best_of_n",
            default_operator_id="complexa_best_of_n_default",
            default_lane_id="complexa_best_of_n",
            default_cost_class="standard",
            runtime_bucket_id=runtime_bucket_id,
            availability="available",
            allowed_params=LOCAL_PARAMS["complexa_best_of_n"],
            notes="best-of-n: replicas independent samples, take the best.",
        ),
        "complexa_fk_steering": Capability(
            family="complexa_fk_steering",
            default_operator_id="complexa_fk_steering_default",
            default_lane_id="complexa_fk_steering",
            default_cost_class="standard",
            runtime_bucket_id=runtime_bucket_id,
            availability="available",
            allowed_params=LOCAL_PARAMS["complexa_fk_steering"],
            notes="fk-steering: log-derivative reward steering at low temp.",
        ),
        "complexa_mcts": Capability(
            family="complexa_mcts",
            default_operator_id="complexa_mcts_default",
            default_lane_id="complexa_mcts",
            default_cost_class="extended",  # MCTS is expensive
            runtime_bucket_id=runtime_bucket_id,
            availability="available",
            allowed_params=LOCAL_PARAMS["complexa_mcts"],
            notes="MCTS: tree search over diffusion trajectories. High compute.",
        ),
        "proteinmpnn_redesign": Capability(
            family="proteinmpnn_redesign",
            default_operator_id="interface_redesign",
            default_lane_id="proteinmpnn_redesign",
            default_cost_class="low",
            runtime_bucket_id=runtime_bucket_id,
            availability="available",
            allowed_params=LOCAL_PARAMS["proteinmpnn_redesign"],
            role="seq_redesign",
            requires_parent_pdb=True,
            outputs_diagnostic_only=True,  # no scoring; refold via AF2 chain
        ),
        # structure_refilter is an AF2 RE-SCORER, NOT a generator. The same
        # backend has two explicit roles in ActionCandidate.refilter_role:
        # canonical_score_conversion (automatic scoring after diagnostic
        # generators; system chain_refilter lane; SU credit resolves upstream)
        # and parent_model_refold (intentional LLM-selected parent-bound AF2
        # retry/refinement; consumes normal E/R/E budget). Never treat the
        # backend family itself as a de-novo productive generator.
        "structure_refilter": Capability(
            family="structure_refilter",
            default_operator_id="af2_consensus",
            default_lane_id="structure_refilter",
            default_cost_class="low",
            runtime_bucket_id=runtime_bucket_id,
            availability="available",
            allowed_params=LOCAL_PARAMS["structure_refilter"],
            role="refilter",
            requires_parent_pdb=True,
            outputs_diagnostic_only=False,  # AF2-calibrated, feeds strict_success
        ),
    }
    # All 4 complexa_* families share the same role metadata (de-novo
    # generator, no parent needed). Their af2folding reward CSV is canonical
    # AF2-Multimer verifier output, so they mint strict/SU directly.
    for fam in ("complexa_beam", "complexa_best_of_n",
                 "complexa_fk_steering", "complexa_mcts"):
        caps[fam] = Capability(
            family=caps[fam].family,
            default_operator_id=caps[fam].default_operator_id,
            default_lane_id=caps[fam].default_lane_id,
            default_cost_class=caps[fam].default_cost_class,
            runtime_bucket_id=caps[fam].runtime_bucket_id,
            availability=caps[fam].availability,
            notes=caps[fam].notes,
            allowed_params=caps[fam].allowed_params,
            role="generator",
            requires_parent_pdb=False,
            outputs_diagnostic_only=False,
        )
    # External generators — diagnostic flag varies by family.
    EXT_ROLE_META = {
        # BindCraft native scores and Binder_RMSD are diagnostic. Its structures require
        # independent standardized AF2 evaluation for the canonical qualification
        # measurements.
        "bindcraft":   {"role": "generator", "requires_parent_pdb": False, "outputs_diagnostic_only": True},
        # BoltzGen: native generator scoring is diagnostic until canonical AF2 score conversion.
        "boltzgen":    {"role": "generator", "requires_parent_pdb": False, "outputs_diagnostic_only": True},
    }
    for ext in ("bindcraft", "boltzgen"):
        meta = EXT_ROLE_META[ext]
        caps[ext] = Capability(
            family=ext,
            default_operator_id=f"{ext}_default",
            default_lane_id=ext,
            default_cost_class="diagnostic",
            runtime_bucket_id=runtime_bucket_id if ext in available_external else None,
            availability="available" if ext in available_external else "unavailable",
            notes="external generator — first use is diagnostic, never extended",
            allowed_params=EXTERNAL_PARAMS[ext] if ext in available_external else {},
            role=meta["role"],  # type: ignore[arg-type]
            requires_parent_pdb=meta["requires_parent_pdb"],
            outputs_diagnostic_only=meta["outputs_diagnostic_only"],
        )
    for u in unavailable:
        caps[u] = Capability(
            family=u,
            default_operator_id=f"{u}_default",
            default_lane_id=u,
            default_cost_class="standard",
            runtime_bucket_id=None,
            availability="unavailable",
            notes="capability_unavailable: weights/databases not staged in this environment",
        )
    # TREX_STANDALONE_EXTENSION_BEGIN: additive operator-installed backends.
    # No adapter is loaded in the publication configuration. Collisions with a
    # built-in family fail closed rather than replacing frozen behavior.
    from .backend_extensions import (
        install_extension_budget_metadata,
        load_backend_adapters,
    )

    adapters = load_backend_adapters()
    install_extension_budget_metadata(adapters)
    for family, adapter in adapters.items():
        if family in caps:
            raise ValueError(
                f"backend extension {family!r} collides with a built-in family"
            )
        caps[family] = adapter.capability
    # TREX_STANDALONE_EXTENSION_END
    return CapabilityRegistry(capabilities=caps)
