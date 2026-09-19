"""Capability registry: which (family, operator, lane) tuples are
launchable in this environment.

The registry is the SOLE source of truth for what the
CandidateBuilder + Selector are allowed to emit. It maps each backend
family to:
  - a default operator id / lane id
  - a resource cost class
  - a preflight check (callable or shell command)
  - a runtime bucket id (when known)
  - an availability status: `available`, `degraded`, `unavailable`, or
    `legacy_archive_only`

In MVP, preflight is `Capability.preflight_status` which is set at run
startup by the operator. The registry does NOT auto-run preflight per
tick.
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
    # Gap A: parameter-space novelty. Each key is a config_delta path the
    # Planner may suggest values for. Numeric → (lo, hi) range; categorical
    # → list of allowed values. Verifier rejects suggestions outside this
    # whitelist.
    allowed_params: dict[str, ParamRange] = field(default_factory=dict)
    # fix20 (2026-05-26): typed metadata for adaptive-route Planner reasoning.
    # The prompt renders the role table from these fields (no hard-coded list),
    # and the Selector/Feasibility uses them deterministically:
    #   role                       — high-level capability class
    #   requires_parent_pdb        — feasibility-time precondition; controller
    #                                pre-checks archive before scheduling
    #   outputs_diagnostic_only    — scoring is not AF2-calibrated, so a
    #                                chain through `structure_refilter` is
    #                                needed to enter the strict_success path
    role: FamilyRole = "generator"
    requires_parent_pdb: bool = False
    outputs_diagnostic_only: bool = False


def validate_config_delta(
    capability: "Capability", config_delta: dict[str, object]
) -> tuple[bool, list[str]]:
    """Return (ok, reasons). reasons is empty when ok=True.

    Two layers of validation (Category A guardrail — bug/budget):
      (1) per-param range/enum check (V2 Gap A)
      (2) eval-budget cap (V5/V6.3 lineage — see compute_eval_budget docstring)
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
    """Return (filtered_delta, dropped_reasons).

    B-010 fix (2026-05-26): The strict `validate_config_delta` dropped the
    ENTIRE config_delta on any single invalid key. In practice the LLM often
    proposes a mix of valid (`temperature=0.05`) and invalid
    (`steering_weight=8.0`) keys per family — strict-all dropped them all and
    every Complexa launch ended up running with default config regardless of
    the LLM's reasoning.

    This partial variant keeps the valid keys, clamps numeric overshoots to the
    nearest allowed bound, drops only invalid categorical/unknown keys, and
    re-runs the eval-budget guardrail against the filtered set. If the eval
    budget is exceeded by the FILTERED set, the budget-driving knobs are
    deterministically repaired down to the family cap. If repair is impossible,
    the whole delta is rejected with an explicit reason. It must never silently
    convert an intended config to defaults.
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
    """Serialize allowed_params for the planner prompt.

    Returns a dict {family: {param: [lo, hi] or [...allowed...]}}. Tuples are
    converted to lists for JSON-safety. B-010 fix: previously the planner
    prompt referenced `allowed_params_per_family` as if it would be injected
    into the evidence, but no caller actually injected it — the LLM was left
    hallucinating param names like `num_outputs`, `steering_weight`, `c_puct`
    that don't exist in any family's allowed_params.
    """
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


# ---------------------------------------------------------------------------
# Per-launch eval-budget cap (V5/V6.3 lineage)
# ---------------------------------------------------------------------------

# Default per-launch eval cap (forward-pass count through the simulator).
# Reference points (from V5/V6.3 archives):
#   - V5 CD45 standard: nsteps=400 × nsamples=4 × bw=4 × n_branch=4 = 25,600
#   - V6.3 typical:     nsteps=300 × nsamples=4 × bw=6 × n_branch=4 = 28,800
#   - V5 SC2RBD beam_then_hallucinate (job 8294854): ~32,000 evals total
# Default cap: 32,768. Rejects gigantic configs like nsteps=500 × bw=32
# × n_branch=16 × nsamples=8 = 2,048,000 evals (OOM-risk).
DEFAULT_EVAL_BUDGET_CAP: int = 32_768

# Per-family cap overrides. External generators are budgeted by trajectories
# / num_designs / num_samples instead of search expansion.
EVAL_BUDGET_CAP_PER_FAMILY: dict[str, int] = {
    # Complexa family - search expansion. B-012 fix (2026-05-26): removed
    # two unsupported legacy aliases. Complexa instantiate_search only knows
    # the implemented algorithms (single-pass, best-of-n, beam-search,
    # fk-steering, mcts). The aliases were exposed via Hydra overrides but
    # the dispatcher would ValueError on them; an earlier B-011 typo masked
    # the failure so launches silently fell back to YAML default (best-of-n).
    # The executor ALWAYS ran 400 steps when nsteps was omitted, so the TRUE per-launch
    # budget was already 2x what the old 200-step accounting reported — i.e. the real
    # ceiling these caps permitted was effectively 2x. Doubling the caps alongside the
    # honest 400-step default PRESERVES the same set of allowed width configs (e.g.
    # beam_width=8,n_branch=4 still passes) rather than silently tightening it.
    "complexa_beam":            65_536,
    "complexa_best_of_n":       65_536,
    "complexa_fk_steering":     65_536,
    "complexa_mcts":            131_072,  # MCTS extended cost class — 2× cap
    # Sequence-only (low cost)
    "proteinmpnn_redesign":     2_048,   # num_seq_per_target
    # Refilters (low cost)
    "structure_refilter":       1_024,
    # External generators — budgeted by max_trajectories / num_designs
    "bindcraft":                64,      # max_trajectories cap (1 traj ≈ 1-2 GPU-h)
    "boltzgen":                 256,     # num_designs × budget cap
}


# llm-002 (2026-06-18): nsteps default is 400, matching the EXECUTOR's behavior when
# nsteps is omitted (the YAML step_checkpoints default [0,100,200,300,400] = 400
# steps; controller._build_complexa_overrides). The old 200 here made
# compute_eval_budget under-count an omitted-nsteps launch by 2x, so the LLM's budget
# reasoning (defaults_when_key_omitted / default_budget in the prompt) disagreed with
# what actually ran. Default complexa_beam budget = 400*4*4*4 = 25600 < cap 65536.
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
    # llm-002 (2026-06-18): shrink WIDTH/parallelism knobs before DEPTH (nsteps), and
    # only shrink knobs the LLM EXPLICITLY set. The old loop iterated dict-insertion
    # order (nsteps FIRST), so an over-asked beam_width/n_branch was preserved while
    # diffusion DEPTH was cut — and nsteps was even INJECTED below its default for
    # configs that never set it (a silent 400->100 depth cut) → shallow searches that
    # convert fewer strict SU per GPU-h. Now: (a) never inject/lower an omitted knob,
    # (b) reduce width knobs first and nsteps last, (c) clamp nsteps no lower than its
    # family default (cutting denoising depth hurts fidelity more than trimming width).
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
    # BoltzDesign1 removed entirely (2026-05-29): the user decided to stop
    # using it. Its integration (executor/parser/registry params) was deleted.
    available_external: tuple[str, ...] = (
        "bindcraft",
        "boltzgen",
    ),
    unavailable: tuple[str, ...] = (),
) -> CapabilityRegistry:
    """Construct the T-ReX MVP capability registry.

    Local Complexa-family + refilters are always `available`. External
    generators default to `available` (caller can degrade them after
    preflight). AlphaFold3 was removed entirely (2026-05-26) — weights not
    available, folder pruned from subgit, family deregistered from T-ReX.
    """
    # Per-family allowed_params: numeric (lo, hi) ranges or categorical lists.
    # Used by Gap A (parameter-space novelty). Ranges drawn from V5/V6.3 sweeps
    # and the Proteina-Complexa `binder_generate.yaml` search defaults.
    #
    # Complexa exposes 7 search algorithms. T-ReX surfaces them as separate
    # families so the Planner can compare them at the family level instead
    # of an LLM-chosen enum. The shared params (beam_width, n_branch,
    # nsamples, nsteps, batch_size) are repeated where relevant.
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
        "complexa_beam": {                 # search.algorithm = beam-search
            # B-020 fix (2026-05-26): removed `keep_lookahead_samples` —
            # it was in the registry but never dispatched to Hydra by
            # _exec_complexa, so any LLM proposal of this key would survive
            # the partial validator and then silently no-op at the worker.
            "beam_width": (2.0, 16.0),
            "n_branch": (2.0, 8.0),
            "nsamples": (1.0, 8.0),
            "nsteps": (100.0, 500.0),
            "batch_size": (1.0, 16.0),
            # E-001 expansion (2026-05-26): V5 exposed these refinement +
            # noise knobs as separate strategy templates (`backbone_noise_beam`,
            # `beam_then_hallucination`, `hallucination_heavy_beam`). T-ReX
            # collapses them into LLM-controllable knobs so the planner can
            # propose `backbone_noise` rescues and `sequence_hallucination`
            # refinement combinations without needing a separate family per
            # combination. Backbone-noise range matches V5's calibrated band
            # (0.10 default → 0.45 strong-noise rescue).
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
        # B-012 fix (2026-05-26): unsupported legacy Complexa aliases
        # removed — Complexa's search_factory only knows 5 algorithms and these
        # two were never implemented. Stochastic-style exploration is achieved
        # via complexa_fk_steering.temperature (low-temp) or via seed variation.
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
            # BUG-A2 fix (2026-05-26): batch_size is dispatched by both
            # _exec_complexa and _exec_complexa_async for every algorithm
            # but was missing from fk_steering and mcts registries — LLM
            # batch_size proposals were silently dropped at validation.
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
            "nsamples": (1.0, 8.0),  # added 2026-05-26 for budget consistency
            "batch_size": (1.0, 16.0),  # BUG-A2 fix (see fk_steering note)
            "sc_scale_noise": (0.05, 0.50),
            "refinement_algorithm": ["", "sequence_hallucination"],
            "n_greedy_iters": (0.0, 30.0),
            "enable_greedy_optimization": [True, False],
            "greedy_percentage": (1.0, 10.0),
            "filter_samples_limit": (1.0, 1000.0),
            **COMPLEXA_REWARD_PARAMS,
        },
        # HIGH 1 fix (2026-05-26): registry keys MUST match executor cd.get keys
        # in controller.py exactly. Same pattern as B-010/B-020 — when
        # the LLM proposes a key the executor doesn't read, validation accepts
        # it (it's in the registry) but the worker silently uses defaults. This
        # made LLM control of Phase 2 backends a no-op for 4 of 5 families.
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
            # Min 2 (smaller = no useful signal even for stalled-fallback bounded
            # probe per smoke 8736558 calibration: 0.58 traj/h ⇒ tj=2 ≈ 3-4 h).
            # Max 64: pairs with EVAL_BUDGET_CAP_PER_FAMILY['bindcraft']=64.
            # §7.0 stalled-fallback uses 4 (bounded probe);
            # productive/rescue can scale up to 16-32 with promotion.
            "max_trajectories": (2.0, 64.0),
            "target_length_min": (40.0, 120.0),
            "target_length_max": (60.0, 200.0),
            # E-002 expansion (2026-05-26): high-value BindCraft advanced
            # settings that V5 hardcoded but T-ReX surfaces as LLM knobs.
            # See subgit/BindCraft/settings_advanced/default_4stage_multimer
            # _mpnn_hardtarget.json for the underlying defaults that T-ReX
            # overrides via a per-launch settings_advanced.json. These are
            # the knobs that directly enable rescue patterns the LLM was
            # asking for ("pLDDT good but iPAE bad → raise weights_pae_inter").
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
            # HIGH 1 fix: align with _exec_boltzgen_async cd.get keys
            # fix20: protocol enum must match `boltzgen run --protocol` whitelist.
            # `protein-protein` is NOT valid — was a fix19 fabrication that made
            # every LLM-chosen run with that protocol exit 2 (CD45 r1/r2,
            # SC2RBD r1 in fix19).
            # Only expose protocols whose required YAML semantics are implemented
            # by the T-ReX launcher. protein-redesign needs explicit design masks;
            # selecting it here would look adaptive while running with incomplete
            # semantics.
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
        # B-012 fix (2026-05-26): unsupported legacy Complexa aliases
        # capability entries removed — see LOCAL_PARAMS comment for context.
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
        # BindCraft: PROVENANCE FIX (2026-05-31). Its native metrics come from
        # the AF2 prediction it OPTIMIZED AGAINST (best-of-5, optimistic) and
        # its CSV "Binder_RMSD" is cross-model/monomer RMSD, NOT the paper's
        # binder-in-complex Cα scRMSD. The Proteina-Complexa paper (App. F)
        # scores ALL methods — incl. BindCraft — under one INDEPENDENT
        # ColabDesign AF2 re-fold (initial-guess + templating). So treat
        # BindCraft as diagnostic-only and chain parseable artifacts through
        # structure_refilter (= that same independent AF2) for the strict gate.
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
            notes="external generator — first use is diagnostic, never extended (plan §12.4)",
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
