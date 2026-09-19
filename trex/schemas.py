"""T-ReX record dataclasses (frozen).

All records are append-only JSONL in the archive. Required fields match
plan §3. Optional later records (PairwiseComparison, MetaReview,
CampaignPrior) are present as stubs but have no MVP producer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ModeName = Literal["exploit", "rescue", "explore"]
ResourceClass = Literal["low", "diagnostic", "standard", "extended"]
RefilterRole = Literal[
    "canonical_score_conversion",
    "parent_model_refold",
]
StateLabel = Literal[
    "productive", "productive_duplicate", "strict_duplicate_collapse",
    "rescue_rich", "stalled", "deep_stall", "low_evidence",
]
# PredictedChange.axis must be exactly one of the 3 success axes (plan §8 schema fix, 2026-05-25).
# "diversity" was removed because it is a panel-level concept, not a per-result metric.
# PreserveConstraint is also limited to strict axes; diversity pressure is handled
# through route/panel evidence, not lifecycle preserve arithmetic.
AxisName = Literal["pLDDT", "iPAE", "binder_scRMSD"]
PreserveAxisName = Literal["pLDDT", "iPAE", "binder_scRMSD"]
CalibrationStatus = Literal["frozen", "provisional", "uncalibrated"]
# §8 schema fix (2026-05-25): LifecycleStatus narrowed from 6 → 4 values.
# "selected" and "queued" were workflow phases never referenced by lifecycle
# arithmetic; if needed they belong on ActionCandidate, not HypothesisCard.
LifecycleStatus = Literal["active", "supported", "contradicted", "retired"]
ParentStratum = Literal[
    "strict_parent",
    "near_pass_parent",
    "weak_scientific_parent",
    "raw_only_external",
    "infrastructure_only_artifact",
]
CreditStatus = Literal["provisional_until_control", "credited", "no_credit"]


def _to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Calibration / runtime / target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricCalibration:
    metric: str
    source: str
    backend: str
    model_digest: str
    calibration_status: CalibrationStatus
    threshold_raw: float
    threshold_calibrated: float
    direction: Literal["increase", "decrease"]
    near_pass_margin: float
    fit_data_scope: str
    fit_date: str


@dataclass(frozen=True)
class RuntimeBucket:
    bucket_id: str
    container_digest: str | None
    ckpt_digests: dict[str, str]
    scoring_script_digest: str | None
    calibration_version: str
    created_at: str


@dataclass(frozen=True)
class TargetConstraint:
    target_id: str
    target_class: str
    hotspots: list[str] = field(default_factory=list)
    forbidden_surfaces: list[str] = field(default_factory=list)
    chain_ids: list[str] = field(default_factory=list)
    assay_geometry: str | None = None
    developability_filters: dict[str, Any] = field(default_factory=dict)
    panel_size_K: int = 8


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultRecord:
    result_id: str
    parent_ids: list[str]
    target_id: str
    backend_family: str
    runtime_bucket_id: str
    metrics: dict[str, float]
    metrics_calibrated: dict[str, float]
    route_lineage: list[str]
    gpu_h: float
    exit_status: Literal["ok", "timeout", "nonzero_exit", "no_artifacts"]
    bins: dict[str, str] = field(default_factory=dict)  # foldseek/contact/epitope/seq
    artifacts: dict[str, str] = field(default_factory=dict)  # PDB/CIF/score paths
    panel_ready: bool = False
    tick_id: str | None = None


# ---------------------------------------------------------------------------
# Evidence summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisStat:
    pass_count: int
    near_pass_count: int
    fail_count: int
    median_raw: float | None
    median_calibrated: float | None
    median_deficit: float | None
    calibration_status: CalibrationStatus
    n: int
    # Two-tier diagnostic axes (v7_3, §4.1 redesign). For DIAGNOSTIC axes
    # pass_count/near_pass_count/fail_count are classified against
    # ``quality_threshold`` (the stricter, discriminative good-interface band);
    # ``below_accept_count`` is the subset of records that fail the tool's
    # OFFICIAL accept floor ``pass_threshold`` (genuine non-binders). These stay
    # at their defaults for the strict-gate axes (pLDDT/iPAE/binder_scRMSD),
    # which are single-tier. pass_threshold=None => quality-only axis (the tool
    # has no real accept gate, e.g. BindCraft dSASA>=1 / pTM=null).
    pass_threshold: float | None = None
    quality_threshold: float | None = None
    below_accept_count: int = 0
    # Diagnostic axes are source-scoped: e.g. design_to_target_iptm is a
    # BoltzGen-native axis, while min_ipae/ipTM usually come from Complexa
    # records. Keep carrier provenance with the aggregate so the LLM cannot
    # silently translate one family's diagnostic blocker into another family's
    # unrelated knob.
    source_families: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class JointPatternCount:
    axes: tuple[str, str]
    pattern: Literal[
        "both_pass", "both_fail", "A_pass_B_fail", "A_fail_B_pass", "both_near_pass"
    ]
    count: int


@dataclass(frozen=True)
class MethodHealthSummary:
    family: str
    attempts: int
    completions: int
    timeouts: int
    nonzero_exits: int
    raw_artifacts: int
    accepted_artifacts: int
    score_files: int
    strict_yield: int
    near_miss_yield: int
    routed_proxy: float | None
    # fix21 (2026-05-26): cumulative GPU-hours invested in this family
    # across the whole run. Lets the Planner reason about under-explored
    # families that genuinely need more budget before being judged. Key
    # use case: BindCraft typically requires ≥1.5–2 cumulative GPU-h on
    # one target before the first strict_success appears (V5/V6.3
    # empirical). Without this, the LLM sees `strict_yield=0, attempts=3`
    # and prematurely deprioritises BindCraft when it has only been
    # given 0.3 gpu-h.
    cumulative_gpu_h: float = 0.0
    # §22.8.11 (2026-05-27 PM): per-family Foldseek-deduped SU count.
    # strict_yield counts RAW strict-passing records — but 7 strict from
    # the same Foldseek cluster only count as 1 unique winner (the run-
    # level run_su_count is already SU-deduped). Without this, the LLM
    # sees `complexa_beam: strict_yield=7` and thinks the family is
    # producing diverse winners when actually all 7 are the same structure.
    # User's primary optimisation target is SU, not raw strict.
    strict_yield_su: int = 0
    # 2026-05-28: the resource-credit signal. SU per GPU-hour = how much
    # structurally-unique success this family bought per unit of the budget
    # it consumed on THIS target. This is the first-class explore/exploit
    # credit: a family that produced SU quickly (high su_per_gpu_h) earns
    # more budget than a slow one, and BindCraft's low-but-nonzero value on
    # a hard target still beats Complexa's zero there. Computed as
    # strict_yield_su / max(cumulative_gpu_h, eps). None until any GPU-h
    # is spent. NOTE: this is a LIFETIME (cumulative-over-run) average — it
    # never decays, so a family that has gone dry keeps a stale-high value.
    su_per_gpu_h: float | None = None
    # 2026-05-30 (G-033): WINDOW-scoped recent versions of the two exploit-
    # ranking signals. The cumulative fields above never decay, so the prompt's
    # "keep exploiting a productive family" rule fired forever on a family that
    # stopped producing. These are computed over the recent window and are the
    # ones the prompt should rank exploit on; the cumulative fields stay as the
    # "lifetime" reference.
    su_per_gpu_h_recent: float | None = None
    near_miss_yield_recent: int = 0
    # SU produced by downstream canonical refiltering of this family's
    # diagnostic-only outputs. Example: BindCraft accepted design →
    # structure_refilter strict success should teach the Planner that the
    # BindCraft settings worked, not only that "structure_refilter" scored.
    chained_strict_yield_su: int = 0
    chained_su_per_gpu_h: float | None = None
    # 2026-05-31 (F7): WINDOW-scoped recent versions of the chained-credit
    # signals — the diagnostic-lane analogue of su_per_gpu_h_recent. The
    # cumulative chained fields above never decay, so a diagnostic generator
    # (BindCraft/BoltzGen/MPNN) that produced chained SU early but has since
    # gone dry kept a stale-high lifetime chained rate, and the prompt judges
    # those lanes BY the chained credit → it could keep funding a dry lane.
    # chained_strict_yield_su_recent counts chained SU clusters FIRST SEEN in
    # the window (marginal, like su_per_gpu_h_recent); chained_su_per_gpu_h_recent
    # divides it by the recent ROUTE GPU-h (upstream + downstream refilter, in
    # the window). These are the ones the prompt should rank diagnostic-lane
    # exploit on; the cumulative chained fields stay as the lifetime reference.
    chained_strict_yield_su_recent: int = 0
    chained_su_per_gpu_h_recent: float | None = None


@dataclass(frozen=True)
class RouteHealthSummary:
    raw_routed: int
    score_files_completed: int
    backlog_used: int
    backlog_cap: int
    near_miss_conversion: float | None
    strict_conversion: float | None
    panel_ready_conversion: float | None


@dataclass(frozen=True)
class LLMHealthSummary:
    model: str
    last_calls_window: list[
        Literal[
            "ok", "parse_fail", "schema_fail", "timeout",
            "call_error", "no_hypotheses_or_candidates",
            "abstain", "low_conf",
        ]
    ]
    parse_fail_rate: float
    schema_fail_rate: float
    timeout_count: int
    median_latency_s: float


@dataclass(frozen=True)
class Exemplar:
    """A concrete best / near-miss binder WITH the full setup that produced it
    (2026-05-30). Joins provenance (family / operator / config_delta) to the
    individual binder's FULL metric vector + per-axis deficits — strictly richer
    than `Recipe` (config-grouped, 3-axis median, no individual binder) and
    `Example` (individual binder, but no config/operator). Lets the Planner see
    "this exact setup produced this exact result", for both the K best proven
    binders (build on them) and the K closest near-misses (diagnose + route the
    remediation by the blocking axis). `kind`: "best" = strict success ranked by
    margin past thresholds (deduped by SU bin → distinct structures);
    "near_miss" = closest-to-passing failures (smallest normalized dominant
    deficit; deduped by (operator, config) signature → diverse failing setups).
    Literal all-axes-fail records are excluded (uninformative noise); known-bad
    *configs* live in `recipes` (joint_fail)."""
    kind: str                                  # "best" | "near_miss"
    result_id: str
    family: str
    operator_id: str | None
    config_delta: dict[str, Any]
    metrics: dict[str, float]                  # FULL vector (3 strict + any diagnostic axes)
    axis_deficits: dict[str, float]
    dominant_deficit_axis: str | None          # worst STRICT axis (pLDDT/iPAE/scRMSD)
    parent_result_id: str | None
    # v7_3: worst ACTIONABLE diagnostic axis for THIS design (vs its quality band),
    # or None. Levered axes only (see DIAGNOSTIC_AXIS_REMEDIATION) so the planner
    # can route remediation — e.g. a near-miss passing all 3 strict axes but
    # blocked by ipTM=0.65 carries diagnostic_blocking_axis="ipTM".
    diagnostic_blocking_axis: str | None = None
    su_bin: str | None = None                  # Foldseek/refilter_source bin (best only)


@dataclass(frozen=True)
class Example:
    result_id: str
    family: str
    parent_id: str | None
    axis_values: dict[str, float]
    axis_deficits: dict[str, float]
    joint_pattern_label: str | None
    # 2026-05-30 (do-now rec 1): the single dominant (worst, margin-normalized)
    # failing axis for THIS candidate — pLDDT (structure), iPAE (interface), or
    # binder_scRMSD (sequence). None when the candidate passes all three. Lets the
    # Planner route a per-candidate remediation (seq-hallucinate / interface-noise /
    # regenerate) instead of reading only aggregate axis_stats. Computed from the
    # already-present axis_deficits — no new metric.
    dominant_deficit_axis: str | None = None


RecipeClass = Literal["strict_success", "panel_ready", "near_miss", "joint_fail"]


@dataclass(frozen=True)
class Recipe:
    """Past (operator, config_delta) signature with its observed outcome class.

    Derived view of the archive — never written as a standalone record.
    EvidenceReducer extracts the top-K per recipe_class on each tick and
    surfaces them in EvidenceSummary.recipes so the Planner can build on
    proven configurations rather than re-inventing them.
    """

    recipe_hash: str
    operator_id: str
    method_family: str
    config_delta: dict[str, Any]
    recipe_class: RecipeClass
    target_id: str
    target_class: str | None
    descendant_count: int
    median_metrics: dict[str, float]
    representative_result_ids: list[str]
    recency_tick: int
    # 2026-05-30 (do-now rec 2): route-level SPEED credit. SU (Foldseek-deduped
    # strict) produced by THIS exact (operator, config_delta) recipe per GPU-hour
    # it consumed. The user's objective is SU per unit time, so among recipes that
    # all yield SU the FASTER one should earn more exploit weight — and credit must
    # be per-ROUTE (e.g. beam+hallucinate vs plain beam), not just per-family.
    # None for non-strict recipe classes (they produced no SU). Only set on
    # strict_success recipes.
    su_per_gpu_h: float | None = None


@dataclass(frozen=True)
class RouteValueSummary:
    """Cost-normalized value for an observed adaptive route.

    This is more specific than MethodHealthSummary. Family-level rows answer
    "which backend family is paying off?", while route rows answer "which exact
    config / parent-context route is buying NEW Foldseek SU per worker-GPU-h?".
    Canonical structure_refilter score-conversion is charged to the upstream
    route; structure_refilter itself remains scoring plumbing unless the row is
    an intentional parent_model_refold advisory route.
    """

    strategy_key: str
    scope: str                           # "family" | "route"
    family: str
    root_family: str | None
    action_family: str
    scoring_family: str | None
    operator_id: str | None
    config_signature: str
    # Human/audit-facing route role. This prevents the physical backend name
    # (especially structure_refilter) from hiding the scientific action:
    # official AF2 score conversion or intentional AF2 parent-model refold.
    route_role: str | None = None
    # Raw refilter role when a refilter-like action/scorer participated in this
    # route; kept alongside route_role for exact audit/provenance checks.
    refilter_role: str | None = None
    config_delta: dict[str, Any] = field(default_factory=dict)
    parent_strategy_key: str | None = None
    route_gpu_h: float = 0.0
    generator_gpu_h: float = 0.0
    canonical_refilter_gpu_h: float = 0.0
    # Number of completed canonical AF2 score conversions attributed to this
    # route at the time of this evidence snapshot.  The controller uses this
    # immutable snapshot to advance one 4->8->16->32 tranche per evidence update
    # instead of repeatedly expanding from the same stale signal.
    canonical_score_conversion_count: int = 0
    attempts: int = 0
    completions: int = 0
    strict_count: int = 0
    new_su: int = 0
    # Record-window marginal SU signal. The record_recent window is the last N
    # ResultRecords, which is responsive but can be misleading for delayed
    # score-conversion routes unless the generator parent GPU-h is charged.
    # Decision paths must prefer gpu_recent or a guarded fallback; medium_recent
    # is a delayed-feedback guard, not the primary rank score.
    record_recent_new_su: int = 0
    record_recent_route_gpu_h: float = 0.0
    record_recent_new_su_per_route_gpu_h: float | None = None
    # Backward-compatible aliases for archives written before the explicit
    # record_recent naming. New code should read/write record_recent_* first and
    # keep these in sync for replay compatibility.
    new_su_recent: int = 0
    new_su_per_route_gpu_h: float | None = None
    recent_route_gpu_h: float = 0.0
    recent_new_su_per_route_gpu_h: float | None = None
    # Worker-GPU-hour recent signal. This complements the record window above so
    # cheap high-throughput routes are not compared to slow routes on a different
    # notion of "recent".
    new_su_recent_gpu: int = 0
    gpu_recent_route_gpu_h: float = 0.0
    gpu_recent_new_su_per_route_gpu_h: float | None = None
    # Secondary worker-GPU-hour window for delayed-feedback routes. This is
    # evidence only: Selector uses it to avoid premature cooldown, not as the
    # primary rank score over the short gpu_recent window.
    medium_recent_new_su: int = 0
    medium_recent_route_gpu_h: float = 0.0
    medium_recent_new_su_per_route_gpu_h: float | None = None
    near_miss_count: int = 0
    near_miss_recent: int = 0
    strict_per_su: float | None = None
    duplicate_bin_fraction: float | None = None
    # promote | healthy | diversify | defer | collapse_risk | untried | plumbing
    # diversify = productive route that is still buying SU, but strict binders are
    # starting to collapse into too few Foldseek bins; keep it alive while
    # penalizing same-route repeats.
    status: str = "observed"
    # Marginal status is the SU/GPU-h decision label. Duplicate features are only
    # warnings here: "dry_duplicate" requires duplicate pressure AND decayed
    # marginal new-SU/GPU-h, while "productive_but_duplicate" means keep buying
    # SU from the route.
    marginal_status: str = "observed"
    # Diagnostic-only generators do not prove or fail strict/SU until their
    # accepted artifacts are canonical AF2 score-converted. Keep this visible so
    # the LLM and Selector do not treat an unscored route as scientifically dry.
    pending_score_conversion_count: int = 0
    # Subset of pending_score_conversion_count with native/proxy evidence
    # strong enough to protect awaiting_score_conversion status. Bulk
    # low-value backlog stays visible but must not keep a dry route alive.
    pending_promising_score_conversion_count: int = 0
    # Canonical cross-family quality, evaluated once per trusted Foldseek bin.
    # These fields are secondary evidence only: SU/GPU-h remains primary.
    strict_quality_n_unique_bins: int = 0
    strict_quality_median: float | None = None
    strict_quality_p25: float | None = None
    strict_quality_axis_margins: dict[str, float] = field(default_factory=dict)
    # Auxiliary route-level diagnostic progress. This never mints SU and never
    # outranks new SU/GPU-h; it keeps a no-SU route scientifically credible when
    # normalized diagnostic axes are near-pass or improving.
    diagnostic_improvement_score: float = 0.0
    diagnostic_improvement_axes: list[str] = field(default_factory=list)
    diagnostic_improvement_n: int = 0
    evidence_refs: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Frozen dataclass normalization: keep the explicit record_recent API and
        # the legacy recent aliases synchronized when reading old/new JSONL.
        if self.record_recent_new_su == 0 and self.new_su_recent:
            object.__setattr__(self, "record_recent_new_su", int(self.new_su_recent))
        elif self.new_su_recent == 0 and self.record_recent_new_su:
            object.__setattr__(self, "new_su_recent", int(self.record_recent_new_su))

        if self.record_recent_route_gpu_h == 0.0 and self.recent_route_gpu_h:
            object.__setattr__(self, "record_recent_route_gpu_h", float(self.recent_route_gpu_h))
        elif self.recent_route_gpu_h == 0.0 and self.record_recent_route_gpu_h:
            object.__setattr__(self, "recent_route_gpu_h", float(self.record_recent_route_gpu_h))

        if self.record_recent_new_su_per_route_gpu_h is None and self.recent_new_su_per_route_gpu_h is not None:
            object.__setattr__(
                self,
                "record_recent_new_su_per_route_gpu_h",
                self.recent_new_su_per_route_gpu_h,
            )
        elif self.recent_new_su_per_route_gpu_h is None and self.record_recent_new_su_per_route_gpu_h is not None:
            object.__setattr__(
                self,
                "recent_new_su_per_route_gpu_h",
                self.record_recent_new_su_per_route_gpu_h,
            )


@dataclass(frozen=True)
class EvidenceSummary:
    tick_id: str
    target_id: str
    target_class: str
    schema_version: str

    # budget
    elapsed_wall_h: float
    remaining_wall_h: float
    completed_children: int
    pending_children: int
    worker_gpu_h_total: float
    worker_gpu_h_last_3_ticks: float

    # success / novelty
    strict_count: int
    global_new_strict: int
    run_su_count: int
    run_su_count_delta: int
    su_per_gpu_h_recent: float | None
    duplicate_fraction: float | None
    top_bin_share: float | None

    # axis + joint (3 success axes — §2.5 SSOT)
    axis_stats: dict[str, AxisStat]
    joint_patterns: list[JointPatternCount]
    near_miss_count: int

    # panel
    panel_ready_count: int
    panel_ready_bins_covered: int

    # health
    method_health: dict[str, MethodHealthSummary]
    route_health: RouteHealthSummary
    llm_health: LLMHealthSummary

    # state classifier
    state_label: StateLabel

    # representative examples (≤ 6)
    examples: list[Example]

    # missingness map
    metric_availability: dict[str, dict[str, bool]]

    # Strict/SU redundancy diagnostics. These are objective-aligned: strict_count
    # is raw canonical strict successes, while run_su_count is SU Foldseek
    # SU@TM0.6. A high ratio means the loop is making strict binders but re-mining
    # the same structural clusters, so raw strict_rate must not drive exploit.
    strict_per_su_total: float | None = None
    strict_duplicate_collapse_signal: bool = False

    # Diagnostic-only generators (BindCraft/BoltzGen/ProteinMPNN) need a
    # downstream structure_refilter record before they can receive canonical
    # strict/SU credit. This archive-derived snapshot prevents "0 chained SU"
    # from being misread when many accepted artifacts are still unscored.
    diagnostic_chain_backlog: dict[str, Any] = field(default_factory=dict)

    # Event-driven admission pressure for not-yet-completed work. MethodHealth
    # and route_values are completed-result ledgers; slow high-cost families can
    # otherwise look "untried" while several workers are already running. Shape:
    # {by_family: {fam: {running, queued, pending_total, inflight_gpu_h}},
    #  high_cost_families: [...]}.
    pending_family_load: dict[str, Any] = field(default_factory=dict)

    # Structure-refilter role accounting. The physical backend may be the same
    # AF2 refold executor, but the scientific meaning is different:
    # automatic canonical score conversion after a diagnostic generator is not
    # the same as an LLM-selected parent-model refold/refinement action.
    refilter_role_health: dict[str, Any] = field(default_factory=dict)

    # archive-derived recipes (Gap A'): proven configurations + recent failures
    recipes: list[Recipe] = field(default_factory=list)

    # Strategy-level outcome ledger for Planner feedback. Unlike `recipes`
    # (top-K per outcome class) and `exemplars` (individual best/near-miss
    # binders), this keeps compact counts of successes, near-misses, and
    # failures for each observed (family, operator, config_delta) strategy so
    # the Planner can exploit what worked, modify what nearly worked, and avoid
    # repeating clearly bad settings.
    strategy_feedback: list[dict[str, Any]] = field(default_factory=list)

    # Route-value ledger for adaptive allocation. Rows are registry/lineage
    # driven and include exact config signatures, so e.g. complexa_beam
    # beam_width=4 and beam_width=8 compete empirically. For
    # proteinmpnn_redesign routes, root_family preserves the original generator
    # that made the parent backbone while action_family remains
    # proteinmpnn_redesign.
    route_values: list[RouteValueSummary] = field(default_factory=list)

    # 2026-05-30: best-K / near-miss-K concrete binders, each WITH the full setup
    # (family/operator/config_delta) that produced it AND its full metric vector.
    # The per-binder join of provenance↔result that neither `recipes` (config-
    # grouped, 3-axis median) nor `examples` (binder, no config) provides.
    exemplars: list[Exemplar] = field(default_factory=list)

    # 2026-05-30 (do-now rec 4): deterministic give-up-and-regenerate signal.
    # Each entry flags a PARENT backbone that refinement is no longer paying off
    # on (>=K non-improving refinement children; K=1 when the parent's dominant
    # block is pLDDT, since a bad fold cannot be fixed by sequence/interface
    # refinement on a fixed backbone). The builder marks further refinements of
    # these parents infeasible, and the Planner is told to regenerate from
    # scratch (fresh family/seed, no parent) rather than keep refining.
    # Shape per entry: {root_result_id, family, dominant_axis, attempts, reason}.
    stuck_lineage_roots: list[dict] = field(default_factory=list)

    # Recent-fallback signal (formerly `mcr_trigger_high_fallback`, renamed
    # 2026-05-26 PM when MCR was fully removed — see plan §10.6). True when
    # recent_fallback_rate >= 0.30. Consumed by Selector §22.8.4 to gate
    # Category B clamps.
    recent_fallback_high: bool = False

    # Diagnostic axes (§4.1, added 2026-05-26). NOT in strict-success rule;
    # surfaced to the LLM for finer scientific hypothesis reasoning. The reducer
    # emits an axis only after enough finite observations (DIAGNOSTIC_MIN_N) in
    # its family-balanced diagnostic window; sparse n<3 axes are intentionally
    # omitted rather than over-weighted. Axes span strict gate support metrics,
    # Complexa interface/proxy terms, BindCraft interface physics, and refilter
    # diagnostic reads. See evidence_reducer.DIAGNOSTIC_AXIS_THRESHOLDS and
    # DIAGNOSTIC_AXIS_REMEDIATION for the current SSOT.
    diagnostic_axis_stats: dict[str, AxisStat] = field(default_factory=dict)
    # Compact deterministic digest of diagnostic_axis_stats shown to Planner and
    # Supervisor. This is stored in the archive so audits can recover the
    # decision-critical diagnostic summary without reconstructing a historical
    # prompt string. "none" means no qualified diagnostic driver was available.
    diagnostic_driver_tldr: str = "none"

    # L-001 (2026-05-26): tick-by-tick trajectory across the current run.
    # Each entry shows how the state evolved and what was launched, so the
    # Planner LLM sees the WHOLE loop (not just the most-recent
    # EvidenceSummary). Entry shape (free-form to stay compact):
    #   {tick_id, state_label, strict_count, run_su_count, launches:
    #    [{family, mode, key_knobs}], dominant_family, notes}.
    # Capped at the last ~10 ticks. Motivated by qwen_trajectory_history
    # _smoke (job 8747422) which showed trajectory 2/4 > compact 1/4 lift.
    recent_ticks_history: list[dict] = field(default_factory=list)
    # v7_3 dispatch-realization audit (2026-06-12): LaunchDecision is selector
    # intent, while DispatchRecord is actual worker start/failure. This compact
    # block lets the Planner/Supervisor and offline audits see whether the LLM
    # E/R/E distribution is becoming real GPU-started work or being lost in a
    # queue/stale-prefetch layer.
    dispatch_realization: dict[str, Any] = field(default_factory=dict)

    # Family-level execution realization, derived from archived
    # SupervisorDecision -> LaunchDecision -> DispatchRecord. This is deliberately
    # separate from method_health: it distinguishes Supervisor proposals and
    # Selector choices from work that actually reached a worker. Unselected
    # proposals are allocation outcomes, not evidence of execution blockage.
    # Shape:
    # {by_family: {fam: {proposed, selected, started, dispatch_deferred,
    #  selected_not_started, proposal_to_start_gap, selection_to_start_gap,
    #  defer_reasons}},
    #  under_started_families: [...]}.
    execution_realization: dict[str, Any] = field(default_factory=dict)


    # Diagnostic native-generator scores live in r.bins (because they are not
    # AF2-calibrated and do not feed strict_success). Exposing these advisory
    # values lets the Planner compare generator families before requesting
    # canonical AF2 score conversion.
    # Shape: { backend_family: { score_key: {median, count, direction, ...} } }
    #   e.g. {"boltzgen": {"design_iptm": {"median": 0.82, "count": 12}}}
    diagnostic_alt_model_scores: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )

    # review #1 (2026-05-31): SU-dedup provenance, so a reader/LLM can tell when
    # run_su_count is genuinely Foldseek-deduped vs incomplete/untrusted.
    # foldseek_su_status: the strict-only clustering status this tick
    #   ("ok" | "no_binary" | "failed" | "no_structures" | "no_strict" | "disabled").
    # foldseek_su_coverage: fraction of strict records that received a foldseek_su
    #   cluster bin. No fallback is allowed for official SU; records without this
    #   bin stay visible as strict/quality evidence but do not mint SU/new-SU.
    foldseek_su_status: str = "ok"
    foldseek_su_coverage: float | None = None
    # Top SU Foldseek-SU bin share among recent raw strict records. This is
    # separate from top_bin_share, which is computed over all scored structures.
    # The selector can use the max of both signals: all-scored collapse catches
    # failure-mode convergence; strict-record concentration catches mined-out
    # live SU bins.
    strict_su_top_bin_share: float | None = None
    # Auxiliary high-TM fine-diversity signal. The headline objective remains
    # full-run live SU@TM0.6; these fields are a bounded recent-window
    # TM0.8 pass over strict successes only, used to tell the LLM whether a
    # productive TM0.6 basin is also spreading into fine-grained TM0.8 bins.
    # They must never replace run_su_count or su_per_gpu_h_recent.
    strict_su_tm08_status: str = "disabled"
    strict_su_tm08_coverage: float | None = None
    strict_su_tm08_recent_count: int | None = None
    strict_su_live_recent_count: int | None = None
    strict_su_tm08_delta_vs_live: int | None = None
    strict_su_tm08_live_split_ratio: float | None = None
    # Legacy archive field names from the TM0.5-objective era. In T-ReX TM0.6
    # runs they mirror the live_* fields so older readers do not break.
    strict_su_tm05_recent_count: int | None = None
    strict_su_tm08_delta_vs_tm05: int | None = None
    strict_su_tm08_split_ratio: float | None = None
    strict_su_tm08_result_scope: str = "disabled"
    # Structural dedup scoping. Current production should use binder_chain; older
    # evidence rows reconstruct as legacy_or_unknown.
    structure_dedup_scope: str = "legacy_or_unknown"
    structure_dedup_fallback_count: int = 0
    # Whole-archive Foldseek provenance for duplicate_fraction/top_bin_share and
    # near-miss dedup. Kept separate from strict-only foldseek_su_* because the
    # two Foldseek calls can degrade independently.
    foldseek_archive_status: str = "legacy_or_unknown"
    foldseek_archive_coverage: float | None = None
    # R1 (2026-06-01): scope of the whole-archive collapse pass that produces
    # duplicate_fraction / top_bin_share. SAFE default for legacy reads;
    # live_tick sets "recent_scored_window_{K}" once the pass is windowed, so the
    # planner reads these as a RECENT-collapse signal, not lifetime dedup.
    foldseek_archive_result_scope: str = "lifetime_or_unknown"
    whole_archive_structure_dedup_scope: str = "legacy_or_unknown"
    whole_archive_structure_dedup_fallback_count: int = 0
    # Exact near-miss-only structural dedup provenance. This is separate from
    # foldseek_archive_* because cumulative per-family near_miss_yield must not
    # depend on a whole-archive/window collapse pass.
    near_miss_dedup_status: str = "legacy_or_unknown"
    near_miss_dedup_coverage: float | None = None

    # Sequence-diversity evidence for wet-lab panel selection. These metrics do
    # NOT change strict_success or run_su_count; they tell the Planner whether
    # the structurally unique winners are also sequence-diverse.
    # sequence_dedup_status: "ok" | "no_binary" | "failed" | "no_sequences" |
    #   "no_strict" | "disabled".
    sequence_dedup_status: str = "disabled"
    sequence_dedup_coverage: float | None = None
    seq_unique_strict_count: int | None = None
    seq_unique_strict_delta: int | None = None
    joint_struct_seq_unique_count: int | None = None
    seq_duplicate_fraction: float | None = None
    top_seq_bin_share: float | None = None

    # LIVE HEADLINE OBJECTIVE (v7_3 2026-07-03): SU per WORKER-WALL GPU-h
    # (elapsed wall-clock × worker slots; excludes the vLLM/controller GPU).
    # This is the fair live throughput denominator for a 3-worker controller.
    #
    # Route feedback still uses completed worker GPU-h (sum of ResultRecord.gpu_h)
    # because per-route attribution must charge only jobs that actually ran.
    # Keep both fields explicit so run-level reporting and route-level learning
    # cannot silently swap denominators.
    run_su_per_worker_gpu_h_total: float | None = None
    worker_wall_gpu_count: float | None = None
    worker_wall_gpu_h_total: float | None = None
    run_su_per_worker_wall_gpu_h_total: float | None = None
    run_su_hwm: int | None = None
    run_su_hwm_delta: int | None = None
    run_su_hwm_per_worker_wall_gpu_h_total: float | None = None
    # F5 plateau governor (2026-06-10): worker GPU-h / ticks since the SU
    # high-water-mark was last raised (running-max, robust to the SU re-cluster
    # dip). deep_stall + the chain-refilter throttle key on gpu_h_since_last_su;
    # both surface to the LLM so it sees plateau DURATION, not just `stalled`.
    gpu_h_since_last_su: float | None = None
    ticks_since_last_su: int | None = None
    # Charged accounting is retained only as raw reservation-overhead metadata
    # (wall × reserved GPUs, incl. vLLM + idle worker time). It is not used as
    # a throughput denominator or decision objective.
    charged_gpu_count: float | None = None
    charged_gpu_h_total: float | None = None
    charged_gpu_h_recent: float | None = None
    charged_gpu_h_scope: str = "unavailable"
    run_su_per_charged_gpu_h_total: float | None = None
    run_su_per_charged_gpu_h_recent: float | None = None

    # Production wet-lab panel snapshot. This is distinct from the online
    # optimisation objective (SU/GPU-h): it tells the Planner/operator what the
    # current deterministic top-N strict-success panel would be after applying
    # quality margin + structural/sequence/contact/source diversity. Near-miss
    # backups are tracked separately and never count as strict successes.
    production_panel_status: str = "not_run"
    production_panel_value: float | None = None
    production_panel_selected_ids: list[str] = field(default_factory=list)
    production_panel_diversity_bins: dict[str, int] = field(default_factory=dict)
    production_panel_gap_reasons: list[str] = field(default_factory=list)
    production_near_miss_ids: list[str] = field(default_factory=list)
    # Concrete result_ids that still have a usable PDB/CIF artifact for
    # parent-required actions (proteinmpnn_redesign and structure_refilter).
    # Builder uses this as a launch gate so
    # LLM-cited concrete parents cannot pass feasibility and then skip at
    # dispatch because the structure artifact is missing.
    parent_artifact_result_ids: list[str] = field(default_factory=list)
    # v7_3 prototype #3 (diagnosis→outcome loop): per blocked-axis remediation
    # outcome over the archive lineage — {axis: {lever, attempts, improved,
    # improve_rate, strict, strict_rate}}. Lets the Planner learn which
    # qualitative diagnoses + remediations actually pay off. Advisory only.
    diagnosis_outcomes: dict[str, dict[str, Any]] = field(default_factory=dict)

    # refold-probe loop closure (#10, 2026-06-13). A parent_model_refold is an
    # advisory re-fold of an existing parent design under a friendlier AF2 config;
    # its axis values are recorded as refold_* and NEVER mint SU. Those numbers
    # used to be written but read by ZERO consumers, so the LLM could be told to
    # fire the probe yet never saw its answer next tick (a dead loop). This
    # aggregate JOINS each advisory refold to its parent's CANONICAL strict metrics
    # (via bins["refilter_source"]) and reports, over the recent window:
    #   probed            : # advisory refold probes observed
    #   structure_limited : # whose refold WOULD pass the strict gate while the
    #                       parent's canonical score did NOT -> the predicted
    #                       backbone was the bottleneck, the sequence is fine ->
    #                       REGENERATE / proteinmpnn_redesign that parent to mint
    #                       SU (re-folding it again cannot, the gate is fixed).
    #   confirmed_limited : # whose refold ALSO fails -> not structure-limited;
    #                       the design itself is the problem (pivot family/seed).
    #   example_parent_ids: up to ~5 structure_limited parent result_ids to act on.
    # Advisory only; never changes strict_success or run_su_count.
    refold_probe_outcomes: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hypothesis + actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedChange:
    axis: AxisName
    direction: Literal["increase", "decrease"]
    baseline_refs: list[str]
    min_relative_deficit_reduction: float
    min_absolute_delta: float | None


@dataclass(frozen=True)
class PreserveConstraint:
    # axis may be any strict success axis. Panel-level diversity is evidence,
    # not a lifecycle preserve axis.
    axis: PreserveAxisName
    max_relative_deficit_increase: float


@dataclass(frozen=True)
class ReasoningTrace:
    """Short structured explanation linking evidence to an action.

    This is an audit aid, not hidden chain-of-thought. The controller does not
    dispatch from these strings; it dispatches from validated families,
    candidates, modes, and config deltas.
    """

    observed_signal: str = ""
    inference: str = ""
    action_implication: str = ""


@dataclass(frozen=True)
class HypothesisCard:
    hypothesis_id: str
    target_id: str
    tick_created: int

    claim: str
    mode_affinity: dict[str, float]
    evidence_refs: list[str]
    predicted_metric_changes: list[PredictedChange]
    preserve_constraints: list[PreserveConstraint]
    recommended_action_families: list[str]

    ttl_ticks: int = 10  # was 3; calibrated to 10 by lifecycle simulation (plan §21 Q30, 2026-05-26)
    status: LifecycleStatus = "active"
    support_points: float = 0.0
    contradiction_points: float = 0.0
    descendants_evaluated: int = 0
    last_evaluated_tick: int | None = None
    # Gap A: LLM-proposed parameter overrides per family (Planner output).
    # CandidateBuilder consults capability_registry.validate_config_delta before
    # accepting these — out-of-range values are dropped, the action falls back
    # to defaults.
    config_delta_suggestions: dict[str, dict[str, Any]] = field(default_factory=dict)
    reasoning_trace: ReasoningTrace = field(default_factory=ReasoningTrace)


@dataclass(frozen=True)
class FeasibilityCheck:
    backend_healthy: bool
    runtime_bucket_id: str | None
    compiler_ok: bool
    verifier_ok: bool
    route_cap_ok: bool
    cost_ok: bool
    reasons: list[str] = field(default_factory=list)

    def all_ok(self) -> bool:
        return (
            self.backend_healthy
            and self.runtime_bucket_id is not None
            and self.compiler_ok
            and self.verifier_ok
            and self.route_cap_ok
            and self.cost_ok
        )


@dataclass(frozen=True)
class ActionCandidate:
    candidate_id: str
    hypothesis_ids: list[str]
    # Concrete execution parent: only set when the worker must consume an
    # existing ResultRecord structure (refilter / redesign families).
    parent_result_id: str | None
    method_family: str
    operator_id: str
    lane_id: str
    config_delta: dict[str, Any]
    downstream_route_plan: list[str]
    estimated_cost_class: ResourceClass
    expected_signal: str
    evidence_refs: list[str]
    feasibility: FeasibilityCheck
    supervisor_mode: ModeName | None = None  # set after supervisor decision
    # Concrete comparison baseline for hypothesis lifecycle updates. This is
    # intentionally separate from parent_result_id: de-novo generators may cite
    # a prior result/recipe as the baseline they aim to improve while consuming
    # no parent PDB at execution time.
    baseline_result_id: str | None = None
    # Role of refilter-like candidates. None for non-refilter families.
    # For structure_refilter, canonical_score_conversion is deterministic
    # system scoring after a diagnostic generator; parent_model_refold is an
    # intentional E/R/E action on an existing parent.
    refilter_role: RefilterRole | None = None


# `MissingCandidateRequest` was removed 2026-05-26 PM along with the rest
# of the MCR machinery (plan §10.6 removal made permanent in code). The
# dataclass, schema field, archive table, and Planner emission are all
# gone. See plan §10.6 for the audit-history removal rationale.


# ---------------------------------------------------------------------------
# LLM outputs (planner / supervisor)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerOutput:
    valid: bool
    abstain: bool
    confidence: float
    fail_reason: str | None
    cards: list[HypothesisCard]
    rationale: str
    raw_text: str
    usage: dict[str, int]


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    mode: ModeName
    rank_in_mode: int
    resource_class: ResourceClass
    what: str
    why: str
    evidence_refs: list[str]
    expected_signal: str
    stop_or_downgrade_if: str
    # Immediate cross-mode launch order from the Supervisor. rank_in_mode keeps
    # the interpretable E/R/E ordering; global_rank answers which candidate the
    # next available worker should run. None preserves legacy archive replay and
    # deterministic/unified fallback construction.
    global_rank: int | None = None
    reasoning_trace: ReasoningTrace = field(default_factory=ReasoningTrace)


@dataclass(frozen=True)
class SupervisorOutput:
    valid: bool
    abstain: bool
    confidence: float
    fail_reason: str | None
    mode_mixture: dict[str, float]
    candidate_decisions: list[CandidateDecision]
    rationale: str
    raw_text: str
    usage: dict[str, int]


@dataclass(frozen=True)
class SupervisorDecision:
    tick_id: str
    mode_mixture: dict[str, float]
    candidate_decisions: list[CandidateDecision]
    clamps_applied: list[str]
    fallback_used: bool
    rationale: str
    # Persisted so evidence-skip can replay the ORIGINAL confidence when it
    # reuses this mixture (a reused low-confidence mixture must keep its Cat-B
    # low-confidence clamp). Default 1.0 keeps legacy records reconstructable.
    confidence: float = 1.0
    # Compact Selector audit trail. This is intentionally informational: it
    # lets the next EvidenceSummary/forensics explain proposed -> selected ->
    # started mismatches without changing the original decision semantics.
    selector_debug: dict[str, Any] = field(default_factory=dict)
    # Compact copy of the read-only selector context shown to the Supervisor.
    # This records the execution/capacity view the LLM actually saw, which makes
    # later selected-vs-started audits reproducible without storing the full
    # prompt payload.
    selector_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMCallRecord:
    call_id: str
    tick_id: str
    role: Literal["planner", "supervisor", "critic"]  # critic added 2026-05-26 (§22.3)
    model: str
    model_digest: str | None
    prompt_hash: str
    schema_version: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    # F6 fix (2026-05-31): widened to the token set _llm_call_record actually
    # stores (= fail_reason.split(":")[0] on the valid=False paths) + the critic's
    # direct "timeout". The valid=False producers are call_error / parse_fail /
    # schema_fail (planner+supervisor) and no_hypotheses_or_candidates (supervisor
    # short-circuit, live_tick.py:998). abstain / low_confidence / empty_cards do
    # NOT appear here — they are returned with valid=True (planner.py:1086) so
    # parse_status is "ok". Was ["ok","parse_fail","schema_fail","timeout"], which
    # omitted call_error + no_hypotheses_or_candidates → archive consumers reading
    # the narrow set drifted.
    parse_status: Literal[
        "ok", "parse_fail", "schema_fail", "timeout",
        "call_error", "no_hypotheses_or_candidates",
    ]
    confidence: float | None
    abstain: bool
    fallback_triggered: bool
    # Full producer-side failure/repair detail. parse_status intentionally stays
    # coarse for aggregation; this field keeps the actionable validator suffix
    # such as "schema_fail:card[0]:unknown_evidence_refs:[x]".
    fail_reason: str | None = None
    # §22.3 critic flags (informational; never overrides Selector).
    # Populated only on role == "critic" calls; empty for planner/supervisor.
    # Sample entries: "(a) target-prior contradicted by evidence: ...",
    # "(c) ignored recent failure: ...", or empty if "no_flags".
    critic_flags: list[str] = field(default_factory=list)
    # Compact prompt audit snapshot. We intentionally do not archive full
    # prompts because they are tens of thousands of tokens per tick; this bounded
    # dict stores the decision-critical headers needed for replay/debugging.
    prompt_audit: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Launch + route + panel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchDecision:
    launch_id: str
    tick_id: str
    candidate_id: str
    status: Literal["launched", "rejected"]
    resource_class_concrete: dict[str, Any]
    why: str
    fallback: bool = False
    is_control: bool = False


@dataclass(frozen=True)
class DispatchRecord:
    """Actual worker-dispatch audit record.

    LaunchDecision is an LLM/Selector scheduling intent. In the event-driven
    controller a selected candidate may sit in the prefetch queue before a GPU
    is free, so actual worker start/failure is tracked separately here.
    """

    dispatch_id: str
    tick_id: str
    candidate_id: str
    status: Literal["started", "dispatch_failed", "parse_failed"]
    launch_id: str | None = None
    worker_slot: str | None = None
    gpu_id: str | None = None
    output_dir: str | None = None
    parent_result_id: str | None = None
    parent_pdb_path: str | None = None
    # Denormalized launch provenance. These mirror ActionCandidate /
    # LaunchDecision metadata so started/failed audit rows remain interpretable
    # without a fragile candidate_id join.
    method_family: str | None = None
    operator_id: str | None = None
    supervisor_mode: ModeName | None = None
    refilter_role: RefilterRole | None = None
    score_credit_basis: str | None = None
    # Process identity for restart fencing. Legacy rows reconstruct with None.
    # The controller uses these only to avoid launching a second copy of a worker
    # that may have survived a controller-process restart.
    worker_pid: int | None = None
    worker_pgid: int | None = None
    worker_pid_start_ticks: int | None = None
    worker_host: str | None = None
    slurm_job_id: str | None = None
    attempt: int = 1
    why: str = ""


@dataclass(frozen=True)
class RouteStageRecord:
    stage: str
    input_count: int
    converted_count: int
    failed_count: int
    score_files: int = 0
    near_miss: int = 0
    strict: int = 0
    panel_ready: int = 0
    calibrated_delta: float | None = None


@dataclass(frozen=True)
class RouteRecord:
    route_id: str
    origin_family: str
    origin_artifact_id: str
    parent_quality_stratum: ParentStratum
    runtime_bucket_id: str
    stages: list[RouteStageRecord]
    matched_control_route_ids: list[str]
    credit_status: CreditStatus
    stage_credit: dict[str, float]


@dataclass(frozen=True)
class PanelSelection:
    panel_id: str
    K: int
    selected_ids: list[str]
    pareto_audit: list[dict[str, Any]]
    diversity_bins: dict[str, int]
    hard_gate_failures: list[str]
    calibration_ref: str
    panel_value: float


# ---------------------------------------------------------------------------
# Optional later (stubs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairwiseComparison:
    comparison_id: str
    candidate_a: str
    candidate_b: str
    winner: str
    margin: float | None
    judge_model: str


@dataclass(frozen=True)
class MetaReview:
    review_id: str
    tick_range: tuple[int, int]
    summary_md: str
    invalid_requests: int
    backend_failures: int


@dataclass(frozen=True)
class CampaignPrior:
    version: str
    data_sources: list[str]
    holdout_targets: list[str]
    posterior_summary: dict[str, Any]
    rollback_path: str | None


def to_jsonable(rec: Any) -> Any:
    """Convert any dataclass record (including nested) to a JSON-serializable dict."""
    return _to_dict(rec)
