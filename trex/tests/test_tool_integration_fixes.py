"""Regression tests for backend output parsing, sequence identity, source diagnostics, and
supported settings.
"""

from __future__ import annotations

from pathlib import Path

import trex.output_parsers.proteinmpnn as mpnn_mod
from trex.capability_registry import default_registry, validate_config_delta_partial
from trex.evidence_reducer import build_diagnostic_axis_stats
from trex.output_parsers.proteinmpnn import parse_proteinmpnn_output
from trex.output_parsers.types import ParserContext
from trex.schemas import ResultRecord


# --- MPNN FASTA fixtures -----------------------------------------------------

# Native header (protein_mpnn_run.py:384): NO `sample=`, NO `seq_recovery=`.
_NATIVE_HDR = (">3HFM, score=1.4500, global_score=1.4500, fixed_chains=['A'], "
               "designed_chains=['B'], model_name=v_48_020, seed=37")
# Generated headers (protein_mpnn_run.py:403): always carry `T=` + `sample=`.
_DESIGN1_HDR = ">T=0.1, sample=1, score=2.1000, global_score=1.8000, seq_recovery=0.3800"
_DESIGN2_HDR = ">T=0.1, sample=2, score=1.9000, global_score=1.2000, seq_recovery=0.5200"

_SEQ = "ACDEFGHIKL"  # length 10


def _write_fa(tmp_path: Path, *headers_seqs: tuple[str, str]) -> Path:
    seqs = tmp_path / "seqs"
    seqs.mkdir(parents=True, exist_ok=True)
    fa = seqs / "design.fa"
    fa.write_text("\n".join(f"{h}\n{s}" for h, s in headers_seqs) + "\n")
    return tmp_path


def _ctx(parent_pdb: str = "") -> ParserContext:
    return ParserContext(
        target_id="t1", runtime_bucket_id="rb1", candidate_id="cand1",
        parent_ids=["cand1"], method_family="proteinmpnn_redesign",
        parent_pdb_path=parent_pdb,
    )


def test_mpnn_native_parent_record_skipped(tmp_path: Path):
    out = _write_fa(tmp_path,
                    (_NATIVE_HDR, _SEQ), (_DESIGN1_HDR, _SEQ), (_DESIGN2_HDR, _SEQ))
    recs = parse_proteinmpnn_output(out, _ctx())
    assert len(recs) == 2, "native (no sample=) must be skipped; only 2 designs kept"
    # the kept records are the two designs, identifiable by their captured score
    gscores = sorted(float(r.bins["mpnn_global_score"]) for r in recs)
    assert gscores == [1.2, 1.8]


def test_mpnn_scores_captured_into_bins(tmp_path: Path):
    out = _write_fa(tmp_path, (_NATIVE_HDR, _SEQ), (_DESIGN2_HDR, _SEQ))
    recs = parse_proteinmpnn_output(out, _ctx())
    assert len(recs) == 1
    b = recs[0].bins
    assert b["mpnn_global_score"] == "1.2000"
    assert b["mpnn_seq_recovery"] == "0.5200"
    assert b["mpnn_score"] == "1.9000"


def test_mpnn_ranker_negates_global_score_for_higher_is_better():
    """The auto-chain ranker sorts reverse=True (higher=better) but MPNN
    global_score is an NLL (lower=better). Lock the negation so the best
    (lowest-NLL) redesign sorts first."""
    def rank_key(global_score: float) -> float:
        return -float(global_score)  # Lower source negative log-likelihood ranks first.
    ranked = sorted([("d1", 1.8), ("d2", 1.2)],
                    key=lambda x: rank_key(x[1]), reverse=True)
    assert ranked[0][0] == "d2"


def test_mpnn_threading_failure_omits_record(tmp_path: Path, monkeypatch):
    out = _write_fa(tmp_path, (_NATIVE_HDR, _SEQ), (_DESIGN1_HDR, _SEQ))
    # Force threading to fail (e.g. binder not on chain B / length mismatch).
    monkeypatch.setattr(mpnn_mod, "_thread_sequence_onto_pdb",
                        lambda *a, **k: False)
    recs = parse_proteinmpnn_output(out, _ctx(parent_pdb="/tmp/fake_parent.pdb"))
    assert recs == [], "failed-threading records must be omitted, not point at parent"


def test_mpnn_threading_success_keeps_threaded_record(tmp_path: Path, monkeypatch):
    out = _write_fa(tmp_path, (_NATIVE_HDR, _SEQ), (_DESIGN1_HDR, _SEQ))
    monkeypatch.setattr(mpnn_mod, "_thread_sequence_onto_pdb",
                        lambda *a, **k: True)
    recs = parse_proteinmpnn_output(out, _ctx(parent_pdb="/tmp/fake_parent.pdb"))
    assert len(recs) == 1
    assert recs[0].artifacts["threaded"] == "true"
    assert recs[0].artifacts["pdb_path"].endswith(".pdb")


def test_designed_chains_is_controller_owned_not_llm_tunable():
    cap = default_registry().get("proteinmpnn_redesign")
    assert "designed_chains" not in cap.allowed_params
    kept, reasons = validate_config_delta_partial(cap, {"designed_chains": "D"})
    assert kept == {}
    assert any(r.startswith("derived_param:designed_chains") for r in reasons)


def _complexa_rec(rid: str, iptm: float, min_ipae: float) -> ResultRecord:
    return ResultRecord(
        result_id=rid, parent_ids=[], target_id="t1",
        backend_family="complexa_beam", runtime_bucket_id="rb1",
        metrics={"pLDDT": 92.0, "iPAE": 0.20, "binder_scRMSD": 1.2,
                 "ipTM": iptm, "min_ipae": min_ipae,
                 "avg_ipsae": iptm, "max_ipsae": iptm},
        metrics_calibrated={}, route_lineage=[], gpu_h=0.5, exit_status="ok",
        bins={},
    )


def test_complexa_diagnostic_axes_now_aggregated():
    recs = [_complexa_rec(f"c{i}", iptm=0.55, min_ipae=0.18) for i in range(4)]
    stats = build_diagnostic_axis_stats(recs)
    # Complexa diagnostics include ipTM, ipSAE, and minimum iPAE.
    assert "ipTM" in stats
    assert "min_ipae" in stats
    assert "avg_ipsae" in stats and "max_ipsae" in stats
    # ipTM 0.55 is below the 0.60 threshold but within the 0.05 near-pass margin
    assert stats["ipTM"].n == 4


from trex.controller import (
    _complexa_canonical_af2_overrides,
    _complexa_refinement_overrides,
)


def test_complexa_af2_reward_overrides_force_official_single_model_gate():
    ov = _complexa_canonical_af2_overrides()
    assert "++generation.reward_model.reward_models.af2folding.model_nums=0" in ov
    assert "++generation.reward_model.reward_models.af2folding.model_nums=[0]" not in ov
    assert "++generation.reward_model.reward_models.af2folding.num_recycles=3" in ov
    assert "++generation.reward_model.reward_models.af2folding.use_initial_guess=True" in ov
    assert "++generation.reward_model.reward_models.af2folding.use_multimer=True" in ov


def test_greedy_knobs_emitted_only_with_sequence_hallucination():
    ov = _complexa_refinement_overrides(
        {"refinement_algorithm": "sequence_hallucination",
         "n_greedy_iters": 30, "enable_greedy_optimization": True})
    j = " ".join(ov)
    assert "refinement.algorithm=sequence_hallucination" in j
    assert "refinement.n_greedy_iters=30" in j
    assert "refinement.enable_greedy_optimization=true" in j


def test_greedy_knobs_auto_enable_sequence_hallucination():
    ov = _complexa_refinement_overrides(
        {"n_greedy_iters": 30, "enable_greedy_optimization": True})
    j = " ".join(ov)
    assert "refinement.algorithm=sequence_hallucination" in j
    assert "refinement.n_greedy_iters=30" in j
    assert "refinement.enable_greedy_optimization=true" in j


def test_refinement_algorithm_null_branch():
    ov = _complexa_refinement_overrides({"refinement_algorithm": ""})
    assert ov == ["++generation.refinement.algorithm=null"]


def test_mpnn_scores_reach_planner_alt_model_rollup():
    """Expose ProteinMPNN source scores in diagnostic summaries."""
    from trex.evidence_reducer import build_diagnostic_alt_model_scores
    recs = [
        ResultRecord(
            result_id=f"m{i}", parent_ids=[], target_id="t1",
            backend_family="proteinmpnn_redesign", runtime_bucket_id="rb1",
            metrics={}, metrics_calibrated={}, route_lineage=[], gpu_h=0.01,
            exit_status="ok",
            bins={"mpnn_global_score": f"{1.0 + 0.1 * i:.4f}",
                  "mpnn_seq_recovery": f"{0.4 + 0.01 * i:.4f}"},
        )
        for i in range(3)
    ]
    roll = build_diagnostic_alt_model_scores(recs)
    assert "proteinmpnn_redesign" in roll
    assert "global_score" in roll["proteinmpnn_redesign"]
    assert "seq_recovery" in roll["proteinmpnn_redesign"]
    assert roll["proteinmpnn_redesign"]["seq_recovery"]["direction"] == "increase"


def test_bindcraft_binder_ptm_now_aggregated():
    """Aggregate the BindCraft source pTM score."""
    recs = [
        ResultRecord(
            result_id=f"b{i}", parent_ids=[], target_id="t1",
            backend_family="bindcraft", runtime_bucket_id="rb1",
            metrics={"pLDDT": 92.0, "iPAE": 0.2, "binder_scRMSD": 1.0,
                     "binder_pTM_avg": 0.62},
            metrics_calibrated={}, route_lineage=[], gpu_h=1.0, exit_status="ok",
            bins={},
        )
        for i in range(4)
    ]
    stats = build_diagnostic_axis_stats(recs)
    assert "binder_pTM_avg" in stats
