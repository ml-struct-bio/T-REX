"""The SU-minting strict gate must be a FIXED function of the design: only the
CANONICAL single-config AF2 score (a canonical score-conversion) writes the
strict keys and mints SU. A config-varying re-fold — an intentional
parent_model_refold OR a multi-model ensemble — is a different MEASUREMENT of the
same design, so it is recorded as ADVISORY (refold_*) and never mints SU. This
stops re-folding a near-miss under a friendlier AF2 config until it passes."""
from __future__ import annotations

import json
from pathlib import Path

from trex.output_parsers.af2_refilter import parse_af2_refilter_output
from trex.output_parsers.types import ParserContext
from trex.refilter_roles import (
    CANONICAL_SCORE_CONVERSION,
    PARENT_MODEL_REFOLD,
)
from trex.success_criteria import is_strict_success


def _write_report(tmp_path: Path, *, model_names) -> Path:
    out = tmp_path / "af2_out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "af2_refilter_result.json").write_text(json.dumps({
        "schema_version": "v5_af2_refilter_result.v1",
        "input_pdb": "/tmp/in.pdb",
        "target_chain": "A",
        "binder_chain": "B",
        "metrics": {
            "plddt": 0.95,             # -> pLDDT 95.0 (passes >= 90)
            "i_pae": 0.10,             # passes <= 7/31
            "binder_scrmsd_ca": 1.0,   # passes <= 1.5
            "iptm": 0.8,
            "model_names": model_names,
            "predicted_pdb": str(out / "pred.pdb"),
        },
    }))
    return out


def _ctx(role: str = "") -> ParserContext:
    return ParserContext(
        target_id="t1", runtime_bucket_id="rb1", candidate_id="cand1",
        parent_ids=["cand1"], method_family="structure_refilter",
        parent_result_id="g1", refilter_role=role,
    )


def test_canonical_single_model_conversion_mints_strict_su(tmp_path: Path):
    rec = parse_af2_refilter_output(
        _write_report(tmp_path, model_names=["model_1_multimer_v3"]),
        _ctx(CANONICAL_SCORE_CONVERSION))[0]
    assert is_strict_success(rec.metrics)
    assert rec.metrics["pLDDT"] == 95.0
    assert rec.bins["af2_strict_basis"] == "canonical"


def test_single_model_parent_model_refold_is_advisory(tmp_path: Path):
    # The KEY option-(b) behavior: an intentional re-fold (e.g. different recycles)
    # is advisory even with a single model — the SAME passing metrics do NOT mint SU.
    rec = parse_af2_refilter_output(
        _write_report(tmp_path, model_names=["model_1_multimer_v3"]),
        _ctx(PARENT_MODEL_REFOLD))[0]
    assert not is_strict_success(rec.metrics)
    assert "pLDDT" not in rec.metrics and "iPAE" not in rec.metrics
    assert rec.metrics["refold_pLDDT"] == 95.0
    assert rec.metrics["refold_iPAE"] == 0.10
    assert rec.bins["af2_strict_basis"] == "advisory_refold"


def test_ensemble_is_advisory_even_without_refold_role(tmp_path: Path):
    # Defensive: a multi-model ensemble is advisory regardless of the role tag.
    rec = parse_af2_refilter_output(_write_report(tmp_path, model_names=[
        "model_1_multimer_v3", "model_2_multimer_v3", "model_3_multimer_v3",
        "model_4_multimer_v3", "model_5_multimer_v3",
    ]), _ctx(CANONICAL_SCORE_CONVERSION))[0]
    assert not is_strict_success(rec.metrics)
    assert rec.metrics["refold_pLDDT"] == 95.0
    assert rec.bins["af2_strict_basis"] == "advisory_refold"
    assert rec.bins["af2_model_count"] == 5


def test_string_model_names_ensemble_is_advisory(tmp_path: Path):
    rec = parse_af2_refilter_output(_write_report(
        tmp_path, model_names="model_1_multimer_v3,model_2_multimer_v3"), _ctx())[0]
    assert not is_strict_success(rec.metrics)
    assert rec.bins["af2_model_count"] == 2


def test_missing_role_and_single_model_defaults_to_canonical_strict(tmp_path: Path):
    # Backward-compat: an older report (no role, single/absent model_names) mints SU.
    out = tmp_path / "af2_out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "af2_refilter_result.json").write_text(json.dumps({
        "metrics": {"plddt": 0.95, "i_pae": 0.10, "binder_scrmsd_ca": 1.0},
    }))
    rec = parse_af2_refilter_output(out, _ctx())[0]
    assert is_strict_success(rec.metrics)
    assert rec.bins["af2_strict_basis"] == "canonical"
