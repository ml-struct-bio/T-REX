"""Keep the manuscript LLM benchmark tools importable without calling a model."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('module', [
    'planner_validation', 'supervisor_validation', 'supervisor_repeatability',
])
def test_llm_benchmark_help_does_not_require_a_server(module):
    completed = subprocess.run(
        [sys.executable, '-m', f'benchmarks.llm_validation.{module}', '--help'],
        cwd=ROOT, text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert '--base-url' in completed.stdout
    assert '--out' in completed.stdout


def test_llm_benchmark_synthetic_inputs_remain_constructible():
    from benchmarks.llm_validation import planner_validation, supervisor_repeatability

    case_names = [
        'case_productive', 'case_rescue_rich', 'case_stalled',
        'case_ambiguous', 'case_panel_ready_diversity_short',
        'case_stress_max_context',
    ]
    for name in case_names:
        label, evidence = getattr(planner_validation, name)('fixture-model')
        assert label and evidence.tick_id and evidence.target_id
    hypotheses, candidates = supervisor_repeatability.build_reliability_case()
    assert len(candidates) == 5
    assert len({candidate.candidate_id for candidate in candidates}) == 5
    assert {candidate.method_family for candidate in candidates}
    assert {hypothesis.hypothesis_id for hypothesis in hypotheses}
