"""Exercise the documented CPU workflow without molecular or clustering tools."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / 'examples/analysis_demo'


@pytest.fixture(scope='module')
def completed_demo(tmp_path_factory):
    destination = tmp_path_factory.mktemp('analysis-example') / 'output'
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith('TREX_') and key != 'PYTHONPATH'}
    subprocess.run(
        [sys.executable, str(DEMO / 'run_demo.py'), '--out', str(destination)],
        cwd=destination.parent, env=environment, check=True, capture_output=True, text=True,
    )
    return destination, environment


def test_example_excludes_unqualified_design_and_copies_exact_fixture(completed_demo):
    destination, _ = completed_demo
    expected = json.loads((DEMO / 'expected_outputs/checks.json').read_text())
    assert json.loads((destination / 'checks.json').read_text()) == expected
    copied = list((destination / 'export/pdbs').glob('*.pdb'))
    assert len(copied) == 1
    assert copied[0].read_bytes() == (DEMO / 'toy.pdb').read_bytes()
    manifest = json.loads((destination / 'export/manifest.json').read_text())
    assert [row['result_id'] for row in manifest['designs']] == ['demo_qualified']


def test_example_preserves_missing_clustering_and_linked_start(completed_demo):
    destination, _ = completed_demo
    validation = json.loads((destination / 'analysis/validate.json').read_text())
    assert validation['ok'] and not validation['errors']
    assert any('Foldseek' in warning for warning in validation['warnings'])
    summary = json.loads((destination / 'analysis/summary.json').read_text())
    assert summary['endpoint']['foldseek_su_status'] == 'not_run'
    assert summary['endpoint']['su_per_worker_wall_gpu_h'] is None
    trace = json.loads((destination / 'analysis/trace.json').read_text())['entries']
    assert trace[0]['hypotheses'][0]['hypothesis_id'] == 'demo_hypothesis_001'
    assert trace[0]['launches'][0]['candidate_id'] == 'demo_candidate_001'
    assert trace[0]['dispatches'][0]['outcome'] == 'started'


def test_example_refuses_existing_output_without_modifying_it(completed_demo):
    destination, environment = completed_demo
    before = {str(path.relative_to(destination)): path.read_bytes()
              for path in destination.rglob('*') if path.is_file()}
    completed = subprocess.run(
        [sys.executable, str(DEMO / 'run_demo.py'), '--out', str(destination)],
        cwd=destination.parent, env=environment, capture_output=True, text=True,
    )
    assert completed.returncode == 2
    assert 'output already exists' in completed.stderr
    after = {str(path.relative_to(destination)): path.read_bytes()
             for path in destination.rglob('*') if path.is_file()}
    assert before == after
