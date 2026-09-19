#!/usr/bin/env python3
"""Create and inspect synthetic records using the installed T-REX interfaces.

No molecular inference, clustering or LLM call is performed. All measurements,
decisions and coordinates are artificial software-test data.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from trex import SCHEMA_VERSION
from trex.archive import Archive
from trex.schemas import (
    ActionCandidate, DispatchRecord, EvidenceSummary, FeasibilityCheck,
    HypothesisCard, LaunchDecision, LLMHealthSummary, ResultRecord,
    RouteHealthSummary, SupervisorDecision,
)

EXAMPLE_ROOT = Path(__file__).resolve().parent


def run_json_command(module: str, *arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, '-m', module, *arguments],
        text=True, capture_output=True, check=True,
    )
    return json.loads(completed.stdout)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')


def create_archive(output_root: Path) -> Path:
    fixture = json.loads((EXAMPLE_ROOT / 'fixture.json').read_text())
    archive_root = output_root / 'archive'
    archive = Archive(archive_root)
    structures = archive_root / 'worker_outputs' / 'synthetic_job'
    structures.mkdir(parents=True)
    structure_path = structures / 'toy.pdb'
    structure_path.write_bytes((EXAMPLE_ROOT / 'toy.pdb').read_bytes())
    candidate_id, hypothesis_id = 'demo_candidate_001', 'demo_hypothesis_001'
    tick_id, launch_id = 'tick_001', 'demo_launch_001'
    family, target_id = 'complexa_beam', fixture['target_id']
    archive.append(HypothesisCard(
        hypothesis_id=hypothesis_id, target_id=target_id, tick_created=1,
        claim='Synthetic example: test an underused route.',
        mode_affinity={'rescue': 0.0, 'explore': 1.0, 'exploit': 0.0},
        evidence_refs=[], predicted_metric_changes=[], preserve_constraints=[],
        recommended_action_families=[family],
    ))
    archive.append(ActionCandidate(
        candidate_id=candidate_id, hypothesis_ids=[hypothesis_id],
        parent_result_id=None, method_family=family, operator_id='demo_generation',
        lane_id='demo_lane', config_delta={}, downstream_route_plan=[],
        estimated_cost_class='low', expected_signal='Synthetic example only.',
        evidence_refs=[], supervisor_mode='explore',
        feasibility=FeasibilityCheck(True, 'demo_runtime', True, True, True, True),
    ))
    archive.append(SupervisorDecision(
        tick_id=tick_id,
        mode_mixture={'rescue': 0.0, 'explore': 1.0, 'exploit': 0.0},
        candidate_decisions=[], clamps_applied=[], fallback_used=False,
        rationale='Synthetic decision; no model was called.',
    ))
    archive.append(LaunchDecision(
        launch_id=launch_id, tick_id=tick_id, candidate_id=candidate_id,
        status='launched', resource_class_concrete={}, why='Synthetic queue admission.',
    ))
    archive.append(DispatchRecord(
        dispatch_id='demo_dispatch_001', tick_id=tick_id, candidate_id=candidate_id,
        launch_id=launch_id, status='started', method_family=family,
        supervisor_mode='explore', why='Synthetic start; no process was launched.',
    ))
    for design in fixture['designs']:
        archive.append(ResultRecord(
            result_id=design['result_id'], parent_ids=[candidate_id],
            target_id=target_id, backend_family=family,
            runtime_bucket_id='demo_runtime', metrics=design['metrics'],
            metrics_calibrated={}, route_lineage=[family], gpu_h=0.0,
            exit_status='ok', artifacts={'pdb_path': str(structure_path)}, tick_id=tick_id,
        ))
    archive.append(EvidenceSummary(
        tick_id=tick_id, target_id=target_id, target_class='synthetic',
        schema_version=SCHEMA_VERSION, elapsed_wall_h=0.0, remaining_wall_h=0.0,
        completed_children=1, pending_children=0, worker_gpu_h_total=0.0,
        worker_gpu_h_last_3_ticks=0.0, strict_count=1, global_new_strict=1,
        run_su_count=0, run_su_count_delta=0, su_per_gpu_h_recent=None,
        duplicate_fraction=None, top_bin_share=None, axis_stats={}, joint_patterns=[],
        near_miss_count=0, panel_ready_count=0, panel_ready_bins_covered=0,
        method_health={}, route_health=RouteHealthSummary(0, 0, 0, 0, None, None, None),
        llm_health=LLMHealthSummary('synthetic-no-model', [], 0.0, 0.0, 0, 0.0),
        state_label='low_evidence', examples=[], metric_availability={},
        foldseek_su_status='not_run', foldseek_su_coverage=0.0,
        sequence_dedup_status='not_run',
        diagnostic_driver_tldr='Synthetic archive for interface inspection.',
    ))
    write_json(archive_root / 'DEMO_ONLY.json', {
        'synthetic': True, 'molecular_inference_run': False, 'clustering_run': False,
        'note': 'SU=0 is an uncomputed fixture placeholder; no throughput is measured.',
    })
    return archive_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    args = parser.parse_args(argv)
    output_root = args.out.expanduser().resolve()
    if output_root.exists():
        parser.error('output already exists; choose a new directory to preserve its contents')
    output_root.mkdir(parents=True)
    archive_root = create_archive(output_root)
    analysis_root = output_root / 'analysis'
    analysis_root.mkdir()
    reports = {}
    for command in ('validate', 'summary', 'trace'):
        reports[command] = run_json_command(
            'trex.analysis', command, '--archive-root', str(archive_root), '--json'
        )
        write_json(analysis_root / f'{command}.json', reports[command])
    write_json(analysis_root / 'schema.json', run_json_command('trex.analysis', 'schema', '--json'))
    export = run_json_command(
        'trex.export_best_n', '--archive-root', str(archive_root),
        '--target-id', 'synthetic_demo', '--n', '10', '--no-dedup',
        '--out-dir', str(output_root / 'export'),
    )
    write_json(analysis_root / 'export.json', export)
    with (output_root / 'export/manifest.csv').open(newline='') as handle:
        exported_designs = list(csv.DictReader(handle))
    observed = {
        'archive_valid': reports['validate']['ok'],
        'result_count': reports['summary']['records']['results'],
        'qualified_fixture_count': reports['summary']['endpoint']['strict_count'],
        'foldseek_status': reports['summary']['endpoint']['foldseek_su_status'],
        'throughput': reports['summary']['endpoint']['su_per_worker_wall_gpu_h'],
        'trace_count': len(reports['trace']['entries']),
        'exported_result_ids': [row['result_id'] for row in exported_designs],
        'copied_pdb_count': len(list((output_root / 'export/pdbs').glob('*.pdb'))),
    }
    expected = json.loads((EXAMPLE_ROOT / 'expected_outputs/checks.json').read_text())
    if observed != expected:
        raise RuntimeError(f'example differs from its documented contract: {observed}')
    write_json(output_root / 'checks.json', observed)
    print(json.dumps({'ok': True, 'synthetic': True, 'output_root': str(output_root), **observed}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
