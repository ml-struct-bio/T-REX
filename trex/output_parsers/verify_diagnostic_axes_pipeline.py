"""End-to-end verification: BindCraft parser → EvidenceSummary
diagnostic_axis_stats. Runs locally, no LLM.

Verifies:
  1. parse_bindcraft_output on real fixture → ResultRecords with
     diagnostic metric values.
  2. reduce_evidence consumes those records → EvidenceSummary with
     populated diagnostic_axis_stats.
  3. Each diagnostic axis pass_count + near_pass_count + fail_count
     equals n_total when coverage is 100%.
  4. JSON round-trip: schema → JSON → schema preserves diagnostic_axis_stats.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..evidence_reducer import reduce_evidence
from ..schemas import EvidenceSummary, to_jsonable
from .bindcraft import parse_bindcraft_output
from .types import ParserContext


FIXTURE = Path(
    "/path/to/trex/archive"
)


def main() -> int:
    ctx = ParserContext(
        target_id="30_SC2RBD",
        runtime_bucket_id="bc_v1",
        candidate_id="test_cand_001",
        parent_ids=["test_cand_001"],
    )
    print(f"Step 1: parse fixture {FIXTURE}")
    records = parse_bindcraft_output(FIXTURE, ctx)
    print(f"  → {len(records)} records")

    if not records:
        print("[FAIL] no records — cannot proceed")
        return 1

    print("Step 2: reduce_evidence over the records")
    evidence = reduce_evidence(
        tick_id="t_test_001",
        target_id="30_SC2RBD",
        target_class="viral_rbd",
        elapsed_wall_h=12.0,
        remaining_wall_h=36.0,
        pending_children=0,
        worker_gpu_h_total=12.0,
        all_results=records,
        window_results=records,
        run_su_count=0,
        run_su_count_delta=0,
        duplicate_fraction=0.0,
        near_miss_count=0,
        top_bin_share=0.5,
        panel_ready_count=0,
        panel_ready_bins_covered=0,
        llm_model="vllm/Qwen/Qwen3.6-27B-FP8",
    )

    print("Step 3: inspect EvidenceSummary.diagnostic_axis_stats")
    if not evidence.diagnostic_axis_stats:
        print("[FAIL] diagnostic_axis_stats is empty")
        return 1
    failures = []
    for axis, stat in evidence.diagnostic_axis_stats.items():
        total = stat.pass_count + stat.near_pass_count + stat.fail_count
        marker = "✓" if total == stat.n else "✗"
        print(f"  {marker}  {axis:25s}  p={stat.pass_count:2d} n={stat.near_pass_count:2d} f={stat.fail_count:2d}  "
              f"total={total} (n={stat.n})  median={stat.median_raw}")
        if total != stat.n:
            failures.append(f"{axis} counts don't sum to n")
    if failures:
        print(f"[FAIL] {failures}")
        return 1

    print("Step 4: JSON round-trip")
    js = to_jsonable(evidence)
    js_str = json.dumps(js)
    # parse back
    js_back = json.loads(js_str)
    n_diag = len(js_back.get("diagnostic_axis_stats", {}))
    print(f"  → JSON length {len(js_str)} bytes, n_diag_axes_serialized={n_diag}")
    if n_diag != len(evidence.diagnostic_axis_stats):
        print(f"[FAIL] diagnostic_axis_stats count differs after round-trip: "
              f"{len(evidence.diagnostic_axis_stats)} → {n_diag}")
        return 1

    # Verify success_axis_stats unchanged
    if len(evidence.axis_stats) != 3:
        print(f"[FAIL] success axis_stats not 3: {list(evidence.axis_stats.keys())}")
        return 1

    print(f"\n[PASS] BindCraft → EvidenceSummary diagnostic pipeline OK.")
    print(f"  Success axes: {list(evidence.axis_stats.keys())}")
    print(f"  Diagnostic axes: {list(evidence.diagnostic_axis_stats.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
