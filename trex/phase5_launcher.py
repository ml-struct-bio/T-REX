"""Phase 5 outer launcher daemon (§22.1).

Async event-driven: maintains a pool of N worker slots; each completion
triggers exactly one new plan_one_slot call + replacement launch on the
freed GPU. NOT a sync 3-batch loop.

Run mode:
  - test : substitutes SLURM submission with a stub that echoes the
           command to a JSONL log + immediately marks the job as
           "complete" once a pre-staged output dir appears. Use for
           §22.1.1 T1 acceptance smoke.

Production live campaigns are handled by controller.py. This
module intentionally keeps live submission unimplemented.

Output parsers are dispatched by ActionCandidate.method_family
(see trex/output_parsers/).
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .archive import Archive
from .schemas import ActionCandidate, DispatchRecord, LaunchDecision, ResultRecord
from .output_parsers import ParseError, ParserContext
from .output_parsers.bindcraft import parse_bindcraft_output
from .output_parsers.complexa import parse_complexa_output


# Per-family parser registry. This launcher is the test-mode smoke harness
# (submit_live_mode is deferred to controller.py for production).
# B-012 cleanup (2026-05-27): unsupported legacy Complexa aliases
# entries removed — capability_registry no longer exposes them.
PARSER_BY_FAMILY = {
    "bindcraft":            parse_bindcraft_output,
    "complexa_beam":        parse_complexa_output,
    "complexa_best_of_n":   parse_complexa_output,
    "complexa_fk_steering": parse_complexa_output,
    "complexa_mcts":        parse_complexa_output,
    # proteinmpnn, refilter, boltzgen — added when test fixtures
    # appear (OQ-004). Production parsing happens in controller.
}


@dataclass
class WorkerSlot:
    slot_id: str
    job_id: str | None = None       # SLURM job id (or TEST_xxx in test mode)
    candidate_id: str | None = None
    launched_at: float | None = None  # epoch seconds
    output_dir: Path | None = None


@dataclass
class DaemonConfig:
    archive_root: Path
    target_id: str
    n_slots: int = 3
    poll_interval_s: float = 30.0
    max_wall_h: float = 48.0
    mode: str = "test"  # "test" | "live"
    test_output_root: Path | None = None  # for test mode pre-staged dirs


# ---------------------------------------------------------------------------
# Submission backends
# ---------------------------------------------------------------------------

def submit_test_mode(candidate: ActionCandidate, output_dir: Path,
                       log_path: Path) -> str:
    """Stub: log the 'submission' to JSONL, return a fake job_id."""
    job_id = f"TEST_{uuid.uuid4().hex[:8]}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "candidate_id": candidate.candidate_id,
            "method_family": candidate.method_family,
            "config_delta": candidate.config_delta,
            "stub_job_id": job_id,
            "expected_output_dir": str(output_dir),
        }) + "\n")
    return job_id


def submit_live_mode(candidate: ActionCandidate, output_dir: Path) -> str:
    """Real sbatch path — DEFERRED. Will dispatch per-family SLURM wrappers."""
    raise NotImplementedError(
        "Live SLURM submission per-family wrappers not yet implemented. "
        "Use --mode=test for §22.1.1 T1 acceptance smoke."
    )


def is_complete(slot: WorkerSlot, mode: str) -> bool:
    """In test mode, completion = output_dir exists with a sentinel file.
    In live mode, completion = sacct shows COMPLETED/FAILED/CANCELLED."""
    if slot.output_dir is None:
        return False
    if mode == "test":
        # Sentinel: pre-staged BindCraft fixture exists at output_dir
        sentinel = slot.output_dir / "designs" / "final_design_stats.csv"
        return sentinel.exists()
    elif mode == "live":
        raise NotImplementedError("Live sacct polling not yet wired.")
    return False


# ---------------------------------------------------------------------------
# Daemon main loop
# ---------------------------------------------------------------------------

def daemon_main_loop(cfg: DaemonConfig) -> dict[str, Any]:
    """Run the async daemon until budget is exhausted or stop signal."""
    archive = Archive(cfg.archive_root)
    submissions_log = cfg.archive_root / "daemon_submissions.jsonl"
    parse_errors_log = cfg.archive_root / "parse_errors.jsonl"

    pool: list[WorkerSlot] = [WorkerSlot(slot_id=f"slot_{i}")
                                for i in range(cfg.n_slots)]
    start_t = time.time()
    n_completed = 0
    n_parse_errors = 0
    n_submitted = 0

    while (time.time() - start_t) / 3600.0 < cfg.max_wall_h:
        # 1. Detect completions
        for slot in pool:
            if slot.job_id is None or not is_complete(slot, cfg.mode):
                continue
            # Parse output and append ResultRecords
            family = _get_candidate_family(archive, slot.candidate_id)
            parser = PARSER_BY_FAMILY.get(family)
            if parser is None:
                _log_parse_error(parse_errors_log, slot, f"no parser for family={family}")
                archive.append(DispatchRecord(
                    dispatch_id=f"daemon_parse_failed_{uuid.uuid4().hex[:12]}",
                    tick_id="daemon",
                    candidate_id=slot.candidate_id or "",
                    status="parse_failed",
                    worker_slot=slot.slot_id,
                    output_dir=str(slot.output_dir) if slot.output_dir else None,
                    why=f"no parser for family={family}",
                ))
                n_parse_errors += 1
                _free_slot(slot)
                continue
            cand = _lookup_candidate(archive, slot.candidate_id)
            rt_bucket = "unknown"
            if cand and cand.feasibility and cand.feasibility.runtime_bucket_id:
                rt_bucket = cand.feasibility.runtime_bucket_id
            ctx = ParserContext(
                target_id=cfg.target_id,
                runtime_bucket_id=rt_bucket,
                candidate_id=slot.candidate_id,
                parent_ids=[slot.candidate_id],
            )
            try:
                records = parser(slot.output_dir, ctx)
                for r in records:
                    archive.append(r)
                n_completed += 1
            except ParseError as e:
                _log_parse_error(parse_errors_log, slot, str(e))
                archive.append(DispatchRecord(
                    dispatch_id=f"daemon_parse_failed_{uuid.uuid4().hex[:12]}",
                    tick_id="daemon",
                    candidate_id=slot.candidate_id or "",
                    status="parse_failed",
                    worker_slot=slot.slot_id,
                    output_dir=str(slot.output_dir) if slot.output_dir else None,
                    why=str(e)[:300],
                ))
                n_parse_errors += 1
            _free_slot(slot)

        # 2. Read new LaunchDecisions and submit on free slots
        free_slots = [s for s in pool if s.job_id is None]
        if not free_slots:
            time.sleep(cfg.poll_interval_s)
            continue

        new_decisions = _iter_unscheduled_launches(archive)
        for slot, decision in zip(free_slots, new_decisions):
            cand = _lookup_candidate(archive, decision.candidate_id)
            if cand is None:
                continue
            # Build expected output dir
            output_dir = _expected_output_dir(cfg, slot, cand)
            if cfg.mode == "test":
                job_id = submit_test_mode(cand, output_dir, submissions_log)
            else:
                job_id = submit_live_mode(cand, output_dir)
            slot.job_id = job_id
            slot.candidate_id = decision.candidate_id
            slot.launched_at = time.time()
            slot.output_dir = output_dir
            archive.append(DispatchRecord(
                dispatch_id=f"daemon_dispatch_{uuid.uuid4().hex[:12]}",
                launch_id=decision.launch_id,
                tick_id=decision.tick_id,
                candidate_id=decision.candidate_id,
                status="started",
                worker_slot=slot.slot_id,
                gpu_id=None,
                output_dir=str(output_dir),
                attempt=1,
                why=f"phase5_launcher submitted job_id={job_id}",
            ))
            n_submitted += 1

        time.sleep(cfg.poll_interval_s)

        # Soft exit when all slots idle and no new LaunchDecisions
        if (all(s.job_id is None for s in pool)
            and not list(_iter_unscheduled_launches(archive))):
            break

    return {
        "n_submitted": n_submitted,
        "n_completed": n_completed,
        "n_parse_errors": n_parse_errors,
        "wall_clock_h": round((time.time() - start_t) / 3600.0, 3),
        "submissions_log": str(submissions_log),
        "parse_errors_log": str(parse_errors_log),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_slot(slot: WorkerSlot) -> None:
    slot.job_id = None
    slot.candidate_id = None
    slot.launched_at = None
    slot.output_dir = None


def _log_parse_error(log_path: Path, slot: WorkerSlot, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "slot_id": slot.slot_id,
            "job_id": slot.job_id,
            "candidate_id": slot.candidate_id,
            "msg": msg,
        }) + "\n")


def _iter_unscheduled_launches(archive: Archive):
    """Yield LaunchDecisions whose candidate_id is not yet in
    daemon_submissions.jsonl."""
    submitted_ids: set[str] = set()
    log_path = archive.root / "daemon_submissions.jsonl"
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                try:
                    submitted_ids.add(json.loads(line)["candidate_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    for L in archive.iter_records(LaunchDecision):
        if getattr(L, "status", None) == "launched" and L.candidate_id not in submitted_ids:
            yield L


def _lookup_candidate(archive: Archive, candidate_id: str | None) -> ActionCandidate | None:
    if candidate_id is None:
        return None
    for c in archive.iter_records(ActionCandidate):
        if c.candidate_id == candidate_id:
            return c
    return None


def _get_candidate_family(archive: Archive, candidate_id: str | None) -> str | None:
    c = _lookup_candidate(archive, candidate_id)
    return c.method_family if c else None


def _expected_output_dir(cfg: DaemonConfig, slot: WorkerSlot,
                          cand: ActionCandidate) -> Path:
    """In test mode: use cfg.test_output_root (operator pre-stages a real
    fixture there). In live mode: derive from SLURM JOB_ID conventions."""
    if cfg.mode == "test" and cfg.test_output_root is not None:
        return cfg.test_output_root
    # Live mode placeholder
    return cfg.archive_root / "worker_outputs" / cand.method_family / slot.slot_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="T-ReX phase5 outer launcher daemon (§22.1)")
    p.add_argument("--archive-root", type=Path, required=True)
    p.add_argument("--target-id", type=str, required=True)
    p.add_argument("--n-slots", type=int, default=3)
    p.add_argument("--poll-interval-s", type=float, default=30.0)
    p.add_argument("--max-wall-h", type=float, default=48.0)
    p.add_argument("--mode", choices=["test", "live"], default="test")
    p.add_argument("--test-output-root", type=Path, default=None,
                    help="(test mode) pre-staged worker output dir to use for "
                          "every test-mode launch — typically a real BindCraft "
                          "fixture for T1 acceptance.")
    args = p.parse_args()

    if args.mode == "test" and args.test_output_root is None:
        p.error("--test-output-root required in test mode")

    cfg = DaemonConfig(
        archive_root=args.archive_root,
        target_id=args.target_id,
        n_slots=args.n_slots,
        poll_interval_s=args.poll_interval_s,
        max_wall_h=args.max_wall_h,
        mode=args.mode,
        test_output_root=args.test_output_root,
    )
    print(f"[phase5_launcher] starting (mode={args.mode}, slots={args.n_slots})")
    summary = daemon_main_loop(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
