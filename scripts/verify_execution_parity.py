#!/usr/bin/env python3
"""Compare launch defaults and shutdown traces without running any backend.

Only the original inline shutdown block and the extracted public shutdown
functions are executed, against fake processes and a fake clock. This is a
bounded compatibility check, not full controller or GPU execution equivalence.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from typing import Any


SOURCE_PACKAGE = "auto" + "research_v7_3_3"

# Wrapper defaults are not evidence of the effective settings of a past run.
# Pin the documented public deployment change; reject other profile changes.
SOURCE_WRAPPER_PROFILE = {
    "time": "48:00:00",
    "gres": "gpu:h100:4",
    "cpus-per-task": "16",
    "mem": "192G",
    "launch_hours": "47.0",
}
PUBLIC_LAUNCH_PROFILE = {
    **SOURCE_WRAPPER_PROFILE,
    "time": "49:00:00",
    "launch_hours": "48.0",
}


def _main_body(path: Path) -> list[ast.stmt]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return next(
        node.body
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def _source_shutdown(path: Path) -> list[ast.stmt]:
    body = _main_body(path)
    index = next(
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "drain_grace_per_slot_s"
            for target in node.targets
        )
    )
    block = body[index : index + 2]
    if len(block) != 2 or not isinstance(block[1], ast.For):
        raise ValueError("original sequential shutdown loop not found")
    if ast.dump(block[1].iter) != ast.dump(ast.Name(id="pool", ctx=ast.Load())):
        raise ValueError("original shutdown loop does not iterate the worker pool")
    return block


def _public_shutdown(trex_root: Path) -> tuple[list[ast.stmt], float]:
    body = _main_body(trex_root / "trex/controller.py")
    calls = [
        node.value
        for node in body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "drain_worker_slots"
    ]
    if len(calls) != 1:
        raise ValueError("expected one top-level shutdown call after the loop")
    grace = next(
        keyword.value
        for keyword in calls[0].keywords
        if keyword.arg == "grace_seconds_per_slot"
    )
    grace_seconds = float(ast.literal_eval(grace))
    path = trex_root / "trex/execution/worker_supervision.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"_method_family", "_target_id", "drain_worker_slots"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise ValueError("incomplete public shutdown implementation")
    return functions, grace_seconds


# Entries are (busy, wait outcome, return code, parse fails).
SHUTDOWN_CASES = {
    "empty": (),
    "idle": ((False, "complete", 0, False),),
    "completed": ((True, "complete", 0, False),),
    "nonzero_exit": ((True, "complete", 7, False),),
    "timeout": ((True, "timeout", 0, False),),
    "parse_failure": ((True, "complete", 0, True),),
    "timeout_parse_failure": ((True, "timeout", 0, True),),
    "mixed_workers": (
        (False, "complete", 0, False),
        (True, "complete", 0, False),
        (True, "timeout", 0, True),
        (True, "complete", 7, True),
        (True, "timeout", 0, False),
    ),
}


def _shutdown_trace(
    statements: list[ast.stmt],
    case: tuple,
    *,
    public_grace: float | None = None,
) -> dict[str, Any]:
    events: list[list[Any]] = []
    now = 100.0
    slots = []
    archive = object()
    target = SimpleNamespace(target_id="fixture")
    chain_sequence = [0]

    def clock() -> float:
        return now

    def make_slot(index: int, spec: tuple) -> SimpleNamespace:
        busy, outcome, return_code, parse_fails = spec
        slot = SimpleNamespace(
            gpu_id=str(index),
            busy=busy,
            launched_at=0.0,
            cand=SimpleNamespace(method_family="test_backend"),
            parse_fails=parse_fails,
        )

        def wait(timeout: float) -> int:
            nonlocal now
            events.append(["wait", index, timeout])
            now += timeout if outcome == "timeout" else 20.0
            if outcome == "timeout":
                raise subprocess.TimeoutExpired("fake-worker", timeout)
            return return_code

        def free() -> None:
            events.append(["free", index])
            slot.busy = False
            slot.proc = None
            slot.cand = None

        slot.proc = SimpleNamespace(index=index, wait=wait)
        slot.free = free
        return slot

    for index, spec in enumerate(case):
        slots.append(make_slot(index, spec))

    def parse(
        slot, *, return_code, archive, target, auto_chain_sequence, elapsed_gpu_hours
    ):
        events.append(
            [
                "parse",
                int(slot.gpu_id),
                return_code,
                elapsed_gpu_hours,
                target.target_id,
                auto_chain_sequence is chain_sequence,
            ]
        )
        if slot.parse_fails:
            raise ValueError("fixture parse failure")
        return 1

    def parse_original(slot, *, rc, archive, target, chain_seq_ref, elapsed_gpu_h):
        return parse(
            slot,
            return_code=rc,
            archive=archive,
            target=target,
            auto_chain_sequence=chain_seq_ref,
            elapsed_gpu_hours=elapsed_gpu_h,
        )

    def record_failure(archive, slot, why, *, target_id):
        events.append(["parse_failure", int(slot.gpu_id), why, target_id])

    def terminate(process):
        events.append(["terminate", process.index])

    namespace = {
        "print": lambda *args, **kwargs: None,
        "subprocess": subprocess,
        "time": SimpleNamespace(time=clock),
        "pool": slots,
        "archive": archive,
        "target": target,
        "chain_seq_ref": chain_sequence,
        "_parse_and_archive_slot": parse_original,
        "_append_parse_failed_dispatch_record": record_failure,
        "_terminate_process_group": terminate,
        "WorkerDrainReport": SimpleNamespace,
    }
    # Deferred annotations avoid importing either controller or external tools.
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *statements,
        ],
        type_ignores=[],
    )
    exec(
        compile(ast.fix_missing_locations(tree), "<shutdown-parity>", "exec"), namespace
    )
    if public_grace is not None:
        namespace["drain_worker_slots"](
            slots,
            grace_seconds_per_slot=public_grace,
            archive=archive,
            target=target,
            auto_chain_sequence=chain_sequence,
            clock=clock,
            dependencies=SimpleNamespace(
                parse_worker_output=parse,
                record_parse_failure=record_failure,
                terminate_worker_process=terminate,
            ),
        )
    return {"events": events, "busy_after": [slot.busy for slot in slots]}


def _launch_profile(path: Path, *, public: bool) -> dict[str, str]:
    text = path.read_text()
    result = {}
    for option in ("time", "gres", "cpus-per-task", "mem"):
        values = re.findall(r"^#SBATCH --" + option + r"=(\S+)$", text, re.MULTILINE)
        if len(values) != 1:
            raise ValueError(f"missing or ambiguous Slurm option: {option}")
        result[option] = values[0]
    name = "TREX_MAX_WALL_H" if public else "MAX_WALL_H"
    values = re.findall(r"\$\{" + name + r":-([^}]+)\}", text)
    if len(set(values)) != 1:
        raise ValueError(f"missing or inconsistent launch-window default: {name}")
    result["launch_hours"] = str(float(values[0]))
    return result


def compare_execution(source_root: Path, trex_root: Path) -> dict[str, Any]:
    try:
        source_profile = _launch_profile(
            source_root / "slurm/v7_3_3_per_target_node.slurm", public=False
        )
        public_profile = _launch_profile(
            trex_root / "slurm/trex_per_target_node.slurm", public=True
        )
        source_statements = _source_shutdown(
            source_root / SOURCE_PACKAGE / "phase2_v7_controller.py"
        )
        public_statements, public_grace = _public_shutdown(trex_root)
        cases = []
        for name, case in SHUTDOWN_CASES.items():
            source_trace = _shutdown_trace(source_statements, case)
            public_trace = _shutdown_trace(
                public_statements, case, public_grace=public_grace
            )
            cases.append(
                {
                    "case": name,
                    "equal": source_trace == public_trace,
                    "source": source_trace,
                    "public": public_trace,
                }
            )
        profiles_equal = source_profile == public_profile
        profiles_verified = (
            source_profile == SOURCE_WRAPPER_PROFILE
            and public_profile == PUBLIC_LAUNCH_PROFILE
        )
        shutdown_equal = all(case["equal"] for case in cases)
        return {
            "schema_version": "trex.execution-parity.v2",
            "passed": profiles_verified and shutdown_equal,
            "execution_equal": profiles_equal and shutdown_equal,
            "shutdown_equal": shutdown_equal,
            "launch_profile": {
                "equal": profiles_equal,
                "matches_documented_profiles": profiles_verified,
                "source": source_profile,
                "public": public_profile,
                "expected_source": SOURCE_WRAPPER_PROFILE,
                "expected_public": PUBLIC_LAUNCH_PROFILE,
                "change": (
                    "Public defaults use a cumulative 48-hour controller limit "
                    "and a 49-hour scheduler reservation for startup/shutdown. "
                    "Historical effective run settings come from provenance."
                ),
            },
            "shutdown_cases": cases,
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trex-root", type=Path, required=True)
    args = parser.parse_args()
    report = compare_execution(args.source_root, args.trex_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
