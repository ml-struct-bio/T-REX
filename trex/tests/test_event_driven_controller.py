"""Test asynchronous worker lifecycle, dispatch, parsing, and chained evaluation with
mocked external processes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from trex.archive import Archive
from trex.controller import (
    FAMILY_TIMEOUT_S,
    LENGTHS_PER_TARGET,
    _WorkerSlot,
    _controller_sleep_seconds,
    _dispatch_candidate_to_gpu,
    _dispatch_pending_to_free_slots,
    _env_executable_path,
    _parse_and_archive_slot,
    _require_p2_runtime_available,
    _resolve_parent_artifact,
    _with_runtime_tool_paths,
)
from trex.campaign.runtime import (
    ControllerLoopConfig,
    RuntimePaths,
)
from trex.execution.progress_checkpoint import ProgressCheckpointResult
from trex.live_tick import LiveTickConfig
from trex.schemas import (
    ActionCandidate,
    FeasibilityCheck,
    ResultRecord,
    TargetConstraint,
    to_jsonable,
)
from trex.success_criteria import STRICT_SUCCESS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _feas() -> FeasibilityCheck:
    return FeasibilityCheck(
        backend_healthy=True,
        runtime_bucket_id="rb_v7",
        compiler_ok=True,
        verifier_ok=True,
        route_cap_ok=True,
        cost_ok=True,
    )


def _cand(family: str = "bindcraft", cid: str = "c1") -> ActionCandidate:
    return ActionCandidate(
        candidate_id=cid,
        hypothesis_ids=["h1"],
        parent_result_id=None,
        method_family=family,
        operator_id="op",
        lane_id=family,
        config_delta={},
        downstream_route_plan=[],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="x",
        evidence_refs=["e1"],
        feasibility=_feas(),
    )


def _target(tid: str = "t1") -> TargetConstraint:
    return TargetConstraint(target_id=tid, target_class="c")


_PASS_METRICS = {
    "pLDDT": STRICT_SUCCESS["pLDDT"][0] + 2.0,
    "iPAE": STRICT_SUCCESS["iPAE"][0] - 0.05,
    "binder_scRMSD": STRICT_SUCCESS["binder_scRMSD"][0] - 0.3,
}


def test_isolated_env_strips_cluster_cuda():
    """Backend library paths must exclude conflicting cluster CUDA libraries."""
    import os
    from pathlib import Path
    from trex.controller import _isolated_env_for_subprocess

    # Simulate the SLURM env that T-REX inherits after `module load foldseek`
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:/something/else"
    env = _isolated_env_for_subprocess(Path("/scratch/bc_env"))
    try:
        # Cluster CUDA gone, only bc_env/lib remains
        assert env["LD_LIBRARY_PATH"] == "/scratch/bc_env/lib"
        assert "/usr/local/cuda-12.8" not in env["LD_LIBRARY_PATH"]
        # PATH gets bc_env/bin prepended (not replaced)
        assert env["PATH"].startswith("/scratch/bc_env/bin:")
    finally:
        os.environ.pop("LD_LIBRARY_PATH", None)


def test_env_executable_path_preserves_venv_symlink(tmp_path: Path, monkeypatch):
    base = tmp_path / "base-python"
    base.write_text("#!/bin/sh\n")
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base)
    monkeypatch.setenv("TEST_VENV_PYTHON", str(venv_python))

    selected = _env_executable_path("TEST_VENV_PYTHON", "/unused")

    assert selected == venv_python
    assert selected != selected.resolve()


def test_controller_long_backoff_requires_all_workers_idle():
    idle = _WorkerSlot(slot_id=0, gpu_id="1")
    busy = _WorkerSlot(slot_id=1, gpu_id="2")
    busy.cand = _cand()
    busy.proc = MagicMock()

    assert (
        _controller_sleep_seconds(
            dispatched=0,
            planned=0,
            pending=[],
            pool=[idle],
            poll_interval_s=5,
        )
        == 120
    )
    assert (
        _controller_sleep_seconds(
            dispatched=0,
            planned=0,
            pending=[],
            pool=[idle, busy],
            poll_interval_s=5,
        )
        == 5
    )


def test_production_loop_has_no_iteration_cap_or_default_evidence_skip():
    assert ControllerLoopConfig().maximum_iterations is None
    assert LiveTickConfig().skip.enabled is False


def test_p2_runtime_preflight_uses_exact_configured_interpreter(tmp_path: Path):
    import trex.controller as controller

    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    community = tmp_path / "community_models"
    (community / "ckpts" / "AF2").mkdir(parents=True)
    weights = community / "ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text("weights")
    completed = MagicMock(returncode=0, stdout="prefix=/fake/venv\n")
    runtime_paths = dataclasses.replace(
        controller.RUNTIME_PATHS,
        repo_root=tmp_path,
        legacy_complexa_repo=tmp_path,
        complexa_python=python,
    )

    with (patch.object(controller.subprocess, "run", return_value=completed) as run,):
        output = _require_p2_runtime_available(
            require_af2=True,
            require_proteinmpnn=True,
            runtime_paths=runtime_paths,
        )

    assert output == "prefix=/fake/venv"
    assert run.call_args.args[0][0] == str(python)
    assert run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_p2_runtime_preflight_fails_on_import_error(tmp_path: Path):
    import trex.controller as controller

    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)
    community = tmp_path / "community_models"
    weights = community / "ProteinMPNN/vanilla_model_weights/v_48_020.pt"
    weights.parent.mkdir(parents=True)
    weights.write_text("weights")
    completed = MagicMock(returncode=1, stdout="ModuleNotFoundError: jax")
    runtime_paths = dataclasses.replace(
        controller.RUNTIME_PATHS,
        legacy_complexa_repo=tmp_path,
        complexa_python=python,
    )
    with (
        patch.object(controller.subprocess, "run", return_value=completed),
        pytest.raises(SystemExit, match="runtime preflight exited 1"),
    ):
        _require_p2_runtime_available(
            require_af2=False,
            require_proteinmpnn=True,
            runtime_paths=runtime_paths,
        )


def test_runtime_paths_configure_foldseek_and_mmseqs_commands(
    tmp_path: Path,
) -> None:
    import trex.controller as controller

    runtime_paths = dataclasses.replace(
        controller.RUNTIME_PATHS,
        foldseek_binary=tmp_path / "tools/foldseek",
        mmseqs_binary=tmp_path / "tools/mmseqs",
    )

    config = _with_runtime_tool_paths(LiveTickConfig(), runtime_paths)

    assert config.foldseek.binary == str(runtime_paths.foldseek_binary)
    assert config.sequence_dedup.binary == str(runtime_paths.mmseqs_binary)


def test_family_timeout_table_covers_all_dispatch_families():
    """Every method_family the controller can dispatch has a timeout."""
    required = {
        "bindcraft",
        "complexa_beam",
        "complexa_best_of_n",
        "complexa_fk_steering",
        "complexa_mcts",
        "structure_refilter",
        "proteinmpnn_redesign",
        "boltzgen",
    }
    assert required.issubset(
        set(FAMILY_TIMEOUT_S)
    ), f"missing: {required - set(FAMILY_TIMEOUT_S)}"


def test_lengths_per_target_known_targets():
    """Registered targets have generation-length defaults."""
    assert "05_CD45" in LENGTHS_PER_TARGET
    assert "23_BetV1" in LENGTHS_PER_TARGET
    assert "30_SC2RBD" in LENGTHS_PER_TARGET


# ---------------------------------------------------------------------------
# _WorkerSlot lifecycle
# ---------------------------------------------------------------------------


def test_worker_slot_starts_idle():
    s = _WorkerSlot(slot_id=0, gpu_id="1")
    assert not s.busy
    assert not s.is_done()
    assert s.cand is None
    assert s.proc is None


def test_worker_slot_busy_when_proc_assigned():
    s = _WorkerSlot(slot_id=0, gpu_id="1")
    proc = MagicMock()
    proc.poll.return_value = None
    s.proc = proc
    s.cand = _cand()
    s.launched_at = 100.0
    assert s.busy
    assert not s.is_done()  # poll returns None → still running


def test_worker_slot_is_done_when_poll_returns_exit_code():
    s = _WorkerSlot(slot_id=0, gpu_id="1")
    proc = MagicMock()
    proc.poll.return_value = 0
    s.proc = proc
    assert s.busy
    assert s.is_done()


def test_worker_slot_free_clears_state():
    s = _WorkerSlot(slot_id=0, gpu_id="1")
    s.proc = MagicMock()
    s.cand = _cand()
    s.out_dir = Path("/tmp/x")
    s.parent_pdb_str = "/p.pdb"
    s.parent_result_id = "r_001"
    s.tick_id = "v7r001"
    s.launched_at = 100.0
    s.free()
    assert not s.busy
    assert s.cand is None
    assert s.out_dir is None
    assert s.parent_pdb_str == ""
    assert s.parent_result_id == ""
    assert s.tick_id == ""
    assert s.launched_at == 0.0


# ---------------------------------------------------------------------------
# _dispatch_candidate_to_gpu — family routing
# ---------------------------------------------------------------------------


def test_dispatch_returns_none_for_unknown_family(tmp_path: Path):
    arc = Archive(tmp_path)
    cand = _cand(family="rfdiffusion")  # not in dispatch tree
    out = _dispatch_candidate_to_gpu(
        cand,
        gpu_id="1",
        archive=arc,
        target=_target(),
        target_pdb="/tmp/none.pdb",
        round_id=1,
        archive_root=tmp_path,
    )
    assert out is None


def test_dispatch_bindcraft_calls_bindcraft_async(tmp_path: Path):
    arc = Archive(tmp_path)
    cand = _cand(family="bindcraft")
    fake_proc = MagicMock(name="proc")
    fake_out = tmp_path / "out_b"
    with patch(
        "trex.controller._exec_bindcraft_async",
        return_value=(fake_proc, fake_out),
    ) as mock_bc:
        out = _dispatch_candidate_to_gpu(
            cand,
            gpu_id="2",
            archive=arc,
            target=_target(),
            target_pdb="/tmp/target.pdb",
            round_id=5,
            archive_root=tmp_path,
        )
    assert out is not None
    (
        proc,
        out_dir,
        parent_pdb_str,
        parent_result_id,
        target_chains_csv,
        binder_chain,
    ) = out
    assert proc is fake_proc
    assert out_dir == fake_out
    assert parent_pdb_str == ""  # bindcraft is not chained
    assert parent_result_id == ""
    assert mock_bc.call_count == 1
    # gpu_id flowed through
    kwargs = mock_bc.call_args.kwargs
    assert kwargs["gpu_id"] == "2"
    assert kwargs["runtime_paths"] is not None


def test_dispatch_structure_refilter_skips_when_no_parent_pdb(tmp_path: Path):
    """Chained refilter needs a parent PDB; absent → return None."""
    arc = Archive(tmp_path)
    cand = _cand(family="structure_refilter")
    # No ResultRecord in archive, so _resolve_parent_artifact returns None.
    out = _dispatch_candidate_to_gpu(
        cand,
        gpu_id="1",
        archive=arc,
        target=_target(),
        target_pdb="/tmp/target.pdb",
        round_id=1,
        archive_root=tmp_path,
    )
    assert out is None


def test_pending_dispatch_exception_is_retryable_not_campaign_fatal(tmp_path: Path):
    arc = Archive(tmp_path)
    slot = _WorkerSlot(slot_id=0, gpu_id="0")
    cand = _cand(family="complexa_beam", cid="c_bad")
    pending = ["c_bad"]
    retries: dict[str, int] = {}

    def _boom(*args: Any, **kwargs: Any):
        raise OSError("fork failed")

    dispatched = _dispatch_pending_to_free_slots(
        [slot],
        pending,
        {"c_bad": cand},
        archive=arc,
        target=_target(),
        target_pdb="/tmp/target.pdb",
        archive_root=tmp_path,
        round_id=1,
        dispatch_fn=_boom,
        dispatch_retries=retries,
    )

    assert dispatched == 0
    assert pending == ["c_bad"]
    assert retries["c_bad"] == 1
    assert not slot.busy


def test_started_worker_is_tracked_even_if_dispatch_record_append_fails(tmp_path: Path):
    class FailingAppendArchive:
        def iter_records(self, cls):
            return iter(())

        def append(self, record):
            raise OSError("archive temporarily unavailable")

    slot = _WorkerSlot(slot_id=0, gpu_id="0")
    cand = _cand(family="complexa_beam", cid="c_started")
    proc = MagicMock()
    proc.poll.return_value = None

    def _started(*args: Any, **kwargs: Any):
        return proc, tmp_path / "worker_out", "", ""

    dispatched = _dispatch_pending_to_free_slots(
        [slot],
        ["c_started"],
        {"c_started": cand},
        archive=FailingAppendArchive(),
        target=_target(),
        target_pdb="/tmp/target.pdb",
        archive_root=tmp_path,
        round_id=1,
        dispatch_fn=_started,
        dispatch_retries={},
    )

    assert dispatched == 1
    assert slot.busy
    assert slot.proc is proc
    assert slot.cand is cand


def test_explicit_parent_missing_artifact_does_not_fallback_to_best_pdb(tmp_path: Path):
    """Explicit parent lineage must be exact.

    If a chain/refilter candidate cites parent A but parent A has no usable PDB,
    falling back to the best unrelated PDB silently scores the wrong binder and
    corrupts downstream attribution.
    """
    arc = Archive(tmp_path)
    best_pdb = tmp_path / "best.pdb"
    best_pdb.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    arc.append(
        ResultRecord(
            result_id="missing_artifact_parent",
            parent_ids=[],
            target_id="t1",
            backend_family="bindcraft",
            runtime_bucket_id="rb_v7",
            metrics={"pLDDT": 91.0},
            metrics_calibrated={},
            route_lineage=[],
            gpu_h=1.0,
            exit_status="ok",  # type: ignore[arg-type]
            bins={},
            artifacts={},
            panel_ready=False,
        )
    )
    arc.append(
        ResultRecord(
            result_id="best_unrelated",
            parent_ids=[],
            target_id="t1",
            backend_family="complexa_beam",
            runtime_bucket_id="rb_v7",
            metrics={"pLDDT": 99.0},
            metrics_calibrated={},
            route_lineage=[],
            gpu_h=0.1,
            exit_status="ok",  # type: ignore[arg-type]
            bins={},
            artifacts={"pdb_path": str(best_pdb)},
            panel_ready=False,
        )
    )
    cand = _cand(family="structure_refilter")
    cand = dataclasses.replace(cand, parent_result_id="missing_artifact_parent")
    assert _resolve_parent_artifact(arc, cand) is None


@pytest.mark.parametrize("archive_name", ["first_run", "relocated_run"])
@pytest.mark.parametrize("candidate_id", ["cmplx_001", "nested/cmplx_001"])
def test_dispatch_complexa_preserves_historical_seed_input(
    tmp_path: Path,
    archive_name: str,
    candidate_id: str,
):
    archive_root = tmp_path / archive_name
    arc = Archive(archive_root)
    cand = _cand(family="complexa_beam", cid=candidate_id)
    fake_proc = MagicMock()
    fake_out = tmp_path / "complexa_out"
    with patch(
        "trex.controller._exec_complexa_async",
        return_value=(fake_proc, fake_out),
    ) as mock_c:
        out = _dispatch_candidate_to_gpu(
            cand,
            gpu_id="3",
            archive=arc,
            target=_target(),
            target_pdb="/tmp/target.pdb",
            round_id=7,
            archive_root=archive_root,
        )
    assert out is not None
    args = mock_c.call_args
    assert args.kwargs["gpu_id"] == "3"
    # run_name is part of the seed hash, not merely an output-directory label.
    # Moving an archive must not change the historical per-launch seed input.
    assert args.args[3] == f"v7_r007_{candidate_id.replace('/', '_')}"
    output_namespace = args.kwargs["output_namespace"]
    assert len(output_namespace) == 12
    assert all(character in "0123456789abcdef" for character in output_namespace)


# ---------------------------------------------------------------------------
# _parse_and_archive_slot
# ---------------------------------------------------------------------------


def test_parse_and_archive_returns_zero_for_empty_slot(tmp_path: Path):
    arc = Archive(tmp_path)
    slot = _WorkerSlot(slot_id=0, gpu_id="1")  # no cand, no out_dir
    n = _parse_and_archive_slot(
        slot,
        rc=0,
        archive=arc,
        target=_target(),
        chain_seq_ref=[0],
    )
    assert n == 0


def test_parse_and_archive_appends_records_from_parser(tmp_path: Path):
    """Mock the parser; verify ResultRecords land in archive."""
    arc = Archive(tmp_path)
    cand = _cand(family="bindcraft", cid="c1")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r001"

    fake_records = [
        ResultRecord(
            result_id="r_001",
            parent_ids=["c1"],
            target_id="t1",
            backend_family="bindcraft",
            runtime_bucket_id="rb_v7",
            metrics=dict(_PASS_METRICS),
            metrics_calibrated={},
            route_lineage=[],
            gpu_h=1.0,
            exit_status="ok",  # type: ignore[arg-type]
            bins={"design": "d1"},
            artifacts={},
            panel_ready=False,
        ),
    ]
    with patch(
        "trex.controller.parse_bindcraft_output",
        return_value=fake_records,
    ):
        n = _parse_and_archive_slot(
            slot,
            rc=0,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
        )
    assert n == 1
    stored = list(arc.iter_records(ResultRecord))
    assert len(stored) == 1
    assert stored[0].result_id == "r_001"


def test_parse_and_archive_resilient_on_nonzero_rc(tmp_path: Path):
    """Attempt to recover usable outputs after a nonzero exit."""
    arc = Archive(tmp_path)
    cand = _cand(family="boltzgen")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r001"

    fake_records = [
        ResultRecord(
            result_id="r_bg",
            parent_ids=["c1"],
            target_id="t1",
            backend_family="boltzgen",
            runtime_bucket_id="rb_v7",
            metrics=dict(_PASS_METRICS),
            metrics_calibrated={},
            route_lineage=[],
            gpu_h=1.0,
            exit_status="ok",  # type: ignore[arg-type]
            bins={"boltzgen_design_iptm": "0.85"},
            artifacts={},
            panel_ready=False,
        ),
    ]
    with patch(
        "trex.controller.parse_boltzgen_output",
        return_value=fake_records,
    ):
        n = _parse_and_archive_slot(
            slot,
            rc=1,  # non-zero
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
        )
    assert n == 1


def test_parse_and_archive_handles_parse_error(tmp_path: Path):
    """ParseError → 0 records, slot still freeable. No archive write."""
    from trex.output_parsers import ParseError

    arc = Archive(tmp_path)
    cand = _cand(family="bindcraft")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r001"

    with patch(
        "trex.controller.parse_bindcraft_output",
        side_effect=ParseError("bad outputs"),
    ):
        n = _parse_and_archive_slot(
            slot,
            rc=0,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
        )
    assert n == 0
    assert list(arc.iter_records(ResultRecord)) == []


def test_parse_emits_synthetic_record_when_zero_records_and_elapsed(tmp_path: Path):
    """Record compute for an empty parsed output after meaningful execution."""
    from trex.output_parsers import ParseError

    arc = Archive(tmp_path)
    cand = _cand(family="bindcraft", cid="c_synth")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r050"

    with patch(
        "trex.controller.parse_bindcraft_output",
        return_value=[],  # zero records (BindCraft killed before any design accepted)
    ):
        n = _parse_and_archive_slot(
            slot,
            rc=-1,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
            elapsed_gpu_h=1.5,  # 90 min spent
        )
    # 0 ResultRecords reported via return, but synthetic record IS in archive
    assert n == 0
    stored = list(arc.iter_records(ResultRecord))
    assert len(stored) == 1
    synth = stored[0]
    assert synth.exit_status == "timeout"
    assert synth.backend_family == "bindcraft"
    assert synth.gpu_h == 1.5
    assert synth.metrics == {}  # no strict_yield, but family tried


def test_parse_no_synthetic_when_elapsed_too_short(tmp_path: Path):
    """Ignore a brief successful empty result below the dispatch-noise threshold."""
    arc = Archive(tmp_path)
    cand = _cand(family="bindcraft", cid="c_quick")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r051"
    with patch(
        "trex.controller.parse_bindcraft_output",
        return_value=[],
    ):
        _parse_and_archive_slot(
            slot,
            rc=0,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
            elapsed_gpu_h=0.002,  # 7s — below 18s threshold (dispatch noise)
        )
    assert list(arc.iter_records(ResultRecord)) == []


def test_parse_synthetic_captures_fast_crash(tmp_path: Path):
    """Record short failed executions even when they produced no designs."""
    arc = Archive(tmp_path)
    cand = _cand(family="bindcraft", cid="c_segfault")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r060"
    with patch(
        "trex.controller.parse_bindcraft_output",
        return_value=[],
    ):
        _parse_and_archive_slot(
            slot,
            rc=-11,
            archive=arc,
            target=_target(),  # -11 = SIGSEGV
            chain_seq_ref=[0],
            elapsed_gpu_h=0.01,  # 36s
        )
    stored = list(arc.iter_records(ResultRecord))
    assert len(stored) == 1
    assert stored[0].exit_status == "nonzero_exit"
    assert stored[0].bins["return_code"] == "-11"
    assert stored[0].gpu_h == 0.01


def test_parse_synthetic_captures_import_crash_below_gpu_threshold(tmp_path: Path):
    """A fast nonzero exit is infrastructure evidence, not dispatch noise."""
    arc = Archive(tmp_path)
    cand = _cand(family="structure_refilter", cid="c_import_error")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r061"
    with patch(
        "trex.controller.parse_af2_refilter_output",
        return_value=[],
    ):
        _parse_and_archive_slot(
            slot,
            rc=1,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
            elapsed_gpu_h=0.001,
        )
    stored = list(arc.iter_records(ResultRecord))
    assert len(stored) == 1
    assert stored[0].exit_status == "nonzero_exit"
    assert stored[0].gpu_h == 0.001


def test_parse_no_synthetic_when_real_records_returned(tmp_path: Path):
    """Synthetic record is only emitted when parser returned 0. Real
    records present → no synthetic."""
    arc = Archive(tmp_path)
    cand = _cand(family="bindcraft", cid="c_ok")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r052"
    real_rec = ResultRecord(
        result_id="r_real",
        parent_ids=["c_ok"],
        target_id="t1",
        backend_family="bindcraft",
        runtime_bucket_id="rb_v7",
        metrics=dict(_PASS_METRICS),
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=0.05,
        exit_status="ok",  # type: ignore[arg-type]
        bins={"design": "d1"},
        artifacts={},
        panel_ready=False,
    )
    with patch(
        "trex.controller.parse_bindcraft_output",
        return_value=[real_rec],
    ):
        _parse_and_archive_slot(
            slot,
            rc=0,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
            elapsed_gpu_h=1.5,
        )
    stored = list(arc.iter_records(ResultRecord))
    assert len(stored) == 1
    assert stored[0].result_id == "r_real"
    assert stored[0].exit_status == "ok"  # not synthetic


def test_parse_normalizes_per_record_gpu_h_to_elapsed(tmp_path: Path):
    """Distribute elapsed job compute across its parsed records without inflation."""
    arc = Archive(tmp_path)
    cand = _cand(family="complexa_best_of_n", cid="c_replica")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r080"

    # Parser emits 4 records each with gpu_h=0.5 (would sum to 2.0 inflated)
    fake_records = [
        ResultRecord(
            result_id=f"r_{i}",
            parent_ids=["c_replica"],
            target_id="t1",
            backend_family="complexa_best_of_n",
            runtime_bucket_id="rb_v7",
            metrics=dict(_PASS_METRICS),
            metrics_calibrated={},
            route_lineage=[],
            gpu_h=0.5,
            exit_status="ok",  # type: ignore[arg-type]
            bins={"sample_index": str(i)},
            artifacts={},
            panel_ready=False,
        )
        for i in range(4)
    ]
    with patch(
        "trex.controller.parse_complexa_output",
        return_value=fake_records,
    ):
        n = _parse_and_archive_slot(
            slot,
            rc=0,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
            elapsed_gpu_h=0.4,  # actual wall time: 24 min ≈ 0.4 GPU-h
        )
    assert n == 4
    stored = list(arc.iter_records(ResultRecord))
    assert len(stored) == 4
    # Each record's gpu_h normalized to elapsed / N = 0.4 / 4 = 0.10
    for r in stored:
        assert abs(r.gpu_h - 0.10) < 1e-9, f"got {r.gpu_h}"
    # Sum equals elapsed exactly
    assert abs(sum(r.gpu_h for r in stored) - 0.4) < 1e-9


def test_parse_skips_gpu_h_rewrite_when_elapsed_zero(tmp_path: Path):
    """Backwards compat: when elapsed_gpu_h is 0 (caller didn't pass it),
    leave parser-emitted gpu_h alone (existing behavior)."""
    arc = Archive(tmp_path)
    cand = _cand(family="complexa_beam", cid="c_nl")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r081"
    fake = [
        ResultRecord(
            result_id="r_x",
            parent_ids=["c_nl"],
            target_id="t1",
            backend_family="complexa_beam",
            runtime_bucket_id="rb_v7",
            metrics=dict(_PASS_METRICS),
            metrics_calibrated={},
            route_lineage=[],
            gpu_h=0.5,
            exit_status="ok",  # type: ignore[arg-type]
            bins={},
            artifacts={},
            panel_ready=False,
        ),
    ]
    with patch(
        "trex.controller.parse_complexa_output",
        return_value=fake,
    ):
        _parse_and_archive_slot(
            slot,
            rc=0,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
            elapsed_gpu_h=0.0,
        )
    stored = list(arc.iter_records(ResultRecord))
    assert stored[0].gpu_h == 0.5  # unchanged


def test_parse_synthetic_exit_status_distinguishes_timeout_from_nonzero(tmp_path: Path):
    """rc=-1 → exit_status='timeout' (killed by salvage path); rc>=0 with
    empty parse → exit_status='no_artifacts' (worker finished cleanly but
    produced nothing parseable)."""
    arc = Archive(tmp_path)
    cand = _cand(family="boltzgen", cid="c_a")
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r053"
    with patch(
        "trex.controller.parse_boltzgen_output",
        return_value=[],
    ):
        _parse_and_archive_slot(
            slot,
            rc=-1,
            archive=arc,
            target=_target(),
            chain_seq_ref=[0],
            elapsed_gpu_h=1.5,
        )
    assert list(arc.iter_records(ResultRecord))[0].exit_status == "timeout"

    # Now rc=0 case (worker exited cleanly with no outputs)
    arc2 = Archive(tmp_path / "arc2")
    slot2 = _WorkerSlot(slot_id=1, gpu_id="2")
    slot2.cand = _cand(family="boltzgen", cid="c_b")
    slot2.out_dir = tmp_path / "arc2"
    slot2.tick_id = "v7r054"
    with patch(
        "trex.controller.parse_boltzgen_output",
        return_value=[],
    ):
        _parse_and_archive_slot(
            slot2,
            rc=0,
            archive=arc2,
            target=_target(),
            chain_seq_ref=[0],
            elapsed_gpu_h=1.5,
        )
    assert list(arc2.iter_records(ResultRecord))[0].exit_status == "no_artifacts"


def test_chain_seq_increments_for_auto_chain_children(tmp_path: Path):
    """Auto-chain children use a shared mutable counter so candidate_ids
    are unique across slots within one controller run."""
    arc = Archive(tmp_path)
    # diagnostic-only generator with a downstream plan
    cand = ActionCandidate(
        candidate_id="bg_parent",
        hypothesis_ids=["h1"],
        parent_result_id=None,
        method_family="boltzgen",
        operator_id="op",
        lane_id="boltzgen",
        config_delta={},
        downstream_route_plan=["structure_refilter"],
        estimated_cost_class="standard",  # type: ignore[arg-type]
        expected_signal="x",
        evidence_refs=["e1"],
        feasibility=_feas(),
    )
    arc.append(cand)
    slot = _WorkerSlot(slot_id=0, gpu_id="1")
    slot.cand = cand
    slot.out_dir = tmp_path
    slot.tick_id = "v7r003"

    rec = ResultRecord(
        result_id="r_bg_001",
        parent_ids=["bg_parent"],
        target_id="t1",
        backend_family="boltzgen",
        runtime_bucket_id="rb_v7",
        metrics=dict(_PASS_METRICS),
        metrics_calibrated={},
        route_lineage=[],
        gpu_h=1.0,
        exit_status="ok",  # type: ignore[arg-type]
        bins={"boltzgen_design_iptm": "0.85"},
        artifacts={"pdb_path": str(tmp_path / "fake.pdb")},  # presence-only check
        panel_ready=False,
    )
    # touch artifact path so the auto-chain scoring loop sees it
    (tmp_path / "fake.pdb").write_text("ATOM  ")

    chain_seq_ref = [0]
    with patch(
        "trex.controller.parse_boltzgen_output",
        return_value=[rec],
    ):
        _parse_and_archive_slot(
            slot,
            rc=0,
            archive=arc,
            target=_target(),
            chain_seq_ref=chain_seq_ref,
        )
    # Counter incremented at least once; auto-chain child appended.
    assert chain_seq_ref[0] >= 1
    children = [
        c
        for c in arc.iter_records(ActionCandidate)
        if c.candidate_id.startswith("chain_")
    ]
    # Auto-chain emission requires the registered family to be "available";
    # if it's not on this cluster the chain skips. Either way, never crash.
    if children:
        assert all("v7r003" in c.candidate_id for c in children)
        assert all(c.parent_result_id == "r_bg_001" for c in children)


@pytest.mark.parametrize("maximum_iterations", [1, 3])
@pytest.mark.parametrize("include_selector", [False, True])
def test_controller_replans_and_waits_when_idle_without_new_results(
    tmp_path: Path,
    maximum_iterations: int,
    include_selector: bool,
) -> None:
    """Preserve historical retries, including an empty completed selection.

    No new results arrive, and no jobs are queued. Each loop must still reach
    planning and the idle backoff instead of gating or terminating early.
    External executables, LLM calls, and sleeping are replaced by test doubles.
    """

    from trex import controller

    archive_root = tmp_path / "archive"
    constraint_path = tmp_path / "target.json"
    target_pdb = tmp_path / "target.pdb"
    constraint_path.write_text(json.dumps(to_jsonable(_target())))
    target_pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  "
        "1.00 20.00           C\nTER\nEND\n"
    )
    runtime_paths = RuntimePaths.from_environment({}, default_repo_root=tmp_path)

    def tick_result(*args, **kwargs):
        result = {
            "evidence": {
                "state_label": "low_evidence",
                "completed_children": 0,
                "run_su_count": 0,
            }
        }
        if include_selector and not kwargs.get("evidence_only", False):
            result["selector"] = {"launched": 0}
        return result

    checkpoint = ProgressCheckpointResult(
        last_checkpoint_at=0.0,
        latest_state=None,
        wrote_checkpoint=False,
    )

    with patch.object(
        controller, "_require_foldseek_available", return_value="/fake/foldseek"
    ), patch.object(
        controller, "run_live_tick", side_effect=tick_result
    ) as live_tick, patch.object(
        controller, "refresh_progress_checkpoint", return_value=checkpoint
    ), patch.object(
        controller.time, "sleep"
    ) as sleep:
        controller.main(
            [
                "--archive-root",
                str(archive_root),
                "--target-constraint",
                str(constraint_path),
                "--target-pdb",
                str(target_pdb),
                "--enabled-families",
                "complexa_beam",
                "--worker-gpus",
                "1",
                "--max-wall-h",
                "1",
            ],
            runtime_paths=runtime_paths,
            loop_config=ControllerLoopConfig(
                maximum_iterations=maximum_iterations,
                poll_interval_seconds=0.0,
                state_probe_cache_seconds=0.0,
            ),
        )

    assert live_tick.call_count == 2 * maximum_iterations
    assert live_tick.call_args_list[0].kwargs["evidence_only"] is True
    assert "evidence_only" not in live_tick.call_args_list[1].kwargs
    planning_calls = [
        call
        for call in live_tick.call_args_list
        if not call.kwargs.get("evidence_only", False)
    ]
    assert [call.kwargs["tick_id"] for call in planning_calls] == [
        f"v7r{index:03d}" for index in range(1, maximum_iterations + 1)
    ]
    assert sleep.call_count == maximum_iterations
    assert all(call.args == (120.0,) for call in sleep.call_args_list)
    assert (archive_root / ".controller.lock").is_file()
