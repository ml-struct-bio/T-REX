"""Run an asynchronous design campaign for one target.

Planner and Supervisor use the LLM endpoint; worker GPUs execute generation,
redesign, and AF2 evaluation jobs. Deterministic code summarizes results,
constructs and validates candidates, maintains the queue, and records actual
worker starts and failures in DispatchRecord entries. The campaign advisory
guard is deterministic.

This is the live runner. phase5_launcher.py supports staged-output simulation.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import shutil
import socket
import subprocess
import time
from dataclasses import replace as dc_replace
from functools import partial
from pathlib import Path
from typing import Any

from .archive import Archive
from .backend_extensions import (
    BackendLaunchContext,
    get_backend_adapter,
    validate_backend_command,
)
from .backends.af2 import prepare_af2_refilter_launch
from .backends.bindcraft import prepare_bindcraft_launch
from .backends.boltzgen import (
    parse_hotspot as _parse_hotspot,
    prepare_boltzgen_launch,
    write_boltzgen_yaml as _write_boltzgen_yaml,
)
from .backends.complexa import (
    build_complexa_overrides,
    build_complexa_shell_command,
    canonical_af2_overrides as _complexa_canonical_af2_overrides,
    refinement_overrides as _complexa_refinement_overrides,
    reward_filter_overrides as _complexa_reward_filter_overrides,
)
from .backends.proteinmpnn import prepare_proteinmpnn_launch
from .backends.process_management import (
    isolated_subprocess_environment as _isolated_env_for_subprocess,
    popen_kwargs as _popen_kwargs,
    terminate_process_group as _terminate_process_group,
)
from .campaign.runtime import (
    BINDCRAFT_FLOOR_S,
    BINDCRAFT_INCREMENTAL_PARSE_INTERVAL_S,
    BINDCRAFT_YIELD_WINDOW_S,
    ControllerLoopConfig,
    DEFAULT_HARD_CEILING_S as _DEFAULT_HARD_CEILING_S,
    FAMILY_TIMEOUT_S,
    HARD_CEILING_S,
    RuntimePaths,
    WorkerSlot as _WorkerSlot,
    controller_sleep_seconds as _controller_sleep_seconds,
    resolve_environment_executable_path as _env_executable_path,
    resolve_environment_path as _env_path,
    worker_timeout_reason as _worker_timeout_reason,
)
from .campaign.controller_args import parse_controller_args
from .execution.circuit_breaker import (
    FamilyCircuitBreakerDependencies,
    FamilyCircuitBreakerRequest,
    evaluate_family_circuit_breakers,
    family_circuit_broken as _family_circuit_broken,
)
from .execution.dispatch import (
    DispatchQueueDependencies,
    MAX_DISPATCH_RETRIES as _MAX_DISPATCH_RETRIES,
    PermanentDispatchSkip,
    dispatch_pending_to_free_slots,
)
from .execution.progress_checkpoint import (
    ProgressCheckpointDependencies,
    ProgressCheckpointRequest,
    build_cheap_checkpoint_evidence as _cheap_checkpoint_evidence,
    read_controller_checkpoint as _read_controller_checkpoint,
    refresh_progress_checkpoint,
    resolve_charged_gpu_count as _checkpoint_charged_gpu_count,
    write_controller_checkpoint as _write_controller_checkpoint,
)
from .execution.result_processing import (
    OutputParserRegistry,
    WorkerOutputRequest,
    create_synthetic_result_record,
    process_worker_output,
    synthetic_result_exit_status as _synthetic_exit_status,
)
from .execution.score_conversion_scheduling import (
    ScoreConversionSchedulingRequest,
    build_score_conversion_schedule,
    canonical_strict_metrics_present,
    deduplicate_scored_parent_results,
    is_canonical_score_conversion_result,
    result_requires_score_conversion,
    results_require_score_conversion,
    score_conversion_candidate_limit,
    score_conversion_parent_identity,
    select_score_conversion_parents,
)
from .execution.worker_supervision import (
    WorkerSupervisionDependencies,
    drain_worker_slots,
    supervise_worker_slots,
)
from .live_tick import (
    LiveTickConfig,
    _diagnostic_chain_backlog,
    _has_usable_structure_artifact,
    _index_actions_by_spawned_result,
    _native_strict_like_pending_score_conversion,
    _proxy_promising_pending_score_conversion,
    run_live_tick,
)
from .output_parsers.af2_refilter import parse_af2_refilter_output
from .output_parsers.bindcraft import parse_bindcraft_output
from .output_parsers.boltzgen import parse_boltzgen_output
from .output_parsers.complexa import parse_complexa_output
from .output_parsers.proteinmpnn import parse_proteinmpnn_output
from .refilter_roles import (
    CANONICAL_SCORE_CONVERSION,
    PARENT_MODEL_REFOLD,
    infer_refilter_role,
    is_canonical_score_conversion,
)
from .schemas import (
    ActionCandidate,
    DispatchRecord,
    EvidenceSummary,
    FeasibilityCheck,
    LaunchDecision,
    ResultRecord,
    TargetConstraint,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = RuntimePaths.from_environment(default_repo_root=_REPO_ROOT)

# Compatibility aliases for code that imported the former controller globals.
# Production launchers consume RuntimePaths directly below.
TREX_REPO_ROOT = RUNTIME_PATHS.repo_root
TREX_EXTERNAL_ROOT = RUNTIME_PATHS.external_root
TREX_COMPLEXA_REPO = RUNTIME_PATHS.complexa_repo
TREX_LEGACY_COMPLEXA_REPO = RUNTIME_PATHS.legacy_complexa_repo
TREX_BINDCRAFT_REPO = RUNTIME_PATHS.bindcraft_repo
TREX_BINDCRAFT_ENV = RUNTIME_PATHS.bindcraft_env
P2_LEGACY_PROTEINA_COMPLEXA = RUNTIME_PATHS.legacy_complexa_repo
P2_LEGACY_COMMUNITY_ROOT = RUNTIME_PATHS.community_models_root
P2_AF2_DATA_DIR = RUNTIME_PATHS.af2_data_dir
P2_PYTHON = RUNTIME_PATHS.complexa_python
P2_BOLTZGEN_BIN = RUNTIME_PATHS.boltzgen_binary
P2_BOLTZGEN_REPO = RUNTIME_PATHS.boltzgen_repo
P2_BOLTZGEN_CACHE = RUNTIME_PATHS.boltzgen_cache


def _exec_af2_refilter_async(
    cand: ActionCandidate,
    out_dir: Path,
    parent_pdb: Path,
    target_chain: str,
    binder_chain: str,
    gpu_id: str,
    runtime_paths: RuntimePaths | None = None,
) -> tuple["subprocess.Popen | None", Path]:
    """Launch standardized AF2 evaluation and return the process handle and output
    directory.

    The caller waits and parses af2_refilter_result.json for designs requiring separate
    evaluation.
    """
    paths = runtime_paths or RUNTIME_PATHS
    out_dir.mkdir(parents=True, exist_ok=True)
    if not paths.complexa_python.exists():
        print(
            "  [SKIP] python missing for AF2 refilter: " f"{paths.complexa_python}",
            flush=True,
        )
        return None, out_dir
    if not parent_pdb.exists():
        print(f"  [SKIP] parent_pdb missing for AF2 refilter: {parent_pdb}", flush=True)
        return None, out_dir
    if not paths.af2_data_dir.exists():
        print(
            f"  [SKIP] AF2 multimer params missing: {paths.af2_data_dir}",
            flush=True,
        )
        return None, out_dir

    launch = prepare_af2_refilter_launch(
        output_dir=out_dir,
        parent_pdb=parent_pdb,
        target_chain=target_chain,
        binder_chain=binder_chain,
        gpu_id=gpu_id,
        python=paths.complexa_python,
        community_root=paths.community_models_root,
        af2_data_dir=paths.af2_data_dir,
        trex_repo_root=paths.repo_root,
        candidate_id=cand.candidate_id,
        campaign_seed=RUN_SEED,
        config_delta=cand.config_delta,
    )
    log_path = out_dir / "worker.log"
    log_fp = open(log_path, "w")
    print(f"  [worker:af2_refilter][gpu={gpu_id}] pdb={parent_pdb.name}", flush=True)
    proc = subprocess.Popen(
        launch.argv,
        env=launch.environment,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        **_popen_kwargs(),
    )
    log_fp.close()  # The child owns a duplicate file descriptor; close the parent handle.
    return proc, launch.output_dir


def _cif_to_pdb(cif_path: Path) -> Path | None:
    """Convert CIF to a cached sibling PDB for PDB-only consumers. Return None on
    conversion failure.
    """
    if cif_path.suffix.lower() != ".cif":
        return cif_path
    pdb_path = cif_path.with_suffix(".converted.pdb")
    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        return pdb_path
    try:
        from Bio.PDB import MMCIFParser, PDBIO  # type: ignore

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(cif_path.stem, str(cif_path))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path))
        if pdb_path.exists() and pdb_path.stat().st_size > 0:
            return pdb_path
    except Exception as e:
        print(f"  [cif_to_pdb] {cif_path.name} failed: {e}", flush=True)
    return None


def _pdb_chain_ids(path: Path) -> tuple[str, ...]:
    """Return distinct chain IDs from ATOM rows in file order.

    Supports PDB and mmCIF input; returns an empty tuple for missing or unreadable files.
    """
    if not path.exists():
        return ()
    out: list[str] = []
    is_cif = path.suffix.lower() in {".cif", ".mmcif"}
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return ()
    for raw in lines:
        if not raw.startswith("ATOM"):
            continue
        if is_cif:
            parts = raw.split()
            chain_id = parts[6].strip() if len(parts) > 6 else ""
        else:
            if len(raw) < 22:
                continue
            chain_id = raw[21].strip()
        if chain_id and chain_id not in out:
            out.append(chain_id)
    return tuple(out)


_STANDARD_AA3 = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


def _noncanonical_residue_names(path: Path, *, chain_id: str | None = None) -> set[str]:
    """Noncanonical ATOM residue names in a PDB, optionally restricted to a chain.

    BoltzGen can emit unknown/nonstandard residue names that ColabDesign maps to
    amino-acid index 20, causing AF2 refilter to crash before producing a
    ResultRecord. Rejecting those parents before dispatch records the candidate
    as a skipped/failed conversion instead of burning a worker on a known crash.
    """
    if not path.exists() or path.suffix.lower() not in {".pdb", ".ent"}:
        return set()
    bad: set[str] = set()
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return set()
    for raw in lines:
        if not raw.startswith("ATOM") or len(raw) < 22:
            continue
        ch = raw[21].strip()
        if chain_id and ch != chain_id:
            continue
        resn = raw[17:20].strip().upper()
        if resn and resn not in _STANDARD_AA3:
            bad.add(resn)
    return bad


def _parent_result_by_id(archive: Archive, result_id: str) -> ResultRecord | None:
    for r in archive.iter_records(ResultRecord):
        if r.result_id == result_id:
            return r
    return None


def _pdb_residues_per_chain(path: Path) -> dict[str, int]:
    """Count distinct residues per chain (ATOM lines only). PDB + mmCIF."""
    if not path.exists():
        return {}
    is_cif = path.suffix.lower() in {".cif", ".mmcif"}
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return {}
    for raw in lines:
        if not raw.startswith("ATOM"):
            continue
        if is_cif:
            parts = raw.split()
            if len(parts) < 9:
                continue
            ch = parts[6].strip()
            resseq = parts[8].strip() if len(parts) > 8 else parts[-1]
        else:
            if len(raw) < 27:
                continue
            ch = raw[21].strip()
            resseq = raw[22:27].strip()
        if not ch:
            continue
        key = (ch, resseq)
        if key in seen:
            continue
        seen.add(key)
        counts[ch] = counts.get(ch, 0) + 1
    return counts


# Set once in main() from --target-pdb. The target chain is COPIED VERBATIM
# into output structures. Production dispatch verifies all target subunit
# sequences against this reference, independently of output chain labels.
# Residue counts remain only for compatibility with legacy helper callers.
TARGET_PDB_PATH: Path | None = None
TARGET_RES_COUNT: int | None = None
TARGET_CHAIN_IDS: tuple[str, ...] = ()


def _csv_chains(chain_ids: list[str] | tuple[str, ...]) -> str:
    return ",".join(c for c in chain_ids if c)


def _first_free_chain_id(used: list[str] | tuple[str, ...], default: str = "B") -> str:
    used_set = {c for c in used if c}
    if default and default not in used_set:
        return default
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c not in used_set:
            return c
    return default or "B"


# Mix the campaign seed into supported backend seeds to distinguish repeated campaigns
# with identical candidate settings.
RUN_SEED: int = 0


def _resolve_pdb_chains(
    pdb_path: Path,
    *,
    target_default: str = "A",
    binder_default: str = "B",
    target_res_count: int | None = None,
    target_chain_ids: list[str] | tuple[str, ...] | None = None,
    target_pdb: Path | None = None,
) -> tuple[str, str]:
    """Introspect a parent PDB to pick (target_chain, binder_chain).

    Parent PDBs reach the AF2 refilter from heterogeneous sources whose chain
    conventions DIFFER. Prefer the target constraint's chain IDs when a
    multi-chain target is preserved. Otherwise use
    residue-count matching only for legacy callers without a target reference.
    All production dispatch paths supply target_pdb and verify exact target
    subunit sequences; unknown or ambiguous roles cannot dispatch.
    """
    if target_pdb is not None:
        from .output_identity import resolve_output_chains
        from .af2_chain_identity import ChainIdentityError

        try:
            mapping = resolve_output_chains(
                pdb_path,
                target_pdb,
                target_chain_ids or TARGET_CHAIN_IDS,
            )
        except ChainIdentityError as exc:
            raise PermanentDispatchSkip(
                f"Unresolved parent output chain identity: {exc}"
            ) from exc
        return ",".join(mapping["target_chains"]), mapping["binder_chain"]
    counts = _pdb_residues_per_chain(pdb_path)
    if len(counts) < 2:
        return target_default, binder_default

    known = tuple(c for c in (target_chain_ids or TARGET_CHAIN_IDS) if c)
    tref = target_res_count if target_res_count is not None else TARGET_RES_COUNT
    if len(known) > 1 and tref is not None:
        tol = 0.15 * tref
        target_like = tuple(c for c, n in counts.items() if abs(n - tref) <= tol)
        if len(target_like) == len(known):
            non_target = [c for c in counts if c not in target_like]
            if non_target:
                binder = max(non_target, key=lambda c: counts[c])
                return _csv_chains(target_like), binder
        collapsed_target = [
            c
            for c, n in counts.items()
            if abs(n - tref * len(known)) <= tol * len(known)
        ]
        if len(collapsed_target) == 1:
            target = collapsed_target[0]
            binder = max((c for c in counts if c != target), key=lambda c: counts[c])
            return target, binder

    if len(known) > 1:
        present_known = tuple(c for c in known if c in counts)
        if len(present_known) == len(known):
            non_target = [c for c in counts if c not in present_known]
            if non_target:
                binder = max(non_target, key=lambda c: counts[c])
                return _csv_chains(present_known), binder

    if tref is not None:
        target = min(counts, key=lambda c: abs(counts[c] - tref))
        binder = max((c for c in counts if c != target), key=lambda c: counts[c])
        if abs(counts[target] - tref) > 0.15 * tref:
            print(
                f"  [WARN _resolve_pdb_chains] no chain matches target size "
                f"~{tref} in {pdb_path.name} (chains={counts}); using closest={target}",
                flush=True,
            )
        return target, binder
    print(
        f"  [WARN _resolve_pdb_chains] target size unknown for {pdb_path.name} "
        f"(chains={counts}); defaulting target={target_default}/binder={binder_default}",
        flush=True,
    )
    return target_default, binder_default


def _resolve_parent_artifact(
    archive: Archive,
    cand: ActionCandidate,
    *,
    metric: str = "pLDDT",
    prefer_high: bool = True,
) -> tuple[Path, str] | None:
    """Resolve the candidate parent, or the highest-pLDDT usable result when no parent is
    specified.

    Return (PDB path, result ID), converting CIF when necessary, or None if no usable
    parent exists.
    """

    def _pdb_of(r: ResultRecord) -> Path | None:
        if not r.artifacts:
            return None
        p = r.artifacts.get("pdb_path")
        if p and Path(p).exists():
            cand_path = Path(p)
            if cand_path.suffix.lower() == ".cif":
                return _cif_to_pdb(cand_path)
            return cand_path
        d = r.artifacts.get("pdb_dir")
        if d:
            pdb_cands = sorted(Path(d).glob("*.pdb"))
            if pdb_cands:
                return pdb_cands[0]
            cif_cands = sorted(Path(d).glob("*.cif"))
            if cif_cands:
                return _cif_to_pdb(cif_cands[0])
        return None

    if cand.parent_result_id:
        for r in archive.iter_records(ResultRecord):
            if r.result_id == cand.parent_result_id:
                pdb = _pdb_of(r)
                if pdb is not None:
                    return (pdb, r.result_id)
                print(
                    f"  [SKIP] parent_result_id={cand.parent_result_id} "
                    "found but has no usable structure artifact; refusing "
                    "best-PDB fallback for explicit parent",
                    flush=True,
                )
                return None
        print(
            f"  [SKIP] parent_result_id={cand.parent_result_id} not found; "
            "refusing best-PDB fallback for explicit parent",
            flush=True,
        )
        return None

    best: tuple[float, Path, str] | None = None
    for r in archive.iter_records(ResultRecord):
        v = (r.metrics or {}).get(metric)
        if v is None:
            continue
        pdb = _pdb_of(r)
        if pdb is None:
            continue
        v = float(v)
        if (
            best is None
            or (prefer_high and v > best[0])
            or (not prefer_high and v < best[0])
        ):
            best = (v, pdb, r.result_id)
    return (best[1], best[2]) if best is not None else None


def _exec_boltzgen_async(
    cand: ActionCandidate,
    out_dir: Path,
    target_id: str,
    target_pdb: str,
    hotspots: list[str],
    chain_ids: list[str],
    length_range: tuple[int, int],
    gpu_id: str,
    runtime_paths: RuntimePaths | None = None,
) -> tuple["subprocess.Popen | None", Path]:
    """Launch BoltzGen and return its diagnostic outputs for parsing and standardized
    evaluation.
    """
    paths = runtime_paths or RUNTIME_PATHS
    out_dir.mkdir(parents=True, exist_ok=True)
    if not paths.boltzgen_binary.exists():
        print(
            f"  [SKIP] boltzgen binary missing: {paths.boltzgen_binary}",
            flush=True,
        )
        return None, out_dir
    binder_chain = _first_free_chain_id(tuple(chain_ids), default="B")
    launch = prepare_boltzgen_launch(
        output_dir=out_dir,
        target_id=target_id,
        target_pdb=target_pdb,
        hotspots=hotspots,
        chain_ids=chain_ids,
        binder_chain=binder_chain,
        length_range=length_range,
        gpu_id=gpu_id,
        binary=paths.boltzgen_binary,
        repo=paths.boltzgen_repo,
        cache=paths.boltzgen_cache,
        config_delta=cand.config_delta,
    )
    design_count = int((cand.config_delta or {}).get("num_designs", 16))
    log_path = out_dir / "worker.log"
    log_fp = open(log_path, "w")
    print(
        f"  [worker:boltzgen][gpu={gpu_id}] "
        f"target={target_id} num_designs={design_count}",
        flush=True,
    )
    proc = subprocess.Popen(
        launch.argv,
        env=launch.environment,
        cwd=str(launch.cwd) if launch.cwd is not None else None,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        **_popen_kwargs(),
    )
    log_fp.close()
    return proc, launch.output_dir


def _exec_proteinmpnn_async(
    cand: ActionCandidate,
    out_dir: Path,
    parent_pdb: Path,
    gpu_id: str,
    binder_chain: str = "B",
    runtime_paths: RuntimePaths | None = None,
) -> tuple["subprocess.Popen | None", Path]:
    """Launch ProteinMPNN sequence redesign.

    The parser threads sequences onto the parent structure; downstream AF2 evaluation
    supplies qualification measurements.
    """
    paths = runtime_paths or RUNTIME_PATHS
    out_dir.mkdir(parents=True, exist_ok=True)
    proteinmpnn_directory = paths.proteinmpnn_dir
    if not proteinmpnn_directory.exists() or not paths.proteinmpnn_weights.exists():
        print(f"  [SKIP] ProteinMPNN scripts/weights missing", flush=True)
        return None, out_dir
    if not paths.complexa_python.exists() or not parent_pdb.exists():
        print(f"  [SKIP] python or parent PDB missing", flush=True)
        return None, out_dir

    launch = prepare_proteinmpnn_launch(
        output_dir=out_dir,
        parent_pdb=parent_pdb,
        binder_chain=binder_chain,
        gpu_id=gpu_id,
        python=paths.complexa_python,
        proteinmpnn_dir=proteinmpnn_directory,
        candidate_id=cand.candidate_id,
        campaign_seed=RUN_SEED,
        config_delta=cand.config_delta,
    )
    sequence_count = int((cand.config_delta or {}).get("num_seq_per_target", 8))
    log_path = out_dir / "worker.log"
    log_fp = open(log_path, "w")
    print(
        f"  [worker:proteinmpnn][gpu={gpu_id}] "
        f"parent={parent_pdb.name} num_seq={sequence_count}",
        flush=True,
    )
    proc = subprocess.Popen(
        launch.argv,
        env=launch.environment,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        **_popen_kwargs(),
    )
    log_fp.close()
    return proc, launch.output_dir


def _exec_bindcraft(
    cand: ActionCandidate,
    out_dir: Path,
    target_id: str,
    target_pdb: str,
    hotspots: str,
    chains: str,
    lengths: tuple[int, int],
    gpu_id: str = "1",
    runtime_paths: RuntimePaths | None = None,
) -> tuple[int, Path]:
    """Run BindCraft on the selected GPU and return (exit_code, output_dir)."""
    paths = runtime_paths or RUNTIME_PATHS
    bindcraft_repository = paths.bindcraft_repo
    bindcraft_environment = paths.bindcraft_env
    launch = prepare_bindcraft_launch(
        output_dir=out_dir,
        repo=bindcraft_repository,
        environment_root=bindcraft_environment,
        target_id=target_id,
        target_pdb=target_pdb,
        hotspots=hotspots,
        chains=chains,
        default_lengths=lengths,
        config_delta=cand.config_delta,
    )
    if launch.advanced_overrides:
        print(
            f"  [worker:bindcraft] advanced overrides: " f"{launch.advanced_overrides}",
            flush=True,
        )
    command = list(launch.argv)
    worker_environment = os.environ.copy()
    worker_environment[
        "PATH"
    ] = f"{bindcraft_environment}/bin:" + worker_environment.get("PATH", "")
    worker_environment[
        "LD_LIBRARY_PATH"
    ] = f"{bindcraft_environment}/lib:" + worker_environment.get("LD_LIBRARY_PATH", "")
    worker_environment["CUDA_VISIBLE_DEVICES"] = gpu_id

    print(
        f"  [worker:bindcraft][gpu={gpu_id}] cmd[:200]: "
        f"{' '.join(shlex.quote(token) for token in command)[:200]}",
        flush=True,
    )
    return_code = subprocess.run(command, env=worker_environment).returncode
    return return_code, out_dir


def _exec_bindcraft_async(
    cand: ActionCandidate,
    out_dir: Path,
    target_id: str,
    target_pdb: str,
    hotspots: str,
    chains: str,
    lengths: tuple[int, int],
    gpu_id: str,
    runtime_paths: RuntimePaths | None = None,
) -> tuple["subprocess.Popen", Path]:
    """Launch BindCraft asynchronously and return the process handle and output directory."""
    paths = runtime_paths or RUNTIME_PATHS
    bindcraft_repository = paths.bindcraft_repo
    bindcraft_environment = paths.bindcraft_env
    launch = prepare_bindcraft_launch(
        output_dir=out_dir,
        repo=bindcraft_repository,
        environment_root=bindcraft_environment,
        target_id=target_id,
        target_pdb=target_pdb,
        hotspots=hotspots,
        chains=chains,
        default_lengths=lengths,
        config_delta=cand.config_delta,
    )
    command = list(launch.argv)
    # Isolate the backend CUDA libraries from cluster modules to avoid incompatible
    # library loading.
    worker_environment = _isolated_env_for_subprocess(bindcraft_environment)
    worker_environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    log_path = out_dir / "worker.log"
    log_fp = open(log_path, "w")
    print(f"  [worker:bindcraft][gpu={gpu_id}] launching → {out_dir.name}", flush=True)
    proc = subprocess.Popen(
        command,
        env=worker_environment,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        **_popen_kwargs(),
    )
    log_fp.close()
    return proc, out_dir


def _exec_complexa(
    cand: ActionCandidate,
    out_dir: Path,
    target_id: str,
    run_name: str,
    gpu_id: str = "1",
    runtime_paths: RuntimePaths | None = None,
) -> tuple[int, Path]:
    """Run Complexa in the configured environment and return (exit_code, output_dir).

    Return exit code 127 when the configured Python executable is unavailable.
    """
    paths = runtime_paths or RUNTIME_PATHS
    out_dir.mkdir(parents=True, exist_ok=True)

    # The configured Proteina-Complexa checkout owns this worker entry point.
    complexa_repository = paths.complexa_repo
    complexa_python = paths.complexa_python

    if not complexa_python.exists():
        print(
            f"  [SKIP] Complexa python missing: {complexa_python}",
            flush=True,
        )
        return 127, out_dir  # 127 = "command not found"
    overrides = build_complexa_overrides(
        family=cand.method_family,
        candidate_id=cand.candidate_id,
        config_delta=cand.config_delta,
        target_id=target_id,
        run_name=run_name,
        repo=complexa_repository,
        campaign_seed=RUN_SEED,
    )
    shell_command = build_complexa_shell_command(
        repo=complexa_repository,
        python=complexa_python,
        overrides=overrides,
    )

    print(
        f"  [worker:{cand.method_family}][gpu={gpu_id}] "
        f"bash_cmd[:200]: {shell_command[:200]}...",
        flush=True,
    )
    worker_environment = os.environ.copy()
    worker_environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    return_code = subprocess.run(
        ["bash", "-c", shell_command],
        env=worker_environment,
    ).returncode

    # Complexa writes to its repository's inference/<auto-named dir>/. Find it by
    # matching target_id + run_name substring (Complexa names vary).
    inference_directory = complexa_repository / "inference"
    matched_output_directories = (
        sorted(
            inference_directory.glob(f"*{target_id}*{run_name}*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if inference_directory.exists()
        else []
    )
    complexa_output_directory = (
        matched_output_directories[0]
        if matched_output_directories
        else inference_directory
        / f"search_binder_local_pipeline_{target_id}_{run_name}"
    )
    return return_code, complexa_output_directory


def _exec_complexa_async(
    cand: ActionCandidate,
    out_dir: Path,
    target_id: str,
    run_name: str,
    gpu_id: str,
    runtime_paths: RuntimePaths | None = None,
    output_namespace: str | None = None,
) -> tuple["subprocess.Popen | None", Path]:
    """Launch Complexa asynchronously with a candidate-specific output directory. Return
    (None, out_dir) when the executable or weights are unavailable.
    """
    paths = runtime_paths or RUNTIME_PATHS
    out_dir.mkdir(parents=True, exist_ok=True)
    complexa_repository = paths.complexa_repo
    complexa_python = paths.complexa_python
    if not complexa_python.exists():
        print(f"  [SKIP] Complexa python missing", flush=True)
        return None, out_dir
    physical_run_name = (
        f"{run_name}_a{output_namespace}" if output_namespace else run_name
    )
    overrides = build_complexa_overrides(
        family=cand.method_family,
        candidate_id=cand.candidate_id,
        config_delta=cand.config_delta,
        target_id=target_id,
        run_name=physical_run_name,
        repo=complexa_repository,
        campaign_seed=RUN_SEED,
        seed_run_name=run_name,
    )
    shell_command = build_complexa_shell_command(
        repo=complexa_repository,
        python=complexa_python,
        overrides=overrides,
    )
    worker_environment = os.environ.copy()
    worker_environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    log_path = out_dir / "worker.log"
    log_fp = open(log_path, "w")
    print(
        f"  [worker:{cand.method_family}][gpu={gpu_id}] launching → {physical_run_name}",
        flush=True,
    )
    proc = subprocess.Popen(
        ["bash", "-c", shell_command],
        env=worker_environment,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        **_popen_kwargs(),
    )
    log_fp.close()
    inference_directory = complexa_repository / "inference"
    complexa_output_directory = (
        inference_directory
        / f"search_binder_local_pipeline_{target_id}_{physical_run_name}"
    )
    return proc, complexa_output_directory


# Controller main loop
# -----------------------------------------------------------------------------


def _bindcraft_accepted_count(out_dir) -> int | None:
    """Number of ACCEPTED BindCraft designs so far (rows in
    designs/final_design_stats.csv, appended one per accepted design; or the
    count of Accepted/*.pdb). None if not found.

    This is the PRODUCTIVITY signal for BindCraft's adaptive timeout. BindCraft
    spins trajectory-after-trajectory toward `number_of_final_designs` accepted
    finals it may never reach on a hard target — so it is always CPU-busy yet
    yielding nothing. A CPU/file-liveness check can't tell "converging" from
    "failing"; the accepted-design count can. (out_dir is the per-launch
    worker_out dir, private to this launch.)
    """
    if out_dir is None:
        return None
    try:
        from pathlib import Path as _P

        d = _P(out_dir)
        csv = next(d.glob("**/final_design_stats.csv"), None)
        if csv is not None and csv.exists():
            with open(csv) as fh:
                n = sum(1 for _ in fh)
            return max(0, n - 1)  # minus header
        acc = next(d.glob("**/Accepted"), None)
        if acc is not None and acc.is_dir():
            return sum(1 for _ in acc.glob("*.pdb"))
    except Exception:  # noqa: BLE001
        return None
    return None


def _bindcraft_scoreable_final_count(out_dir) -> int | None:
    """Number of final BindCraft PDBs eligible for canonical AF2 scoring.

    BindCraft's native Accepted/Rejected split is not the T-REX strict gate.
    Both final directories contain complete binder candidates and live archives
    show canonical strict successes from both. Trajectory/MPNN intermediates are
    deliberately excluded because they are not final score-conversion inputs.
    """
    if out_dir is None:
        return None
    try:
        d = Path(out_dir)
        found_final_dir = False
        paths: set[str] = set()
        for dirname in ("Accepted", "Rejected"):
            final_dir = next(d.glob(f"**/{dirname}"), None)
            if final_dir is None or not final_dir.is_dir():
                continue
            found_final_dir = True
            paths.update(str(p.resolve()) for p in final_dir.glob("*.pdb"))
        if found_final_dir:
            return len(paths)
    except Exception:  # noqa: BLE001
        return None
    # Some BindCraft layouts write final_design_stats.csv before materializing
    # Accepted/. Use the accepted-row count when final structures are absent.
    return _bindcraft_accepted_count(out_dir)


def _csv_body_row_count(path: Path) -> int | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        with open(path) as fh:
            n = sum(1 for line in fh if line.strip())
        return max(0, n - 1)
    except Exception:  # noqa: BLE001
        return None


def _bindcraft_progress_count(out_dir) -> int | None:
    """Summarize BindCraft artifacts for provenance and diagnostics.

    Intermediate trajectory and MPNN rows do not establish scoreable output.
    The timeout watchdog uses accepted/final designs, not this signature.
    """
    if out_dir is None:
        return None
    try:
        d = Path(out_dir)
        counts: list[int] = []
        acc = _bindcraft_accepted_count(d)
        if acc is not None:
            counts.append(acc)
        for name in (
            "trajectory_stats.csv",
            "mpnn_design_stats.csv",
            "final_design_stats.csv",
        ):
            for csv in d.glob(f"**/{name}"):
                n = _csv_body_row_count(csv)
                if n is not None:
                    counts.append(n)
        for pattern in (
            "**/Trajectory/*.pdb",
            "**/Trajectory/Relaxed/*.pdb",
            "**/Trajectory/LowConfidence/*.pdb",
            "**/Trajectory/Clashing/*.pdb",
            "**/MPNN/*.pdb",
            "**/MPNN/Relaxed/*.pdb",
            "**/Accepted/*.pdb",
        ):
            n = sum(1 for _ in d.glob(pattern))
            if n > 0:
                counts.append(n)
        if not counts:
            return None
        return sum(max(0, int(x)) for x in counts)
    except Exception:  # noqa: BLE001
        return None


LENGTHS_PER_TARGET: dict[str, tuple[int, int]] = {
    "05_CD45": (80, 200),
    "23_BetV1": (70, 185),
    "30_SC2RBD": (80, 120),
    # Official Proteina-Complexa Table 4 target variants.
    "28_HER2_AAV": (60, 100),
    "26_CbAgo": (70, 160),
    "31_IL7RA": (50, 120),
    "32_PDL1_ALPHA_REPACK": (50, 120),
    "36_VEGFA": (50, 140),
    # Historical aliases kept for replay/offline analysis only.
    "27_HER2_AAV": (60, 100),
    "25_CbAgo": (70, 160),
    "02_PDL1": (64, 155),
}


def _high_cost_pending_families() -> set[str]:
    # Production default caps only BindCraft concentration. BindCraft has long
    # delayed feedback, so a single early hit must not fill all worker slots.
    # Set TREX_HIGH_COST_PENDING_FAMILIES="" for an explicit no-cap ablation.
    raw = os.environ.get("TREX_HIGH_COST_PENDING_FAMILIES", "bindcraft")
    return {f.strip() for f in raw.split(",") if f.strip()}


def _pending_family_load_summary(
    pool, pending: list[str], cand_by_id: dict[str, ActionCandidate]
) -> dict[str, Any]:
    """Family-level load from running + queued event-driven work.

    Completed-result health cannot see slow workers until they finish. This
    summary lets the LLM/selector know that a high-cost probe is already in
    flight, preventing repeated BindCraft/MCTS stacking before feedback lands.
    """
    now = time.time()
    by_family: dict[str, dict[str, Any]] = {}

    def row(fam: str) -> dict[str, Any]:
        return by_family.setdefault(
            fam,
            {"running": 0, "queued": 0, "pending_total": 0, "inflight_gpu_h": 0.0},
        )

    for slot in pool:
        if not getattr(slot, "busy", False) or slot.cand is None:
            continue
        fam = slot.cand.method_family
        r = row(fam)
        r["running"] += 1
        r["pending_total"] += 1
        if slot.launched_at > 0:
            r["inflight_gpu_h"] += max(0.0, (now - slot.launched_at) / 3600.0)
    active_ids = {
        slot.cand.candidate_id
        for slot in pool
        if getattr(slot, "busy", False) and slot.cand is not None
    }
    for cid in pending:
        if cid in active_ids:
            continue
        cand = cand_by_id.get(cid)
        if cand is None:
            continue
        fam = cand.method_family
        r = row(fam)
        r["queued"] += 1
        r["pending_total"] += 1
    for r in by_family.values():
        r["inflight_gpu_h"] = round(float(r.get("inflight_gpu_h", 0.0) or 0.0), 4)
    return {
        "by_family": dict(sorted(by_family.items())),
        "high_cost_families": sorted(_high_cost_pending_families()),
    }


def _pending_load_cache_key(load: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    rows = load.get("by_family", {}) if isinstance(load, dict) else {}
    out = []
    for fam, row in sorted(rows.items()):
        if not isinstance(row, dict):
            continue
        out.append(
            (fam, int(row.get("running", 0) or 0), int(row.get("queued", 0) or 0))
        )
    return tuple(out)


def _latest_evidence_for_dispatch(archive) -> EvidenceSummary | None:
    try:
        evs = list(archive.iter_records(EvidenceSummary))
        return evs[-1] if evs else None
    except Exception:  # noqa: BLE001
        return None


def _mh_get(h: Any, key: str, default: Any = 0) -> Any:
    if h is None:
        return default
    if isinstance(h, dict):
        return h.get(key, default)
    return getattr(h, key, default)


def _repeated_support_dispatch_cap(
    evidence: EvidenceSummary | None, family: str
) -> int | None:
    if evidence is None:
        return None
    er = getattr(evidence, "execution_realization", {}) or {}
    by_family = er.get("by_family", {}) if isinstance(er, dict) else {}
    row = by_family.get(family, {}) if isinstance(by_family, dict) else {}
    if not isinstance(row, dict):
        return None
    try:
        proposed = int(row.get("proposed", 0) or 0)
        started = int(row.get("started", 0) or 0)
        deferred = int(row.get("dispatch_deferred", 0) or 0)
        selected_not_started = int(row.get("selected_not_started", 0) or 0)
    except (TypeError, ValueError):
        return None
    min_proposed = _bounded_int_env(
        "TREX_HIGH_COST_REPEATED_SUPPORT_MIN_PROPOSED", 6, min_value=1
    )
    if proposed < min_proposed:
        return None
    if deferred <= 0 and selected_not_started <= 0:
        return None
    max_started_fraction = float(
        os.environ.get("TREX_HIGH_COST_REPEATED_SUPPORT_MAX_STARTED_FRACTION", "0.35")
    )
    if started / max(1, proposed) > max_started_fraction:
        return None
    state = str(getattr(evidence, "state_label", "") or "")
    run_su = int(getattr(evidence, "run_su_count", 0) or 0)
    try:
        dry_gpu_h = float(getattr(evidence, "gpu_h_since_last_su", 0.0) or 0.0)
    except (TypeError, ValueError):
        dry_gpu_h = 0.0
    productive_duplicate_dry_collapse = (
        state == "productive_duplicate"
        and bool(getattr(evidence, "strict_duplicate_collapse_signal", False))
        and dry_gpu_h >= 1.0
    )
    if (
        state
        not in {"low_evidence", "stalled", "deep_stall", "strict_duplicate_collapse"}
        and run_su >= 4
        and not productive_duplicate_dry_collapse
    ):
        return None
    mh = (getattr(evidence, "method_health", {}) or {}).get(family)
    gpu_h = float(_mh_get(mh, "cumulative_gpu_h", 0.0) or 0.0)
    timeouts = int(_mh_get(mh, "timeout_count", _mh_get(mh, "timeouts", 0)) or 0)
    strict = int(_mh_get(mh, "strict_yield_su", 0) or 0)
    chained = int(_mh_get(mh, "chained_strict_yield_su", 0) or 0)
    near = int(_mh_get(mh, "near_miss_yield", 0) or 0)
    recent_near = int(_mh_get(mh, "near_miss_yield_recent", 0) or 0)
    max_negative_gpu_h = float(
        os.environ.get("TREX_HIGH_COST_REPEATED_SUPPORT_MAX_NEGATIVE_GPU_H", "6.0")
    )
    max_timeouts = _bounded_int_env(
        "TREX_HIGH_COST_REPEATED_SUPPORT_MAX_TIMEOUTS", 1, min_value=0
    )
    if (
        gpu_h >= max_negative_gpu_h
        and timeouts > max_timeouts
        and strict == 0
        and chained == 0
        and near == 0
        and recent_near == 0
    ):
        return None
    return _bounded_int_env(
        "TREX_HIGH_COST_INFLIGHT_CAP_REPEATED_SUPPORT", 2, min_value=1
    )


def _high_cost_dispatch_cap(evidence: EvidenceSummary | None, family: str) -> int:
    if family not in _high_cost_pending_families():
        return 10**9
    if evidence is None:
        return _bounded_int_env("TREX_HIGH_COST_INFLIGHT_CAP", 1, min_value=1)

    def _float_env(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    from .selector import high_cost_cap_for_evidence

    cap, _source = high_cost_cap_for_evidence(
        evidence,
        family,
        high_cost_pending_cap=_bounded_int_env(
            "TREX_HIGH_COST_INFLIGHT_CAP", 1, min_value=1
        ),
        high_cost_pending_promoted_cap=_bounded_int_env(
            "TREX_HIGH_COST_INFLIGHT_CAP_PROMOTED", 2, min_value=1
        ),
        high_cost_pending_deep_stall_cap=_bounded_int_env(
            "TREX_HIGH_COST_INFLIGHT_CAP_DEEP_STALL", 1, min_value=1
        ),
        high_cost_pending_dry_pivot_cap=_bounded_int_env(
            "TREX_HIGH_COST_INFLIGHT_CAP_DRY_PIVOT", 2, min_value=1
        ),
        high_cost_pending_strong_cap=_bounded_int_env(
            "TREX_HIGH_COST_INFLIGHT_CAP_STRONG", 2, min_value=1, max_value=3
        ),
        high_cost_dry_pivot_min_gpu_h=_float_env(
            "TREX_HIGH_COST_DRY_PIVOT_MIN_GPU_H", 0.75
        ),
        high_cost_dry_pivot_min_completed_children=_bounded_int_env(
            "TREX_HIGH_COST_DRY_PIVOT_MIN_COMPLETED", 32, min_value=1
        ),
        high_cost_dry_pivot_max_best_recent_su_per_gpu_h=_float_env(
            "TREX_HIGH_COST_DRY_PIVOT_MAX_BEST_RECENT_SU_PER_GPU_H", 0.25
        ),
        high_cost_strong_min_su=_bounded_int_env(
            "TREX_HIGH_COST_STRONG_MIN_SU", 2, min_value=1
        ),
        high_cost_strong_min_gpu_h=_float_env("TREX_HIGH_COST_STRONG_MIN_GPU_H", 1.0),
        high_cost_strong_min_recent_su_per_gpu_h=_float_env(
            "TREX_HIGH_COST_STRONG_MIN_RECENT_SU_PER_GPU_H", 0.50
        ),
        high_cost_strong_best_fraction=_float_env(
            "TREX_HIGH_COST_STRONG_BEST_FRACTION", 0.80
        ),
        high_cost_stale_recent_gpu_h=_float_env(
            "TREX_HIGH_COST_STALE_RECENT_GPU_H", 1.0
        ),
        min_su_per_gpu_h=_float_env("TREX_COST_AWARE_MIN_SU_PER_GPU_H", 0.25),
        high_cost_repeated_support_min_proposed=_bounded_int_env(
            "TREX_HIGH_COST_REPEATED_SUPPORT_MIN_PROPOSED", 6, min_value=1
        ),
        high_cost_repeated_support_max_started_fraction=_float_env(
            "TREX_HIGH_COST_REPEATED_SUPPORT_MAX_STARTED_FRACTION", 0.35
        ),
        high_cost_repeated_support_cap=_bounded_int_env(
            "TREX_HIGH_COST_INFLIGHT_CAP_REPEATED_SUPPORT", 2, min_value=1
        ),
        high_cost_repeated_support_max_negative_gpu_h=_float_env(
            "TREX_HIGH_COST_REPEATED_SUPPORT_MAX_NEGATIVE_GPU_H", 6.0
        ),
        high_cost_repeated_support_max_timeouts=_bounded_int_env(
            "TREX_HIGH_COST_REPEATED_SUPPORT_MAX_TIMEOUTS", 1, min_value=0
        ),
    )
    return cap


def _high_cost_dispatch_cap_for_pool(
    evidence: EvidenceSummary | None,
    family: str,
    pool,
) -> int:
    cap = _high_cost_dispatch_cap(evidence, family)
    if family not in _high_cost_pending_families():
        return cap
    pool_n = len(pool or [])
    if pool_n <= 1:
        return cap
    reserve = _bounded_int_env(
        "TREX_HIGH_COST_RESERVE_NON_HIGH_COST_SLOTS",
        1,
        min_value=0,
    )
    if reserve <= 0:
        return cap
    return max(1, min(cap, pool_n - reserve))


def _high_cost_dispatch_defer_reason(
    pool, cand: ActionCandidate, archive
) -> str | None:
    fam = str(getattr(cand, "method_family", "") or "")
    if not fam or archive is None or fam not in _high_cost_pending_families():
        return None
    evidence = _latest_evidence_for_dispatch(archive)
    cap = _high_cost_dispatch_cap_for_pool(evidence, fam, pool)
    active = sum(
        1
        for slot in pool
        if getattr(slot, "busy", False)
        and slot.cand is not None
        and slot.cand.method_family == fam
    )
    if active >= cap:
        state = (
            str(getattr(evidence, "state_label", "unknown") or "unknown")
            if evidence
            else "no_evidence"
        )
        return f"high_cost_inflight_cap:{fam}:active={active} cap={cap} state={state}"
    return None


def _dispatch_candidate_to_gpu(
    cand: "ActionCandidate",
    *,
    gpu_id: str,
    archive: "Archive",
    target: "TargetConstraint",
    target_pdb: str,
    round_id: int,
    archive_root: Path,
    runtime_paths: RuntimePaths | None = None,
) -> tuple[object, Path, str, str, str, str] | None:
    """Family-aware worker dispatch for one candidate.

    Returns (proc, out_dir, parent_pdb_str, parent_result_id, target_chains_csv, binder_chain) on
    successful Popen, or None when the family is unknown / parent PDB
    missing / executor unavailable. Identical dispatch tree as the
    previous batch loop — extracted so the event-driven main loop
    can call it per slot.
    """
    paths = runtime_paths or RUNTIME_PATHS
    worker_out = archive_root / (
        f"worker_outputs/r{round_id:03d}_{cand.method_family}_{cand.candidate_id}"
    )
    lengths = LENGTHS_PER_TARGET.get(target.target_id, (80, 150))
    parent_pdb_str = ""
    parent_result_id = ""
    target_chains_csv = _csv_chains(tuple(target.chain_ids))
    binder_chain = ""

    if cand.method_family == "bindcraft":
        proc, out_dir = _exec_bindcraft_async(
            cand,
            worker_out,
            target.target_id,
            target_pdb,
            ",".join(target.hotspots),
            _csv_chains(tuple(target.chain_ids)),
            lengths,
            gpu_id=gpu_id,
            runtime_paths=paths,
        )
    elif cand.method_family.startswith("complexa_"):
        cid_safe = cand.candidate_id.replace("/", "_")
        run_name = f"v7_r{round_id:03d}_{cid_safe}"
        namespace_material = (
            f"{Path(archive_root).resolve()}|{os.getpid()}|{time.time_ns()}"
        )
        output_namespace = hashlib.sha256(namespace_material.encode()).hexdigest()[:12]
        proc, out_dir = _exec_complexa_async(
            cand,
            worker_out,
            target.target_id,
            run_name,
            gpu_id=gpu_id,
            runtime_paths=paths,
            output_namespace=output_namespace,
        )
        if proc is None:
            print(f"  [SKIP] {cand.method_family}", flush=True)
            return None
    elif cand.method_family == "structure_refilter":
        parent = _resolve_parent_artifact(archive, cand)
        if parent is None:
            print(f"  [SKIP] structure_refilter: no parent PDB found", flush=True)
            return None
        parent_pdb, parent_result_id = parent
        parent_pdb_str = str(parent_pdb)
        tgt_ch, bnd_ch = _resolve_pdb_chains(
            parent_pdb,
            target_chain_ids=tuple(target.chain_ids),
            target_pdb=Path(target_pdb),
        )
        target_chains_csv, binder_chain = tgt_ch, bnd_ch
        if (tgt_ch, bnd_ch) != ("A", "B"):
            print(
                f"  [af2_refilter] non-default chains for "
                f"{parent_pdb.name}: target={tgt_ch} binder={bnd_ch}",
                flush=True,
            )
        parent_rec = _parent_result_by_id(archive, parent_result_id)
        if parent_rec is not None and parent_rec.backend_family == "boltzgen":
            bad_res = _noncanonical_residue_names(parent_pdb, chain_id=bnd_ch)
            if bad_res:
                reason = (
                    "structure_refilter permanent skip: boltzgen parent has "
                    f"noncanonical binder residues {sorted(bad_res)} in chain {bnd_ch}"
                )
                print(f"  [SKIP] {reason}", flush=True)
                raise PermanentDispatchSkip(reason)
        proc, out_dir = _exec_af2_refilter_async(
            cand,
            worker_out,
            parent_pdb,
            target_chain=tgt_ch,
            binder_chain=bnd_ch,
            gpu_id=gpu_id,
            runtime_paths=paths,
        )
        if proc is None:
            return None
    elif cand.method_family == "proteinmpnn_redesign":
        parent = _resolve_parent_artifact(archive, cand)
        if parent is None:
            print(f"  [SKIP] proteinmpnn: no parent PDB found", flush=True)
            return None
        parent_pdb, parent_result_id = parent
        parent_pdb_str = str(parent_pdb)
        tgt_ch, bnd_ch = _resolve_pdb_chains(
            parent_pdb,
            target_chain_ids=tuple(target.chain_ids),
            target_pdb=Path(target_pdb),
        )
        target_chains_csv, binder_chain = tgt_ch, bnd_ch
        proc, out_dir = _exec_proteinmpnn_async(
            cand,
            worker_out,
            parent_pdb,
            gpu_id=gpu_id,
            binder_chain=bnd_ch,
            runtime_paths=paths,
        )
        if proc is None:
            return None
    elif cand.method_family == "boltzgen":
        binder_chain = _first_free_chain_id(tuple(target.chain_ids), default="B")
        proc, out_dir = _exec_boltzgen_async(
            cand,
            worker_out,
            target.target_id,
            target_pdb,
            list(target.hotspots),
            list(target.chain_ids),
            lengths,
            gpu_id=gpu_id,
            runtime_paths=paths,
        )
        if proc is None:
            return None
    else:
        adapter = get_backend_adapter(cand.method_family)
        if adapter is None:
            print(f"  [SKIP] family {cand.method_family} has no executor", flush=True)
            return None
        parent_pdb: Path | None = None
        if adapter.capability.requires_parent_pdb:
            parent = _resolve_parent_artifact(archive, cand)
            if parent is None:
                print(
                    f"  [SKIP] {cand.method_family}: no parent PDB found",
                    flush=True,
                )
                return None
            parent_pdb, parent_result_id = parent
            parent_pdb_str = str(parent_pdb)
            target_chains_csv, binder_chain = _resolve_pdb_chains(
                parent_pdb,
                target_chain_ids=tuple(target.chain_ids),
                target_pdb=Path(target_pdb),
            )
        else:
            binder_chain = _first_free_chain_id(tuple(target.chain_ids), default="B")
        context = BackendLaunchContext(
            candidate=cand,
            output_root=worker_out,
            target=target,
            target_pdb=Path(target_pdb),
            gpu_id=gpu_id,
            length_range=lengths,
            parent_pdb=parent_pdb,
            parent_result_id=parent_result_id,
            target_chains_csv=target_chains_csv,
            binder_chain=binder_chain,
        )
        command = validate_backend_command(adapter.build_command(context), worker_out)
        command.output_dir.mkdir(parents=True, exist_ok=True)
        worker_out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(command.env)
        # The operator adapter may select program options, but the controller
        # remains authoritative for worker-slot GPU isolation.
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        log_path = worker_out / "worker.log"
        log_fp = log_path.open("w")
        try:
            proc = subprocess.Popen(
                list(command.argv),
                cwd=str(command.cwd) if command.cwd is not None else None,
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                **_popen_kwargs(),
            )
        finally:
            log_fp.close()
        out_dir = command.output_dir
        print(
            f"  [worker:{cand.method_family}][gpu={gpu_id}] launching extension",
            flush=True,
        )
    return (
        proc,
        out_dir,
        parent_pdb_str,
        parent_result_id,
        target_chains_csv,
        binder_chain,
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _complexa_native_refilter_score(r: ResultRecord) -> float | None:
    """Rank Complexa parents for canonical AF2 score-conversion.

    Legacy helper for ranking old Complexa diagnostic archives. New T-REX
    Complexa launches are direct-scored and should not enter score conversion.
    The score is ranking-only for stale chain candidates or offline replay.
    """
    if not r.backend_family.startswith("complexa_"):
        return None
    m = r.metrics or {}
    score = 0.0
    seen = False

    plddt = _float_or_none(m.get("complexa_native_pLDDT"))
    ipae = _float_or_none(m.get("complexa_native_iPAE"))
    rmsd = _float_or_none(m.get("complexa_native_binder_scRMSD"))
    if plddt is not None:
        score += plddt / 100.0
        seen = True
    if ipae is not None:
        score += max(0.0, (0.50 - ipae) / 0.50)
        seen = True
    if rmsd is not None:
        score += max(0.0, (4.0 - rmsd) / 4.0)
        seen = True
    if (
        plddt is not None
        and plddt >= 90.0
        and ipae is not None
        and ipae <= (7.0 / 31.0)
        and rmsd is not None
        and rmsd < 1.5
    ):
        score += 10.0

    # Use auxiliary diagnostics only as tiebreakers so their different scales cannot
    # overwhelm the qualification-based ranking.
    _ADDON_W = 0.25
    for key in (
        "ipTM",
        "avg_ipsae",
        "max_ipsae",
        "contact_density",
        "interface_contact_density",
    ):
        value = _float_or_none(m.get(key))
        if value is not None:
            score += _ADDON_W * value
            seen = True
    min_ipae = _float_or_none(m.get("min_ipae"))
    if min_ipae is not None:
        score += _ADDON_W * max(0.0, (0.50 - min_ipae) / 0.50)
        seen = True

    return score if seen else None


def _route_value_score_modifier(family: str, evidence: EvidenceSummary | None) -> float:
    if evidence is None:
        return 0.0
    best = 0.0
    for row in getattr(evidence, "route_values", None) or []:
        get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
        if get("scope") != "family":
            continue
        fam = get("action_family") or get("family")
        if fam != family:
            continue
        status = str(get("status", ""))
        rate = _route_row_current_rate(row)
        mod = min(rate, 5.0) * 0.20
        if status == "promote":
            mod += 1.0
        elif status == "diversify":
            mod -= 0.5
        elif status in {"defer", "collapse_risk"}:
            mod -= 2.0
        best = max(best, mod) if mod >= 0 else min(best, mod)
    return best


def _boltzgen_refilter_proxy_score(r: ResultRecord) -> float | None:
    bins = r.bins or {}
    seen = False
    score = 0.0
    weights = {
        "boltzgen_design_iptm": 2.0,
        "boltzgen_design_to_target_iptm": 1.5,
        "boltzgen_design_iiptm": 1.2,
        "boltzgen_structure_confidence": 1.0,
    }
    for key, weight in weights.items():
        value = _float_or_none(bins.get(key))
        if value is not None:
            score += weight * value
            seen = True
    min_pae = _float_or_none(bins.get("boltzgen_min_design_to_target_pae"))
    if min_pae is not None:
        score += 1.5 * max(0.0, (12.0 - min_pae) / 12.0)
        seen = True
    rmsd_refolded = _float_or_none(bins.get("boltzgen_native_rmsd_refolded"))
    if rmsd_refolded is not None:
        score += max(0.0, (3.0 - rmsd_refolded) / 3.0)
        seen = True
    if _proxy_promising_pending_score_conversion(r):
        score += 2.0
        seen = True
    return score if seen else None


def _proteinmpnn_refilter_proxy_score(r: ResultRecord) -> float | None:
    bins = r.bins or {}
    native_score = _float_or_none(bins.get("mpnn_global_score"))
    if native_score is None:
        native_score = _float_or_none(bins.get("mpnn_score"))
    seq_recovery = _float_or_none(bins.get("mpnn_seq_recovery"))
    if native_score is None and seq_recovery is None:
        return None
    score = 0.0
    if native_score is not None:
        # ProteinMPNN score is per-residue NLL; lower is better. Keep positive
        # scale for sorting and cap very poor sequences at zero.
        score += max(0.0, 2.5 - native_score)
    if seq_recovery is not None:
        # Mid-range recovery tends to preserve a usable scaffold while still
        # changing sequence enough to escape duplicate SU bins.
        score += max(0.0, 1.0 - abs(seq_recovery - 0.50) * 2.0) * 0.25
    if _proxy_promising_pending_score_conversion(r):
        score += 1.0
    return score


def _diagnostic_refilter_score(
    r: ResultRecord, evidence: EvidenceSummary | None = None
) -> float:
    """Higher-is-better parent ranking for diagnostic to canonical refilter."""
    route_modifier = _route_value_score_modifier(r.backend_family, evidence)

    if r.backend_family == "boltzgen":
        score = _boltzgen_refilter_proxy_score(r)
        if score is not None:
            return score + route_modifier
    if r.backend_family == "proteinmpnn_redesign":
        score = _proteinmpnn_refilter_proxy_score(r)
        if score is not None:
            return score + route_modifier

    bins = r.bins or {}
    for key in ("bindcraft_rank_iptm",):
        value = _float_or_none(bins.get(key))
        if value is not None:
            return value + route_modifier

    complexa_score = _complexa_native_refilter_score(r)
    if complexa_score is not None:
        return complexa_score + route_modifier

    return route_modifier


def _route_key_for_launch_candidate(
    archive: Archive,
    family: str,
    source_candidate: ActionCandidate,
) -> str:
    """Exact route key for a just-completed diagnostic launch.

    Must match evidence_reducer.build_route_values() and _chain_source_context():
    parent-bound actions such as Complexa->ProteinMPNN are learned as
    root-route -> action-route, not as a context-free ProteinMPNN route.
    """
    from .evidence_reducer import resolve_generating_record, route_component_key

    action_op = source_candidate.operator_id or f"{family}_default"
    action_cfg = dict(source_candidate.config_delta or {})
    action_comp = route_component_key(family, action_op, action_cfg)
    root_comp = action_comp

    parent_id = source_candidate.parent_result_id
    if parent_id:
        results = list(archive.iter_records(ResultRecord))
        by_result_id = {r.result_id: r for r in results}
        parent = by_result_id.get(parent_id)
        if parent is not None:
            actions = list(archive.iter_records(ActionCandidate))
            spawning_actions = _spawn_index_for_results(actions, results)
            root = resolve_generating_record(
                parent,
                by_result_id=by_result_id,
                spawning_actions=spawning_actions,
            )
            root_action = spawning_actions.get(root.result_id)
            root_family = root.backend_family
            root_op = (
                root_action.operator_id
                if root_action is not None
                else f"{root_family}_default"
            )
            root_cfg = dict(
                (root_action.config_delta or {}) if root_action is not None else {}
            )
            root_comp = route_component_key(root_family, root_op, root_cfg)

    return (
        f"route::{action_comp}"
        if root_comp == action_comp
        else f"route::{root_comp}->{action_comp}"
    )


def _exact_route_signal_for_launch(
    archive: Archive,
    family: str,
    source_candidate: ActionCandidate | None,
) -> dict[str, Any]:
    """Latest exact route/config value for the diagnostic launch being parsed.

    Family rows are too coarse here: if one complexa_fk_steering config is
    productive, another nearby config should not automatically receive the
    high score-conversion cap. Exact route rows preserve operator/config, and
    parent-bound routes preserve root context.
    """
    if source_candidate is None:
        return {}
    try:
        strategy_key = _route_key_for_launch_candidate(
            archive, family, source_candidate
        )
        evs = list(archive.iter_records(EvidenceSummary))[-6:]
    except Exception:  # noqa: BLE001
        return {}
    for ev in reversed(evs):
        for row in getattr(ev, "route_values", None) or []:
            get = (
                row.get
                if isinstance(row, dict)
                else lambda k, d=None: getattr(row, k, d)
            )
            if get("scope") != "route":
                continue
            if str(get("strategy_key", "") or "") != strategy_key:
                continue
            return {
                "strategy_key": strategy_key,
                "status": str(get("status", "") or ""),
                "record_recent_new_su": int(
                    get("record_recent_new_su", get("new_su_recent", 0)) or 0
                ),
                "record_recent_rate": _route_row_current_rate(row),
                "rate": float(get("new_su_per_route_gpu_h", 0.0) or 0.0),
            }
    return {"strategy_key": strategy_key}


def _bounded_int_env(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Parse an integer env override without letting bad env crash the controller."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = default
    if min_value is not None:
        value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


# Compatibility aliases for tests and downstream code that imported the former
# controller-private helpers. New code should use trex.execution directly.
_score_conversion_identity = score_conversion_parent_identity
_dedupe_score_conversion_parents = deduplicate_scored_parent_results
_record_has_canonical_strict_axes = canonical_strict_metrics_present
_record_needs_score_conversion = result_requires_score_conversion
_records_need_score_conversion = results_require_score_conversion
_is_canonical_score_conversion_record = is_canonical_score_conversion_result


def _auto_chain_cap_for_diagnostic_launch(
    archive: Archive,
    family: str,
    scored_records: list[ResultRecord],
    source_candidate: ActionCandidate | None = None,
) -> int:
    """Compatibility wrapper for the former archive-aware controller helper."""

    del archive, source_candidate
    return score_conversion_candidate_limit(family, scored_records)


def _select_score_conversion_parents(
    scored: list[tuple[float, ResultRecord]],
    cap: int,
) -> list[tuple[float, ResultRecord]]:
    """Compatibility wrapper using the new descriptive scheduling API."""

    return select_score_conversion_parents(scored, cap)


def _chain_fair_probe_per_route() -> int:
    """Max canonical-score probes for an unproven route before evidence updates.

    This is deliberately route/config scoped rather than target or family prior:
    every observed diagnostic route gets a small official-AF2 scoring probe, but
    an early-returning route cannot monopolize freed GPUs unless the evidence
    already says that route is buying new SU or has native-strict-like parents.
    """
    try:
        return max(
            1, int(os.environ.get("TREX_CHAIN_REFILTER_FAIR_PROBE_PER_ROUTE", "4"))
        )
    except ValueError:
        return 4


def _chain_weak_family_probe_cap() -> int:
    """Max weak canonical AF2 conversions per unproven diagnostic family.

    Per-route fair probes are not enough when one diagnostic generator creates
    many exact route/config groups. Keep native-like, proxy-promising, or
    promoted parents uncapped; this only bounds low-value weak-tail backlog after
    enough failed score-conversion feedback exists.
    """
    try:
        return max(
            1, int(os.environ.get("TREX_CHAIN_REFILTER_WEAK_FAMILY_PROBE_CAP", "24"))
        )
    except ValueError:
        return 24


def _chain_family_score_conversion_tick_cap() -> int:
    """Per-tick cap for weak score-conversion from one diagnostic family.

    Route-level fair probes alone are not enough: a single generator can create
    many exact route/config groups and thereby consume the whole reserve even
    when every group is low value. Keep this cap target/family-agnostic; route
    evidence can still bypass it through the high-value cap below.
    """
    try:
        return max(
            1, int(os.environ.get("TREX_CHAIN_REFILTER_FAMILY_PER_TICK_CAP", "4"))
        )
    except ValueError:
        return 4


def _chain_family_score_conversion_high_value_tick_cap() -> int:
    """Per-tick cap for evidence-backed score-conversion from one family."""
    try:
        return max(
            1,
            int(
                os.environ.get(
                    "TREX_CHAIN_REFILTER_FAMILY_PER_TICK_HIGH_VALUE_CAP", "16"
                )
            ),
        )
    except ValueError:
        return 16


def _chain_route_tranches() -> list[int]:
    raw = os.environ.get("TREX_CHAIN_REFILTER_ROUTE_TRANCHES", "4,8,16,32,64")
    vals: list[int] = []
    for part in raw.split(","):
        try:
            val = int(part.strip())
        except ValueError:
            continue
        if val > 0:
            vals.append(val)
    vals = sorted(set(vals))
    return vals or [4, 8, 16, 32, 64]


def _chain_route_tranche_cap(
    *,
    fair_cap: int,
    route_row: Any,
    prior_route_row: Any,
) -> int:
    """Authorize at most one completed-evidence tranche for an exact route.

    The live dispatch count is applied by the caller against the immutable
    EvidenceSummary authorization returned here.  Keeping those states separate
    is what makes the configured tranche ladder real: repeated one-slot reserve
    calls cannot advance again until a new evidence snapshot reports that the
    current tranche completed with useful canonical signal.  Native/proxy
    diagnostics still rank the first four, but they do not bypass the ladder.
    """
    base = max(1, int(fair_cap))
    tranches = sorted({base, *[v for v in _chain_route_tranches() if v > base]})
    max_configured_tranche = tranches[-1]
    observed = int(
        _route_row_value(route_row, "canonical_score_conversion_count", 0) or 0
    )
    if observed < base:
        return base

    # A snapshot taken part-way through an already-authorized tranche may finish
    # that tranche, but cannot unlock the following one.
    if observed not in tranches:
        for tranche in tranches:
            if observed < tranche:
                return tranche
        elastic = max_configured_tranche
        while observed > elastic:
            elastic *= 2
        if observed < elastic:
            return elastic

    current_idx = tranches.index(observed) if observed in tranches else None

    previous_observed = int(
        _route_row_value(prior_route_row, "canonical_score_conversion_count", 0) or 0
    )
    if previous_observed >= observed:
        return observed

    def _increased(name: str, *, epsilon: float = 0.0) -> bool:
        current = _route_row_value(route_row, name, 0) or 0
        previous = _route_row_value(prior_route_row, name, 0) or 0
        try:
            return float(current) > float(previous) + epsilon
        except (TypeError, ValueError):
            return False

    # Lexicographic evidence: SU first, then deduped canonical strict/near
    # structural signal, then quality/diagnostic progress for genuinely hard
    # targets. Raw strict_count is intentionally excluded: duplicate stricts
    # should not turn the 4->8->16->32 ladder into an unbounded scoring drain.
    useful_signal = (
        _increased("new_su")
        or _increased("near_miss_count")
        or _increased("strict_quality_n_unique_bins")
        or _increased("strict_quality_p25", epsilon=1e-6)
        or _increased("strict_quality_median", epsilon=1e-6)
        or _increased("diagnostic_improvement_score", epsilon=1e-6)
    )
    if not useful_signal:
        return observed

    if current_idx is not None and current_idx < len(tranches) - 1:
        return tranches[current_idx + 1]

    # After the configured ladder, keep productive routes alive with an elastic
    # doubling tail. This preserves bounded evidence-gated scoring for dry routes
    # while avoiding a hard cap on routes still minting new SU or quality gains.
    return max(observed + 1, observed * 2)


def _is_recoverable_dispatch_failure(d: DispatchRecord) -> bool:
    # Temporary dispatch_failed rows should be retried after capacity or state
    # changes. Other dispatch_failed rows remain terminal.
    if d.status != "dispatch_failed":
        return False
    why = d.why or ""
    dispatch_id = d.dispatch_id or ""
    return (
        why.startswith("high_cost_inflight_cap")
        or dispatch_id.startswith("deep_stall_throttle_")
        or "current tick state=deep_stall" in why
    )


def _fallback_chain_source_family(candidate_id: str) -> str:
    """Infer a source family from an archived or synthetic chain identifier.

    Live scheduling uses the parent ResultRecord. This fallback preserves
    underscores within the family name and returns ``_unknown`` for an
    unrecognized identifier.
    """
    if not candidate_id.startswith("chain_") or "_to_" not in candidate_id:
        return "_unknown"
    try:
        left = candidate_id[len("chain_") :].split("_to_", 1)[0]
        return left.split("_", 1)[1] if "_" in left else "_unknown"
    except Exception:  # noqa: BLE001
        return "_unknown"


def _spawn_index_for_results(
    actions: list[ActionCandidate],
    results: list[ResultRecord],
) -> dict[str, ActionCandidate]:
    by_cid = {a.candidate_id: a for a in actions}
    out: dict[str, ActionCandidate] = {}
    for r in results:
        for pid in r.parent_ids or []:
            ac = by_cid.get(pid)
            if ac is not None:
                out[r.result_id] = ac
                break
    return out


def _route_value_index(
    evidence: EvidenceSummary | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_route: dict[str, Any] = {}
    by_family: dict[str, Any] = {}
    if evidence is None:
        return by_route, by_family
    for row in getattr(evidence, "route_values", None) or []:
        get = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)
        scope = str(get("scope", "") or "")
        key = str(get("strategy_key", "") or "")
        fam = str(get("action_family", "") or get("family", "") or "")
        if scope == "route" and key:
            by_route[key] = row
        elif scope == "family" and fam:
            by_family[fam] = row
    return by_route, by_family


def _route_row_value(row: Any, key: str, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _route_row_float(row: Any, key: str, default: Any = None) -> float | None:
    value = _route_row_value(row, key, default)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _route_row_int(row: Any, key: str, default: Any = 0) -> int:
    value = _route_row_value(row, key, default)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _route_row_allows_record_recent(row: Any) -> bool:
    """Whether record-count recent fields are safe for route decisions.

    Tiny canonical AF2 score-conversion rows can make a delayed diagnostic route
    look recently productive while omitting the generator GPU-h that bought the
    artifacts. Use record windows only for direct-scored routes; score-conversion
    routes must rely on GPU/medium windows or lifetime as explicitly labelled.
    """
    route_role = str(_route_row_value(row, "route_role", "") or "")
    canonical_refilter_gpu_h = (
        _route_row_float(row, "canonical_refilter_gpu_h", 0.0) or 0.0
    )
    return canonical_refilter_gpu_h <= 0.0 and "score_conversion" not in route_role


def _route_row_recent_su(row: Any) -> int:
    recent = max(
        _route_row_int(row, "new_su_recent_gpu", 0),
        _route_row_int(row, "medium_recent_new_su", 0),
    )
    if _route_row_allows_record_recent(row):
        recent = max(
            recent,
            _route_row_int(
                row, "record_recent_new_su", _route_row_value(row, "new_su_recent", 0)
            ),
        )
    return recent


def _route_row_current_rate(row: Any) -> float:
    """Decision-safe route value for launch/allocation paths.

    Prefer the short worker-GPU-hour recent window, then the medium GPU-hour
    window. The record window is only a fallback for direct-scored routes;
    delayed AF2 score-conversion routes can otherwise contain only tiny
    refilter records while omitting the generator GPU-h.
    """
    value = _route_row_float(row, "gpu_recent_new_su_per_route_gpu_h", None)
    if value is not None:
        return max(0.0, value)

    value = _route_row_float(row, "medium_recent_new_su_per_route_gpu_h", None)
    if value is not None:
        return max(0.0, value)

    record_value = _route_row_float(
        row,
        "record_recent_new_su_per_route_gpu_h",
        _route_row_value(row, "recent_new_su_per_route_gpu_h", None),
    )
    if record_value is not None and _route_row_allows_record_recent(row):
        return max(0.0, record_value)

    lifetime = _route_row_float(row, "new_su_per_route_gpu_h", None)
    return max(0.0, lifetime or 0.0)


def _chain_source_context(
    c: ActionCandidate,
    *,
    by_result_id: dict[str, ResultRecord],
    spawning_actions: dict[str, ActionCandidate],
) -> dict[str, Any]:
    """Resolve the upstream family and exact route for a chain refilter.

    For direct diagnostic generators this returns their family/operator/config
    route. For parent-bound redesigns such as proteinmpnn_redesign, it preserves
    the root generator context in the route key, matching build_route_values().
    """
    from .evidence_reducer import route_component_key

    parent = by_result_id.get(c.parent_result_id or "")
    if parent is None:
        fam = _fallback_chain_source_family(c.candidate_id)
        group = f"legacy::{c.candidate_id}" if fam == "_unknown" else f"family::{fam}"
        return {
            "source_family": fam,
            "route_key": f"family::{fam}",
            "route_group": group,
            "parent": None,
            "score": 0.0,
            "native_like": False,
        }

    action = spawning_actions.get(parent.result_id)
    action_family = parent.backend_family
    action_op = action.operator_id if action is not None else f"{action_family}_default"
    action_cfg = dict((action.config_delta or {}) if action is not None else {})
    action_comp = route_component_key(action_family, action_op, action_cfg)

    root_comp = action_comp
    root_family = action_family
    if action is not None and action.parent_result_id:
        root_parent = by_result_id.get(action.parent_result_id)
        if root_parent is not None:
            try:
                from .evidence_reducer import resolve_generating_record

                root = resolve_generating_record(
                    root_parent,
                    by_result_id=by_result_id,
                    spawning_actions=spawning_actions,
                )
            except Exception:  # noqa: BLE001
                root = root_parent
            root_action = spawning_actions.get(root.result_id)
            root_family = root.backend_family
            root_op = (
                root_action.operator_id
                if root_action is not None
                else f"{root_family}_default"
            )
            root_cfg = dict(
                (root_action.config_delta or {}) if root_action is not None else {}
            )
            root_comp = route_component_key(root_family, root_op, root_cfg)

    route_key = (
        f"route::{action_comp}"
        if root_comp == action_comp
        else f"route::{root_comp}->{action_comp}"
    )
    score = _diagnostic_refilter_score(parent)
    return {
        "source_family": action_family,
        "root_family": root_family,
        "route_key": route_key,
        "route_group": route_key,
        "parent": parent,
        "score": score,
        "native_like": _native_strict_like_pending_score_conversion(parent),
        "proxy_promising": _proxy_promising_pending_score_conversion(parent),
    }


def _chain_route_is_promoted(
    ctx: dict[str, Any],
    route_row: Any,
    family_row: Any,
    *,
    allow_family_fallback: bool = False,
) -> bool:
    """Whether an exact score-conversion route may exceed the fair probe cap.

    Native-like parents can always escape the cap. Otherwise promotion is route
    scoped: a productive family average must not let every sibling config or
    parent-bound redesign spend unlimited AF2 conversion slots. Family fallback
    is kept only for explicit legacy callers with no exact route row.
    """
    if bool(ctx.get("native_like")):
        return True
    rows = [route_row]
    if allow_family_fallback and route_row is None:
        rows.append(family_row)
    for row in rows:
        if row is None:
            continue
        status = str(_route_row_value(row, "status", "") or "")
        if status == "promote":
            return True
        if _route_row_recent_su(row) > 0:
            return True
        if _route_row_current_rate(row) > 0.0:
            return True
        if (
            status == "healthy"
            and int(_route_row_value(row, "near_miss_recent", 0) or 0) > 0
        ):
            return True
    return False


def _parse_and_archive_worker_output(
    slot: _WorkerSlot,
    *,
    return_code: int,
    archive: "Archive",
    target: "TargetConstraint",
    auto_chain_sequence: list[int],
    elapsed_gpu_hours: float = 0.0,
    incremental: bool = False,
) -> int:
    """Parse a completed slot's output, append ResultRecords + auto-chain
    children. Returns the number of records appended.

    `auto_chain_sequence` is a single-element list used as a mutable int — the
    caller-shared counter that ensures auto-chain candidate_ids stay
    unique across slots within the same controller run.

    `elapsed_gpu_hours` is the actual wall time the worker ran on the GPU.
    In incremental mode (currently BindCraft only), the parser may see the
    same rows/PDBs repeatedly while the worker is still running. We append only
    fresh result_ids and charge only the elapsed GPU-h delta since the previous
    archive pass for this slot, so incremental feedback cannot double-count SU
    or worker GPU-h.
    """
    candidate = slot.cand
    requested_output_directory = slot.out_dir
    if candidate is None or requested_output_directory is None:
        return 0
    gpu_id = slot.gpu_id
    tick_id = slot.tick_id

    if return_code != 0:
        # Attempt parsing on nonzero exits because usable outputs may already exist.
        print(
            f"  [worker exit_code={return_code}][gpu={gpu_id}] "
            f"{candidate.method_family} — "
            f"attempting parse anyway (resilient mode)",
            flush=True,
        )

    parent_ids: list[str] = [candidate.candidate_id]
    if slot.parent_result_id:
        parent_ids.append(slot.parent_result_id)
    parent_backend_family = ""
    if candidate.method_family == "structure_refilter" and slot.parent_result_id:
        for parent_result in archive.iter_records(ResultRecord):
            if parent_result.result_id == slot.parent_result_id:
                parent_backend_family = parent_result.backend_family
                break
    try:
        existing_result_ids = {
            result.result_id for result in archive.iter_records(ResultRecord)
        }
    except Exception:  # noqa: BLE001
        existing_result_ids = set()
    processing_request = WorkerOutputRequest(
        candidate=candidate,
        requested_output_directory=requested_output_directory,
        target=target,
        tick_id=tick_id,
        target_pdb_path=str(TARGET_PDB_PATH) if TARGET_PDB_PATH else "",
        parent_pdb_path=slot.parent_pdb_str,
        parent_result_id=slot.parent_result_id,
        target_chains_csv=getattr(slot, "target_chains_csv", ""),
        binder_chain=getattr(slot, "binder_chain", ""),
        parent_backend_family=parent_backend_family,
        existing_result_ids=frozenset(existing_result_ids),
        previously_archived_result_ids=frozenset(slot.archived_result_ids),
        elapsed_gpu_hours=elapsed_gpu_hours,
        previously_archived_gpu_hours=slot.archived_elapsed_gpu_h,
    )
    processed_output = process_worker_output(
        processing_request,
        parser_registry=OutputParserRegistry(
            bindcraft=parse_bindcraft_output,
            af2_refilter=parse_af2_refilter_output,
            proteinmpnn=parse_proteinmpnn_output,
            boltzgen=parse_boltzgen_output,
            complexa=parse_complexa_output,
        ),
    )
    if processed_output.output_directory != requested_output_directory:
        print(
            f"  [complexa_globbed] {requested_output_directory.name} → "
            f"{processed_output.output_directory.name}",
            flush=True,
        )
    if processed_output.parser_error:
        print(
            f"  [parse_error][gpu={gpu_id}] {processed_output.parser_error}",
            flush=True,
        )
    records = list(processed_output.result_records)
    parsed_record_count = processed_output.parsed_record_count
    had_previously_archived_records = bool(slot.archived_result_ids)
    elapsed_gpu_hours_delta = processed_output.elapsed_gpu_hours_delta
    parser_context = processed_output.parser_context
    if processed_output.duplicate_record_count:
        print(
            f"  [dedupe_parse][gpu={gpu_id}] {candidate.method_family} "
            f"parsed={parsed_record_count} fresh={len(records)} "
            f"already_archived={processed_output.duplicate_record_count}",
            flush=True,
        )

    for result in records:
        archive.append(result)
        slot.archived_result_ids.add(result.result_id)
    if records:
        slot.archived_elapsed_gpu_h = max(
            slot.archived_elapsed_gpu_h,
            elapsed_gpu_hours,
        )
    parse_label = "incremental_gpu" if incremental else "gpu"
    logged_gpu_hours_per_record = (
        elapsed_gpu_hours_delta / len(records) if records else 0.0
    )
    print(
        f"  [{parse_label}={gpu_id}] {candidate.method_family} → "
        f"{len(records)} records "
        f"(parsed={parsed_record_count}; elapsed_gpu_h={elapsed_gpu_hours:.2f}; "
        f"delta_gpu_h={elapsed_gpu_hours_delta:.2f}; "
        f"per-record={logged_gpu_hours_per_record:.3f})",
        flush=True,
    )

    # When no design records were parsed, retain failure status and elapsed compute.
    # Record all failures and successful empty jobs lasting at least 18 seconds; shorter
    # successful runs are treated as dispatch noise.
    final_parse_replayed_known_records = (
        (not incremental)
        and len(records) == 0
        and (had_previously_archived_records or parsed_record_count > 0)
    )
    if final_parse_replayed_known_records:
        print(
            f"  [completion_no_new_records][gpu={gpu_id}] "
            f"{candidate.method_family} parsed={parsed_record_count}; "
            f"prior_archived={len(slot.archived_result_ids)} "
            "clean completion after incremental/deduped parse",
            flush=True,
        )
        if (had_previously_archived_records or return_code != 0) and (
            return_code != 0 or elapsed_gpu_hours_delta >= 0.005
        ):
            synthetic_result = create_synthetic_result_record(
                candidate=candidate,
                target_id=target.target_id,
                parent_ids=parent_ids,
                tick_id=tick_id,
                return_code=return_code,
                elapsed_gpu_hours=elapsed_gpu_hours_delta,
                refilter_role=parser_context.refilter_role,
                refilter_source_family=parser_context.refilter_source_family,
                completion_after_incremental_parse=True,
            )
            archive.append(synthetic_result)
            slot.archived_result_ids.add(synthetic_result.result_id)
            slot.archived_elapsed_gpu_h = max(
                slot.archived_elapsed_gpu_h,
                elapsed_gpu_hours,
            )
    elif (
        (not incremental)
        and len(records) == 0
        and (return_code != 0 or elapsed_gpu_hours_delta >= 0.005)
    ):
        synthetic_result = create_synthetic_result_record(
            candidate=candidate,
            target_id=target.target_id,
            parent_ids=parent_ids,
            tick_id=tick_id,
            return_code=return_code,
            elapsed_gpu_hours=elapsed_gpu_hours_delta,
            refilter_role=parser_context.refilter_role,
            refilter_source_family=parser_context.refilter_source_family,
        )
        archive.append(synthetic_result)
        slot.archived_result_ids.add(synthetic_result.result_id)
        slot.archived_elapsed_gpu_h = max(
            slot.archived_elapsed_gpu_h,
            elapsed_gpu_hours,
        )
        print(
            f"  [synth_summary][gpu={gpu_id}] {candidate.method_family} "
            f"exit_status={synthetic_result.exit_status} "
            f"gpu_h={elapsed_gpu_hours_delta:.2f} "
            f"(rc={return_code}; no records; "
            f"{elapsed_gpu_hours_delta*60:.1f} min newly spent — "
            "tracked for truthful method health)",
            flush=True,
        )
    elif (not incremental) and len(records) == 0:
        # Very short workers (<18s) are too small to count as meaningful GPU-h
        # in method_health, but they must not leave route_health stuck as
        # "started forever". Mark the dispatch as parse/no-artifact failed so
        # queued-route accounting can clear it without fabricating a ResultRecord.
        _append_parse_failed_dispatch_record(
            archive,
            slot,
            (
                "parser returned zero records below synthetic ResultRecord "
                f"threshold (elapsed_gpu_h={elapsed_gpu_hours:.5f}, "
                f"rc={return_code})"
            ),
        )

    # Auto-chain: emit follow-up ActionCandidates for downstream families.
    if records and candidate.downstream_route_plan:
        latest_evidence = archive.latest_evidence(target.target_id)
        score_conversion_schedule = build_score_conversion_schedule(
            ScoreConversionSchedulingRequest(
                source_candidate=candidate,
                result_records=tuple(records),
                tick_id=tick_id,
                starting_sequence_number=auto_chain_sequence[0],
            ),
            score_parent_result=lambda parent_result: _diagnostic_refilter_score(
                parent_result,
                latest_evidence,
            ),
        )
        auto_chain_sequence[0] = score_conversion_schedule.final_sequence_number
        for scheduled_candidate in score_conversion_schedule.scheduled_candidates:
            archive.append(scheduled_candidate)
            print(
                f"  [auto_chain] {candidate.method_family} → "
                f"{scheduled_candidate.method_family} "
                f"parent={scheduled_candidate.parent_result_id} "
                f"candidate={scheduled_candidate.candidate_id}",
                flush=True,
            )
    return len(records)


def _parse_and_archive_slot(
    slot: _WorkerSlot,
    *,
    rc: int,
    archive: "Archive",
    target: "TargetConstraint",
    chain_seq_ref: list[int],
    elapsed_gpu_h: float = 0.0,
    incremental: bool = False,
) -> int:
    """Compatibility wrapper for the former abbreviated controller API."""

    return _parse_and_archive_worker_output(
        slot,
        return_code=rc,
        archive=archive,
        target=target,
        auto_chain_sequence=chain_seq_ref,
        elapsed_gpu_hours=elapsed_gpu_h,
        incremental=incremental,
    )


def _archive_incremental_bindcraft_output(
    slot: _WorkerSlot,
    *,
    archive: "Archive",
    target: "TargetConstraint",
    auto_chain_sequence: list[int],
    observed_at: float,
) -> int:
    """Archive fresh BindCraft artifacts before process exit.

    BindCraft can keep a worker alive for hours after it has already written
    Accepted/Rejected PDBs. Without this pass, the controller sees no official
    AF2 feedback and the high-cost cap stays full. Incremental parse is
    duplicate-safe in _parse_and_archive_slot and charges only elapsed GPU-h
    since the previous archive pass.
    """
    if not slot.busy or slot.cand is None or slot.out_dir is None:
        return 0
    if slot.cand.method_family != "bindcraft":
        return 0
    if slot.launched_at <= 0.0:
        return 0
    if (
        observed_at - slot.last_incremental_parse_at
        < BINDCRAFT_INCREMENTAL_PARSE_INTERVAL_S
    ):
        return 0
    slot.last_incremental_parse_at = observed_at
    elapsed_gpu_hours = max(0.0, (observed_at - slot.launched_at) / 3600.0)
    if elapsed_gpu_hours <= slot.archived_elapsed_gpu_h + 1e-6:
        return 0
    return _parse_and_archive_worker_output(
        slot,
        return_code=0,
        archive=archive,
        target=target,
        auto_chain_sequence=auto_chain_sequence,
        elapsed_gpu_hours=elapsed_gpu_hours,
        incremental=True,
    )


def _chain_backfill_ids(
    archive,
    seen: set,
    max_n: int,
    *,
    admission_tick_id: str | None = None,
    allow_exhausted_background: bool = False,
) -> list[str]:
    """Feasible, not-yet-dispatched chain refilters for score conversion.

    The lane is official-SU plumbing, but it must still be allocated like a
    scarce resource: first-returning diagnostic routes get a bounded fair probe,
    then only evidence-promoted/native-like routes may exceed that probe. This
    prevents one Complexa best-of-n batch from draining many quick AF2 refilters
    before other generator routes have a chance to produce evidence.
    """
    if max_n <= 0:
        return []

    actions = list(archive.iter_records(ActionCandidate))
    results = list(archive.iter_records(ResultRecord))
    by_result_id = {r.result_id: r for r in results}
    spawning_actions = _spawn_index_for_results(actions, results)
    try:
        evs = list(archive.iter_records(EvidenceSummary))
        latest_ev = evs[-1] if evs else None
    except Exception:  # noqa: BLE001
        latest_ev = None
    route_rows, family_rows = _route_value_index(latest_ev)
    route_history: dict[str, list[Any]] = {}
    for ev in evs[:-1]:
        for row in getattr(ev, "route_values", None) or []:
            if _route_row_value(row, "scope", "") != "route":
                continue
            key = str(_route_row_value(row, "strategy_key", "") or "")
            if key:
                route_history.setdefault(key, []).append(row)

    dispatches = list(archive.iter_records(DispatchRecord))
    started_ids = {d.candidate_id for d in dispatches if d.status == "started"}
    terminal_or_started = {
        d.candidate_id
        for d in dispatches
        if d.status == "started"
        or d.status == "parse_failed"
        or (d.status == "dispatch_failed" and not _is_recoverable_dispatch_failure(d))
    }

    action_by_id = {a.candidate_id: a for a in actions}
    launch_decisions = list(archive.iter_records(LaunchDecision))
    served_by_route: dict[str, int] = {}
    served_weak_by_family: dict[str, int] = {}
    for cid in started_ids | set(seen):
        ac = action_by_id.get(cid)
        if (
            ac is None
            or not cid.startswith("chain_")
            or ac.method_family != "structure_refilter"
        ):
            continue
        ctx = _chain_source_context(
            ac, by_result_id=by_result_id, spawning_actions=spawning_actions
        )
        family = str(ctx.get("source_family") or "_unknown")
        parent = ctx.get("parent")
        if parent is not None and not _record_needs_score_conversion(family, parent):
            continue
        route_key = str(ctx.get("route_key") or "")
        route_group = str(ctx.get("route_group") or route_key or "_unknown")
        served_by_route[route_group] = served_by_route.get(route_group, 0) + 1
        route_row = route_rows.get(route_key)
        family_row = family_rows.get(family)
        high_value = (
            _chain_route_is_promoted(ctx, route_row, family_row)
            or bool(ctx.get("native_like"))
            or bool(ctx.get("proxy_promising"))
        )
        if not high_value:
            served_weak_by_family[family] = served_weak_by_family.get(family, 0) + 1

    fair_cap = _chain_fair_probe_per_route()
    weak_family_cap = _chain_weak_family_probe_cap()
    grouped_items: dict[str, list[dict[str, Any]]] = {}
    for c in actions:
        cid = c.candidate_id
        if not cid.startswith("chain_") or c.method_family != "structure_refilter":
            continue
        if cid in terminal_or_started or cid in seen:
            continue
        try:
            if c.feasibility is not None and not c.feasibility.all_ok():
                continue
        except Exception:  # noqa: BLE001
            pass

        ctx = _chain_source_context(
            c, by_result_id=by_result_id, spawning_actions=spawning_actions
        )
        route_key = str(ctx.get("route_key") or "")
        group = str(ctx.get("route_group") or route_key or "_unknown")
        family = str(ctx.get("source_family") or "_unknown")
        parent = ctx.get("parent")
        if parent is not None and not _record_needs_score_conversion(family, parent):
            continue
        route_row = route_rows.get(route_key)
        family_row = family_rows.get(family)
        status = str(
            _route_row_value(route_row, "status", "")
            or _route_row_value(family_row, "status", "")
            or ""
        )
        promoted = _chain_route_is_promoted(ctx, route_row, family_row)
        native_like = bool(ctx.get("native_like"))
        proxy_promising = bool(ctx.get("proxy_promising"))
        proxy_score = float(ctx.get("score", 0.0) or 0.0)
        if promoted or native_like:
            priority = 0
        elif proxy_promising:
            priority = 1
        elif status == "diversify":
            priority = 3
        elif status in {"defer", "collapse_risk"}:
            priority = 4
        else:
            priority = 2
        grouped_items.setdefault(group, []).append(
            {
                "cid": cid,
                "priority": priority,
                "score": proxy_score,
                "promoted": promoted,
                "native_like": native_like,
                "proxy_promising": proxy_promising,
                "status": status,
                "family": family,
            }
        )

    by_group: dict[str, list[dict[str, Any]]] = {}
    exhausted_background: list[dict[str, Any]] = []
    for group, items in grouped_items.items():
        items.sort(
            key=lambda x: (int(x["priority"]), -float(x["score"]), str(x["cid"]))
        )
        route_key = str(group if group.startswith("route::") else "")
        family = str(items[0].get("family") or "_unknown")
        route_row = route_rows.get(route_key)
        family_row = family_rows.get(family)
        status = str(
            _route_row_value(route_row, "status", "")
            or _route_row_value(family_row, "status", "")
            or ""
        )
        served = served_by_route.get(group, 0)
        observed = int(
            _route_row_value(route_row, "canonical_score_conversion_count", 0) or 0
        )
        previous_boundary = max(
            [0] + [t for t in _chain_route_tranches() if t < observed]
        )
        prior_route_row = None
        for prior in reversed(route_history.get(route_key, [])):
            prior_observed = int(
                _route_row_value(prior, "canonical_score_conversion_count", 0) or 0
            )
            if prior_observed <= previous_boundary:
                prior_route_row = prior
                break
        route_cap = _chain_route_tranche_cap(
            fair_cap=fair_cap,
            route_row=route_row,
            prior_route_row=prior_route_row,
        )
        allowed = max(0, min(len(items), route_cap - served))
        if allowed <= 0:
            if allow_exhausted_background and items:
                exhausted_background.append(
                    {
                        "cid": str(items[0]["cid"]),
                        "family": family,
                        "priority": int(items[0]["priority"]),
                        "score": float(items[0]["score"]),
                        "high_value": bool(
                            items[0].get("promoted")
                            or items[0].get("native_like")
                            or items[0].get("proxy_promising")
                        ),
                    }
                )
            continue
        selected_items = []
        for x in items[:allowed]:
            high_value = (
                bool(x.get("promoted"))
                or bool(x.get("native_like"))
                or bool(x.get("proxy_promising"))
            )
            selected_items.append(
                {
                    "cid": str(x["cid"]),
                    "family": family,
                    "priority": int(x["priority"]),
                    "score": float(x["score"]),
                    "high_value": high_value,
                }
            )
        by_group[group] = selected_items

    out: list[str] = []
    planned_total_by_family: dict[str, int] = {}
    if admission_tick_id:
        for decision in launch_decisions:
            if decision.tick_id != admission_tick_id or decision.status != "launched":
                continue
            action = action_by_id.get(decision.candidate_id)
            if action is None or not is_canonical_score_conversion(action):
                continue
            ctx = _chain_source_context(
                action,
                by_result_id=by_result_id,
                spawning_actions=spawning_actions,
            )
            family = str(ctx.get("source_family") or "_unknown")
            planned_total_by_family[family] = planned_total_by_family.get(family, 0) + 1
    planned_weak_by_family: dict[str, int] = {}
    family_tick_cap = _chain_family_score_conversion_tick_cap()
    family_high_tick_cap = _chain_family_score_conversion_high_value_tick_cap()
    # Round-robin across exact route/config groups, so an early high-output route
    # cannot occupy the whole reserve before other pending routes are probed.
    active = sorted(by_group)
    while len(out) < max_n and active:
        nxt: list[str] = []
        for group in active:
            if len(out) >= max_n:
                break
            items = by_group[group]
            while items and len(out) < max_n:
                x = items.pop(0)
                cid = str(x["cid"])
                family = str(x.get("family") or "_unknown")
                high_value = bool(x.get("high_value"))
                total_planned = planned_total_by_family.get(family, 0)
                tick_cap = family_high_tick_cap if high_value else family_tick_cap
                if total_planned >= tick_cap:
                    continue
                if not high_value:
                    weak_remaining = (
                        weak_family_cap
                        - served_weak_by_family.get(family, 0)
                        - planned_weak_by_family.get(family, 0)
                    )
                    if weak_remaining <= 0:
                        continue
                    planned_weak_by_family[family] = (
                        planned_weak_by_family.get(family, 0) + 1
                    )
                planned_total_by_family[family] = total_planned + 1
                out.append(cid)
                break
            if items:
                nxt.append(group)
        active = nxt
    if allow_exhausted_background and len(out) < max_n and exhausted_background:
        exhausted_background.sort(
            key=lambda x: (int(x["priority"]), -float(x["score"]), str(x["cid"]))
        )
        for item in exhausted_background:
            family = str(item["family"])
            high_value = bool(item["high_value"])
            tick_cap = family_high_tick_cap if high_value else family_tick_cap
            if planned_total_by_family.get(family, 0) >= tick_cap:
                continue
            out.append(str(item["cid"]))
            break
    return out[:max_n]


def _completed_candidate_ids(archive) -> set[str]:
    completed: set[str] = set()
    for r in archive.iter_records(ResultRecord):
        for pid in r.parent_ids or []:
            completed.add(pid)
    return completed


def _pid_start_ticks(pid: int | None) -> int | None:
    """Read Linux /proc start ticks, which disambiguate PID reuse."""
    if pid is None or pid <= 1:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # comm is parenthesized and may itself contain spaces or ')'. Fields
        # after the final ')' begin with state (field 3); starttime is field 22.
        fields = stat[stat.rfind(")") + 2 :].split()
        if not fields or fields[0] in {"Z", "X"}:
            return None
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # kill(pid, 0) also succeeds for zombies; /proc starttime treats those as
    # terminal so restart recovery does not wait forever on an unreapable row.
    return _pid_start_ticks(pid) is not None


def _pgid_alive(pgid: int | None) -> bool:
    if pgid is None or pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fence_started_dispatch_for_recovery(dispatch: DispatchRecord) -> bool:
    """Return whether a previously-started worker is safe to relaunch.

    New rows carry enough process identity to fence a worker that survived a
    controller restart in the same SLURM allocation. Legacy rows are not blindly
    relaunched by default because that can create two writers for one output
    directory; operators can opt into the historical behavior only when they
    know the old allocation is gone.
    """
    recorded_host = str(getattr(dispatch, "worker_host", "") or "")
    current_host = socket.gethostname()
    recorded_job = str(getattr(dispatch, "slurm_job_id", "") or "")
    current_job = str(os.environ.get("SLURM_JOB_ID", "") or "")
    pid = getattr(dispatch, "worker_pid", None)
    pgid = getattr(dispatch, "worker_pgid", None)
    recorded_start_ticks = getattr(dispatch, "worker_pid_start_ticks", None)

    if recorded_job and current_job and recorded_job != current_job:
        # SLURM fences processes at allocation teardown. A different job id is a
        # new allocation, so the archived intent is safe to replay.
        return True
    if recorded_host and recorded_host != current_host:
        return False
    pid_alive = _pid_alive(pid)
    group_alive = _pgid_alive(pgid)
    if not pid_alive and not group_alive:
        if pid is not None or pgid is not None:
            return True
        return os.environ.get("TREX_RECOVER_LEGACY_STARTED", "0") == "1"
    if recorded_start_ticks is None:
        # A live PID without an archived birth identity might have been reused;
        # never signal it or launch a duplicate based on PID alone.
        return False
    # If the leader is gone but descendants remain in its process group, there
    # is no /proc/<pgid> entry. That is still the archived worker tree and must
    # be fenced before replay. A different live leader birth time means the
    # numeric PID/PGID was reused after the old group exited; do not signal it.
    current_start_ticks = _pid_start_ticks(pid if pid_alive else pgid)
    if current_start_ticks is not None and current_start_ticks != int(
        recorded_start_ticks
    ):
        return True
    if (
        recorded_host != current_host
        or recorded_job != current_job
        or pgid is None
        or pgid <= 1
    ):
        return False

    try:
        os.killpg(int(pgid), signal.SIGTERM)
        deadline = time.monotonic() + 5.0
        while _pgid_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _pgid_alive(pgid):
            os.killpg(int(pgid), signal.SIGKILL)
            deadline = time.monotonic() + 2.0
            while _pgid_alive(pgid) and time.monotonic() < deadline:
                time.sleep(0.05)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return not _pgid_alive(pgid)


def _recover_undispatched_launch_ids(
    archive,
    chain_backfill_seen: set[str] | None = None,
) -> list[str]:
    """Recover selected-but-not-dispatched candidates after controller restart.

    The event controller's `pending` queue and worker pool are in-memory. If the
    process exits after LaunchDecision is archived, or after DispatchRecord(started)
    but before ResultRecord parsing, those candidates otherwise look selected
    forever while no current worker can finish them. Recover any non-terminal,
    incomplete launch intent at startup.
    """
    actions = {c.candidate_id: c for c in archive.iter_records(ActionCandidate)}
    dispatches = list(archive.iter_records(DispatchRecord))
    terminal = {
        d.candidate_id
        for d in archive.iter_records(DispatchRecord)
        if d.status == "parse_failed"
        or (d.status == "dispatch_failed" and not _is_recoverable_dispatch_failure(d))
    }
    latest_dispatch: dict[str, DispatchRecord] = {}
    for dispatch in dispatches:
        latest_dispatch[dispatch.candidate_id] = dispatch
    completed = _completed_candidate_ids(archive)
    out: list[str] = []
    seen: set[str] = set()
    for launch in archive.iter_records(LaunchDecision):
        if launch.status != "launched":
            continue
        cid = launch.candidate_id
        if cid in seen or cid in terminal or cid in completed:
            continue
        cand = actions.get(cid)
        if cand is None:
            continue
        try:
            if cand.feasibility is not None and not cand.feasibility.all_ok():
                continue
        except Exception:  # noqa: BLE001
            pass
        dispatch = latest_dispatch.get(cid)
        if (
            dispatch is not None
            and dispatch.status == "started"
            and not _fence_started_dispatch_for_recovery(dispatch)
        ):
            continue
        out.append(cid)
        seen.add(cid)
    if chain_backfill_seen is not None:
        chain_backfill_seen.update(cid for cid in out if cid.startswith("chain_"))
    return out


def _short_id(value: str, n: int = 10) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:n]


def _latest_launch_decision(archive, candidate_id: str) -> LaunchDecision | None:
    if archive is None:
        return None
    latest: LaunchDecision | None = None
    for rec in archive.iter_records(LaunchDecision):
        if rec.candidate_id == candidate_id and rec.status == "launched":
            latest = rec
    return latest


def _latest_launch_decisions_for_pending(
    archive, pending: list[str]
) -> dict[str, LaunchDecision]:
    """Return latest LaunchDecision rows for a pending queue in one archive scan."""
    if archive is None or not pending:
        return {}
    wanted = set(pending)
    latest: dict[str, LaunchDecision] = {}
    try:
        for rec in archive.iter_records(LaunchDecision):
            if rec.status == "launched" and rec.candidate_id in wanted:
                latest[rec.candidate_id] = rec
    except Exception:  # noqa: BLE001
        return {}
    return latest


def _tick_sort_value(tick_id: str | None) -> int:
    """Parse v7rNNN-style tick ids for ordering; unknown ticks sort oldest."""
    s = str(tick_id or "")
    pos = s.rfind("r")
    if pos < 0:
        return -1
    digits = []
    for ch in s[pos + 1 :]:
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    if not digits:
        return -1
    try:
        return int("".join(digits))
    except ValueError:
        return -1


def _archive_resume_state(
    archive: Archive,
    *,
    target_id: str | None = None,
) -> tuple[int, float]:
    """Return the next-safe round base and cumulative elapsed wall hours."""
    max_round = 0
    for cls in (EvidenceSummary, LaunchDecision, DispatchRecord, ResultRecord):
        for record in archive.iter_records(cls):
            max_round = max(
                max_round, _tick_sort_value(getattr(record, "tick_id", None))
            )
    elapsed_wall_h = max(
        (float(e.elapsed_wall_h or 0.0) for e in archive.iter_records(EvidenceSummary)),
        default=0.0,
    )
    checkpoint = _read_controller_checkpoint(archive)
    if checkpoint is not None and (
        target_id is None or str(checkpoint.get("target_id")) == str(target_id)
    ):
        max_round = max(max_round, int(checkpoint.get("round_id", 0) or 0))
        elapsed_wall_h = max(
            elapsed_wall_h,
            float(checkpoint.get("elapsed_wall_h", 0.0) or 0.0),
        )
    return max_round, max(0.0, elapsed_wall_h)


def _acquire_controller_lock(root: Path) -> int:
    """Hold a non-blocking exclusive lease for one controller per archive."""
    path = root / ".controller.lock"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise SystemExit(
            f"[v7_controller][FATAL] archive already has an active controller: {root}"
        ) from exc
    os.ftruncate(fd, 0)
    os.write(
        fd,
        (
            f"pid={os.getpid()} host={socket.gethostname()} "
            f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}\n"
        ).encode("utf-8"),
    )
    return fd


def _pending_signal_priority_text(
    cand: ActionCandidate | None,
    launch_decision: LaunchDecision | None,
) -> str:
    parts: list[str] = []
    if cand is not None:
        parts.append(str(getattr(cand, "expected_signal", "") or ""))
        parts.append(str(getattr(cand, "supervisor_mode", "") or ""))
    if launch_decision is not None:
        parts.append(str(getattr(launch_decision, "why", "") or ""))
        meta = getattr(launch_decision, "resource_class_concrete", None) or {}
        parts.append(str(meta.get("mode", "") or ""))
    return " ".join(parts).lower()


def _pending_priority_key(
    cid: str,
    cand: ActionCandidate | None,
    launch_decision: LaunchDecision | None,
    original_index: int,
) -> tuple[int, int, int, int]:
    """Priority-sort queued intents before dispatch.

    Prefetch means selected candidates can sit in pending while newer evidence
    arrives. FIFO dispatch then turns the loop into delayed feedback. This key
    keeps score-conversion plumbing first, favors GPU/medium-recent route-value
    signals and newer decisions, and demotes lifetime-memory-only exact replays.
    """
    if cid.startswith("chain_"):
        return (
            0,
            0,
            -_tick_sort_value(getattr(launch_decision, "tick_id", None)),
            original_index,
        )

    text = _pending_signal_priority_text(cand, launch_decision)
    current_signal = (
        "source=gpu_recent" in text
        or "source=medium_recent" in text
        or "new_su_recent" in text
        or "recent_su" in text
        or "recent_near_miss" in text
    )
    diagnostic_signal = (
        "diagnostic_improvement_score" in text or "diagnostic_axes=" in text
    )
    stale_lifetime_only = (
        "source=lifetime_memory" in text and "no_recent_signal=1" in text
    )
    if current_signal:
        signal_rank = 1
    elif diagnostic_signal:
        signal_rank = 2
    elif stale_lifetime_only:
        signal_rank = 5
    else:
        # Ordinary Supervisor candidates and cross-family escape-floor candidates
        # live in the same tier; newer evidence should decide their order.
        signal_rank = 3

    return (
        signal_rank,
        0 if launch_decision is not None else 1,
        -_tick_sort_value(getattr(launch_decision, "tick_id", None)),
        original_index,
    )


def _prioritize_pending_queue(
    archive,
    pending: list[str],
    cand_by_id: dict[str, ActionCandidate],
) -> None:
    """Reorder selected candidates so actual starts track current evidence."""
    if archive is None or len(pending) <= 1:
        return
    latest = _latest_launch_decisions_for_pending(archive, pending)
    if not latest and not any(str(cid).startswith("chain_") for cid in pending):
        return
    indexed = list(enumerate(pending))
    indexed.sort(
        key=lambda item: _pending_priority_key(
            item[1], cand_by_id.get(item[1]), latest.get(item[1]), item[0]
        )
    )
    pending[:] = [cid for _, cid in indexed]


def _action_candidate_by_id(
    archive, candidate_id: str | None
) -> ActionCandidate | None:
    if archive is None or not candidate_id:
        return None
    latest: ActionCandidate | None = None
    try:
        for rec in archive.iter_records(ActionCandidate):
            if rec.candidate_id == candidate_id:
                latest = rec
    except Exception:  # noqa: BLE001
        return None
    return latest


def _score_credit_basis_for_candidate(cand: ActionCandidate | None) -> str | None:
    if cand is None:
        return None
    refilter_role = infer_refilter_role(cand)
    if (
        cand.method_family == "structure_refilter"
        and refilter_role == CANONICAL_SCORE_CONVERSION
    ):
        return "official_score_conversion"
    if (
        cand.method_family == "structure_refilter"
        and refilter_role == PARENT_MODEL_REFOLD
    ):
        return "advisory_refold_only"
    try:
        from .capability_registry import default_registry

        cap = default_registry().get(cand.method_family)
        if bool(getattr(cap, "outputs_diagnostic_only", False)):
            return "requires_canonical_score_conversion"
    except Exception:  # noqa: BLE001
        pass
    return "direct_official_strict_metrics"


def _dispatch_record_metadata(
    archive=None,
    *,
    cand: ActionCandidate | None = None,
    candidate_id: str | None = None,
    launch_decision: LaunchDecision | None = None,
) -> dict[str, Any]:
    """Denormalized candidate provenance for DispatchRecord rows.

    DispatchRecord is the started/failed ground truth. Store the same route role
    vocabulary that the Planner/Supervisor saw so log readers do not need to
    reconstruct candidate joins before judging selected->started alignment.
    """
    if cand is None:
        cand = _action_candidate_by_id(archive, candidate_id)
    meta = (
        getattr(launch_decision, "resource_class_concrete", None)
        if launch_decision is not None
        else None
    ) or {}
    supervisor_mode = (
        getattr(cand, "supervisor_mode", None) if cand is not None else None
    )
    if supervisor_mode is None:
        supervisor_mode = meta.get("mode")
    return {
        "method_family": getattr(cand, "method_family", None)
        if cand is not None
        else None,
        "operator_id": getattr(cand, "operator_id", None) if cand is not None else None,
        "supervisor_mode": supervisor_mode,
        "refilter_role": infer_refilter_role(cand) if cand is not None else None,
        "score_credit_basis": _score_credit_basis_for_candidate(cand),
    }


def _append_parse_failed_dispatch_record(
    archive, slot, reason: str, *, target_id: str | None = None
) -> None:
    if archive is None:
        return
    if slot.cand is None:
        return
    cid = slot.cand.candidate_id
    launch_decision = _latest_launch_decision(archive, cid)
    archive.append(
        DispatchRecord(
            dispatch_id=(
                f"parse_failed_{slot.tick_id or 'unknown'}_"
                f"{_short_id(cid + ':' + str(time.time()))}"
            ),
            launch_id=(
                launch_decision.launch_id if launch_decision is not None else None
            ),
            tick_id=slot.tick_id or "unknown",
            candidate_id=cid,
            status="parse_failed",
            worker_slot=str(slot.slot_id),
            gpu_id=str(slot.gpu_id),
            output_dir=str(slot.out_dir) if slot.out_dir is not None else None,
            parent_result_id=slot.parent_result_id or None,
            parent_pdb_path=slot.parent_pdb_str or None,
            **_dispatch_record_metadata(
                cand=slot.cand, launch_decision=launch_decision
            ),
            attempt=1,
            why=reason[:300],
        )
    )
    # Retain elapsed compute after parsing fails when target_id is known. Empty
    # measurements prevent this failure record from qualifying.
    if target_id and slot.launched_at and slot.launched_at > 0:
        elapsed_gpu_h = max(0.0, (time.time() - slot.launched_at) / 3600.0)
        if elapsed_gpu_h >= 0.005:
            from .schemas import ResultRecord as _RR

            fam = slot.cand.method_family if slot.cand is not None else "unknown"
            pids = [cid]
            if slot.parent_result_id:
                pids.append(slot.parent_result_id)
            role = infer_refilter_role(slot.cand) if slot.cand is not None else None
            archive.append(
                _RR(
                    result_id=f"synth_parsefail_{cid}_{int(time.time()):x}"[:80],
                    parent_ids=pids,
                    target_id=target_id,
                    backend_family=fam,
                    runtime_bucket_id="rb_v7",
                    metrics={},
                    metrics_calibrated={},
                    route_lineage=[fam],
                    gpu_h=elapsed_gpu_h,
                    exit_status="no_artifacts",  # type: ignore[arg-type]
                    bins={"refilter_role": role} if role else {},
                    artifacts={},
                    panel_ready=False,
                    tick_id=slot.tick_id or None,
                )
            )


def _chain_refilter_reserve_limit(queue_room: int, backlog_n: int | None = None) -> int:
    """Per-round score-conversion reserve for diagnostic auto-chain refilters.

    Idle-only backfill under-credits diagnostic lanes: when the generator queue
    stays full, accepted BindCraft/MPNN/BoltzGen artifacts can sit unscored for
    hours and read as 0 SU. Reserve a slice of newly-free capacity so canonical
    structure_refilter scoring (the ONLY SU-minting step for those families)
    progresses without letting refilters monopolize all workers.

    Two modes:
      - ``backlog_n is None`` (legacy / single-arg): fixed floor reserve of
        ``TREX_CHAIN_REFILTER_RESERVE_PER_ROUND`` (default 1), bounded by
        ``queue_room``. Preserved for callers/tests that don't measure backlog.
      - ``backlog_n`` given: size the reserve to the actual unscored backlog.
        The default lane cap is 1, but rises to 2 when the high-value backlog is
        large (``TREX_CHAIN_REFILTER_HIGH_BACKLOG_MIN``, default 32). Canonical
        scoring therefore progresses asynchronously without occupying all free
        workers or delaying fresh scientific planning. Explicit
        ``TREX_CHAIN_REFILTER_MAX_CONCURRENT`` still overrides the adaptive cap
        for ablations.
    """
    if queue_room <= 0:
        return 0
    raw = os.environ.get("TREX_CHAIN_REFILTER_RESERVE_PER_ROUND", "1")
    try:
        floor = int(raw)
    except ValueError:
        floor = 1
    floor = max(0, floor)
    if backlog_n is None:
        return max(0, min(queue_room, floor))
    if backlog_n <= 0:
        return 0
    try:
        frac = float(os.environ.get("TREX_CHAIN_REFILTER_RESERVE_MAX_FRACTION", "0.5"))
    except ValueError:
        frac = 0.5
    frac = min(max(frac, 0.0), 1.0)
    gen_floor = 1 if queue_room >= 2 else 0  # keep a generation slot when room>1
    frac_cap = math.ceil(queue_room * frac)
    raw_lane_cap = os.environ.get("TREX_CHAIN_REFILTER_MAX_CONCURRENT")
    if raw_lane_cap is None or not raw_lane_cap.strip():
        try:
            high_backlog_min = max(
                1, int(os.environ.get("TREX_CHAIN_REFILTER_HIGH_BACKLOG_MIN", "32"))
            )
        except ValueError:
            high_backlog_min = 32
        lane_cap = 2 if backlog_n >= high_backlog_min else 1
    else:
        try:
            lane_cap = max(1, int(raw_lane_cap))
        except ValueError:
            lane_cap = 1
    max_reserve = max(
        0,
        min(queue_room - gen_floor, max(floor, frac_cap), lane_cap),
    )
    return max(0, min(backlog_n, max_reserve))


def _chain_candidate_native_like(archive, candidate_id: str) -> bool:
    try:
        actions = list(archive.iter_records(ActionCandidate))
        results = list(archive.iter_records(ResultRecord))
        by_result_id = {r.result_id: r for r in results}
        spawning_actions = _spawn_index_for_results(actions, results)
        action_by_id = {a.candidate_id: a for a in actions}
        cand = action_by_id.get(candidate_id)
        if cand is None:
            return False
        ctx = _chain_source_context(
            cand, by_result_id=by_result_id, spawning_actions=spawning_actions
        )
        return bool(ctx.get("native_like"))
    except Exception:  # noqa: BLE001
        return False


def _chain_escape_share_allowance(archive, available_ids: list[str]) -> int:
    native_ids = [
        cid for cid in available_ids if _chain_candidate_native_like(archive, cid)
    ]
    if native_ids:
        try:
            evs = list(archive.iter_records(EvidenceSummary))
            state = str(getattr(evs[-1], "state_label", "") or "") if evs else ""
        except Exception:  # noqa: BLE001
            state = ""
        if state == "deep_stall":
            return 1
        native_cap = _bounded_int_env(
            "TREX_CHAIN_REFILTER_NATIVE_ESCAPE_MAX", 2, min_value=1
        )
        return min(native_cap, len(native_ids))
    return (
        1
        if any(_chain_candidate_has_escape_value(archive, cid) for cid in available_ids)
        else 0
    )


def _chain_candidate_has_escape_value(archive, candidate_id: str) -> bool:
    """Whether one score-conversion candidate may bypass the recent share cap.

    Native-strict-like parents and exact route-promoted parents are the cases
    where suppressing the only score-conversion slot can hide real SU. This is
    intentionally exact-route aware; a productive family average alone is not
    enough to bypass the share cap.
    """
    try:
        actions = list(archive.iter_records(ActionCandidate))
        results = list(archive.iter_records(ResultRecord))
        by_result_id = {r.result_id: r for r in results}
        spawning_actions = _spawn_index_for_results(actions, results)
        action_by_id = {a.candidate_id: a for a in actions}
        cand = action_by_id.get(candidate_id)
        if cand is None:
            return False
        ctx = _chain_source_context(
            cand, by_result_id=by_result_id, spawning_actions=spawning_actions
        )
        if bool(ctx.get("native_like")):
            return True
        if bool(ctx.get("proxy_promising")):
            return True
        evs = list(archive.iter_records(EvidenceSummary))
        latest_ev = evs[-1] if evs else None
        route_rows, family_rows = _route_value_index(latest_ev)
        route_row = route_rows.get(str(ctx.get("route_key") or ""))
        family_row = family_rows.get(str(ctx.get("source_family") or ""))
        return _chain_route_is_promoted(ctx, route_row, family_row)
    except Exception:  # noqa: BLE001
        return False


def _chain_refilter_recent_share_cap(
    archive,
    requested: int,
    available_ids: list[str],
) -> int:
    """Cap canonical score-conversion launches by recent launch share.

    The reserve cap is per call, but AF2 score-conversion jobs can be short and
    repeatedly free the same worker. This recent-window cap prevents the plumbing
    lane from occupying nearly every launch on productive targets while keeping a
    one-slot escape valve for native-like/exact-route-promoted parents.
    """
    if requested <= 0 or not available_ids:
        return 0
    try:
        window = max(1, int(os.environ.get("TREX_CHAIN_REFILTER_SHARE_WINDOW", "40")))
    except ValueError:
        window = 40
    try:
        min_n = max(0, int(os.environ.get("TREX_CHAIN_REFILTER_SHARE_MIN_N", "20")))
    except ValueError:
        min_n = 20
    try:
        base_share = float(os.environ.get("TREX_CHAIN_REFILTER_MAX_SHARE", "0.60"))
    except ValueError:
        base_share = 0.60
    try:
        dup_share = float(
            os.environ.get("TREX_CHAIN_REFILTER_MAX_SHARE_DUPLICATE", "0.50")
        )
    except ValueError:
        dup_share = 0.50
    base_share = min(max(base_share, 0.0), 1.0)
    dup_share = min(max(dup_share, 0.0), 1.0)

    starts = [
        d
        for d in archive.iter_records(DispatchRecord)
        if getattr(d, "status", "") == "started"
    ][-window:]
    launch_by_id = {
        str(getattr(ld, "candidate_id", "") or ""): ld
        for ld in archive.iter_records(LaunchDecision)
        if getattr(ld, "status", "") == "launched"
    }

    def _dispatch_is_score_conversion(d: DispatchRecord) -> bool:
        cid = str(getattr(d, "candidate_id", "") or "")
        if cid.startswith("chain_"):
            return True
        ld = launch_by_id.get(cid)
        rc = getattr(ld, "resource_class_concrete", {}) if ld is not None else {}
        return isinstance(rc, dict) and rc.get("mode") == "chain_refilter"

    chain_n = sum(1 for d in starts if _dispatch_is_score_conversion(d))
    try:
        evs = list(archive.iter_records(EvidenceSummary))
        state = str(getattr(evs[-1], "state_label", "") or "") if evs else ""
    except Exception:  # noqa: BLE001
        state = ""
    max_share = (
        dup_share
        if state in {"productive_duplicate", "strict_duplicate_collapse"}
        else base_share
    )
    if len(starts) < min_n:
        # The bootstrap allowance uses the same share budget as later dispatches
        # unless explicitly overridden.
        raw_bootstrap = os.environ.get("TREX_CHAIN_REFILTER_BOOTSTRAP_MAX")
        if raw_bootstrap is None or str(raw_bootstrap).strip() == "":
            bootstrap_max = int(math.floor(max_share * max(1, min_n)))
        else:
            bootstrap_max = _bounded_int_env(
                "TREX_CHAIN_REFILTER_BOOTSTRAP_MAX",
                int(math.floor(max_share * max(1, min_n))),
                min_value=0,
            )
        room = max(0, bootstrap_max - chain_n)
        if room > 0:
            return min(requested, room)
        return min(requested, _chain_escape_share_allowance(archive, available_ids))
    allowed_total = int(math.floor(max_share * window))
    room = max(0, allowed_total - chain_n)
    if room > 0:
        return min(requested, room)
    return min(requested, _chain_escape_share_allowance(archive, available_ids))


def _chain_escape_first(archive, ids: list[str]) -> list[str]:
    """Move the one allowed native/promoted escape to the front under share caps."""
    for i, cid in enumerate(ids):
        if _chain_candidate_has_escape_value(archive, cid):
            if i == 0:
                return ids
            return [cid] + ids[:i] + ids[i + 1 :]
    return ids


def _append_chain_launch_decisions(
    archive,
    ids: list[str],
    *,
    tick_id: str,
    source: str,
    why: str,
) -> None:
    for j, cid in enumerate(ids):
        archive.append(
            LaunchDecision(
                launch_id=f"{source}_{tick_id}_{j:02d}_{_short_id(cid)}",
                tick_id=tick_id,
                candidate_id=cid,
                status="launched",
                resource_class_concrete={
                    "class": "low",
                    # System-reserved canonical scoring lane. Do not tag as
                    # "rescue": recent mode windows use {exploit,rescue,explore}
                    # to preserve LLM allocation shares, and deterministic
                    # chain-refilter backfill should not consume that rescue budget.
                    "mode": "chain_refilter",
                    "refilter_role": CANONICAL_SCORE_CONVERSION,
                    "source": source,
                },
                why=why,
            )
        )


def _has_native_strict_like_pending_refilter(archive) -> bool:
    """Whether deep_stall must keep a minimal canonical scoring floor.

    Native-strict-like artifacts are the highest-value case, but any accepted
    diagnostic artifact that has not yet received canonical AF2 scoring is sunk
    generator compute until structure_refilter converts it. During deep_stall we
    still cap this floor at one slot in _queue_chain_refilter_reserve; this
    predicate only decides whether that one-slot escape path should remain open.
    """
    try:
        backlog = _diagnostic_chain_backlog(
            list(archive.iter_records(ActionCandidate)),
            list(archive.iter_records(ResultRecord)),
            list(archive.iter_records(LaunchDecision)),
            list(archive.iter_records(DispatchRecord)),
        )
    except Exception:  # noqa: BLE001
        return False
    # Allow a minimal evaluation reserve while accepted diagnostic artifacts await
    # scoring. The per-round cap still bounds actual starts.
    if int(backlog.get("total_unscored_diagnostic_artifacts", 0) or 0) > 0:
        return True
    for row in (backlog.get("by_family") or {}).values():
        if (
            isinstance(row, dict)
            and int(row.get("native_strict_like_pending_refilter", 0) or 0) > 0
        ):
            return True
    return False


def _has_pending_score_conversion_refilter(archive) -> bool:
    """Whether accepted diagnostic artifacts still need canonical AF2 scoring.

    This is broader than the deep-stall native-like floor. In the live event
    loop the common case is one GPU freeing at a time; if that single slot is
    always withheld from score conversion, diagnostic generators can look like
    zero-SU failures for hours even though their accepted artifacts were never
    officially scored.
    """
    try:
        backlog = _diagnostic_chain_backlog(
            list(archive.iter_records(ActionCandidate)),
            list(archive.iter_records(ResultRecord)),
            list(archive.iter_records(LaunchDecision)),
            list(archive.iter_records(DispatchRecord)),
        )
    except Exception:  # noqa: BLE001
        return False
    return int(backlog.get("total_unscored_diagnostic_artifacts", 0) or 0) > 0


def _score_conversion_backlog_count(archive, *, prefer_high_value: bool = True) -> int:
    """Count pending score-conversion work for reserve sizing."""
    try:
        backlog = _diagnostic_chain_backlog(
            list(archive.iter_records(ActionCandidate)),
            list(archive.iter_records(ResultRecord)),
            list(archive.iter_records(LaunchDecision)),
            list(archive.iter_records(DispatchRecord)),
        )
    except Exception:  # noqa: BLE001
        return 0
    if prefer_high_value:
        high = int(backlog.get("native_or_proxy_pending_refilter", 0) or 0)
        if high > 0:
            return high
    return int(backlog.get("total_unscored_diagnostic_artifacts", 0) or 0)


def _high_value_score_conversion_pending(archive) -> bool:
    """Whether high-value diagnostic artifacts need canonical AF2 scoring."""
    try:
        backlog = _diagnostic_chain_backlog(
            list(archive.iter_records(ActionCandidate)),
            list(archive.iter_records(ResultRecord)),
            list(archive.iter_records(LaunchDecision)),
            list(archive.iter_records(DispatchRecord)),
        )
    except Exception:  # noqa: BLE001
        return False
    return int(backlog.get("native_or_proxy_pending_refilter", 0) or 0) > 0


def _busy_score_conversion_count(pool) -> int:
    """Currently running canonical score-conversion workers."""
    n = 0
    for slot in pool or []:
        if not getattr(slot, "busy", False):
            continue
        cand = getattr(slot, "cand", None)
        if cand is not None and is_canonical_score_conversion(cand):
            n += 1
    return n


def _score_conversion_feedback_inflight(archive, pool) -> bool:
    """Whether checkpoint monitoring should refresh conversion feedback.

    Canonical scoring is asynchronous plumbing.  Its presence is useful for
    cheap monitoring refreshes, but must not fence LLM planning or generator
    dispatch.
    """
    return (
        _high_value_score_conversion_pending(archive)
        and _busy_score_conversion_count(pool) > 0
    )


def _score_conversion_reserve_room(archive, queue_room: int) -> tuple[int, bool]:
    """Room visible to the score-conversion reserve.

    With >1 available slots, the downstream reserve limiter still keeps at least
    one generation slot. With exactly one slot, allow canonical score conversion
    to take it only when unscored diagnostic artifacts exist; otherwise preserve
    the fresh LLM-planned-action slot.
    """
    if queue_room <= 0:
        return 0, False
    if _has_pending_score_conversion_refilter(archive):
        return queue_room, True
    return max(0, queue_room - 1), False


try:
    _FAMILY_MAX_NOYIELD_KILLS = max(
        1,
        int(
            os.environ.get(
                "TREX_FAMILY_MAX_NOYIELD_KILLS",
                os.environ.get("TREX_BINDCRAFT_MAX_TIMEOUTS", "2"),
            )
        ),
    )
except ValueError:
    _FAMILY_MAX_NOYIELD_KILLS = 2
try:
    _FAMILY_CIRCUIT_MIN_GPU_H = max(
        0.0, float(os.environ.get("TREX_FAMILY_CIRCUIT_MIN_GPU_H", "3.0"))
    )
except ValueError:
    _FAMILY_CIRCUIT_MIN_GPU_H = 3.0


_LAZY_RECHAIN_BUFFER = int(os.environ.get("TREX_LAZY_RECHAIN_BUFFER", "16"))


def _lazy_rechain_stranded_diagnostic_artifacts(
    archive,
    *,
    tick_id: str,
    chain_seq_ref: list[int],
    target_buffer: int,
) -> int:
    """Recover missing evaluation candidates for diagnostic artifacts.

    Top up to target_buffer in source-family round-robin order. This queues work;
    dispatch throttles and round limits still control execution.
    """
    if target_buffer <= 0:
        return 0
    results = list(archive.iter_records(ResultRecord))
    actions = list(archive.iter_records(ActionCandidate))
    dispatches = list(archive.iter_records(DispatchRecord))
    terminal_or_started = {
        d.candidate_id
        for d in dispatches
        if d.status == "started"
        or d.status == "parse_failed"
        or (d.status == "dispatch_failed" and not _is_recoverable_dispatch_failure(d))
    }
    # generator artifacts already officially scored (canonical structure_refilter child)
    refilter_sources = {
        (r.bins or {}).get("refilter_source")
        for r in results
        if _is_canonical_score_conversion_record(r)
    }
    # generator artifacts that ALREADY have a chain_* candidate (any status), and the
    # count of still-pending (not started/failed) chain candidates for buffer sizing.
    have_chain_parent: set[str] = set()
    pending_chain = 0
    for c in actions:
        if not (
            c.candidate_id.startswith("chain_")
            and c.method_family == "structure_refilter"
        ):
            continue
        if c.parent_result_id:
            have_chain_parent.add(c.parent_result_id)
        if c.candidate_id not in terminal_or_started:
            pending_chain += 1
    result_by_id = {r.result_id: r for r in results}
    scored_or_queued_identities: set[tuple[str, str]] = set()
    for rid in refilter_sources | have_chain_parent:
        rec = result_by_id.get(rid)
        if rec is not None:
            scored_or_queued_identities.add(_score_conversion_identity(rec))
    room = target_buffer - pending_chain
    if room <= 0:
        return 0
    by_src: dict[str, list[ResultRecord]] = {}
    for r in results:
        if (
            _record_needs_score_conversion(r.backend_family, r)
            and r.result_id not in refilter_sources
            and r.result_id not in have_chain_parent
            and _score_conversion_identity(r) not in scored_or_queued_identities
        ):
            scored_or_queued_identities.add(_score_conversion_identity(r))
            by_src.setdefault(r.backend_family, []).append(r)
    if not by_src:
        return 0
    from .capability_registry import default_registry as _dr

    sr_cap = _dr().get("structure_refilter")
    if sr_cap is None or sr_cap.availability != "available":
        return 0
    feas = FeasibilityCheck(
        backend_healthy=True,
        runtime_bucket_id="rb_v7",
        compiler_ok=True,
        verifier_ok=True,
        route_cap_ok=True,
        cost_ok=True,
    )
    minted = 0
    active = [s for s in by_src if by_src[s]]
    while minted < room and active:
        nxt: list[str] = []
        for s in active:
            if minted >= room:
                break
            parent_rec = by_src[s].pop(0)
            chain_seq_ref[0] += 1
            child = ActionCandidate(
                candidate_id=(
                    f"chain_{tick_id}_{s}_to_structure_refilter_"
                    f"{chain_seq_ref[0]:03d}"
                ),
                hypothesis_ids=[],
                parent_result_id=parent_rec.result_id,
                method_family="structure_refilter",
                operator_id=sr_cap.default_operator_id,
                lane_id=sr_cap.default_lane_id,
                config_delta={},
                downstream_route_plan=[],
                estimated_cost_class=sr_cap.default_cost_class,
                expected_signal=(
                    f"lazy_rechain:{s}->structure_refilter "
                    f"parent={parent_rec.result_id} role={CANONICAL_SCORE_CONVERSION}"
                ),
                evidence_refs=[parent_rec.result_id],
                feasibility=feas,
                baseline_result_id=parent_rec.result_id,
                refilter_role=CANONICAL_SCORE_CONVERSION,
            )
            archive.append(child)
            minted += 1
            if by_src[s]:
                nxt.append(s)
        active = nxt
    if minted:
        print(
            f"  [lazy_rechain] minted {minted} stranded diagnostic chain "
            f"refilter candidate(s) (pending_buffer={pending_chain}->{pending_chain+minted})",
            flush=True,
        )
    return minted


def _queue_chain_refilter_reserve(
    archive,
    pending: list[str],
    chain_backfill_seen: set[str],
    *,
    tick_id: str,
    queue_room: int,
    source: str,
    why: str,
    throttled: bool = False,
    max_reserve_cap: int | None = None,
    chain_seq_ref: list[int] | None = None,
) -> list[str]:
    """Reserve bounded evaluation work ahead of queued generation.

    Diagnostic artifacts need standardized scoring before they can qualify. Deep-stall
    throttling retains a minimal reserve for promising artifacts without allowing
    evaluation to monopolize workers.
    """
    # Recover missing evaluation candidates before draining; queue creation does not
    # consume a worker slot.
    if chain_seq_ref is not None:
        _lazy_rechain_stranded_diagnostic_artifacts(
            archive,
            tick_id=tick_id,
            chain_seq_ref=chain_seq_ref,
            target_buffer=max(queue_room, _LAZY_RECHAIN_BUFFER),
        )
    if throttled and not _has_native_strict_like_pending_refilter(archive):
        return []
    action_by_id = {c.candidate_id: c for c in archive.iter_records(ActionCandidate)}
    if any(is_canonical_score_conversion(action_by_id.get(cid)) for cid in pending):
        return []
    # Probe the actual eligible backlog (bounded by queue_room — we never reserve
    # more than that) FIRST, then size the reserve to it so a large unscored pile
    # drains adaptively instead of 1/round. Probing does not mutate `seen`.
    available = _chain_escape_first(
        archive,
        _chain_backfill_ids(
            archive,
            chain_backfill_seen,
            queue_room,
            admission_tick_id=tick_id,
        ),
    )
    reserve_n = _chain_refilter_reserve_limit(queue_room, backlog_n=len(available))
    if throttled:
        reserve_n = min(1, reserve_n)
    # Per-ROUND cap: the caller passes how many reserves remain in this round's
    # budget so predispatch + refill reserves cannot COMPOUND to fill the whole
    # pool and starve fresh generation (the gen-slot guarantee is otherwise only
    # per-call). None leaves the per-round reservation uncapped.
    if max_reserve_cap is not None:
        reserve_n = min(reserve_n, max(0, max_reserve_cap))
    reserve_n = _chain_refilter_recent_share_cap(archive, reserve_n, available)
    ids = available[:reserve_n]
    if not ids:
        return []
    pending[0:0] = ids
    chain_backfill_seen.update(ids)
    _append_chain_launch_decisions(
        archive,
        ids,
        tick_id=tick_id,
        source=source,
        why=why,
    )
    return ids


def _cancel_pending_chain_reserves_for_deep_stall(
    archive,
    pending: list[str],
    reserved_ids: list[str],
    *,
    tick_id: str,
) -> list[str]:
    """Cancel not-yet-dispatched score-conversion reserves on new deep_stall.

    Reserve candidates are queued before the current tick's EvidenceSummary is
    computed so the controller can keep GPUs busy. If that same tick newly
    diagnoses ``deep_stall``, those chain refilters should not be dispatched in
    postdispatch. Mark them terminal as dispatch_failed so restart recovery does
    not resurrect work the current evidence explicitly throttled.
    """
    if not reserved_ids:
        return []
    reserved = set(reserved_ids)
    cancelled: list[str] = []
    kept: list[str] = []
    for cid in pending:
        if cid in reserved and cid.startswith("chain_"):
            cancelled.append(cid)
        else:
            kept.append(cid)
    if not cancelled:
        return []
    pending[:] = kept
    for cid in cancelled:
        launch_decision = _latest_launch_decision(archive, cid)
        archive.append(
            DispatchRecord(
                dispatch_id=(f"deep_stall_throttle_{tick_id}_{_short_id(cid)}"),
                launch_id=(
                    launch_decision.launch_id if launch_decision is not None else None
                ),
                tick_id=tick_id,
                candidate_id=cid,
                status="dispatch_failed",
                worker_slot=None,
                gpu_id=None,
                output_dir=None,
                parent_result_id=None,
                parent_pdb_path=None,
                **_dispatch_record_metadata(
                    archive, candidate_id=cid, launch_decision=launch_decision
                ),
                attempt=1,
                why=(
                    "current tick state=deep_stall; cancelled deterministic "
                    "diagnostic score-conversion reserve before dispatch"
                ),
            )
        )
    return cancelled


def _should_pause_initial_no_evidence_prefetch(
    *,
    completed_children: int,
    busy_count: int,
    pool_size: int,
) -> bool:
    """Avoid blind scientific prefetch before the first worker result.

    The first tick already launches the deterministic root-coverage warmstart.
    While all worker slots are occupied and no ResultRecord exists yet, extra
    Planner/Supervisor calls only see the same low-evidence prompt and tend to
    queue cheap Complexa variants. Those queued candidates can later run under
    the soft stale policy before real feedback is incorporated. Pause refill
    until the first completed worker result creates actual evidence.
    """
    return (
        int(completed_children or 0) <= 0
        and int(busy_count or 0) >= int(pool_size or 0) > 0
    )


def _drop_stale_scientific_pending(
    archive,
    pending: list[str],
    *,
    current_completed_children: int | None,
    current_run_su_count: int | None,
    current_state_label: str | None,
    tick_id: str,
) -> list[str]:
    """Drop queued jobs when their eligibility is invalidated.

    The default soft policy retains jobs after ordinary evidence growth.
    TREX_PREFETCH_STALE_POLICY=hard also invalidates them on completed-count, SU-count,
    or state changes.
    """
    policy = os.environ.get("TREX_PREFETCH_STALE_POLICY", "soft").strip().lower()
    hard_policy = policy in {"hard", "strict", "count", "counts"}

    dropped: list[str] = []
    kept: list[str] = []
    for cid in pending:
        if cid.startswith("chain_"):
            kept.append(cid)
            continue

        launch_decision = _latest_launch_decision(archive, cid)
        meta = (
            getattr(launch_decision, "resource_class_concrete", None)
            if launch_decision is not None
            else None
        ) or {}
        planned_children = meta.get("planned_completed_children")
        planned_su = meta.get("planned_run_su_count")
        planned_state = meta.get("planned_state_label")
        planned_mode = meta.get("mode")

        stale = False
        stale_reasons: list[str] = []
        # Soft policy keeps ordinary selected work aligned with GPU starts, but
        # the initial cold-start no-evidence prefetch is different: it was chosen
        # before any worker result existed. Once the first result arrives, replan
        # from real evidence instead of starting blind queued Complexa variants.
        if (
            planned_children is not None
            and current_completed_children is not None
            and int(planned_children) <= 0
            and int(current_completed_children) > 0
            and str(planned_state or "") == "low_evidence"
        ):
            stale = True
            stale_reasons.append(
                f"initial_no_evidence_prefetch completed_children {planned_children}->{current_completed_children}"
            )
        if (
            hard_policy
            and planned_children is not None
            and current_completed_children is not None
        ):
            if int(planned_children) != int(current_completed_children):
                stale = True
                stale_reasons.append(
                    f"completed_children {planned_children}->{current_completed_children}"
                )
        if hard_policy and planned_su is not None and current_run_su_count is not None:
            if int(planned_su) != int(current_run_su_count):
                stale = True
                stale_reasons.append(
                    f"run_su_count {planned_su}->{current_run_su_count}"
                )
        if (
            hard_policy
            and planned_state is not None
            and current_state_label is not None
        ):
            if str(planned_state) != str(current_state_label):
                stale = True
                stale_reasons.append(f"state {planned_state}->{current_state_label}")

        if not stale:
            kept.append(cid)
            continue

        dropped.append(cid)
        archive.append(
            DispatchRecord(
                dispatch_id=f"stale_prefetch_{tick_id}_{_short_id(cid)}",
                launch_id=(
                    launch_decision.launch_id if launch_decision is not None else None
                ),
                tick_id=tick_id,
                candidate_id=cid,
                status="dispatch_failed",
                worker_slot=None,
                gpu_id=None,
                output_dir=None,
                parent_result_id=None,
                parent_pdb_path=None,
                **_dispatch_record_metadata(
                    archive, candidate_id=cid, launch_decision=launch_decision
                ),
                attempt=1,
                why=(
                    "stale scientific prefetch: "
                    + ("; ".join(stale_reasons) if stale_reasons else "invalidated")
                    + f" (policy={policy})"
                ),
            )
        )

    if dropped:
        pending[:] = kept
    return dropped


def _dispatch_pending_to_free_slots(
    pool,
    pending,
    cand_by_id,
    *,
    archive,
    target,
    target_pdb,
    archive_root,
    round_id,
    dispatch_fn,
    dispatch_retries: dict[str, int] | None = None,
    dispatch_defer_seen: set[str] | None = None,
) -> int:
    """Compatibility wrapper for the extracted dispatch queue state machine."""

    dependencies = DispatchQueueDependencies(
        prioritize_pending_queue=_prioritize_pending_queue,
        high_cost_defer_reason=_high_cost_dispatch_defer_reason,
        latest_launch_decision=_latest_launch_decision,
        dispatch_record_metadata=_dispatch_record_metadata,
        short_id=_short_id,
        process_start_ticks=_pid_start_ticks,
    )
    return dispatch_pending_to_free_slots(
        worker_slots=pool,
        pending_candidate_ids=pending,
        candidates_by_id=cand_by_id,
        archive=archive,
        target=target,
        target_pdb=target_pdb,
        archive_root=archive_root,
        round_id=round_id,
        dispatch_candidate=dispatch_fn,
        dependencies=dependencies,
        retry_counts=dispatch_retries,
        deferred_audits=dispatch_defer_seen,
        max_dispatch_retries=_MAX_DISPATCH_RETRIES,
    )


def _require_foldseek_available(binary: str = "foldseek") -> str:
    # Fail fast when official Foldseek-deduped SU cannot be computed.
    path = shutil.which(binary)
    if path:
        return path
    raise SystemExit(
        "[v7_controller][FATAL] Foldseek binary not found on PATH. "
        "Official SU requires Foldseek-deduped bins and has no per-result fallback. "
        f"Requested binary={binary!r}."
    )


def _require_p2_runtime_available(
    *,
    require_af2: bool,
    require_proteinmpnn: bool,
    timeout_s: float = 90.0,
    runtime_paths: RuntimePaths | None = None,
) -> str | None:
    """Fail before worker dispatch if the shared AF2/ProteinMPNN venv is unusable."""
    if not (require_af2 or require_proteinmpnn):
        return None
    paths = runtime_paths or RUNTIME_PATHS
    if not paths.complexa_python.is_file() or not os.access(
        paths.complexa_python,
        os.X_OK,
    ):
        raise SystemExit(
            f"[v7_controller][FATAL] AF2/ProteinMPNN Python is not executable: "
            f"{paths.complexa_python}"
        )
    if require_af2 and not paths.af2_data_dir.is_dir():
        raise SystemExit(
            f"[v7_controller][FATAL] AF2 parameter directory is missing: "
            f"{paths.af2_data_dir}"
        )
    if require_proteinmpnn:
        weights = paths.proteinmpnn_weights
        if not weights.is_file():
            raise SystemExit(
                f"[v7_controller][FATAL] ProteinMPNN weights are missing: {weights}"
            )

    modules = ["jax"]
    if require_af2:
        modules.append("community_models.colabdesign")
    if require_proteinmpnn:
        modules.append("torch")
    probe = (
        "import importlib,sys;"
        f"mods={modules!r};"
        "[importlib.import_module(m) for m in mods];"
        "print('executable='+sys.executable);"
        "print('prefix='+sys.prefix);"
        "print('modules='+','.join(mods))"
    )
    env = _isolated_env_for_subprocess(paths.complexa_python.parent.parent)
    env["PYTHONPATH"] = f"{paths.repo_root}:" + env.get("PYTHONPATH", "")
    # Imports exercise environment activation without reserving a worker GPU.
    env["CUDA_VISIBLE_DEVICES"] = ""
    try:
        completed = subprocess.run(
            [str(paths.complexa_python), "-c", probe],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(
            f"[v7_controller][FATAL] AF2/ProteinMPNN runtime preflight failed "
            f"for {paths.complexa_python}: {type(exc).__name__}: {exc}"
        ) from exc
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        raise SystemExit(
            f"[v7_controller][FATAL] AF2/ProteinMPNN runtime preflight exited "
            f"{completed.returncode} for {paths.complexa_python}:\n"
            f"{output[-4000:]}"
        )
    return output


def _with_runtime_tool_paths(
    config: LiveTickConfig,
    runtime_paths: RuntimePaths,
) -> LiveTickConfig:
    """Bind resolved Foldseek/MMseqs commands to one live-tick config."""

    return dc_replace(
        config,
        foldseek=dc_replace(
            config.foldseek,
            binary=runtime_paths.foldseek_command,
        ),
        sequence_dedup=dc_replace(
            config.sequence_dedup,
            binary=runtime_paths.mmseqs_command,
        ),
    )


def _resolve_worker_hard_ceiling_seconds(method_family: str) -> float:
    """Resolve one built-in or extension backend's hang backstop."""

    extension_adapter = get_backend_adapter(method_family)
    return float(
        HARD_CEILING_S.get(
            method_family,
            extension_adapter.hard_ceiling_seconds
            if extension_adapter is not None
            else _DEFAULT_HARD_CEILING_S,
        )
    )


def _worker_supervision_dependencies() -> WorkerSupervisionDependencies:
    """Bind controller parsing and process operations to lifecycle supervision."""

    return WorkerSupervisionDependencies(
        parse_worker_output=_parse_and_archive_worker_output,
        record_parse_failure=_append_parse_failed_dispatch_record,
        parse_incremental_bindcraft_output=_archive_incremental_bindcraft_output,
        count_bindcraft_scoreable_results=_bindcraft_scoreable_final_count,
        resolve_hard_ceiling_seconds=_resolve_worker_hard_ceiling_seconds,
        terminate_worker_process=_terminate_process_group,
    )


def _family_circuit_breaker_dependencies() -> FamilyCircuitBreakerDependencies:
    """Bind lineage attribution to the run-level family breaker."""

    return FamilyCircuitBreakerDependencies(
        index_actions_by_spawned_result=_index_actions_by_spawned_result,
        is_family_circuit_broken=_family_circuit_broken,
    )


def _progress_checkpoint_dependencies() -> ProgressCheckpointDependencies:
    """Bind controller evidence callbacks to durable progress reporting."""

    return ProgressCheckpointDependencies(
        score_conversion_feedback_inflight=_score_conversion_feedback_inflight,
        summarize_pending_family_load=_pending_family_load_summary,
        run_live_tick=run_live_tick,
        charged_gpu_count=_checkpoint_charged_gpu_count,
        build_cheap_evidence=_cheap_checkpoint_evidence,
        write_checkpoint=_write_controller_checkpoint,
    )


def main(
    argv: list[str] | None = None,
    *,
    runtime_paths: RuntimePaths | None = None,
    loop_config: ControllerLoopConfig | None = None,
) -> None:
    args = parse_controller_args(argv)
    active_loop_config = loop_config or ControllerLoopConfig()
    active_runtime_paths = runtime_paths or RuntimePaths.from_environment(
        default_repo_root=_REPO_ROOT
    )

    global RUN_SEED
    RUN_SEED = int(args.seed)
    print(f"[v7_controller] run_seed={RUN_SEED}", flush=True)
    print(
        "[v7_controller] runtime_paths="
        + json.dumps(active_runtime_paths.as_dict(), sort_keys=True),
        flush=True,
    )

    archive = Archive(args.archive_root)
    # Keep the descriptor alive for the duration of main(). The OS releases the
    # flock on every exit path, including exceptions and scheduler termination.
    _controller_lock_fd = _acquire_controller_lock(args.archive_root)
    assert _controller_lock_fd >= 0
    tgt_data = json.loads(args.target_constraint.read_text())
    target = TargetConstraint(**tgt_data)

    # Bind output chain verification to the configured target reference.
    global TARGET_PDB_PATH, TARGET_RES_COUNT, TARGET_CHAIN_IDS
    TARGET_PDB_PATH = Path(args.target_pdb).resolve()
    TARGET_CHAIN_IDS = tuple(target.chain_ids)
    _tcounts = _pdb_residues_per_chain(Path(args.target_pdb))
    TARGET_RES_COUNT = max(_tcounts.values()) if _tcounts else None
    print(
        f"[v7_controller] target_res_count={TARGET_RES_COUNT} "
        f"target_chains={_csv_chains(TARGET_CHAIN_IDS)} (pdb_chains={_tcounts})",
        flush=True,
    )

    # One controller process serves one target. Reject archives containing another
    # target to avoid mixing evidence.
    foreign_targets = {r.target_id for r in archive.iter_records(ResultRecord)} - {
        target.target_id
    }
    if foreign_targets:
        raise SystemExit(
            f"[v7_controller] archive {args.archive_root} contains foreign "
            f"target_ids {sorted(foreign_targets)} — refuse to run multi-target"
        )

    # Limit the available registry to the explicitly enabled families.
    from .capability_registry import default_registry

    backend_registry = default_registry()
    all_backend_families = tuple(backend_registry.capabilities.keys())
    if args.enabled_families.strip():
        whitelist = {f.strip() for f in args.enabled_families.split(",") if f.strip()}
        # Diagnostic-only families require standardized evaluation. Enable
        # structure_refilter when needed so their outputs can qualify.
        diagnostic_native_families = {
            family
            for family in whitelist
            if (capability := backend_registry.get(family)) is not None
            and getattr(capability, "outputs_diagnostic_only", False)
            and getattr(capability, "role", "generator")
            in ("generator", "seq_redesign", "refilter")
        }
        if diagnostic_native_families and "structure_refilter" not in whitelist:
            print(
                "[v7_controller][WARN] diagnostic-only families "
                f"{sorted(diagnostic_native_families)} "
                f"need structure_refilter to yield official strict/SU (their artifacts "
                f"auto-chain through it); it was missing from --enabled-families → "
                f"auto-enabling structure_refilter to avoid a dead lane.",
                flush=True,
            )
            whitelist.add("structure_refilter")
        configured_unavailable_families: tuple[str, ...] = tuple(
            family for family in all_backend_families if family not in whitelist
        )
        print(f"[v7_controller] enabled families: {sorted(whitelist)}", flush=True)
        print(
            "[v7_controller] disabled by --enabled-families: "
            f"{sorted(configured_unavailable_families)}",
            flush=True,
        )
    else:
        configured_unavailable_families = ()
    worker_gpus = [g.strip() for g in args.worker_gpus.split(",") if g.strip()]
    if not worker_gpus:
        worker_gpus = ["1", "2", "3"]
    if len(set(worker_gpus)) != len(worker_gpus):
        raise SystemExit(
            "[v7_controller][FATAL] TREX_WORKER_GPUS must contain unique GPU identifiers"
        )
    base_cfg = _with_runtime_tool_paths(
        LiveTickConfig(),
        active_runtime_paths,
    )
    foldseek_cfg = dc_replace(
        base_cfg.foldseek,
        min_tm_score=float(args.foldseek_su_tm_score),
        collapse_tm_score=float(args.foldseek_collapse_tm_score),
    )
    selector_cfg = dc_replace(
        base_cfg.selector,
        quota_realization=args.selector_quota_realization,
        mode_window_k=int(args.selector_mode_window_k),
        adaptive_mode_window_k=bool(args.selector_adaptive_mode_window_k),
    )
    cfg = dc_replace(
        base_cfg,
        planner=dc_replace(
            base_cfg.planner, base_url=args.vllm_base_url, model=args.llm_model
        ),
        supervisor=dc_replace(
            base_cfg.supervisor, base_url=args.vllm_base_url, model=args.llm_model
        ),
        critic=dc_replace(
            base_cfg.critic,
            base_url=args.vllm_base_url,
            model=args.llm_model,
            enabled=bool(args.enable_critic),
        ),
        selector=selector_cfg,
        foldseek=foldseek_cfg,
        builder=dc_replace(
            base_cfg.builder,
            unavailable_backends_override=configured_unavailable_families,
        ),
        skip=dc_replace(base_cfg.skip, enabled=bool(args.enable_evidence_skip)),
        enable_exemplars=bool(args.enable_exemplars),
        worker_wall_gpu_count=len(worker_gpus),
    )
    enabled_backend_families = set(all_backend_families) - set(
        configured_unavailable_families
    )
    p2_runtime = _require_p2_runtime_available(
        require_af2="structure_refilter" in enabled_backend_families,
        require_proteinmpnn="proteinmpnn_redesign" in enabled_backend_families,
        runtime_paths=active_runtime_paths,
    )
    if p2_runtime:
        print(
            "[v7_controller] AF2/ProteinMPNN runtime preflight=ok "
            + " | ".join(p2_runtime.splitlines()),
            flush=True,
        )
    foldseek_path = _require_foldseek_available(cfg.foldseek.binary)
    print(f"[v7_controller] foldseek_binary={foldseek_path}", flush=True)
    print(
        f"[v7_controller] exemplars={'on' if args.enable_exemplars else 'off'}",
        flush=True,
    )
    print(
        f"[v7_controller] evidence_skip={'on' if args.enable_evidence_skip else 'off'}",
        flush=True,
    )
    print(
        f"[v7_controller] selector_quota_realization={cfg.selector.quota_realization} "
        f"mode_window_k={cfg.selector.mode_window_k} "
        f"adaptive_mode_window_k={int(cfg.selector.adaptive_mode_window_k)}",
        flush=True,
    )
    print(
        f"[v7_controller] foldseek_su_tm={cfg.foldseek.min_tm_score:.2f} "
        f"foldseek_collapse_tm={cfg.foldseek.collapse_tm_score:.2f} "
        "structure_scope=binder_chain alignment_type=1 "
        "min_seq_id=0.0 cov_mode=0 threads=1",
        flush=True,
    )

    round_id, elapsed_offset_h = _archive_resume_state(
        archive, target_id=target.target_id
    )
    controller_started_monotonic = time.monotonic()
    # Keep selected jobs queued so a free worker can start while planning replenishes
    # the queue.
    pending: list[str] = []
    chain_backfill_seen: set[
        str
    ] = set()  # Evaluation candidates already queued for idle-slot backfill.
    dispatch_retries: dict[
        str, int
    ] = {}  # Transient dispatch retry counts per candidate.
    dispatch_defer_seen: set[
        str
    ] = set()  # One audit row per candidate and temporary capacity limit.
    # Restore campaign state so resumed dispatch applies the current evaluation
    # throttle.
    prior_evidence_summaries = list(archive.iter_records(EvidenceSummary))
    latest_state: str | None = (
        prior_evidence_summaries[-1].state_label if prior_evidence_summaries else None
    )
    state_probe_cache_key: tuple[
        int, int, int, int, tuple[tuple[str, int, int], ...]
    ] | None = None
    state_probe_cache_at = 0.0
    state_probe_cache_summary: dict | None = None
    state_probe_cache_ttl_seconds = active_loop_config.state_probe_cache_seconds
    progress_checkpoint_at = 0.0
    progress_checkpoint_interval_seconds = (
        active_loop_config.progress_checkpoint_interval_seconds
    )
    resume_checkpoint = _read_controller_checkpoint(archive)
    if (
        resume_checkpoint is not None
        and str(resume_checkpoint.get("target_id")) == target.target_id
    ):
        checkpoint_state = (resume_checkpoint.get("evidence") or {}).get("state_label")
        if checkpoint_state:
            latest_state = str(checkpoint_state)
    print(
        f"[v7_controller] target={target.target_id} start={time.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        flush=True,
    )
    print(f"[v7_controller] archive={args.archive_root}", flush=True)
    print(f"[v7_controller] vLLM={args.vllm_base_url}", flush=True)
    print(
        f"[v7_controller] worker_gpus={worker_gpus} "
        f"worker_wall_gpu_count={len(worker_gpus)}",
        flush=True,
    )
    if round_id or elapsed_offset_h:
        print(
            f"[v7_controller] resumed archive at round={round_id} "
            f"elapsed_wall_h={elapsed_offset_h:.3f}",
            flush=True,
        )

    # Reap each completed worker independently and refill free slots while other jobs
    # continue.
    pool = [_WorkerSlot(slot_id=i, gpu_id=g) for i, g in enumerate(worker_gpus)]
    dispatch_candidate = partial(
        _dispatch_candidate_to_gpu,
        runtime_paths=active_runtime_paths,
    )
    worker_supervision = _worker_supervision_dependencies()
    circuit_breaker_dependencies = _family_circuit_breaker_dependencies()
    progress_checkpoint_dependencies = _progress_checkpoint_dependencies()
    poll_interval_s = active_loop_config.poll_interval_seconds
    chain_seq_ref = [0]  # mutable counter for auto_chain candidate_ids
    recovered_pending = _recover_undispatched_launch_ids(
        archive,
        chain_backfill_seen=chain_backfill_seen,
    )
    if recovered_pending:
        pending.extend(recovered_pending)
        print(
            f"[v7_controller] recovered {len(recovered_pending)} "
            f"selected-but-not-dispatched candidate(s) from archive",
            flush=True,
        )
    blocked_families: set[str] = set()
    completed_iterations = 0
    while True:
        if (
            active_loop_config.maximum_iterations is not None
            and completed_iterations >= active_loop_config.maximum_iterations
        ):
            print(
                "[v7_controller] bounded loop completed "
                f"{completed_iterations} iteration(s); draining workers",
                flush=True,
            )
            break
        completed_iterations += 1
        elapsed_h = (
            elapsed_offset_h
            + (time.monotonic() - controller_started_monotonic) / 3600.0
        )
        if elapsed_h >= args.max_wall_h:
            print(
                f"[v7_controller] max wall reached ({elapsed_h:.2f} h); "
                f"draining remaining workers and exit",
                flush=True,
            )
            break

        # 0. Disable persistently unproductive families using archived evidence.
        circuit_breaker_result = evaluate_family_circuit_breakers(
            FamilyCircuitBreakerRequest(
                archive=archive,
                all_families=all_backend_families,
                unavailable_families=configured_unavailable_families,
                already_blocked_families=frozenset(blocked_families),
                pending_candidate_ids=pending,
                backend_registry=backend_registry,
                max_no_yield_timeouts=_FAMILY_MAX_NOYIELD_KILLS,
                minimum_gpu_hours=_FAMILY_CIRCUIT_MIN_GPU_H,
            ),
            circuit_breaker_dependencies,
        )
        if circuit_breaker_result.newly_blocked_families:
            blocked_families = set(circuit_breaker_result.blocked_families)
            pending[:] = circuit_breaker_result.pending_candidate_ids
            unavailable_families = tuple(
                sorted(set(configured_unavailable_families) | blocked_families)
            )
            cfg = dc_replace(
                cfg,
                builder=dc_replace(
                    cfg.builder,
                    unavailable_backends_override=unavailable_families,
                ),
            )
            for blocked_family in circuit_breaker_result.newly_blocked_families:
                print(
                    f"  [circuit_breaker] {blocked_family} disabled after >= "
                    f"{_FAMILY_MAX_NOYIELD_KILLS} no-yield kills, "
                    f">={_FAMILY_CIRCUIT_MIN_GPU_H:.1f} GPU-h, and 0 "
                    "strict/near-miss; removed from planner/builder + pending "
                    "for the rest of the run",
                    flush=True,
                )

        # 1. Reap completed slots and salvage timed-out workers.
        supervise_worker_slots(
            pool,
            archive=archive,
            target=target,
            auto_chain_sequence=chain_seq_ref,
            dependencies=worker_supervision,
        )

        # Persist wall progress between scientific ticks without consuming an
        # LLM call or adding checkpoint-only rows to scientific history.
        checkpoint_result = refresh_progress_checkpoint(
            ProgressCheckpointRequest(
                archive=archive,
                target=target,
                worker_slots=pool,
                pending_candidate_ids=pending,
                live_tick_config=cfg,
                round_id=round_id,
                elapsed_wall_hours=elapsed_h,
                maximum_wall_hours=args.max_wall_h,
                elapsed_offset_hours=elapsed_offset_h,
                controller_started_monotonic=controller_started_monotonic,
                worker_gpu_count=len(worker_gpus),
                latest_state=latest_state,
                last_checkpoint_at=progress_checkpoint_at,
                interval_seconds=progress_checkpoint_interval_seconds,
            ),
            progress_checkpoint_dependencies,
        )
        progress_checkpoint_at = checkpoint_result.last_checkpoint_at
        latest_state = checkpoint_result.latest_state

        # Dispatch queued jobs before planning so free workers do not wait for LLM
        # calls.
        free_slots = sum(1 for s in pool if not s.busy)
        # Share one evaluation-reserve budget across predispatch and refill, leaving
        # capacity for generation.
        round_backlog_n = max(
            len(pool),
            _score_conversion_backlog_count(archive, prefer_high_value=True),
        )
        round_reserve_budget = _chain_refilter_reserve_limit(
            len(pool),
            backlog_n=round_backlog_n,
        )
        reserved_this_round = 0
        if free_slots > 0:
            # Refresh campaign state before reserving evaluation slots. This
            # evidence-only probe neither archives a summary nor calls the LLMs.
            _probe_now = time.time()
            _probe_inflight_gpu_h = sum(
                max(0.0, (_probe_now - s.launched_at) / 3600.0)
                for s in pool
                if s.busy and s.launched_at > 0
            )
            _probe_tick_id = f"v7probe{max(round_id + 1, 1):03d}"
            try:
                _probe_result_count = sum(
                    1
                    for r in archive.iter_records(ResultRecord)
                    if r.target_id == target.target_id
                )
                _probe_busy_count = sum(1 for s in pool if s.busy)
                _probe_cand_by_id = {
                    c.candidate_id: c for c in archive.iter_records(ActionCandidate)
                }
                _probe_pending_family_load = _pending_family_load_summary(
                    pool, pending, _probe_cand_by_id
                )
                # The evidence-only state probe includes in-flight worker GPU-h
                # in dry-timer classification. Bucket it coarsely so the cache
                # still avoids churn, but cannot hide a stalled/deep-stall
                # transition while long workers keep running.
                _probe_inflight_bucket = int(_probe_inflight_gpu_h * 4.0)
                _probe_key = (
                    _probe_result_count,
                    _probe_busy_count,
                    free_slots,
                    _probe_inflight_bucket,
                    _pending_load_cache_key(_probe_pending_family_load),
                )
                if (
                    state_probe_cache_summary is not None
                    and state_probe_cache_key == _probe_key
                    and (_probe_now - state_probe_cache_at)
                    < state_probe_cache_ttl_seconds
                ):
                    _probe_summary = state_probe_cache_summary
                else:
                    _probe_summary = run_live_tick(
                        archive,
                        target,
                        tick_id=_probe_tick_id,
                        tick_id_int=round_id + 1,
                        elapsed_wall_h=elapsed_h,
                        remaining_wall_h=max(0.0, args.max_wall_h - elapsed_h),
                        pending_children=sum(1 for s in pool if s.busy),
                        cfg=cfg,
                        available_slots_override=max(1, free_slots),
                        inflight_gpu_h=_probe_inflight_gpu_h,
                        pending_family_load=_probe_pending_family_load,
                        evidence_only=True,
                    )
                    state_probe_cache_key = _probe_key
                    state_probe_cache_at = _probe_now
                    state_probe_cache_summary = _probe_summary
                latest_state = _probe_summary.get("evidence", {}).get(
                    "state_label", latest_state
                )
                _probe_evidence = _probe_summary.get("evidence", {})
                if latest_state == "deep_stall":
                    _stale_chain_ids = [
                        cid for cid in pending if cid.startswith("chain_")
                    ]
                    if (
                        _has_native_strict_like_pending_refilter(archive)
                        and _stale_chain_ids
                    ):
                        _stale_chain_ids = _stale_chain_ids[1:]
                    cancelled = _cancel_pending_chain_reserves_for_deep_stall(
                        archive, pending, _stale_chain_ids, tick_id=_probe_tick_id
                    )
                    if cancelled:
                        print(
                            f"  [chain_refilter_predispatch] cancelled "
                            f"{len(cancelled)} stale queued reserve(s) after "
                            f"state probe=deep_stall",
                            flush=True,
                        )
                dropped = _drop_stale_scientific_pending(
                    archive,
                    pending,
                    current_completed_children=_probe_evidence.get("n_all_results"),
                    current_run_su_count=_probe_evidence.get("run_su_count"),
                    current_state_label=_probe_evidence.get("state_label"),
                    tick_id=_probe_tick_id,
                )
                if dropped:
                    print(
                        f"  [stale_prefetch] dropped {len(dropped)} stale "
                        "scientific pending candidate(s); will replan from "
                        "current evidence",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [state_probe_err] {type(exc).__name__}: {exc}; "
                    f"using previous state={latest_state}",
                    flush=True,
                )
        # Apply the deep-stall cap to the combined predispatch and refill reserve using
        # the refreshed state.
        if latest_state == "deep_stall":
            round_reserve_budget = min(round_reserve_budget, 1)
        if free_slots > 0:
            # Usually leave the sole newly-freed worker for a fresh
            # LLM/Selector decision. Exception: if accepted diagnostic artifacts
            # are still unscored, canonical score conversion may use that single
            # slot so route evidence is not poisoned as a false zero-SU failure.
            (
                predispatch_reserve_room,
                predispatch_score_slot,
            ) = _score_conversion_reserve_room(archive, free_slots)
            predispatch_reserved = _queue_chain_refilter_reserve(
                archive,
                pending,
                chain_backfill_seen,
                tick_id=f"v7r{max(round_id, 1):03d}",
                queue_room=predispatch_reserve_room,
                source="chain_refilter_predispatch",
                why=(
                    "reserved diagnostic score-conversion lane before stale "
                    "prefetch work: accepted diagnostic artifact needs "
                    "canonical structure_refilter before strict/SU credit"
                ),
                throttled=(latest_state == "deep_stall"),
                max_reserve_cap=max(
                    0,
                    round_reserve_budget
                    - reserved_this_round
                    - _busy_score_conversion_count(pool),
                ),
                chain_seq_ref=chain_seq_ref,
            )
            reserved_this_round += len(predispatch_reserved)
            if free_slots > 0 and predispatch_reserve_room == 0:
                print(
                    "  [chain_refilter_predispatch] left single free slot for "
                    "fresh LLM-planned action before chain reserve",
                    flush=True,
                )
            elif predispatch_score_slot and free_slots == 1:
                print(
                    "  [chain_refilter_predispatch] single free slot allowed "
                    "for pending diagnostic score conversion",
                    flush=True,
                )
            if predispatch_reserved:
                print(
                    f"  [chain_refilter_predispatch] prioritized "
                    f"{len(predispatch_reserved)} chain refilter(s) ahead of "
                    f"stale pending work",
                    flush=True,
                )
        _cand_by_id = {c.candidate_id: c for c in archive.iter_records(ActionCandidate)}
        try:
            dispatched_pre = _dispatch_pending_to_free_slots(
                pool,
                pending,
                _cand_by_id,
                archive=archive,
                target=target,
                target_pdb=args.target_pdb,
                archive_root=args.archive_root,
                round_id=round_id,
                dispatch_fn=dispatch_candidate,
                dispatch_retries=dispatch_retries,
                dispatch_defer_seen=dispatch_defer_seen,
            )
        except Exception as exc:  # noqa: BLE001
            dispatched_pre = 0
            print(
                f"  [dispatch_loop_err][predispatch] {type(exc).__name__}: {exc}",
                flush=True,
            )

        # 3. Refill the queue to ~one pool's worth so the NEXT freed slot has
        # work ready. This planning overlaps the workers dispatched in step 2.
        n_busy = sum(1 for s in pool if s.busy)
        queue_room = max(0, len(pool) - len(pending))
        planned = 0
        chain_reserved = 0
        if queue_room > 0:
            try:
                _completed_for_prefetch = sum(
                    1
                    for r in archive.iter_records(ResultRecord)
                    if r.target_id == target.target_id
                )
            except Exception:  # noqa: BLE001
                _completed_for_prefetch = 0
            if _should_pause_initial_no_evidence_prefetch(
                completed_children=_completed_for_prefetch,
                busy_count=n_busy,
                pool_size=len(pool),
            ):
                queue_room = 0
        if queue_room > 0:
            round_id += 1
            tick_id = f"v7r{round_id:03d}"
            # Leave at least one just-freed slot for the fresh LLM-planned
            # action unless diagnostic score-conversion feedback itself is
            # pending. In the one-slot event-driven case, scoring an accepted
            # diagnostic artifact is often the fastest way to update the next
            # scientific decision with true SU/GPU-h evidence.
            pre_reserve_room, reserve_score_slot = _score_conversion_reserve_room(
                archive, queue_room
            )
            reserved_ids = _queue_chain_refilter_reserve(
                archive,
                pending,
                chain_backfill_seen,
                tick_id=tick_id,
                queue_room=pre_reserve_room,
                source="chain_refilter_reserve",
                why=(
                    "reserved diagnostic score-conversion lane: accepted "
                    "diagnostic artifact needs canonical structure_refilter "
                    "before strict/SU credit"
                ),
                throttled=(latest_state == "deep_stall"),
                max_reserve_cap=max(
                    0,
                    round_reserve_budget
                    - reserved_this_round
                    - _busy_score_conversion_count(pool),
                ),
                chain_seq_ref=chain_seq_ref,
            )
            reserved_this_round += len(reserved_ids)
            chain_reserved = len(reserved_ids)
            if queue_room > 0 and pre_reserve_room == 0:
                print(
                    "  [chain_refilter_reserve] left single free slot for "
                    "fresh LLM-planned action before chain backfill",
                    flush=True,
                )
            elif reserve_score_slot and queue_room == 1:
                print(
                    "  [chain_refilter_reserve] single refill slot allowed "
                    "for pending diagnostic score conversion",
                    flush=True,
                )
            if chain_reserved:
                print(
                    f"  [chain_refilter_reserve] queued {chain_reserved} "
                    f"chain refilter(s) before generator planning/pending work",
                    flush=True,
                )
            refill_n = max(0, len(pool) - len(pending))
            print(
                f"\n[v7_controller] ===== round {round_id} (elapsed {elapsed_h:.2f} h, "
                f"busy={n_busy}/{len(pool)}, queued={len(pending)}, refill={refill_n}) =====",
                flush=True,
            )
            if refill_n > 0:
                # Include running-worker compute in the time since the last new SU; idle
                # workers contribute zero.
                _now = time.time()
                inflight_gpu_h = sum(
                    max(0.0, (_now - s.launched_at) / 3600.0)
                    for s in pool
                    if s.busy and s.launched_at > 0
                )
                try:
                    _cand_by_id_for_load = {
                        c.candidate_id: c for c in archive.iter_records(ActionCandidate)
                    }
                    _pending_family_load = _pending_family_load_summary(
                        pool, pending, _cand_by_id_for_load
                    )
                    summary = run_live_tick(
                        archive,
                        target,
                        tick_id=tick_id,
                        tick_id_int=round_id,
                        elapsed_wall_h=elapsed_h,
                        remaining_wall_h=max(0.0, args.max_wall_h - elapsed_h),
                        pending_children=n_busy,
                        cfg=cfg,
                        available_slots_override=refill_n,
                        inflight_gpu_h=inflight_gpu_h,
                        pending_family_load=_pending_family_load,
                    )
                except Exception as e:  # noqa: BLE001
                    print(
                        f"  [live_tick exception] {type(e).__name__}: {e}", flush=True
                    )
                    time.sleep(60)
                    continue
                # Carry the returned campaign state into subsequent evaluation
                # throttling.
                latest_state = summary.get("evidence", {}).get(
                    "state_label", latest_state
                )
                if latest_state == "deep_stall" and reserved_ids:
                    cancel_ids = list(reserved_ids)
                    if _has_native_strict_like_pending_refilter(archive):
                        cancel_ids = cancel_ids[1:]
                    cancelled = _cancel_pending_chain_reserves_for_deep_stall(
                        archive, pending, cancel_ids, tick_id=tick_id
                    )
                    if cancelled:
                        print(
                            f"  [chain_refilter_reserve] cancelled "
                            f"{len(cancelled)} queued reserve(s) after current "
                            f"tick entered deep_stall",
                            flush=True,
                        )
                new_ids = [
                    L.candidate_id
                    for L in archive.iter_records(LaunchDecision)
                    if L.tick_id == tick_id
                    and L.status == "launched"
                    and L.candidate_id not in reserved_ids
                ]
                for cid in new_ids:
                    if cid.startswith("chain_"):
                        chain_backfill_seen.add(cid)
                    if cid not in pending:
                        pending.append(cid)
                planned = len(new_ids)
                # Backfill only unused queue capacity. Deep-stall throttling leaves
                # capacity for subsequent generation.
                _bf_cand_by_id = {
                    c.candidate_id: c for c in archive.iter_records(ActionCandidate)
                }
                _score_lane_occupied = _busy_score_conversion_count(pool) + sum(
                    1
                    for cid in pending
                    if is_canonical_score_conversion(_bf_cand_by_id.get(cid))
                )
                bf = (
                    []
                    if latest_state == "deep_stall" or _score_lane_occupied > 0
                    else _chain_backfill_ids(
                        archive,
                        chain_backfill_seen,
                        min(1, max(0, len(pool) - len(pending))),
                        admission_tick_id=tick_id,
                        allow_exhausted_background=True,
                    )
                )
                if bf:
                    pending.extend(bf)
                    chain_backfill_seen.update(bf)
                    _append_chain_launch_decisions(
                        archive,
                        bf,
                        tick_id=tick_id,
                        source="chain_backfill",
                        why="idle-slot chain refilter backfill (G-032)",
                    )
                    print(
                        f"  [chain_backfill] queued {len(bf)} chain refilter(s) "
                        f"for idle slots",
                        flush=True,
                    )
                state = summary.get("evidence", {}).get("state_label", "?")
            else:
                state = "chain_refilter_reserved"
            print(
                f"  state={state}  planned={planned}  "
                f"chain_reserved={chain_reserved}  queued_now={len(pending)}",
                flush=True,
            )

        # Dispatch free slots after planning replenishes the queue.
        if pending:
            _cand_by_id = {
                c.candidate_id: c for c in archive.iter_records(ActionCandidate)
            }
            try:
                dispatched_post = _dispatch_pending_to_free_slots(
                    pool,
                    pending,
                    _cand_by_id,
                    archive=archive,
                    target=target,
                    target_pdb=args.target_pdb,
                    archive_root=args.archive_root,
                    round_id=round_id,
                    dispatch_fn=dispatch_candidate,
                    dispatch_retries=dispatch_retries,
                    dispatch_defer_seen=dispatch_defer_seen,
                )
            except Exception as exc:  # noqa: BLE001
                dispatched_post = 0
                print(
                    f"  [dispatch_loop_err][postdispatch] {type(exc).__name__}: {exc}",
                    flush=True,
                )
        else:
            dispatched_post = 0

        # 4. Sleep. Back off only when there is genuinely no work in flight or
        # queued; otherwise poll briefly so reaps + refills stay responsive.
        sleep_s = _controller_sleep_seconds(
            dispatched=dispatched_pre + dispatched_post,
            planned=planned,
            pending=pending,
            pool=pool,
            poll_interval_s=poll_interval_s,
        )
        if sleep_s == 120.0:
            print("  [no_progress] sleeping 120s", flush=True)
        time.sleep(sleep_s)

    # Preserve the original sequential shutdown: each busy slot gets a wait
    # of up to 10 minutes, followed by salvage/termination on timeout.
    # This is not a shared completion deadline. The loop-start wall check and
    # estimated-runtime feasibility checks are not a hard per-start cutoff.
    # Startup consumes scheduler time before the controller clock starts;
    # the allocation can therefore end before every slot finishes draining.
    drain_worker_slots(
        pool,
        grace_seconds_per_slot=600.0,
        archive=archive,
        target=target,
        auto_chain_sequence=chain_seq_ref,
        dependencies=worker_supervision,
    )

    print(f"[v7_controller] done at {time.strftime('%Y-%m-%dT%H:%M:%SZ')}", flush=True)


if __name__ == "__main__":
    main()
