"""Reproducibility metadata for T-ReX production runs.

The scientific archive is append-only, but a result is not reproducible unless
the controller source, model, target, external backends, and runtime settings
are identified alongside it. This module captures that information without
depending on the scientific controller or third-party packages.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

try:
    import fcntl
except ImportError:  # pragma: no cover - production runs on Linux.
    fcntl = None


PROVENANCE_SCHEMA = "v7.3.3_run_provenance_v2"
LEGACY_PROVENANCE_SCHEMAS = frozenset(
    {
        "v7.3.3_run_provenance_v1",
    }
)
SUPPORTED_PROVENANCE_SCHEMAS = LEGACY_PROVENANCE_SCHEMAS | {PROVENANCE_SCHEMA}
MODEL_MANIFEST_SCHEMA = "v7.3.3_model_manifest_v1"
PROVENANCE_CAPTURE_OUTPUT_SCHEMA_VERSION = "trex.provenance-capture.v1"
_MODEL_MANIFEST_NAME = ".trex_model_manifest.json"
_SOURCE_SUFFIXES = {".json", ".patch", ".py", ".sh", ".slurm", ".txt", ".yaml", ".yml"}
_GIT_LFS_DISABLED = [
    "git",
    "-c",
    "filter.lfs.required=false",
    "-c",
    "filter.lfs.smudge=",
    "-c",
    "filter.lfs.clean=",
    "-c",
    "filter.lfs.process=",
]


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_rows(rows: Iterable[tuple[str, int, str]]) -> str:
    digest = hashlib.sha256()
    for relative, size, content_sha256 in sorted(rows):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_tree_digest(source_root: Path) -> tuple[str, int]:
    """Hash the publication-relevant T-ReX source and launch configuration."""
    roots = [
        source_root / "trex",
        source_root / "config" / "trex",
        source_root / "config" / "targets",
        source_root / "config" / "reproducibility",
        source_root / "scripts",
    ]
    explicit = [
        source_root / "slurm" / "trex_per_target_node.slurm",
        source_root / "pyproject.toml",
        source_root / "README.md",
        source_root / "docs" / "all_prompts.py",
        source_root / "docs" / "all_prompts_snapshot.txt",
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix in _SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
                and ".pytest_cache" not in path.parts
            )
    files.extend(path for path in explicit if path.is_file())
    rows = [
        (str(path.relative_to(source_root)), path.stat().st_size, _sha256_file(path))
        for path in sorted(set(files))
    ]
    if not rows:
        raise RuntimeError(f"no T-ReX source files found under {source_root}")
    return _digest_rows(rows), len(rows)


def build_model_manifest(model_path: Path) -> dict[str, Any]:
    """Hash every model file once and return a portable content manifest."""
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    files: list[dict[str, Any]] = []
    for path in sorted(model_path.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        if path.name == _MODEL_MANIFEST_NAME:
            continue
        relative = str(path.relative_to(model_path))
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"model directory is empty: {model_path}")
    rows = [(row["path"], int(row["size"]), row["sha256"]) for row in files]
    return {
        "schema_version": MODEL_MANIFEST_SCHEMA,
        "model_name": model_path.name,
        "content_sha256": _digest_rows(rows),
        "n_files": len(files),
        "total_bytes": sum(int(row["size"]) for row in files),
        "files": files,
    }


def verify_model_manifest(
    model_path: Path,
    manifest: dict[str, Any],
    *,
    verify_content: bool = False,
) -> str:
    """Validate a content manifest and optionally rehash every declared file.

    The default path is suitable for campaign startup: it validates manifest
    integrity plus every declared file name and byte size. ``verify_content``
    additionally compares every file's SHA256 and is intended for release
    audits because model trees can contain tens of gigabytes.
    """
    if manifest.get("schema_version") != MODEL_MANIFEST_SCHEMA:
        raise ValueError("unsupported model manifest schema")
    expected = manifest.get("files") or []
    if not expected or not manifest.get("content_sha256"):
        raise ValueError("model manifest is incomplete")
    rows: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for row in expected:
        relative = str(row.get("path", ""))
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in seen
        ):
            raise ValueError(f"unsafe or duplicate model path: {relative!r}")
        seen.add(relative)
        try:
            size = int(row["size"])
            expected_sha = str(row["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid model manifest row: {row!r}") from exc
        if (
            size < 0
            or len(expected_sha) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha)
        ):
            raise ValueError(f"invalid model manifest row: {row!r}")
        rows.append((relative, size, expected_sha))
        path = model_path / relative
        if not path.is_file():
            raise FileNotFoundError(f"model file missing: {path}")
        if path.stat().st_size != size:
            raise ValueError(f"model file size changed: {path}")
        if verify_content and _sha256_file(path) != expected_sha:
            raise ValueError(f"model file content changed: {path}")
    if manifest.get("n_files") not in (None, len(rows)):
        raise ValueError("model manifest n_files does not match its file rows")
    total_bytes = sum(size for _, size, _ in rows)
    if manifest.get("total_bytes") not in (None, total_bytes):
        raise ValueError("model manifest total_bytes does not match its file rows")
    content_sha256 = _digest_rows(rows)
    if content_sha256 != manifest["content_sha256"]:
        raise ValueError("model manifest content digest does not match its file rows")
    return str(manifest["content_sha256"])


def model_digest_from_env() -> str | None:
    value = os.environ.get("TREX_MODEL_DIGEST", "").strip()
    return value or None


def _run_text(args: list[str], cwd: Path, *, attempts: int = 8) -> str | None:
    """Run a read-only command, retrying transient shared-filesystem failures."""
    for attempt in range(max(1, attempts)):
        try:
            proc = subprocess.run(
                args,
                cwd=str(cwd),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is not None and proc.returncode == 0:
            return proc.stdout.strip()
        if attempt + 1 < attempts:
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
    return None


@contextlib.contextmanager
def _git_probe_lock(path: Path):
    lock_dir = os.environ.get("TREX_PROVENANCE_LOCK_DIR", "").strip()
    if not lock_dir or fcntl is None:
        yield
        return
    root = Path(lock_dir)
    lock_name = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    try:
        root.mkdir(parents=True, exist_ok=True)
        handle = (root / f"{lock_name}.lock").open("w")
    except OSError:
        yield
        return
    with handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _lfs_exclude_pathspecs(path: Path) -> list[str]:
    attrs = path / ".gitattributes"
    if not attrs.is_file():
        return []
    excludes: list[str] = []
    for raw in attrs.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "filter=lfs" not in line:
            continue
        pattern = line.split()[0].strip()
        if pattern and not pattern.startswith("!"):
            excludes.append(f":!{pattern}")
    return excludes


def _lfs_excluded_status_and_diff(
    path: Path, *, include_diff_hash: bool
) -> tuple[str | None, str | None, list[str]]:
    excludes = _lfs_exclude_pathspecs(path)
    if not excludes:
        return None, None, []
    pathspec = ["--", ".", *excludes]
    status = _run_text(
        [
            *_GIT_LFS_DISABLED,
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
            *pathspec,
        ],
        path,
    )
    diff = (
        _run_text(
            [
                *_GIT_LFS_DISABLED,
                "diff",
                "HEAD",
                "--binary",
                "--no-ext-diff",
                *pathspec,
            ],
            path,
        )
        if include_diff_hash
        else None
    )
    return status, diff, excludes


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def git_state(
    path: Path,
    *,
    include_diff_hash: bool = True,
    diff_output: Path | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Record a revision, all working-tree paths, and the tracked diff.

    Untracked source changes are not representable in ``git diff HEAD``. They
    are nevertheless part of the execution state and must make ``dirty`` true;
    the T-ReX source-tree digest records their bytes independently.
    """
    with _git_probe_lock(path):
        head = _run_text(["git", "rev-parse", "HEAD"], path)
        status = _run_text(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"], path
        )
        diff = (
            _run_text(["git", "diff", "HEAD", "--binary", "--no-ext-diff"], path)
            if include_diff_hash
            else None
        )
        capture_mode = "normal"
        lfs_excludes: list[str] = []
        if status is None or (include_diff_hash and status and diff is None):
            (
                fallback_status,
                fallback_diff,
                lfs_excludes,
            ) = _lfs_excluded_status_and_diff(path, include_diff_hash=include_diff_hash)
            if fallback_status is not None:
                status = fallback_status
                capture_mode = "lfs_excluded"
            if fallback_diff is not None:
                diff = fallback_diff
    capture_error: str | None = None
    if head is not None and status is None:
        capture_error = f"could not capture tracked git status for {path}"
        if strict:
            raise RuntimeError(capture_error)
    if head is not None and include_diff_hash and status and diff is None:
        capture_error = f"could not capture tracked git diff for {path}"
        if strict:
            raise RuntimeError(capture_error)
    artifact: str | None = None
    if diff and diff_output is not None:
        _atomic_write_text(diff_output, diff)
        artifact = str(diff_output)
    untracked_paths = sorted(
        line[3:] for line in (status or "").splitlines() if line.startswith("?? ")
    )
    return {
        "path": str(path.resolve()),
        "git_head": head,
        "dirty": bool(status) if status is not None else None,
        "status_sha256": hashlib.sha256(status.encode()).hexdigest()
        if status
        else None,
        "tracked_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()
        if diff
        else None,
        "tracked_diff_artifact": artifact,
        "untracked_paths": untracked_paths,
        "untracked_path_count": len(untracked_paths),
        "capture_error": capture_error,
        "capture_mode": capture_mode,
        "tracked_diff_lfs_excludes": lfs_excludes,
    }


def _package_versions() -> dict[str, str | None]:
    packages = ("vllm", "torch", "openai", "pydantic", "numpy", "scipy")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def capture_run_provenance(args: argparse.Namespace) -> dict[str, Any]:
    from .backend_extensions import backend_extension_provenance
    from .prompt_catalog import runtime_prompt_metadata

    source_root = Path(args.source_root).resolve()
    model_path = Path(args.model_path).resolve()
    target_pdb = Path(args.target_pdb).resolve()
    target_config = Path(args.target_config).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    model_manifest = json.loads(model_manifest_path.read_text())
    p2_python = os.environ.get("TREX_COMPLEXA_PYTHON") or None
    p2_python_resolved = (
        str(Path(p2_python).expanduser().resolve()) if p2_python else None
    )
    model_digest = verify_model_manifest(model_path, model_manifest)
    source_digest, source_file_count = source_tree_digest(source_root)
    output_path = Path(args.out).resolve()
    patch_root = output_path.parent / "repository_patches"
    backends: dict[str, dict[str, Any]] = {
        "trex": git_state(source_root, include_diff_hash=False),
    }
    for name, path in (
        ("complexa", os.environ.get("TREX_COMPLEXA_REPO", "")),
        ("legacy_complexa", os.environ.get("TREX_LEGACY_COMPLEXA_REPO", "")),
        ("bindcraft", os.environ.get("TREX_BINDCRAFT_REPO", "")),
        ("boltzgen", os.environ.get("TREX_BOLTZGEN_REPO", "")),
    ):
        if path and Path(path).is_dir():
            backends[name] = git_state(
                Path(path), diff_output=patch_root / f"{name}.patch"
            )
    charged_gpus_raw = getattr(args, "charged_gpus", None)
    charged_gpus = (
        float(charged_gpus_raw) if charged_gpus_raw not in (None, "") else None
    )
    critic_enabled = bool(int(args.critic_enabled))
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA,
        "source": {
            "tree_sha256": source_digest,
            "n_files": source_file_count,
            "release_archive_sha256": os.environ.get("TREX_RELEASE_ARCHIVE_SHA256")
            or None,
        },
        "model": {
            "served_name": args.served_model,
            "client_model": getattr(args, "llm_model", None),
            "path": str(model_path),
            "content_sha256": model_digest,
            "manifest_path": str(model_manifest_path),
            "n_files": model_manifest.get("n_files"),
            "total_bytes": model_manifest.get("total_bytes"),
        },
        "target": {
            "name": args.target,
            "target_id": args.target_id,
            "pdb_path": str(target_pdb),
            "pdb_sha256": _sha256_file(target_pdb),
            "config_path": str(target_config),
            "config_sha256": _sha256_file(target_config),
        },
        "controller": {
            "max_wall_h": float(args.max_wall_h),
            "foldseek_su_tm_score": float(args.foldseek_su_tm_score),
            "foldseek_collapse_tm_score": float(args.foldseek_collapse_tm_score),
            "enabled_families": [x for x in args.enabled_families.split(",") if x],
            "worker_gpus": [x for x in args.worker_gpus.split(",") if x],
            "charged_gpus": charged_gpus,
            "seed": int(args.seed),
            "critic_enabled": critic_enabled,
            "evidence_skip_enabled": bool(int(args.evidence_skip_enabled)),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "packages": _package_versions(),
            "af2_proteinmpnn_python": p2_python,
            "af2_proteinmpnn_python_resolved": p2_python_resolved,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "repositories": backends,
        "backend_extensions": backend_extension_provenance(),
        "prompts": runtime_prompt_metadata(
            model=getattr(args, "llm_model", None),
            base_url=getattr(args, "llm_base_url", None),
            critic_enabled=critic_enabled,
        ),
    }
    campaign_name = getattr(args, "campaign_name", None)
    campaign_source_sha256 = getattr(args, "campaign_source_sha256", None)
    if campaign_name or campaign_source_sha256:
        payload["campaign"] = {
            "name": campaign_name,
            "source_path": getattr(args, "campaign_source_path", None),
            "source_sha256": campaign_source_sha256,
            "input_artifact": getattr(args, "campaign_input_artifact", None),
            "input_artifact_sha256": getattr(
                args, "campaign_input_artifact_sha256", None
            ),
            "resolved_artifact": getattr(args, "campaign_resolved_artifact", None),
            "resolved_artifact_sha256": getattr(
                args, "campaign_resolved_artifact_sha256", None
            ),
        }
    _atomic_write_json(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    model = sub.add_parser("hash-model", help="create a full content-hash manifest")
    model.add_argument("--model-path", required=True)
    model.add_argument("--out", required=True)

    digest = sub.add_parser(
        "model-digest", help="verify a manifest and print its digest"
    )
    digest.add_argument("--model-path", required=True)
    digest.add_argument("--manifest", required=True)
    digest.add_argument(
        "--full-content",
        action="store_true",
        help="rehash every declared file instead of checking names and sizes only",
    )

    capture = sub.add_parser("capture", help="write run_provenance.json")
    capture.add_argument("--out", required=True)
    capture.add_argument("--source-root", required=True)
    capture.add_argument("--model-path", required=True)
    capture.add_argument("--model-manifest", required=True)
    capture.add_argument("--served-model", required=True)
    capture.add_argument(
        "--llm-model",
        help="exact provider-qualified model identifier passed to LLM clients",
    )
    capture.add_argument(
        "--llm-base-url",
        help="exact OpenAI-compatible endpoint passed to LLM clients",
    )
    capture.add_argument("--target", required=True)
    capture.add_argument("--target-id", required=True)
    capture.add_argument("--target-pdb", required=True)
    capture.add_argument("--target-config", required=True)
    capture.add_argument("--max-wall-h", required=True)
    capture.add_argument("--foldseek-su-tm-score", required=True)
    capture.add_argument("--foldseek-collapse-tm-score", required=True)
    capture.add_argument("--enabled-families", required=True)
    capture.add_argument("--worker-gpus", required=True)
    capture.add_argument(
        "--charged-gpus",
        help="optional reserved-allocation GPU count (audit metadata only)",
    )
    capture.add_argument("--seed", required=True)
    capture.add_argument("--critic-enabled", required=True)
    capture.add_argument("--evidence-skip-enabled", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "hash-model":
        manifest = build_model_manifest(Path(args.model_path))
        _atomic_write_json(Path(args.out), manifest)
        print(manifest["content_sha256"])
        return 0
    if args.command == "model-digest":
        manifest = json.loads(Path(args.manifest).read_text())
        print(
            verify_model_manifest(
                Path(args.model_path),
                manifest,
                verify_content=args.full_content,
            )
        )
        return 0
    if args.command == "capture":
        payload = capture_run_provenance(args)
        print(
            json.dumps(
                {
                    "source_tree_sha256": payload["source"]["tree_sha256"],
                    "schema_version": PROVENANCE_CAPTURE_OUTPUT_SCHEMA_VERSION,
                    "model_content_sha256": payload["model"]["content_sha256"],
                    "target_pdb_sha256": payload["target"]["pdb_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
