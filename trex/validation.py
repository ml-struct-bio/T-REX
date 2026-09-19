"""Fail-closed installation and target validation for T-ReX.

This module validates the *execution contract* without launching an LLM or a
GPU worker.  It is intentionally stricter than an import smoke: enabled
families must have the executables and assets that their dispatch paths use,
and registered targets must match both their JSON constraint and PDB content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from Bio.PDB import PDBParser

from .capability_registry import default_registry
from .provenance import verify_model_manifest
from .resource_paths import publication_data_path


REPO_ROOT = (
    Path(os.environ.get("TREX_REPO_ROOT", Path(__file__).resolve().parents[1]))
    .expanduser()
    .resolve()
)
TARGET_REGISTRY = publication_data_path(REPO_ROOT, "config/targets/registry.json")
TARGET_ASSET_MANIFEST = publication_data_path(
    REPO_ROOT, "config/targets/assets.sha256.json"
)
PRODUCTION_STACK = publication_data_path(
    REPO_ROOT, "config/reproducibility/production_stack.json"
)
AF2_ASSET_MANIFEST = publication_data_path(
    REPO_ROOT, "config/reproducibility/af2_parameters_manifest.json"
)
PROTEINMPNN_ASSET_MANIFEST = publication_data_path(
    REPO_ROOT, "config/reproducibility/proteinmpnn_weights_manifest.json"
)
DEFAULT_FAMILIES = (
    "bindcraft",
    "boltzgen",
    "complexa_beam",
    "complexa_best_of_n",
    "complexa_fk_steering",
    "complexa_mcts",
    "proteinmpnn_redesign",
    "structure_refilter",
)
INSTALLATION_VALIDATION_SCHEMA_VERSION = "trex.installation-validation.v1"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.status == "ok" or not self.required


def _env_path(name: str, default: Path | None = None) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if raw:
        return Path(raw).expanduser()
    return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _family_set(value: str | Iterable[str]) -> set[str]:
    items = value.split(",") if isinstance(value, str) else value
    return {str(item).strip() for item in items if str(item).strip()}


def _which(explicit: str | None, fallback: str) -> str | None:
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.is_file() and os.access(path, os.X_OK) else None
    return shutil.which(fallback)


def _git_head(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_diff_sha256(path: Path) -> str | None:
    """Hash the tracked working-tree diff using the provenance convention."""
    try:
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
            cwd=path,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    diff = proc.stdout.strip()
    return hashlib.sha256(diff.encode()).hexdigest() if diff else None


def _target_checks(
    target: str,
    *,
    target_config: Path | None,
    target_pdb: Path | None,
    asset_root: Path | None,
    verify_asset_hash: bool,
) -> tuple[list[Check], Path | None, Path | None, str | None]:
    checks: list[Check] = []
    if TARGET_REGISTRY.is_file():
        registry = json.loads(TARGET_REGISTRY.read_text())
        entry = registry.get("targets", {}).get(target)
    else:
        registry = {}
        entry = None
    if entry is None and target_config is None:
        detail = (
            f"unknown target without --target-config: {target}"
            if TARGET_REGISTRY.is_file()
            else "registered-target data are unavailable; use a source checkout, set "
            "TREX_REPO_ROOT, or pass --target-config and --target-pdb"
        )
        return [Check("target registry", "fail", detail)], None, None, None

    if entry is None:
        assert target_config is not None
        config = target_config
        pdb = target_pdb
        expected_id = None
        checks.append(
            Check(
                "target registry",
                "warn",
                "custom target; registry hash is unavailable",
                required=False,
            )
        )
    else:
        config = target_config or publication_data_path(
            REPO_ROOT, Path("config/targets") / entry["config"]
        )
        pdb = target_pdb or (asset_root / entry["pdb"] if asset_root else None)
        expected_id = str(entry["target_id"])
    checks.append(
        Check("target config", "ok" if config.is_file() else "fail", str(config))
    )
    if not config.is_file():
        return checks, config, pdb, expected_id

    payload = json.loads(config.read_text())
    actual_id = str(payload.get("target_id", ""))
    if expected_id is None:
        expected_id = actual_id or None
    checks.append(
        Check(
            "target id",
            "ok" if actual_id == expected_id else "fail",
            f"expected={expected_id} observed={actual_id or '<missing>'}",
        )
    )
    chains = [str(chain) for chain in payload.get("chain_ids", [])]
    checks.append(
        Check(
            "target chains",
            "ok" if chains and len(chains) == len(set(chains)) else "fail",
            ",".join(chains) if chains else "chain_ids is empty",
        )
    )
    if pdb is None:
        checks.append(
            Check(
                "target PDB",
                "warn",
                "not configured; set TREX_TARGET_ASSET_ROOT or pass --target-pdb",
                required=False,
            )
        )
        return checks, config, pdb, expected_id
    checks.append(Check("target PDB", "ok" if pdb.is_file() else "fail", str(pdb)))
    if not pdb.is_file():
        return checks, config, pdb, expected_id

    try:
        structure = PDBParser(QUIET=True).get_structure(target, str(pdb))
        model = next(structure.get_models())
        residues_by_chain = {
            chain.id: {
                (int(residue.id[1]), str(residue.id[2]).strip())
                for residue in chain.get_residues()
                if residue.id[0] == " "
            }
            for chain in model
        }
    except Exception as exc:
        checks.append(Check("PDB parse", "fail", f"{type(exc).__name__}: {exc}"))
        return checks, config, pdb, expected_id

    missing_chains = [chain for chain in chains if chain not in residues_by_chain]
    checks.append(
        Check(
            "configured chains in PDB",
            "fail" if missing_chains else "ok",
            f"missing={missing_chains}" if missing_chains else ",".join(chains),
        )
    )

    bad_hotspots: list[str] = []
    for hotspot in payload.get("hotspots", []):
        text = str(hotspot)
        match = re.fullmatch(r"(.)(-?\d+)([A-Za-z]?)", text)
        if match is None:
            bad_hotspots.append(text)
            continue
        chain, residue_no, insertion = (
            match.group(1),
            int(match.group(2)),
            match.group(3),
        )
        if (residue_no, insertion) not in residues_by_chain.get(chain, set()):
            bad_hotspots.append(text)
    checks.append(
        Check(
            "hotspots in PDB",
            "fail" if bad_hotspots else "ok",
            f"missing_or_invalid={bad_hotspots}"
            if bad_hotspots
            else ",".join(payload.get("hotspots", [])),
        )
    )

    if verify_asset_hash and entry is not None and TARGET_ASSET_MANIFEST.is_file():
        manifest = json.loads(TARGET_ASSET_MANIFEST.read_text())
        expected_sha = manifest.get("targets", {}).get(target, {}).get("sha256")
        observed_sha = _sha256(pdb)
        checks.append(
            Check(
                "target PDB SHA256",
                "ok" if expected_sha and observed_sha == expected_sha else "fail",
                f"expected={expected_sha or '<missing>'} observed={observed_sha}",
            )
        )
    return checks, config, pdb, expected_id


def validate_install(
    *,
    target: str,
    enabled_families: str | Iterable[str] = DEFAULT_FAMILIES,
    target_config: Path | None = None,
    target_pdb: Path | None = None,
    asset_root: Path | None = None,
    require_backends: bool = False,
    require_model: bool = False,
    verify_asset_hash: bool = True,
    verify_backend_revisions: bool = False,
    verify_checkpoint_content: bool = False,
) -> list[Check]:
    """Return all preflight checks; callers decide how to render failures."""
    families = _family_set(enabled_families)
    checks, _, _, _ = _target_checks(
        target,
        target_config=target_config,
        target_pdb=target_pdb,
        asset_root=asset_root,
        verify_asset_hash=verify_asset_hash,
    )

    try:
        registry = default_registry()
    except Exception as exc:  # noqa: BLE001 - malformed operator plugin.
        checks.append(
            Check(
                "backend extension registry",
                "fail",
                f"{type(exc).__name__}: {exc}",
            )
        )
        registry = None
    if registry is not None:
        unknown = sorted(family for family in families if registry.get(family) is None)
        unavailable = sorted(
            family
            for family in families
            if (cap := registry.get(family)) is not None
            and cap.availability != "available"
        )
        checks.append(
            Check(
                "enabled family registry",
                "fail" if unknown or unavailable else "ok",
                f"unknown={unknown} unavailable={unavailable}"
                if unknown or unavailable
                else ",".join(sorted(families)) or "none",
            )
        )
        diagnostic = sorted(
            family
            for family in families
            if (cap := registry.get(family)) is not None and cap.outputs_diagnostic_only
        )
        if diagnostic and "structure_refilter" not in families:
            # Match the controller's fail-safe auto-enable behavior, and make
            # the resulting AF2 dependency visible during preflight.
            families.add("structure_refilter")
            checks.append(
                Check(
                    "canonical score-conversion route",
                    "warn",
                    f"auto-required for diagnostic families: {','.join(diagnostic)}",
                    required=False,
                )
            )
        for family in sorted(families):
            cap = registry.get(family)
            if cap is None or cap.preflight is None:
                continue
            try:
                ok, detail = cap.preflight()
            except Exception as exc:  # noqa: BLE001 - operator hook boundary.
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            checks.append(
                Check(
                    f"{family} adapter preflight",
                    "ok" if ok else ("fail" if require_backends else "warn"),
                    detail,
                    required=require_backends,
                )
            )

    foldseek = _which(os.environ.get("TREX_FOLDSEEK_BIN"), "foldseek")
    checks.append(
        Check(
            "Foldseek executable",
            "ok" if foldseek else ("fail" if require_backends else "warn"),
            foldseek or "not found; set TREX_FOLDSEEK_BIN or PATH",
            required=require_backends,
        )
    )
    mmseqs = _which(os.environ.get("TREX_MMSEQS_BIN"), "mmseqs")
    checks.append(
        Check(
            "MMseqs2 executable",
            "ok" if mmseqs else "warn",
            mmseqs or "not found; sequence diversity will be unavailable",
            required=False,
        )
    )

    need_complexa = any(family.startswith("complexa_") for family in families)
    need_af2 = bool({"structure_refilter", "proteinmpnn_redesign"} & families)
    need_bindcraft = "bindcraft" in families
    need_boltzgen = "boltzgen" in families
    complexa = _env_path(
        "TREX_COMPLEXA_REPO", REPO_ROOT / "external" / "Proteina-Complexa"
    )
    legacy = _env_path("TREX_LEGACY_COMPLEXA_REPO", complexa)
    complexa_python = _env_path(
        "TREX_COMPLEXA_PYTHON",
        complexa / ".venv" / "bin" / "python" if complexa else None,
    )
    bindcraft = _env_path("TREX_BINDCRAFT_REPO", REPO_ROOT / "external" / "BindCraft")
    bindcraft_env = _env_path(
        "TREX_BINDCRAFT_ENV",
        bindcraft / ".venv" if bindcraft else None,
    )
    boltzgen = _env_path("TREX_BOLTZGEN_REPO", REPO_ROOT / "external" / "BoltzGen")
    boltzgen_bin = _env_path("TREX_BOLTZGEN_BIN")

    def path_check(
        name: str, path: Path | None, required: bool, kind: str = "exists"
    ) -> None:
        if kind == "file":
            ok = bool(path and path.is_file())
        elif kind == "exec":
            ok = bool(path and path.is_file() and os.access(path, os.X_OK))
        else:
            ok = bool(path and path.exists())
        checks.append(
            Check(
                name,
                "ok" if ok else ("fail" if required and require_backends else "warn"),
                str(path) if path else "not configured",
                required=required and require_backends,
            )
        )

    path_check("Proteina-Complexa checkout", complexa, need_complexa)
    path_check(
        "Proteina-Complexa env.sh",
        complexa / "env.sh" if complexa else None,
        need_complexa,
        "file",
    )
    path_check(
        "Complexa/AF2 Python", complexa_python, need_complexa or need_af2, "exec"
    )
    path_check(
        "Complexa checkpoint",
        complexa / "ckpts" / "complexa.ckpt" if complexa else None,
        need_complexa,
        "file",
    )
    path_check(
        "Complexa AE checkpoint",
        complexa / "ckpts" / "complexa_ae.ckpt" if complexa else None,
        need_complexa,
        "file",
    )
    af2_root = legacy / "community_models" / "ckpts" / "AF2" if legacy else None
    path_check("AF2 parameters", af2_root, need_af2)
    for model_index in range(1, 6):
        path_check(
            f"AF2 multimer-v3 model {model_index}",
            af2_root / f"params_model_{model_index}_multimer_v3.npz"
            if af2_root
            else None,
            need_af2,
            "file",
        )
    mpnn_root = (
        legacy / "community_models" / "ProteinMPNN" / "vanilla_model_weights"
        if legacy
        else None
    )
    for model_name in ("v_48_002", "v_48_010", "v_48_020", "v_48_030"):
        path_check(
            f"ProteinMPNN weight {model_name}",
            mpnn_root / f"{model_name}.pt" if mpnn_root else None,
            "proteinmpnn_redesign" in families,
            "file",
        )

    def manifest_check(name: str, root: Path | None, manifest_path: Path) -> None:
        path_check(f"{name} manifest", manifest_path, True, "file")
        if not root or not root.is_dir() or not manifest_path.is_file():
            return
        try:
            digest = verify_model_manifest(
                root,
                json.loads(manifest_path.read_text()),
                verify_content=verify_checkpoint_content,
            )
        except Exception as exc:
            checks.append(
                Check(
                    f"{name} verification",
                    "fail" if require_backends else "warn",
                    str(exc),
                    required=require_backends,
                )
            )
        else:
            mode = "full SHA256" if verify_checkpoint_content else "manifest + sizes"
            checks.append(
                Check(
                    f"{name} verification",
                    "ok",
                    f"{digest} ({mode})",
                    required=require_backends,
                )
            )

    if need_af2:
        manifest_check("AF2 parameters", af2_root, AF2_ASSET_MANIFEST)
    if "proteinmpnn_redesign" in families:
        manifest_check("ProteinMPNN weights", mpnn_root, PROTEINMPNN_ASSET_MANIFEST)

    path_check(
        "BindCraft checkout",
        bindcraft / "bindcraft.py" if bindcraft else None,
        need_bindcraft,
        "file",
    )
    path_check(
        "BindCraft Python",
        bindcraft_env / "bin" / "python" if bindcraft_env else None,
        need_bindcraft,
        "exec",
    )
    path_check("BoltzGen checkout", boltzgen, need_boltzgen)
    path_check("BoltzGen executable", boltzgen_bin, need_boltzgen, "exec")

    if require_model:
        model_path = _env_path("TREX_QWEN_MODEL_PATH")
        model_manifest = _env_path(
            "TREX_MODEL_MANIFEST",
            publication_data_path(
                REPO_ROOT, "config/trex/qwen3_6_27b_fp8_model_manifest.json"
            ),
        )
        checks.append(
            Check(
                "Qwen model directory",
                "ok" if model_path and model_path.is_dir() else "fail",
                str(model_path) if model_path else "not configured",
            )
        )
        checks.append(
            Check(
                "Qwen model manifest",
                "ok" if model_manifest and model_manifest.is_file() else "fail",
                str(model_manifest) if model_manifest else "not configured",
            )
        )
        if (
            model_path
            and model_path.is_dir()
            and model_manifest
            and model_manifest.is_file()
        ):
            try:
                digest = verify_model_manifest(
                    model_path,
                    json.loads(model_manifest.read_text()),
                    verify_content=verify_checkpoint_content,
                )
            except Exception as exc:
                checks.append(
                    Check("Qwen model manifest verification", "fail", str(exc))
                )
            else:
                checks.append(Check("Qwen model manifest verification", "ok", digest))

    if verify_backend_revisions:
        if not PRODUCTION_STACK.is_file():
            checks.append(
                Check(
                    "production stack manifest",
                    "fail",
                    f"not found: {PRODUCTION_STACK}",
                )
            )
            return checks
        stack = json.loads(PRODUCTION_STACK.read_text())
        paths = {
            "proteina_complexa": (complexa, need_complexa or need_af2),
            "proteina_complexa_community": (
                legacy,
                need_af2 or "proteinmpnn_redesign" in families,
            ),
            "bindcraft": (bindcraft, need_bindcraft),
            "boltzgen": (boltzgen, need_boltzgen),
        }
        for name, (path, enabled) in paths.items():
            if not enabled:
                continue
            expected = stack.get("components", {}).get(name, {}).get("git_commit")
            if path and path.is_dir() and expected:
                observed = _git_head(path)
                checks.append(
                    Check(
                        f"{name} revision",
                        "ok" if observed == expected else "fail",
                        f"expected={expected} observed={observed or '<unavailable>'}",
                    )
                )
            if name == "bindcraft" and path and path.is_dir():
                component = stack.get("components", {}).get(name, {})
                patch_name = component.get("patch")
                expected_patch_sha = component.get("patch_file_sha256")
                if patch_name and expected_patch_sha:
                    patch_path = publication_data_path(REPO_ROOT, str(patch_name))
                    observed_patch_sha = (
                        _sha256(patch_path) if patch_path.is_file() else None
                    )
                    checks.append(
                        Check(
                            "bindcraft patch artifact",
                            "ok"
                            if observed_patch_sha == expected_patch_sha
                            else "fail",
                            f"expected={expected_patch_sha} "
                            f"observed={observed_patch_sha or '<missing>'}",
                        )
                    )
                expected_diff = component.get("production_tracked_diff_sha256")
                if expected_diff:
                    observed_diff = _git_diff_sha256(path)
                    checks.append(
                        Check(
                            "bindcraft tracked patch",
                            "ok" if observed_diff == expected_diff else "fail",
                            f"expected={expected_diff} "
                            f"observed={observed_diff or '<clean-or-unavailable>'}",
                        )
                    )
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", required=True, help="registered key or custom target label"
    )
    parser.add_argument("--target-config", type=Path)
    parser.add_argument("--target-pdb", type=Path)
    parser.add_argument(
        "--asset-root", type=Path, default=_env_path("TREX_TARGET_ASSET_ROOT")
    )
    parser.add_argument(
        "--enabled-families",
        default=os.environ.get("TREX_ENABLED_FAMILIES", ",".join(DEFAULT_FAMILIES)),
    )
    parser.add_argument("--require-backends", action="store_true")
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--skip-asset-hash", action="store_true")
    parser.add_argument("--verify-backend-revisions", action="store_true")
    parser.add_argument(
        "--verify-checkpoint-content",
        action="store_true",
        help="rehash model/checkpoint bytes; slower than the default file/size check",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    checks = validate_install(
        target=args.target,
        enabled_families=args.enabled_families,
        target_config=args.target_config,
        target_pdb=args.target_pdb,
        asset_root=args.asset_root,
        require_backends=args.require_backends,
        require_model=args.require_model,
        verify_asset_hash=not args.skip_asset_hash,
        verify_backend_revisions=args.verify_backend_revisions,
        verify_checkpoint_content=args.verify_checkpoint_content,
    )
    failed = [check for check in checks if check.required and check.status == "fail"]
    if args.as_json:
        print(
            json.dumps(
                {
                    "ok": not failed,
                    "schema_version": INSTALLATION_VALIDATION_SCHEMA_VERSION,
                    "failure_count": len(failed),
                    "checks": [asdict(check) for check in checks],
                },
                indent=2,
            )
        )
    else:
        width = max(len(check.name) for check in checks)
        for check in checks:
            print(f"{check.status.upper():4}  {check.name:<{width}}  {check.detail}")
        print(
            "\nPreflight passed."
            if not failed
            else f"\nPreflight failed: {len(failed)} required check(s).",
            file=sys.stdout if not failed else sys.stderr,
        )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
