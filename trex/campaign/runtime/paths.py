"""Resolve backend paths once for a controller process.

``BackendPaths`` is the user-facing campaign input and may contain omitted
values. ``RuntimePaths`` is the fully materialized controller view: core
backend locations are concrete absolute paths, executable symlinks are
preserved, and derived AF2/ProteinMPNN locations have one definition.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ...resource_paths import publication_data_path


BACKEND_PATH_ENVIRONMENT_VARIABLES: Mapping[str, str] = MappingProxyType(
    {
        "repo_root": "TREX_REPO_ROOT",
        "external_root": "TREX_EXTERNAL_ROOT",
        "complexa_repo": "TREX_COMPLEXA_REPO",
        "legacy_complexa_repo": "TREX_LEGACY_COMPLEXA_REPO",
        "complexa_python": "TREX_COMPLEXA_PYTHON",
        "bindcraft_repo": "TREX_BINDCRAFT_REPO",
        "bindcraft_env": "TREX_BINDCRAFT_ENV",
        "boltzgen_repo": "TREX_BOLTZGEN_REPO",
        "boltzgen_binary": "TREX_BOLTZGEN_BIN",
        "boltzgen_cache": "TREX_BOLTZGEN_CACHE",
        "foldseek_binary": "TREX_FOLDSEEK_BIN",
        "mmseqs_binary": "TREX_MMSEQS_BIN",
        "qwen_model_path": "TREX_QWEN_MODEL_PATH",
        "qwen_model_manifest": "TREX_MODEL_MANIFEST",
    }
)

EXECUTABLE_BACKEND_PATH_FIELDS = frozenset(
    {
        "complexa_python",
        "boltzgen_binary",
        "foldseek_binary",
        "mmseqs_binary",
    }
)


def resolve_environment_path(
    environment_name: str,
    default: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve a directory/file environment override to an absolute path."""

    values = os.environ if environment is None else environment
    configured_value = values.get(environment_name, "").strip()
    selected_path = Path(configured_value) if configured_value else Path(default)
    return selected_path.expanduser().resolve()


def resolve_environment_executable_path(
    environment_name: str,
    default: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve an executable without dereferencing virtualenv symlinks."""

    values = os.environ if environment is None else environment
    configured_value = values.get(environment_name, "").strip()
    selected_path = Path(configured_value) if configured_value else Path(default)
    expanded_path = selected_path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path
    return Path(os.path.abspath(expanded_path))


@dataclass(frozen=True)
class RuntimePaths:
    """Concrete paths consumed by backend launchers in one controller run."""

    repo_root: Path
    external_root: Path
    complexa_repo: Path
    legacy_complexa_repo: Path
    complexa_python: Path
    bindcraft_repo: Path
    bindcraft_env: Path
    boltzgen_repo: Path
    boltzgen_binary: Path
    boltzgen_cache: Path
    foldseek_binary: Path | None = None
    mmseqs_binary: Path | None = None
    qwen_model_path: Path | None = None
    qwen_model_manifest: Path | None = None

    @property
    def community_models_root(self) -> Path:
        return self.legacy_complexa_repo / "community_models"

    @property
    def af2_data_dir(self) -> Path:
        return self.community_models_root / "ckpts" / "AF2"

    @property
    def proteinmpnn_dir(self) -> Path:
        return self.community_models_root / "ProteinMPNN"

    @property
    def proteinmpnn_weights(self) -> Path:
        return self.proteinmpnn_dir / "vanilla_model_weights" / "v_48_020.pt"

    @property
    def foldseek_command(self) -> str:
        return str(self.foldseek_binary) if self.foldseek_binary else "foldseek"

    @property
    def mmseqs_command(self) -> str:
        return str(self.mmseqs_binary) if self.mmseqs_binary else "mmseqs"

    def as_dict(self) -> dict[str, str | None]:
        """Return direct and derived paths for logs and reproducibility records."""

        output: dict[str, str | None] = {}
        for field_definition in fields(self):
            value = getattr(self, field_definition.name)
            output[field_definition.name] = (
                str(value) if value is not None else None
            )
        output.update(
            {
                "community_models_root": str(self.community_models_root),
                "af2_data_dir": str(self.af2_data_dir),
                "proteinmpnn_dir": str(self.proteinmpnn_dir),
                "proteinmpnn_weights": str(self.proteinmpnn_weights),
            }
        )
        return output

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        default_repo_root: str | Path | None = None,
    ) -> "RuntimePaths":
        """Resolve one immutable runtime snapshot from ``TREX_*`` settings."""

        values = os.environ if environment is None else environment
        package_repo_root = (
            Path(default_repo_root)
            if default_repo_root is not None
            else Path(__file__).resolve().parents[3]
        )

        def configured_path(field_name: str, default: Path | None) -> Path | None:
            environment_name = BACKEND_PATH_ENVIRONMENT_VARIABLES[field_name]
            raw_value = values.get(environment_name, "").strip()
            if not raw_value and default is None:
                return None
            selected_path = Path(raw_value) if raw_value else default
            assert selected_path is not None
            if field_name in EXECUTABLE_BACKEND_PATH_FIELDS:
                return resolve_environment_executable_path(
                    environment_name,
                    selected_path,
                    environment=values,
                )
            return resolve_environment_path(
                environment_name,
                selected_path,
                environment=values,
            )

        repo_root = configured_path("repo_root", package_repo_root)
        assert repo_root is not None
        external_root = configured_path("external_root", repo_root / "external")
        assert external_root is not None
        complexa_repo = configured_path(
            "complexa_repo",
            repo_root / "external" / "Proteina-Complexa",
        )
        assert complexa_repo is not None
        legacy_complexa_repo = configured_path(
            "legacy_complexa_repo",
            complexa_repo,
        )
        assert legacy_complexa_repo is not None
        bindcraft_repo = configured_path(
            "bindcraft_repo",
            external_root / "BindCraft",
        )
        assert bindcraft_repo is not None
        boltzgen_repo = configured_path(
            "boltzgen_repo",
            external_root / "BoltzGen",
        )
        assert boltzgen_repo is not None

        search_path = values.get("PATH", "")
        foldseek_on_path = shutil.which("foldseek", path=search_path)
        mmseqs_on_path = shutil.which("mmseqs", path=search_path)
        complexa_python = configured_path(
            "complexa_python",
            complexa_repo / ".venv" / "bin" / "python",
        )
        bindcraft_env = configured_path(
            "bindcraft_env",
            bindcraft_repo / ".venv",
        )
        boltzgen_binary = configured_path(
            "boltzgen_binary",
            boltzgen_repo / ".venv" / "bin" / "boltzgen",
        )
        boltzgen_cache = configured_path(
            "boltzgen_cache",
            external_root / "checkpoints" / "boltzgen",
        )
        assert complexa_python is not None
        assert bindcraft_env is not None
        assert boltzgen_binary is not None
        assert boltzgen_cache is not None

        return cls(
            repo_root=repo_root,
            external_root=external_root,
            complexa_repo=complexa_repo,
            legacy_complexa_repo=legacy_complexa_repo,
            complexa_python=complexa_python,
            bindcraft_repo=bindcraft_repo,
            bindcraft_env=bindcraft_env,
            boltzgen_repo=boltzgen_repo,
            boltzgen_binary=boltzgen_binary,
            boltzgen_cache=boltzgen_cache,
            foldseek_binary=configured_path(
                "foldseek_binary",
                Path(foldseek_on_path) if foldseek_on_path else None,
            ),
            mmseqs_binary=configured_path(
                "mmseqs_binary",
                Path(mmseqs_on_path) if mmseqs_on_path else None,
            ),
            qwen_model_path=configured_path("qwen_model_path", None),
            qwen_model_manifest=configured_path(
                "qwen_model_manifest",
                publication_data_path(
                    repo_root, "config/trex/qwen3_6_27b_fp8_model_manifest.json"
                ),
            ),
        )
