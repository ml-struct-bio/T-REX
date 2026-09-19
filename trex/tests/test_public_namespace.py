from pathlib import Path

import trex
from trex import controller
from trex.llm import create_client


def test_public_namespace_exports_runtime():
    assert trex.SCHEMA_VERSION == "v7.3.3-reward-guard"
    assert callable(controller.main)
    assert callable(create_client)


def test_release_contains_no_legacy_package_namespace():
    root = Path(__file__).resolve().parents[2]
    forbidden = "auto" + "research"
    ignored = {
        ".git", "build", ".pytest_cache", "__pycache__",
        "slurm_logs", "runs", "archives", "outputs", "results", "external",
    }
    offenders = []
    for path in root.rglob("*"):
        if (
            not path.is_file()
            or any(part in ignored or part.endswith(".egg-info") for part in path.parts)
            or path.suffix == ".pyc"
        ):
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if forbidden in text.lower():
            offenders.append(str(path.relative_to(root)))
    assert offenders == []


def test_legacy_package_directories_are_absent():
    root = Path(__file__).resolve().parents[2]
    old_runtime = "auto" + "research_v7_3_3"
    old_common = "auto" + "research_common"
    assert not (root / old_runtime).exists()
    assert not (root / old_common).exists()
