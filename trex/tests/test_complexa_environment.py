"""Dependency installation guards and transitive-extra checks, without model execution."""
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from packaging.markers import default_environment

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup = load_script("setup_complexa_env")
check = load_script("check_complexa_env")


class Distribution:
    def __init__(self, name, version, requires=()):
        self.metadata = {"Name": name}
        self.version = version
        self.requires = list(requires)


def test_dependency_check_follows_required_extras():
    packages = [
        Distribution("app", "1.0", ["engine[gpu]>=1"]),
        Distribution("engine", "1.0", ['cuda-helper>=2; extra == "gpu"']),
    ]
    assert check.dependency_errors(packages, default_environment()) == [
        'engine: missing cuda-helper>=2; extra == "gpu"'
    ]
    packages.append(Distribution("cuda-helper", "2.0"))
    assert check.dependency_errors(packages, default_environment()) == []


def test_dependency_check_reports_version_conflict_and_ignores_inactive_extra():
    packages = [
        Distribution("engine", "1.0", ["numpy<2", 'missing-docs; extra == "docs"']),
        Distribution("numpy", "2.4.3"),
    ]
    assert check.dependency_errors(packages, default_environment()) == [
        "engine: requires numpy<2; installed 2.4.3"
    ]


def test_unmarked_existing_environment_is_preserved(tmp_path):
    environment = tmp_path / "working-env"
    environment.mkdir()
    marker = environment / "keep.txt"
    marker.write_text("existing work")
    with pytest.raises(ValueError, match="preserved"):
        setup.check_destination(environment, tmp_path / "repo")
    assert marker.read_text() == "existing work"


def test_environment_cannot_be_reused_for_different_source(tmp_path):
    environment = tmp_path / "env"
    environment.mkdir()
    (environment / setup.MARKER).write_text(
        json.dumps({"repository": str(tmp_path / "other")})
    )
    with pytest.raises(ValueError, match="different"):
        setup.check_destination(environment, tmp_path / "repo")


def test_environment_symlink_is_rejected(tmp_path):
    environment = tmp_path / "env"
    environment.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        setup.check_destination(environment, tmp_path / "repo")


def test_profile_lock_and_metadata_patch_match_published_hashes():
    profile = json.loads(setup.PROFILE.read_text())
    setup.checked_bytes(ROOT / profile["lock"], profile["lock_sha256"])
    patch = setup.checked_bytes(
        ROOT / profile["dependency_patch"], profile["dependency_patch_sha256"]
    ).decode()
    assert [x for x in patch.splitlines() if x.startswith("+++ ")] == [
        "+++ b/pyproject.toml"
    ]
    changed = [
        x
        for x in patch.splitlines()
        if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))
    ]
    assert len(changed) == 4
    assert all("einops==" in x or "scipy==" in x for x in changed)


def test_corrupt_profile_artifact_is_rejected(tmp_path):
    artifact = tmp_path / "lock"
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        setup.checked_bytes(artifact, hashlib.sha256(b"expected").hexdigest())


def test_source_check_works_without_lfs_but_rejects_modified_code(tmp_path):
    repository = tmp_path / "backend"
    repository.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    git("init")
    (repository / ".gitattributes").write_text("dataset.csv filter=lfs -text\n")
    (repository / "dataset.csv").write_text("original data\n")
    project = repository / "pyproject.toml"
    project.write_text("[project]\nname='fixture'\n")
    source = repository / "implementation.py"
    source.write_text("VALUE = 1\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "Fixture",
    )
    git("config", "filter.lfs.process", "definitely-missing-git-lfs")
    git("config", "filter.lfs.required", "true")
    (repository / "dataset.csv").write_text("materialized dataset contents\n")
    profile = {
        "git_commit": git("rev-parse", "HEAD").stdout.strip(),
        "patched_pyproject_sha256": hashlib.sha256(project.read_bytes()).hexdigest(),
    }
    setup.prepare_source(repository, profile)
    source.write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="Tracked backend changes"):
        setup.prepare_source(repository, profile)
