from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts/verify_study_environment.py"
_SPEC = importlib.util.spec_from_file_location("study_environment_check", _SCRIPT)
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


class Distribution:
    def __init__(self, name, version, requires=(), origin=None):
        self.metadata = {"Name": name}
        self.version = version
        self.requires = requires
        self.origin = origin

    def read_text(self, name):
        assert name == "direct_url.json"
        return json.dumps(self.origin) if self.origin is not None else None


@pytest.fixture
def inventory():
    lock = b"consumer==1.0\nnumpy==2.4.4\ntransformers @ git+https://example.org/model.git@abcdef\n"
    profile = {
        "lock_sha256": hashlib.sha256(lock).hexdigest(),
        "python": "3.12.13",
        "platform": "linux",
        "machine": "x86_64",
        "recorded_dependency_exceptions": [
            {
                "package": "consumer",
                "requirement": "numpy<2.3",
                "installed_version": "2.4.4",
            },
        ],
    }
    distributions = [
        Distribution("consumer", "1.0", ["numpy<2.3", 'optional; extra == "test"']),
        Distribution("numpy", "2.4.4"),
        Distribution(
            "transformers",
            "5.6.0.dev0",
            origin={
                "url": "https://example.org/model.git",
                "vcs_info": {"vcs": "git", "commit_id": "abcdef"},
            },
        ),
    ]
    environment = {
        "python_full_version": "3.12.13",
        "sys_platform": "linux",
        "platform_machine": "x86_64",
    }
    return lock, profile, distributions, environment


def test_matches_inventory_with_explicit_exceptions(inventory):
    report = checker.verify_environment(*inventory)
    assert report["matches_study_inventory"]
    assert report["locked_packages"] == 3
    assert len(report["recorded_dependency_exceptions"]) == 1


@pytest.mark.parametrize(
    "change,fragment",
    [
        ("pin", "expected ==2.4.4"),
        ("source", "installed source"),
        ("missing", "missing package"),
        ("dependency", "missing required dependency"),
        ("conflict", "unexpected dependency conflict"),
        ("exception", "recorded dependency exception is absent"),
        ("python", "python_full_version"),
        ("inventory", "SHA256"),
    ],
)
def test_rejects_environment_drift(inventory, change, fragment):
    lock, profile, distributions, environment = inventory
    if change == "pin":
        distributions[1].version = "2.2.0"
    elif change == "source":
        distributions[2].origin["vcs_info"]["commit_id"] = "changed"
    elif change == "missing":
        distributions.pop()
    elif change == "dependency":
        distributions[0].requires.append("missing-dependency>=1")
    elif change == "conflict":
        distributions[2].requires = ["numpy<2"]
    elif change == "exception":
        distributions[0].requires = []
    elif change == "python":
        environment["python_full_version"] = "3.12.12"
    elif change == "inventory":
        lock += b"# changed inventory\n"
    report = checker.verify_environment(lock, profile, distributions, environment)
    assert not report["matches_study_inventory"]
    assert any(fragment in item for item in report["failures"])
