"""Keep public guides focused on usage and reproducibility."""
import io
import json
from pathlib import Path
import tokenize

ROOT = Path(__file__).resolve().parents[2]
INTERNAL_MARKERS = (
    "Review-found ",
    "Internal review notes",
    "Debugging history",
    "TODO:",
    "FIXME:",
)


def test_public_guides_exclude_internal_review_records():
    paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    paths += list((ROOT / "docs").glob("*.md"))
    for path in paths:
        text = path.read_text()
        for marker in INTERNAL_MARKERS:
            assert marker not in text, (path, marker)
        for private_root in ("/home/", "/scratch/", "/projects/"):
            assert private_root not in text, (path, private_root)
    assert not (ROOT / "docs/implementation-and-reproducibility.md").exists()
    assert (ROOT / "docs/reproducibility.md").is_file()


def test_runtime_comments_exclude_internal_review_markers():
    for path in (ROOT / "trex").rglob("*.py"):
        if "tests" in path.relative_to(ROOT).parts:
            continue
        for token in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
            if token.type == tokenize.COMMENT:
                for marker in INTERNAL_MARKERS:
                    assert marker not in token.string, (path, token.start, marker)


def test_bundled_guides_keep_scope_without_internal_execution_history():
    manifest = json.loads((ROOT / "config/reproducibility/assets_manifest.json").read_text())
    documents = manifest["components"]["documentation"]["files"]
    for entry in documents:
        for marker in INTERNAL_MARKERS:
            assert marker not in entry["text"], (entry["path"], marker)
    versions = json.loads(next(entry["text"] for entry in documents if entry["path"] == "VERSIONS.json"))
    assert versions["observed_backend_environments"]["bindcraft"]["whole_job_completed"] is False
    assert versions["validation_scope"]["full_48_hour_benchmark"] == "not tested"
    assert "provenance_note" not in versions["validation_scope"]
