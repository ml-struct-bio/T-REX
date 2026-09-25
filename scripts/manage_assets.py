#!/usr/bin/env python3
"""Download, verify and connect pinned assets without importing model code."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "config/reproducibility/assets_manifest.json"
CHUNK_SIZE = 8 * 1024 * 1024


def relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or not path.parts
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {".", ".."} for part in path.parts)
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path


def asset_path(root: Path, name: str) -> Path:
    path = root / relative_path(name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes root through a symlink: {name}")
    return path


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != "trex.assets.v1":
        raise ValueError("Unsupported asset manifest schema")
    components = manifest.get("components")
    if not isinstance(components, dict) or not components:
        raise ValueError("Manifest has no components")
    seen = set()
    for name, component in components.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError(f"Invalid component name: {name}")
        if not component.get("files"):
            raise ValueError(f"Component has no files: {name}")
        members = set()
        for entry in component["files"]:
            relative_path(entry["path"])
            if entry["path"] in seen:
                raise ValueError(f"Duplicate asset path: {entry['path']}")
            seen.add(entry["path"])
            if type(entry["size"]) is not int or entry["size"] < 0:
                raise ValueError("Asset size must be a nonnegative integer")
            if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
                raise ValueError("Invalid SHA256")
            if "archive_member" in entry:
                member = entry["archive_member"]
                relative_path(member)
                if member in members or not component.get("archive_url"):
                    raise ValueError("Duplicate archive member or missing archive URL")
                members.add(member)
    return manifest


def selected_components(manifest: dict, names: list[str] | None) -> dict:
    all_components = manifest["components"]
    names = list(dict.fromkeys(names or all_components))
    unknown = set(names) - set(all_components)
    if unknown:
        raise ValueError(f"Unknown components: {', '.join(sorted(unknown))}")
    return {name: all_components[name] for name in names}


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(path: Path, entry: dict, *, size_only: bool = False) -> None:
    if not path.is_file():
        raise ValueError(f"Missing asset: {path}")
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"Size mismatch: {path}")
    if not size_only and digest_file(path) != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch: {path}")


def verify(root: Path, components: dict, *, size_only: bool = False) -> None:
    errors = []
    for name, component in components.items():
        for entry in component["files"]:
            try:
                check_file(asset_path(root, entry["path"]), entry, size_only=size_only)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        print(f"Checked {name}: {len(component['files'])} files", flush=True)
    if errors:
        raise ValueError("Asset verification failed:\n" + "\n".join(errors))
    print(
        "Sizes verified (content not hashed)."
        if size_only
        else "All SHA256 hashes verified.",
        flush=True,
    )


def install_stream(stream, dest: Path, entry: dict) -> None:
    """Commit a complete verified file; refuse replacement, including races."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".trex-asset-", dir=dest.parent)
    temporary = Path(temporary)
    try:
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(fd, "wb") as out:
            for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
                size += len(block)
                if size > entry["size"]:
                    raise ValueError(f"Download exceeds expected size: {entry['path']}")
                digest.update(block)
                out.write(block)
        if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Downloaded bytes do not match manifest: {entry['path']}")
        # link() is atomic and refuses to replace a concurrently created file.
        os.link(temporary, dest)
    finally:
        temporary.unlink(missing_ok=True)


def open_url(url: str):
    parsed = urllib.parse.urlsplit(url)
    local_test = parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }
    if (
        (parsed.scheme != "https" and not local_test)
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Downloads require HTTPS (HTTP loopback is allowed for tests)")
    return urllib.request.urlopen(url, timeout=120)


def fetch_archive(root: Path, url: str, entries: list[dict]) -> None:
    wanted = {entry["archive_member"]: entry for entry in entries}
    with closing(open_url(url)) as response, tarfile.open(
        fileobj=response, mode="r|*"
    ) as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if name not in wanted:
                continue
            entry = wanted.pop(name)
            if not member.isfile() or member.size != entry["size"]:
                raise ValueError(f"Invalid archive member: {name}")
            with closing(archive.extractfile(member)) as stream:
                install_stream(stream, asset_path(root, entry["path"]), entry)
            print(f"Downloaded {entry['path']}", flush=True)
            if not wanted:
                break
    if wanted:
        raise ValueError(f"Archive missing declared members: {', '.join(wanted)}")


def fetch(
    root: Path, components: dict, *, base_url: str | None, source_dir: Path | None
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    # Check every existing file before transferring any missing ones.
    for component in components.values():
        for entry in component["files"]:
            dest = asset_path(root, entry["path"])
            if dest.exists() or dest.is_symlink():
                check_file(dest, entry)
    for component in components.values():
        archived = []
        for entry in component["files"]:
            dest = asset_path(root, entry["path"])
            if dest.exists():
                continue
            if "text" in entry:
                stream = io.BytesIO(entry["text"].encode())
            elif source_dir is not None:
                stream = asset_path(source_dir, entry["path"]).open("rb")
            elif base_url:
                stream = open_url(
                    base_url.rstrip("/")
                    + "/"
                    + urllib.parse.quote(entry["path"], safe="/")
                )
            elif "archive_member" in entry:
                archived.append(entry)
                continue
            elif entry.get("url"):
                stream = open_url(entry["url"])
            else:
                raise ValueError(f"No download source for {entry['path']}")
            with closing(stream):
                install_stream(stream, dest, entry)
            print(f"Downloaded {entry['path']}", flush=True)
        if archived:
            fetch_archive(root, component["archive_url"], archived)
    print(
        "Fetch complete; every existing and downloaded file passed SHA256 verification.",
        flush=True,
    )


def import_zip(root: Path, archive_path: Path, components: dict) -> None:
    """Import only manifest-declared regular files from the release ZIP."""
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for info in archive.infolist():
            name = info.filename.rstrip("/")
            relative_path(name)
            if name == "T-REX-assets" and info.is_dir():
                continue
            if not name.startswith("T-REX-assets/"):
                raise ValueError(f"Unexpected ZIP root: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"ZIP symlinks are not accepted: {info.filename}")
            if name in members:
                raise ValueError(f"Duplicate ZIP member: {name}")
            members[name] = info
        entries = [entry for c in components.values() for entry in c["files"]]
        for entry in entries:
            name = "T-REX-assets/" + entry["path"]
            info = members.get(name)
            if info is None or info.is_dir() or info.file_size != entry["size"]:
                raise ValueError(f"Missing or invalid ZIP asset: {entry['path']}")
            dest = asset_path(root, entry["path"])
            if dest.exists() or dest.is_symlink():
                check_file(dest, entry)
        for entry in entries:
            dest = asset_path(root, entry["path"])
            if dest.exists():
                continue
            with archive.open(members["T-REX-assets/" + entry["path"]]) as source:
                install_stream(source, dest, entry)
            print(f"Imported {entry['path']}", flush=True)
    print("ZIP import complete; all selected file hashes verified.", flush=True)


def fetch_drive(
    root: Path, components: dict, url: str, release_path: Path, cache: Path
) -> None:
    """Fetch a public Drive ZIP, verify the release hash, then import assets."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "drive.google.com":
        raise ValueError("Use the HTTPS Google Drive file sharing URL")
    if "/folders/" in parsed.path or parsed.username or parsed.password:
        raise ValueError("Provide the ZIP file link, not a folder link")
    release = json.loads(release_path.read_text())
    sha = release.get("sha256", "")
    if not isinstance(sha, str) or not re.fullmatch(r"[a-f0-9]{64}", sha):
        raise ValueError(
            "The release manifest does not yet contain a verified ZIP SHA256"
        )
    size = release.get("size")
    if type(size) is not int or size <= 0:
        raise ValueError("Invalid release ZIP size")
    missing = False
    for component in components.values():
        for entry in component["files"]:
            dest = asset_path(root, entry["path"])
            if dest.exists() or dest.is_symlink():
                check_file(dest, entry)
            else:
                missing = True
    if not missing:
        print("All selected assets already present and hash-verified.")
        return
    cache.mkdir(parents=True, exist_ok=True)
    download = cache / (sha + ".zip")
    if download.is_symlink():
        raise ValueError("Download cache archive must not be a symlink")
    if not download.exists():
        try:
            import gdown
        except ImportError as exc:
            raise ValueError(
                "Google Drive support requires: python -m pip install -e '.[assets]'"
            ) from exc
        try:
            result = gdown.download(
                url=url,
                output=str(download),
                resume=True,
                use_cookies=False,
            )
        except Exception as exc:
            raise ValueError(
                "Drive download failed "
                f"({type(exc).__name__}: {exc}); check public file access, quota "
                "and network, then retry"
            ) from exc
        if result is None:
            raise ValueError(
                "Drive download failed; check public download permission and retry"
            )
    check_file(download, {"size": size, "sha256": sha})
    import_zip(root, download, components)


def verify_bindcraft(manifest: dict, python: Path) -> None:
    """Read installed metadata and hash package weights without loading models."""
    component = selected_components(manifest, ["colabdesign_bindcraft"])[
        "colabdesign_bindcraft"
    ]
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import json; from importlib.metadata import distribution; "
            "d = distribution('colabdesign'); "
            "print(json.dumps({'version': d.version, "
            "'site': str(d.locate_file(''))}))",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    installed = json.loads(result.stdout)
    if installed["version"] != component["package_version"]:
        raise ValueError(
            f"ColabDesign version: expected {component['package_version']}, "
            f"got {installed['version']}"
        )
    prefix = "checkpoints/colabdesign_bindcraft/"
    files = [
        {**entry, "path": entry["path"].removeprefix(prefix)}
        for entry in component["files"]
        if entry["path"].startswith(prefix + "colabdesign/")
    ]
    if not files:
        raise ValueError("Manifest has no ColabDesign package weights")
    verify(Path(installed["site"]), {"installed_colabdesign_weights": {"files": files}})
    print(
        "Installed package weights match. Backend dependencies/inference are separate checks."
    )


def configure(
    root: Path,
    manifest: dict,
    complexa: Path,
    community: Path,
    bindcraft: Path,
    env_out: Path,
) -> None:
    """Connect asset files to existing backend checkouts; no environment installs."""
    required = ("complexa", "af2", "proteinmpnn", "boltzgen", "qwen", "targets")
    components = selected_components(manifest, list(required))
    for path in (complexa, community, bindcraft):
        if not path.is_dir():
            raise ValueError(f"Backend checkout does not exist: {path}")
    verify(root, components)
    links = {}
    for name, component in components.items():
        for entry in component["files"]:
            source = asset_path(root, entry["path"])
            basename = source.name
            destinations = []
            if name == "complexa" and basename.endswith(".ckpt"):
                destinations = [complexa / "ckpts" / basename]
            elif name == "af2" and (basename.endswith(".npz") or basename == "LICENSE"):
                destinations = [
                    repo / "community_models/ckpts/AF2" / basename
                    for repo in (complexa, community)
                ]
                destinations.append(bindcraft / "params" / basename)
            elif name == "proteinmpnn" and basename.endswith(".pt"):
                destinations = [
                    repo
                    / "community_models/ProteinMPNN/vanilla_model_weights"
                    / basename
                    for repo in (complexa, community)
                ]
            for dest in destinations:
                if dest.exists() or dest.is_symlink():
                    check_file(dest, entry)
                else:
                    if not any(
                        dest.parent.resolve().is_relative_to(checkout)
                        for checkout in (complexa, community, bindcraft)
                    ):
                        raise ValueError(
                            f"Backend asset destination escapes checkouts: {dest}"
                        )
                    links[dest] = source
    if env_out in links:
        raise ValueError("Environment file conflicts with an asset link")
    values = {
        "TREX_REPO_ROOT": REPO,
        "TREX_ASSET_ROOT": root,
        "TREX_TARGET_ASSET_ROOT": root / "targets",
        "TREX_COMPLEXA_REPO": complexa,
        "TREX_LEGACY_COMPLEXA_REPO": community,
        "TREX_BINDCRAFT_REPO": bindcraft,
        "TREX_BOLTZGEN_CACHE": root / "checkpoints/boltzgen",
        "TREX_QWEN_MODEL_PATH": root / "checkpoints/Qwen3.6-27B-FP8",
        "TREX_MODEL_MANIFEST": REPO / "config/trex/qwen3_6_27b_fp8_model_manifest.json",
    }
    content = "# Asset paths only; configure backend executables separately.\n"
    content += (
        "\n".join(
            f"export {key}={shlex.quote(str(value))}" for key, value in values.items()
        )
        + "\n"
    )
    if env_out.exists() or env_out.is_symlink():
        if (
            env_out.is_symlink()
            or not env_out.is_file()
            or env_out.read_text() != content
        ):
            raise ValueError(f"Refusing to overwrite environment file: {env_out}")
    created = []
    try:
        for dest, source in links.items():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(source)
            created.append(dest)
        if not env_out.exists():
            env_out.parent.mkdir(parents=True, exist_ok=True)
            with env_out.open("x") as stream:
                created.append(env_out)
                stream.write(content)
    except OSError:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    print(f"Created {len(links)} asset links. Environment: {env_out}")
    print(
        "Backend Python environments, binaries and the LLM server still require installation."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    installed = subparsers.add_parser(
        "verify-bindcraft", help="Verify ColabDesign weights in a BindCraft environment"
    )
    installed.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    installed.add_argument("--python", type=Path, required=True)
    for command in ("list", "fetch", "verify", "configure", "import-zip"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        if command != "list":
            sub.add_argument("--root", type=Path, required=True)
        if command != "configure":
            sub.add_argument(
                "--component",
                action="append",
                help="Repeat to select components; default: all",
            )
        if command == "verify":
            sub.add_argument(
                "--size-only",
                action="store_true",
                help="Fast presence/size check; does not verify content",
            )
        if command == "fetch":
            sources = sub.add_mutually_exclusive_group()
            sources.add_argument(
                "--base-url",
                help="Published mirror root, preferably pinned to a release revision",
            )
            sources.add_argument(
                "--source-dir",
                type=Path,
                help="Copy from a local portable asset bundle",
            )
            sources.add_argument(
                "--drive-url", help="Public Google Drive ZIP file sharing URL"
            )
            sub.add_argument(
                "--release-manifest",
                type=Path,
                default=REPO / "config/reproducibility/assets_release.json",
            )
            sub.add_argument(
                "--download-cache",
                type=Path,
                help="Keep the downloaded ZIP here for reuse",
            )
        if command == "import-zip":
            sub.add_argument("--archive", type=Path, required=True)
        if command == "configure":
            sub.add_argument("--complexa-repo", type=Path, required=True)
            sub.add_argument("--community-repo", type=Path, required=True)
            sub.add_argument("--bindcraft-repo", type=Path, required=True)
            sub.add_argument("--env-out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.command == "verify-bindcraft":
            # Resolving a venv executable symlink changes the selected environment.
            verify_bindcraft(manifest, args.python.absolute())
            return 0
        components = selected_components(manifest, getattr(args, "component", None))
        if args.command == "list":
            for name, component in components.items():
                size = sum(f["size"] for f in component["files"])
                print(
                    f"{name:12s} {len(component['files']):3d} files {size / 1e9:8.3f} GB  {component.get('license', '')}"
                )
            print(manifest.get("scope", ""))
        elif args.command == "fetch":
            if args.drive_url:
                cache = (
                    args.download_cache
                    or args.root.resolve().parent / ".trex-downloads"
                )
                fetch_drive(
                    args.root.resolve(),
                    components,
                    args.drive_url,
                    args.release_manifest,
                    cache,
                )
            else:
                fetch(
                    args.root.resolve(),
                    components,
                    base_url=args.base_url,
                    source_dir=args.source_dir,
                )
        elif args.command == "import-zip":
            import_zip(args.root.resolve(), args.archive, components)
        elif args.command == "verify":
            verify(args.root.resolve(), components, size_only=args.size_only)
        else:
            configure(
                args.root.resolve(),
                manifest,
                args.complexa_repo.resolve(),
                args.community_repo.resolve(),
                args.bindcraft_repo.resolve(),
                args.env_out.absolute(),
            )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        tarfile.TarError,
        zipfile.BadZipFile,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Asset setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
