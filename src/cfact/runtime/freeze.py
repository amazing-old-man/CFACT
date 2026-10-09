"""Offline, content-addressed release inventory and deterministic source archive."""

from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import shutil

ROOT_FILES = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitignore",
)
DIRECTORIES = ("src", "scripts", "requirements", "configs", "tests", "examples", "docs", ".github")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for name in DIRECTORIES:
        if any(p.is_symlink() for p in (root / name).rglob("*")):
            raise ValueError("Release source directories must not contain symlinks")
        paths.extend(
            p
            for p in (root / name).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not any(part.endswith(".egg-info") for part in p.parts)
            and p.suffix not in {".pyc", ".pyo"}
        )
    if not (root / "src/cfact/collection/worker.py").is_file():
        raise ValueError("A complete CFACT source checkout is required")
    if any(p.is_symlink() or not p.resolve().is_relative_to(root.resolve()) for p in paths):
        raise ValueError("Release source files must be regular files inside the checkout")
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}


def fingerprint(files: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify(root: Path) -> dict:
    record = json.loads((root / "FREEZE.json").read_text())
    actual = inventory(root)
    expected = record["files"]
    changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
    if record.get("schema") != "cfact.source-freeze.v1" or fingerprint(expected) != record.get(
        "content_sha256"
    ):
        raise ValueError("Invalid source freeze manifest")
    if changed:
        raise ValueError("Frozen release differs: " + ", ".join(changed))
    return {
        "status": "PASS",
        "release": record["release"],
        "files": len(actual),
        "content_sha256": record["content_sha256"],
    }


def create(root: Path, release: str) -> dict:
    if not release or Path(release).name != release or release in {".", ".."}:
        raise ValueError("release must be a single path component")
    files = inventory(root)
    record = {
        "schema": "cfact.source-freeze.v1",
        "release": release,
        "content_sha256": fingerprint(files),
        "files": files,
    }
    with (root / "FREEZE.json").open("x") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return verify(root)


def archive(root: Path, destination: Path) -> str:
    record = verify(root)
    names = [*json.loads((root / "FREEZE.json").read_text())["files"], "FREEZE.json"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # gzip and tar timestamps/owners are fixed, so two archives of the freeze have identical bytes.
    with (
        destination.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for name in sorted(names):
                path = root / name
                info = tarfile.TarInfo(record["release"] + "/" + name)
                info.size = path.stat().st_size
                info.mode = 0o644
                with path.open("rb") as stream:
                    tar.addfile(info, stream)
    return sha256(destination)


def verify_archive(root: Path, archive_path: Path) -> dict:
    """Check every archive member against the independent source copy."""
    record = verify(root)
    expected = {**inventory(root), "FREEZE.json": sha256(root / "FREEZE.json")}
    observed = {}
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            prefix = record["release"] + "/"
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError("Unexpected archive member")
            name = member.name[len(prefix):]
            if name in observed:
                raise ValueError("Duplicate archive member")
            observed[name] = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
    if observed != expected:
        raise ValueError("Archive/source contents differ")
    return {"status": "PASS", "archive_sha256": sha256(archive_path), "files": len(observed)}


def copy_source(root: Path, destination: Path) -> dict:
    """Independent regular-file copy, excluding old freeze and mutable run artifacts."""
    if destination.exists():
        raise FileExistsError(destination)
    files = inventory(root)
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
        if target.stat().st_ino == (root / name).stat().st_ino:
            raise ValueError("Release must not hardlink active sources")
    if inventory(destination) != files:
        raise ValueError("Source changed during release copy")
    return files


def verify_release(release: Path) -> dict:
    record = json.loads((release / "FREEZE.json").read_text())
    if record.get("schema") != "cfact.code-release.v2":
        raise ValueError("Unknown code release schema")
    source = release / "source"
    actual = verify(source)
    if actual["content_sha256"] != record["code_content_sha256"]:
        raise ValueError("Release code content identity mismatch")
    for relative, digest in record["artifact_sha256"].items():
        path = release / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise ValueError(f"Release artifact mismatch: {relative}")
    expected_artifacts = {name for name in record["artifact_sha256"] if name.startswith("artifacts/")}
    actual_artifacts = {p.relative_to(release).as_posix() for p in (release / "artifacts").rglob("*") if p.is_file()}
    if actual_artifacts != expected_artifacts:
        raise ValueError("Release artifact inventory contains missing or additional files")
    expected_checksums = "".join(
        f"{digest}  source/{name}\n" for name, digest in sorted(inventory(source).items())
    )
    if (release / "FILES.sha256").read_text() != expected_checksums:
        raise ValueError("FILES.sha256 differs from source inventory")
    archive_report = verify_archive(source, release / record["archive_path"])
    return {"status": "PASS", "release_id": record["release_id"],
            "code_content_sha256": actual["content_sha256"], "archive": archive_report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "create", "archive", "verify-release", "verify-archive"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--release")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "verify-release":
        result = verify_release(root)
    elif args.action == "verify-archive":
        if not args.output:
            parser.error("verify-archive requires --output archive path")
        result = verify_archive(root, args.output)
    elif args.action == "create":
        if not args.release:
            parser.error("create requires --release")
        result = create(root, args.release)
    elif args.action == "archive":
        if not args.output:
            parser.error("archive requires --output")
        result = {"archive": str(args.output), "sha256": archive(root, args.output)}
    else:
        result = verify(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
