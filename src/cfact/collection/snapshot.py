"""Hash and freeze the complete collection package, including nested modules."""

from __future__ import annotations
import hashlib
import tarfile
from pathlib import Path


def source_files(root: Path) -> list[Path]:
    package = root / "src/cfact"
    if not (package / "collection/worker.py").is_file():
        raise FileNotFoundError(f"No collection source at {root}")
    return sorted(
        p
        for p in package.rglob("*")
        if p.is_file() and p.suffix in {".py", ".json", ".html"} and "__pycache__" not in p.parts
    )


def source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in source_files(root):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def archive_source(root: Path, destination: Path, config: Path, manifest: Path) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        for path in source_files(root):
            archive.add(path, arcname=str(path.relative_to(root)))
        archive.add(config, arcname="config_frozen.json")
        archive.add(manifest, arcname="source_split_frozen.json")


def unpack_source(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or not (
                member.isfile() or member.isdir()
            ):
                raise ValueError(f"Unsupported source archive entry: {member.name}")
        archive.extractall(destination)


def implementation_digest() -> str:
    """Digest the imported package, including when installed from a wheel."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in {".py", ".json", ".html"}
            or "__pycache__" in path.parts
        ):
            continue
        digest.update(("src/cfact/" + path.relative_to(package).as_posix()).encode())
        digest.update(b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
