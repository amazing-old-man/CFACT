"""Extract downloaded archives without following links or escaping the destination."""

from __future__ import annotations
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import zipfile


def member_path(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    result = root.joinpath(*path.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes destination: {name}")
    return result


def extract_tar(source: Path, destination: Path, *, episode_indices: list[int] | None = None) -> None:
    """Publish a new extracted directory; preserve incomplete attempts for diagnosis."""
    destination = destination.resolve()
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(f"Choose a new extraction directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:*") as archive:
        members = archive.getmembers()
        for item in members:
            member_path(partial, item.name)
            if not (item.isfile() or item.isdir()):
                raise ValueError(f"Links and special files are unsupported: {item.name}")
        if episode_indices is not None:
            episodes = {f"episode_{index:06d}" for index in episode_indices}
            def selected(item):
                parts = PurePosixPath(item.name).parts
                return ("meta" in parts or item.name.endswith("extras/dataset_meta.json")
                        or any(PurePosixPath(part).stem in episodes for part in parts))
            members = [item for item in members if selected(item)]
        required = sum(item.size for item in members if item.isfile())
        if required + 1024**3 > shutil.disk_usage(destination.parent).free:
            raise OSError(f"Insufficient disk space to extract {required} bytes")
        partial.mkdir()
        for item in members:
            target = member_path(partial, item.name)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(item) as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
    partial.rename(destination)


def extract_asset_zip(source: Path, destination: Path) -> dict:
    """Merge assets; existing files must match, and are never overwritten."""
    from cfact.sources.manifest import sha256_file
    import hashlib

    destination = destination.resolve()
    with zipfile.ZipFile(source) as archive:
        for item in archive.infolist():
            member_path(destination, item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Asset archive contains a symlink: {item.filename}")
        destination.mkdir(parents=True, exist_ok=True)
        required = sum(item.file_size for item in archive.infolist())
        if required + 1024**3 > shutil.disk_usage(destination).free:
            raise OSError(f"Insufficient disk space for asset archive ({required} bytes)")
        count = 0
        for item in archive.infolist():
            target = member_path(destination, item.filename)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                digest = hashlib.sha256()
                with archive.open(item) as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                if sha256_file(target) != digest.hexdigest():
                    raise FileExistsError(f"Existing asset differs: {target}")
            else:
                partial = target.with_name(target.name + ".partial")
                with archive.open(item) as stream, partial.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                partial.rename(target)
            count += 1
    return {"files": count, "uncompressed_bytes": required}
