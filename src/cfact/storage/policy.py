"""Artifact path containment and actual target filesystem resource contracts."""
from __future__ import annotations
import math
import os
from pathlib import Path
import shutil


def contained_path(path: Path, allowed_root: Path) -> Path:
    path, allowed_root = Path(path).expanduser().resolve(), Path(allowed_root).expanduser().resolve()
    if not path.is_relative_to(allowed_root):
        raise ValueError("ARTIFACT_PATH_OUTSIDE_ALLOWED_ROOT")
    return path


def artifact_root(config: dict, project: Path, override: Path | None = None) -> Path:
    requested = override if override is not None else config.get("artifact_root")
    if requested is None:
        return (project / "output").resolve()
    requested = Path(requested).expanduser()
    if not requested.is_absolute():
        raise ValueError("ARTIFACT_ROOT_MUST_BE_ABSOLUTE")
    allowed = config.get("allowed_write_root", requested)
    return contained_path(requested, Path(allowed))


def target_filesystem(directory: Path, policy: dict) -> dict:
    """Probe the actual existing target, never the repository or a guessed disk."""
    directory = Path(directory).resolve()
    parent = directory
    while not parent.exists():
        if parent == parent.parent:
            raise FileNotFoundError(directory)
        parent = parent.parent
    usage, filesystem = shutil.disk_usage(parent), os.statvfs(parent)
    minimum = policy.get("minimum_host_free_gib", 0)
    inode_minimum = policy.get("minimum_free_inodes", 0)
    if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum) or minimum < 0
            or type(inode_minimum) is not int or inode_minimum < 0):
        raise ValueError("INVALID_STORAGE_RESERVE_POLICY")
    failures = []
    if usage.free < minimum * 2**30:
        failures.append("DISK_RESERVE")
    if filesystem.f_favail < inode_minimum:
        failures.append("INODE_RESERVE")
    return {"status": "PASS" if not failures else "STORAGE_GUARD_STOP",
            "failures": failures, "write_target_realpath": str(directory),
            "filesystem_probe_realpath": str(parent), "filesystem_device": parent.stat().st_dev,
            "filesystem_free_bytes": usage.free, "filesystem_free_inodes": filesystem.f_favail,
            "minimum_filesystem_free_bytes": int(minimum * 2**30),
            "minimum_filesystem_free_inodes": inode_minimum}
