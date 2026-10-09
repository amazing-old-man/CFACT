"""Stable identities and idempotent, relative-path release persistence."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
from cfact.io.serialization import atomic_json
from cfact.sources.manifest import sha256_file

SCHEMA = "cfact.dual-view.v1"


def stable_id(kind: str, payload) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return kind + "_" + hashlib.sha256(canonical.encode()).hexdigest()[:24]


def inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected path relative to release root: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes release: {relative}")
    return result


def copy_file(source: Path, destination: Path) -> str:
    """Copy independent bytes; never hardlink to a read-only historical dataset."""
    digest = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise ValueError(f"Resume found different bytes: {destination}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    # A prior interrupted copy is safe to replace; final files are immutable.
    with source.open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    temporary.replace(destination)
    return digest


def write_json(path: Path, value) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Resume metadata differs: {path}")
        return
    atomic_json(path, value)


def portable(value, paths: dict[str, str] | None = None):
    """Preserve metadata while replacing legacy file references with release references.

    External paths are provenance names, not dereferenceable dataset links.
    Their content hashes / original source identities remain in source manifests.
    """
    paths = paths or {}
    if isinstance(value, dict):
        return {key: portable(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, paths) for item in value]
    if isinstance(value, str):
        if value in paths:
            return paths[value]
        if value.startswith("/"):
            return {
                "external_basename": Path(value).name,
                "availability": "EXTERNAL_PROVENANCE_ONLY",
            }
    return value
