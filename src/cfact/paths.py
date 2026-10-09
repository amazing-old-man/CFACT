"""Explicit paths for external workspaces and source snapshots."""

from pathlib import Path
import os


def workspace_root() -> Path:
    return Path(os.environ.get("CFACT_WORKSPACE", Path.cwd())).expanduser().resolve()


def source_root() -> Path:
    checkout = Path.cwd() if (Path.cwd() / "src/cfact/collection/worker.py").is_file() else Path(__file__).resolve().parents[2]
    root = (
        Path(os.environ.get("CFACT_SOURCE_ROOT", checkout))
        .expanduser()
        .resolve()
    )
    if not (root / "src/cfact/collection/worker.py").is_file():
        raise FileNotFoundError(
            "Collection requires a source checkout: set --source-root or CFACT_SOURCE_ROOT. Reading data only needs the installed package."
        )
    return root
