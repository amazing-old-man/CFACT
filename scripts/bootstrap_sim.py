#!/usr/bin/env python3
"""Create an isolated, pinned CFACT collection runtime (Python 3.10, Linux)."""

from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
import shlex
import subprocess
import sys


def run(*command: str, **kwargs):
    print("+ " + shlex.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def bootstrap(root: Path, python: str, checkout: Path) -> None:
    lock = json.loads((checkout / "src/cfact/resources/runtime.json").read_text())
    version = subprocess.check_output(
        [python, "-c", 'import sys; print("%d.%d" % sys.version_info[:2])'], text=True
    ).strip()
    if version != lock["python"]:
        raise ValueError(f"Use Python {lock['python']}, got {version}: {python}")
    root = root.expanduser().resolve()
    if root == checkout:
        raise ValueError("The runtime must be a separate directory, such as .runtime")
    os.environ.setdefault("PIP_CACHE_DIR", str(root / "cache/pip"))
    root.mkdir(parents=True, exist_ok=True)
    sources = root / "sources"
    sources.mkdir(exist_ok=True)
    for name, spec in lock["repositories"].items():
        destination = sources / name
        if not destination.exists():
            run("git", "init", str(destination))
            run("git", "-C", str(destination), "remote", "add", "origin", spec["url"])
        url = subprocess.check_output(
            ["git", "-C", str(destination), "remote", "get-url", "origin"], text=True
        ).strip()
        if url != spec["url"]:
            raise ValueError(f"Unexpected origin at {destination}: {url}")
        head = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], capture_output=True, text=True
        )
        if head.returncode != 0:
            # Resume an interrupted initial fetch, without resetting existing checkouts.
            run("git", "-C", str(destination), "fetch", "--depth", "1", "origin", spec["revision"])
            run("git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD")
        else:
            if head.stdout.strip() != spec["revision"]:
                raise ValueError(f"Wrong revision at {destination}; choose a new runtime directory")
        dirty = subprocess.check_output(
            ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        if dirty.strip():
            raise ValueError(f"Tracked source changes at {destination}; refusing to overwrite")
    venv = root / "venv"
    if not venv.exists():
        run(python, "-m", "venv", str(venv))
    executable = str(venv / "bin/python")
    run(executable, "-m", "pip", "install", "pip==25.3", "setuptools==80.9.0", "wheel==0.45.1")
    run(
        executable,
        "-m",
        "pip",
        "install",
        "-c",
        str(checkout / "requirements/simulation-lock.txt"),
        "-r",
        str(checkout / "requirements/simulation.txt"),
    )
    run(executable, "-m", "pip", "install", "--no-deps", "-e", str(checkout))
    site = Path(
        subprocess.check_output(
            [executable, "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True
        ).strip()
    )
    # Expose only these pinned source trees. Do not install their unrelated training extras.
    (site / "cfact_simulator_sources.pth").write_text(
        "".join(str(sources / name) + "\n" for name in lock["repositories"])
    )
    (root / "runtime-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    with (root / "python-packages.txt").open("w") as stream:
        run(executable, "-m", "pip", "freeze", stdout=stream)
    activation = "\n".join(
        [
            "# Source this file from bash after installation.",
            f"source {shlex.quote(str(venv / 'bin/activate'))}",
            f"export CFACT_RUNTIME_ROOT={shlex.quote(str(root))}",
            f"export CFACT_SOURCE_ROOT={shlex.quote(str(checkout))}",
            "export MUJOCO_GL=egl",
            f"export HF_HOME={shlex.quote(str(root / 'cache/huggingface'))}",
            f"export HF_XET_CACHE={shlex.quote(str(root / 'cache/xet'))}",
            f"export CFACT_EXPORT_PYTHON={shlex.quote(str(root / 'export-venv/bin/python'))}",
            "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
            "export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}",
            "export PYTHONDONTWRITEBYTECODE=1",
            f"export NUMBA_CACHE_DIR={shlex.quote(str(root / 'cache/numba'))}",
            f"export MPLCONFIGDIR={shlex.quote(str(root / 'cache/matplotlib'))}",
            "",
        ]
    )
    (root / "activate.sh").write_text(activation)
    print(f"Runtime ready. Run: source {shlex.quote(str(root / 'activate.sh'))}")
    print("Next: cfact download-assets, then cfact doctor --render --gpu 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable, help="Python 3.10 executable")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("The tested collection profile currently targets Linux with NVIDIA EGL")
    bootstrap(args.runtime_root, args.python, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
