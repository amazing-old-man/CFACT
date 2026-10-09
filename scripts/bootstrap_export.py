#!/usr/bin/env python3
"""Install the independent official LeRobot v3 export/reader environment."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from bootstrap_sim import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        parser.error("Run this installer with Python 3.10")
    root = args.runtime_root.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[1]
    if root == checkout:
        parser.error("The runtime must be a separate directory, such as .runtime")
    os.environ.setdefault("PIP_CACHE_DIR", str(root / "cache/pip"))
    venv = root / "export-venv"
    if not venv.exists():
        run(sys.executable, "-m", "venv", str(venv))
    python = str(venv / "bin/python")
    run(python, "-m", "pip", "install", "pip==25.3", "setuptools==80.9.0", "wheel==0.45.1")
    run(
        python,
        "-m",
        "pip",
        "install",
        "-c",
        str(checkout / "requirements/export-lock.txt"),
        "torch==2.7.1",
        "torchvision==0.22.1",
        "--index-url",
        "https://download.pytorch.org/whl/cpu",
    )
    run(
        python,
        "-m",
        "pip",
        "install",
        "-c",
        str(checkout / "requirements/export-lock.txt"),
        "-r",
        str(checkout / "requirements/export.txt"),
    )
    run(python, "-m", "pip", "install", "--no-deps", "-e", str(checkout))
    run(python, "-m", "pip", "check")
    run(
        python,
        "-c",
        'from cfact.export.shards import official_api; official_api(); print("LeRobot 0.4.0 import passed")',
    )
    with (root / "export-python-packages.txt").open("w") as stream:
        subprocess.run([python, "-m", "pip", "freeze"], stdout=stream, check=True)
    print(f"Export interpreter: {python}")
    print("The simulator activate.sh sets CFACT_EXPORT_PYTHON to this interpreter.")


if __name__ == "__main__":
    main()
