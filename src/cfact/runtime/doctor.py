"""Report actionable collection runtime checks; optionally test actual EGL rendering."""

from __future__ import annotations
import argparse
import contextlib
import importlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from cfact.io.serialization import atomic_json
from cfact.runtime.assets import load_lock, runtime_root


def check_runtime(root: Path, render: bool = False, gpu: int = 0) -> dict:
    checks = []

    def check(name, function):
        try:
            detail = function()
            checks.append({"check": name, "status": "PASS", "detail": detail})
        except Exception as error:
            checks.append({"check": name, "status": "FAIL", "detail": str(error)})

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)
        return message

    lock = load_lock()
    check(
        "python",
        lambda: require(
            sys.version_info[:2] == (3, 10),
            f"Python 3.10 required, current: {sys.version.split()[0]}",
        ),
    )
    check(
        "runtime_lock",
        lambda: require(
            json.loads((root / "runtime-lock.json").read_text()) == lock,
            "Runtime lock must match packaged runtime.json",
        ),
    )
    for name, spec in lock["repositories"].items():

        def source_check(name=name, spec=spec):
            path = root / "sources" / name
            head = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            require(head == spec["revision"], f"{name} revision differs: {head}")
            dirty = subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
            require(not dirty, f"{name} has tracked modifications: {dirty}")
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                module = importlib.import_module(name)
            require(
                Path(module.__file__).resolve().is_relative_to(path.resolve()),
                f"{name} imported from unexpected location: {module.__file__}",
            )
            return {"revision": head, "path": module.__file__}

        check("source:" + name, source_check)
    for name, expected in {
        "numpy": "2.2.5",
        "mujoco": "3.3.1",
        "numba": "0.61.2",
        "scipy": "1.15.3",
        "gymnasium": "0.29.1",
    }.items():
        check(
            "package:" + name,
            lambda n=name, e=expected: require(
                version(n) == e, f"{n}=={e}; installed {version(n)}"
            ),
        )
    for item in lock["assets"]:
        folder = root / "sources/robocasa/robocasa/models/assets" / item["folder"]
        check(
            "assets:" + item["name"],
            lambda p=folder: require(
                p.is_dir() and any(f.is_file() for f in p.rglob("*")), f"Asset directory: {p}"
            ),
        )

    def ffmpeg():
        executable = shutil.which("ffmpeg")
        require(executable is not None, "Install system FFmpeg (sudo apt-get install ffmpeg)")
        result = subprocess.run(
            [executable, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True
        )
        require("libx264" in result.stdout, "FFmpeg must provide libx264")
        return executable

    check("ffmpeg", ffmpeg)

    def gpu_check():
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader"], text=True
        )
        require(
            any(line.split(",")[0].strip() == str(gpu) for line in output.splitlines()),
            f"Physical GPU index {gpu} is unavailable",
        )
        return output.strip()

    check("nvidia_gpu", gpu_check)
    if render:

        def egl():
            env = os.environ.copy()
            env.update({"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu)})
            code = """import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, height=32, width=32)
r.update_scene(d)
a = r.render()
assert a.shape == (32, 32, 3)
r.close()
print('EGL 32x32 RGB render passed')
"""
            result = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=90
            )
            require(result.returncode == 0, result.stderr[-3000:] or result.stdout)
            return result.stdout.strip()

        check("egl_render", egl)
    return {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "runtime_root": str(root),
        "python": sys.executable,
        "checks": checks,
        "scope": "Asset presence and optional EGL; use quickstart to validate actual task replay.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional JSON diagnostic report")
    args = parser.parse_args()
    result = check_runtime(runtime_root(args.runtime_root), args.render, args.gpu)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
