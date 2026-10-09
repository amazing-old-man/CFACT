"""Validate authoritative replay fields without inventing missing state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from cfact.io.serialization import natural_key
from cfact.sources.manifest import sha256_file


def validate_raw(path: Path, task: str | None = None) -> dict:
    rows = []
    with h5py.File(path, "r") as source:
        data = source["data"]
        env = json.loads(data.attrs["env_args"])
        if not env.get("env_name") or (task and env["env_name"] != task):
            raise ValueError("Source task identity mismatch")
        kwargs = env.get("env_kwargs", {})
        frequency = kwargs.get("control_freq", 20)
        frequency_origin = "dataset_env_args" if "control_freq" in kwargs else "pinned_RoboCasa_Kitchen_default"
        if frequency != 20:
            raise ValueError("Frozen execution requires authoritative control_freq=20")
        if not kwargs.get("controller_configs"):
            raise ValueError("Missing recorded controller configuration")
        for name in sorted(data, key=natural_key):
            demo = data[name]
            actions, states = np.asarray(demo["actions"]), np.asarray(demo["states"])
            if actions.ndim != 2 or actions.shape[1] != 12 or not len(actions):
                raise ValueError(f"{name}: expected nonempty [T,12] recorded actions")
            if states.ndim != 2 or len(states) != len(actions) or states.shape[1] < 3:
                raise ValueError(f"{name}: invalid aligned MuJoCo state rows")
            if not np.isfinite(actions).all() or not np.isfinite(states).all():
                raise ValueError(f"{name}: nonfinite actions/states")
            intervals = np.diff(states[:, 0])
            if (intervals <= 0).any():
                raise ValueError(f"{name}: non-increasing recorded simulation clock")
            irregular = np.flatnonzero(~np.isclose(intervals, .05, atol=1e-6, rtol=0))
            xml = demo.attrs["model_file"]
            if ET.fromstring(xml).tag != "mujoco":
                raise ValueError(f"{name}: invalid scene XML")
            meta = json.loads(demo.attrs["ep_meta"])
            if not isinstance(meta, dict) or not meta:
                raise ValueError(f"{name}: missing episode metadata")
            rows.append({"demo": name, "original_episode_id": str(demo.attrs.get("original_episode_id", name)),
                         "steps": len(actions), "state_width": states.shape[1],
                         "start_time_s": float(states[0, 0]), "end_time_s": float(states[-1, 0]),
                         "irregular_clock_intervals": [{"after_row": int(i), "delta_s": float(intervals[i])}
                                                       for i in irregular]})
        if not rows:
            raise ValueError("Source has no episodes")
    return {"status": "SCHEMA_VALID_REPLAY_NOT_TESTED", "sha256": sha256_file(path),
            "task": env["env_name"], "control_hz": 20, "control_hz_origin": frequency_origin, "env_args": env,
            "episodes": rows, "physics_replay_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", action="store_true", help="Actually replay one fixed demo under resource admission")
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/quickstart.yaml"))
    args = parser.parse_args()
    report = validate_raw(args.source, args.task)
    if args.replay:
        from cfact.run import load_config, parameters
        from cfact.runtime.resources import admission
        from cfact.sources.replay import validate_replay
        config = parameters(load_config(args.config), 1, None)
        gate = admission(Path.cwd(), [args.gpu], 1, config["resource"])
        if gate["status"] != "PASS":
            if args.output:
                from cfact.io.serialization import atomic_json
                atomic_json(args.output, gate)
            raise RuntimeError("NOT_RUN_RESOURCE_BLOCKED")
        report["replay"] = validate_replay(args.source, args.demo, args.gpu)
        report["physics_replay_verified"] = True
        report["status"] = "REPLAY_EXECUTED"
    if args.output:
        from cfact.io.serialization import atomic_json
        atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
