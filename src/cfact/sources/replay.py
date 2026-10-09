"""Measure Source replay and snapshot repeatability with the frozen simulator adapter."""

from pathlib import Path

import h5py
import numpy as np

from cfact.collection.runtime import _create_env, _load_root
from cfact.execution.snapshot import capture_snapshot, restore_snapshot


def validate_replay(path: Path, demo: str, gpu: int, seed: int = 20260922) -> dict:
    root = _load_root(path, demo)
    env = _create_env(root["env_meta"], seed, "robot0_agentview_left", gpu)
    try:
        with h5py.File(path, "r") as source:
            recorded = np.asarray(source[f"data/{demo}/states"])
        env.reset_to(root["initial"])
        initial = np.asarray(env.env.sim.get_state().flatten()).copy()
        if not np.array_equal(initial, recorded[0]):
            raise ValueError("Source initial qpos/qvel/time restoration differs from official recorded state")
        snapshot = capture_snapshot(env)
        env.step(root["actions"][0])
        first = np.asarray(env.env.sim.get_state().flatten()).copy()
        restored = restore_snapshot(env, snapshot, initial=root["initial"])
        if not restored["complete"] or not np.array_equal(env.env.sim.get_state().flatten(), initial):
            raise ValueError("Source snapshot restore did not recover the exact initial state")
        env.step(root["actions"][0])
        repeated = np.asarray(env.env.sim.get_state().flatten()).copy()
        if not np.allclose(first, repeated, atol=1e-10, rtol=0):
            raise ValueError("Independent action replay from the same snapshot is not repeatable")
        restore_snapshot(env, snapshot, initial=root["initial"])
        success, errors = [], []
        for index, action in enumerate(root["actions"]):
            env.step(action)
            success.append(bool(env.is_success()["task"]))
            if index + 1 < len(recorded):
                state = np.asarray(env.env.sim.get_state().flatten())
                errors.append({"action_index": index,
                    "clock_difference_s": float(state[0] - recorded[index + 1, 0]),
                    "maximum_state_difference": float(np.max(np.abs(state[1:] - recorded[index + 1, 1:])))})
        return {"status": "REPLAY_EXECUTED", "demo": demo, "steps": len(root["actions"]),
                "initial_state_exact": True, "snapshot_restore_exact": True,
                "independent_first_action_max_abs_difference": float(np.max(np.abs(first - repeated))),
                "official_success_first": next((i for i, value in enumerate(success) if value), None),
                "official_success_ever": any(success), "official_success_last": success[-1],
                "recorded_state_comparison": errors,
                "recorded_trajectory_bit_exact": all(abs(e["clock_difference_s"]) < 1e-9 and e["maximum_state_difference"] < 1e-9 for e in errors),
                "recorded_controller_internals_available": False,
                "scope": "Recorded initial state restored; recorded actions executed unchanged. Raw Source omits controller internals; recorded-trajectory differences are reported, never repaired."}
    finally:
        env.env.close()
