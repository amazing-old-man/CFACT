"""Explicit runtime snapshots for the pinned MuJoCo / robosuite backend.

Only value state is copied: official simulator state, active controller values,
interpolators, gripper commands, task/fixture values and actual RNG engines.
No Python object graph is pickled and restoration never steps the simulator.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

SNAPSHOT_SCHEMA = "cfact.backend_snapshot.v1"
DATA_FIELDS = ("ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata", "eq_active")
MODEL_FIELDS = ("body_pos", "body_quat", "geom_rgba", "site_rgba", "site_size", "mat_rgba", "mat_emission", "eq_active0")


def _is_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return True
    if isinstance(value, np.ndarray):
        return value.dtype.kind in "biufcSU" or (
            value.dtype.kind == "O" and all(_is_value(item) for item in value.flat)
        )
    if isinstance(value, (list, tuple)):
        return all(_is_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, (str, int)) and _is_value(v) for k, v in value.items())
    return False


def _values(obj: Any) -> dict[str, Any]:
    return {name: copy.deepcopy(value) for name, value in vars(obj).items()
            if not name.startswith("_cfact") and name not in {"_obs_cache", "_observations"}
            and _is_value(value)}


def _restore_values(obj: Any, values: dict[str, Any]) -> None:
    for name, value in values.items():
        setattr(obj, name, copy.deepcopy(value))


def _rngs(obj: Any) -> dict[str, Any]:
    result = {}
    for name, rng in vars(obj).items():
        if isinstance(rng, np.random.Generator):
            result[name] = {"kind": "Generator", "state": copy.deepcopy(rng.bit_generator.state)}
        elif isinstance(rng, np.random.RandomState):
            result[name] = {"kind": "RandomState", "state": copy.deepcopy(rng.get_state())}
        elif isinstance(rng, random.Random):
            result[name] = {"kind": "Random", "state": copy.deepcopy(rng.getstate())}
    return result


def _restore_rngs(obj: Any, values: dict[str, Any]) -> None:
    for name, entry in values.items():
        rng = getattr(obj, name)
        if entry["kind"] == "Generator":
            rng.bit_generator.state = copy.deepcopy(entry["state"])
        elif entry["kind"] == "RandomState":
            rng.set_state(entry["state"])
        else:
            rng.setstate(entry["state"])


def _robot_state(robot: Any) -> dict:
    composite = robot.composite_controller
    parts = {}
    for name, controller in composite.part_controllers.items():
        parts[name] = {
            "class": f"{type(controller).__module__}.{type(controller).__name__}",
            "values": _values(controller),
            "interpolators": {
                key: _values(getattr(controller, key))
                for key in ("interpolator", "interpolator_pos", "interpolator_ori")
                if getattr(controller, key, None) is not None
            },
        }
    buffers = {}
    for name, value in vars(robot).items():
        if not name.startswith("recent_"):
            continue
        if isinstance(value, dict):
            buffers[name] = {key: _values(item) for key, item in value.items()}
        elif hasattr(value, "__dict__"):
            buffers[name] = _values(value)
    return {
        "values": _values(robot), "composite": _values(composite), "controllers": parts,
        "grippers": {name: _values(gripper) for name, gripper in robot.gripper.items()},
        "buffers": buffers, "rng": _rngs(robot),
    }


def capture_snapshot(env: Any) -> dict[str, Any]:
    base = env.env
    # Use the official simulator interface directly. Wrapper get_state also
    # calls task.get_ep_meta(), which can randomly select language and consume
    # the environment RNG just by taking a snapshot.
    state = np.asarray(base.sim.get_state().flatten()).copy()
    model = base.sim.model.get_xml() if hasattr(base.sim.model, "get_xml") else env.get_state().get("model")
    return {
        "schema": SNAPSHOT_SCHEMA, "states": state,
        "sim_time_s": float(base.sim.data.time),
        "time_origin": "official_simulator_state",
        "official_metadata": {"state_api": "sim.get_state().flatten"},
        "model_sha256": hashlib.sha256(model.encode()).hexdigest() if model else None,
        "sim_data": {name: np.array(getattr(base.sim.data, name), copy=True) for name in DATA_FIELDS if hasattr(base.sim.data, name)},
        "model_values": {name: np.array(getattr(base.sim.model, name), copy=True) for name in MODEL_FIELDS if hasattr(base.sim.model, name)},
        "environment": _values(base), "wrapper": _values(env),
        "observables": {name: _values(observable) for name, observable in getattr(base, "_observables", {}).items()},
        "observation_cache": copy.deepcopy(getattr(base, "_obs_cache", {})),
        "robots": [_robot_state(robot) for robot in base.robots],
        "fixtures": {name: {"values": _values(fixture), "rng": _rngs(fixture)} for name, fixture in getattr(base, "fixtures", {}).items()},
        "rng": {"numpy_global": np.random.get_state(), "python_global": random.getstate(), "environment": _rngs(base)},
        "capabilities": {
            "physics": True, "controller_goal_and_interpolation": True,
            "gripper_internal_command": True, "task_fixture_value_state": True,
            "rng": True, "render_bit_exact": False,
            "excluded": ["unused plugins", "renderer caches", "opaque non-value task extensions"],
        },
    }


def restore_snapshot(env: Any, snapshot: dict[str, Any], initial: dict | None = None,
                     *, allow_legacy: bool = False) -> dict[str, Any]:
    started = time.monotonic()
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        if not allow_legacy:
            raise ValueError("legacy_snapshot_missing_controller_gripper_rng: limited restore only")
        if initial is not None:
            env.reset_to(initial)
        env.reset_to({"states": snapshot["states"]})
        return {"complete": False, "reason": "legacy_physics_only", "restore_wall_s": time.monotonic() - started}
    required = {"states", "sim_time_s", "sim_data", "robots", "fixtures", "rng", "environment", "wrapper"}
    if not required.issubset(snapshot):
        raise ValueError("incomplete_backend_snapshot")
    if initial is not None:
        # Reset before restoring controller and gripper values, never after.
        env.reset_to(initial)
    base = env.env
    if len(base.robots) != len(snapshot["robots"]):
        raise ValueError("snapshot_robot_layout_mismatch")
    for name, value in snapshot.get("model_values", {}).items():
        getattr(base.sim.model, name)[...] = value
    model = base.sim.model.get_xml() if hasattr(base.sim.model, "get_xml") else env.get_state().get("model")
    actual_model_hash = hashlib.sha256(model.encode()).hexdigest() if model else None
    if snapshot.get("model_sha256") is None or actual_model_hash != snapshot["model_sha256"]:
        raise ValueError("snapshot_model_identity_mismatch")
    base.sim.set_state_from_flattened(snapshot["states"])
    # forward recomputes derived MuJoCo arrays; it must not call env.update_state,
    # which can trigger fixture events or increment timers at restore time.
    base.sim.forward()
    for name, value in snapshot["sim_data"].items():
        getattr(base.sim.data, name)[...] = value
    _restore_values(base, snapshot["environment"])
    _restore_values(env, snapshot["wrapper"])
    for name, values in snapshot.get("observables", {}).items():
        _restore_values(base._observables[name], values)
    if hasattr(base, "_obs_cache"):
        base._obs_cache = copy.deepcopy(snapshot.get("observation_cache", {}))
    for robot, saved in zip(base.robots, snapshot["robots"]):
        _restore_values(robot, saved["values"])
        _restore_values(robot.composite_controller, saved["composite"])
        if set(robot.composite_controller.part_controllers) != set(saved["controllers"]):
            raise ValueError("snapshot_controller_layout_mismatch")
        for name, data in saved["controllers"].items():
            controller = robot.composite_controller.part_controllers[name]
            if data["class"] != f"{type(controller).__module__}.{type(controller).__name__}":
                raise ValueError("snapshot_controller_class_mismatch")
            _restore_values(controller, data["values"])
            for attr, values in data["interpolators"].items():
                _restore_values(getattr(controller, attr), values)
        for name, values in saved["grippers"].items():
            _restore_values(robot.gripper[name], values)
        for name, values in saved["buffers"].items():
            buffer = getattr(robot, name)
            if isinstance(buffer, dict):
                for key, entry in values.items():
                    _restore_values(buffer[key], entry)
            else:
                _restore_values(buffer, values)
        _restore_rngs(robot, saved["rng"])
    for name, saved in snapshot["fixtures"].items():
        _restore_values(base.fixtures[name], saved["values"])
        _restore_rngs(base.fixtures[name], saved["rng"])
    _restore_rngs(base, snapshot["rng"]["environment"])
    np.random.set_state(snapshot["rng"]["numpy_global"])
    random.setstate(snapshot["rng"]["python_global"])
    actual = np.asarray(base.sim.get_state().flatten())
    if not np.array_equal(actual, snapshot["states"]) or float(base.sim.data.time) != snapshot["sim_time_s"]:
        raise RuntimeError("snapshot_restore_changed_physical_state_or_time")
    return {"complete": True, "restore_wall_s": time.monotonic() - started,
            "physics_steps": 0, "sim_time_s": snapshot["sim_time_s"], "capabilities": snapshot["capabilities"]}


def _pack(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            if not _is_value(value):
                raise TypeError("snapshot_opaque_object_array_not_supported")
            return {"__object_array__": [_pack(item) for item in value.flat], "shape": value.shape}
        return {"__array__": base64.b64encode(value.tobytes()).decode(), "dtype": value.dtype.str, "shape": value.shape}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {"__tuple__": [_pack(item) for item in value]}
    if isinstance(value, dict):
        return {"__dict__": [[_pack(k), _pack(v)] for k, v in value.items()]}
    if isinstance(value, list):
        return [_pack(item) for item in value]
    return value


def _unpack(value: Any) -> Any:
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__object_array__" in value:
        entries = [_unpack(item) for item in value["__object_array__"]]
        result = np.empty(len(entries), dtype=object)
        result[:] = entries
        return result.reshape(value["shape"])
    if "__array__" in value:
        return np.frombuffer(base64.b64decode(value["__array__"]), dtype=value["dtype"]).reshape(value["shape"]).copy()
    if "__tuple__" in value:
        return tuple(_unpack(item) for item in value["__tuple__"])
    return {_unpack(k): _unpack(v) for k, v in value["__dict__"]}


def save_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(_pack(snapshot), separators=(",", ":")))
    temporary.replace(path)


def load_snapshot(path: Path) -> dict[str, Any]:
    return _unpack(json.loads(Path(path).read_text()))
