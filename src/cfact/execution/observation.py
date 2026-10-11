"""Current-state observations on private MjData, without changing execution.

The controller / predicates retain the backend's native cache semantics. Only
archived observations and labels use this explicitly refreshed boundary.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import dataclass

import numpy as np

PHASE = "post_control_integration_shadow_forward.v1"
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
SCENE_FIELDS = (
    "body_pos", "body_quat", "geom_pos", "geom_quat", "geom_rgba", "site_pos",
    "site_quat", "site_rgba", "site_size", "mat_rgba", "mat_emission",
    "cam_pos", "cam_quat", "cam_fovy", "cam_bodyid", "cam_mode", "eq_active0",
)


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@dataclass
class Observation:
    sim: object
    state: np.ndarray
    integration_state: np.ndarray | None
    metadata: dict

    def render(self, *, height, width, camera_name, depth=False):
        source = getattr(self.sim, "_cfact_execution_sim", None)
        if source is not None:
            from robosuite.utils.binding_utils import MjRenderContextOffscreen

            original = source._render_context_offscreen
            if original is None:
                raise RuntimeError("observation_requires_initialized_native_renderer")
            if self.sim._render_context_offscreen is None:
                context = MjRenderContextOffscreen(self.sim, device_id=original.device_id)
                # Match native visual/collision visibility and render options exactly.
                context.vopt = copy.copy(original.vopt)
                context.scn.flags[:] = original.scn.flags
            else:
                self.sim._render_context_offscreen.vopt = copy.copy(original.vopt)
                self.sim._render_context_offscreen.scn.flags[:] = original.scn.flags
        kwargs = dict(height=height, width=width, camera_name=camera_name)
        if depth:
            kwargs["depth"] = True
        value = self.sim.render(**kwargs)
        if isinstance(value, tuple):
            return tuple(np.array(v, copy=True) for v in value)
        return np.array(value, copy=True)

    def archive_values(self):
        if self.integration_state is None:
            return {}
        result = {"integration_state": self.integration_state.copy()}
        for name in SCENE_FIELDS:
            if hasattr(self.sim.model, name):
                result["model_" + name] = np.array(getattr(self.sim.model, name), copy=True)
        return result


def observe(env) -> Observation:
    """Copy mjSTATE_INTEGRATION, forward the copy, never step or forward live data."""
    source = env.env.sim
    state = np.asarray(source.get_state().flatten()).copy()
    model = getattr(getattr(source, "model", None), "_model", None)
    data = getattr(source.data, "_data", None)
    if model is None and data is None:
        # Small non-MuJoCo test doubles do not claim the physical phase contract.
        return Observation(source, state, None, {"observation_phase": "non_mujoco_test_double"})
    if model is None or data is None:
        raise TypeError("observation_requires_native_model_and_data")
    if model.nplugin:
        raise ValueError("opaque_plugin_visual_state_not_supported")
    import mujoco
    from robosuite.utils.binding_utils import MjSim

    shadow = getattr(source, "_cfact_observation_sim", None)
    if shadow is None:
        shadow = MjSim(model)
        shadow._cfact_execution_sim = source
        source._cfact_observation_sim = shadow
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, values, spec)
    if not np.isfinite(values).all():
        raise ValueError("observation_nonfinite_integration_state")
    mujoco.mj_setState(model, shadow.data._data, values, spec)
    mujoco.mj_forward(model, shadow.data._data)
    after = np.empty_like(values)
    mujoco.mj_getState(model, data, after, spec)
    if not np.array_equal(values, after) or not np.array_equal(state, source.get_state().flatten()):
        raise RuntimeError("observation_changed_execution_state")
    scene = {name: array_sha(getattr(model, name)) for name in SCENE_FIELDS if hasattr(model, name)}
    cameras = []
    for name in CAMERAS:
        cid = source.model.camera_name2id(name)
        cameras.append({"name": name, "id": int(cid), "fovy": float(model.cam_fovy[cid]),
                        "pos": model.cam_pos[cid].tolist(), "quat": model.cam_quat[cid].tolist(),
                        "body": int(model.cam_bodyid[cid]),
                        "world_pos": shadow.data.cam_xpos[cid].tolist(),
                        "world_rot": shadow.data.cam_xmat[cid].tolist()})
    renderer = source._render_context_offscreen
    render_options = {}
    if renderer is not None:
        for name in ("geomgroup", "sitegroup", "jointgroup", "tendongroup", "actuatorgroup", "flags"):
            render_options[name] = np.asarray(getattr(renderer.vopt, name)).tolist()
    metadata = {
        "observation_phase": PHASE, "pose_source": "shadow_mj_forward",
        "state_sha256": array_sha(state), "integration_state_sha256": array_sha(values),
        "scene_sha256": hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest(),
        "camera_config_sha256": hashlib.sha256(json.dumps(cameras, sort_keys=True).encode()).hexdigest(),
        "camera_config": cameras, "sim_time_s": float(data.time),
        "physics_tick": int(round(data.time / model.opt.timestep)),
        "physics_timestep_s": float(model.opt.timestep),
        "integration_state_spec": int(spec),
        "render_options": render_options,
        "render_scene_flags": np.asarray(renderer.scn.flags).tolist() if renderer is not None else None,
        "refresh_action": "mj_forward_on_private_MjData", "capture_integration_steps": 0,
    }
    return Observation(shadow, state, values, metadata)


def _shadow_interface_view(interface, sim):
    """Rebind Python descriptors, rejecting opaque instance captures of live data.

    resolve_role_binding restores get_object_pose as an instance attribute.
    A shallow copy keeps that bound method's original __self__, so redirecting
    env.sim alone does not redirect the actual kinematic read.
    """
    view = copy.copy(interface)
    if view is interface:
        raise TypeError("observation_unsupported_binding_copy")
    environment = copy.copy(interface.env)
    if environment is interface.env:
        raise TypeError("observation_unsupported_binding_copy")
    view.env = environment
    view.env.sim = sim
    for name, value in vars(interface).items():
        if not callable(value):
            continue
        if (not inspect.ismethod(value) or value.__self__ is not interface
                or value.__func__.__closure__):
            raise TypeError(f"observation_unsupported_binding:{name}")
        setattr(view, name, value.__func__.__get__(view, type(view)))
    required = ["get_object_poses", "get_robot_eef_pose"]
    if hasattr(interface, "get_object_pose"):
        required.append("get_object_pose")
    for name in required:
        getter = getattr(view, name, None)
        if (not inspect.ismethod(getter) or getter.__self__ is not view
                or getter.__func__.__closure__):
            raise TypeError(f"observation_unsupported_binding:{name}")
    return view


def observation_poses(env, interface):
    """Read only pose APIs through a private view; never rerun task predicates."""
    if interface.env.sim is not env.env.sim:
        raise ValueError("observation_interface_sim_binding_mismatch")
    captured = observe(env)
    if captured.integration_state is None:
        return None
    view = _shadow_interface_view(interface, captured.sim)
    poses, eef = view.get_object_poses(), view.get_robot_eef_pose()
    if not isinstance(poses, dict) or not poses:
        raise ValueError("observation_missing_pose_roles")
    for pose in [*poses.values(), eef]:
        value = np.asarray(pose)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError("observation_invalid_current_pose")
    captured.metadata["pose_binding"] = "private_interface_python_descriptors.v1"
    return poses, eef, captured.metadata
