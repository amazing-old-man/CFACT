"""Explicit low-dimensional RoboCasa365 bridge for MimicGen preparation.

The historical MimicGen script imports broad robomimic training modules even
though preparation only needs three tiny utility surfaces. RoboCasa365 also
ships a modern environment wrapper whose observation module imports torch even
when images are disabled. Preparation is deliberately low-dimensional, so this
module installs narrow, process-local shims for exactly those unused-heavy
imports and leaves the official ``prepare_src_dataset`` function body intact.
"""

from __future__ import annotations

import json
import importlib
import sys
from types import ModuleType
from typing import Any

import h5py
import numpy as np


def _list_of_flat_dict_to_dict_of_list(
    rows: list[dict[str, Any]],
) -> dict[str, list[Any]]:
    """Transpose a non-empty list of equally-keyed flat dictionaries."""

    if not rows:
        return {}
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("MimicGen datagen dictionaries do not share identical keys")
    return {key: [row[key] for row in rows] for key in keys}


def _install_prepare_import_shims() -> None:
    """Install only the legacy utility API exercised by the official script."""

    file_utils = ModuleType("robomimic.utils.file_utils")

    def get_env_metadata_from_dataset(
        dataset_path: str,
        ds_format: str = "robomimic",
    ) -> dict[str, Any]:
        if ds_format != "robomimic":
            raise ValueError(f"unsupported prepare dataset format: {ds_format}")
        with h5py.File(dataset_path, "r") as dataset:
            value = dataset["data"].attrs["env_args"]
        if isinstance(value, bytes):
            value = value.decode()
        return json.loads(value) if isinstance(value, str) else dict(value)

    def url_is_alive(*_args: Any, **_kwargs: Any) -> bool:
        raise RuntimeError("network URL checks are unavailable in offline prepare mode")

    file_utils.get_env_metadata_from_dataset = get_env_metadata_from_dataset
    file_utils.url_is_alive = url_is_alive

    tensor_utils = ModuleType("robomimic.utils.tensor_utils")
    tensor_utils.list_of_flat_dict_to_dict_of_list = _list_of_flat_dict_to_dict_of_list

    # Replaced with SourceEnvUtils immediately after the upstream module loads.
    env_utils = ModuleType("robomimic.utils.env_utils")

    utils_package = importlib.import_module("robomimic.utils")
    for short_name, module in (
        ("file_utils", file_utils),
        ("tensor_utils", tensor_utils),
        ("env_utils", env_utils),
    ):
        setattr(utils_package, short_name, module)
        sys.modules[f"robomimic.utils.{short_name}"] = module

    mimicgen_file_utils = ModuleType("mimicgen.utils.file_utils")

    def get_all_demos_from_dataset(
        dataset_path: str,
        filter_key: str | None = None,
        start: int | None = None,
        n: int | None = None,
    ) -> list[str]:
        with h5py.File(dataset_path, "r") as dataset:
            if filter_key is None:
                demos = list(dataset["data"].keys())
            else:
                demos = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in dataset[f"mask/{filter_key}"][()]
                ]
        demos.sort(key=lambda value: int(value.rsplit("_", 1)[-1]))
        if start is not None:
            demos = demos[start:]
        if n is not None:
            demos = demos[:n]
        return demos

    mimicgen_file_utils.get_all_demos_from_dataset = get_all_demos_from_dataset
    mimicgen_utils_package = importlib.import_module("mimicgen.utils")
    setattr(mimicgen_utils_package, "file_utils", mimicgen_file_utils)
    sys.modules["mimicgen.utils.file_utils"] = mimicgen_file_utils


def lowdim_env_wrapper_class():
    """Load RoboCasa's wrapper with a no-image observation utility surface."""

    module_name = "robocasa.utils.robomimic.robomimic_obs_utils"
    obs_utils = ModuleType(module_name)
    obs_utils.OBS_KEYS_TO_MODALITIES = {}

    def initialize_obs_utils_with_obs_specs(specs: dict[str, Any]) -> None:
        mapping: dict[str, str] = {}
        for groups in specs.values():
            for modality, keys in groups.items():
                mapping.update({key: modality for key in keys})
        obs_utils.OBS_KEYS_TO_MODALITIES = mapping

    def key_is_obs_modality(key: str, obs_modality: str) -> bool:
        return obs_utils.OBS_KEYS_TO_MODALITIES.get(key) == obs_modality

    def process_obs(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("visual observation processing is disabled during preparation")

    obs_utils.initialize_obs_utils_with_obs_specs = initialize_obs_utils_with_obs_specs
    obs_utils.key_is_obs_modality = key_is_obs_modality
    obs_utils.process_obs = process_obs
    sys.modules[module_name] = obs_utils

    from robocasa.utils.robomimic.robomimic_env_wrapper import EnvRobocasa

    return EnvRobocasa


def synchronize_fixed_fixture_poses(env: Any) -> None:
    """Align fixed fixture Python references with a loaded source-model XML.

    RoboCasa's wrapper reloads MuJoCo XML but retains fixture objects created by
    the preceding reset.  Predicates such as the blender-lid target use those
    objects' XML poses, so a randomized pre-reset fixture pose can disagree with
    the loaded simulator even when the MuJoCo state is exact.
    """

    import robosuite.utils.transform_utils as transform_utils

    for fixture in getattr(env, "fixtures", {}).values():
        # Free auxiliary fixtures are dynamic objects; their pose must continue
        # to come from MuJoCo rather than being written into the static model.
        if not hasattr(fixture, "set_pos") or not hasattr(fixture, "set_euler"):
            continue
        try:
            body_id = env.sim.model.body_name2id(fixture.root_body)
        except (KeyError, ValueError):
            continue
        if int(body_id) < 0:
            continue
        if int(env.sim.model.body_jntnum[body_id]) > 0:
            continue
        # Fixture MJCF poses are relative to their parent body.  Using the
        # world-space values from ``sim.data.body_xpos/body_xquat`` works only
        # for fixtures parented directly under worldbody; nested fixtures such
        # as sinks would acquire the parent transform twice.  The compiled
        # model retains the exact source-XML local pose required here.
        fixture.set_pos(np.asarray(env.sim.model.body_pos[body_id], dtype=float))
        quat_wxyz = np.asarray(env.sim.model.body_quat[body_id], dtype=float)
        quat_xyzw = transform_utils.convert_quat(quat_wxyz, to="xyzw")
        fixture.set_euler(transform_utils.mat2euler(transform_utils.quat2mat(quat_xyzw)))
        # Some fixtures cache a target derived from their static XML pose.
        if hasattr(fixture, "lid_closed_pos"):
            fixture.lid_closed_pos = None
    env.update_state()


def robocasa_env_wrapper_class():
    """Return the low-dimensional wrapper with source-model pose repair."""

    BaseEnvRobocasa = lowdim_env_wrapper_class()

    class SourceEnvRobocasa(BaseEnvRobocasa):
        def reset_to(self, state):
            observation = super().reset_to(state)
            if "model" in state:
                synchronize_fixed_fixture_poses(self.env)
            return observation

    return SourceEnvRobocasa


def official_prepare_function():
    """Return official prepare with a process-local modern environment factory."""

    _install_prepare_import_shims()

    from robomimic.envs.env_base import EnvBase

    EnvRobocasa = robocasa_env_wrapper_class()
    from mimicgen.scripts import prepare_src_dataset as upstream
    from mimicgen.env_interfaces.base import make_interface
    from cfact.adapters.action_codec import install_action_codec

    def codec_interface(*args, **kwargs):
        return install_action_codec(make_interface(*args, **kwargs))

    def extract_with_codec(env, env_interface, initial_state, states, actions):
        """Read s[t], decode a[t], and retain desired-goal history across t.

        Source state files do not contain controller internals. Reconstruct
        command goals in episode order from the official reset initialization;
        refresh kinematics without executing a physical transition.
        """
        if len(states) != len(actions):
            raise ValueError("source requires one pre-action state per command")
        env.reset()
        env.reset_to(initial_state)
        interface = install_action_codec(env_interface)
        composite = interface.env.robots[0].composite_controller
        composite.update_state()
        composite.reset()
        rows = []
        for state, action in zip(states, actions):
            env.reset_to({"states": state})
            composite.update_state()
            for controller in composite.part_controllers.values():
                controller.new_update = True
            rows.append(interface.get_datagen_info(action=np.array(action, copy=True)).to_dict())
            # Use the real backend to carry desired goals, interpolation state,
            # and accumulated gripper commands. This does not call env.step.
            composite.set_goal(np.array(action, copy=True))
        result = interface.postprocess_datagen_info(_list_of_flat_dict_to_dict_of_list(rows))
        for key, values in result.items():
            if key in {"object_poses", "subtask_term_signals"}:
                result[key] = {k: np.asarray(v) for k, v in _list_of_flat_dict_to_dict_of_list(values).items()}
            else:
                result[key] = np.asarray(values)
        # The official writer serializes this result, so metadata is appended
        # after it closes the file by the wrapper below.
        controller_records.append(interface.action_codec.semantics())
        return result

    controller_records = []

    class SourceEnvRobocasa(EnvRobocasa, EnvBase):
        pass

    class SourceEnvUtils:
        @staticmethod
        def create_env_for_data_processing(
            env_meta: dict[str, Any],
            camera_names: list[str],
            camera_height: int,
            camera_width: int,
            reward_shaping: bool,
            seed: int | None = None,
        ) -> SourceEnvRobocasa:
            if camera_names:
                raise ValueError("modern MimicGen preparation must remain low-dimensional")
            kwargs = dict(env_meta["env_kwargs"])
            for key in (
                "env_name",
                "camera_names",
                "camera_height",
                "camera_width",
                "camera_heights",
                "camera_widths",
                "reward_shaping",
            ):
                kwargs.pop(key, None)
            if seed is not None:
                kwargs["seed"] = seed
            return SourceEnvRobocasa.create_for_data_processing(
                env_name=env_meta["env_name"],
                camera_names=camera_names,
                camera_height=camera_height,
                camera_width=camera_width,
                reward_shaping=reward_shaping,
                **kwargs,
            )

        @staticmethod
        def is_robosuite_env(env_meta: dict[str, Any]) -> bool:
            return int(env_meta["type"]) == 1

    upstream.EnvUtils = SourceEnvUtils
    upstream.make_interface = codec_interface
    upstream.extract_datagen_info_from_trajectory = extract_with_codec

    def prepare(**kwargs):
        controller_records.clear()
        upstream.prepare_src_dataset(**kwargs)
        output = kwargs.get("output_path") or kwargs["dataset_path"]
        demos = upstream.MG_FileUtils.get_all_demos_from_dataset(
            output, filter_key=kwargs.get("filter_key"), n=kwargs.get("n")
        )
        with h5py.File(output, "a") as dataset:
            for demo, semantics in zip(demos, controller_records):
                group = dataset[f"data/{demo}/datagen_info"]
                group.attrs["cfact_controller_semantics"] = json.dumps(semantics, sort_keys=True)
                group.attrs["cfact_goal_state_origin"] = "sequential_raw_prestate_and_official_reset; controller internals absent from raw"
                group.attrs["cfact_target_state_index"] = "pre_action_s[t]"
        return None

    return prepare
