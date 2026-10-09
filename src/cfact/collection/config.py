"""Reject unsupported settings before freezing a plan or starting physical workers."""

from __future__ import annotations
import math


def validate_config(config: dict) -> None:
    def integer(section: str, key: str, minimum: int) -> int:
        value = config[section][key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{section}.{key} must be an integer >= {minimum}")
        return value

    for key in (
        "max_executed_options_per_node",
        "max_raw_proposals_per_generator",
        "max_selected_g0",
    ):
        integer("candidate_pool", key, 1)
    # Coverage selection always considers one candidate per nonempty family.
    # Setting max_selected_g0=0 is not an implemented way to disable G0.
    depth = integer("tree_limits", "maximum_decision_depth", 1)
    if depth > 3:
        raise ValueError("maximum_decision_depth above 3 is not implemented")
    for key in ("second_depth_pools", "third_depth_pools"):
        integer("tree_limits", key, 0)
    integer("tree_limits", "candidate_pools_per_root", 1)
    for key in (
        "control_hz",
        "local_timeout_steps",
        "max_env_steps_per_root",
        "max_task_path_steps",
        "interpolation_steps",
    ):
        integer("execution", key, 1)
    integer("execution", "continuation_steps", 0)
    fps = integer("video", "fps", 1)
    if fps != config["execution"]["control_hz"]:
        raise ValueError("video.fps must equal execution.control_hz; export never resamples")
    cameras = config["video"]["cameras"]
    if not cameras or len(cameras) != len(set(cameras)):
        raise ValueError("video.cameras must contain unique camera names")
    for key, minimum in (("maximum_output_gb", 0), ("minimum_host_free_gib", 0)):
        value = config["resource"][key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < minimum
        ):
            raise ValueError(f"resource.{key} must be a finite nonnegative number")
    if config["resource"]["maximum_output_gb"] == 0:
        raise ValueError("resource.maximum_output_gb must be positive")
    profile = config.get("engineering_profile")
    if profile is not None:
        for key in ("max_candidates_per_family", "max_children_per_parent",
                    "max_decision_parents_per_root", "max_workers"):
            integer("engineering_profile", key, 1)
        if profile.get("name") == "coverage":
            from cfact.collection.manifest_runner import read_profile

            if profile != read_profile("coverage"):
                raise ValueError("coverage must use the frozen unmodified engineering profile")
