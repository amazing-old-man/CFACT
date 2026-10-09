"""Freeze explicit prepared sources and donors into a portable collection plan."""

from __future__ import annotations
import argparse
import copy
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import h5py
import numpy as np
from cfact import __version__
from cfact.io.serialization import atomic_json
from cfact.tasks import target_task
from cfact.paths import workspace_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_plan(specification: Path, output: Path, parameters: Path | None = None) -> dict:
    specification = specification.resolve()
    requested = json.loads(specification.read_text())
    config = (
        json.loads(parameters.read_text())
        if parameters
        else json.loads(files("cfact.resources").joinpath("collection.json").read_text())
    )
    config = copy.deepcopy(config)
    from cfact.collection.config import validate_config

    validate_config(config)
    policies = json.loads(files("cfact.resources").joinpath("tasks.json").read_text())
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new plan directory: {output}")
    if not requested.get("tasks"):
        raise ValueError("sources must contain a nonempty tasks list")
    requests, prepared_files, task_rows, config_rows = {}, {}, [], []
    seen_tasks = set()
    for row in requested["tasks"]:
        task = row["task"]
        target_task(task)
        if task in seen_tasks:
            raise ValueError(f"duplicate task: {task}")
        seen_tasks.add(task)
        path = Path(row["prepared_hdf5"]).expanduser()
        path = (specification.parent / path).resolve() if not path.is_absolute() else path.resolve()
        roots, donors = row["roots"], row.get("donors", [])
        for group in [*roots, *donors]:
            if not isinstance(group, str) or not group or "/" in group or group in {".", ".."}:
                raise ValueError(f"invalid HDF5 episode group: {group!r}")
        if not roots or len(set(roots)) != len(roots) or len(set(donors)) != len(donors):
            raise ValueError("roots must be nonempty; roots and donors must each be unique")
        if set(roots) & set(donors):
            raise ValueError("root and donor groups must be disjoint")
        file_hash = sha256_file(path)
        from cfact.sources.prepare import validate_prepared_identity

        prepared_identity = validate_prepared_identity(path, required_episodes=roots + donors)
        policy = policies[task]
        with h5py.File(path, "r") as dataset:
            meta = dataset["data"].attrs["env_args"]
            if isinstance(meta, bytes):
                meta = meta.decode()
            if json.loads(meta)["env_name"] != task:
                raise ValueError(f"source task metadata differs from {task}")
            provenance = json.loads(dataset["data"].attrs.get("cfact_source_provenance", "{}"))
            original_demo_ids = {}
            source_videos = {}
            content_groups = {}
            for group in [*roots, *donors]:
                episode = dataset[f"data/{group}"]
                source_videos[group] = json.loads(episode.attrs.get("cfact_source_videos", "{}"))
                episode_index = episode.attrs.get("lerobot_episode_index")
                original_demo_ids[group] = (
                    f"episode_{int(episode_index):06d}" if episode_index is not None else group
                )
                actions, states = episode["actions"], episode["states"]
                if (
                    actions.ndim != 2
                    or states.ndim != 2
                    or not len(actions)
                    or len(actions) != len(states)
                    or not np.isfinite(actions[()]).all()
                    or not np.isfinite(states[()]).all()
                ):
                    raise ValueError(f"invalid source actions/states: {task}/{group}")
                for key in ("model_file", "ep_meta"):
                    if key not in episode.attrs:
                        raise ValueError(f"missing replay attribute {key}: {group}")
                for key in (
                    "eef_pose",
                    "target_pose",
                    "gripper_action",
                    "object_poses/" + policy["event_object_ref"],
                    "object_poses/" + policy["operation_object_ref"],
                    "subtask_term_signals/" + policy["event_signal"],
                ):
                    if len(episode["datagen_info/" + key]) != len(actions):
                        raise ValueError(f"unaligned datagen field {key}: {group}")
                content = hashlib.sha256(
                    actions[()].tobytes()
                    + states[()].tobytes()
                    + str(episode.attrs["model_file"]).encode()
                    + str(episode.attrs["ep_meta"]).encode()
                ).hexdigest()
                if content in content_groups:
                    raise ValueError(
                        f"duplicate recorded trajectory in {group} and {content_groups[content]}"
                    )
                content_groups[content] = group
        prepared_files[str(path)] = {
            "task": task,
            "sha256": file_hash,
            "groups": sorted(set(roots + donors)),
            "prepared_identity": prepared_identity,
        }

        def route(group):
            package_hash = provenance.get("package_sha256")
            original_id = original_demo_ids[group]
            source_id = (
                f"RoboCasa@sha256:{package_hash}:{task}:{original_id}"
                if package_hash
                else f"sha256:{file_hash}:{group}"
            )
            identity = {
                "source_id": source_id,
                "original_episode_id": original_id,
                "prepared_sha256": file_hash,
                **provenance,
            }
            return {
                "task": task,
                "source_id": source_id,
                "original_episode_id": original_id,
                "episode_id": source_id,
                "prepared_hdf5": str(path),
                "prepared_group": group,
                "prepared_sha256": file_hash,
                "source_class": "user_prepared_dataset",
                "source_identity": identity,
                "source_identity_sha256": digest_json(identity),
                "source_video_paths": {
                    camera: row["path"] for camera, row in source_videos[group].items()
                },
                "source_video_sha256": {
                    camera: row["sha256"] for camera, row in source_videos[group].items()
                },
                "source_video_status": "SOURCE_VIDEO_PRESENT"
                if source_videos[group]
                else "SOURCE_VIDEO_MISSING",
            }

        donor_rows = [route(group) for group in donors]
        for group in roots:
            item = route(group)
            request_id = digest_json({"task": task, "source": item["source_id"]})[:24]
            requests[request_id] = {
                **item,
                "request_id": request_id,
                "root_id": request_id,
                "request_status": "REQUESTED_NEW",
                "role_in_release": "USER_DECLARED_ROOT",
                "donor_source_ids": [donor["source_id"] for donor in donor_rows],
            }
        task_rows.append(
            {
                "task": task,
                "status": "READY_FOR_DEVELOPMENT",
                "event_signal": policy["event_signal"],
                "donors": donor_rows,
            }
        )
        config_rows.append({"task": task, "prepared_hdf5": str(path), **policy})
    # All validation above is read-only; a plan is only created after it passes.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "2.0.0",
        "manifest_status": "FROZEN_EXPLICIT_SOURCES",
        "release_id": output.name,
        "new_request_count": len(requests),
        "queue_order": list(requests),
        "pilot_request_ids": list(requests)[:1],
        "tasks": task_rows,
        "requests_by_id": requests,
        "prepared_files": prepared_files,
        "raw_source_routes": "source_stage/raw_source_routes.json",
        "prepared_source_manifest": "source_stage/prepared_source_manifest.json",
    }
    raw_path = output / manifest["raw_source_routes"]
    prepared_path = output / manifest["prepared_source_manifest"]
    atomic_json(
        raw_path,
        {
            "source_specification_sha256": sha256_file(specification),
            "routes": list(requests.values()),
        },
    )
    atomic_json(prepared_path, {"status": "PASS", "files": prepared_files})
    manifest["raw_source_routes_sha256"] = sha256_file(raw_path)
    manifest["prepared_source_manifest_sha256"] = sha256_file(prepared_path)
    config.update(
        {
            "generator_revision": f"cfact-{__version__}",
            "tasks": config_rows,
            "output_root": str(
                (
                    workspace_root()
                    / requested.get(
                        "output_root",
                        os.environ.get(
                            "CFACT_OUTPUT_ROOT",
                            config.get("storage", {}).get("output_root", str(output / "runs")),
                        ),
                    )
                ).resolve()
            ),
            "release_root": str(output),
            "prepared_source_manifest_sha256": manifest["prepared_source_manifest_sha256"],
            "raw_routes_sha256": manifest["raw_source_routes_sha256"],
        }
    )
    atomic_json(output / "sources.json", requested)
    atomic_json(output / "config.json", config)
    atomic_json(output / "source_manifest.json", manifest)
    return {
        "plan": str(output),
        "tasks": len(task_rows),
        "roots": len(requests),
        "config": str(output / "config.json"),
        "manifest": str(output / "source_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--parameters", type=Path, help="Full collection settings JSON; packaged defaults otherwise"
    )
    args = parser.parse_args()
    print(json.dumps(build_plan(args.sources, args.output, args.parameters), indent=2))


if __name__ == "__main__":
    main()
