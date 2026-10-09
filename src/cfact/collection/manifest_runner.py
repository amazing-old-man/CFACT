"""Generic explicit-manifest lifecycle using the official preparation/collection APIs."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from cfact.io.serialization import atomic_json, RunLock
from cfact.paths import source_root
from cfact.runtime.freeze import verify


def read_profile(name: str | None) -> dict:
    if name is None:
        return {}
    if name != "coverage":
        raise ValueError("unknown engineering profile")
    from importlib.resources import files

    return json.loads(files("cfact.resources").joinpath(name + ".json").read_text())


def validate_manifest(path: Path, profile: dict) -> dict:
    import h5py
    from cfact.tasks import target_task

    manifest = json.loads(path.read_text())
    tasks, requests = set(), set()
    if not manifest.get("tasks"):
        raise ValueError("explicit tasks and root/demo lists are required")
    for row in manifest["tasks"]:
        task = row["task"]
        target_task(task)
        if task in tasks:
            raise ValueError("duplicate task")
        tasks.add(task)
        roots, donors = row["roots"], row.get("donors", [])
        if not roots or len(roots) != len(set(roots)) or len(donors) != len(set(donors)):
            raise ValueError("nonempty unique roots and unique donors required")
        if set(roots) & set(donors):
            raise ValueError("root/donor overlap")
        if profile and len(roots) != profile["roots_per_task"]:
            raise ValueError("profile requires three distinct root demos per task")
        for demo in roots + donors:
            if not isinstance(demo, str) or not demo or "/" in demo or demo in {".", ".."}:
                raise ValueError("invalid episode group")
        key = "raw_hdf5" if "raw_hdf5" in row else "prepared_hdf5"
        candidate = Path(row[key]).expanduser()
        candidate = (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        row[key] = str(candidate)
        with h5py.File(candidate, "r") as dataset:
            for demo in roots + donors:
                if f"data/{demo}" not in dataset:
                    raise ValueError(f"missing requested episode: {task}/{demo}")
        requests.update((task, demo) for demo in roots)
    if profile and (len(tasks) != profile["tasks"] or len(requests) != profile["unique_roots"]):
        raise ValueError("coverage requires exactly 20 tasks and 60 unique roots")
    return manifest


def prepare_manifest(manifest_path: Path, work: Path, profile: dict, gpu: int,
                     parameters: Path | None) -> dict:
    from importlib.resources import files
    from cfact.sources.prepare import validate_prepared_identity
    from cfact.sources.manifest import build_plan

    requested = validate_manifest(manifest_path, profile)
    config = json.loads(parameters.read_text()) if parameters else json.loads(
        files("cfact.resources").joinpath("collection.json").read_text()
    )
    if profile:
        config["engineering_profile"] = profile
        config["resource"].update(profile["resource"])
        config["resource"]["requested_scene_workers"] = profile["target_workers"]
    identity = {"manifest": requested, "profile": profile, "parameters": config,
                "source_content_sha256": verify(source_root())["content_sha256"]}
    marker = work / "request_identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("manifest/profile/parameters/code changed; use a new work directory")
    work.mkdir(parents=True, exist_ok=True)
    with RunLock(work / "prepare.lock"):
        atomic_json(marker, identity)
        spec = copy.deepcopy(requested)
        spec["output_root"] = str(work / "runs")
        for row in spec["tasks"]:
            groups = row["roots"] + row.get("donors", [])
            if "raw_hdf5" in row:
                raw = Path(row.pop("raw_hdf5"))
                output = work / "prepared" / (row["task"] + ".hdf5")
                if output.exists():
                    validate_prepared_identity(output, raw=raw, required_episodes=groups)
                else:
                    from cfact.collection.coordinator import _resource_admission

                    admission = _resource_admission([gpu], 1, config["resource"])
                    atomic_json(work / "prepare_resource_admission.json", admission)
                    if admission["status"] != "PASS":
                        raise RuntimeError("preparation resource admission failed")
                    command = [sys.executable, "-m", "cfact", "prepare-source",
                               "--raw", str(raw), "--output", str(output),
                               "--task", row["task"], "--gpu", str(gpu), "--resume"]
                    for demo in groups:
                        command.extend(["--demo", demo])
                    subprocess.run(command, check=True)
                row["prepared_hdf5"] = str(output)
            else:
                validate_prepared_identity(Path(row["prepared_hdf5"]), required_episodes=groups)
        if profile:
            config["engineering_profile"] = profile
            config["resource"].update(profile["resource"])
            config["resource"]["requested_scene_workers"] = profile["target_workers"]
        spec_path, params_path = work / "sources.json", work / "parameters.json"
        atomic_json(spec_path, spec)
        atomic_json(params_path, config)
        plan = work / "plan"
        if plan.exists():
            frozen_spec = json.loads((plan / "sources.json").read_text())
            if frozen_spec != spec:
                raise ValueError("existing plan differs from requested sources")
            from cfact.collection.coordinator import _verify_prepared_sources

            frozen_cfg = json.loads((plan / "config.json").read_text())
            frozen_manifest = json.loads((plan / "source_manifest.json").read_text())
            report = _verify_prepared_sources(frozen_manifest, work / "prepared_check.json", frozen_cfg)
            if report["status"] != "PASS":
                raise ValueError("prepared sources changed since plan creation")
            return {"status": "RESUMED_PREPARED", "plan": str(plan)}
        return build_plan(spec_path, plan, params_path)


def collection_run(work: Path) -> Path:
    return work / "runs" / "collection" / "_work" / "collection"


def status(work: Path) -> dict:
    run = collection_run(work)
    outcomes = []
    for path in sorted((run / "data").glob("*/*/*/root_result.json")):
        value = json.loads(path.read_text())
        outcomes.append({"path": str(path.relative_to(run)), "task": value.get("task"),
                         "root_id": value.get("root_id"), "status": value.get("status")})
    release_path = work / "runs/collection/release.json"
    released = json.loads(release_path.read_text()) if release_path.is_file() else {}
    return {"run": str(run), "outcomes": outcomes,
            "pause_requested": (run / "runtime/pause_requested.json").exists(),
            "safely_paused": (run / "runtime/paused.json").exists(),
            "release_complete": released.get("status") == "COMPLETE"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "prepare", "collect", "resume", "pause", "status", "publish"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--profile", choices=("coverage",))
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    freeze = verify(source_root())
    if args.action == "verify":
        print(json.dumps(freeze, indent=2))
        return
    if args.work_dir is None:
        parser.error("--work-dir required")
    work = args.work_dir.resolve()
    profile = read_profile(args.profile)
    workers = args.workers or profile.get("target_workers", 1)
    if not 1 <= workers <= profile.get("max_workers", 30):
        raise ValueError("worker count exceeds profile")
    if args.dry_run:
        manifest = validate_manifest(args.manifest.resolve(), profile) if args.manifest else None
        print(json.dumps({"status": "DRY_RUN", "action": args.action, "freeze": freeze,
                          "roots": sum(len(r["roots"]) for r in manifest["tasks"]) if manifest else None,
                          "workers": workers, "physics_started": False,
                          "work_dir": str(work)}, indent=2))
        return
    if args.action == "prepare":
        if args.manifest is None:
            parser.error("prepare requires --manifest")
        result = prepare_manifest(args.manifest.resolve(), work, profile,
                                  int(args.gpu_devices.split(",")[0]), args.parameters)
    elif args.action in {"collect", "resume"}:
        command = [sys.executable, "-m", "cfact", "collect", "--config", str(work / "plan/config.json"),
                   "--source-manifest", str(work / "plan/source_manifest.json"), "--stage", "formal",
                   "--scene-workers", str(workers), "--gpu-devices", args.gpu_devices,
                   "--run-label", "collection"]
        if args.action == "resume":
            command.append("--resume")
        subprocess.run(command, check=True)
        result = status(work)
    elif args.action == "pause":
        marker = collection_run(work) / "runtime/pause_requested.json"
        atomic_json(marker, {"reason": "explicit_pause_request", "mode": "drain_active_roots"})
        result = {"status": "PAUSE_REQUESTED", "marker": str(marker)}
    elif args.action == "status":
        result = status(work)
    else:
        export_python = os.environ.get("CFACT_EXPORT_PYTHON", sys.executable)
        output = work / "runs/collection"
        command = [export_python, "-m", "cfact", "publish", "--run", str(collection_run(work)),
                   "--output", str(output), "--release-id", "collection"]
        if (output / "release.json").exists():
            command.append("--resume")
        run = collection_run(work)
        from cfact.collection.coordinator import _encode_pending

        config = json.loads((run / "config_frozen.json").read_text())
        with RunLock(run / "runtime/coordinator.lock"):
            encoded = _encode_pending(run, workers=config["resource"].get("ffmpeg_workers", 1))
            if encoded.get("failures"):
                raise RuntimeError("media encoding incomplete; publication deferred")
            subprocess.run(command, check=True)
        result = status(work)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
