"""Immutable run identity and stable lifecycle for the frozen physical generator."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from importlib.metadata import distributions
from importlib.resources import files
import json
from pathlib import Path
import re
import subprocess
import sys
import uuid

import yaml
import h5py

from cfact.io.serialization import atomic_json, RunLock
from cfact.sources.manifest import build_plan, sha256_file, digest_json
from cfact.sources.prepare import validate_prepared_identity
from cfact.runtime.resources import admission
from cfact.storage.policy import artifact_root, contained_path


def load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if value.get("schema") != "cfact.run.v1" or not value.get("tasks"):
        raise ValueError("Expected cfact.run.v1 with explicit tasks")
    for task in value["tasks"]:
        if not task.get("roots") or set(task["roots"]) & set(task.get("donors", [])):
            raise ValueError("Nonempty roots and disjoint donors are required")
        if len(set(task["roots"])) != len(task["roots"]):
            raise ValueError("Duplicate root demos")
    return value


def parameters(config: dict, workers: int, export_python: str | None) -> dict:
    result = json.loads(files("cfact.resources").joinpath("collection.json").read_text())
    # Only bounded engineering budgets are selected here. Scientific defaults stay intact.
    profile = json.loads(files("cfact.resources").joinpath("coverage.json").read_text())
    profile["name"] = "bounded"
    profile["tasks"] = len(config["tasks"])
    profile["unique_roots"] = sum(len(t["roots"]) for t in config["tasks"])
    profile["target_workers"] = workers
    result["engineering_profile"] = profile
    result["run_seed"] = config["seed"]
    result["video"].update(config.get("camera", {}))
    result["tree_limits"]["maximum_decision_depth"] = config.get("maximum_decision_depth", 2)
    if config.get("max_control_steps") is not None:
        result["execution"]["max_env_steps_per_root"] = config["max_control_steps"]
        result["execution"]["max_task_path_steps"] = config["max_control_steps"]
    result["resource"].update(config["resource"])
    result["resource"]["requested_scene_workers"] = workers
    result["scheduling"] = {"max_active_roots_per_task": config.get("max_active_roots_per_task")}
    result["storage"].update({"auto_publish": False, "flat_trees": True, "export_python": export_python})
    if config.get("artifact_root"):
        result["storage"].update({key: config[key] for key in
            ("artifact_root", "allowed_write_root", "quota_root") if key in config})
    from cfact.collection.config import validate_config
    validate_config(result)
    return result


def status(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    native = run / "runtime/collection/_work/collection"
    outcomes = []
    for path in sorted(native.glob("data/*/*/*/root_result.json")):
        result = json.loads(path.read_text())
        outcomes.append({"task": result["task"], "root_id": result.get("root_id"),
                         "status": result["status"], "edges": len(result.get("edges", []))})
    return {"run_id": manifest["run_id"], "run": str(run), "requested_roots": manifest["requested_roots"],
            "completed_root_records": len(outcomes), "outcomes": outcomes,
            "export": json.loads((run / "release.json").read_text()).get("status") if (run / "release.json").exists() else "NOT_RUN",
            "state": json.loads((run / "runtime/state.json").read_text()) if (run / "runtime/state.json").exists() else {}}


def verify_manifest(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text())
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if digest_json(payload) != manifest["manifest_sha256"]:
        raise ValueError("Immutable run manifest changed")
    if manifest.get("artifact_root"):
        contained_path(run, Path(manifest["artifact_root"]))
    for name, expected in manifest["immutable_files"].items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"Immutable run input changed: {name}")
    return manifest


def collect(run: Path, *, resume: bool) -> dict:
    manifest = verify_manifest(run)
    from cfact.collection.snapshot import implementation_digest
    from cfact.collection.coordinator import COMPLETED_ROOT_STATUSES
    if implementation_digest() != manifest["code_sha256"]:
        raise ValueError("Run code changed; resume requires the original package")
    for source in json.loads((run / "source_manifest.json").read_text())["tasks"]:
        if sha256_file(Path(source["prepared_path"])) != source["sha256"]:
            raise ValueError("Prepared source changed after run initialization")
    current = status(run)
    if resume and current["completed_root_records"] == manifest["requested_roots"] and all(
        row["status"] in COMPLETED_ROOT_STATUSES for row in current["outcomes"]
    ):
        return {**current, "resume": "COMPLETE_NO_PHYSICS_REPEATED"}
    plan = run / "runtime/plan"
    config = json.loads((plan / "config.json").read_text())
    devices = manifest["gpu_devices"]
    gate = admission(run, devices, manifest["workers"], config["resource"])
    atomic_json(run / "reports/resource_admission.json", gate)
    if gate["status"] != "PASS":
        atomic_json(run / "runtime/state.json", gate)
        raise RuntimeError("NOT_RUN_RESOURCE_BLOCKED; see reports/resource_admission.json")
    command = [sys.executable, "-m", "cfact.collection.coordinator", "--config", str(plan / "config.json"),
               "--source-manifest", str(plan / "source_manifest.json"), "--stage", "formal",
               "--scene-workers", str(manifest["workers"]), "--gpu-devices", ",".join(map(str, devices)),
               "--run-label", "collection"]
    if resume:
        command.append("--resume")
    with RunLock(run / "runtime/lifecycle.lock"):
        atomic_json(run / "runtime/state.json", {"status": "COLLECTING"})
        with (run / "runtime/collection.log").open("a") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        atomic_json(run / "runtime/state.json", {"status": "COLLECTED" if result.returncode == 0 else "COLLECTION_FAILED", "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"Collection exited {result.returncode}; same-run resume retains root identities")
    return status(run)


def generate(args) -> dict:
    config = load_config(args.config)
    if args.task:
        requested = set(args.task)
        config["tasks"] = [row for row in config["tasks"] if row["task"] in requested]
        if {row["task"] for row in config["tasks"]} != requested:
            raise ValueError("Requested task is absent from the selected configuration")
    for field, selections in (("roots", args.root_demo), ("donors", args.donor_demo)):
        explicit = {}
        for selection in selections or []:
            task, separator, demo = selection.partition("=")
            if not separator or not demo:
                raise ValueError("Explicit demos use TASK=demo_N")
            explicit.setdefault(task, []).append(demo)
        if set(explicit) - {row["task"] for row in config["tasks"]}:
            raise ValueError("Explicit demo belongs to an unselected task")
        for row in config["tasks"]:
            if row["task"] in explicit:
                row[field] = explicit[row["task"]]
    if args.max_roots_per_task is not None:
        if args.max_roots_per_task < 1:
            raise ValueError("max-roots-per-task must be positive")
        for row in config["tasks"]:
            row["roots"] = row["roots"][:args.max_roots_per_task]
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_control_steps is not None:
        config["max_control_steps"] = args.max_control_steps
    for row in config["tasks"]:
        if not row["roots"] or set(row["roots"]) & set(row.get("donors", [])):
            raise ValueError("Roots and donors must remain nonempty and disjoint")
    workers = args.workers or config.get("workers", 1)
    devices = [int(v) for v in args.gpu_devices.split(",")]
    if not 1 <= workers <= 6 or not devices:
        raise ValueError("Use 1 to 6 workers with explicit available GPU devices")
    config = copy.deepcopy(config)
    artifact = artifact_root(config, Path.cwd().resolve(), getattr(args, "artifact_root", None))
    config["artifact_root"] = str(artifact)
    params = parameters(config, workers, args.export_python)
    if args.dry_run:
        return {"status": "DRY_RUN", "physics_started": False, "workers": workers,
                "root_count": sum(len(x["roots"]) for x in config["tasks"]), "config": config,
                "artifact_root": str(artifact),
                "resources": admission(next(p for p in (artifact, *artifact.parents) if p.exists()), devices, workers, params["resource"])}
    prepared_dir = args.prepared_dir or Path(config.get("source_dir", ".cache")) / "prepared"
    task_rows, source_rows, datasets = [], [], set()
    for row in config["tasks"]:
        path = (prepared_dir / (row["task"] + ".hdf5")).resolve()
        identity = validate_prepared_identity(path, required_episodes=row["roots"] + row.get("donors", []))
        with h5py.File(path, "r") as dataset:
            provenance = json.loads(dataset["data"].attrs.get("cfact_source_provenance", "{}"))
        datasets.add(provenance.get("official_download", {}).get("repo_id", "local-hdf5").split("/")[-1])
        task_rows.append({**row, "prepared_hdf5": str(path)})
        source_rows.append({"task": row["task"], "prepared_path": str(path), "sha256": sha256_file(path),
                            "identity": identity, "source_provenance": provenance})
    if len(datasets) != 1 or config["source_dataset"] != next(iter(datasets)):
        raise ValueError("Configured dataset name differs from the actual Source provenance")
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src/cfact").is_dir():
        raise ValueError("Run generate from the CFACT project root")
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", config["source_dataset"]).strip(".-")[:100]
    if not slug:
        raise ValueError("Source dataset name has no safe slug")
    created = datetime.now(timezone.utc)
    run_id = uuid.uuid4().hex[:12]
    output_base = contained_path(artifact / config.get("output_subdir", ""), artifact)
    run = output_base / f"{slug}__{created:%Y%m%dT%H%M%SZ}__{run_id}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "runtime").mkdir()
    (run / "reports").mkdir()
    spec = {"tasks": task_rows, "output_root": str(run / "runtime")}
    atomic_json(run / "runtime/sources.json", spec)
    atomic_json(run / "runtime/parameters.json", params)
    build_plan(run / "runtime/sources.json", run / "runtime/plan", run / "runtime/parameters.json")
    atomic_json(run / "source_manifest.json", {"dataset": config["source_dataset"], "tasks": source_rows})
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    from cfact.collection.snapshot import implementation_digest
    immutable = ["config.yaml", "source_manifest.json", "runtime/plan/config.json", "runtime/plan/source_manifest.json"]
    manifest = {"schema": "cfact.run.v1", "run_id": run_id, "created_utc": created.isoformat(),
                "artifact_root": str(artifact), "allowed_write_root": config.get("allowed_write_root", str(artifact)),
                "source_dataset": config["source_dataset"], "requested_roots": sum(len(x["roots"]) for x in task_rows),
                "tasks": task_rows, "seed": config["seed"], "workers": workers, "gpu_devices": devices,
                "camera_contract": params["video"], "config_sha256": digest_json(params),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"])),
                "code_sha256": implementation_digest(), "git_commit": revision.stdout.strip() if revision.returncode == 0 else "UNCOMMITTED",
                "dependencies": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
                "immutable_files": {name: sha256_file(run / name) for name in immutable}}
    manifest["manifest_sha256"] = digest_json(manifest)
    atomic_json(run / "manifest.json", manifest)
    print(json.dumps({"run": str(run), "run_id": run_id}), flush=True)
    return collect(run, resume=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--config", type=Path, required=True)
    gen.add_argument("--prepared-dir", type=Path,
                     help="Prepared inputs; defaults to config source_dir/prepared, or legacy .cache/prepared")
    gen.add_argument("--artifact-root", type=Path, help="Absolute output filesystem root; frozen in run identity")
    gen.add_argument("--workers", type=int)
    gen.add_argument("--gpu-devices", default="0")
    gen.add_argument("--export-python")
    gen.add_argument("--task", action="append", help="Select a configured task; repeatable")
    gen.add_argument("--root-demo", action="append", help="Explicit TASK=demo_N root; repeatable")
    gen.add_argument("--donor-demo", action="append", help="Explicit TASK=demo_N donor; repeatable")
    gen.add_argument("--max-roots-per-task", type=int, help="Limit the declared root list before run initialization")
    gen.add_argument("--seed", type=int)
    gen.add_argument("--max-control-steps", type=int, help="Per-root and task-path budget in actual control steps")
    gen.add_argument("--dry-run", action="store_true")
    for name in ("status", "resume", "pause", "export", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        if name == "export":
            command.add_argument("--media", choices=("full", "playlist"), default="full")
            command.add_argument("--by-tree", action="store_true")
    args = parser.parse_args()
    if args.action == "generate":
        report = generate(args)
    else:
        run = args.run.resolve()
        verify_manifest(run)
        if args.action == "status":
            report = status(run)
        elif args.action == "resume":
            report = collect(run, resume=True)
        elif args.action == "pause":
            atomic_json(run / "runtime/collection/_work/collection/runtime/pause_requested.json",
                        {"mode": "drain_active_roots", "reason": "explicit_user_request"})
            report = {"status": "PAUSE_REQUESTED", "run": str(run)}
        elif args.action == "export":
            from cfact.export.dataset import export_run
            report = export_run(run, full_media=args.media == "full", by_tree=args.by_tree)
        else:
            from cfact.export.verification import verify_run
            report = verify_run(run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
