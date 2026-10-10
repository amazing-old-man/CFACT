#!/usr/bin/env python3
"""Single-writer coordinator for independent object-effect scene processes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cfact.paths import workspace_root, source_root
from cfact.collection.snapshot import source_digest, archive_source, unpack_source
from cfact.io.serialization import RunLock, atomic_json
from cfact.io.review_video import encode_deferred_video


PROJECT = workspace_root()
COMPLETED_ROOT_STATUSES = {"PHYSICS_COMPLETE", "BLOCKED_SOURCE_REPLAY", "NO_SUPPORTED_CONTINUATION"}


def _source_hash() -> str:
    return source_digest(source_root())


def _jobs(manifest: dict[str, Any], split: str, config: dict[str, Any]) -> list[dict[str, str]]:
    if split not in {"formal", "pilot"} or "requests_by_id" not in manifest:
        raise ValueError("collection requires an explicit prepared-source request manifest")
    request_ids = manifest["pilot_request_ids"] if split == "pilot" else manifest["queue_order"]
    return [
        {
            "task": manifest["requests_by_id"][request_id]["task"],
            "root_id": request_id,
            "split": "formal",
            "role_in_release": manifest["requests_by_id"][request_id].get(
                "role_in_release", "FORMAL_REQUEST"
            ),
        }
        for request_id in request_ids
    ]


def _snapshot(run: Path, config: Path, source_manifest: Path) -> None:
    source = source_root()
    freeze = None
    if json.loads(config.read_text()).get("source_mode") == "development":
        from cfact.runtime.freeze import inventory, fingerprint
        current = inventory(source)
        freeze = {"status": "DEVELOPMENT_SNAPSHOT", "release": "0.7.2.dev0",
                  "content_sha256": fingerprint(current), "files": current}
    elif (source / "FREEZE.json").exists():
        from cfact.runtime.freeze import verify

        freeze = verify(source)
    snapshot = run / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True, capture_output=True
    )
    commit = revision.stdout.strip() if revision.returncode == 0 else "UNCOMMITTED"
    (snapshot / "base_commit.txt").write_text(commit + "\n")
    diff = subprocess.run(["git", "diff", "--binary"], cwd=source, text=True, capture_output=True)
    (snapshot / "dirty.diff").write_text(diff.stdout if diff.returncode == 0 else "")
    archive_path = snapshot / "original_sources.tar.gz"
    archive_source(source, archive_path, config, source_manifest)
    runtime = run / "source_runtime"
    unpack_source(archive_path, runtime)
    if freeze and freeze["status"] == "DEVELOPMENT_SNAPSHOT":
        atomic_json(snapshot / "DEVELOPMENT_FREEZE.json", freeze)
    elif freeze:
        shutil.copy2(source / "FREEZE.json", snapshot / "FREEZE.json")
    expected = source_digest(source)
    if source_digest(runtime) != expected:
        raise RuntimeError("frozen source differs from source checkout")
    (run / "config_frozen.json").write_bytes(config.read_bytes())
    (run / "source_split_frozen.json").write_bytes(source_manifest.read_bytes())
    from importlib.metadata import distributions

    runtime_versions = {
        "python": sys.version,
        "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
    }
    atomic_json(snapshot / "runtime_versions.json", runtime_versions)
    atomic_json(
        run / "run_manifest.json",
        {
            "generator_revision": json.loads(config.read_text())["generator_revision"],
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "source_split_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
            "source_snapshot": "source_snapshot/original_sources.tar.gz",
            "source_base_commit": commit,
            "source_freeze": freeze,
            "runtime_versions": "source_snapshot/runtime_versions.json",
            "source_sha256": expected,
            "requested_scene_workers": json.loads(config.read_text())["resource"][
                "requested_scene_workers"
            ],
        },
    )


def _verify_prepared_sources(
    manifest: dict[str, Any], destination: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Verify the prepared manifest, hash each routed HDF5 once, and check its groups."""
    import h5py

    release_root = Path(config["release_root"]).resolve()
    prepared_manifest_path = release_root / manifest["prepared_source_manifest"]
    prepared_manifest_digest = (
        hashlib.sha256(prepared_manifest_path.read_bytes()).hexdigest()
        if prepared_manifest_path.is_file()
        else None
    )
    route_manifest_path = release_root / manifest["raw_source_routes"]
    route_manifest_digest = (
        hashlib.sha256(route_manifest_path.read_bytes()).hexdigest()
        if route_manifest_path.is_file()
        else None
    )
    failures: list[dict[str, str]] = []
    if prepared_manifest_digest != manifest.get("prepared_source_manifest_sha256"):
        failures.append(
            {
                "path": str(prepared_manifest_path),
                "reason": "PREPARED_SOURCE_MANIFEST_SHA256_MISMATCH",
            }
        )
    if prepared_manifest_digest != config.get("prepared_source_manifest_sha256"):
        failures.append(
            {
                "path": str(prepared_manifest_path),
                "reason": "CONFIG_PREPARED_MANIFEST_SHA256_MISMATCH",
            }
        )
    if route_manifest_digest != config.get("raw_routes_sha256"):
        failures.append(
            {"path": str(route_manifest_path), "reason": "RAW_ROUTE_MANIFEST_SHA256_MISMATCH"}
        )
    if (
        not prepared_manifest_path.is_file()
        or json.loads(prepared_manifest_path.read_text()).get("status") != "PASS"
    ):
        failures.append(
            {"path": str(prepared_manifest_path), "reason": "PREPARED_SOURCE_MANIFEST_NOT_PASS"}
        )
    records = manifest.get("prepared_files", {})
    task_event_signals = {row["task"]: row.get("event_signal") for row in manifest.get("tasks", [])}
    checked: list[dict[str, Any]] = []
    for path_text, record in sorted(records.items()):
        path = Path(path_text)
        if not path.is_file():
            failures.append({"path": path_text, "reason": "MISSING_FILE"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != record.get("sha256"):
            failures.append(
                {
                    "path": path_text,
                    "reason": "SHA256_MISMATCH",
                    "expected": str(record.get("sha256")),
                    "actual": actual,
                }
            )
            continue
        bad_groups = []
        try:
            from cfact.sources.prepare import validate_prepared_identity

            validate_prepared_identity(path, required_episodes=record.get("groups", []))
            with h5py.File(path, "r") as source:
                for name in record.get("groups", []):
                    group_name = name if name.startswith("data/") else f"data/{name}"
                    if group_name not in source:
                        bad_groups.append({"group": name, "reason": "MISSING_GROUP"})
                        continue
                    group = source[group_name]
                    if not all(key in group for key in ("actions", "states")):
                        bad_groups.append({"group": name, "reason": "MISSING_ACTIONS_OR_STATES"})
                        continue
                    actions, states = group["actions"], group["states"]
                    if len(actions) < 1 or len(states) < 1 or len(actions) != len(states):
                        bad_groups.append({"group": name, "reason": "INVALID_ACTION_STATE_LENGTHS"})
                    event_signal = task_event_signals.get(record.get("task"))
                    signal_group = group.get("datagen_info/subtask_term_signals")
                    if event_signal and (signal_group is None or event_signal not in signal_group):
                        bad_groups.append(
                            {
                                "group": name,
                                "reason": "MISSING_FROZEN_EVENT_SIGNAL",
                                "event_signal": str(event_signal),
                            }
                        )
        except Exception as error:
            bad_groups.append(
                {"group": "*", "reason": f"HDF5_READ_ERROR:{type(error).__name__}:{error}"}
            )
        if bad_groups:
            failures.append(
                {
                    "path": path_text,
                    "reason": "GROUP_VALIDATION_FAILED",
                    "groups": json.dumps(bad_groups, ensure_ascii=False),
                }
            )
        checked.append(
            {
                "path": path_text,
                "sha256": actual,
                "groups_checked": len(record.get("groups", [])),
                "bytes": path.stat().st_size,
            }
        )
    result = {
        "status": "PASS" if not failures and checked else "SOURCE_INTEGRITY_FAILED",
        "checked_files": checked,
        "checked_file_count": len(checked),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_prepared_file_count": len(records),
    }
    atomic_json(destination, result)
    return result


def _result_paths(run: Path, job: dict[str, str]) -> list[Path]:
    return list(
        (run / "data" / job["split"] / job["task"]).glob(
            f"{job['task']}__{job['root_id']}__*/completed.json"
        )
    )


def _quarantine_interrupted_job(run: Path, job: dict[str, str]) -> Path | None:
    """Preserve an interrupted root and its media before one explicit resume.

    A root's frame spool has the same deterministic name as a retry. Moving only
    the data directory leaves stale frames in place and makes every retried edge
    fail. The run lock must already be held by this coordinator.
    """
    split, task, root_id = job["split"], job["task"], job["root_id"]
    root_pattern = f"{task}__{root_id}__*"
    data_roots = sorted((run / "data" / split / task).glob(root_pattern))
    paths = list(data_roots)
    for method in ("reference", "G0", "G1", "G2"):
        paths.extend(sorted((run / "videos" / method / split / task).glob(root_pattern)))
    paths.extend(sorted((run / "videos" / "lineages" / split / task).glob(f"{task}__{root_id}__*")))
    for extra in (
        run / "history" / split / task / root_id,
        run / "runtime" / "scene_alive" / f"{split}__{task}__{root_id}.json",
        run / "runtime" / "job_results" / split / task / f"{root_id}.json",
        run / "runtime" / "logs" / split / task / f"{root_id}.log",
    ):
        if extra.exists():
            paths.append(extra)
    if not paths:
        return None
    alive = run / "runtime" / "scene_alive" / f"{split}__{task}__{root_id}.json"
    if alive.exists():
        try:
            pid = int(json.loads(alive.read_text())["pid"])
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, ProcessLookupError):
            command = b""
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot verify scene owner {alive}") from error
        if (
            b"cfact.collection.worker" in command
            and f"--root-id {root_id}".encode() in command
            and f"--task {task}".encode() in command
        ):
            raise RuntimeError(f"scene worker still alive with pid {pid}: {alive}")
    for root in data_roots:
        lock = root / "run.lock"
        if not lock.exists():
            continue
        try:
            payload = json.loads(lock.read_text())
            pid = int(payload["pid"])
            expected = payload.get("process_start_ticks")
            actual = RunLock._process_start_ticks(pid)
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot verify interrupted root lock {lock}") from error
        if actual is not None and (expected is None or actual == int(expected)):
            raise RuntimeError(f"root worker still alive with pid {pid}: {root}")
    # Even if the data root was already moved by an interrupted recovery,
    # discover and preserve its matching media spool before dispatching a retry.
    archive = (
        run
        / "runtime"
        / "interrupted_attempts"
        / split
        / task
        / root_id
        / f"attempt_{time.time_ns()}_{os.getpid()}"
    )
    archive.mkdir(parents=True, exist_ok=False)
    records = []
    for source in paths:
        relative = source.relative_to(run)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        records.append({"original": str(relative), "archived": str(target.relative_to(run))})
    atomic_json(
        archive / "quarantine.json",
        {
            "reason": "interrupted_or_noncomplete_root_resume",
            "job": job,
            "moved": records,
            "archive_wall_time_ns": time.time_ns(),
        },
    )
    return archive


def _gpu_status() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines():
        try:
            index_text, uuid, used_text, total_text = [value.strip() for value in line.split(",")]
            index, used, total = int(index_text), int(used_text), int(total_text)
        except ValueError:
            continue
        rows.append(
            {
                "index": index,
                "uuid": uuid,
                "used_mib": used,
                "total_mib": total,
                "free_mib": total - used,
            }
        )
    return rows


def _resource_admission(
    devices: list[int], workers: int, resource: dict[str, Any] | None = None
) -> dict[str, Any]:
    rows = {row["index"]: row for row in _gpu_status()}
    resource = resource or {}
    required_per_scene_mib = int(resource.get("admission_gpu_mib_per_scene", 1800))
    reserve_mib = int(resource.get("admission_gpu_reserve_mib", 8192))
    counts = {device: 0 for device in devices}
    for index in range(workers):
        counts[devices[index % len(devices)]] += 1
    failures = [
        device
        for device, count in counts.items()
        if device not in rows
        or rows[device]["free_mib"] < count * required_per_scene_mib + reserve_mib
    ]
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemAvailable:", "SwapFree:")):
            key, value = line.split(":", 1)
            memory[key] = int(value.strip().split()[0]) // 1024
    ram_per_scene = int(resource.get("admission_ram_mib_per_scene", 3584))
    ram_reserve = int(resource.get("admission_ram_reserve_mib", 12288))
    required_ram_mib = workers * ram_per_scene + ram_reserve
    ram_short = memory.get("MemAvailable", 0) < required_ram_mib
    return {
        "status": "PASS" if not failures and not ram_short else "PARALLELISM_RESOURCE_LIMIT",
        "requested_workers": workers,
        "per_scene_admission_mib": required_per_scene_mib,
        "reserve_per_gpu_mib": reserve_mib,
        "device_worker_counts": counts,
        "gpu_observation": list(rows.values()),
        "insufficient_devices": failures,
        "ram_observation_mib": memory,
        "required_available_ram_mib": required_ram_mib,
        "ram_per_scene_admission_mib": ram_per_scene,
        "ram_reserve_mib": ram_reserve,
        "ram_basis": resource.get("ram_basis", "development measurement plus safety margin"),
        "ram_short": ram_short,
    }


def _dispatch_resource_admission(
    devices: list[int], active: dict[int, Any], next_gpu: int, resource: dict[str, Any]
) -> dict[str, Any]:
    rows = {row["index"]: row for row in _gpu_status()}
    per_scene = int(resource.get("admission_gpu_mib_per_scene", 1800))
    reserve = int(resource.get("admission_gpu_reserve_mib", 8192))
    counts = {device: 0 for device in devices}
    for _, item in active.items():
        gpu = int(item[1].get("physical_gpu_index", -1))
        if gpu in counts:
            counts[gpu] += 1
    counts[next_gpu] = counts.get(next_gpu, 0) + 1
    failures = [
        device
        for device, count in counts.items()
        if device not in rows or rows[device]["free_mib"] < count * per_scene + reserve
    ]
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            memory["MemAvailable"] = int(line.split(":", 1)[1].strip().split()[0]) // 1024
    ram_per = int(resource.get("admission_ram_mib_per_scene", 3584))
    ram_reserve = int(resource.get("admission_ram_reserve_mib", 12288))
    required_ram = (len(active) + 1) * ram_per + ram_reserve
    thermal = {"thermal": [], "failures": []}
    if "maximum_gpu_temperature_c" in resource:
        from cfact.runtime.resources import thermal_admission
        thermal = thermal_admission([next_gpu], resource)
    return {
        "status": "PASS"
        if not failures and not thermal["failures"] and memory.get("MemAvailable", 0) >= required_ram
        else "PARALLELISM_RESOURCE_LIMIT",
        "thermal": thermal,
        "active_workers_before_dispatch": len(active),
        "next_gpu": next_gpu,
        "device_worker_counts_after_dispatch": counts,
        "gpu_observation": list(rows.values()),
        "insufficient_devices": failures,
        "ram_observation_mib": memory,
        "required_available_ram_mib": required_ram,
    }


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _storage_root(run: Path, config: dict[str, Any]) -> Path:
    # The plan and dataset may live on different disks. Keep source-manifest
    # resolution under release_root, but account for the actual saved dataset.
    if config.get("storage", {}).get("schema") == "cfact.dual-view.v1":
        return run.resolve().parent.parent
    return Path(config.get("release_root", run)).resolve()


def _storage_admission(run: Path, config: dict[str, Any]) -> dict[str, Any]:
    resource = config["resource"]
    release_root = _storage_root(run, config)
    usage = shutil.disk_usage(release_root)
    baseline_path = run / "runtime" / "storage_baseline.json"
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text())
        delta_bytes = max(0, int(baseline["filesystem_free_bytes"]) - int(usage.free))
        release_bytes = int(baseline.get("release_bytes", 0)) + delta_bytes
        run_bytes = delta_bytes
        spool_bytes = None  # the filesystem delta includes camera PNG and review panel spools
        accounting = "conservative_release_filesystem_delta_from_collection_start"
    else:
        release_bytes = _tree_bytes(release_root)
        run_bytes = _tree_bytes(run)
        spool_bytes = sum(_tree_bytes(path) for path in run.rglob("*.frames") if path.is_dir())
        accounting = "initial_release_and_run_tree_scan"
    maximum = int(resource["maximum_output_gb"] * 1_000_000_000)
    minimum_free = int(resource["minimum_host_free_gib"] * 1024**3)
    ok = release_bytes <= maximum and usage.free >= minimum_free
    return {
        "status": "PASS" if ok else "STORAGE_GUARD_STOP",
        "release_bytes": release_bytes,
        "run_and_spool_bytes": run_bytes,
        "frame_spool_bytes": spool_bytes,
        "accounting": accounting,
        "maximum_release_bytes": maximum,
        "filesystem_free_bytes": usage.free,
        "minimum_filesystem_free_bytes": minimum_free,
    }


def _next_dispatch_index(pending: list[dict], active_jobs: list[dict], task_limit: int | None) -> int | None:
    """Keep source order while allowing independently capped task queues to progress."""
    for index, job in enumerate(pending):
        if task_limit is None or sum(row["task"] == job["task"] for row in active_jobs) < task_limit:
            return index
    return None


def _run_batch(
    *,
    run: Path,
    jobs: list[dict[str, str]],
    config: Path,
    split_manifest: Path,
    workers: int,
    devices: list[int],
    resume: bool,
) -> list[dict[str, Any]]:
    config_payload = json.loads(config.read_text())
    task_limit = config_payload.get("scheduling", {}).get("max_active_roots_per_task")
    pending = []
    for job in jobs:
        completed = _result_paths(run, job)
        if resume and completed and len(completed) == 1:
            previous = json.loads(completed[0].read_text()).get("status")
            if previous in COMPLETED_ROOT_STATUSES:
                continue
        if resume:
            _quarantine_interrupted_job(run, job)
        pending.append(job)
    active: dict[int, tuple[subprocess.Popen[str], dict[str, str], Any, float]] = {}
    results: list[dict[str, Any]] = []
    baseline = _gpu_status()
    peak = {x["index"]: x["used_mib"] for x in baseline}
    prior_workers_path = run / "runtime" / "workers.json"
    prior_workers = (
        json.loads(prior_workers_path.read_text())
        if resume and prior_workers_path.is_file()
        else {}
    )
    for index, used in prior_workers.get("gpu_peak_used_mib", {}).items():
        peak[int(index)] = max(peak.get(int(index), 0), int(used))
    peak_active = int(prior_workers.get("peak_simultaneous_workers", 0))
    peak_ready_scenes = int(prior_workers.get("peak_simultaneous_ready_scenes", 0))
    launched = 0
    last_sample = 0.0
    samples_path = run / "runtime" / "resource_samples.jsonl"
    pause_file = run / "runtime" / "pause_requested.json"
    while pending or active:
        if pause_file.exists() and not active:
            atomic_json(run / "runtime" / "paused.json", {
                "status": "PAUSED", "pending_jobs": pending,
                "finished_jobs": results, "active_workers": 0,
            })
            break
        while pending and len(active) < workers and not pause_file.exists():
            dispatch_index = _next_dispatch_index(pending, [item[1] for item in active.values()], task_limit)
            if dispatch_index is None:
                break
            worker_slot = launched % workers
            gpu = devices[worker_slot % len(devices)]
            dispatch_gate = _dispatch_resource_admission(
                devices, active, gpu, config_payload["resource"]
            )
            atomic_json(run / "runtime" / "dispatch_resource_admission.json", dispatch_gate)
            if dispatch_gate["status"] != "PASS":
                break
            storage = _storage_admission(run, config_payload)
            atomic_json(run / "runtime" / "storage_admission.json", storage)
            if storage["status"] != "PASS":
                for job in pending:
                    results.append(
                        {**job, "status": "COLLECTION_BUDGET_CENSORED", "reason": storage["status"]}
                    )
                pending.clear()
                break
            expected_source = json.loads((run / "run_manifest.json").read_text())["source_sha256"]
            if _source_hash() != expected_source:
                raise RuntimeError(
                    "source changed after run freeze; refusing mixed-version workers"
                )
            job = dict(pending.pop(dispatch_index))
            observed_gpu = next((row for row in _gpu_status() if row["index"] == gpu), None)
            job["physical_gpu_index"] = gpu
            job["physical_gpu_uuid"] = observed_gpu["uuid"] if observed_gpu else "UNKNOWN"
            job["cuda_visible_devices"] = str(gpu)
            job["mujoco_egl_device_id"] = str(gpu)
            env = os.environ.copy()
            env.update(
                {
                    "CFACT_WORKSPACE": str(PROJECT),
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "MUJOCO_EGL_DEVICE_ID": str(gpu),
                    "MUJOCO_GL": "egl",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "CFACT_DEFER_VIDEO_ENCODING": "1",
                    "CFACT_GPU_UUID": job["physical_gpu_uuid"],
                }
            )
            log = run / "runtime" / "logs" / job["split"] / job["task"] / f"{job['root_id']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            stream = log.open("w")
            frozen_worker = run / "source_runtime" / "src/cfact/collection/worker.py"
            if not frozen_worker.is_file():
                raise FileNotFoundError(f"frozen worker missing: {frozen_worker}")
            env["PYTHONPATH"] = (
                str(run / "source_runtime/src") + os.pathsep + env.get("PYTHONPATH", "")
            )
            command = [
                sys.executable,
                "-m",
                "cfact.collection.worker",
                "--config",
                str(config),
                "--source-manifest",
                str(split_manifest),
                "--task",
                job["task"],
                "--root-id",
                job["root_id"],
                "--split",
                job["split"],
                "--run",
                str(run),
                "--gpu",
                str(gpu),
            ]
            process = subprocess.Popen(
                command, cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True
            )
            active[process.pid] = (process, job, stream, time.monotonic())
            launched += 1
            peak_active = max(peak_active, len(active))
        gpu_rows = _gpu_status()
        ready_scenes = len(list((run / "runtime" / "scene_alive").glob("*.json")))
        peak_ready_scenes = max(peak_ready_scenes, ready_scenes)
        for row in gpu_rows:
            peak[row["index"]] = max(peak.get(row["index"], 0), row["used_mib"])
        atomic_json(
            run / "runtime" / "workers.json",
            {
                "requested_workers": workers,
                "max_active_roots_per_task": task_limit,
                "active_workers": len(active),
                "peak_simultaneous_workers": peak_active,
                "ready_scene_instances": ready_scenes,
                "peak_simultaneous_ready_scenes": peak_ready_scenes,
                "active": [{"pid": pid, **item[1]} for pid, item in active.items()],
                "gpu_baseline_mib": baseline,
                "gpu_peak_used_mib": peak,
            },
        )
        if time.monotonic() - last_sample >= 10.0:
            memory = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    memory["MemAvailable"] = int(line.split(":", 1)[1].strip().split()[0]) // 1024
            sample = {
                "sampled_at": time.time(),
                "active_workers": len(active),
                "pending_workers": len(pending),
                "gpu": gpu_rows,
                "mem_available_mib": memory.get("MemAvailable"),
                "filesystem_free_bytes": shutil.disk_usage(run).free,
                "storage": _storage_admission(run, config_payload),
            }
            with samples_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
                stream.flush()
            last_sample = time.monotonic()
        for pid, (process, job, stream, started) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            del active[pid]
            marker = _result_paths(run, job)
            status = (
                json.loads(marker[0].read_text())["status"]
                if code == 0 and marker
                else "TECHNICAL_ERROR"
            )
            result = {
                **job,
                "status": status,
                "exit_code": code,
                "wall_seconds": time.monotonic() - started,
                "completed_marker": None if not marker else str(marker[0].relative_to(run)),
            }
            results.append(result)
            atomic_json(
                run
                / "runtime"
                / "job_results"
                / job["split"]
                / job["task"]
                / f"{job['root_id']}.json",
                result,
            )
            atomic_json(
                run / "runtime" / "progress.json",
                {
                    "split": job["split"],
                    "completed_jobs": len(results),
                    "requested_jobs": len(jobs),
                    "active_workers": len(active),
                    "ready_jobs": len(pending),
                    "results": results,
                },
            )
        if pending or active:
            time.sleep(2)
    return results


def _encode_pending(run: Path, workers: int = 2) -> dict[str, Any]:
    pending = []
    for media_path in sorted((run / "data").rglob("media.json")):
        media = json.loads(media_path.read_text())
        if media.get("encoding_status") == "PENDING":
            pending.append((media_path, media))
    atomic_json(
        run / "runtime" / "media_progress.json",
        {
            "status": "MEDIA_PENDING" if pending else "COMPLETE",
            "queued": len(pending),
            "completed": 0,
            "failed": 0,
            "maximum_simultaneous_encoders": workers,
        },
    )
    results = []

    def encode(item: tuple[Path, dict[str, Any]]) -> dict[str, Any]:
        media_path, media = item
        video = run / media["video"]
        try:
            encode_deferred_video(video, int(media["fps"]))
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=nb_frames,width,height",
                    "-of",
                    "json",
                    str(video),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            streams = json.loads(probe.stdout)["streams"]
            if not streams or int(streams[0].get("nb_frames", 0)) != len(media["frame_sim_time_s"]):
                raise RuntimeError("encoded frame count differs from physical frame ledger")
            media["encoding_status"] = "COMPLETE"
            media["ffprobe"] = streams[0]
            atomic_json(media_path, media)
            frame_spool = video.with_suffix(".frames")
            for panel_frame in frame_spool.glob("*.jpg"):
                panel_frame.unlink()
            return {
                "media": str(media_path.relative_to(run)),
                "status": "COMPLETE",
                "clean_camera_frames_retained": True,
            }
        except Exception as error:
            return {
                "media": str(media_path.relative_to(run)),
                "status": "ENCODING_ERROR",
                "error": f"{type(error).__name__}: {error}",
            }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(encode, item) for item in pending]):
            results.append(future.result())
            atomic_json(
                run / "runtime" / "media_progress.json",
                {
                    "status": "MEDIA_PENDING",
                    "queued": len(pending),
                    "completed": sum(row["status"] == "COMPLETE" for row in results),
                    "failed": sum(row["status"] != "COMPLETE" for row in results),
                    "maximum_simultaneous_encoders": workers,
                },
            )
    report = {
        "maximum_simultaneous_encoders": workers,
        "queued": len(pending),
        "results": results,
        "failures": [row for row in results if row["status"] != "COMPLETE"],
    }
    atomic_json(run / "runtime" / "encoding.json", report)
    atomic_json(
        run / "runtime" / "media_progress.json",
        {
            "status": "COMPLETE" if not report["failures"] else "ENCODING_ERROR",
            "queued": len(pending),
            "completed": len(results) - len(report["failures"]),
            "failed": len(report["failures"]),
            "maximum_simultaneous_encoders": workers,
        },
    )
    return report


def _export_lerobot(run: Path, config: dict[str, Any]) -> dict[str, Any]:
    specification = config.get("dataset_export")
    if not specification:
        return {"status": "NOT_REQUESTED"}
    output = run / "lerobot_dataset"
    report_path = output / "lerobot_readback.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        prior_collection_path = run / "readback.json"
        prior_collection = (
            json.loads(prior_collection_path.read_text()).get("collection_status")
            if prior_collection_path.is_file()
            else None
        )
        if report.get("status") == "PASS" and prior_collection == "COMPLETE":
            return {"status": "PASS", "readback": report, "resumed_from_completed_export": True}
        if output.exists():
            output.rename(output.with_name(output.name + f".incomplete_{time.time_ns()}"))
    admission = _storage_admission(run, config)
    if admission["status"] != "PASS":
        result = {"status": admission["status"], "storage_admission": admission}
        atomic_json(run / "runtime" / "lerobot_export.json", result)
        return result
    runtime_manifest = Path(specification["runtime_manifest"])
    runtime_hash = hashlib.sha256(runtime_manifest.read_bytes()).hexdigest()
    if runtime_hash != specification["runtime_manifest_sha256"]:
        raise RuntimeError("LeRobot runtime manifest changed after the formal config freeze")
    script = run / "source_runtime" / "src/cfact/export/lerobot.py"
    if not script.is_file():
        raise FileNotFoundError(f"frozen LeRobot exporter missing: {script}")
    command = [
        specification["python"],
        "-m",
        "cfact.export.lerobot",
        "--run",
        str(run),
        "--output",
        str(output),
        "--repo-id",
        specification["repo_id"],
        "--python-packages",
        specification["python_packages"],
        "--cleanup-camera-spools",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(run / "source_runtime/src") + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        {
            "HF_HOME": str(run / "runtime" / "hf_home"),
            "HF_LEROBOT_HOME": str(output),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "WANDB_DISABLED": "true",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    log = run / "runtime" / "logs" / "lerobot_export.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        command, cwd=PROJECT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    log.write_text(process.stdout)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    output_json = None
    if process.returncode == 0 and lines:
        try:
            output_json = json.loads(lines[-1])
        except json.JSONDecodeError:
            output_json = None
    result = {
        "status": "PASS"
        if process.returncode == 0 and output_json and output_json.get("status") == "PASS"
        else "EXPORT_ERROR",
        "returncode": process.returncode,
        "command": command,
        "log": str(log.relative_to(run)),
        "storage_admission_before_export": admission,
        "runtime_manifest_sha256": runtime_hash,
        "readback": output_json,
    }
    atomic_json(run / "runtime" / "lerobot_export.json", result)
    return result


def _export_lineages(run: Path) -> dict[str, Any]:
    """Use only ancestor local segments, never an evaluated fixed tail, in a lineage."""
    results = []
    by_task: dict[str, Path] = {}
    split_priority = {"confirmation": 0, "development": 1, "adapter": 2}
    for tree in sorted(
        (run / "data").glob("*/*/*/tree.json"),
        key=lambda path: (split_priority.get(path.relative_to(run).parts[1], 9), str(path)),
    ):
        task = tree.parent.parent.name
        by_task.setdefault(task, tree)
    for task, tree_path in by_task.items():
        tree = json.loads(tree_path.read_text())
        root_edges = [row for row in tree.get("edges", []) if row.get("edge_path")]
        children = tree.get("child_pools", [])
        thirds = tree.get("third_pools", [])
        selected = []
        for first in root_edges:
            chain = [first]
            child = next(
                (
                    pool
                    for pool in children
                    if pool.get("parent_edge") == first["edge_path"] and pool.get("edges")
                ),
                None,
            )
            if child:
                second = next((row for row in child["edges"] if row.get("edge_path")), None)
                if second:
                    chain.append(second)
                    third = next(
                        (
                            pool
                            for pool in thirds
                            if pool.get("parent_edge") == second["edge_path"] and pool.get("edges")
                        ),
                        None,
                    )
                    if third:
                        last = next((row for row in third["edges"] if row.get("edge_path")), None)
                        if last:
                            chain.append(last)
            if len(chain) > len(selected):
                selected = chain
        clips = []
        for edge in selected:
            media_path = run / edge["edge_path"] / "media.json"
            if not media_path.is_file():
                break
            media = json.loads(media_path.read_text())
            video = run / media["video"]
            if media.get("encoding_status") != "COMPLETE" or not video.is_file():
                break
            phases = media["frame_phase"]
            end = next(
                (i for i, phase in enumerate(phases) if phase == "fixed_continuation"), len(phases)
            )
            clips.append((video, end, edge["edge_path"]))
        if not clips:
            continue
        root_id = tree_path.parent.name
        output = run / "videos" / "lineages" / task / f"{root_id}__lineage.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.is_file() and output.stat().st_size:
            results.append(
                {
                    "task": task,
                    "video": str(output.relative_to(run)),
                    "edge_paths": [item[2] for item in clips],
                    "status": "COMPLETE",
                }
            )
            continue
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for video, _, _ in clips:
            command.extend(["-i", str(video)])
        filters = []
        for index, (_, end, _) in enumerate(clips):
            filters.append(
                f"[{index}:v]trim=start_frame=0:end_frame={end},setpts=PTS-STARTPTS[v{index}]"
            )
        filters.append(
            "".join(f"[v{index}]" for index in range(len(clips)))
            + f"concat=n={len(clips)}:v=1:a=0[out]"
        )
        command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[out]",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                str(output),
            ]
        )
        try:
            subprocess.run(command, check=True)
            results.append(
                {
                    "task": task,
                    "video": str(output.relative_to(run)),
                    "edge_paths": [item[2] for item in clips],
                    "status": "COMPLETE",
                    "includes_fixed_continuation": False,
                }
            )
        except subprocess.CalledProcessError as error:
            results.append({"task": task, "status": "ENCODING_ERROR", "error": str(error)})
    report = {
        "lineages": results,
        "failures": [row for row in results if row["status"] != "COMPLETE"],
    }
    atomic_json(run / "videos" / "lineages" / "index.json", report)
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _export_reports(run: Path) -> dict[str, Any]:
    import numpy as np

    node_rows = []
    option_rows = []
    media_rows = []
    cost_rows = []
    consequence_rows = []
    problems = []
    for node in sorted((run / "data").glob("*/*/*/nodes/*")):
        if not node.is_dir():
            continue
        parts = node.relative_to(run).parts
        split, task, root_id, node_id = parts[1], parts[2], parts[3], parts[5]
        context = node / "context.json"
        observation = node / "observation.npz"
        if not context.is_file() or not observation.is_file():
            problems.append(
                {"node": str(node.relative_to(run)), "reason": "missing_context_or_observation"}
            )
            continue
        with np.load(observation) as observed:
            rgb_valid = "rgb" in observed and observed["rgb"].shape[0] == 3
            measured_valid = (
                len(observed["actual_eef_pose"]) == 1 and len(observed["object_poses"]) == 1
            )
        node_rows.append(
            {
                "split": split,
                "task": task,
                "root_id": root_id,
                "node_id": node_id,
                "path": str(node.relative_to(run)),
                "three_clean_cameras": rgb_valid,
                "measured_parent": measured_valid,
            }
        )
        if not rgb_valid or not measured_valid:
            problems.append(
                {"node": str(node.relative_to(run)), "reason": "invalid_clean_observation"}
            )
        proposals = node / "proposals.jsonl"
        if proposals.is_file():
            for line in proposals.read_text().splitlines():
                if not line:
                    continue
                option = json.loads(line)
                edge = node / "edges" / option["candidate_id"]
                local = edge / "local_result.json"
                technical = edge / "technical_error.json"
                status = (
                    "EXECUTED"
                    if local.is_file()
                    else "TECHNICAL_ERROR"
                    if technical.is_file()
                    else "REJECTED_OR_BUDGET_CENSORED"
                )
                option_rows.append(
                    {
                        "split": split,
                        "task": task,
                        "root_id": root_id,
                        "node_id": node_id,
                        "candidate_id": option["candidate_id"],
                        "source_method": option["source_method"],
                        "subtype": option["subtype"],
                        "status": status,
                        "reason": option.get("rejection_reason") or "",
                        "source_episode": option.get("source_episode") or "",
                    }
                )
        for edge in sorted((node / "edges").glob("*")):
            if not edge.is_dir():
                continue
            execution = edge / "execution.npz"
            if execution.is_file():
                try:
                    with np.load(execution) as trace:
                        count = len(trace["commanded_actions"])
                        if (
                            len(trace["sim_time"]) != count + 1
                            or len(trace["actual_eef_pose"]) != count + 1
                            or len(trace["object_poses"]) != count + 1
                            or len(trace["robot_qpos"]) != count + 1
                        ):
                            raise ValueError("unaligned_measured_trace")
                        if count == 0:
                            raise ValueError("empty_physical_execution")
                        obj = trace["object_poses"][:, 0, :3, 3]
                        joint = trace["relevant_fixture_qpos"]
                        command_hash = hashlib.sha256(
                            np.asarray(trace["commanded_actions"], dtype="<f8").tobytes()
                        ).hexdigest()
                        state_hash = hashlib.sha256(
                            np.asarray(obj, dtype="<f8").tobytes()
                        ).hexdigest()
                        local_path = edge / "local_result.json"
                        local = json.loads(local_path.read_text()) if local_path.is_file() else {}
                        card = json.loads((edge / "option.json").read_text())
                        consequence_rows.append(
                            {
                                "split": split,
                                "task": task,
                                "root_id": root_id,
                                "node_id": node_id,
                                "candidate_id": edge.name,
                                "source_method": card["source_method"],
                                "subtype": card["subtype"],
                                "local_event": local.get("termination_reason"),
                                "official_success_at_child": local.get("official_success_at_child"),
                                "Y_fixed_status": local.get("Y_fixed", {}).get("status"),
                                "object_displacement_m": float(np.linalg.norm(obj[-1] - obj[0])),
                                "fixture_joint_displacement": (
                                    float(np.max(np.abs(joint[-1] - joint[0])))
                                    if joint.shape[1]
                                    else None
                                ),
                                "command_sha256": command_hash,
                                "object_path_sha256": state_hash,
                                "control_steps_local": count,
                            }
                        )
                except (OSError, KeyError, ValueError) as error:
                    problems.append({"edge": str(edge.relative_to(run)), "reason": str(error)})
            media = edge / "media.json"
            if media.is_file():
                card = json.loads((edge / "option.json").read_text())
                meta = json.loads(media.read_text())
                video_path = run / meta["video"]
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    problems.append(
                        {"edge": str(edge.relative_to(run)), "reason": "missing_or_empty_video"}
                    )
                if len(set(meta["cameras"])) != 3 or not meta["frame_sim_time_s"]:
                    problems.append(
                        {
                            "edge": str(edge.relative_to(run)),
                            "reason": "invalid_camera_rig_or_timestamps",
                        }
                    )
                media_rows.append(
                    {
                        "split": split,
                        "task": task,
                        "root_id": root_id,
                        "node_id": node_id,
                        "candidate_id": edge.name,
                        "source_method": card["source_method"],
                        "video_path": meta["video"],
                        "frame_count": len(meta["frame_sim_time_s"]),
                        "render_source": meta["render_source"],
                    }
                )
            costs = edge / "costs.json"
            if costs.is_file():
                row = json.loads(costs.read_text())
                cost_rows.append(
                    {
                        "split": split,
                        "task": task,
                        "root_id": root_id,
                        "node_id": node_id,
                        "candidate_id": edge.name,
                        **row,
                    }
                )
    for original in sorted((run / "videos" / "original").glob("*/*/index.json")):
        row = json.loads(original.read_text())
        media_rows.append(
            {
                "split": "source",
                "task": original.parent.parent.name,
                "root_id": original.parent.name,
                "node_id": "",
                "candidate_id": "",
                "source_method": "original",
                "video_path": row.get("video_path", ""),
                "frame_count": "",
                "render_source": "native_single_view",
            }
        )
    tables = run / "tables"
    _write_csv(tables / "nodes.csv", node_rows)
    _write_csv(tables / "options.csv", option_rows)
    _write_csv(tables / "cost_summary.csv", cost_rows)
    _write_csv(tables / "consequences.csv", consequence_rows)
    _write_csv(run / "videos" / "index.csv", media_rows)
    for source in ("G0", "G1", "G2", "reference", "original"):
        rows = [row for row in media_rows if row["source_method"].lower() == source.lower()]
        atomic_json(
            run / "videos" / source / "index.json",
            {
                "source_method": source,
                "video_count": len(rows),
                "entries": rows,
                "empty_reason": None if rows else "no_executed_applicable_option",
            },
        )
    counts = {}
    for option in option_rows:
        key = option["source_method"]
        bucket = counts.setdefault(key, {"proposed": 0, "executed": 0, "rejected_or_censored": 0})
        bucket["proposed"] += 1
        bucket["executed" if option["status"] == "EXECUTED" else "rejected_or_censored"] += 1
    _write_csv(
        tables / "generator_summary.csv",
        [{"method": key, **value} for key, value in sorted(counts.items())],
    )
    outcome_counts = {}
    for edge in consequence_rows:
        bucket = outcome_counts.setdefault(
            edge["source_method"],
            {
                "measured_executions": 0,
                "local_events": 0,
                "fixed_successes": 0,
                "fixed_horizon_censored": 0,
            },
        )
        bucket["measured_executions"] += 1
        bucket["local_events"] += edge["local_event"] == "local_event_reached"
        bucket["fixed_successes"] += edge["Y_fixed_status"] == "official_success"
        bucket["fixed_horizon_censored"] += edge["Y_fixed_status"] == "horizon_censored"
    _write_csv(
        tables / "outcome_summary.csv",
        [{"method": method, **row} for method, row in sorted(outcome_counts.items())],
    )
    _write_csv(
        tables / "task_summary.csv",
        [
            {
                "task": task,
                "nodes": sum(x["task"] == task for x in node_rows),
                "executed_options": sum(
                    x["task"] == task and x["status"] == "EXECUTED" for x in option_rows
                ),
            }
            for task in sorted({x["task"] for x in node_rows})
        ],
    )
    atomic_json(
        run / "schema.json",
        {
            "schema_version": "3.0.0",
            "action_alignment": "K commands; K+1 physical states; frame0 is before the first action",
            "poses": "4x4 world transforms in metres",
            "sim_time": "seconds",
            "fixture_joint_units": "joint-dependent metres or radians; names in node context",
            "videos": "review media only; clean observations in nodes/*/observation.npz",
            "outcome": "Y_fixed is one observed rollout within a bounded horizon, not a probability",
        },
    )
    report_dir = run / "reports"
    report_dir.mkdir(exist_ok=True)
    by_task = {}
    for row in node_rows:
        if row["node_id"] == "n_root":
            by_task.setdefault(row["task"], row)
    review = ["# Author review", "", "Paths below are relative to this run directory.", ""]
    for task, row in sorted(by_task.items()):
        root = row["path"].split("/nodes/")[0]
        review += [
            f"## {task}",
            "",
            f"- Source: `videos/original/{task}/{row['root_id'].split('__')[1]}/source_original.mp4`",
            f"- Candidate pool: `{row['path']}/candidate_pool.json`",
            f"- Actual tree: `{root}/tree.json`",
            "",
        ]
    (report_dir / "AUTHOR_REVIEW.md").write_text("\n".join(review))
    (report_dir / "IMPLEMENTATION_AND_LIMITATIONS.md").write_text(
        "# 实现说明与限制\n\n"
        "候选路径根据源对象或把手的实测运动生成，由 MimicGen waypoint 和 RoboCasa `env.step` 执行。"
        "每个局部事件子状态先保存，再从该 child state 恢复并执行固定后续路径。\n\n"
        "释放事件依据张开命令、对象稳定性与末端分离信号判断；不宣称直接测得支撑面接触。"
        "静态几何检查不能保证连续无碰撞，也不能保证任务成功。深度为 MuJoCo 原始 render buffer。\n"
    )
    return {
        "nodes": len(node_rows),
        "options": len(option_rows),
        "videos": len(media_rows),
        "readback_problems": problems,
        "generator_counts": counts,
        "outcome_counts": outcome_counts,
        "measured_executions": len(consequence_rows),
    }


def _readback(run: Path) -> dict[str, Any]:
    roots = []
    edges = []
    for path in sorted((run / "data").glob("*/*/*/root_result.json")):
        result = json.loads(path.read_text())
        roots.append(
            {
                "split": result["split"],
                "task": result["task"],
                "root_id": result["root_id"],
                "status": result["status"],
                "nodes": len(result["nodes"]),
                "edges": len(result["edges"]),
                "observed_max_decision_depth": result.get("observed_max_decision_depth", 1),
                "charged_control_steps": result.get("charged_control_steps"),
            }
        )
        for edge in result["edges"]:
            edges.append(
                {
                    "split": result["split"],
                    "task": result["task"],
                    "root_id": result["root_id"],
                    **edge,
                }
            )
    table_dir = run / "tables"
    table_dir.mkdir(exist_ok=True)
    for name, rows in (("roots", roots), ("edges", edges)):
        if rows:
            columns = sorted({key for row in rows for key in row})
            with (table_dir / f"{name}.csv").open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
    job_results = [
        json.loads(path.read_text())
        for path in sorted((run / "runtime" / "job_results").glob("*/*/*.json"))
    ]
    requested_jobs = []
    ledger = run / "runtime" / "requested_jobs.json"
    if ledger.is_file():
        requested_jobs = json.loads(ledger.read_text()).get("jobs", [])
    completed_identities = {
        (row["split"], row["task"], row["root_id"].split("__")[1]) for row in roots
    }
    missing_requested = [
        job
        for job in requested_jobs
        if (job["split"], job["task"], job["root_id"]) not in completed_identities
    ]
    technical_jobs = [row for row in job_results if row["status"] == "TECHNICAL_ERROR"]
    status = {
        "roots": len(roots),
        "edges": len(edges),
        "development_roots": sum(x["split"] == "development" for x in roots),
        "adapter_roots": sum(x["split"] == "adapter" for x in roots),
        "confirmation_roots": sum(x["split"] == "confirmation" for x in roots),
        "technical_roots": (
            sum(
                x["status"] not in {"PHYSICS_COMPLETE", "PARTIAL", "BLOCKED_SOURCE_REPLAY"}
                for x in roots
            )
            + len(technical_jobs)
        ),
        "blocked_source_replay_roots": sum(x["status"] == "BLOCKED_SOURCE_REPLAY" for x in roots),
        "requested_jobs": len(requested_jobs),
        "job_results": len(job_results),
        "technical_jobs_without_root": technical_jobs,
        "missing_requested_jobs": missing_requested,
        "completed_jobs": {
            "finished": len(job_results),
            "physics_complete": sum(x["status"] == "PHYSICS_COMPLETE" for x in job_results),
            "partial": sum(x["status"] == "PARTIAL" for x in job_results),
            "technical": len(technical_jobs),
            "blocked": sum(
                x["status"] in {"COLLECTION_BUDGET_CENSORED", "BLOCKED", "BLOCKED_SOURCE_REPLAY"}
                for x in job_results
            ),
        },
    }
    status.update(_export_reports(run))
    formal_jobs = [job for job in requested_jobs if job.get("split") == "formal"]
    formal_results = [row for row in job_results if row.get("split") == "formal"]
    formal_roots = [root for root in roots if root.get("split") == "formal"]
    source_manifest = json.loads((run / "source_split_frozen.json").read_text())
    requested = (
        int(source_manifest["new_request_count"])
        if "requests_by_id" in source_manifest
        else len(formal_jobs)
    )
    lerobot_readback_path = run / "lerobot_dataset" / "lerobot_readback.json"
    lerobot_readback = (
        json.loads(lerobot_readback_path.read_text()) if lerobot_readback_path.is_file() else None
    )
    lerobot_complete = bool(lerobot_readback and lerobot_readback.get("status") == "PASS")
    if lerobot_readback is not None:
        status["lerobot_readback"] = lerobot_readback
        status["lerobot_episode_count"] = lerobot_readback.get("episode_count", 0)
        status["lerobot_frame_count"] = lerobot_readback.get("frame_count", 0)
    status["historical_r2_physics_complete_reused"] = source_manifest.get(
        "historical_r2_request_count", 0
    )
    status["formal_new_source_requests"] = source_manifest.get("new_request_count", 0)
    export_requested = bool(
        json.loads((run / "config_frozen.json").read_text()).get("dataset_export")
    )
    status["lerobot_export_requested"] = export_requested
    formal_complete = (
        requested > 0
        and len(formal_results) == requested
        and len(formal_roots) == requested
        and all(
            row.get("status") in {"PHYSICS_COMPLETE", "BLOCKED_SOURCE_REPLAY"}
            for row in formal_results
        )
        and all(
            root.get("status") in {"PHYSICS_COMPLETE", "BLOCKED_SOURCE_REPLAY"}
            for root in formal_roots
        )
        and (lerobot_complete or not export_requested)
    )
    confirmation_complete = all(
        x["status"] == "PHYSICS_COMPLETE" for x in roots if x["split"] == "confirmation"
    )
    media_complete = all(
        json.loads(path.read_text()).get("encoding_status") == "COMPLETE"
        for path in (run / "data").rglob("media.json")
    )
    formal_mode = bool(formal_jobs) or any(root.get("split") == "formal" for root in roots)
    collection = (
        "COMPLETE"
        if (
            formal_complete
            if formal_mode
            else status["confirmation_roots"] == requested and confirmation_complete
        )
        and media_complete
        and not status["readback_problems"]
        else "PARTIAL"
    )
    methods = status["outcome_counts"]
    peak_ready = 0
    workers_path = run / "runtime" / "workers.json"
    if workers_path.is_file():
        peak_ready = int(
            json.loads(workers_path.read_text()).get("peak_simultaneous_ready_scenes", 0)
        )
    admission_path = run / "runtime" / "resource_admission.json"
    admission = json.loads(admission_path.read_text()) if admission_path.is_file() else None
    status["peak_simultaneous_ready_scenes"] = peak_ready
    status["collection_status"] = collection
    status["capability_status"] = {
        "source_and_option_contracts": "COMPLETE"
        if collection == "COMPLETE" and status["options"]
        else ("PARTIAL" if status["options"] else "NOT_OBSERVED"),
        "G0_legal_local_variants": "COMPLETE"
        if collection == "COMPLETE" and methods.get("G0", {}).get("measured_executions")
        else ("PARTIAL" if methods.get("G0", {}).get("measured_executions") else "NOT_OBSERVED"),
        "G1_object_motion_modes": "COMPLETE"
        if collection == "COMPLETE" and methods.get("G1", {}).get("measured_executions")
        else ("PARTIAL" if methods.get("G1", {}).get("measured_executions") else "NOT_OBSERVED"),
        "G2_constraint_targets_and_execution": "COMPLETE"
        if collection == "COMPLETE" and methods.get("G2", {}).get("measured_executions")
        else ("PARTIAL" if methods.get("G2", {}).get("measured_executions") else "NOT_OBSERVED"),
        "actual_event_children_and_recursion": "COMPLETE"
        if any(x["observed_max_decision_depth"] >= 3 for x in roots)
        else ("PARTIAL" if any(x["nodes"] > 1 for x in roots) else "NOT_OBSERVED"),
        "measured_state_recording": "COMPLETE"
        if collection == "COMPLETE"
        and status["measured_executions"]
        and not status["readback_problems"]
        else ("PARTIAL" if status["measured_executions"] else "NOT_OBSERVED"),
        "parallel_scene_execution": (
            "COMPLETE"
            if peak_ready >= int((admission or {}).get("requested_workers", 1))
            else (
                "PARTIAL"
                if peak_ready
                else "BLOCKED"
                if admission and admission["status"] != "PASS"
                else "NOT_OBSERVED"
            )
        ),
        "three_view_video_export": "COMPLETE"
        if collection == "COMPLETE"
        and lerobot_complete
        and len((lerobot_readback or {}).get("camera_features", [])) == 3
        else ("PARTIAL" if status["videos"] else "NOT_OBSERVED"),
        "declared_source_collection": collection,
    }
    capability_names = {
        "source_and_option_contracts": "源与候选契约",
        "G0_legal_local_variants": "G0 局部合法变体",
        "G1_object_motion_modes": "G1 对象运动模式",
        "G2_constraint_targets_and_execution": "G2 约束目标与执行",
        "actual_event_children_and_recursion": "实测事件子节点与树展开",
        "measured_state_recording": "实测状态记录",
        "parallel_scene_execution": "并行场景执行",
        "three_view_video_export": "三路相机视频导出",
        "declared_source_collection": "声明来源范围内的采集",
    }
    summary = [
        "# CFACT 采集阶段报告",
        "",
        f"- 开发树数量：{status['development_roots']}",
        f"- 适配树数量：{status['adapter_roots']}",
        f"- 正式树数量：{len(formal_roots)}/{requested}",
        f"- 节点数量：{status['nodes']}",
        f"- 边数量：{status['edges']}",
        f"- 回看视频数量（含源原视频）：{status['videos']}",
        f"- LeRobot v3 episode 数：{status.get('lerobot_episode_count', 0)}",
        f"- 冻结新源请求：{status.get('formal_new_source_requests', requested)}",
        f"- 复用历史 R2 物理结果：{status.get('historical_r2_physics_complete_reused', 0)}",
        f"- 读回问题数：{len(status['readback_problems'])}",
        f"- 当前队列状态：{collection}",
        "",
        "## 能力覆盖",
        "",
        *[
            f"- {capability_names.get(key, key)}：{value}"
            for key, value in status["capability_status"].items()
        ],
        "",
        "结果是有限执行时域内实际观测到的物理 rollout。超时或预算截断不等同于物理失败。",
        "",
    ]
    (run / "reports" / "FINAL_REPORT.md").write_text("\n".join(summary))
    atomic_json(run / "readback.json", status)
    return status


def _require_stage_success(
    results: list[dict[str, Any]],
    encoding: dict[str, Any],
    lineages: dict[str, Any],
    export: dict[str, Any],
) -> None:
    """A finished coordinator must not hide failed workers or incomplete output."""
    allowed = COMPLETED_ROOT_STATUSES
    failed = [row for row in results if row.get("status") not in allowed]
    if (
        failed
        or encoding.get("failures")
        or lineages.get("failures")
        or export.get("status")
        not in {
            "PASS",
            "DEFERRED_UNTIL_FULL_QUEUE",
            "DEFERRED_TO_RELEASE_PUBLICATION",
            "NOT_REQUESTED",
        }
    ):
        raise RuntimeError(
            "collection stage has failed or incomplete outputs; see runtime stage and job reports"
        )


def _export_python(config: dict[str, Any]) -> str:
    import os

    configured = config.get("storage", {}).get("export_python") or os.environ.get(
        "CFACT_EXPORT_PYTHON"
    )
    return str(configured or sys.executable)


def _check_export_runtime(config: dict[str, Any]) -> None:
    process = subprocess.run(
        [
            _export_python(config),
            "-c",
            "from importlib.metadata import version; assert version('lerobot') == '0.4.0'; "
            "from lerobot.datasets.lerobot_dataset import LeRobotDataset",
        ],
        capture_output=True,
        text=True,
    )
    if process.returncode:
        raise RuntimeError(
            "Dual-view publication needs the LeRobot 0.4.0 export environment. "
            "Set CFACT_EXPORT_PYTHON or storage.export_python. " + process.stderr[-1000:]
        )


def _publish_dual_view(run: Path, release: Path, config: dict[str, Any]) -> None:
    import os

    command = [
        _export_python(config),
        "-m",
        "cfact",
        "publish",
        "--run",
        str(run),
        "--output",
        str(release),
        "--resume",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(run / "source_runtime/src")
    # The single coordinator publishes after every scene worker and review encoder has finished.
    log = run / "runtime/publication.log"
    with log.open("a") as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("preflight", "pilot", "formal", "report-only"), default="preflight"
    )
    parser.add_argument("--scene-workers", type=int, default=1)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan and frozen code without physics or writes")
    args = parser.parse_args()
    if (
        not args.run_label
        or Path(args.run_label).name != args.run_label
        or args.run_label in {".", ".."}
    ):
        parser.error("--run-label must be a single directory name")
    args.config = args.config.resolve()
    args.source_manifest = args.source_manifest.resolve()
    if not 1 <= args.scene_workers <= 30:
        raise ValueError("scene-workers must be in [1, 30]")
    requested_config = json.loads(args.config.read_text())
    from cfact.collection.config import validate_config

    validate_config(requested_config)
    profile = requested_config.get("engineering_profile", {})
    if profile and args.scene_workers > int(profile["max_workers"]):
        raise ValueError("scene-workers exceeds engineering profile maximum")
    if args.dry_run:
        from cfact.runtime.freeze import verify

        freeze = verify(source_root())
        manifest = json.loads(args.source_manifest.read_text())
        jobs = _jobs(manifest, "pilot" if args.stage == "pilot" else "formal", requested_config)
        print(json.dumps({"status": "DRY_RUN", "freeze": freeze, "jobs": jobs,
                          "physics_started": False}, sort_keys=True))
        return
    output_root = Path(requested_config.get("output_root", str(PROJECT / "runs")))
    release = output_root / args.run_label
    dual_view = requested_config.get("storage", {}).get("schema") == "cfact.dual-view.v1"
    run = release / "_work/collection" if dual_view else release
    auto_publish = requested_config.get("storage", {}).get("auto_publish", True)
    if dual_view and auto_publish and args.stage not in {"preflight", "report-only"}:
        _check_export_runtime(requested_config)
    run.mkdir(parents=True, exist_ok=True)
    lock = RunLock(run / "runtime" / "coordinator.lock")
    global_lock = RunLock(output_root / "global_physics_coordinator.lock")
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    lock.acquire()
    global_acquired = False
    try:
        if args.stage not in {"preflight", "report-only"}:
            global_lock.acquire()
            global_acquired = True
        if args.resume and args.stage in {"pilot", "formal"}:
            # Clear pause only after obtaining ownership; a competing resume
            # must never restart dispatch in a coordinator still draining.
            for name in ("pause_requested.json", "paused.json"):
                (run / "runtime" / name).unlink(missing_ok=True)
        frozen = run / "config_frozen.json"
        split_frozen = run / "source_split_frozen.json"
        if not frozen.exists():
            _snapshot(run, args.config, args.source_manifest)
        elif not args.resume:
            raise FileExistsError("run exists; use --resume")
        if frozen.read_bytes() != args.config.read_bytes():
            raise RuntimeError("requested config differs from frozen run config")
        if source_digest(run / "source_runtime") != _source_hash():
            raise RuntimeError("frozen execution source changed after snapshot")
        if _source_hash() != json.loads((run / "run_manifest.json").read_text())["source_sha256"]:
            raise RuntimeError("source changed after run freeze")
        if split_frozen.read_bytes() != args.source_manifest.read_bytes():
            raise RuntimeError("requested source manifest differs from frozen run manifest")
        config = json.loads(frozen.read_text())
        manifest = json.loads(split_frozen.read_text())
        if args.stage == "report-only":
            encoding = _encode_pending(run, workers=config["resource"].get("ffmpeg_workers", 2))
            lineages = (
                {"status": "DEFERRED_TO_PLAYLIST_PUBLICATION"}
                if dual_view
                else _export_lineages(run)
            )
            print(
                json.dumps(
                    {"readback": _readback(run), "encoding": encoding, "lineages": lineages},
                    sort_keys=True,
                )
            )
            return
        devices = [int(x) for x in args.gpu_devices.split(",")]
        available = {x["index"] for x in _gpu_status()}
        remaining_physics = True
        if args.resume and args.stage in {"pilot", "formal"}:
            jobs = _jobs(manifest, "pilot" if args.stage == "pilot" else "formal", config)
            remaining_physics = any(
                len(paths := _result_paths(run, job)) != 1
                or json.loads(paths[0].read_text()).get("status")
                not in COMPLETED_ROOT_STATUSES
                for job in jobs
            )
        if remaining_physics and (not available or any(device not in available for device in devices)):
            atomic_json(
                run / "runtime" / "progress.json",
                {
                    "status": "PARALLELISM_RESOURCE_LIMIT",
                    "requested_workers": args.scene_workers,
                    "actual_workers": 0,
                    "reason": "requested_GPU_unavailable",
                },
            )
            raise RuntimeError("requested GPUs unavailable")
        if args.stage == "preflight":
            source_integrity = _verify_prepared_sources(
                manifest, run / "runtime" / "prepared_source_integrity.json", config
            )
            if source_integrity["status"] != "PASS":
                raise RuntimeError("prepared source integrity verification failed")
            admission = _resource_admission(devices, args.scene_workers, config["resource"])
            admission["storage"] = _storage_admission(run, config)
            admission["source_hash_verified"] = True
            admission["prepared_source_integrity"] = source_integrity
            admission["frozen_split_status"] = [row.get("status") for row in manifest["tasks"]]
            atomic_json(run / "runtime" / "preflight.json", admission)
            print(json.dumps(admission, sort_keys=True))
            return
        if args.stage in {"pilot", "formal"}:
            if "queue_order" not in manifest or "requests_by_id" not in manifest:
                raise ValueError("pilot/formal stages require a frozen source request manifest")
            source_integrity = _verify_prepared_sources(
                manifest, run / "runtime" / "prepared_source_integrity.json", config
            )
            if source_integrity["status"] != "PASS":
                raise RuntimeError("prepared source integrity verification failed")
            baseline_path = run / "runtime" / "storage_baseline.json"
            if not baseline_path.exists():
                release_root = _storage_root(run, config)
                usage = shutil.disk_usage(release_root)
                atomic_json(
                    baseline_path,
                    {
                        "captured_at": time.time(),
                        "filesystem_free_bytes": usage.free,
                        "release_bytes": _tree_bytes(release_root),
                        "release_root": str(release_root),
                        "scope_request_count": len(manifest["queue_order"]),
                    },
                )
            formal_jobs = _jobs(manifest, "pilot" if args.stage == "pilot" else "formal", config)
            atomic_json(
                run / "runtime" / "requested_jobs.json",
                {
                    "jobs": formal_jobs,
                    "stage": args.stage,
                    "request_count": len(formal_jobs),
                    "full_frozen_request_count": manifest["new_request_count"],
                },
            )
            admission_workers = (
                min(args.scene_workers, 20) if args.stage == "pilot" else args.scene_workers
            )
            admission = (
                _resource_admission(devices, admission_workers, config["resource"])
                if remaining_physics else {"status": "PASS", "reason": "all_root_jobs_completed_no_physics"}
            )
            admission["storage"] = _storage_admission(run, config)
            atomic_json(run / "runtime" / "resource_admission.json", admission)
            if admission["status"] != "PASS" or admission["storage"]["status"] != "PASS":
                atomic_json(run / "runtime" / "progress.json", {**admission, "actual_workers": 0})
                raise RuntimeError("PARALLELISM_RESOURCE_LIMIT or STORAGE_GUARD_STOP")
            results = _run_batch(
                run=run,
                jobs=formal_jobs,
                config=frozen,
                split_manifest=split_frozen,
                workers=admission_workers,
                devices=devices,
                resume=args.resume,
            )
            if (run / "runtime" / "pause_requested.json").exists():
                print(json.dumps({"status": "PAUSED", "run": str(run),
                                  "active_workers": 0, "publication": "DEFERRED"}))
                return
            encoding = _encode_pending(run, workers=config["resource"].get("ffmpeg_workers", 2))
            lineages = (
                {"status": "DEFERRED_TO_PLAYLIST_PUBLICATION"}
                if dual_view
                else _export_lineages(run)
            )
            if args.stage == "pilot":
                lerobot_export = {"status": "DEFERRED_UNTIL_FULL_QUEUE"}
                readback = _readback(run)
            else:
                lerobot_export = (
                    {"status": "DEFERRED_TO_RELEASE_PUBLICATION"}
                    if dual_view
                    else _export_lerobot(run, config)
                )
                readback = _readback(run)
            atomic_json(
                run
                / "runtime"
                / ("pilot_stage.json" if args.stage == "pilot" else "formal_stage.json"),
                {
                    "job_count": len(formal_jobs),
                    "stage": args.stage,
                    "full_frozen_request_count": manifest["new_request_count"],
                    "results": len(results),
                    "encoding": encoding,
                    "lineages": lineages,
                    "lerobot_export": lerobot_export,
                    "readback": readback,
                },
            )
            _require_stage_success(results, encoding, lineages, lerobot_export)
            if dual_view and auto_publish:
                _publish_dual_view(run, release, config)

    finally:
        if global_acquired:
            global_lock.release()
        lock.release()


if __name__ == "__main__":
    main()
