"""Host observations and configurable admission checks before physical work."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def thermal_admission(devices: list[int], policy: dict) -> dict:
    failures = []
    probe = subprocess.run(["nvidia-smi", "--query-gpu=index,temperature.gpu,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True)
    thermal = []
    if probe.returncode:
        failures.append("GPU_TEMPERATURE_UNAVAILABLE")
    else:
        for line in probe.stdout.splitlines():
            index, temperature, utilization = [int(x.strip()) for x in line.split(",")]
            if index in devices:
                thermal.append({"index": index, "temperature_c": temperature, "utilization_percent": utilization})
                if temperature >= policy.get("maximum_gpu_temperature_c", 85):
                    failures.append("GPU_TEMPERATURE_LIMIT")
        if {row["index"] for row in thermal} != set(devices):
            failures.append("GPU_TEMPERATURE_UNAVAILABLE")
    return {"thermal": thermal, "failures": failures}


def admission(directory: Path, devices: list[int], workers: int, policy: dict) -> dict:
    from cfact.collection.coordinator import _resource_admission

    report = _resource_admission(devices, workers, policy)
    failures = [] if report["status"] == "PASS" else ["RAM_OR_VRAM_RESERVE"]
    usage = shutil.disk_usage(directory)
    report["disk_free_bytes"] = usage.free
    if usage.free < policy["minimum_host_free_gib"] * 2**30:
        failures.append("DISK_RESERVE")
    thermal = thermal_admission(devices, policy)
    failures.extend(thermal["failures"])
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True)
    report.update({"status": "PASS" if not failures else "NOT_RUN_RESOURCE_BLOCKED",
                   "failures": failures, "thermal": thermal["thermal"],
                   "gpu_processes": processes.stdout.splitlines(),
                   "other_process_policy": "memory accounted at admission; existing processes are never stopped"})
    return report
