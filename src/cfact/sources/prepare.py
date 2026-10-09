"""Annotate raw replay HDF5 in a separate process using pinned MimicGen."""

from __future__ import annotations
import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import h5py
from cfact.data.validation import validate_prepared_dataset
from cfact.io.serialization import RunLock, atomic_json, natural_key
from cfact.sources.manifest import sha256_file
from cfact.tasks import target_task

PREPARED_SCHEMA = "cfact.prepared.v2"
OPERATION_SCHEMA = "cfact.operation_context.v1"


def preparer_code_identity() -> str:
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in ("sources/prepare.py", "adapters/action_codec.py", "adapters/interfaces.py", "adapters/robocasa.py"):
        digest.update(relative.encode() + b"\0" + (package / relative).read_bytes())
    return digest.hexdigest()


def prepared_identity(raw: Path, episodes: list[str]) -> dict:
    with h5py.File(raw, "r") as dataset:
        metadata = dataset["data"].attrs["env_args"]
        if isinstance(metadata, bytes):
            metadata = metadata.decode()
        metadata = json.loads(metadata) if isinstance(metadata, str) else dict(metadata)
    return {
        "schema": PREPARED_SCHEMA, "operation_schema": OPERATION_SCHEMA,
        "raw_sha256": sha256_file(raw), "episodes": episodes,
        "env_metadata": metadata, "preparer_code_sha256": preparer_code_identity(),
    }


def validate_prepared_identity(path: Path, *, raw: Path | None = None,
                               required_episodes: list[str] | None = None) -> dict:
    """Reject legacy or stale annotations rather than silently treating them as new."""
    with h5py.File(path, "r") as dataset:
        encoded = dataset.attrs.get("cfact_prepared_identity")
        if encoded is None:
            raise ValueError("prepared_identity_missing: rebuild into a new output file")
        identity = json.loads(encoded)
        if identity.get("schema") != PREPARED_SCHEMA or identity.get("operation_schema") != OPERATION_SCHEMA:
            raise ValueError("prepared_schema_mismatch")
        if identity.get("preparer_code_sha256") != preparer_code_identity():
            raise ValueError("prepared_code_identity_mismatch")
        if raw is not None and identity.get("raw_sha256") != sha256_file(raw):
            raise ValueError("prepared_raw_identity_mismatch")
        if required_episodes and not set(required_episodes).issubset(identity.get("episodes", [])):
            raise ValueError("prepared_episode_identity_mismatch")
        for episode in required_episodes or identity["episodes"]:
            attrs = dataset[f"data/{episode}/datagen_info"].attrs
            if "cfact_controller_semantics" not in attrs:
                raise ValueError("prepared_controller_identity_missing")
            actual = hashlib.sha256(attrs["cfact_controller_semantics"].encode()).hexdigest()
            if identity.get("controller_semantics_sha256", {}).get(episode) != actual:
                raise ValueError("prepared_controller_identity_mismatch")
    return identity


def _copy_subset(raw: Path, output: Path, episodes: int | list[str]) -> None:
    if isinstance(episodes, int) and episodes < 1:
        raise ValueError("episodes must be positive")
    with h5py.File(raw, "r") as source, h5py.File(output, "x") as destination:
        destination.attrs.update(source.attrs)
        data = destination.create_group("data")
        data.attrs.update(source["data"].attrs)
        groups = sorted(source["data"], key=natural_key)[:episodes] if isinstance(episodes, int) else list(episodes)
        if len(set(groups)) != len(groups):
            raise ValueError("duplicate source episode request")
        if not groups:
            raise ValueError("raw dataset has no episodes")
        for group in groups:
            source.copy(source["data"][group], data, name=group)
        data.attrs["total"] = sum(len(data[group]["actions"]) for group in groups)


def worker(raw: Path, output: Path, task_name: str, episodes: int | list[str] | None) -> None:
    import mimicgen.env_interfaces.robocasa  # noqa: F401
    import cfact.adapters.interfaces  # noqa: F401
    from cfact.adapters.robocasa import official_prepare_function

    task = target_task(task_name)
    with h5py.File(raw, "r") as source:
        selected = sorted(source["data"], key=natural_key)
    selected = selected[:episodes] if isinstance(episodes, int) else (episodes or selected)
    identity = prepared_identity(raw, selected)
    if episodes is not None:
        # Upstream copies every episode but annotates only n; create a true subset
        # first so unannotated episodes cannot leak into the prepared output.
        _copy_subset(raw, output, episodes)
        raw = output
    official_prepare_function()(
        dataset_path=str(raw),
        output_path=None if episodes is not None else str(output),
        env_interface_name=task.adapter_interface or task.mimicgen_interface,
        env_interface_type="robosuite",
        n=None,
    )
    with h5py.File(output, "a") as dataset:
        identity["controller_semantics_sha256"] = {
            demo: hashlib.sha256(dataset[f"data/{demo}/datagen_info"].attrs["cfact_controller_semantics"].encode()).hexdigest()
            for demo in selected
        }
        dataset.attrs["cfact_prepared_identity"] = json.dumps(identity, sort_keys=True)
    validate_prepared_identity(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--demo", action="append", help="Explicit source episode; repeat for root and donor")
    parser.add_argument("--resume", action="store_true", help="Preserve and quarantine an interrupted partial output before retry")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    task = target_task(args.task)
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    if args.demo and args.max_episodes is not None:
        parser.error("use either --demo or --max-episodes")
    raw, output = args.raw.resolve(), args.output.resolve()
    if not raw.is_file():
        raise FileNotFoundError(raw)
    if output.exists():
        raise FileExistsError(output)
    if args.worker:
        worker(raw, output, args.task, args.demo or args.max_episodes)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    lock = RunLock(output.with_suffix(output.suffix + ".lock"))
    lock.acquire()
    try:
        if partial.exists():
            if not args.resume:
                raise FileExistsError(f"Inspect the previous incomplete attempt: {partial}")
            partial.rename(partial.with_name(partial.name + f".interrupted-{time.time_ns()}"))
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(args.gpu),
                "MUJOCO_EGL_DEVICE_ID": str(args.gpu),
                "MUJOCO_GL": "egl",
            }
        )
        command = [
            sys.executable,
            "-m",
            "cfact.sources.prepare",
            "--worker",
            "--raw",
            str(raw),
            "--output",
            str(partial),
            "--task",
            args.task,
        ]
        if args.max_episodes:
            command.extend(["--max-episodes", str(args.max_episodes)])
        for demo in args.demo or []:
            command.extend(["--demo", demo])
        log = output.with_suffix(output.suffix + ".log")
        with log.open("a" if args.resume else "x") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = validate_prepared_dataset(
            partial,
            expected_task=args.task,
            expected_interface=task.adapter_interface or task.mimicgen_interface,
        )
        report["prepared_identity"] = validate_prepared_identity(partial, raw=raw)
        partial.rename(output)
        report.update(
            {
                "raw_source": str(raw),
                "raw_sha256": sha256_file(raw),
                "prepared_sha256": sha256_file(output),
            }
        )
        atomic_json(output.with_suffix(output.suffix + ".validation.json"), report)
        print(json.dumps({"output": str(output), **report}, indent=2))
    finally:
        lock.release()


if __name__ == "__main__":
    main()
