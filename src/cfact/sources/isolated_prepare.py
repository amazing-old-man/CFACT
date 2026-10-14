"""Prepare selected recorded Source episodes inside the round's write boundary.

This wrapper does not alter the pinned preparation identity or physics. The
upstream annotator reads recorded states; its env.step calls are counted here.
"""
import argparse
import os
from pathlib import Path
import resource
import time

from cfact.io.serialization import RunLock, atomic_json
from cfact.storage.policy import contained_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--demo", action="append", required=True)
    args = p.parse_args()
    root = args.round.resolve()
    out = args.output.resolve()
    contained_path(out, root)
    if os.environ.get("MUJOCO_GL") != "osmesa":
        raise ValueError("SOURCE_PREPARATION_REQUIRES_EXPLICIT_CPU_OSMESA")
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    if partial.exists():
        raise FileExistsError("Preserve and inspect the existing partial: " + str(partial))
    started = time.monotonic()
    steps = [0]
    status, error = "TECHNICAL_ERROR", None
    lock = RunLock(out.with_suffix(out.suffix + ".lock"))
    lock.acquire()
    try:
        from cfact.runtime.readonly_assets import install_generated_xml_redirect, install_write_boundary
        install_generated_xml_redirect(root)
        install_write_boundary(root)
        # Count any upstream physical control action instead of assuming that
        # annotation performs none. All normal annotation is reset_to(state).
        import robosuite.environments.base as base
        original = base.MujocoEnv.step
        def counted(self, *a, **kw):
            steps[0] += 1
            return original(self, *a, **kw)
        base.MujocoEnv.step = counted
        from cfact.sources.prepare import worker, validate_prepared_identity
        from cfact.data.validation import validate_prepared_dataset
        from cfact.sources.manifest import sha256_file
        from cfact.tasks import target_task
        worker(args.raw.resolve(), partial, args.task, args.demo)
        task = target_task(args.task)
        validation = validate_prepared_dataset(partial, expected_task=args.task,
            expected_interface=task.adapter_interface or task.mimicgen_interface)
        identity = validate_prepared_identity(partial, raw=args.raw.resolve(), required_episodes=args.demo)
        partial.rename(out)
        atomic_json(out.with_suffix(out.suffix + ".validation.json"),
            {**validation, "prepared_identity": identity, "prepared_sha256": sha256_file(out),
             "raw_source": str(args.raw.resolve()), "raw_sha256": identity["raw_sha256"]})
        status = "COMPLETE"
    except Exception as exc:
        error = repr(exc)
        raise
    finally:
        elapsed = time.monotonic() - started
        from cfact.collection.snapshot import implementation_digest
        receipt = dict(job_id="prepare_" + out.stem, task=args.task, status=status,
            error=error, run=str(out), source_groups=args.demo, backend="osmesa",
            cpu_worker_seconds=elapsed, gpu_worker_seconds=0., wall_seconds=elapsed,
            control_steps=steps[0], peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            code_sha256=implementation_digest(), recorded_state_annotation=True)
        atomic_json(root/"logs/receipts"/(receipt["job_id"]+".json"), receipt)
        lock.release()


if __name__ == "__main__":
    main()
