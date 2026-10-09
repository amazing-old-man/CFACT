"""Run one physical tree from an original RoboCasa bundle and verify every saved trace."""

from __future__ import annotations
import argparse
from collections import Counter
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
import h5py
import numpy as np
from cfact import TreeDataset, ReleaseDataset
from cfact.data.example import summarize
from cfact.io.serialization import atomic_json
from cfact.runtime.assets import runtime_root
from cfact.runtime.doctor import check_runtime
from cfact.sources.manifest import build_plan
from cfact.tasks import SUPPORTED_TASK_NAMES


def validate_run(run: Path, task: str) -> dict:
    dataset = TreeDataset(run)
    roots = dataset.roots(task=task)
    if len(roots) != 1 or roots[0]["status"] != "PHYSICS_COMPLETE":
        raise RuntimeError(f"Expected one PHYSICS_COMPLETE root, got {roots}")
    row = roots[0]
    keys = {key: row[key] for key in ("task", "root_id", "split")}
    edges = list(dataset.iter_edges(**keys))
    statuses = dict(Counter(edge.get("status") for edge in edges))
    if not edges or any(edge.get("status") != "PHYSICS_COMPLETE" for edge in edges):
        raise RuntimeError(f"Physical edges missing or incomplete: {statuses}")
    segments, nonempty, actions = 0, 0, 0
    for edge in edges:
        for segment in ("local", "fixed_continuation"):
            trace = dataset.read_segment(**keys, edge_id=edge["edge_path"], segment=segment)
            count = len(trace["commanded_actions"])
            if not np.isfinite(trace["commanded_actions"]).all():
                raise ValueError("Nonfinite commanded actions")
            # TreeDataset checks every K/K+1 field before returning the segment.
            segments += 1
            nonempty += int(count > 0)
            actions += count
    if not actions:
        raise RuntimeError("No executed action transitions were produced")
    return {
        "roots": len(roots),
        "edge_statuses": statuses,
        "edges": len(edges),
        "trace_segments": segments,
        "nonempty_segments": nonempty,
        "actions": actions,
    }


def quickstart(source: Path, output: Path, task: str, gpu: int, episodes: int) -> dict:
    if episodes < 2:
        raise ValueError("Use at least 2 episodes: one root and one disjoint donor")
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "task": task, "source": str(source), "steps": []}
    report_path = output / "acceptance.json"
    atomic_json(report_path, state)
    try:
        doctor = check_runtime(runtime_root(), render=True, gpu=gpu)
        atomic_json(output / "environment.json", doctor)
        if doctor["status"] != "PASS":
            raise RuntimeError(f"Runtime checks failed; see {output / 'environment.json'}")
        state["steps"].append("doctor")

        def invoke(name, *arguments):
            log = output / f"{name}.log"
            print(f"{name}: {log}", flush=True)
            with log.open("x") as stream:
                result = subprocess.run(
                    [sys.executable, "-m", "cfact", *map(str, arguments)],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}); see {log}")
            state["steps"].append(name)
            atomic_json(report_path, state)

        raw, prepared = output / "raw.hdf5", output / "prepared.hdf5"
        invoke(
            "import",
            "import-source",
            "--source",
            source,
            "--output",
            raw,
            "--task",
            task,
            "--max-episodes",
            episodes,
        )
        invoke(
            "prepare_source",
            "prepare-source",
            "--raw",
            raw,
            "--output",
            prepared,
            "--task",
            task,
            "--gpu",
            gpu,
        )
        with h5py.File(prepared, "r") as dataset:
            groups = sorted(dataset["data"], key=lambda value: int(value.rsplit("_", 1)[-1]))
        if len(groups) < 2:
            raise ValueError("The source bundle needs at least two distinct demonstrations")
        sources = output / "sources.json"
        atomic_json(
            sources,
            {
                "tasks": [
                    {
                        "task": task,
                        "prepared_hdf5": str(prepared),
                        "roots": groups[:1],
                        "donors": groups[1:],
                    }
                ],
                "output_root": str(output / "releases"),
            },
        )
        parameters = json.loads(files("cfact.resources").joinpath("collection.json").read_text())
        # Use the published protocol unchanged; only the source inventory is small.
        atomic_json(output / "parameters.json", parameters)
        build_plan(sources, output / "plan", output / "parameters.json")
        state["steps"].append("prepare_plan")
        invoke(
            "collect",
            "collect",
            "--config",
            output / "plan/config.json",
            "--source-manifest",
            output / "plan/source_manifest.json",
            "--stage",
            "pilot",
            "--scene-workers",
            "1",
            "--gpu-devices",
            gpu,
            "--run-label",
            "pilot",
        )
        release = output / "releases/pilot"
        run = release / "_work/collection"
        traces = validate_run(run, task)
        public = ReleaseDataset(release)
        readback = json.loads(public.inside(public.manifest["official_readback"]).read_text())
        if (
            readback["status"] != "PASS"
            or public.manifest["frames"] != traces["actions"]
            or public.manifest["episodes"] != traces["nonempty_segments"]
        ):
            raise RuntimeError("Published release does not match measured collection traces")
        example = summarize(release, task, output / "outcome_pairs.jsonl")
        if example["synthetic_fixture"] or example["first_transition"] is None:
            raise RuntimeError("Physical reader acceptance did not produce a transition")
        state.update(
            {
                "status": "PASS",
                "run": str(run),
                "traces": traces,
                "reader": example,
                "profile": "one root and disjoint donors; unchanged packaged generation protocol",
                "release": str(release),
                "scope": "End-to-end execution and format integrity, not all-task scientific validation",
            }
        )
        state["steps"].append("readback")
    except Exception as error:
        state.update({"status": "FAIL", "error": str(error)})
        raise
    finally:
        atomic_json(report_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Original tar or extracted LeRobot directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory outside the code repository"
    )
    parser.add_argument("--task", choices=SUPPORTED_TASK_NAMES, default="OpenDrawer")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            quickstart(args.source, args.output, args.task, args.gpu, args.max_episodes), indent=2
        )
    )


if __name__ == "__main__":
    main()
