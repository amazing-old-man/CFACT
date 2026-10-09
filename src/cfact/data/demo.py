"""Create a tiny synthetic format fixture; never use it as physical evidence."""

from pathlib import Path
import json
import numpy as np


def make_demo(run: str | Path) -> Path:
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "FIXTURE.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "purpose": "reader demonstration only; no robot or simulator was executed",
            },
            indent=2,
        )
    )
    root = run / "data/example/ExampleTask/root1"
    root.mkdir(parents=True)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    write(
        root / "root_result.json",
        {
            "task": "ExampleTask",
            "root_id": "root1",
            "split": "example",
            "status": "PHYSICS_COMPLETE",
            "synthetic_fixture": True,
        },
    )
    write(root / "root_manifest.json", {"source_episode": "synthetic_source"})

    def edge(node, identity, outcome, offset):
        folder = root / "nodes" / node / "edges" / identity
        folder.mkdir(parents=True)
        write(folder.parent.parent / "context.json", {"node_id": node, "synthetic_fixture": True})
        np.savez(
            folder / "execution.npz",
            commanded_actions=np.ones((2, 12)),
            raw_planned_actions=np.ones((2, 12)),
            controller_target_poses=np.repeat(np.eye(4)[None], 2, axis=0),
            local_event_flags=np.array([False, True]),
            sim_time=np.array([0.0, 0.05, 0.10]) + offset,
            robot_qpos=np.zeros((3, 7)),
        )
        write(
            folder / "option.json",
            {
                "parent_context_id": f"root1/{node}",
                "source_method": "Example",
                "synthetic_fixture": True,
            },
        )
        return {
            "candidate_id": identity,
            "edge_path": str(folder.relative_to(run)),
            "status": "PHYSICS_COMPLETE",
            "Y_fixed": outcome,
        }

    a = edge("n_root", "a", "horizon_censored", 0.0)
    b = edge("n_root", "b", "official_success", 0.0)
    c = edge("n_child", "c", "official_success", 0.10)
    write(
        root / "tree.json",
        {
            "root": "n_root",
            "edges": [a, b],
            "child_pools": [{"node_id": "n_child", "parent_edge": a["edge_path"], "edges": [c]}],
            "third_pools": [],
            "synthetic_fixture": True,
        },
    )
    return run
