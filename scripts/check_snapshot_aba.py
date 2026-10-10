import argparse
import json
from pathlib import Path
import numpy as np
from cfact.collection.runtime import _load_root, _create_env
from cfact.execution.snapshot import (
    load_snapshot,
    capture_snapshot,
    restore_snapshot,
    save_snapshot,
)
from cfact.io.review_video import QuadReviewWriter
from cfact.runtime.resources import admission
from cfact.run import load_config, parameters

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--gpu", type=int, default=0)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
manifest = json.loads((a.run / "manifest.json").read_text())
cfg = parameters(load_config(a.run / "config.yaml"), 1, None)
reports = []
for taskrow in manifest["tasks"]:
    task = taskrow["task"]
    rootdemo = "demo_0"
    source = _load_root(Path(taskrow["prepared_hdf5"]), rootdemo)
    gate = admission(a.output, [a.gpu], 1, cfg["resource"])
    if gate["status"] != "PASS":
        raise RuntimeError("ABA_RESOURCE_BLOCKED")
    trees = [json.loads(path.read_text()) for path in (a.run / "trees" / task).glob("*/tree.json")]
    # Freeze the demo_0 tree; no selection based on success or structural richness.
    source_id = next(
        json.loads(p.read_text())["source_id"]
        for p in (a.run / "sources" / task).glob("*/episode_000000/source_manifest.json")
    )
    tree = next(t for t in trees if t["source_id"] == source_id)
    decision = [node for node in tree["nodes"] if node["decision_node"]]
    env = _create_env(source["env_meta"], manifest["seed"], "robot0_agentview_left", a.gpu)
    try:
        env.reset_to(source["initial"])
        for node in decision:
            folder = a.output / task / node["node_id"]
            folder.mkdir(parents=True)
            context = json.loads((a.run / node["context"]).read_text())
            snapshot = load_snapshot(a.run / context["snapshot"]["path"])
            parent_state = snapshot["states"].copy()
            edges = [e for e in tree["edges"] if e["parent_node"] == node["node_id"]]
            actions = []
            for edge in edges:
                path = a.run / edge["path"] / "local/execution.npz"
                if path.exists():
                    with np.load(path) as data:
                        if len(data["commanded_actions"]):
                            actions.append((edge["edge_id"], 0, data["commanded_actions"][0].copy()))
            if not actions:
                reports.append(
                    {
                        "task": task,
                        "node": node["node_id"],
                        "status": "NOT_VERIFIED_NO_ISSUED_COMMAND",
                    }
                )
                continue
            command_a = actions[0]
            command_b = next(
                (x for x in actions[1:] if not np.array_equal(x[2], command_a[2])), None
            )
            if command_b is None:
                # First commands can coincide after interpolation. Use an already issued
                # later command from a sibling for this restore diagnostic, without
                # modifying a candidate or inventing a perturbation.
                for edge in edges[1:]:
                    path = a.run / edge["path"] / "local/execution.npz"
                    if not path.exists():
                        continue
                    with np.load(path) as data:
                        for index, command in enumerate(data["commanded_actions"]):
                            if not np.array_equal(command, command_a[2]):
                                command_b = (edge["edge_id"], index, command.copy())
                                break
                    if command_b is not None:
                        break
            command_b = command_b or actions[-1]
            trials = []
            for ordinal, (label, command) in enumerate(
                [("A", command_a), ("B", command_b), ("A", command_a)]
            ):
                restore_report = restore_snapshot(env, snapshot, initial=source["initial"])
                restored = capture_snapshot(env)
                before_time = float(env.env.sim.data.time)
                before_rgb = np.stack(
                    [
                        env.env.sim.render(height=224, width=224, camera_name=c)[::-1].copy()
                        for c in QuadReviewWriter.CAMERAS
                    ]
                )
                if env.env.sim.data.time != before_time:
                    raise RuntimeError("RENDER_CHANGED_CLOCK")
                if (
                    not np.array_equal(restored["states"], parent_state)
                    or restore_report["physics_steps"] != 0
                ):
                    raise RuntimeError("RESTORE_CHANGED_STATE")
                save_snapshot(folder / f"{ordinal}_{label}_restored_snapshot.json", restored)
                env.step(command[2])
                after = capture_snapshot(env)
                after_rgb = np.stack(
                    [
                        env.env.sim.render(height=224, width=224, camera_name=c)[::-1].copy()
                        for c in QuadReviewWriter.CAMERAS
                    ]
                )
                np.savez_compressed(
                    folder / f"{ordinal}_{label}.npz",
                    restored_state=restored["states"],
                    executed_command=command[2],
                    end_state=after["states"],
                    rgb_before=before_rgb,
                    rgb_after=after_rgb,
                    time_before=before_time,
                    time_after=float(env.env.sim.data.time),
                )
                trials.append(
                    {
                        "label": label,
                        "edge_command_origin": command[0],
                        "edge_command_index": command[1],
                        "restored_state": restored["states"],
                        "end_state": after["states"],
                        "rgb_before": before_rgb,
                        "rgb_after": after_rgb,
                        "restore_report": restore_report,
                    }
                )
            state_delta = float(np.abs(trials[0]["end_state"] - trials[2]["end_state"]).max())
            rgb_delta = int(
                np.abs(
                    trials[0]["rgb_after"].astype(int) - trials[2]["rgb_after"].astype(int)
                ).max()
            )
            if state_delta > 1e-10:
                raise RuntimeError("ABA_NONREPEATABLE_PHYSICS")
            reports.append(
                {
                    "task": task,
                    "node": node["node_id"],
                    "layer": "Root" if node["node_id"] == tree["root_node"] else "Desc",
                    "status": "PASS",
                    "A_B_commands_differ": not np.array_equal(command_a[2], command_b[2]),
                    "A_repeat_state_abs_max": state_delta,
                    "A_repeat_rgb_abs_max": rgb_delta,
                    "issued_command_origins": [
                        {"trial": i, "label": trial["label"],
                         "edge_id": trial["edge_command_origin"],
                         "action_index": trial["edge_command_index"]}
                        for i, trial in enumerate(trials)
                    ],
                    "restore_clock_exact": True,
                    "hidden_restore_steps": 0,
                    "camera_shape": [3, 224, 224, 3],
                    "diagnostic_control_steps": 3,
                    "render_bit_exact_claim": rgb_delta == 0,
                }
            )
    finally:
        env.env.close()
(a.output / "ABA_VALIDATION.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
print(json.dumps(reports, indent=2))
