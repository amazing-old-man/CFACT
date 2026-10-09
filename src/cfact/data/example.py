"""Extract sibling outcome contrasts and read one measured transition."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from cfact import TreeDataset


def summarize(run: Path, task: str | None = None, output: Path | None = None) -> dict:
    if (run / "release.json").is_file():
        return summarize_release(run, task, output)
    dataset = TreeDataset(run)
    roots = dataset.roots(task=task)
    if not roots:
        raise ValueError("No roots matched the requested task")
    counts = {"roots": len(roots), "physical_edges": 0, "outcome_pairs": 0}
    transition = None
    stream = None
    if output:
        output = output.resolve()
        if output.is_relative_to(dataset.run_root):
            raise ValueError("Write derived pairs outside the source dataset")
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = output.open("x", encoding="utf-8")
    try:
        for row in roots:
            if row["status"] != "PHYSICS_COMPLETE":
                continue
            keys = {"task": row["task"], "root_id": row["root_id"], "split": row["split"]}
            for pair in dataset.iter_outcome_pairs(**keys):
                counts["outcome_pairs"] += 1
                if stream:
                    stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
            for edge in dataset.iter_edges(**keys):
                if edge.get("status") != "PHYSICS_COMPLETE" or not edge.get("edge_path"):
                    continue
                counts["physical_edges"] += 1
                if transition is None:
                    values = dataset.read_segment(**keys, edge_id=edge["edge_path"])
                    if len(values["commanded_actions"]):
                        item = dataset.read_transition(**keys, edge_id=edge["edge_path"], index=0)
                        transition = {
                            "task": row["task"],
                            "root_id": row["root_id"],
                            "edge": edge["edge_path"],
                            "action_shape": list(item["action_t"].shape),
                            "time_t": float(item["state_t"]["sim_time"]),
                            "time_t_plus_1": float(item["state_t_plus_1"]["sim_time"]),
                        }
    finally:
        if stream:
            stream.close()
    return {
        **counts,
        "first_transition": transition,
        "synthetic_fixture": (dataset.run_root / "FIXTURE.json").exists(),
        "label_note": "horizon_censored is not a proven failure; pairs are not preference labels",
    }


def summarize_release(run: Path, task: str | None, output: Path | None) -> dict:
    from cfact import ReleaseDataset

    dataset = ReleaseDataset(run)
    roots = dataset.roots(task=task)
    if not roots:
        raise ValueError("No roots matched the requested task")
    stream = None
    if output:
        if output.resolve().is_relative_to(dataset.root):
            raise ValueError("Write derived pairs outside the source release")
        output.parent.mkdir(parents=True, exist_ok=True)
        stream = output.open("x")
    pairs, edge_count, first = 0, 0, None
    try:
        for root in roots:
            for pair in dataset.iter_outcome_pairs(root["root_id"]):
                pairs += 1
                if stream:
                    stream.write(json.dumps(pair) + "\n")
            for edge in dataset.tree(root["root_id"])["edges"]:
                if edge["status"] != "PHYSICS_COMPLETE":
                    continue
                edge_count += 1
                if first is None:
                    trace = dataset.trace(edge["edge_id"])
                    if len(trace["commanded_actions"]):
                        first = {
                            "task": root["task"],
                            "root_id": root["root_id"],
                            "edge": edge["edge_id"],
                            "action_shape": list(trace["commanded_actions"][0].shape),
                            "time_t": float(trace["sim_time"][0]),
                            "time_t_plus_1": float(trace["sim_time"][1]),
                        }
    finally:
        if stream:
            stream.close()
    return {
        "roots": len(roots),
        "physical_edges": edge_count,
        "outcome_pairs": pairs,
        "first_transition": first,
        "synthetic_fixture": False,
        "label_note": "horizon_censored is not a proven failure; pairs are not preference labels",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Existing raw tree dataset")
    source.add_argument("--demo", type=Path, help="Create a new synthetic format fixture here")
    parser.add_argument("--task")
    parser.add_argument("--output", type=Path, help="New JSONL output outside the source dataset")
    args = parser.parse_args()
    if args.demo:
        from .demo import make_demo

        args.run = make_demo(args.demo)
    print(json.dumps(summarize(args.run, args.task, args.output), indent=2))


if __name__ == "__main__":
    main()
