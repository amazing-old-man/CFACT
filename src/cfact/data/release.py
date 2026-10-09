"""Read portable tree/LeRobot releases and resolve both directions of their mappings."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from cfact.storage.layout import SCHEMA, inside


class ReleaseDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "release.json").read_text())
        if self.manifest.get("schema") != SCHEMA or self.manifest.get("status") != "COMPLETE":
            raise ValueError("Expected a complete cfact.dual-view.v1 release")
        self._trees = [json.loads(inside(self.root, p).read_text()) for p in self.manifest["trees"]]
        self._episodes = None
        self._terminals = None

    def inside(self, relative: str) -> Path:
        return inside(self.root, relative)

    def roots(self, task: str | None = None, demo_id: str | None = None) -> list[dict]:
        keys = ("task", "source_package", "source_id", "demo_id", "root_id", "status", "tree_path")
        return [
            {k: t[k] for k in keys}
            for t in self._trees
            if (task is None or t["task"] == task) and (demo_id is None or t["demo_id"] == demo_id)
        ]

    def tree(self, root_id: str) -> dict:
        rows = [t for t in self._trees if t["root_id"] == root_id]
        if len(rows) != 1:
            raise KeyError(root_id)
        return rows[0]

    def edge(self, edge_id: str) -> dict:
        rows = [e for t in self._trees for e in t["edges"] if e["edge_id"] == edge_id]
        if len(rows) != 1:
            raise KeyError(edge_id)
        return rows[0]

    def node(self, root_id: str, node_id: str) -> dict:
        from cfact.storage.schema import validate_child_context

        matches = [n for n in self.tree(root_id)["nodes"] if n["node_id"] == node_id]
        if len(matches) != 1:
            raise KeyError(node_id)
        row = matches[0]
        context = json.loads(self.inside(row["context"]).read_text())
        with np.load(self.inside(row["state"]), allow_pickle=False) as saved:
            return validate_child_context(context, saved["sim_state"])

    def episodes(self, edge_id: str | None = None) -> list[dict]:
        if self._episodes is None:
            import pyarrow.parquet as pq

            self._episodes = pq.read_table(
                self.inside(self.manifest["episode_tree_map"])
            ).to_pylist()
        return [r for r in self._episodes if edge_id is None or r["edge_id"] == edge_id]

    def resolve_episode(self, shard_id: str, episode_index: int) -> dict:
        rows = [
            r
            for r in self.episodes()
            if r["shard_id"] == shard_id and r["episode_index"] == episode_index
        ]
        if len(rows) != 1:
            raise KeyError((shard_id, episode_index))
        return rows[0]

    def terminals(self, edge_id: str | None = None) -> list[dict]:
        if self._terminals is None:
            import pyarrow.parquet as pq

            self._terminals = pq.read_table(
                self.inside(self.manifest["terminal_observation_map"])
            ).to_pylist()
        return [r for r in self._terminals if edge_id is None or r["edge_id"] == edge_id]

    def trace(self, edge_id: str, segment: str = "local") -> dict:
        if segment not in {"local", "fixed_continuation"}:
            raise ValueError("Unknown segment")
        edge = self.edge(edge_id)
        with np.load(
            self.inside(edge["path"]) / segment / "execution.npz", allow_pickle=False
        ) as values:
            arrays = {k: values[k] for k in values.files}
        count = len(arrays["commanded_actions"])
        if len(arrays["sim_time"]) != count + 1:
            raise ValueError("Measured trace violates K/K+1")
        return arrays

    def iter_outcome_pairs(self, root_id: str):
        from itertools import combinations

        observed = {
            "official_success",
            "horizon_censored",
            "observed_failure",
            "observed_operation_failure",
        }
        tree = self.tree(root_id)
        for node in tree["nodes"]:
            rows = []
            for edge in tree["edges"]:
                if edge["parent_node"] != node["node_id"] or edge["status"] != "PHYSICS_COMPLETE":
                    continue
                result = json.loads(self.inside(edge["result"]).read_text())
                outcome = result["fixed_continuation"].get("status")
                if outcome in observed:
                    rows.append((edge["edge_id"], outcome))
            for left, right in combinations(rows, 2):
                if left[1] != right[1]:
                    yield {
                        "task": tree["task"],
                        "root_id": root_id,
                        "parent_node": node["node_id"],
                        "left_edge": left[0],
                        "right_edge": right[0],
                        "left_outcome": left[1],
                        "right_outcome": right[1],
                    }

    def open_shard(self, row: dict):
        from cfact.export.shards import open_local_shard

        return open_local_shard(self.root, row)
