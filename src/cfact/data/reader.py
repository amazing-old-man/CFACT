"""Read measured physical rollout trees without flattening their branches."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from cfact.storage.schema import training_observation, validate_child_context


class TreeDataset:
    """Read roots, nodes, edges, transitions, and connected edge paths from one run."""

    def __init__(self, run_root: str | Path):
        self.run_root = Path(run_root).resolve()
        if not (self.run_root / "data").is_dir():
            raise FileNotFoundError(f"Expected a raw tree run containing data/: {self.run_root}")
        self._roots = list((self.run_root / "data").glob("*/*/*/root_result.json"))

    @cached_property
    def _records(self):
        """Read root metadata once for a frozen dataset, avoiding quadratic disk I/O."""
        rows = []
        for path in sorted(self._roots):
            manifest_path = path.parent / "root_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            rows.append((path, json.loads(path.read_text()), manifest))
        return rows

    def _inside(self, relative: str | Path) -> Path:
        path = (self.run_root / relative).resolve()
        if not path.is_relative_to(self.run_root):
            raise ValueError(f"path escapes run root: {relative}")
        return path

    def roots(self, *, task: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
        result = []
        for path, root, manifest in self._records:
            if task is not None and root.get("task") != task:
                continue
            if split is not None and root.get("split") != split:
                continue
            result.append(
                {
                    "task": root.get("task"),
                    "split": root.get("split"),
                    "request_id": manifest.get("source_episode"),
                    "root_id": root.get("root_id"),
                    "role_in_release": manifest.get("role_in_release"),
                    "status": root.get("status"),
                    "root_result": str(path.relative_to(self.run_root)),
                    "tree": str((path.parent / "tree.json").relative_to(self.run_root)),
                }
            )
        return result

    def _root_dir(self, task: str, root_id: str, split: str | None = None) -> Path:
        matches = []
        for path, result, manifest in self._records:
            if result.get("task") != task:
                continue
            if split is not None and result.get("split") != split:
                continue
            identities = {str(result.get("root_id", "")), str(manifest.get("source_episode", ""))}
            if root_id in identities:
                matches.append(path.parent)
        if len(matches) != 1:
            raise KeyError(f"expected one root for {task}/{root_id}, found {len(matches)}")
        return matches[0]

    def read_tree(self, task: str, root_id: str, *, split: str | None = None) -> dict[str, Any]:
        path = self._root_dir(task, root_id, split) / "tree.json"
        return json.loads(path.read_text())

    def read_node(
        self, task: str, root_id: str, node_id: str, *, split: str | None = None
    ) -> dict[str, Any]:
        root_dir = self._root_dir(task, root_id, split)
        if not node_id or Path(node_id).name != node_id or node_id in {".", ".."}:
            raise ValueError("node_id must be a single directory name")
        path = self._inside(root_dir / "nodes" / node_id)
        context_path = path / "context.json"
        if not context_path.is_file():
            raise KeyError(f"node not found: {node_id}")
        context = json.loads(context_path.read_text())
        state_path = path / "context.npz"
        if state_path.is_file():
            with np.load(state_path, allow_pickle=False) as saved:
                context = validate_child_context(context, saved["sim_state"])
        context["path"] = str(path.relative_to(self.run_root))
        return context

    def _pools(self, tree: dict[str, Any]) -> Iterator[tuple[str | None, list[dict[str, Any]]]]:
        yield None, tree.get("edges", [])
        for key in ("child_pools", "third_pools"):
            for pool in tree.get(key, []):
                yield pool.get("parent_edge"), pool.get("edges", [])

    def read_tree_path(
        self, task: str, root_id: str, edge_ids: list[str], *, split: str | None = None
    ) -> list[dict[str, Any]]:
        """Resolve an explicitly selected root-to-child chain by candidate ID or edge path."""
        tree = self.read_tree(task, root_id, split=split)
        selected: list[dict[str, Any]] = []
        parent_edge: str | None = None
        for edge_identity in edge_ids:
            candidates = [
                edge
                for parent, edges in self._pools(tree)
                if parent == parent_edge
                for edge in edges
                if edge.get("candidate_id") == edge_identity
                or edge.get("edge_path") == edge_identity
            ]
            if len(candidates) != 1:
                raise KeyError(f"edge {edge_identity!r} is not a unique child of {parent_edge!r}")
            edge = candidates[0]
            if not edge.get("edge_path"):
                raise ValueError(f"selected edge has no physical trace: {edge_identity}")
            selected.append(edge)
            parent_edge = edge["edge_path"]
        return selected

    def read_edge(
        self, task: str, root_id: str, edge_id: str, *, split: str | None = None
    ) -> dict[str, Any]:
        tree = self.read_tree(task, root_id, split=split)
        matches = [
            edge
            for _, edges in self._pools(tree)
            for edge in edges
            if edge_id in {edge.get("candidate_id"), edge.get("edge_path")}
        ]
        if len(matches) != 1:
            raise KeyError(
                f"expected one edge for {edge_id!r}, found {len(matches)}; use its edge_path"
            )
        row = dict(matches[0])
        if row.get("edge_path"):
            path = self._inside(row["edge_path"])
            for key, relative in (
                ("option", "option.json"),
                ("local_result", "local_result.json"),
                ("fixed_continuation_result", "continuation/result.json"),
                ("media", "media.json"),
                ("goal_realization", "goal_realization.json"),
            ):
                record_path = path / relative
                if record_path.is_file():
                    row[key] = json.loads(record_path.read_text())
        return row

    def read_segment(
        self,
        task: str,
        root_id: str,
        edge_id: str,
        segment: str = "local",
        *,
        split: str | None = None,
    ) -> dict[str, np.ndarray]:
        if segment not in {"local", "fixed_continuation"}:
            raise ValueError("segment must be 'local' or 'fixed_continuation'")
        edge = self.read_edge(task, root_id, edge_id, split=split)
        edge_dir = self._inside(edge["edge_path"])
        path = edge_dir / ("execution.npz" if segment == "local" else "continuation/execution.npz")
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name].copy() for name in archive.files}
        count = len(values["commanded_actions"])
        if len(values["sim_time"]) != count + 1:
            raise ValueError(f"segment violates K/K+1 alignment: {path}")
        for key in ("raw_planned_actions", "controller_target_poses", "local_event_flags"):
            if key not in values or len(values[key]) != count:
                raise ValueError(f"segment violates action alignment for {key}: {path}")
        for key in (
            "robot_qpos",
            "robot_qvel",
            "actual_eef_pose",
            "object_poses",
            "fixture_poses",
            "task_success",
        ):
            if key in values and len(values[key]) != count + 1:
                raise ValueError(f"segment violates state alignment for {key}: {path}")
        if count and (
            not np.isfinite(values["commanded_actions"]).all()
            or not np.isfinite(values["sim_time"]).all()
            or np.any(np.diff(values["sim_time"]) <= 0)
        ):
            raise ValueError(f"segment has invalid actions or timestamps: {path}")
        return values

    def read_transition(
        self,
        task: str,
        root_id: str,
        edge_id: str,
        index: int,
        segment: str = "local",
        *,
        split: str | None = None,
    ) -> dict[str, Any]:
        values = self.read_segment(task, root_id, edge_id, segment, split=split)
        count = len(values["commanded_actions"])
        if not 0 <= index < count:
            raise IndexError(f"transition index {index} outside [0, {count})")
        transition_keys = [
            "commanded_actions",
            "raw_planned_actions",
            "controller_target_poses",
            "local_event_flags",
        ]
        return {
            "index": index,
            "segment": segment,
            "state_t": training_observation(values, index),
            "action_t": values["commanded_actions"][index],
            "raw_planned_action_t": values["raw_planned_actions"][index],
            "controller_target_pose_t": values["controller_target_poses"][index],
            "local_event_reached": bool(values["local_event_flags"][index]),
            "controller_transition_reason": str(values["controller_transition_reasons"][index])
            if "controller_transition_reasons" in values
            else None,
            "state_t_plus_1": training_observation(values, index + 1),
            "outcome": {
                "task_success_before": bool(values["task_success"][index])
                if "task_success" in values else None,
                "task_success_after": bool(values["task_success"][index + 1])
                if "task_success" in values else None,
            },
            "transition_fields": transition_keys,
        }

    def iter_edges(self, task: str, root_id: str, *, split: str | None = None):
        """Yield edge summaries with their real incoming parent edge."""
        for parent, edges in self._pools(self.read_tree(task, root_id, split=split)):
            for edge in edges:
                yield {**edge, "parent_edge": parent}

    def iter_outcome_pairs(self, task: str, root_id: str, *, split: str | None = None):
        """Yield observed sibling outcome contrasts, without inventing preferences.

        Horizon censoring remains a literal label. Unknown or technical results
        are excluded. These records are useful for outcome auditing; they do not
        assert that one action is universally better than another.
        """
        from itertools import combinations

        observed = {
            "official_success",
            "horizon_censored",
            "observed_failure",
            "observed_operation_failure",
        }
        for parent, edges in self._pools(self.read_tree(task, root_id, split=split)):
            valid = [
                edge
                for edge in edges
                if edge.get("status") == "PHYSICS_COMPLETE"
                and edge.get("edge_path")
                and edge.get("Y_fixed") in observed
            ]
            for left, right in combinations(valid, 2):
                if left["Y_fixed"] == right["Y_fixed"]:
                    continue
                yield {
                    "task": task,
                    "root_id": root_id,
                    "split": split,
                    "parent_edge": parent,
                    "left_edge": left["edge_path"],
                    "left_outcome": left["Y_fixed"],
                    "right_edge": right["edge_path"],
                    "right_outcome": right["Y_fixed"],
                }


__all__ = ["TreeDataset"]
