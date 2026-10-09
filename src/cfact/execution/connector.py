"""Budgeted joint-space connector for a caller-owned, frozen collision world.

The callback must validate the actual robot, gripper, held object and current
scene at every query. This module only searches; it never changes a simulator.
It checks joint-space segments at a declared resolution, not continuously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ConnectorBudget:
    maximum_samples: int = 512
    maximum_validity_queries: int = 10000
    steering_step_rad: float = 0.18
    validation_step_rad: float = 0.035
    maximum_path_length_rad: float = 8.0
    goal_bias: float = 0.1


class BoundedJointConnector:
    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        state_valid: Callable[[np.ndarray], bool],
        *,
        budget: ConnectorBudget = ConnectorBudget(),
        seed: int = 0,
    ):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if (
            self.lower.ndim != 1
            or self.upper.shape != self.lower.shape
            or not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or not np.all(self.lower < self.upper)
            or budget.maximum_samples < 1
            or budget.maximum_validity_queries < 2
            or budget.steering_step_rad <= 0
            or budget.validation_step_rad <= 0
            or budget.maximum_path_length_rad <= 0
            or not 0 <= budget.goal_bias <= 1
        ):
            raise ValueError("invalid_connector_configuration")
        self.state_valid = state_valid
        self.budget = budget
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.queries = 0

    def _valid(self, q: np.ndarray) -> bool:
        if self.queries >= self.budget.maximum_validity_queries:
            raise _BudgetExhausted
        self.queries += 1
        return bool(self.state_valid(q))

    def _segment_valid(self, a: np.ndarray, b: np.ndarray) -> bool:
        length = float(np.linalg.norm(b - a))
        count = max(1, int(np.ceil(length / self.budget.validation_step_rad)))
        for part in range(1, count + 1):
            if not self._valid(a + (b - a) * (part / count)):
                return False
        return True

    def _extend(
        self, tree: list[tuple[np.ndarray, int]], target: np.ndarray
    ) -> tuple[str, int | None]:
        nearest = min(
            range(len(tree)), key=lambda index: float(np.linalg.norm(tree[index][0] - target))
        )
        origin = tree[nearest][0]
        distance = float(np.linalg.norm(target - origin))
        if distance < 1e-12:
            return "REACHED", nearest
        step = min(distance, self.budget.steering_step_rad)
        proposed = origin + (target - origin) * (step / distance)
        if not self._segment_valid(origin, proposed):
            return "TRAPPED", None
        tree.append((proposed, nearest))
        return ("REACHED" if step == distance else "ADVANCED"), len(tree) - 1

    @staticmethod
    def _branch(tree: list[tuple[np.ndarray, int]], index: int) -> list[np.ndarray]:
        path = []
        while index != -1:
            q, index = tree[index]
            path.append(q)
        return path[::-1]

    def plan(self, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray | None, dict]:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if (
            start.shape != self.lower.shape
            or goal.shape != self.lower.shape
            or not np.isfinite(start).all()
            or not np.isfinite(goal).all()
            or np.any(start < self.lower)
            or np.any(start > self.upper)
            or np.any(goal < self.lower)
            or np.any(goal > self.upper)
        ):
            raise ValueError("invalid_connector_endpoint")
        status = "NO_PLAN_FOUND_WITHIN_BUDGET"
        samples = 0
        path = None
        try:
            if not self._valid(start) or not self._valid(goal):
                status = "INVALID_ENDPOINT"
            elif self._segment_valid(start, goal):
                if np.linalg.norm(goal - start) <= self.budget.maximum_path_length_rad:
                    path = [start, goal]
                    status = "DIRECT_PATH"
                else:
                    status = "PATH_LENGTH_LIMIT"
            else:
                trees = [[(start, -1)], [(goal, -1)]]
                for samples in range(1, self.budget.maximum_samples + 1):
                    side = (samples - 1) % 2
                    source, other = trees[side], trees[1 - side]
                    target = (
                        other[0][0]
                        if self.rng.random() < self.budget.goal_bias
                        else self.rng.uniform(self.lower, self.upper)
                    )
                    _, index = self._extend(source, target)
                    if index is None:
                        continue
                    while True:
                        state, other_index = self._extend(other, source[index][0])
                        if state == "TRAPPED":
                            break
                        if state == "REACHED":
                            assert other_index is not None
                            left = self._branch(trees[0], index if side == 0 else other_index)
                            right = self._branch(trees[1], other_index if side == 0 else index)
                            candidate = left + right[::-1][1:]
                            length = sum(
                                float(np.linalg.norm(b - a))
                                for a, b in zip(candidate, candidate[1:])
                            )
                            if length <= self.budget.maximum_path_length_rad:
                                path = candidate
                                status = "CONNECTED"
                            else:
                                status = "PATH_LENGTH_LIMIT"
                            break
                    if path is not None or status == "PATH_LENGTH_LIMIT":
                        break
        except _BudgetExhausted:
            status = "NO_PLAN_FOUND_WITHIN_BUDGET"
        report = {
            "backend": "bounded_bidirectional_joint_rrt_connect",
            "status": status,
            "seed": self.seed,
            "samples": samples,
            "validity_queries": self.queries,
            "maximum_samples": self.budget.maximum_samples,
            "maximum_validity_queries": self.budget.maximum_validity_queries,
            "steering_step_rad": self.budget.steering_step_rad,
            "validation_step_rad": self.budget.validation_step_rad,
            "maximum_path_length_rad": self.budget.maximum_path_length_rad,
            "goal_bias": self.budget.goal_bias,
            "continuous_collision": "NOT_CHECKED",
            "path_length_rad": (
                sum(float(np.linalg.norm(b - a)) for a, b in zip(path, path[1:]))
                if path is not None
                else None
            ),
        }
        return (np.asarray(path) if path is not None else None), report


class _BudgetExhausted(Exception):
    pass
