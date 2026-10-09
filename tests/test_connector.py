"""Budgeted connector resolves a blocked route and reports bounded failure."""

from __future__ import annotations

import numpy as np

from cfact.execution.connector import BoundedJointConnector, ConnectorBudget


def test_connector_routes_around_obstacle_with_checked_segments() -> None:
    lower = np.array([-2.0, -2.0])
    upper = np.array([2.0, 2.0])

    def valid(q):
        return np.linalg.norm(q) >= 0.55

    budget = ConnectorBudget(
        maximum_samples=512,
        maximum_validity_queries=10000,
        steering_step_rad=0.2,
        validation_step_rad=0.02,
        maximum_path_length_rad=6.0,
    )
    connector = BoundedJointConnector(lower, upper, valid, budget=budget, seed=7)
    path, report = connector.plan(np.array([-1.0, 0.0]), np.array([1.0, 0.0]))
    assert path is not None and report["status"] == "CONNECTED"
    assert report["validity_queries"] <= budget.maximum_validity_queries
    assert np.array_equal(path[0], [-1.0, 0.0]) and np.array_equal(path[-1], [1.0, 0.0])
    for a, b in zip(path, path[1:]):
        for q in np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a) / 0.02)) + 1)):
            assert valid(q)


def test_connector_rejects_path_exceeding_length_and_query_budget() -> None:
    lower = np.array([-2.0, -2.0])
    upper = np.array([2.0, 2.0])
    start = np.array([-1.0, 0.0])
    goal = np.array([1.0, 0.0])
    short = BoundedJointConnector(
        lower, upper, lambda q: True, budget=ConnectorBudget(maximum_path_length_rad=1.5), seed=1
    )
    path, report = short.plan(start, goal)
    assert path is None and report["status"] == "PATH_LENGTH_LIMIT"
    limited = BoundedJointConnector(
        lower,
        upper,
        lambda q: np.linalg.norm(q) >= 0.55,
        budget=ConnectorBudget(maximum_validity_queries=5),
        seed=1,
    )
    path, report = limited.plan(start, goal)
    assert path is None and report["status"] == "NO_PLAN_FOUND_WITHIN_BUDGET"
    assert report["validity_queries"] == 5
