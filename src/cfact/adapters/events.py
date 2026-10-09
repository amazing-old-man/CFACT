"""Small simulator-facing event predicates for modern RoboCasa adapters."""

from __future__ import annotations

from typing import Any

import numpy as np


def microwave_button_signals(env: Any) -> dict[str, int]:
    """Separate start-button contact from the unchanged official success."""

    button_geom = f"{env.microwave.name}_start_button"
    return {
        "stage_contact_button": int(env.check_contact(env.robots[0].gripper["right"], button_geom)),
        "success": int(env._check_success()),
    }


def coffee_machine_button_geoms(env: Any) -> tuple[str, ...]:
    """Return all start-button geoms for the selected coffee machine."""

    prefix = env.coffee_machine.naming_prefix
    return tuple(sorted(f"{prefix}{name}" for name in env.coffee_machine._start_button_names))


def select_coffee_machine_button(env: Any, eef_position: np.ndarray) -> str:
    """Choose the contacted button, otherwise the nearest one, deterministically."""

    geoms = coffee_machine_button_geoms(env)
    if not geoms:
        raise ValueError("coffee machine exposes no start-button geom")
    gripper = env.robots[0].gripper["right"]
    contacted = [geom for geom in geoms if env.check_contact(gripper, geom)]
    if contacted:
        return contacted[0]
    position = np.asarray(eef_position, dtype=float)
    return min(
        geoms,
        key=lambda geom: (
            float(
                np.linalg.norm(
                    np.asarray(env.sim.data.geom_xpos[env.sim.model.geom_name2id(geom)]) - position
                )
            ),
            geom,
        ),
    )


def coffee_machine_button_signals(env: Any) -> dict[str, int]:
    """Separate any real coffee-button contact from official withdrawn success."""

    gripper = env.robots[0].gripper["right"]
    return {
        "stage_contact_button": int(
            any(env.check_contact(gripper, geom) for geom in coffee_machine_button_geoms(env))
        ),
        "success": int(env._check_success()),
    }
