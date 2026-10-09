"""Small coordinate-frame helpers shared by modern interface adapters."""

from __future__ import annotations

import numpy as np


def rotate_planar_base_delta_to_world(values: np.ndarray, base_yaw: float) -> np.ndarray:
    """Rotate an ``(x, y)`` controller delta from robot-base to world axes."""

    result = np.array(values, dtype=float, copy=True)
    cosine = float(np.cos(base_yaw))
    sine = float(np.sin(base_yaw))
    base_x = float(result[0])
    base_y = float(result[1])
    result[0] = cosine * base_x - sine * base_y
    result[1] = sine * base_x + cosine * base_y
    return result
