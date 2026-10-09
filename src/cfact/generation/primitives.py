"""Reusable operation-family legality and descriptor normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class OperationCandidate:
    operation_type: str
    parameters: Mapping[str, float | str]
    target_ids: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    legal: bool
    reason: str
    normalized_descriptor: tuple[float, ...]
    units: tuple[str, ...]


@dataclass(frozen=True)
class OperationPlugin:
    family: str
    parameter_names: tuple[str, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    units: tuple[str, ...]

    def validate(self, candidate: OperationCandidate) -> ValidationResult:
        if candidate.operation_type != self.family:
            return ValidationResult(False, "operation_family_mismatch", (), self.units)
        if not candidate.target_ids:
            return ValidationResult(False, "no_executable_target", (), self.units)
        values = []
        for name, low, high in zip(self.parameter_names, self.lower_bounds, self.upper_bounds):
            if name not in candidate.parameters:
                return ValidationResult(False, f"missing_parameter:{name}", (), self.units)
            value = float(candidate.parameters[name])
            if value < low or value > high:
                return ValidationResult(False, f"out_of_bounds:{name}", (), self.units)
            width = high - low
            values.append(0.0 if width == 0.0 else (value - low) / width)
        return ValidationResult(True, "accepted", tuple(values), self.units)


def default_operation_plugins() -> dict[str, OperationPlugin]:
    """Return family plugins; task-specific roles remain configuration data."""

    definitions: Sequence[OperationPlugin] = (
        OperationPlugin(
            "placement", ("x_m", "y_m", "z_m"), (-1.0, -1.0, -0.2), (1.0, 1.0, 1.0), ("m", "m", "m")
        ),
        OperationPlugin("linear_joint", ("joint_fraction",), (0.0,), (1.0,), ("fraction",)),
        OperationPlugin("rotary_joint", ("angle_rad",), (-6.2832,), (6.2832,), ("rad",)),
        OperationPlugin("button", ("press_depth_m",), (0.0,), (0.08,), ("m",)),
        OperationPlugin("lever", ("angle_rad",), (-3.1416,), (3.1416,), ("rad",)),
        OperationPlugin("knob", ("angle_rad",), (-6.2832,), (6.2832,), ("rad",)),
    )
    return {plugin.family: plugin for plugin in definitions}
