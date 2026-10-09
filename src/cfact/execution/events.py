"""Event observations used by the adaptive tree policy.

The detector deliberately consumes already defined physical predicates.  It
does not infer task labels from RGB and it keeps source boundaries, planned
command changes, and observed physical events as separate fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class EventObservation:
    event_name: str
    first_step: int
    confirmed_step: int
    minimum_duration_steps: int
    source_boundary_step: int | None
    planned_command_step: int | None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


class HysteresisEventDetector:
    """Confirm boolean events only after a shared minimum duration.

    A confirmed event is emitted once per false-to-true episode.  The event is
    re-armed only after a shared consecutive-false window.  This merges short
    contact dropouts without turning numerical jitter into artificial tree
    depth.
    """

    def __init__(
        self,
        minimum_duration_steps: int,
        rearm_after_false_steps: int | None = None,
    ):
        if minimum_duration_steps < 1:
            raise ValueError("minimum_duration_steps must be positive")
        self.minimum_duration_steps = int(minimum_duration_steps)
        if rearm_after_false_steps is None:
            rearm_after_false_steps = minimum_duration_steps
        if rearm_after_false_steps < 1:
            raise ValueError("rearm_after_false_steps must be positive")
        self.rearm_after_false_steps = int(rearm_after_false_steps)
        self._runs: dict[str, tuple[int, int]] = {}
        self._false_runs: dict[str, int] = {}
        self._emitted: set[str] = set()

    def update(
        self,
        step: int,
        predicates: Mapping[str, bool],
        *,
        source_boundaries: Mapping[str, int] | None = None,
        planned_command_steps: Mapping[str, int] | None = None,
    ) -> list[EventObservation]:
        observed: list[EventObservation] = []
        source_boundaries = source_boundaries or {}
        planned_command_steps = planned_command_steps or {}
        for name, value in predicates.items():
            if not value:
                self._runs.pop(name, None)
                false_count = self._false_runs.get(name, 0) + 1
                self._false_runs[name] = false_count
                if false_count >= self.rearm_after_false_steps:
                    self._emitted.discard(name)
                continue
            self._false_runs.pop(name, None)
            first, count = self._runs.get(name, (int(step), 0))
            count += 1
            self._runs[name] = (first, count)
            if count >= self.minimum_duration_steps and name not in self._emitted:
                self._emitted.add(name)
                observed.append(
                    EventObservation(
                        event_name=name,
                        first_step=first,
                        confirmed_step=int(step),
                        minimum_duration_steps=self.minimum_duration_steps,
                        source_boundary_step=source_boundaries.get(name),
                        planned_command_step=planned_command_steps.get(name),
                    )
                )
        return observed
