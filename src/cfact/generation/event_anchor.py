"""Stages derived only from already executed local observations."""
from dataclasses import dataclass, asdict
from enum import Enum

class OperationStage(str, Enum):
    APPROACH = "APPROACH"
    CONTACT_ESTABLISHED = "CONTACT_ESTABLISHED"
    PARTIAL_TASK_EFFECT = "PARTIAL_TASK_EFFECT"
    HELD_TRANSPORT = "HELD_TRANSPORT"
    PLACE_READY = "PLACE_READY"
    RELEASE_AND_RETREAT = "RELEASE_AND_RETREAT"
    RECOVERABLE_MISS = "RECOVERABLE_MISS"
    TERMINAL_SUCCESS = "TERMINAL_SUCCESS"
    TERMINAL_UNRECOVERABLE = "TERMINAL_UNRECOVERABLE"
    UNSUPPORTED = "UNSUPPORTED"

@dataclass(frozen=True)
class EventAnchor:
    stage: OperationStage
    sim_time_s: float
    source_cursor: int
    evidence: dict
    allowed_continuation: str | None
    reason: str | None = None

    def record(self):
        return {**asdict(self), "stage": self.stage.value,
                "label_origin": "measured_local_child_before_fixed_tail"}

def classify_stage(effect, *, source_cursor, local_event=False, release=False,
                   verified_recovery=False):
    """A horizon stop alone is neither a failure nor a recoverable miss."""
    S=OperationStage
    stage=S.APPROACH; continuation="bounded_current_geometry_source_reanchor"
    reason=None
    if effect["official_success"]:
        stage=S.TERMINAL_SUCCESS; continuation=None
    elif not effect["supported"]:
        stage=S.UNSUPPORTED; continuation=None; reason=effect["capability_reason"]
    elif release:
        stage=S.RELEASE_AND_RETREAT; continuation=None
        reason="release_endpoint_requires_new_verified_stage"
    elif effect["family"]=="held_object_placement":
        if effect["held_verified"]:
            stage=S.PLACE_READY if local_event else S.HELD_TRANSPORT
            continuation="recorded_release_suffix" if local_event else "bounded_current_geometry_source_reanchor"
        elif verified_recovery:
            stage=S.RECOVERABLE_MISS; continuation="verified_geometric_reapproach"
        else:
            stage=S.TERMINAL_UNRECOVERABLE; continuation=None
            reason="NO_VERIFIED_HELD_OR_RECOVERY_CONTINUATION"
    elif effect["activation"] is True:
        stage=S.PARTIAL_TASK_EFFECT
    elif effect["physical_progress_above_floor"]:
        stage=S.PARTIAL_TASK_EFFECT
    elif effect["contact"]:
        stage=S.CONTACT_ESTABLISHED
    return EventAnchor(stage, effect["sim_time_s"], int(source_cursor), effect,
                       continuation, reason)
