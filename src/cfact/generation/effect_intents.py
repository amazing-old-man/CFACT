"""Outcome-blind effect requests; never measured labels."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class EffectIntent:
    task: str
    operation_family: str
    stage: str
    affected_entity: str
    target_effect_type: str
    proposed_magnitude: dict
    geometry_frame: str
    task_goal_binding: dict
    contact_constraints: dict
    feasibility_constraints: dict
    horizon_budget: int
    source_provenance: tuple
    def record(self):
        return {"schema":"cfact.effect_intent.v1", **asdict(self),
                "is_ground_truth":False}

def intent_for_option(option, task):
    context=option.operation_context
    delta=np.asarray(option.poses)[:, :3, 3]-context.current_eef_pose[:3, 3]
    return EffectIntent(task, option.operation_type, context.stage,
        context.object_roles["manipulated"],
        option.goal_binding.get("kind", "source_relative_motion"),
        {"maximum_requested_tool_translation_m":float(np.linalg.norm(delta,axis=1).max()),
         "candidate_edit":dict(option.candidate_edit),
         "local_goal":dict(option.goal_binding)},
        option.reference_frame, dict(context.task_goal),
        dict(option.entry_constraints), dict(option.precondition_checks),
        option.local_timeout, tuple(option.provenance))
