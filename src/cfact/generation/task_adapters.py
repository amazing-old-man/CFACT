"""Four reusable physical-effect adapters; topology comes from actual bindings."""
import json
import numpy as np
from cfact.generation.event_anchor import classify_stage

FLOORS={"slide":0.005,"hinge":0.05,"object":0.01,"activation":1}

def family_for_task(task):
    if task.name in {"OpenBlenderLid","CloseBlenderLid"}:return "free_body_lid"
    if task.primitive_family=="placement":return "held_object_placement"
    if task.primitive_family=="button":return "button_activation"
    return "articulated_contact"

def activation_state(env):
    """Read the official fixture value without invoking update_state/stepping."""
    base=env.env
    for name in ("microwave","coffee_machine"):
        fixture=getattr(base,name,None)
        if fixture is not None and callable(getattr(fixture,"get_state",None)):
            state=fixture.get_state()
            if "turned_on" in state:
                return {"status":"MEASURED", "fixture":name,
                        "turned_on":bool(state["turned_on"]),
                        "origin":"official_fixture.get_state.turned_on"}
    return {"status":"NOT_APPLICABLE_OR_UNAVAILABLE"}

class TaskEffectAdapter:
    def __init__(self, task, binding=None):
        self.task=task; self.family=family_for_task(task); self.binding=binding or {}
    def inspect_capability(self, arrays):
        if self.family=="free_body_lid":
            return False,"UNSUPPORTED_FREE_BODY_LID: no verified grasp/release compiler"
        if self.family=="articulated_contact":
            types=np.asarray(arrays.get("fixture_joint_types",[])).astype(str)
            q=np.asarray(arrays.get("relevant_fixture_qpos",[]))
            if len(types)!=1 or types[0] not in {"slide","hinge"} or q.ndim!=2 or q.shape[1]!=1:
                return False,"NO_UNAMBIGUOUS_ACTUAL_HINGE_OR_SLIDE_BINDING"
        if self.family=="button_activation":
            records=[json.loads(str(x)) for x in arrays.get("fixture_activation_json",[])]
            if len(records)!=len(arrays["sim_time"]) or not all(r.get("status")=="MEASURED" for r in records):
                return False,"ACTIVATION_STATE_UNMEASURED"
        return True,None
    def observe_effect(self, arrays, *, event_signal=None):
        supported,reason=self.inspect_capability(arrays)
        poses=np.asarray(arrays["object_poses"])[:,0]
        signals=[json.loads(str(x)) for x in arrays.get("subtask_signals_json",[])]
        contact=bool(signals and (signals[-1].get(event_signal,False) if event_signal else
                     any(v for k,v in signals[-1].items() if k.startswith("stage_contact"))))
        displacement=float(np.linalg.norm(poses[-1,:3,3]-poses[0,:3,3]))
        eef=np.asarray(arrays["actual_eef_pose"])
        relative=np.linalg.inv(eef)@poses
        actions=np.asarray(arrays["commanded_actions"])
        stable=False
        if len(relative)>=4:
            drift=np.linalg.norm(np.diff(relative[-4:,:3,3],axis=0),axis=1)
            rotations=np.linalg.norm(np.diff(relative[-4:,:3,:3],axis=0),axis=(1,2))
            stable=bool(np.max(drift)<=0.003 and np.max(rotations)<=0.03)
        held=bool(self.family=="held_object_placement" and contact and stable and
                  len(actions) and actions[-1,actions.shape[1]-6]>0)
        activation=None;value=poses[-1,:3,3].tolist();units="m_object_translation"
        progress=displacement>=FLOORS["object"]
        if self.family=="articulated_contact" and supported:
            q=np.asarray(arrays["relevant_fixture_qpos"])
            kind=str(arrays["fixture_joint_types"][0]);units="m" if kind=="slide" else "rad"
            value=q[-1].tolist();progress=bool(abs(q[-1,0]-q[0,0])>=FLOORS[kind])
        if self.family=="button_activation" and supported:
            activation=json.loads(str(arrays["fixture_activation_json"][-1]))["turned_on"]
            value=[int(activation)];units="boolean";progress=bool(activation)
        return {"schema":"cfact.observed_task_effect.v1", "family":self.family,
                "supported":supported,"capability_reason":reason,
                "sim_time_s":float(arrays["sim_time"][-1]),"value":value,"units":units,
                "object_displacement_m":displacement,"contact":contact,
                "held_verified":held,"holding_evidence":{"relative_transform_stable":stable,
                  "task_contact":contact,"closed_gripper":bool(len(actions) and actions[-1,actions.shape[1]-6]>0)},
                "activation":activation,"physical_progress_above_floor":progress,
                "official_success":bool(arrays["task_success"][-1])}
    def classify_stage(self, arrays, *, source_cursor, local_event=False, release=False, event_signal=None):
        return classify_stage(self.observe_effect(arrays,event_signal=event_signal),
                              source_cursor=source_cursor,local_event=local_event,release=release)
    def can_expand_child(self, anchor):
        return anchor.allowed_continuation is not None
    def task_effect_distance(self, a, b):
        if a["family"]!=b["family"] or a["units"]!=b["units"]:raise ValueError("incompatible_effect_units")
        floor=FLOORS["activation"] if a["units"]=="boolean" else FLOORS["slide"] if a["units"]=="m" else FLOORS["hinge"] if a["units"]=="rad" else FLOORS["object"]
        return float(np.linalg.norm(np.asarray(a["value"])-np.asarray(b["value"]))/floor)
