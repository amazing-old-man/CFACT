from pathlib import Path
R=Path('/mnt/ssd1/yanghc/CFACT_Research/repo')
files={
'src/cfact/generation/event_anchor.py':'''"""Stages derived only from already executed local observations."""
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
''',
'src/cfact/generation/effect_intents.py':'''"""Outcome-blind effect requests; never measured labels."""
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
''',
'src/cfact/generation/task_adapters.py':'''"""Four reusable physical-effect adapters; topology comes from actual bindings."""
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
                  len(actions) and actions[-1,6]>0)
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
                  "task_contact":contact,"closed_gripper":bool(len(actions) and actions[-1,6]>0)},
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
''',
'src/cfact/generation/frontier.py':'''"""Post-measurement acquisition policy and budgeted, arbitrary-depth traversal."""
from dataclasses import dataclass

@dataclass(frozen=True)
class FrontierChild:
    option: object
    edge: dict
    anchor: object
    decision_depth: int

def select_effect_frontier(children, adapter, maximum):
    """Greedy farthest measured child; no family label or evaluated tail is read."""
    remaining=sorted(children,key=lambda c:c.edge["edge_path"]);chosen=[]
    while remaining and len(chosen)<maximum:
        if not chosen:
            index=max(range(len(remaining)),key=lambda i:(
                remaining[i].anchor.evidence["physical_progress_above_floor"],
                remaining[i].anchor.evidence["held_verified"],
                remaining[i].anchor.evidence["contact"]))
        else:
            index=max(range(len(remaining)),key=lambda i:min(
                adapter.task_effect_distance(remaining[i].anchor.evidence,c.anchor.evidence)
                + float(remaining[i].anchor.stage!=c.anchor.stage) for c in chosen))
        chosen.append(remaining.pop(index))
    return chosen

def expand_frontier(initial, *, adapter, expand, maximum_depth, maximum_pools,
                    parents_per_level, remaining_budget, minimum_pool_steps):
    """Expand saved local children breadth-first; preserve all observations/dispositions."""
    frontier=list(initial);pools=[];decisions=[]
    while frontier and len(pools)<maximum_pools:
        depth=frontier[0].decision_depth
        candidates=[c for c in frontier if c.decision_depth==depth]
        frontier=[c for c in frontier if c.decision_depth!=depth]
        legal=[]
        for child in candidates:
            reason=child.anchor.reason
            if depth>=maximum_depth:reason="DEPTH_BUDGET_EXHAUSTED"
            elif not adapter.can_expand_child(child.anchor):reason=reason or "NO_SUPPORTED_CONTINUATION"
            if reason:
                decisions.append({"edge":child.edge["edge_path"],"decision":"NOT_EXPANDED","reason":reason,"anchor":child.anchor.record()})
            else:legal.append(child)
        picked=select_effect_frontier(legal,adapter,min(parents_per_level,maximum_pools-len(pools)))
        picked_paths={c.edge["edge_path"] for c in picked}
        for child in legal:
            if child.edge["edge_path"] not in picked_paths:
                decisions.append({"edge":child.edge["edge_path"],"decision":"NOT_EXPANDED","reason":"FRONTIER_POOL_BUDGET","anchor":child.anchor.record()})
        for child in picked:
            if remaining_budget()<minimum_pool_steps:
                decisions.append({"edge":child.edge["edge_path"],"decision":"NOT_EXPANDED","reason":"CONTROL_STEP_BUDGET","anchor":child.anchor.record()});continue
            pool,next_children=expand(child)
            pool["decision_depth"]=depth+1
            pools.append(pool);frontier.extend(next_children)
            decisions.append({"edge":child.edge["edge_path"],"decision":"EXPANDED","anchor":child.anchor.record(),"node":pool["node_id"]})
    for child in frontier:
        decisions.append({"edge":child.edge["edge_path"],"decision":"NOT_EXPANDED","reason":"ROOT_POOL_BUDGET","anchor":child.anchor.record()})
    return pools,decisions
'''}
for path,value in files.items():(R/path).write_text(value)
# Observation is separate from native execution caches and controller semantics.
p=R/'src/cfact/execution/options.py';s=p.read_text();s=s.replace('    _capture_failed: bool = field(', '    fixture_activation_json: list[str] = field(default_factory=list)\n    _capture_failed: bool = field(')
s=s.replace('        self._capture_failed = False\n\n    def after_step', '        from cfact.generation.task_adapters import activation_state\n        self.fixture_activation_json.append(json.dumps(activation_state(env), sort_keys=True))\n        self._capture_failed = False\n\n    def after_step')
s=s.replace('            "robot_contacts_json": np.asarray(self.robot_contacts_json),','            "robot_contacts_json": np.asarray(self.robot_contacts_json),\n            "fixture_activation_json": np.asarray(self.fixture_activation_json),');p.write_text(s)
p=R/'src/cfact/collection/config.py';s=p.read_text().replace('    if depth > 3:', '    if depth > (8 if config.get("method", {}).get("task_effect_v2") else 3):').replace('maximum_decision_depth above 3 is not implemented','maximum_decision_depth exceeds implemented bounded policy');p.write_text(s)
p=R/'src/cfact/collection/worker.py';s=p.read_text();s=s.replace('    return {\n        "candidate_id": option.candidate_id,','    from cfact.generation.effect_intents import intent_for_option\n    return {\n        "effect_intent": intent_for_option(option, option.parent_context_id.split("__")[0]).record() if option.operation_context else None,\n        "candidate_id": option.candidate_id,',1)
needle='    source_cursor = dict(local.get("source_cursor", {}))\n'
s=s.replace(needle,needle+'''    from cfact.generation.task_adapters import TaskEffectAdapter
    task_name = edge_dir.parents[4].name.split("__")[0]
    # The declared task comes from the root directory, not the proposed goal.
    task_name = next(part.split("__")[0] for part in edge_dir.parts if "__" in part and part.split("__")[0] in {t.name for t in __import__("cfact.tasks", fromlist=["TARGET_TASKS"]).TARGET_TASKS})
    adapter = TaskEffectAdapter(target_task(task_name), option.operation_context.joint_binding)
    anchor = adapter.classify_stage(trace.arrays(), source_cursor=source_cursor["next_action_index"],
        local_event=local["local_event_reached"], release=option.subtype == "release_and_retreat")
    atomic_json(edge_dir / "observed_effect.json", anchor.evidence)
    atomic_json(edge_dir / "event_anchor.json", anchor.record())
''',1)
s=s.replace('        source_cursor=int(source_cursor["next_action_index"]),\n    )','        source_cursor=int(source_cursor["next_action_index"]),\n        stage=anchor.stage.value,\n    )',1)
# The caller selects continuation stages, never a future evaluated fixed tail.
s=s.replace('    source_transform: Any,\n) -> tuple[dict[str, Any], int, list[Any]]:', '    source_transform: Any,\n    stage: str | None = None,\n) -> tuple[dict[str, Any], int, list[Any]]:',1)
s=s.replace('"operation_stage": "placement_approach"\n            if contract.holding_object\n            else "articulation_followthrough",','"operation_stage": stage or ("placement_approach" if contract.holding_object else "articulation_followthrough"),',1)
s=s.replace('stage="placement_approach" if contract.holding_object else "articulation_followthrough",','stage=stage or ("placement_approach" if contract.holding_object else "articulation_followthrough"),',1)
# Release returns its compiled options to the common next-child classifier.
s=s.replace(') -> tuple[dict[str, Any], int]:\n    """Third semantic stage:',') -> tuple[dict[str, Any], int, list[Any]]:\n    """Recorded semantic stage:',1)
s=s.replace('return {"node_id": node_id, "status": reason, "edges": []}, charged_steps','return {"node_id": node_id, "status": reason, "edges": []}, charged_steps, []',1)
s=s.replace('    }, charged_steps\n\n\ndef _source_attempt_result','    }, charged_steps, selected\n\n\ndef _source_attempt_result',1)
s=s.replace('third_pool, charged_steps = _expand_release_pool(', 'third_pool, charged_steps, _ = _expand_release_pool(',1)
# Keep the old policy available for an explicit ablation / legacy configurations.
start=s.index('        child_pools = []\n',s.index('def run_root'))
end=s.index('        all_edges = (\n',start)
old=s[start:end]
new='''        if cfg.get("method", {}).get("task_effect_v2"):
            from cfact.generation.frontier import FrontierChild, expand_frontier
            from cfact.generation.event_anchor import EventAnchor, OperationStage
            from cfact.generation.task_adapters import TaskEffectAdapter
            adapter = TaskEffectAdapter(task, joint_binding)
            def measured_children(options, physical_edges, depth):
                by_id = {o.candidate_id:o for o in options}
                found=[]
                for edge in physical_edges:
                    if edge.get("status") != "PHYSICS_COMPLETE" or edge["candidate_id"] not in by_id:
                        continue
                    anchor_path=run / edge["edge_path"] / "event_anchor.json"
                    rec=json.loads(anchor_path.read_text())
                    anchor=EventAnchor(OperationStage(rec["stage"]),rec["sim_time_s"],rec["source_cursor"],rec["evidence"],rec["allowed_continuation"],rec.get("reason"))
                    found.append(FrontierChild(by_id[edge["candidate_id"]], edge, anchor, depth))
                return found
            def expand_actual_child(child):
                nonlocal charged_steps
                common=dict(args=args,cfg=cfg,env=env,interface=interface,root=root,
                    ancestor_option=child.option,ancestor_edge=run / child.edge["edge_path"],
                    root_dir=root_dir,root_key=root_key,object_ref=object_ref,
                    fixture_ref=fixture_ref,contract=contract,joint_spec=joint_spec,charged_steps=charged_steps)
                if child.anchor.allowed_continuation=="recorded_release_suffix":
                    pool,charged_steps,options=_expand_release_pool(**common)
                else:
                    pool,charged_steps,options=_expand_child_pool(**common,root_segment=segment,
                        donors=donors,source_transform=pose_utils.transform_source_data_segment_using_object_pose,
                        stage=child.anchor.stage.value)
                return pool,measured_children(options,pool["edges"],child.decision_depth+1)
            limit=cfg["tree_limits"]
            pool_cap=min(limit["candidate_pools_per_root"]-1,
                int(cfg.get("engineering_profile",{}).get("max_decision_parents_per_root",100000))-1)
            child_pools, frontier_decisions = expand_frontier(measured_children(selected,edges,1),
                adapter=adapter,expand=expand_actual_child,
                maximum_depth=limit["maximum_decision_depth"],maximum_pools=max(0,pool_cap),
                parents_per_level=min(limit["second_depth_pools"],
                    int(cfg.get("engineering_profile",{}).get("max_children_per_parent",100000))),
                remaining_budget=lambda:cfg["execution"]["max_env_steps_per_root"]-ledger(env).control_steps,
                minimum_pool_steps=cfg["execution"]["local_timeout_steps"]+cfg["execution"]["continuation_steps"])
            third_pools=[]
            frontier_priority=[];child_frontier=[];third_frontier=[]
            atomic_json(root_dir / "frontier_decisions.json", {"policy":"measured_effect_diversity", "decisions":frontier_decisions, "fixed_tail_outcomes_used":False})
        else:
'''
s=s[:start]+new+''.join('    '+line if line.strip() else line for line in old.splitlines(True))+s[end:]
s=s.replace('            "frontier_selection": {\n                "family_priority": frontier_priority,','            "frontier_selection": {\n                "policy": "measured_effect_diversity" if cfg.get("method", {}).get("task_effect_v2") else "fixed_family_priority",\n                "decisions": frontier_decisions if cfg.get("method", {}).get("task_effect_v2") else [],\n                "family_priority": frontier_priority,',1)
s=s.replace('            "observed_max_decision_depth": (\n                3','            "observed_max_decision_depth": max([1] + [p.get("decision_depth", 2) for p in child_pools]) if cfg.get("method", {}).get("task_effect_v2") else (\n                3',1)
p.write_text(s)
# Generalize display lineage selection without changing source local segments.
p=R/'src/cfact/collection/coordinator.py';s=p.read_text();start=s.index('        children = tree.get("child_pools", [])',s.index('def _export_lineages'));end=s.index('        clips = []',start)
s=s[:start]+'''        pools=tree.get("child_pools", [])+tree.get("third_pools", [])
        by_parent={p.get("parent_edge"):p for p in pools}
        def longest(edge, seen):
            path=edge["edge_path"]
            if path in seen: raise ValueError("lineage_cycle")
            children=[e for e in by_parent.get(path,{}).get("edges",[]) if e.get("edge_path") and e.get("status")=="PHYSICS_COMPLETE"]
            tails=[longest(e,seen|{path}) for e in children]
            return [edge]+max(tails,key=len,default=[])
        selected=max([longest(e,set()) for e in root_edges if e.get("status")=="PHYSICS_COMPLETE"],key=len,default=[])
'''+s[end:];p.write_text(s)
