"""Post-measurement acquisition policy and budgeted, arbitrary-depth traversal."""
from dataclasses import dataclass

@dataclass(frozen=True)
class FrontierChild:
    option: object
    edge: dict
    anchor: object
    decision_depth: int

def select_effect_frontier(children, adapter, maximum, source_first=False):
    """Greedy farthest measured child; no family label or evaluated tail is read."""
    remaining=sorted(children,key=lambda c:c.edge["edge_path"]);chosen=[]
    while remaining and len(chosen)<maximum:
        references=[i for i,c in enumerate(remaining) if getattr(c.option,"source_method",None)=="Reference"]
        if source_first and references:
            index=references[0]
        elif not chosen:
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
                    parents_per_level, remaining_budget, minimum_pool_steps,
                    maximum_attempted_pools=None, source_first=False):
    """Expand saved local children breadth-first; preserve all observations/dispositions."""
    frontier=list(initial);pools=[];decisions=[];productive=0
    attempt_limit=maximum_pools if maximum_attempted_pools is None else maximum_attempted_pools
    while frontier and productive<maximum_pools and len(pools)<attempt_limit:
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
        requested=len(legal) if maximum_attempted_pools is not None else min(parents_per_level,maximum_pools-productive)
        picked=select_effect_frontier(legal,adapter,requested,source_first=source_first)
        used=set();level_productive=0
        for child in picked:
            if level_productive>=parents_per_level or productive>=maximum_pools or len(pools)>=attempt_limit:break
            used.add(child.edge["edge_path"])
            if remaining_budget()<minimum_pool_steps:
                decisions.append({"edge":child.edge["edge_path"],"decision":"NOT_EXPANDED","reason":"CONTROL_STEP_BUDGET","anchor":child.anchor.record()});continue
            pool,next_children=expand(child)
            pool["decision_depth"]=depth+1
            pools.append(pool);frontier.extend(next_children)
            good=bool(next_children) or sum(e.get("status")=="PHYSICS_COMPLETE" for e in pool.get("edges",[]))>=2
            if "edges" not in pool:good=True
            if good:productive+=1;level_productive+=1
            decisions.append({"edge":child.edge["edge_path"],"decision":"EXPANDED" if good else "COMPILATION_UNAVAILABLE",
                "anchor":child.anchor.record(),"node":pool["node_id"],"pool_status":pool.get("status")})
        for child in legal:
            if child.edge["edge_path"] not in used:
                decisions.append({"edge":child.edge["edge_path"],"decision":"NOT_EXPANDED","reason":"FRONTIER_POOL_BUDGET","anchor":child.anchor.record()})
    for child in frontier:
        decisions.append({"edge":child.edge["edge_path"],"decision":"NOT_EXPANDED","reason":"ROOT_POOL_BUDGET","anchor":child.anchor.record()})
    return pools,decisions
