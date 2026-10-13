"""Post-measurement acquisition policy and budgeted, arbitrary-depth traversal."""
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
        picked=sorted(legal,key=lambda c: ({"G2":0,"G1":1,"Reference":2,"G0":3}.get(c.option.source_method,4),c.edge["edge_path"]))[:min(parents_per_level,maximum_pools-len(pools))]
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
