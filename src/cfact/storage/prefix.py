"""Export one shared, physically recorded source-prefix segment per tree."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from cfact.io.serialization import atomic_json
from cfact.storage.layout import stable_id
from cfact.sources.manifest import sha256_file


def publish_prefix(output,published):
    tree=published["tree"];media=tree.get("source_replay_prefix_media")
    if not media or media["control_steps"]==0:return None,None
    directory=output/media["history_directory"]
    data=[]
    for name in media["history_frames"]:
        with np.load(directory/name,allow_pickle=False) as values:
            data.append({k:values[k].copy() for k in values.files})
    required={"robot_qpos","robot_qvel","actual_eef_pose","task_success","robot_qpos_joint_names"}
    if any(not required.issubset(d) for d in data):
        return None,{"status":"LEGACY_NATIVE_HISTORY_ONLY","reason":"prefix robot/eef fields not archived during acquisition"}
    if any("action_from_previous" not in d for d in data[1:]):raise ValueError("PREFIX_NOT_CONTIGUOUS_ACTION_HISTORY")
    state=np.stack([d["sim_state"] for d in data])
    root=next(n for n in tree["nodes"] if n["node_id"]==tree["root_node"])
    with np.load(output/root["state"],allow_pickle=False) as saved:
        if not np.array_equal(state[-1],saved["sim_state"]):raise ValueError("PREFIX_TERMINAL_IS_NOT_ROOT_PARENT")
    times=np.asarray([float(d["sim_time_s"]) for d in data]);fps=media["fps"]
    if not np.allclose(np.diff(times),1/fps,atol=1e-9,rtol=0):raise ValueError("PREFIX_CONTROL_CLOCK")
    base=Path(tree["tree_path"]).parent/"source_prefix";target=output/base;target.mkdir(parents=True,exist_ok=True)
    trace={key:np.stack([d[key] for d in data]) for key in ["robot_qpos","robot_qvel","actual_eef_pose","task_success"]}
    trace.update(sim_time=times,sim_state=state,commanded_actions=np.stack([d["action_from_previous"] for d in data[1:]]),
                 robot_qpos_joint_names=data[0]["robot_qpos_joint_names"])
    np.savez_compressed(target/"execution.npz",**trace)
    cameras=media["cameras"];paths={c:[] for c in cameras}
    for i,d in enumerate(data):
        if d["rgb"].shape!=(3,224,224,3):raise ValueError("PREFIX_NATIVE_RGB_SHAPE")
        for j,camera in enumerate(cameras):
            path=base/"rgb"/camera/f"{i:07d}.png";(output/path).parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(d["rgb"][j]).save(output/path)
            paths[camera].append(str(path))
    atomic_json(target/"media.json",dict(cameras=cameras,fps=fps,clean_camera_frame_paths=paths,frame_sim_time_s=times.tolist()))
    sid=stable_id("prefix",tree["root_id"])
    segment=dict(segment_id=sid,task=tree["task"],source_id=tree["source_id"],root_id=tree["root_id"],
        edge_id=sid,parent_node=None,actual_child=tree["root_node"],segment="source_prefix",
        edge_path=str(base),result_path=tree["root_context"],actual_child_state=root["state"],
        trace_path=str(base/"execution.npz"),trace_sha256=sha256_file(target/"execution.npz"),
        action_count=len(data)-1,media=str(base/"media.json"),review_video=media["video"],
        camera_frame_indices=list(range(len(data))),fps=fps,acquisition_edge_key=sid,
        generator="SourceReplay",operation_type=tree["task"],donor_source_id=None,
        record_kind="shared_source_replay_prefix_not_candidate_edge")
    terminal={k:v for k,v in segment.items() if k!="camera_frame_indices"}
    terminal.update(terminal_state_index=len(data)-1,terminal_sim_time_s=float(times[-1]),
        terminal_images={c:p[-1] for c,p in paths.items()},terminal_review_frame=len(data)-1,
        zero_actions=False,official_success=bool(trace["task_success"][-1]))
    tree["official_source_prefix"]=segment
    atomic_json(output/tree["tree_path"],tree)
    return segment,terminal
