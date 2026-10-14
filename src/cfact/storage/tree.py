"""Publish measured acquisition records as a portable tree view, without re-execution."""

from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
from cfact.export.lerobot import _load_trace
from cfact.io.serialization import array_digest
from cfact.storage.layout import copy_file, inside, portable, stable_id, write_json
from cfact.sources.manifest import sha256_file
from cfact.storage.media import review_video
from cfact.storage.integrity import media_spool, replay_history_path


def read(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _publish_snapshot(context: dict, state: np.ndarray, source: Path,
                      output: Path, target: Path, run: Path) -> dict:
    """Require new/declared snapshots and bind their time/state before copying."""
    from cfact.execution.snapshot import load_snapshot, SNAPSHOT_SCHEMA
    from cfact.storage.schema import CHILD_SCHEMA

    context = dict(context)
    declared = context.get("snapshot")
    required = context.get("schema") == CHILD_SCHEMA or bool(declared)
    if declared:
        reference = declared.get("path") if isinstance(declared, dict) else declared
        if not isinstance(reference, str) or not reference:
            raise ValueError("Declared backend snapshot has no file reference")
        referenced = Path(reference)
        referenced = referenced.resolve() if referenced.is_absolute() else (source.parent / referenced).resolve()
        if referenced != source.resolve():
            raise ValueError("Declared backend snapshot identity differs from acquisition location")
    if not source.is_file():
        if required:
            raise ValueError(f"Declared actual child/parent backend snapshot is missing: {source}")
        context["snapshot"] = {"status": "LEGACY_UNAVAILABLE", "capabilities": "physics_state_only"}
        return context
    if source.is_symlink() or not source.resolve().is_relative_to(run.resolve()):
        raise ValueError("Backend snapshot must be an independent acquisition file")
    snapshot = load_snapshot(source)
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported declared backend snapshot schema")
    if not np.array_equal(np.asarray(snapshot.get("states")), state):
        raise ValueError("Backend snapshot differs from saved actual child/parent state")
    if not np.isclose(float(snapshot.get("sim_time_s", np.nan)), float(state[0]), atol=1e-9, rtol=0):
        raise ValueError("Backend snapshot time differs from saved actual child/parent")
    required_fields = {"robots", "fixtures", "rng", "environment", "wrapper", "sim_data", "model_sha256"}
    if not required_fields.issubset(snapshot) or not snapshot["model_sha256"]:
        raise ValueError("Declared backend snapshot is incomplete")
    digest = copy_file(source, output / target)
    context["snapshot"] = {
        **(declared if isinstance(declared, dict) else {}),
        "path": str(target), "schema": SNAPSHOT_SCHEMA, "sha256": digest,
        "capabilities": snapshot.get("capabilities", {}),
    }
    return context


def _validate_history_refs(context: dict, state: np.ndarray, run: Path,
                           expected_history: Path | None) -> None:
    refs = context.get("history_refs", [])
    if context.get("history_complete") and len(refs) < 3:
        raise ValueError("Complete parent history requires three temporal frames")
    previous = None
    terminal_state = None
    for row in refs:
        directory = Path(row["directory"])
        directory = directory.resolve() if directory.is_absolute() else (run / directory).resolve()
        if not directory.is_relative_to(run.resolve()):
            raise ValueError("History reference escapes acquisition run")
        if expected_history is None or directory != expected_history:
            raise ValueError("History reference does not belong to this root's published ledger")
        frame = inside(directory, row["frame"])
        with np.load(frame, allow_pickle=False) as values:
            current = float(values["sim_time_s"])
            terminal_state = np.asarray(values["sim_state"])
            if (not np.isclose(current, row["sim_time_s"], atol=1e-9, rtol=0)
                    or not np.isclose(current, terminal_state[0], atol=1e-9, rtol=0)):
                raise ValueError("History reference time differs from measured frame")
            if row.get("has_action_from_previous") and "action_from_previous" not in values:
                raise ValueError("History reference is missing its real intermediate action")
        if previous is not None and current <= previous:
            raise ValueError("Parent history must contain distinct increasing temporal frames")
        previous = current
    if refs and not np.array_equal(terminal_state, state):
        raise ValueError("History terminal frame differs from actual child/parent state")


def source_record(route: dict, task: str, output: Path, run: Path, request: str | None) -> dict:
    identity = route.get("source_identity") or {}
    matched = re.search(r"@sha256:([a-f0-9]{64}):[^:]+:(.+)$", route.get("source_id", ""))
    package_hash = identity.get("package_sha256") or (
        matched.group(1) if matched else route.get("prepared_sha256")
    )
    if not package_hash:
        raise ValueError("A source package or prepared HDF5 content hash is required")
    package = stable_id("pkg", {"sha256": package_hash})
    demo = (
        identity.get("original_episode_id")
        or route.get("original_episode_id")
        or (matched.group(2) if matched else route.get("prepared_group"))
    )
    if not isinstance(demo, str) or "/" in demo or demo in ("", ".", ".."):
        raise ValueError(f"Invalid source demo identity: {demo}")
    sid = stable_id("src", {"package": package_hash, "task": task, "demo": demo})
    base = Path("sources") / task / package / demo
    original = (
        read(run / "videos/original" / task / str(request) / "index.json", {}) if request else {}
    )
    videos = []
    for row in original.get("source_videos", []):
        path = inside(run, row["video_path"])
        if not path.exists():
            continue
        digest = sha256_file(path)
        if row.get("sha256") and digest != row["sha256"]:
            raise ValueError(f"Source video hash mismatch: {path}")
        target = Path("videos/source") / (digest + path.suffix)
        copy_file(path, output / target)
        videos.append({"path": str(target), "sha256": digest, "camera": row.get("camera")})
    record = {
        "source_id": sid,
        "task": task,
        "source_package": package,
        "demo_id": demo,
        "package_sha256": package_hash,
        "identity_basis": "original_package"
        if identity.get("package_sha256") or matched
        else "prepared_hdf5",
        "original_source_id": route.get("source_id"),
        "prepared_sha256": route.get("prepared_sha256"),
        "prepared_group": route.get("prepared_group"),
        "identity": portable({k: v for k, v in identity.items() if not k.endswith("_path")}),
        "manifest": str(base / "source_manifest.json"),
        "original_video": str(base / "original_video.json"),
    }
    write_json(
        output / base / "source_manifest.json",
        {
            k: v
            for k, v in record.items()
            if k not in {"original_source_id", "prepared_sha256", "prepared_group", "identity"}
        },
    )
    # A shared source may appear first as a donor. Only roots have original media indices.
    media_path = output / base / "original_video.json"
    media = {
        "status": "SOURCE_VIDEO_PRESENT" if videos else "SOURCE_VIDEO_MISSING",
        "videos": videos,
    }
    if not media_path.exists() or videos:
        from cfact.io.serialization import atomic_json

        atomic_json(media_path, media)
    return record


def publish_tree(root_path: Path, run: Path, output: Path, config: dict, scope: dict) -> dict:
    from cfact.storage.schema import validate_child_context
    result = read(root_path / "root_result.json")
    manifest = read(root_path / "root_manifest.json")
    task = result["task"]
    request = str(manifest["source_episode"])
    route = scope["requests_by_id"][request]
    source = source_record(route, task, output, run, request)
    # Donors are explicitly separate records, never inferred to be the root demo.
    donor_routes = manifest.get("donor_source_routes", [])
    donor_records = [source_record(d, task, output, run, None) for d in donor_routes]
    donor_by_original = {d["original_source_id"]: d for d in donor_records}
    context = read(root_path / "nodes/n_root/context.json", {})
    protocol = {
        key: config.get(key)
        for key in [
            "generator_revision",
            "run_seed",
            "candidate_pool",
            "tree_limits",
            "execution",
            "contact_guard",
            "motion_validation",
        ]
    }
    protocol["task"] = next((v for v in config["tasks"] if v["task"] == task), {})
    protocol["task"] = {k: v for k, v in protocol["task"].items() if k != "prepared_hdf5"}
    protocol["implementation_sha256"] = {
        p["module"]: p["sha256"] for p in manifest.get("original_module_provenance", [])
    }
    protocol["collection_source_sha256"] = read(run / "run_manifest.json", {}).get("source_sha256")
    protocol["donor_sources"] = [d["source_id"] for d in donor_records]
    protocol_id = stable_id("protocol", protocol)
    root_identity = {
        "source_id": source["source_id"],
        "source_package": source["source_package"],
        "demo_id": source["demo_id"],
        "source_action_prefix": context.get("source_action_prefix"),
        "sim_time_s": context.get("sim_time_s"),
        "parent_state_sha256": context.get("sim_state_sha256"),
        "protocol_id": protocol_id,
    }
    rid = stable_id("root", root_identity)
    base = (Path("trees") / task / rid if config.get("storage", {}).get("flat_trees")
            else Path("trees") / task / source["source_id"] / rid)
    media_root = Path("media" if config.get("storage", {}).get("flat_trees") else "videos")
    nodes, edges, terminals, segments = [], [], [], []
    contexts = {p.parent.name: read(p) for p in sorted(root_path.glob("nodes/*/context.json"))}
    expanded = {c["parent_edge"]: name for name, c in contexts.items() if c.get("parent_edge")}
    node_ids = {name: stable_id("node", {"root": rid, "decision_node": name}) for name in contexts}
    edge_paths = sorted(root_path.glob("nodes/*/edges/*"))
    edge_ids = {
        str(p.relative_to(run)): stable_id(
            "edge", {"root": rid, "parent": p.parent.parent.name, "candidate": p.name}
        )
        for p in edge_paths
    }
    path_map = {old: str(base / "edges" / eid) for old, eid in edge_ids.items()}
    path_map.update({str(p): path_map[str(p.relative_to(run))] for p in edge_paths})
    history = replay_history_path(run, context, manifest, result)
    history_rel = None
    prefix_media = None
    if history:
        history_rel = base / "history"
        path_map[str(history)] = str(history_rel)
        for path in sorted(history.iterdir()):
            if not path.is_file() or path.is_symlink():
                raise ValueError("History must contain independent regular files")
            target = output / history_rel / path.name
            if path.suffix == ".json":
                write_json(target, portable(read(path), path_map))
            elif path.suffix == ".jsonl":
                target.parent.mkdir(parents=True, exist_ok=True)
                rows = [portable(json.loads(line), path_map) for line in path.read_text().splitlines() if line]
                content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
                if target.exists() and target.read_text() != content:
                    raise ValueError("Resume history index differs")
                if not target.exists():
                    target.write_text(content)
            else:
                copy_file(path, target)
        from cfact.export.lineage_video import render_replay_prefix

        prefix_video = media_root / "replay_prefix" / rid / "review.mp4"
        prefix_record_path = (output / prefix_video).with_suffix(".mp4.json")
        if prefix_record_path.exists() and (output / prefix_video).exists():
            prefix_media = read(prefix_record_path)
            if sha256_file(output / prefix_video) != prefix_media["sha256"]:
                raise ValueError("Replay prefix video integrity mismatch")
        else:
            prefix_media = render_replay_prefix(
                output / history_rel, output / prefix_video, fps=config["video"]["fps"],
                canonical_frames=[row["frame"] for row in context.get("history_refs",[])]
                if config.get("method",{}).get("recursive_goals_v2") else None,
            )
        prefix_media.update({"video": str(prefix_video), "history_directory": str(history_rel)})
        # Convert renderer's local output receipt to release-relative paths.
        from cfact.io.serialization import atomic_json

        atomic_json(prefix_record_path, prefix_media)
    by_candidate = {
        (Path(e.get("edge_path", "")).parent.parent.name, e.get("candidate_id")): e
        for e in result.get("edges", [])
        if e.get("edge_path")
    }
    old_tree = read(root_path / "tree.json", {})
    for pool in [
        {"node_id": "n_root", "edges": old_tree.get("edges", [])},
        *old_tree.get("child_pools", []),
        *old_tree.get("third_pools", []),
    ]:
        for edge in pool.get("edges", []):
            by_candidate[(pool["node_id"], edge.get("candidate_id"))] = edge
    for name, ctx in contexts.items():
        nid = node_ids[name]
        old = root_path / "nodes" / name
        state = old / "context.npz"
        state_rel = base / "nodes" / nid / "state.npz"
        digest = copy_file(state, output / state_rel)
        with np.load(state, allow_pickle=False) as values:
            if array_digest(values["sim_state"]) != ctx["sim_state_sha256"]:
                raise ValueError(f"Parent state hash mismatch: {state}")
            ctx = validate_child_context(ctx, values["sim_state"])
            saved_state = values["sim_state"].copy()
            _validate_history_refs(ctx, saved_state, run, history)
        snapshot = old / "parent_snapshot.json"
        if ctx.get("parent_edge"):
            snapshot = run / ctx["parent_edge"] / "child_snapshot.json"
        ctx = _publish_snapshot(ctx, saved_state, snapshot, output,
                                base / "nodes" / nid / "backend_snapshot.json", run)
        node = {
            "node_id": nid,
            "state": str(state_rel),
            "state_file_sha256": digest,
            "parent_edge": edge_ids.get(ctx.get("parent_edge")),
            "decision_node": True,
            "context": str(base / "nodes" / nid / "context.json"),
        }
        write_json(
            output / node["context"],
            {**portable(ctx, path_map), **node, "legacy_node_id": name, "node_id": nid},
        )
        for filename in [
            "observation.npz",
            "cameras.json",
            "candidate_pool.json",
            "proposals.jsonl",
        ]:
            p = old / filename
            if p.exists():
                if p.suffix == ".json":
                    write_json(
                        output / base / "nodes" / nid / filename, portable(read(p), path_map)
                    )
                elif p.suffix == ".jsonl":
                    rows = [
                        portable(json.loads(line), path_map)
                        for line in p.read_text().splitlines()
                        if line
                    ]
                    target = output / base / "nodes" / nid / filename
                    target.write_text("".join(json.dumps(row) + "\n" for row in rows))
                else:
                    copy_file(p, output / base / "nodes" / nid / filename)
        nodes.append(node)
    for old in edge_paths:
        old_rel = str(old.relative_to(run))
        eid = edge_ids[old_rel]
        parent = node_ids[old.parent.parent.name]
        edge_base = base / "edges" / eid
        option = read(old / "option.json", {})
        summary = by_candidate.get((old.parent.parent.name, old.name), {})
        local_result = read(old / "local_result.json", {})
        fixed_result = read(old / "continuation/result.json", {})
        child = None
        if (old / "child_context.npz").exists():
            child = node_ids.get(expanded.get(old_rel)) or stable_id(
                "node", {"root": rid, "actual_child_of": eid}
            )
            state_rel = base / "nodes" / child / "state.npz"
            child_ctx = read(old / "child_context.json", {})
            with np.load(old / "child_context.npz", allow_pickle=False) as saved:
                terminal_time = None
                if (old / "execution.npz").is_file():
                    with np.load(old / "execution.npz", allow_pickle=False) as trace:
                        terminal_time = float(trace["sim_time"][-1])
                child_ctx = validate_child_context(
                    child_ctx, saved["sim_state"], terminal_time=terminal_time
                )
                _validate_history_refs(child_ctx, saved["sim_state"], run, history)
                snapshot = old / "child_snapshot.json"
                child_ctx = _publish_snapshot(
                    child_ctx, saved["sim_state"], snapshot, output,
                    base / "nodes" / child / "backend_snapshot.json", run,
                )
                expected = child_ctx.get("state_sha256") or local_result.get("child_state_sha256")
                if expected and array_digest(saved["sim_state"]) != expected:
                    raise ValueError(f"Actual child state hash mismatch: {old}")
                if (output / state_rel).exists():
                    with np.load(output / state_rel, allow_pickle=False) as parent_state:
                        if not np.array_equal(saved["sim_state"], parent_state["sim_state"]):
                            raise ValueError("Recursive parent differs from measured local child")
                else:
                    copy_file(old / "child_context.npz", output / state_rel)
            if not any(n["node_id"] == child for n in nodes):
                node = {
                    "node_id": child,
                    "state": str(state_rel),
                    "state_file_sha256": sha256_file(output / state_rel),
                    "parent_edge": eid,
                    "decision_node": False,
                    "context": str(base / "nodes" / child / "context.json"),
                }
                write_json(
                    output / node["context"],
                    {
                        **portable(child_ctx, path_map),
                        **node,
                        "origin": "measured_local_child_before_fixed_continuation",
                    },
                )
                nodes.append(node)
        donor = (
            donor_by_original.get(option.get("source_episode"))
            if option.get("source_method") == "G1"
            else None
        )
        if option.get("source_method") == "G1" and donor is None:
            raise ValueError("G1 edge donor is absent from the recorded donor population")
        edge = {
            "edge_id": eid,
            "parent_node": parent,
            "actual_child": child,
            "generator": option.get("source_method"),
            "operation_type": option.get("operation_type"),
            "root_source_id": source["source_id"],
            "donor_source_id": donor["source_id"] if donor else None,
            "donor_original_id": option.get("source_episode")
            if option.get("source_method") == "G1"
            else None,
            "status": summary.get("status", "TECHNICAL_UNKNOWN"),
            "path": str(edge_base),
            "option": str(edge_base / "option.json"),
            "result": str(edge_base / "result.json"),
            "media": str(edge_base / "media.json"),
            "acquisition_edge_key": stable_id("acquisition", old_rel),
        }
        write_json(output / edge["option"], portable(option, path_map))
        write_json(
            output / edge["result"],
            {
                "summary": portable(summary, path_map),
                "local": portable(local_result, path_map),
                "fixed_continuation": portable(fixed_result, path_map),
            },
        )
        for filename in [
            "plan.npz",
            "executable_plan.npz",
            "connection.json",
            "costs.json",
            "goal_realization.json",
        ]:
            p = old / filename
            if p.exists():
                if p.suffix == ".json":
                    write_json(output / edge_base / filename, portable(read(p), path_map))
                else:
                    copy_file(p, output / edge_base / filename)
        media = read(old / "media.json", {})
        if (old / "execution.npz").exists() and history is not None:
            from cfact.storage.ledger_media import clean_media_from_ledger
            media = clean_media_from_ledger(run,old,media,history,
                derived_root=output/".derived_camera_frames")
        from cfact.storage.media import validate_media

        if (old / "execution.npz").exists():
            validate_media(media)
        media_base = media_root / "edges" / eid
        clean = {}
        spool = Path(media["frame_spool"]) if media.get("ledger_reconstruction_receipt") else media_spool(run, media)
        review_record = review_video(run, output, media, spool, media_base / "review.mp4")
        review = review_record["review_video"]
        for camera, paths in media.get("clean_camera_frame_paths", {}).items():
            clean[camera] = []
            for index, path in enumerate(paths):
                src = inside(spool, path)
                if not src.is_file():
                    raise FileNotFoundError(f"Clean camera frame is unavailable: {src}")
                target = media_base / "clean_cameras" / camera / f"{index:06d}{src.suffix}"
                copy_file(src, output / target)
                clean[camera].append(str(target))
        new_media = {
            k: v
            for k, v in media.items()
            if k not in ["frame_spool", "video", "clean_camera_frame_paths"]
        }
        if media.get("ledger_reconstruction_receipt"):
            receipt=read(Path(media["ledger_reconstruction_receipt"]))
            for frame in receipt["frames"]:
                frame["ledger_frame"]=str(history_rel / Path(frame["ledger_frame"]).name)
            receipt_path=edge_base / "ledger_rgb_receipt.json"
            write_json(output / receipt_path,receipt)
            new_media["ledger_reconstruction_receipt"]=str(receipt_path)
        new_media.update(
            {
                **review_record,
                "clean_camera_frame_paths": clean,
                "source_encoding_status": media.get("encoding_status"),
                "encoding_status": "COMPLETE" if review else "MISSING",
            }
        )
        write_json(output / edge["media"], new_media)
        for kind, filename in [
            ("local", "execution.npz"),
            ("fixed_continuation", "continuation/execution.npz"),
        ]:
            if not (old / filename).exists():
                continue
            trace = _load_trace(old, kind)
            count = len(trace["commanded_actions"])
            if count and not review:
                raise ValueError(f"Nonempty measured {kind} segment is missing review media: {old}")
            if count and not np.allclose(
                np.diff(trace["sim_time"]), 1 / media["fps"], atol=1e-5, rtol=0
            ):
                raise ValueError(
                    "Action times do not match recorded camera fps; resampling is forbidden"
                )
            if kind == "local" and summary.get("status") == "PHYSICS_COMPLETE" and not child:
                raise ValueError("Completed physical edge has no saved actual child state")
            for field in ("commanded_actions", "robot_qpos", "robot_qvel", "actual_eef_pose"):
                if not np.isfinite(trace[field]).all():
                    raise ValueError(f"Nonfinite measured {field}: {old}")
            if kind == "local" and child:
                if not np.isclose(
                    trace["sim_time"][-1], child_ctx["sim_time_s"], atol=1e-8, rtol=0
                ):
                    raise ValueError("Local terminal time differs from actual child state")
            if kind == "fixed_continuation" and (old / "execution.npz").exists():
                local_trace = _load_trace(old, "local")
                for field in ("sim_time", "robot_qpos", "robot_qvel", "actual_eef_pose"):
                    if not np.allclose(local_trace[field][-1], trace[field][0], atol=1e-8, rtol=0):
                        raise ValueError(
                            f"Fixed continuation does not start at actual local child: {field}"
                        )
            trace_rel = edge_base / kind / "execution.npz"
            digest = copy_file(old / filename, output / trace_rel)
            segment_id = stable_id("segment", {"edge": eid, "kind": kind})
            frame_times = np.asarray(media.get("frame_sim_time_s", []))
            frame_indices = []
            for t in trace["sim_time"]:
                if not len(frame_times):
                    raise ValueError(f"Missing media timestamps for {eid}")
                idx = int(np.argmin(abs(frame_times - t)))
                if abs(float(frame_times[idx]) - float(t)) > 1e-5:
                    raise ValueError(f"Camera timestamps do not match measured states: {eid}")
                frame_indices.append(idx)
            segment = {
                "segment_id": segment_id,
                "task": task,
                "source_id": source["source_id"],
                "root_id": rid,
                "edge_id": eid,
                "parent_node": parent,
                "actual_child": child,
                "segment": kind,
                "edge_path": str(edge_base),
                "result_path": edge["result"],
                "actual_child_state": str(base / "nodes" / child / "state.npz") if child else None,
                "trace_path": str(trace_rel),
                "trace_sha256": digest,
                "action_count": count,
                "media": edge["media"],
                "review_video": review,
                "camera_frame_indices": frame_indices,
                "fps": media.get("fps", config["video"]["fps"]),
                "acquisition_edge_key": stable_id("acquisition", old_rel),
                "generator": edge["generator"],
                "operation_type": edge["operation_type"],
                "donor_source_id": edge["donor_source_id"],
            }
            segments.append(segment)
            terminal = {k: v for k, v in segment.items() if k != "camera_frame_indices"}
            terminal.update(
                {
                    "terminal_state_index": count,
                    "terminal_sim_time_s": float(trace["sim_time"][-1]),
                    "terminal_images": {c: paths[frame_indices[-1]] for c, paths in clean.items()},
                    "terminal_review_frame": frame_indices[-1],
                    "episode_index": None,
                    "compatible_schema": None,
                    "shard_id": None,
                    "shard_path": None,
                    "repo_id": None,
                    "zero_actions": count == 0,
                }
            )
            terminals.append(terminal)
        edges.append(edge)
    for node in nodes:
        write_json(
            output / base / "nodes" / node["node_id"] / "outgoing_edges.json",
            [e["edge_id"] for e in edges if e["parent_node"] == node["node_id"]],
        )
    tree = {
        "schema": "cfact.dual-view.v1",
        "task": task,
        "source_id": source["source_id"],
        "source_package": source["source_package"],
        "demo_id": source["demo_id"],
        "root_id": rid,
        "root_node": node_ids.get("n_root"),
        "status": result["status"],
        "root_identity": root_identity,
        "source_manifest": source["manifest"],
        "donor_manifests": [d["manifest"] for d in donor_records],
        "root_context": str(base / "root_context.json"),
        "nodes": nodes,
        "edges": edges,
        "tree_path": str(base / "tree.json"),
        "measured_history": str(history_rel / "index.json") if history_rel else None,
        "source_replay_prefix_media": prefix_media,
        "source_replay_prefix_status": "AVAILABLE" if prefix_media else "LEGACY_OR_UNRECORDED_MISSING",
    }
    write_json(
        output / tree["root_context"],
        {
            "identity": root_identity,
            "protocol": protocol,
            "root_source_route": portable(route),
            "donor_source_routes": portable(donor_routes),
            "root_node": node_ids.get("n_root"),
            "source_manifest": source["manifest"],
            "donor_manifests": tree["donor_manifests"],
            "collection_result": portable(result, path_map),
        },
    )
    write_json(output / tree["tree_path"], tree)
    # Playlists share media and contain only true ancestor locals plus the selected leaf tail.
    incoming = {e["actual_child"]: e for e in edges if e["actual_child"]}
    by_edge_segment = {(s["edge_id"], s["segment"]): s for s in segments}
    lineages = []
    for leaf in edges:
        if any(e["parent_node"] == leaf["actual_child"] for e in edges):
            continue
        chain, current, seen = [leaf], leaf, {leaf["edge_id"]}
        while current["parent_node"] in incoming:
            current = incoming[current["parent_node"]]
            if current["edge_id"] in seen:
                raise ValueError("Cycle in measured tree lineage")
            seen.add(current["edge_id"])
            chain.insert(0, current)
        parts = [
            by_edge_segment[(e["edge_id"], "local")]
            for e in chain
            if (e["edge_id"], "local") in by_edge_segment
        ]
        tail = by_edge_segment.get((leaf["edge_id"], "fixed_continuation"))
        if tail and tail["action_count"]:
            parts.append(tail)
        playlist = []
        if prefix_media and prefix_media["frame_count"] > 1:
            playlist.append({
                "segment": "source_prefix", "video": prefix_media["video"],
                "start_s": 0,
                "end_s": (prefix_media["frame_count"] - 1) / prefix_media["fps"],
                "boundary_policy": "omit_shared_terminal_parent_frame",
            })
        for s in parts:
            if s["action_count"] == 0 and s is not parts[-1]:
                continue
            indices = s["camera_frame_indices"]
            if s["review_video"]:
                playlist.append(
                    {
                        "edge_id": s["edge_id"],
                        "segment": s["segment"],
                        "video": s["review_video"],
                        "start_s": indices[0] / s["fps"],
                        "end_s": (indices[-1] + (1 if s is parts[-1] else 0)) / s["fps"],
                        "trace_path": s["trace_path"],
                        "boundary_policy": "shared_state_once",
                    }
                )
        lid = stable_id("lineage", {"root": rid, "edges": [e["edge_id"] for e in chain]})
        path = str(media_root / "lineages" / (lid + ".json"))
        line = {
            "lineage_id": lid,
            "root_id": rid,
            "edge_ids": [e["edge_id"] for e in chain],
            "playlist": path,
            "segments": playlist,
        }
        write_json(output / path, line)
        lineages.append(line)
    return {"tree": tree, "segments": segments, "terminals": terminals, "lineages": lineages}
