#!/usr/bin/env python3
"""Read-only local viewer for measured CF20 trees."""

from __future__ import annotations
import argparse
import json
import math
import mimetypes
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Viewer:
    def __init__(self, run, lerobot_root=None):
        self.run = Path(run).resolve()
        self.lerobot = (
            Path(lerobot_root).resolve() if lerobot_root else self.run / "lerobot_dataset"
        )

    def inside(self, relative):
        path = (self.run / relative).resolve()
        if not path.is_relative_to(self.run):
            raise ValueError("path escapes run root")
        return path

    def roots(self, task=None):
        rows = []
        for p in sorted((self.run / "data").glob("*/*/*/root_result.json")):
            try:
                result = json.loads(p.read_text())
                manifest_path = p.parent / "root_manifest.json"
                manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
                done_path = p.parent / "completed.json"
                done = json.loads(done_path.read_text()) if done_path.is_file() else {}
            except (OSError, json.JSONDecodeError):
                continue
            if task and result.get("task") != task:
                continue
            if not (p.parent / "tree.json").is_file():
                continue
            rows.append(
                {
                    "task": result.get("task"),
                    "split": result.get("split"),
                    "request_id": manifest.get("source_episode"),
                    "root_id": result.get("root_id"),
                    "status": done.get("status", result.get("status", "INCOMPLETE")),
                    "root_result": str(p.relative_to(self.run)),
                }
            )
        return rows

    def root(self, task, identity):
        rows = [r for r in self.roots(task) if identity in {r["root_id"], r["request_id"]}]
        if len(rows) != 1:
            raise KeyError("expected one root; found " + str(len(rows)))
        return rows[0], self.inside(rows[0]["root_result"]).parent

    def summary_edge(self, edge):
        out = dict(edge)
        if not edge.get("edge_path"):
            return out
        d = self.inside(edge["edge_path"])

        def read(n):
            p = d / n
            try:
                return json.loads(p.read_text()) if p.is_file() else {}
            except (OSError, json.JSONDecodeError):
                return {}

        option, local, fixed = (
            read("option.json"),
            read("local_result.json"),
            read("continuation/result.json"),
        )
        out["_viewer"] = {
            "method": option.get("source_method", "unknown"),
            "subtype": option.get("subtype"),
            "operation": option.get("operation_type"),
            "parent": option.get("parent_context_id"),
            "local_steps": local.get("control_steps"),
            "event": local.get("local_event_reached"),
            "stop": local.get("termination_reason"),
            "fixed": fixed.get("status", local.get("Y_fixed", {}).get("status")),
            "fixed_steps": fixed.get("actual_steps", fixed.get("control_steps")),
        }
        return out

    def tree(self, task, identity):
        row, rootdir = self.root(task, identity)
        t = json.loads((rootdir / "tree.json").read_text())
        t["edges"] = [self.summary_edge(e) for e in t.get("edges", [])]
        for key in ("child_pools", "third_pools"):
            for pool in t.get(key, []):
                pool["edges"] = [self.summary_edge(e) for e in pool.get("edges", [])]
        return {"root": row, "tree": t}

    def episode_rows(self, edge_id):
        index = self.lerobot / "indices" / "episode_tree_map.jsonl"
        if not index.is_file():
            return []
        rows = []
        with index.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("edge_id") == edge_id:
                    rows.append(
                        {
                            k: row.get(k)
                            for k in (
                                "dataset_id",
                                "shard_id",
                                "episode_index",
                                "segment_kind",
                                "dataset_start_index",
                                "frame_count",
                                "task",
                                "node_id",
                                "edge_id",
                            )
                        }
                    )
        return rows

    def edge(self, task, identity, edge_id):
        root, rootdir = self.root(task, identity)
        t = json.loads((rootdir / "tree.json").read_text())
        pools = [t.get("edges", [])] + [
            p.get("edges", []) for k in ("child_pools", "third_pools") for p in t.get(k, [])
        ]
        matches = [
            e
            for edges in pools
            for e in edges
            if edge_id in {e.get("candidate_id"), e.get("edge_path")}
        ]
        if len(matches) != 1:
            raise KeyError("expected one edge; found " + str(len(matches)))
        edge = matches[0]
        d = self.inside(edge["edge_path"])

        def read(n):
            p = d / n
            return json.loads(p.read_text()) if p.is_file() else None

        media = read("media.json") or {}
        video = media.get("video")
        return {
            "root": root,
            "edge": edge,
            "option": read("option.json"),
            "local_result": read("local_result.json"),
            "fixed_result": read("continuation/result.json"),
            "video": video,
            "media_summary": {
                k: media.get(k)
                for k in ("encoding_status", "fps", "resolution", "camera_observation_provenance")
            },
            "video_url": "/file?path=" + urllib.parse.quote(str(video), safe="/")
            if video
            else None,
            "local_url": "/file?path="
            + urllib.parse.quote(str(Path(edge["edge_path"]) / "execution.npz"), safe="/"),
            "fixed_url": "/file?path="
            + urllib.parse.quote(
                str(Path(edge["edge_path"]) / "continuation" / "execution.npz"), safe="/"
            ),
            "episode_map": self.episode_rows(str(edge.get("candidate_id", ""))),
        }

    def transition(self, task, identity, edge_id, segment, index):
        if segment not in {"local", "fixed_continuation"}:
            raise ValueError("invalid segment")
        data = self.edge(task, identity, edge_id)
        d = self.inside(data["edge"]["edge_path"])
        p = d / ("execution.npz" if segment == "local" else "continuation/execution.npz")
        with np.load(p, allow_pickle=False) as z:
            count = len(z["commanded_actions"])
            if not 0 <= index < count:
                raise IndexError("index outside [0,%d)" % count)
            names = (
                "sim_time",
                "robot_qpos",
                "robot_qvel",
                "actual_eef_pose",
                "object_poses",
                "object_velocities",
                "fixture_poses",
                "relevant_fixture_qpos",
                "relevant_fixture_qvel",
                "task_success",
            )

            def state(i):
                return {k: z[k][i].tolist() for k in names if k in z}

            return clean(
                {
                    "task": task,
                    "root_id": identity,
                    "edge_id": edge_id,
                    "segment": segment,
                    "transition_index": index,
                    "K": count,
                    "state_t": state(index),
                    "action_sent": z["commanded_actions"][index].tolist(),
                    "state_t_plus_1": state(index + 1),
                    "raw_planned_action": z["raw_planned_actions"][index].tolist()
                    if "raw_planned_actions" in z
                    else None,
                    "local_event_reached": bool(z["local_event_flags"][index])
                    if segment == "local" and "local_event_flags" in z
                    else None,
                }
            )

    def progress(self):
        manifest_path = self.run / "source_split_frozen.json"
        try:
            manifest = json.loads(manifest_path.read_text())
            queue = manifest.get("queue_order", [])
            task_by_id = manifest.get("requests_by_id", {})
            statuses = {}
            for path in (self.run / "data" / "formal").rglob("completed.json"):
                try:
                    marker = json.loads(path.read_text())
                    root_manifest = json.loads((path.parent / "root_manifest.json").read_text())
                    request_id = root_manifest.get("source_episode")
                    if request_id:
                        statuses[request_id] = marker.get("status", "PENDING")
                except (OSError, json.JSONDecodeError):
                    continue
            for path in (self.run / "runtime" / "job_results" / "formal").rglob("*.json"):
                try:
                    result = json.loads(path.read_text())
                    request_id = result.get("root_id", path.stem)
                    if request_id not in statuses:
                        statuses[request_id] = result.get("status", "PENDING")
                except (OSError, json.JSONDecodeError):
                    continue
            workers_path = self.run / "runtime" / "workers.json"
            active = (
                json.loads(workers_path.read_text()).get("active", [])
                if workers_path.is_file()
                else []
            )
            active_ids = {str(row.get("root_id")) for row in active}
            counts = {}
            by_task = {}
            for request_id in queue:
                status = statuses.get(request_id)
                if request_id in active_ids:
                    status = "ACTIVE"
                elif status is None:
                    status = "PENDING"
                counts[status] = counts.get(status, 0) + 1
                task = task_by_id.get(request_id, {}).get("task", "unknown")
                task_counts = by_task.setdefault(task, {})
                task_counts[status] = task_counts.get(status, 0) + 1
            return {
                "sampled_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "total_requests": len(queue),
                "effective_status_counts": counts,
                "per_task_status_counts": by_task,
                "live_active_worker_count": len(active),
                "requested_worker_cap": json.loads(workers_path.read_text()).get(
                    "requested_workers"
                )
                if workers_path.is_file()
                else None,
                "counting_basis": "frozen queue + completed markers + latest job results + workers.json",
            }
        except (OSError, json.JSONDecodeError):
            return {"sampled_at": None, "effective_status_counts": {}}


class Handler(BaseHTTPRequestHandler):
    viewer = None

    def log_message(self, fmt, *args):
        print(self.log_date_time_string(), fmt % args, flush=True)

    def json(self, value, status=200):
        body = json.dumps(clean(value), ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path in {"/", "/index.html"}:
                body = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif u.path == "/api/progress":
                self.json(self.viewer.progress())
            elif u.path == "/api/roots":
                self.json({"roots": self.viewer.roots(q.get("task", [None])[0])})
            elif u.path == "/api/tree":
                self.json(self.viewer.tree(q["task"][0], q["root_id"][0]))
            elif u.path == "/api/edge":
                self.json(self.viewer.edge(q["task"][0], q["root_id"][0], q["edge_id"][0]))
            elif u.path == "/api/transition":
                self.json(
                    self.viewer.transition(
                        q["task"][0],
                        q["root_id"][0],
                        q["edge_id"][0],
                        q.get("segment", ["local"])[0],
                        int(q.get("index", ["0"])[0]),
                    )
                )
            elif u.path == "/file":
                self.file(q["path"][0])
            else:
                self.json({"error": "not found"}, 404)
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as e:
            self.json({"error": str(e)}, 400)

    def file(self, relative):
        p = self.viewer.inside(urllib.parse.unquote(relative))
        if not p.is_file():
            self.json({"error": "file not found"}, 404)
            return
        size = p.stat().st_size
        start, end = 0, size - 1
        partial = False
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            try:
                a, b = header[6:].split("-", 1)
                start = int(a) if a else 0
                end = int(b) if b else size - 1
                if start < 0 or end < start or start >= size:
                    raise ValueError()
                end = min(end, size - 1)
                partial = True
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */" + str(size))
                self.end_headers()
                return
        length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        with p.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining:
                b = f.read(min(1024 * 1024, remaining))
                if not b:
                    break
                self.wfile.write(b)
                remaining -= len(b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--lerobot-root", type=Path)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    if (a.run / "release.json").is_file():
        from cfact.viewer.release import serve

        serve(a.run, a.host, a.port)
        return
    Handler.viewer = Viewer(a.run, a.lerobot_root)
    if not (Handler.viewer.run / "data").is_dir():
        p.error("--run must contain data/")
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print("CFACT viewer at http://%s:%d/ for %s" % (a.host, a.port, Handler.viewer.run), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
