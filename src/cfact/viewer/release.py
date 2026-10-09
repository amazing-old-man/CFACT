"""Read-only browser for portable dual-view releases."""

from __future__ import annotations
import json
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from cfact.data.release import ReleaseDataset
from cfact.viewer.server import Handler as FileHandler


class ReleaseHandler(FileHandler):
    dataset: ReleaseDataset

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                body = (self.dataset.root / "viewer/index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/roots":
                self.json({"roots": self.dataset.roots()})
            elif url.path == "/api/tree":
                self.json(self.dataset.tree(query["root_id"][0]))
            elif url.path == "/api/edge":
                edge = self.dataset.edge(query["edge_id"][0])
                result = {
                    **edge,
                    "episodes": self.dataset.episodes(edge["edge_id"]),
                    "terminals": self.dataset.terminals(edge["edge_id"]),
                }
                for name in ["option", "result", "media"]:
                    result[name] = json.loads(self.dataset.inside(edge[name]).read_text())
                self.json(result)
            elif url.path == "/api/arrays":
                arrays = self.dataset.trace(query["edge_id"][0], query.get("segment", ["local"])[0])
                index = int(query.get("index", ["0"])[0])
                count = len(arrays["commanded_actions"])
                if not 0 <= index <= count:
                    raise ValueError("State index out of range")
                self.json(
                    {
                        "K": count,
                        "state_index": index,
                        "sim_time": float(arrays["sim_time"][index]),
                        "robot_qpos": arrays["robot_qpos"][index].tolist(),
                        "robot_qvel": arrays["robot_qvel"][index].tolist(),
                        "action": arrays["commanded_actions"][index].tolist()
                        if index < count
                        else None,
                        "fields": {
                            k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                            for k, v in arrays.items()
                        },
                    }
                )
            elif url.path == "/file":
                self.file(query["path"][0])
            else:
                self.json({"error": "not found"}, 404)
        except (KeyError, ValueError, OSError) as error:
            self.json({"error": str(error)}, 400)


def serve(run, host, port):
    dataset = ReleaseDataset(run)
    handler = type(
        "BoundReleaseHandler", (ReleaseHandler,), {"dataset": dataset, "viewer": dataset}
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"CFACT release viewer: http://{host}:{port}/", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
