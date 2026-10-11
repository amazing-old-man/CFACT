"""One ledger for every real wrapper step, with shared on-disk history frames."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import numpy as np
from cfact.io.serialization import atomic_json
from cfact.io.review_video import QuadReviewWriter
from cfact.execution.observation import observe, array_sha


class ControlStepLedger:
    def __init__(self, env: Any, directory: Path, *, width: int, height: int):
        self.env = env
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "frames.jsonl").exists() or any(self.directory.glob("frame_*.npz")):
            raise FileExistsError("existing_control_history_requires_resume_before_replay")
        self.width, self.height = int(width), int(height)
        self.control_steps = 0
        self.restore_wall_seconds = 0.0
        self.phase = "source_prefix"
        self.segment = "source_prefix"
        self._fresh = True
        self._original = env.step
        self.rows: list[dict[str, Any]] = []
        env.step = self.step

    def begin(self, segment: str, phase: str):
        self.segment, self.phase, self._fresh = segment, phase, True

    def capture(self, action=None):
        observation = observe(self.env)
        sim = observation.sim
        sequence = len(self.rows)
        path = self.directory / f"frame_{sequence:07d}.npz"
        images = np.stack(
            [
                np.asarray(
                    observation.render(height=self.height, width=self.width, camera_name=camera)[::-1],
                    dtype=np.uint8,
                )
                for camera in QuadReviewWriter.CAMERAS
            ]
        )
        state = observation.state
        values = {"sim_state": state, "sim_time_s": np.asarray(float(sim.data.time)), "rgb": images}
        values.update(observation.archive_values())
        model = getattr(getattr(sim, "model", None), "_model", None)
        if model is not None:
            source = self.env.env.sim
            if not hasattr(source, "_cfact_observation_model_sha"):
                import hashlib
                xml = source.model.get_xml()
                sha = hashlib.sha256(xml.encode()).hexdigest()
                (self.directory / f"model_{sha}.xml").write_text(xml)
                source._cfact_observation_model_sha = sha
            observation.metadata["model_xml_sha256"] = source._cfact_observation_model_sha
            if not hasattr(source, "_cfact_observation_model_binary_sha"):
                # XML decimal serialization is not a lossless MjModel archive.
                import mujoco
                binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
                mujoco.mj_saveModel(model, buffer=binary)
                binary_sha = array_sha(binary)
                (self.directory / f"model_{binary_sha}.mjb").write_bytes(binary.tobytes())
                source._cfact_observation_model_binary_sha = binary_sha
            observation.metadata["model_binary_sha256"] = source._cfact_observation_model_binary_sha
        if action is not None:
            values["action_from_previous"] = np.asarray(action).copy()
        np.savez_compressed(path, **values)
        row = {
            **observation.metadata,
            "frame": path.name,
            "segment": self.segment,
            "phase": self.phase,
            "sim_time_s": float(sim.data.time),
            "control_step_total": self.control_steps,
            "has_action_from_previous": action is not None,
            "image_sha256": [array_sha(frame) for frame in images],
            "action_time_before_s": getattr(self, "_action_time_before", None) if action is not None else None,
            "action_time_after_s": float(sim.data.time) if action is not None else None,
        }
        self.rows.append(row)
        with (self.directory / "frames.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    def step(self, action):
        if self._fresh:
            self.capture()
            self._fresh = False
        before = float(self.env.env.sim.data.time)
        self._action_time_before = before
        try:
            result = self._original(action)
        except BaseException:
            if float(self.env.env.sim.data.time) > before:
                self.control_steps += 1
                self.capture(action)
            raise
        self.control_steps += 1
        self.capture(action)
        return result

    def references(self, *, segment: str | None = None, last: int | None = None):
        rows = [row for row in self.rows if segment is None or row["segment"] == segment]
        if last is not None:
            rows = rows[-last:]
        return [{**row, "directory": str(self.directory)} for row in rows]

    def finish(self):
        atomic_json(
            self.directory / "index.json",
            {
                "schema": "cfact.control-history.v1",
                "control_steps": self.control_steps,
                "restore_wall_seconds": self.restore_wall_seconds,
                "camera_names": list(QuadReviewWriter.CAMERAS),
                "width": self.width,
                "height": self.height,
                "frames": "frames.jsonl",
                "frame_count": len(self.rows),
                "history_origin": "actual_current_replay_and_branch_execution",
                "source_original_video_is_distinct": True,
            },
        )


def ledger(env):
    return getattr(env, "_cfact_step_ledger", None)
