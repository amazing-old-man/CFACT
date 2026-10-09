"""Synchronous three-camera review video; model observations remain separate."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cfact.io.video import FFmpegRGBWriter


class QuadReviewWriter:
    CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")

    def __init__(
        self,
        path: Path,
        *,
        task: str,
        root: str,
        node: str,
        candidate: str,
        method: str,
        intent: str,
        fps: int = 20,
        cell_width: int = 640,
        cell_height: int = 360,
    ) -> None:
        self.path = path
        self.deferred = os.environ.get("CFACT_DEFER_VIDEO_ENCODING") == "1"
        self.frames_dir = path.with_suffix(".frames") if self.deferred else None
        if self.frames_dir is not None:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            if any(self.frames_dir.iterdir()):
                raise FileExistsError(f"nonempty video frame spool: {self.frames_dir}")
            self.writer = None
        else:
            self.writer = FFmpegRGBWriter(path, width=cell_width, height=cell_height, fps=fps)
        self.task, self.root, self.node = task, root, node
        self.candidate, self.method, self.intent = candidate, method, intent
        self.cell_width, self.cell_height, self.fps = cell_width, cell_height, fps
        self.times: list[float] = []
        self.phases: list[str] = []
        self.calibration: dict[str, Any] = {}
        self.camera_extrinsics_per_frame: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self.CAMERAS
        }
        self.clean_camera_frame_paths: dict[str, list[str]] = {name: [] for name in self.CAMERAS}
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        try:
            self.font = ImageFont.truetype(font_path, 23)
        except OSError:
            self.font = ImageFont.load_default()

    def append(self, env: Any, sim_time: float, phase: str) -> None:
        sim = env.env.sim
        initial_time = float(sim.data.time)
        if abs(initial_time - sim_time) > 1e-8:
            raise ValueError("camera frame and recorded physics time differ")
        panel = Image.new("RGB", (2 * self.cell_width, 2 * self.cell_height), "#10151b")
        for index, camera in enumerate(self.CAMERAS):
            try:
                frame = np.asarray(
                    sim.render(height=self.cell_height, width=self.cell_width, camera_name=camera)[
                        ::-1
                    ]
                )
            except Exception as error:
                raise RuntimeError(f"true camera {camera} unavailable: {error}") from error
            if float(sim.data.time) != initial_time:
                raise RuntimeError("render advanced simulation clock")
            x, y = (index % 2) * self.cell_width, (index // 2) * self.cell_height
            camera_image = Image.fromarray(frame.astype(np.uint8))
            panel.paste(camera_image, (x, y))
            if self.frames_dir is not None:
                clean_dir = self.frames_dir / "clean_cameras" / camera
                clean_dir.mkdir(parents=True, exist_ok=True)
                compact_size = (max(1, self.cell_width // 2), max(1, self.cell_height // 2))
                compact_image = camera_image.resize(compact_size, Image.Resampling.BILINEAR)
                clean_path = clean_dir / f"{len(self.times):06d}.jpg"
                compact_image.save(clean_path, "JPEG", quality=80, optimize=False)
                self.clean_camera_frame_paths[camera].append(
                    str(clean_path.relative_to(self.frames_dir))
                )
            if camera not in self.calibration:
                cam_id = sim.model.camera_name2id(camera)
                self.calibration[camera] = {
                    "camera_id": int(cam_id),
                    "world_position_m": np.asarray(sim.data.cam_xpos[cam_id]).tolist(),
                    "world_rotation_row_major": np.asarray(sim.data.cam_xmat[cam_id])
                    .reshape(3, 3)
                    .tolist(),
                    "fovy_deg": float(sim.model.cam_fovy[cam_id]),
                    "moving_camera": camera == "robot0_eye_in_hand",
                }
            cam_id = sim.model.camera_name2id(camera)
            self.camera_extrinsics_per_frame[camera].append(
                {
                    "world_position_m": np.asarray(sim.data.cam_xpos[cam_id]).tolist(),
                    "world_rotation_row_major": np.asarray(sim.data.cam_xmat[cam_id])
                    .reshape(3, 3)
                    .tolist(),
                }
            )
        draw = ImageDraw.Draw(panel)
        x, y = self.cell_width + 18, self.cell_height + 18
        for line in (
            f"任务：{self.task}",
            f"源：{self.root}  节点：{self.node}",
            f"候选：{self.candidate}",
            f"来源：{self.method}",
            f"意图：{self.intent}",
            f"阶段：{phase}",
            f"物理时间：{sim_time:.3f} s",
        ):
            draw.text((x, y), line, font=self.font, fill="white")
            y += 39
        review_panel = panel.resize((self.cell_width, self.cell_height), Image.Resampling.BILINEAR)
        if self.frames_dir is not None:
            review_panel.save(
                self.frames_dir / f"{len(self.times):06d}.jpg", "JPEG", quality=70, optimize=False
            )
        else:
            self.writer.append_data(np.asarray(review_panel, dtype=np.uint8))
        self.times.append(sim_time)
        self.phases.append(phase)

    def close(self) -> dict[str, Any]:
        if self.writer is not None:
            self.writer.close()
        return {
            "video": str(self.path),
            "cameras": list(self.CAMERAS),
            "calibration": self.calibration,
            "frame_sim_time_s": self.times,
            "camera_extrinsics_per_frame": self.camera_extrinsics_per_frame,
            "frame_phase": self.phases,
            "clean_camera_frame_paths": self.clean_camera_frame_paths,
            "clean_camera_frame_encoding": "jpeg_q80_bilinear_half_resolution",
            "camera_render_resolution": [self.cell_width, self.cell_height],
            "camera_feature_resolution": [
                max(1, self.cell_width // 2),
                max(1, self.cell_height // 2),
            ],
            "camera_observation_provenance": "native_simulator_camera_bilinear_downsampled_to_jpeg_q80",
            "render_source": "executed_physics_state",
            "fps": self.fps,
            "resolution": [self.cell_width, self.cell_height],
            "layout": "three_true_views_plus_downsampled_review_panel",
            "encoding_status": "PENDING" if self.deferred else "COMPLETE",
            "frame_spool": str(self.frames_dir) if self.frames_dir is not None else None,
        }


def encode_deferred_video(path: Path, fps: int) -> None:
    """Encode a completed frame spool without reading or changing physics state."""
    frames = path.with_suffix(".frames")
    if not frames.is_dir() or not list(frames.glob("*.jpg")):
        raise FileNotFoundError(f"missing completed frame spool: {frames}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".encoding.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames / "%06d.jpg"),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    os.replace(temporary, path)
