"""Preserve recorded review media independently of clean training camera frames."""

from pathlib import Path
import json
import subprocess
from PIL import Image
import numpy as np
from cfact.io.video import FFmpegRGBWriter
from cfact.storage.layout import copy_file


def review_video(run: Path, output: Path, media: dict, spool: Path, target: Path) -> dict:
    from cfact.storage.layout import inside

    if media.get("video") and inside(run, media["video"]).is_file():
        copy_file(inside(run, media["video"]), output / target)
        return {"review_video": str(target), "review_status": "COPIED_RECORDED_VIDEO"}
    count = len(media.get("frame_sim_time_s", []))
    frames = [spool / f"{index:06d}.jpg" for index in range(count)]
    if not frames or not all(path.is_file() for path in frames):
        return {"review_video": None, "review_status": "REVIEW_MEDIA_MISSING"}
    destination = output / target
    if not destination.exists():
        temporary = destination.with_suffix(".partial.mp4")
        with Image.open(frames[0]) as image:
            width, height = image.size
        with FFmpegRGBWriter(
            temporary, width=width, height=height, fps=int(media["fps"])
        ) as writer:
            for path in frames:
                with Image.open(path) as image:
                    writer.append_data(np.asarray(image.convert("RGB")))
        # A complete movie is committed atomically; interrupted encoding never masquerades as final.
    else:
        temporary = destination
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=nb_frames,width,height",
            "-of",
            "json",
            str(temporary),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    if not streams or int(streams[0].get("nb_frames", 0)) != count:
        raise ValueError("Review video frame count differs from the recorded time ledger")
    if temporary != destination:
        temporary.replace(destination)
    return {
        "review_video": str(target),
        "review_status": "ENCODED_RECORDED_REVIEW_FRAMES",
        "review_ffprobe": streams[0],
    }


def validate_media(media: dict) -> None:
    cameras = media.get("cameras", [])
    times = np.asarray(media.get("frame_sim_time_s", []), dtype=float)
    paths = media.get("clean_camera_frame_paths", {})
    fps = media.get("fps", 0)
    if not cameras or len(cameras) != len(set(cameras)) or set(paths) != set(cameras):
        raise ValueError("Camera names must match the clean camera frame ledger")
    if any(not isinstance(name, str) or "/" in name or name in {".", "..", ""} for name in cameras):
        raise ValueError("Invalid camera name")
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) < -1e-9)
    ):
        raise ValueError("Camera times must be finite and ordered")
    if len(media.get("frame_phase", [])) != len(times) or any(
        len(p) != len(times) for p in paths.values()
    ):
        raise ValueError("Camera time, phase and frame counts differ")
    if not np.isfinite(fps) or fps <= 0 or int(fps) != fps:
        raise ValueError("Official export requires a positive integral recorded fps")
