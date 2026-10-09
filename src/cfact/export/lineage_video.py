"""Render saved replay history and shared lineage playlists without physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cfact.io.review_video import QuadReviewWriter
from cfact.io.serialization import atomic_json
from cfact.io.video import FFmpegRGBWriter


def _inside(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError(f"media_path_outside_root:{relative}")
    return path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _review_panel(images: np.ndarray, sim_time: float, phase: str) -> np.ndarray:
    """The established three-camera/fourth-text-panel layout, with no crop."""
    if images.ndim != 4 or images.shape[0] != 3 or images.shape[-1] != 3:
        raise ValueError("history_requires_three_rgb_cameras")
    if images.dtype != np.uint8:
        raise ValueError("history_rgb_requires_uint8")
    _, height, width, _ = images.shape
    panel = Image.new("RGB", (2 * width, 2 * height), "#10151b")
    for index, frame in enumerate(images):
        panel.paste(Image.fromarray(frame), ((index % 2) * width, (index // 2) * height))
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 23)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(panel)
    for offset, line in enumerate(("Source replay prefix", "Actual recorded physics",
                                   f"Phase: {phase}", f"Time: {sim_time:.3f} s")):
        draw.text((width + 18, height + 18 + offset * 39), line, font=font, fill="white")
    return np.asarray(panel.resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def render_replay_prefix(
    history_directory: Path, output: Path, *, fps: int = 20
) -> dict:
    """Render the latest actual source-prefix replay from a control ledger.

    Source-original video is never an input. A reset creates a distinct
    replay; only the last contiguous replay ending at the parent is selected.
    All raw history remains shared in its original ledger.
    """
    history_directory, output = Path(history_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if fps <= 0:
        raise ValueError("video_fps_must_be_positive")
    index = json.loads((history_directory / "index.json").read_text())
    if index.get("camera_names") != list(QuadReviewWriter.CAMERAS):
        raise ValueError("history_camera_order_mismatch")
    rows_path = _inside(history_directory, index.get("frames", "frames.jsonl"))
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    candidates = [row for row in rows if row["segment"] == "source_prefix"]
    if not candidates:
        raise ValueError("actual_source_prefix_history_missing")
    selected: list[dict] = []
    discarded = 0
    for row in candidates:
        if selected and (not row["has_action_from_previous"]
                         or float(row["sim_time_s"]) <= float(selected[-1]["sim_time_s"])):
            discarded += len(selected)
            selected = []
        selected.append(row)
    width, height = int(index["width"]), int(index["height"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".rendering.mp4")
    frame_times = []
    try:
        with FFmpegRGBWriter(temporary, width=width, height=height, fps=fps) as writer:
            for row in selected:
                with np.load(_inside(history_directory, row["frame"]), allow_pickle=False) as data:
                    images = np.asarray(data["rgb"])
                    sim_time = float(data["sim_time_s"])
                if images.shape != (3, height, width, 3):
                    raise ValueError("history_rgb_resolution_mismatch")
                if abs(sim_time - float(row["sim_time_s"])) > 1e-9:
                    raise ValueError("history_frame_time_mismatch")
                writer.append_data(_review_panel(images, sim_time, row["phase"]))
                frame_times.append(sim_time)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    record = {
        "schema": "cfact.replay-prefix-video.v1",
        "video": str(output), "sha256": _sha(output), "fps": fps,
        "resolution": [width, height], "cameras": list(QuadReviewWriter.CAMERAS),
        "camera_render_resolution": [width, height],
        "frame_count": len(selected), "frame_sim_time_s": frame_times,
        "control_steps": sum(bool(row["has_action_from_previous"]) for row in selected),
        "history_directory": str(history_directory),
        "history_frames": [row["frame"] for row in selected],
        "discarded_frames_from_earlier_replays": discarded,
        "initial_state_available": not bool(selected[0]["has_action_from_previous"]),
        "render_source": "actual_current_source_replay_history",
        "source_original_video_is_distinct": True,
        "layout": "three_true_views_plus_downsampled_review_panel",
    }
    atomic_json(output.with_suffix(output.suffix + ".json"), record)
    return record


def _video_info(path: Path) -> dict:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise RuntimeError("system ffprobe is required for lineage video")
    result = subprocess.run(
        [executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("lineage_video_requires_one_video_stream")
    row = streams[0]
    return {"width": int(row["width"]), "height": int(row["height"]),
            "fps": float(Fraction(row["avg_frame_rate"])),
            "frame_count": int(row["nb_frames"]) if row.get("nb_frames", "N/A") != "N/A" else None}


def _load_lineage(release: Path, lineage_id: str) -> dict:
    manifest = json.loads((release / "release.json").read_text())
    mapping = _inside(release, manifest["lineage_paths"])
    matches = [row for line in mapping.read_text().splitlines() if line.strip()
               if (row := json.loads(line))["lineage_id"] == lineage_id]
    if len(matches) != 1:
        raise ValueError(f"lineage_id_must_resolve_once:{lineage_id}")
    indexed = matches[0]
    playlist = json.loads(_inside(release, indexed["playlist"]).read_text())
    if playlist["lineage_id"] != lineage_id or playlist["edge_ids"] != indexed["edge_ids"]:
        raise ValueError("lineage_playlist_identity_mismatch")
    return playlist


def render_lineage(release: Path, lineage_id: str, output: Path) -> dict:
    """Materialize an exact playlist, sharing ancestor files until rendering."""
    release, output = Path(release).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("system ffmpeg is required for lineage video")
    lineage = _load_lineage(release, lineage_id)
    segments = lineage["segments"]
    if not segments:
        raise ValueError("lineage_playlist_has_no_media")
    inputs, filters, records = [], [], []
    shape = None
    previous_edge = -1
    total_frames = 0
    for index, segment in enumerate(segments):
        if "edge_id" in segment and segment["edge_id"] is not None:
            if segment["edge_id"] not in lineage["edge_ids"]:
                raise ValueError("lineage_segment_is_not_an_ancestor")
            position = lineage["edge_ids"].index(segment["edge_id"])
            if position < previous_edge:
                raise ValueError("lineage_segments_are_out_of_ancestor_order")
            if segment["segment"] == "fixed_continuation" and position != len(lineage["edge_ids"]) - 1:
                raise ValueError("only_selected_leaf_may_contribute_fixed_continuation")
            previous_edge = position
        elif segment["segment"] != "source_prefix" or index != 0:
            raise ValueError("non_edge_lineage_segment_must_be_initial_source_prefix")
        media = _inside(release, segment["video"])
        info = _video_info(media)
        current_shape = (info["width"], info["height"], info["fps"])
        if shape is not None and current_shape != shape:
            raise ValueError("lineage_video_resolution_or_fps_mismatch")
        shape = current_shape
        start = float(segment["start_s"]) * info["fps"]
        end = float(segment["end_s"]) * info["fps"]
        first, last = round(start), round(end)
        if (first < 0 or last <= first or abs(start - first) > 1e-6
                or abs(end - last) > 1e-6
                or (info["frame_count"] is not None and last > info["frame_count"])):
            raise ValueError("lineage_video_interval_is_not_a_valid_frame_range")
        inputs.extend(["-i", str(media)])
        filters.append(f"[{index}:v]trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS[v{index}]")
        records.append({**segment, "source_start_frame": first, "source_end_frame_exclusive": last,
                        "output_start_frame": total_frames, "output_end_frame_exclusive": total_frames + last - first,
                        "source_video_sha256": _sha(media)})
        total_frames += last - first
    labels = "".join(f"[v{index}]" for index in range(len(segments)))
    filters.append(f"{labels}concat=n={len(segments)}:v=1:a=0[out]")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".rendering.mp4")
    try:
        subprocess.run(
            [executable, "-hide_banner", "-loglevel", "error", "-y", *inputs,
             "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt",
             "yuv420p", "-threads", "2", "-movflags", "+faststart", str(temporary)],
            check=True, capture_output=True, text=True,
        )
        rendered = _video_info(temporary)
        if rendered["frame_count"] != total_frames:
            raise ValueError("rendered_lineage_frame_count_mismatch")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    record = {"schema": "cfact.lineage-video.v1", "lineage_id": lineage_id,
              "root_id": lineage["root_id"], "edge_ids": lineage["edge_ids"],
              "video": str(output), "sha256": _sha(output),
              "frame_count": total_frames, "fps": shape[2], "resolution": list(shape[:2]),
              "segments": records, "source_prefix_included": segments[0]["segment"] == "source_prefix",
              "shared_ancestor_media_copied": False}
    atomic_json(output.with_suffix(output.suffix + ".json"), record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render_lineage(args.release, args.lineage_id, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
