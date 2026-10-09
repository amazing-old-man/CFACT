from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cfact.io.video import FFmpegRGBWriter, ffmpeg_available


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="system ffmpeg absent")
def test_system_ffmpeg_rgb_writer_creates_nonempty_mp4(tmp_path: Path) -> None:
    if not ffmpeg_available():
        pytest.skip("system ffmpeg lacks libx264")
    output = tmp_path / "smoke.mp4"
    with FFmpegRGBWriter(output, width=16, height=16, fps=5) as writer:
        writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))
        writer.append_data(np.full((16, 16, 3), 255, dtype=np.uint8))
    assert output.stat().st_size > 0
    probed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "default=noprint_wrappers=1",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "width=16" in probed.stdout
    assert "height=16" in probed.stdout


def test_system_ffmpeg_rgb_writer_rejects_wrong_frame_shape(
    tmp_path: Path,
) -> None:
    if not ffmpeg_available():
        pytest.skip("system ffmpeg lacks libx264")
    with FFmpegRGBWriter(tmp_path / "wrong.mp4", width=16, height=16, fps=5) as writer:
        with pytest.raises(ValueError, match="video frame shape"):
            writer.append_data(np.zeros((8, 8, 3), dtype=np.uint8))
        writer.append_data(np.zeros((16, 16, 3), dtype=np.uint8))
