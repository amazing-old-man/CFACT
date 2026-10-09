"""Small system-FFmpeg writer used when ImageIO has no bundled backend."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import TracebackType

import numpy as np


class FFmpegRGBWriter:
    """Stream fixed-size RGB uint8 frames to a deterministic H.264 MP4."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("system ffmpeg executable is required for MP4 output")
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("video width, height, and fps must be positive")
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._process = subprocess.Popen(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self.frame_count = 0
        self._closed = False

    def append_data(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed video writer")
        array = np.asarray(frame)
        expected = (self.height, self.width, 3)
        if array.shape != expected:
            raise ValueError(f"video frame shape {array.shape} != {expected}")
        if array.dtype != np.uint8:
            raise ValueError(f"video frame dtype {array.dtype} != uint8")
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(array).tobytes())
        except BrokenPipeError as error:
            stderr = self._read_stderr()
            raise RuntimeError(f"ffmpeg stopped while writing video: {stderr}") from error
        self.frame_count += 1

    def _read_stderr(self) -> str:
        if self._process.stderr is None:
            return ""
        return self._process.stderr.read().decode(errors="replace").strip()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = self._read_stderr()
        returncode = self._process.wait()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg exited with code {returncode}: {stderr or 'no diagnostic'}")
        if self.frame_count == 0 or not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced no non-empty video")

    def __enter__(self) -> "FFmpegRGBWriter":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            self.close()
        except Exception:
            if exception_type is None:
                raise
        return False


def ffmpeg_available() -> bool:
    """Return whether the required system encoder is available on PATH."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        return False
    completed = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and "libx264" in completed.stdout
