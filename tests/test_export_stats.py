import numpy as np
import pytest
from cfact.export.lerobot import _matrix_safe_episode_stats


def vector_stats(data, features, quantile_list=None):
    # Reproduce the upstream vector accumulator's handling of leading axes.
    result = {}
    for key, value in data.items():
        vectors = value if len(value) == 1 else value.reshape(-1, value.shape[-1])
        result[key] = {
            "mean": vectors.mean(axis=0),
            "std": vectors.std(axis=0),
            "count": np.array([len(value)]),
        }
    return result


@pytest.mark.parametrize("length", [1, 3, 10])
def test_matrix_statistics_reduce_time_without_collapsing_pose_axes(length):
    pose = np.arange(length * 16, dtype=np.float32).reshape(length, 4, 4)
    before = pose.copy()
    spec = {"pose": {"dtype": "float32", "shape": (4, 4), "names": None}}
    result = _matrix_safe_episode_stats({"pose": pose}, spec, compute=vector_stats)["pose"]
    np.testing.assert_allclose(result["mean"], pose.mean(axis=0))
    np.testing.assert_allclose(result["std"], pose.std(axis=0))
    assert result["mean"].shape == (4, 4)
    assert result["count"].tolist() == [length]
    np.testing.assert_array_equal(pose, before)
    assert spec["pose"]["shape"] == (4, 4)


def test_mismatched_feature_shape_is_rejected():
    with pytest.raises(ValueError, match="recorded shape"):
        _matrix_safe_episode_stats(
            {"pose": np.zeros((2, 4, 4))},
            {"pose": {"dtype": "float32", "shape": (16,)}},
            compute=vector_stats,
        )


@pytest.mark.parametrize("length", [1, 11])
def test_short_episode_h264_has_no_reordered_frames(tmp_path, length):
    import json
    import shutil
    import subprocess
    from PIL import Image
    from cfact.export.lerobot import _encode_h264_frames

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires ffmpeg and ffprobe")
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(length):
        Image.new("RGB", (64, 48), (i * 10, 30, 70)).save(frames / f"frame-{i:06d}.png")
    video = tmp_path / "episode.mp4"
    _encode_h264_frames(frames, video, 20)
    result = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,dts",
            "-of",
            "json",
            str(video),
        ],
        text=True,
    )
    packets = json.loads(result)["packets"]
    assert len(packets) == length
    assert all(p["pts"] == p["dts"] for p in packets)
    assert all(b["dts"] > a["dts"] for a, b in zip(packets, packets[1:]))


def test_export_requires_a_new_output_and_allows_no_automatic_export_config(tmp_path):
    import argparse
    import json
    from cfact.export.lerobot import export

    run = tmp_path / "run"
    run.mkdir()
    (run / "config_frozen.json").write_text(json.dumps({"video": {"fps": 20}}))
    output = run / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing data")
    args = argparse.Namespace(
        run=run,
        output=output,
        repo_id="local/test",
        python_packages=None,
        cleanup_camera_spools=False,
    )
    with pytest.raises(FileExistsError):
        export(args)
    assert sentinel.read_text() == "existing data"
