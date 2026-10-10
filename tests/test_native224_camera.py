import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from cfact.execution.history import ControlStepLedger
from cfact.io.review_video import QuadReviewWriter
from cfact.collection.config import validate_config
from importlib.resources import files


class Sim:
    def __init__(self):
        self.data = SimpleNamespace(
            time=1.0, cam_xpos=np.zeros((3, 3)), cam_xmat=np.tile(np.eye(3).reshape(1, 9), (3, 1))
        )
        self.model = SimpleNamespace(
            camera_name2id=lambda n: QuadReviewWriter.CAMERAS.index(n), cam_fovy=np.ones(3) * 45
        )
        self.calls = []

    def render(self, *, height, width, camera_name):
        self.calls.append((width, height, camera_name))
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[..., 0] = 240
        rgb[: height // 2, :, 1] = 100
        return rgb

    def get_state(self):
        return np.array([self.data.time])


def test_native_capture_history_and_clean_export_no_bgr_or_resize(tmp_path, monkeypatch):
    monkeypatch.setenv("CFACT_DEFER_VIDEO_ENCODING", "1")
    sim = Sim()
    env = SimpleNamespace(env=SimpleNamespace(sim=sim), step=lambda a: None)
    history = ControlStepLedger(env, tmp_path / "history", width=224, height=224)
    history.capture()
    history.finish()
    with np.load(tmp_path / "history/frame_0000000.npz") as npz:
        assert npz["rgb"].shape == (3, 224, 224, 3)
        assert (npz["rgb"][..., 0] == 240).all()
        assert (npz["rgb"][..., 2] == 0).all()
        assert (npz["rgb"][:, 0, :, 1] == 0).all()  # actual MuJoCo vertical flip
    writer = QuadReviewWriter(
        tmp_path / "video.mp4",
        task="any",
        root="root",
        node="p",
        candidate="i",
        method="method",
        intent="",
    )
    writer.append(env, 1.0, "local")
    meta = writer.close()
    assert meta["camera_render_resolution"] == meta["camera_feature_resolution"] == [224, 224]
    assert meta["feature_scale"] == 1 and meta["crop"] is None
    for camera in writer.CAMERAS:
        with Image.open(writer.frames_dir / meta["clean_camera_frame_paths"][camera][0]) as im:
            assert im.size == (224, 224)
            rgb = np.asarray(im)
            assert rgb[..., 0].mean() > 230 and rgb[..., 2].mean() < 10
    assert all(w == h == 224 for w, h, camera in sim.calls)
    assert sim.data.time == 1.0


def test_native224_schema_fails_wrong_dimensions():
    config = json.loads(files("cfact.resources").joinpath("collection.json").read_text())
    validate_config(config)
    config["video"]["cell_width"] = 398
    with pytest.raises(ValueError, match="NATIVE_RGB224"):
        validate_config(config)
