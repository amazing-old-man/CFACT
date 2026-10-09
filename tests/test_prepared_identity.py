import hashlib
import json
import h5py
import numpy as np
import pytest

from cfact.sources.prepare import _copy_subset, prepared_identity, validate_prepared_identity


def test_explicit_subset_preserves_source_and_rejects_stale_identity(tmp_path):
    raw = tmp_path / "raw.hdf5"
    prepared = tmp_path / "prepared.hdf5"
    with h5py.File(raw, "w") as dataset:
        data = dataset.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "OpenDrawer", "env_kwargs": {"controller_configs": {"input_type": "delta"}}})
        for demo in ("demo_0", "demo_7", "demo_10"):
            data.create_group(demo).create_dataset("actions", data=np.ones((3, 12)))
    raw_bytes = raw.read_bytes()
    _copy_subset(raw, prepared, ["demo_7", "demo_10"])
    with pytest.raises(ValueError, match="prepared_identity_missing"):
        validate_prepared_identity(prepared)
    identity = prepared_identity(raw, ["demo_7", "demo_10"])
    semantics = json.dumps({"schema": "cfact.action_codec.v1", "input_ref_frame": "base"})
    identity["controller_semantics_sha256"] = {demo: hashlib.sha256(semantics.encode()).hexdigest() for demo in identity["episodes"]}
    with h5py.File(prepared, "a") as dataset:
        dataset.attrs["cfact_prepared_identity"] = json.dumps(identity)
        for demo in identity["episodes"]:
            dataset[f"data/{demo}"].create_group("datagen_info").attrs["cfact_controller_semantics"] = semantics
    assert validate_prepared_identity(prepared, raw=raw, required_episodes=["demo_10"])["episodes"] == ["demo_7", "demo_10"]
    assert raw.read_bytes() == raw_bytes
    with pytest.raises(ValueError, match="prepared_episode_identity_mismatch"):
        validate_prepared_identity(prepared, required_episodes=["demo_0"])
    with h5py.File(prepared, "a") as dataset:
        identity["preparer_code_sha256"] = "legacy"
        dataset.attrs["cfact_prepared_identity"] = json.dumps(identity)
    with pytest.raises(ValueError, match="prepared_code_identity_mismatch"):
        validate_prepared_identity(prepared)
