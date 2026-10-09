import json

import pytest

from cfact.run import verify_manifest
from cfact.sources.manifest import digest_json, sha256_file


def test_resume_rejects_changed_config_and_manifest(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("seed: 7\n")
    record = {"run_id": "fixed", "immutable_files": {"config.yaml": sha256_file(config)}}
    record["manifest_sha256"] = digest_json(record)
    (tmp_path / "manifest.json").write_text(json.dumps(record))
    assert verify_manifest(tmp_path)["run_id"] == "fixed"
    config.write_text("seed: 8\n")
    with pytest.raises(ValueError, match="Immutable run input"):
        verify_manifest(tmp_path)
    record["run_id"] = "replacement"
    (tmp_path / "manifest.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Immutable run manifest"):
        verify_manifest(tmp_path)
