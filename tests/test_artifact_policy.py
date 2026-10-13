from pathlib import Path
from types import SimpleNamespace
import pytest
from cfact.storage.policy import artifact_root, contained_path, target_filesystem
from cfact.run import verify_manifest
from cfact.sources.manifest import digest_json


def test_artifact_root_rejects_relative_and_symlink_escape(tmp_path):
    allowed = tmp_path / "ssd1"
    allowed.mkdir()
    outside = tmp_path / "ssd2"
    outside.mkdir()
    (allowed / "redirect").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="ABSOLUTE"):
        artifact_root({"artifact_root": "relative"}, tmp_path)
    with pytest.raises(ValueError, match="OUTSIDE"):
        artifact_root({"artifact_root": str(allowed / "redirect"),
                       "allowed_write_root": str(allowed)}, tmp_path)
    assert contained_path(allowed / "new/nested", allowed) == allowed / "new/nested"


def test_probe_checks_actual_target_disk_and_inode_reserve(tmp_path, monkeypatch):
    seen = []
    def usage(path):
        seen.append(path)
        return SimpleNamespace(free=40 * 2**30)
    monkeypatch.setattr("cfact.storage.policy.shutil.disk_usage", usage)
    monkeypatch.setattr("cfact.storage.policy.os.statvfs", lambda _: SimpleNamespace(f_favail=4))
    target = tmp_path / "not_created/run"
    report = target_filesystem(target, {"minimum_host_free_gib": 30, "minimum_free_inodes": 5})
    assert seen == [tmp_path]
    assert report["failures"] == ["INODE_RESERVE"]
    assert report["write_target_realpath"] == str(target)
    report = target_filesystem(target, {"minimum_host_free_gib": 41, "minimum_free_inodes": 0})
    assert report["failures"] == ["DISK_RESERVE"]


def test_manifest_rejects_relocated_run_outside_frozen_artifact_root(tmp_path):
    run = tmp_path / "copied"
    run.mkdir()
    record = {"run_id": "fixed", "artifact_root": str(tmp_path / "original"), "immutable_files": {}}
    record["manifest_sha256"] = digest_json(record)
    (run / "manifest.json").write_text(__import__("json").dumps(record))
    with pytest.raises(ValueError, match="OUTSIDE"):
        verify_manifest(run)
