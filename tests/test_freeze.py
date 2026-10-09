import json
import tarfile
import pytest
from cfact.runtime.freeze import create, verify, archive, copy_source, verify_archive, verify_release, inventory
from cfact.collection.snapshot import implementation_digest, source_digest


def fixture_checkout(tmp_path):
    root = tmp_path / "checkout"
    (root / "src/cfact/collection").mkdir(parents=True)
    (root / "src/cfact/collection/worker.py").write_text("# recorded worker\n")
    (root / "README.md").write_text("# reproducible release\n")
    return root


def test_freeze_inventory_and_reproducible_archive(tmp_path):
    root = fixture_checkout(tmp_path)
    result = create(root, "cfact-test")
    assert result == verify(root)
    first, second = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    assert archive(root, first) == archive(root, second)
    with tarfile.open(first) as file:
        file.extractall(tmp_path / "unpacked")
    assert verify(tmp_path / "unpacked/cfact-test") == result
    with pytest.raises(FileExistsError):
        create(root, "cfact-test")


@pytest.mark.parametrize("change", ["edit", "add", "delete", "manifest"])
def test_freeze_rejects_changed_source_inventory(tmp_path, change):
    root = fixture_checkout(tmp_path)
    create(root, "cfact-test")
    if change == "edit":
        (root / "README.md").write_text("changed")
    elif change == "add":
        (root / "src/new_module.py").write_text("# unexpected")
    elif change == "delete":
        (root / "README.md").unlink()
    else:
        p = root / "FREEZE.json"
        record = json.loads(p.read_text())
        record["content_sha256"] = "bad"
        p.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        verify(root)


def test_imported_package_digest_matches_collection_snapshot():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert implementation_digest() == source_digest(root)


def test_release_source_copy_is_independent_and_excludes_previous_freeze(tmp_path):
    original = fixture_checkout(tmp_path)
    create(original, "previous-release")
    destination = tmp_path / "release/source"
    before = (original / "FREEZE.json").read_bytes()
    copy_source(original, destination)
    assert not (destination / "FREEZE.json").exists()
    source_file = original / "src/cfact/collection/worker.py"
    copied_file = destination / "src/cfact/collection/worker.py"
    assert source_file.stat().st_ino != copied_file.stat().st_ino
    assert not copied_file.is_symlink()
    copied_file.write_text("# isolated development copy\n")
    assert source_file.read_text() == "# recorded worker\n"
    assert (original / "FREEZE.json").read_bytes() == before
    with pytest.raises(FileExistsError):
        copy_source(original, destination)


def test_archive_verification_checks_exact_source_bytes_and_extra_members(tmp_path):
    root = fixture_checkout(tmp_path)
    create(root, "new-release")
    artifact = tmp_path / "source.tar.gz"
    archive(root, artifact)
    assert verify_archive(root, artifact)["status"] == "PASS"
    changed = tmp_path / "changed.tar.gz"
    with tarfile.open(artifact, "r:gz") as source, tarfile.open(changed, "w:gz") as target:
        for member in source.getmembers():
            target.addfile(member, source.extractfile(member))
        target.addfile(tarfile.TarInfo("new-release/unrecorded.py"))
    with pytest.raises(ValueError, match="contents differ"):
        verify_archive(root, changed)


def test_release_checks_artifact_inventory_and_independent_file_list(tmp_path):
    original = fixture_checkout(tmp_path)
    release = tmp_path / "release"
    source = release / "source"
    copy_source(original, source)
    frozen = create(source, "new-release")
    artifact = release / "artifacts/source.tar.gz"
    digest = archive(source, artifact)
    checksums = "".join(f"{value}  source/{name}\n" for name, value in sorted(inventory(source).items()))
    (release / "FILES.sha256").write_text(checksums)
    (release / "FREEZE.json").write_text(json.dumps({
        "schema": "cfact.code-release.v2", "release_id": "new-release",
        "code_content_sha256": frozen["content_sha256"], "archive_path": "artifacts/source.tar.gz",
        "artifact_sha256": {"artifacts/source.tar.gz": digest},
    }))
    assert verify_release(release)["status"] == "PASS"
    extra = release / "artifacts/unrecorded.whl"
    extra.write_bytes(b"unrecorded artifact")
    with pytest.raises(ValueError, match="artifact inventory"):
        verify_release(release)
    extra.unlink()
    (release / "FILES.sha256").write_text(checksums + "extra\n")
    with pytest.raises(ValueError, match="FILES.sha256"):
        verify_release(release)
