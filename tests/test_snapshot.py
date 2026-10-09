import io
import tarfile
import pytest
from cfact.collection.snapshot import source_digest, archive_source, unpack_source


def test_source_snapshot_includes_nested_code_and_resources(tmp_path):
    root = tmp_path / "checkout"
    for name, value in {
        "src/cfact/collection/worker.py": "# worker",
        "src/cfact/generation/options.py": "# options",
        "src/cfact/resources/tasks.json": "{}",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    config = tmp_path / "config.json"
    config.write_text("{}")
    archive = tmp_path / "source.tar.gz"
    archive_source(root, archive, config, config)
    runtime = tmp_path / "runtime"
    unpack_source(archive, runtime)
    assert source_digest(root) == source_digest(runtime)
    (runtime / "src/cfact/generation/options.py").write_text("# changed")
    assert source_digest(root) != source_digest(runtime)


@pytest.mark.parametrize("kind", ["traversal", "symlink"])
def test_snapshot_rejects_paths_and_links_outside_archive(tmp_path, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        item = tarfile.TarInfo("../../outside" if kind == "traversal" else "link")
        if kind == "symlink":
            item.type = tarfile.SYMTYPE
            item.linkname = "/outside"
        out.addfile(item, io.BytesIO())
    with pytest.raises(ValueError):
        unpack_source(archive, tmp_path / "destination")
