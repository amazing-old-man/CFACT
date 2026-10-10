import io
import tarfile
import h5py
import numpy as np
import pandas as pd
import pytest
import json
import shutil
from pathlib import Path
from cfact.sources.archives import extract_tar
from cfact.sources.importer import import_source
from test_lerobot_input import _fixture


def test_real_parquet_directory_and_archive_import(tmp_path):
    source = _fixture(tmp_path / "original")
    pd.DataFrame(
        {
            "action": [np.arange(12), np.arange(12) + 1],
            "next.reward": [0.0, 1.0],
            "next.done": [False, True],
        }
    ).to_parquet(source / "data/chunk-000/episode_000000.parquet")
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source, arcname="lerobot")
    report = import_source(archive, tmp_path / "raw.hdf5", "OpenDrawer", 1)
    assert report["episode_count"] == 1
    assert report["source_archive_sha256"]
    assert len(report["source_files_sha256"]) == 8
    with h5py.File(report["output"]) as file:
        assert file["data"].attrs["total"] == 2  # actual transitions, not episodes
        assert file["data/demo_0/actions"][0].tolist() == [5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]
    with pytest.raises(FileExistsError):
        import_source(archive, tmp_path / "raw.hdf5", "OpenDrawer")
    with pytest.raises(ValueError):
        import_source(source, tmp_path / "negative.hdf5", "OpenDrawer", -1)


@pytest.mark.parametrize("name,link", [("../escape", False), ("/absolute", False), ("link", True)])
def test_archive_rejects_traversal_and_links(tmp_path, name, link):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as tar:
        item = tarfile.TarInfo(name)
        if link:
            item.type = tarfile.SYMTYPE
            item.linkname = "/tmp"
            tar.addfile(item)
        else:
            item.size = 1
            tar.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        extract_tar(path, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


def test_prepared_subset_copies_only_selected_demos_without_touching_raw(tmp_path):
    from cfact.sources.prepare import _copy_subset
    from cfact.sources.manifest import sha256_file

    raw = tmp_path / "raw.hdf5"
    with h5py.File(raw, "x") as file:
        file.attrs["origin"] = "source"
        data = file.create_group("data")
        data.attrs["total"] = 10
        data.attrs["cfact_source_provenance"] = "preserved"
        for name, count in [("demo_10", 5), ("demo_2", 3), ("demo_0", 2)]:
            demo = data.create_group(name)
            demo.attrs["original_episode_index"] = int(name.split("_")[-1])
            demo.create_dataset("actions", data=np.zeros((count, 12)))
    before = sha256_file(raw)
    output = tmp_path / "subset.hdf5"
    _copy_subset(raw, output, 2)
    assert sha256_file(raw) == before
    with h5py.File(output, "r") as file:
        assert set(file["data"]) == {"demo_0", "demo_2"}
        assert file["data"].attrs["total"] == 5
        assert file["data"].attrs["cfact_source_provenance"] == "preserved"
        assert file["data/demo_2"].attrs["original_episode_index"] == 2
    with pytest.raises(FileExistsError):
        _copy_subset(raw, output, 1)


def test_explicit_sparse_source_identity_and_archive_extraction(tmp_path):
    source = _fixture(tmp_path / "original")
    parquet = source / "data/chunk-000/episode_000000.parquet"
    pd.DataFrame({"action": [np.arange(12), np.arange(12) + 1],
                  "next.reward": [0., 1.], "next.done": [False, True]}).to_parquet(parquet)
    for index in (7, 19):
        shutil.copytree(source / "extras/episode_000000", source / f"extras/episode_{index:06d}")
        shutil.copyfile(parquet, parquet.with_name(f"episode_{index:06d}.parquet"))
    (source / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": i}) for i in (0, 7, 19)))
    archive = tmp_path / "official.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(source, arcname="lerobot")
    report = import_source(archive, tmp_path / "sparse.hdf5", "OpenDrawer", demos=["demo_19", "demo_7"])
    with h5py.File(report["output"]) as dataset:
        assert set(dataset["data"]) == {"demo_7", "demo_19"}
        assert dataset["data/demo_19"].attrs["lerobot_episode_index"] == 19
    assert report["demo_episode_mapping"] == {"demo_7": "episode_000007", "demo_19": "episode_000019"}
    assert not (Path(report["source_root"]) / "extras/episode_000000").exists()
    with pytest.raises(ValueError, match="missing or duplicated"):
        import_source(source, tmp_path / "missing.hdf5", "OpenDrawer", demos=["demo_3"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        import_source(source, tmp_path / "duplicate.hdf5", "OpenDrawer", demos=["demo_7", "demo_7"])
