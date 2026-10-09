"""Import an official RoboCasa LeRobot directory or tar bundle as replay HDF5."""

from __future__ import annotations
import argparse
import json
import hashlib
import shutil
import h5py
from pathlib import Path
from cfact.adapters.lerobot import inspect_lerobot_dataset, import_raw_replay_hdf5
from cfact.io.serialization import RunLock, atomic_json
from cfact.sources.archives import extract_tar
from cfact.sources.manifest import sha256_file
from cfact.tasks import SUPPORTED_TASK_NAMES


def import_source(
    source: Path,
    output: Path,
    task: str,
    max_episodes: int | None = None,
    extract_to: Path | None = None,
) -> dict:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    if output.exists() or output.with_suffix(output.suffix + ".import.json").exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = RunLock(output.with_suffix(output.suffix + ".lock"))
    lock.acquire()
    try:
        archive_hash = None
        if source.is_file() and h5py.is_hdf5(source):
            from cfact.sources.validation import validate_raw
            validation = validate_raw(source, task)
            temporary = output.with_suffix(output.suffix + ".partial")
            with source.open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst)
            if sha256_file(temporary) != validation["sha256"]:
                raise RuntimeError("Source changed during local HDF5 import")
            temporary.rename(output)
            result = {"source_format": "local_robocasa_hdf5", "task": task,
                      "input_sha256": validation["sha256"], "output": str(output),
                      "max_episodes": max_episodes,
                      "subset_policy": "HDF5 import preserves all demos; select explicit demos during preparation",
                      "validation": validation}
            atomic_json(output.with_suffix(output.suffix + ".import.json"), result)
            return result
        if source.is_file():
            archive_hash = sha256_file(source)
            extracted = extract_to or output.parent / (output.stem + ".source")
            extract_tar(source, extracted)
            directory = extracted
        elif source.is_dir():
            if extract_to is not None:
                raise ValueError("--extract-to only applies to archive inputs")
            directory = source
        else:
            raise FileNotFoundError(source)
        inspection = inspect_lerobot_dataset(directory, expected_task=task)
        selected = inspection.episodes[:max_episodes] if max_episodes else inspection.episodes
        consumed = [inspection.root / "meta" / name for name in ["info.json", "modality.json"]]
        consumed.append(inspection.root / "extras/dataset_meta.json")
        if (inspection.root / "meta/episodes.jsonl").exists():
            consumed.append(inspection.root / "meta/episodes.jsonl")
        for episode in selected:
            consumed.extend(
                [
                    episode.parquet_path,
                    episode.states_path,
                    episode.model_xml_path,
                    episode.episode_meta_path,
                ]
            )
        hashes = {str(p.relative_to(inspection.root)): sha256_file(p) for p in consumed}
        result = import_raw_replay_hdf5(inspection, output, max_episodes=max_episodes)
        # Detect mutation during import rather than recording misleading provenance.
        for path, digest in hashes.items():
            if sha256_file(inspection.root / path) != digest:
                raise RuntimeError(f"Source changed while importing: {path}")
        package_digest = archive_hash
        package_kind = "downloaded_archive"
        if package_digest is None:
            # A directory input has no tar hash; hash the complete replay bundle,
            # independently of --max-episodes so root identity is stable across subsets.
            bundle = dict(hashes)
            for episode in inspection.episodes:
                for path in (
                    episode.parquet_path,
                    episode.states_path,
                    episode.model_xml_path,
                    episode.episode_meta_path,
                ):
                    key = str(path.relative_to(inspection.root))
                    if key not in bundle:
                        bundle[key] = sha256_file(path)
            package_digest = hashlib.sha256(json.dumps(bundle, sort_keys=True).encode()).hexdigest()
            package_kind = "replay_directory_digest"
        provenance = {
            "package_sha256": package_digest,
            "package_kind": package_kind,
            "source_format": result["source_format"],
        }
        download_receipt = source.with_suffix(source.suffix + ".download.json")
        if source.is_file() and download_receipt.is_file():
            downloaded = json.loads(download_receipt.read_text())
            if downloaded["sha256"] != archive_hash or downloaded["task"] != task:
                raise ValueError("Official download receipt does not match the imported archive")
            provenance["official_download"] = {
                k: downloaded[k] for k in ("repo_id", "filename", "resolved_revision", "sha256")
            }
        with h5py.File(output, "a") as dataset:
            dataset["data"].attrs["cfact_source_provenance"] = json.dumps(
                provenance, sort_keys=True
            )
            for index, episode in enumerate(selected):
                videos = {}
                for video in inspection.root.glob(
                    f"videos/*/observation.images.*/episode_{episode.index:06d}.mp4"
                ):
                    camera = video.parent.name.removeprefix("observation.images.")
                    if camera in videos:
                        raise ValueError(f"Duplicate source camera video: {camera}")
                    videos[camera] = {"path": str(video.resolve()), "sha256": sha256_file(video)}
                dataset[f"data/demo_{index}"].attrs["cfact_source_videos"] = json.dumps(
                    videos, sort_keys=True
                )

        result["source_provenance"] = provenance
        result["output_bytes"] = output.stat().st_size
        result.update(
            {
                "source": str(source),
                "source_root": str(inspection.root),
                "source_archive_sha256": archive_hash,
                "source_files_sha256": hashes,
                "output": str(output),
                "output_sha256": sha256_file(output),
            }
        )
        atomic_json(output.with_suffix(output.suffix + ".import.json"), result)
        return result
    finally:
        lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Extracted LeRobot root or tar archive"
    )
    parser.add_argument("--output", type=Path, required=True, help="New raw replay HDF5")
    parser.add_argument("--task", choices=SUPPORTED_TASK_NAMES, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--extract-to", type=Path, help="New directory; defaults beside output HDF5"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            import_source(args.source, args.output, args.task, args.max_episodes, args.extract_to),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
