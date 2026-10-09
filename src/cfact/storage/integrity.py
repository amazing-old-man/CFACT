"""Content checks for publication inputs and immutable completed release files."""

from pathlib import Path
import hashlib
import json
from cfact.sources.manifest import sha256_file
from cfact.storage.layout import inside


def media_spool(run: Path, media: dict) -> Path:
    if media.get("video"):
        relocated = inside(run, media["video"]).with_suffix(".frames")
        if relocated.is_dir():
            return relocated
    spool = Path(media.get("frame_spool", ""))
    return spool if spool.is_absolute() else run / spool


def replay_history_path(run: Path, *records: dict) -> Path | None:
    """Resolve the same measured-history identity for normal and blocked roots."""
    paths = set()
    for record in records:
        value = record.get("source_replay_history")
        if not value:
            continue
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (run / path).resolve()
        if not path.is_relative_to(run.resolve()):
            raise ValueError("Measured history escapes acquisition run")
        paths.add(path)
    if len(paths) > 1:
        raise ValueError("Conflicting measured replay history identities")
    if not paths:
        return None
    path = paths.pop()
    if not (path / "index.json").is_file():
        raise ValueError("Declared measured replay history index is missing")
    return path


def input_fingerprint(run: Path, roots: list[Path]) -> str:
    inventory = {}

    def add(key, path):
        inventory[key] = sha256_file(path) if path.is_file() else "MISSING"

    for filename in ["config_frozen.json", "source_split_frozen.json", "run_manifest.json"]:
        add(filename, run / filename)
    for root in roots:
        context_path = root / "nodes/n_root/context.json"
        context = json.loads(context_path.read_text()) if context_path.is_file() else {}
        manifest = json.loads((root / "root_manifest.json").read_text())
        result = json.loads((root / "root_result.json").read_text())
        history = replay_history_path(run, context, manifest, result)
        if history:
            for path in sorted(history.glob("*")):
                if path.is_file():
                    add(path.relative_to(run).as_posix(), path)
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".npz", ".jsonl"}:
                add(path.relative_to(run).as_posix(), path)
        for path in sorted(root.glob("nodes/*/edges/*/media.json")):
            media = json.loads(path.read_text())
            prefix = path.relative_to(run).as_posix()
            spool = media_spool(run, media)
            for camera, paths in media.get("clean_camera_frame_paths", {}).items():
                for index, value in enumerate(paths):
                    add(f"{prefix}:camera:{camera}:{index}", inside(spool, value))
            video = inside(run, media["video"]) if media.get("video") else None
            if video is not None:
                add(prefix + ":review", video)
            if video is None or not video.is_file():
                for index in range(len(media.get("frame_sim_time_s", []))):
                    add(f"{prefix}:review-frame:{index}", spool / f"{index:06d}.jpg")
        index = inside(
            run, f"videos/original/{result['task']}/{manifest['source_episode']}/index.json"
        )
        add(index.relative_to(run).as_posix(), index)
        if index.is_file():
            for row in json.loads(index.read_text()).get("source_videos", []):
                add("source-video:" + row["video_path"], inside(run, row["video_path"]))
    return hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()


def release_inventory(output: Path) -> dict[str, str]:
    inventory = {}
    for folder in ["sources", "trees", "lerobot", "mappings", "videos", "viewer", "media"]:
        for path in sorted((output / folder).rglob("*")):
            if path.is_file() and path != output / "mappings/file_integrity.json":
                relative = path.relative_to(output).as_posix()
                inside(output, relative)
                inventory[relative] = sha256_file(path)
    return inventory


def verify_release(output: Path, record: dict) -> None:
    expected = json.loads(inside(output, record["file_integrity"]).read_text())
    actual = release_inventory(output)
    different = sorted(k for k in set(expected) | set(actual) if actual.get(k) != expected.get(k))
    if different:
        raise ValueError("Completed release integrity differs: " + ", ".join(different[:10]))
