"""Verify frozen root membership, official archive, raw and prepared identity; no physics."""

import argparse
import json
from pathlib import Path
import h5py
from cfact.sources.download import download_source
from cfact.sources.manifest import sha256_file
from cfact.sources.prepare import validate_prepared_identity
from cfact.sources.validation import validate_raw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("configs/source_train40_v1.json"))
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    m = json.loads(a.manifest.read_text())
    source = a.source_dir or Path(m["source_dir"])
    reports = []
    for t in m["tasks"]:
        roots = t["roots"]
        donors = t["donors_by_group"]
        backups = t["backups"]
        split = t["split"]
        if len(roots) != 20 or len(set(roots)) != 20 or roots[:18] != t["original_18"]:
            raise ValueError("frozen 18+2 roots changed")
        train = split["train_candidate"]
        if (
            roots != train[2:22]
            or donors["train_candidate"] != train[:2]
            or backups != train[22:27]
        ):
            raise ValueError("training hash order/membership changed")
        if (
            set(roots) & set(sum(donors.values(), []))
            or set(roots) & set(split["development"])
            or set(roots) & set(split["validation_candidate"] + split["test_candidate"])
        ):
            raise ValueError("source role leakage")
        role_sets = [
            set(donors[group])
            for group in [
                "train_candidate",
                "validation_candidate",
                "test_candidate",
                "development",
            ]
        ]
        if any(x & y for i, x in enumerate(role_sets) for y in role_sets[i + 1 :]):
            raise ValueError("donor cross-group leakage")
        task = t["task"]
        archive = source / "archives" / (task + ".tar")
        receipt = download_source(task, archive, revision=m["revision"], offline=True)
        if (
            receipt["sha256"] != t["archive_sha256"]
            or receipt["resolved_revision"] != m["revision"]
        ):
            raise ValueError("official archive identity mismatch")
        raw = source / "raw" / (task + ".hdf5")
        prepared = source / "prepared" / (task + ".hdf5")
        raw_schema = validate_raw(raw, task)
        required = ["demo_" + str(int(ep[8:])) for ep in roots + donors["train_candidate"]]
        identity = validate_prepared_identity(prepared, raw=raw, required_episodes=required)
        with h5py.File(raw, "r") as f:
            for demo in required:
                if int(f["data/" + demo].attrs["lerobot_episode_index"]) != int(demo[5:]):
                    raise ValueError("sparse demo/official identity changed")
                videos = json.loads(f["data/" + demo].attrs["cfact_source_videos"])
                if len(videos) != 3:
                    raise ValueError("missing three Source cameras")
                for video in videos.values():
                    path = Path(video["path"])
                    if (
                        not path.resolve().is_relative_to(source.resolve())
                        or sha256_file(path) != video["sha256"]
                    ):
                        raise ValueError("Source media path or SHA mismatch")
        reports.append(
            {
                "task": task,
                "roots": 20,
                "train_donors": 2,
                "raw_episodes": len(raw_schema["episodes"]),
                "archive_sha256": receipt["sha256"],
                "raw_sha256": raw_schema["sha256"],
                "prepared_sha256": sha256_file(prepared),
                "prepared_identity": identity,
                "status": "PASS",
            }
        )
    result = {"status": "PASS", "physics_started": False, "tasks": reports}
    if a.output:
        from cfact.io.serialization import atomic_json

        atomic_json(a.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
