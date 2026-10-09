"""Publish a new portable tree + official LeRobot release from measured collection output."""

from __future__ import annotations
import argparse
import json
import os
from importlib.resources import files
from pathlib import Path
import sys
import pyarrow as pa
import pyarrow.parquet as pq
from cfact.io.serialization import RunLock, atomic_json
from cfact.collection.snapshot import implementation_digest
from cfact.storage.integrity import input_fingerprint, release_inventory, verify_release
from cfact.storage.layout import SCHEMA, write_json
from cfact.storage.tree import publish_tree
from cfact.export.shards import write_segment, reuse_official_dataset, verify_shard


def write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(path)


def publish(
    run: Path,
    output: Path,
    *,
    resume: bool = False,
    reuse_lerobot: Path | None = None,
    root_ids: list[str] | None = None,
    verify: bool = True,
) -> dict:
    run, output = run.resolve(), output.resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError("Publish into a new version directory outside the historical input run")
    if output.exists() and not resume:
        # Automatic collection reserves this directory and writes only its private workspace.
        if set(p.name for p in output.iterdir()) - {"_work"}:
            raise FileExistsError("Release exists; use --resume with identical inputs")
    output.mkdir(parents=True, exist_ok=True)
    lock = RunLock(output / ".publish.lock")
    lock.acquire()
    try:
        roots = [p.parent for p in sorted(run.glob("data/*/*/*/root_result.json"))]
        if root_ids:
            roots = [p for p in roots if p.name in root_ids]
            if {p.name for p in roots} != set(root_ids):
                raise ValueError("Requested root IDs are missing from the input collection")
        if not roots:
            raise ValueError("No measured root results found")
        fingerprint = input_fingerprint(run, roots)
        publisher = implementation_digest()
        mode = "reuse" if reuse_lerobot else "native"
        manifest_path = output / "release.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["input_fingerprint"] != fingerprint or previous["schema"] != SCHEMA:
                raise ValueError("Resume source inventory / schema differs; create a new release")
            if (
                previous.get("publisher_sha256") != publisher
                or previous.get("export_mode") != mode
                or previous.get("official_verification") != verify
            ):
                raise ValueError("Resume publisher / export mode differs; create a new release")
            if previous["status"] == "COMPLETE":
                verify_release(output, previous)
                # No writing or encoding on a complete resume.
                return previous
        config = json.loads((run / "config_frozen.json").read_text())
        scope = json.loads((run / "source_split_frozen.json").read_text())
        record = {
            "schema": SCHEMA,
            "release_id": output.name,
            "input_fingerprint": fingerprint,
            "publisher_sha256": publisher,
            "export_mode": mode,
            "official_verification": verify,
            "file_integrity": "mappings/file_integrity.json",
            "status": "BUILDING",
            "trees": [],
            "episode_tree_map": "mappings/episode_tree_map.parquet",
            "terminal_observation_map": "mappings/terminal_observation_map.parquet",
            "lineage_paths": "mappings/lineage_paths.jsonl",
            "lerobot_version": "0.4.0",
        }
        atomic_json(manifest_path, record)
        published = [publish_tree(root, run, output, config, scope) for root in roots]
        if len({p["tree"]["root_id"] for p in published}) != len(published):
            raise ValueError("Duplicate root identity in the selected acquisition records")
        segments = [s for p in published for s in p["segments"]]
        if reuse_lerobot:
            rows = reuse_official_dataset(reuse_lerobot.resolve(), output, segments)
        else:
            rows = [write_segment(output, s) for s in segments if s["action_count"]]
        if len({r["segment_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate segment-to-episode mapping")
        if len({(r["shard_path"], r["episode_index"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate official episode mapping")
        by_segment = {r["segment_id"]: r for r in rows}
        terminals = [t for p in published for t in p["terminals"]]
        for terminal in terminals:
            row = by_segment.get(terminal["segment_id"])
            if row:
                terminal.update(
                    {
                        k: row[k]
                        for k in [
                            "compatible_schema",
                            "shard_id",
                            "shard_path",
                            "episode_index",
                            "repo_id",
                        ]
                    }
                )
            elif not terminal["zero_actions"]:
                raise ValueError("Nonempty segment has no official episode")
        write_parquet(output / record["episode_tree_map"], rows)
        write_parquet(output / record["terminal_observation_map"], terminals)
        lineages = [line for p in published for line in p["lineages"]]
        path = output / record["lineage_paths"]
        path.write_text("".join(json.dumps(line, sort_keys=True) + "\n" for line in lineages))
        reports = [verify_shard(output, row) for row in rows] if verify else []
        write_json(
            output / "mappings/official_readback.json",
            {
                "status": ("PASS" if rows else "NO_EPISODES") if verify else "NOT_RUN",
                "samples": reports,
                "rows_decoded": sum(r["decoded_rows"] for r in reports),
            },
        )
        (output / "viewer").mkdir(exist_ok=True)
        (output / "viewer/index.html").write_text(
            files("cfact.viewer").joinpath("release.html").read_text()
        )
        record.update(
            {
                "status": "COMPLETE",
                "trees": [p["tree"]["tree_path"] for p in published],
                "roots": len(published),
                "nodes": sum(len(p["tree"]["nodes"]) for p in published),
                "edges": sum(len(p["tree"]["edges"]) for p in published),
                "episodes": len(rows),
                "frames": sum(row["frame_count"] for row in rows),
                "terminals": len(terminals),
                "zero_action_terminals": sum(t["zero_actions"] for t in terminals),
                "shards": len({row["shard_path"] for row in rows}),
                "official_readback": "mappings/official_readback.json",
            }
        )
        atomic_json(output / record["file_integrity"], release_inventory(output))
        atomic_json(manifest_path, record)
        return record
    finally:
        lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--release-id", help="Used under outputs/CFACT-20 when --output is omitted")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--root-id", action="append", help="Optional exact existing root ID, repeatable"
    )
    parser.add_argument(
        "--reuse-lerobot", type=Path, help="Verified existing official export with matching traces"
    )
    parser.add_argument(
        "--python-packages", type=Path, help="Existing compatible export dependency directory"
    )
    args = parser.parse_args()
    if args.python_packages:
        sys.path.insert(0, str(args.python_packages.resolve()))
    if args.output is None:
        if (
            not args.release_id
            or Path(args.release_id).name != args.release_id
            or args.release_id in (".", "..")
        ):
            parser.error("Supply --output or a single-component --release-id")
        args.output = (
            Path(os.environ.get("CFACT_OUTPUT_ROOT", "outputs/CFACT-20")) / args.release_id
        )
    print(
        json.dumps(
            publish(
                args.run,
                args.output,
                resume=args.resume,
                reuse_lerobot=args.reuse_lerobot,
                root_ids=args.root_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
