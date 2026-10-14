"""Public command-line interface; optional dependencies load only when needed."""

from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import sys

COMMANDS = {
    "recursive": ("cfact.collection.round_runner", "Run a frozen recursive generation queue with server resource checks"),
    "audit-recursive": ("cfact.collection.recursive_audit", "Audit physical sibling effects and actual child ancestry"),
    "calibrate-recursive": ("cfact.collection.recursive_calibration", "Execute actual same-parent A/B/A noise calibration"),
    "export-recursive": ("cfact.export.recursive", "Export measured recursive trees and read all official LeRobot rows"),
    "download": ("cfact.sources.download", "Download or verify a fixed official Source archive"),
    "validate-source": ("cfact.sources.validation", "Validate authoritative replay fields and identity"),
    "manifest": ("cfact.collection.manifest_runner", "Prepare and run explicit resumable root jobs"),
    "render-lineage": ("cfact.export.lineage_video", "Render one measured replay-prefix and actual-child lineage"),
    "publish": ("cfact.export.release", "Publish tree and official LeRobot views"),
    "import-source": ("cfact.sources.importer", "Import RoboCasa LeRobot replay data"),
    "download-source": (
        "cfact.sources.download",
        "Download an official supported RoboCasa source bundle",
    ),
    "download-assets": ("cfact.runtime.assets", "Download pinned kitchen assets"),
    "doctor": ("cfact.runtime.doctor", "Check simulator sources, resources, and EGL"),
    "quickstart": ("cfact.sources.quickstart", "Run a small end-to-end physical collection"),
    "collect": ("cfact.collection.coordinator", "Collect physical action trees"),
    "plan": ("cfact.sources.manifest", "Freeze a collection plan from prepared HDF5 inputs"),
    "prepare": ("cfact.prepare", "Prepare explicit Source demos with resource admission"),
    "prepare-source": ("cfact.sources.prepare", "Add MimicGen annotations to source HDF5"),
    "export-segments": ("cfact.export.lerobot", "Export measured segments to LeRobot"),
    "view": ("cfact.viewer.server", "Browse a run locally"),
    "example": ("cfact.data.example", "Extract same-parent outcomes and one transition"),
}
LIFECYCLE = {"generate", "status", "resume", "pause", "export", "verify"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CFACT physical counterfactual action trees")
    parser.add_argument("--workspace", type=Path, help="External data/output workspace")
    parser.add_argument(
        "--source-root", type=Path, help="CFACT source checkout for collection snapshots"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Summarize an existing raw tree dataset")
    inspect.add_argument("--run", type=Path, required=True)
    inspect.add_argument("--task")
    sub.add_parser("list", help="List maintained commands")
    for name, (_, description) in COMMANDS.items():
        sub.add_parser(name, help=description, add_help=False)
    for name in sorted(LIFECYCLE):
        sub.add_parser(name, help=f"{name.capitalize()} a CFACT run", add_help=False)
    args, remaining = parser.parse_known_args(argv)
    if args.workspace:
        os.environ["CFACT_WORKSPACE"] = str(args.workspace.expanduser().resolve())
    if args.source_root:
        os.environ["CFACT_SOURCE_ROOT"] = str(args.source_root.expanduser().resolve())
    if args.command in {"inspect", "list"} and remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if args.command == "list":
        for name, (_, description) in COMMANDS.items():
            print(f"{name:16s} {description}")
    elif args.command == "inspect":
        from cfact import TreeDataset

        if (args.run / "release.json").is_file():
            from cfact import ReleaseDataset

            dataset = ReleaseDataset(args.run)
        else:
            dataset = TreeDataset(args.run)
        rows = dataset.roots(task=args.task)
        print(
            json.dumps(
                {
                    "run": str(args.run.resolve()),
                    "roots": len(rows),
                    "status": dict(Counter(row["status"] for row in rows)),
                },
                indent=2,
            )
        )
    else:
        if remaining[:1] == ["--"]:
            remaining = remaining[1:]
        previous = sys.argv
        try:
            sys.argv = [f"cfact {args.command}", *([args.command] if args.command in LIFECYCLE else []), *remaining]
            runpy.run_module("cfact.run" if args.command in LIFECYCLE else COMMANDS[args.command][0], run_name="__main__")
        finally:
            sys.argv = previous
