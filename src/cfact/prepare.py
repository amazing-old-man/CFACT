"""Resource-checked preparation of the exact selected root and donor demos."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from cfact.io.serialization import atomic_json
from cfact.run import load_config, parameters
from cfact.runtime.resources import admission
from cfact.sources.prepare import validate_prepared_identity
from cfact.sources.validation import validate_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--demo", action="append", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/quickstart.yaml"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    validate_raw(args.source, args.task)
    if args.output.exists():
        identity = validate_prepared_identity(args.output, raw=args.source, required_episodes=args.demo)
        print(json.dumps({"status": "REUSED_VERIFIED_PREPARED", "identity": identity}, indent=2))
        return
    config = parameters(load_config(args.config), 1, None)
    gate = admission(Path.cwd(), [args.gpu], 1, config["resource"])
    atomic_json(args.output.with_suffix(".admission.json"), gate)
    if gate["status"] != "PASS":
        raise RuntimeError("NOT_RUN_RESOURCE_BLOCKED; see preparation admission report")
    command = [sys.executable, "-m", "cfact.sources.prepare", "--raw", str(args.source),
               "--output", str(args.output), "--task", args.task, "--gpu", str(args.gpu)]
    for demo in args.demo:
        command += ["--demo", demo]
    if args.resume:
        command.append("--resume")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
