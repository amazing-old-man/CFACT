"""Check a new release identity without rewriting the historical FREEZE.json."""

import argparse
import json
import subprocess
from pathlib import Path
from cfact.collection.snapshot import implementation_digest
from cfact.sources.manifest import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    args = parser.parse_args()
    release = json.loads(args.release.read_text())
    root = args.checkout.resolve()
    if release.get("status") != "FROZEN":
        raise ValueError("Release was not admitted for freezing")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    tag = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", release["tag"] + "^{commit}"], text=True
    ).strip()
    if head != release["git_commit"] or tag != head:
        raise ValueError("Release HEAD/tag differs")
    for path, digest in release["source_file_sha256"].items():
        if sha256_file(root / path) != digest:
            raise ValueError("Frozen source file changed: " + path)
    if implementation_digest() != release["generator_code_sha256"]:
        raise ValueError("Actual imported generator differs")
    print(
        json.dumps(
            {
                "status": "PASS",
                "git_commit": head,
                "tag": release["tag"],
                "generator_code_sha256": release["generator_code_sha256"],
                "physics_started": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
