"""Download the pinned upstream kitchen assets into the isolated simulator tree."""

from __future__ import annotations
import argparse
from importlib.resources import files
import json
import os
from pathlib import Path
from cfact.io.serialization import RunLock, atomic_json
from cfact.sources.archives import extract_asset_zip
from cfact.sources.manifest import sha256_file


def runtime_root(value: Path | None = None) -> Path:
    requested = value or os.environ.get("CFACT_RUNTIME_ROOT")
    if not requested:
        raise ValueError("Source the runtime activate.sh or supply --runtime-root")
    return Path(requested).expanduser().resolve()


def load_lock() -> dict:
    return json.loads(files("cfact.resources").joinpath("runtime.json").read_text())


def download_assets(root: Path, names: list[str] | None = None) -> dict:
    from huggingface_hub import hf_hub_download

    lock = load_lock()
    if json.loads((root / "runtime-lock.json").read_text()) != lock:
        raise ValueError("Runtime lock differs from this CFACT version; bootstrap a new runtime")
    asset_root = root / "sources/robocasa/robocasa/models/assets"
    records = root / "asset-receipts"
    records.mkdir(exist_ok=True)
    guard = RunLock(root / "assets.lock")
    guard.acquire()
    results = []
    try:
        for item in lock["assets"]:
            if names and item["name"] not in names:
                continue
            receipt = records / (item["name"] + ".json")
            folder = asset_root / item["folder"]
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if (
                    saved["revision"] != item["revision"]
                    or not folder.is_dir()
                    or not any(folder.rglob("*"))
                ):
                    raise ValueError(f"Existing asset receipt is inconsistent: {receipt}")
                results.append(saved)
                continue
            print(f"Downloading {item['filename']} at {item['revision']}", flush=True)
            archive = Path(
                hf_hub_download(
                    repo_id=item["repo_id"],
                    repo_type="dataset",
                    filename=item["filename"],
                    revision=item["revision"],
                )
            )
            # Upstream archives contain the folder basename; objects live one level deeper.
            info = extract_asset_zip(archive, folder.parent)
            if not folder.is_dir() or not any(folder.rglob("*")):
                raise ValueError(f"Archive did not create the expected asset folder: {folder}")
            saved = {**item, **info, "archive_sha256": sha256_file(archive), "folder": str(folder)}
            atomic_json(receipt, saved)
            results.append(saved)
        return {"assets": results, "runtime_root": str(root)}
    finally:
        guard.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--types", nargs="+", choices=[i["name"] for i in load_lock()["assets"]])
    parser.add_argument(
        "--plan", action="store_true", help="Print download inventory without network or writes"
    )
    args = parser.parse_args()
    if args.plan:
        root = runtime_root(args.runtime_root)
        selected = [a for a in load_lock()["assets"] if not args.types or a["name"] in args.types]
        print(
            json.dumps(
                {
                    "runtime_root": str(root),
                    "archives": selected,
                    "archive_count": len(selected),
                    "downloads_started": False,
                },
                indent=2,
            )
        )
        return
    print(json.dumps(download_assets(runtime_root(args.runtime_root), args.types), indent=2))


if __name__ == "__main__":
    main()
