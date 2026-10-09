"""Download one official RoboCasa replay bundle, retaining its revision and hash."""

from __future__ import annotations
import argparse
from importlib.resources import files
import json
from pathlib import Path
import shutil
from cfact.io.serialization import atomic_json
from cfact.sources.manifest import sha256_file
from cfact.tasks import SUPPORTED_TASK_NAMES


def download_source(
    task: str,
    output: Path,
    split: str = "pretrain",
    source: str = "human",
    revision: str | None = None,
    offline: bool = False,
    cache_dir: Path | None = None,
) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    catalog = json.loads(files("cfact.resources").joinpath("datasets.json").read_text())
    filename = catalog["datasets"].get(task, {}).get(f"{split}/{source}")
    if filename is None:
        raise ValueError(f"No registered {split}/{source} dataset for {task}")
    output = output.expanduser().resolve()
    revision = revision or catalog["default_revision"]
    receipt = output.with_suffix(output.suffix + ".download.json")
    if output.exists():
        if not receipt.is_file():
            raise ValueError("Existing archive has no download receipt")
        saved = json.loads(receipt.read_text())
        if any(saved.get(k) != v for k, v in {
            "task": task, "repo_id": catalog["repo_id"], "filename": filename,
            "requested_revision": revision,
        }.items()) or sha256_file(output) != saved["sha256"]:
            raise ValueError("Cached source identity or SHA256 differs")
        return {**saved, "cache_reused": True, "offline": offline}
    if offline:
        raise FileNotFoundError("Offline mode requires the archive and its verified receipt")
    # Resolve a branch once, then download its immutable snapshot.
    resolved = HfApi().dataset_info(catalog["repo_id"], revision=revision).sha
    cached = Path(
        hf_hub_download(
            repo_id=catalog["repo_id"], repo_type="dataset", filename=filename,
            revision=resolved, cache_dir=cache_dir,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with cached.open("rb") as src, partial.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    partial.rename(output)
    result = {
        "task": task,
        "repo_id": catalog["repo_id"],
        "filename": filename,
        "requested_revision": revision,
        "resolved_revision": resolved,
        "sha256": sha256_file(output),
        "output": str(output),
    }
    atomic_json(output.with_suffix(output.suffix + ".download.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=SUPPORTED_TASK_NAMES, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New tar file")
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    parser.add_argument("--source", choices=["human", "mimicgen"], default="human")
    parser.add_argument("--revision", help="HF commit or branch; resolved SHA is recorded")
    parser.add_argument("--offline", action="store_true", help="Verify and reuse an existing archive")
    parser.add_argument("--cache-dir", type=Path, help="Explicit Hugging Face cache directory")
    args = parser.parse_args()
    try:
        result = download_source(args.task, args.output, args.split, args.source, args.revision,
                                 args.offline, args.cache_dir)
    except Exception as error:
        # No token is read or printed; authentication is managed by the official HF client.
        from huggingface_hub.errors import HfHubHTTPError

        if isinstance(error, HfHubHTTPError) and error.response.status_code in (401, 403):
            parser.exit(
                1,
                "Dataset access was denied. Open the official dataset page, accept any "
                "required terms, then run `hf auth login` and retry. An existing local "
                "bundle can be passed directly to `cfact import-source`.\n",
            )
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
