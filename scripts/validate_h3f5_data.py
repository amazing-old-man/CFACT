import argparse
import json
import gzip
import csv
from pathlib import Path
from collections import Counter
from itertools import combinations
import numpy as np
import av
from PIL import Image
from cfact_models.data.run import FrozenRun
from cfact_models.data.temporal import (
    AlignedFlatDataset,
    ParentStarDataset,
    TemporalSpec,
    star_collate,
)
from cfact_models.data.normalization import Normalizer
from cfact_models.data.media import lewm_transform
from cfact_models.utils.io import sha256, ContractError

p = argparse.ArgumentParser()
p.add_argument("--run", type=Path, required=True)
p.add_argument("--audit", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
run = FrozenRun(a.run)
with gzip.open(a.audit / "window_registry.json.gz", "rt") as f:
    allrecords = json.load(f)
allgroups = json.loads((a.audit / "star_registry.json").read_text())
records = [r for r in allrecords if (r["history"], r["frameskip"], r["horizon"]) == (3, 5, 5)]
groups = [g for g in allgroups if (g["history"], g["frameskip"], g["horizon"]) == (3, 5, 5)]
normalizer = Normalizer(
    np.zeros(12, np.float32), np.ones(12, np.float32), (), tuple(["diagnostic_identity"] * 12)
)
flat = AlignedFlatDataset(a.run, records, normalizer, pixel_profile="native224")
stars = ParentStarDataset(flat, groups)
counts = Counter()
physical = []
starfacts = []
sourcefacts = []
for i, r in enumerate(records):
    sample = flat[i]
    assert list(sample["pixels"].shape) == [4, 3, 224, 224]
    assert list(sample["action"].shape) == [3, 60]
    assert len(r["action_refs"]) == 15
    assert abs(r["future_refs"][0]["time"] - r["context_refs"][-1]["time"] - 0.25) < 1e-8
    assert np.isfinite(sample["pixels"].numpy()).all()
    counts[(r["task"], r["layer"], r["segment"], bool(r["parent_qualified"]))] += 1
for i, g in enumerate(groups):
    sample = stars[i]
    pixels = [branch["pixels"] for branch in sample["branches"]]
    assert all(np.array_equal(pixels[0][:3].numpy(), v[:3].numpy()) for v in pixels[1:])
    rows = [records[flat.by_key[k]] for k in g["window_keys"]]
    for r in rows:
        native = run.trace(r["segment_id"])
        x = flat[flat.by_key[r["window_key"]]]["action"].numpy().reshape(15, 12)
        assert np.array_equal(x[-5:], native["commanded_actions"][:5].astype(np.float32))
        assert r["segment"] == "local" and r["anchor"] == 0
    for ra, rb in combinations(rows, 2):
        ta, tb = run.trace(ra["segment_id"]), run.trace(rb["segment_id"])
        pair = {
            "task": g["task"],
            "root": g["root_id"],
            "parent": g["parent_node"],
            "layer": g["layer"],
            "edges": [ra["edge_id"], rb["edge_id"]],
            "issued_action_h5_abs_max": float(
                np.abs(ta["commanded_actions"][:5] - tb["commanded_actions"][:5]).max()
            ),
        }
        for field in ["robot_qpos", "robot_qvel", "actual_eef_pose", "object_poses", "relevant_fixture_qpos"]:
            if field in ta and field in tb and ta[field][5].shape == tb[field][5].shape:
                pair[field + "_h5_abs_max"] = float(
                    np.abs(ta[field][5] - tb[field][5]).max(initial=0)
                )
        physical.append(pair)
    starfacts.append(
        {
            "task": g["task"],
            "root": g["root_id"],
            "layer": g["layer"],
            "parent": g["parent_node"],
            "siblings": len(rows),
            "pairs": len(rows) * (len(rows) - 1) // 2,
            "canonical_history_byte_equal": True,
            "history_identity": g["history_identity"],
            "window_keys": g["window_keys"],
        }
    )
# No dropped unpaired Flat rows: Star is strictly an index into the complete Flat registry.
keys = {k for g in groups for k in g["window_keys"]}
unpaired = [r["window_key"] for r in records if r["window_key"] not in keys]
reject = Counter()
with (a.audit / "STAR_CANDIDATES.csv").open() as f:
    for row in csv.DictReader(f):
        if (int(row["history"]), int(row["frameskip"]), int(row["horizon"])) == (3, 5, 5):
            reject.update(json.loads(row["branch_rejections"]).values())
negatives = []
for anchor, count, shared in [(9, 30, False), (0, 4, True)]:
    try:
        TemporalSpec(3, 5, 5).indices(anchor, count, shared)
        raise AssertionError("illegal accepted")
    except ContractError as e:
        negatives.append(
            {"anchor": anchor, "actions": count, "shared": shared, "rejection": str(e)}
        )
if groups:
    padded = star_collate([stars[i] for i in range(len(stars))])
    padding = {
        "branch_mask_shape": list(padded["branch_mask"].shape),
        "has_natural_padding": bool((~padded["branch_mask"]).any()),
    }
else:
    padding = {}
# Inspect the actual three Source videos and the same production image transform without any model.
transform = lewm_transform()
for taskrow in run.manifest["tasks"]:
    task = taskrow["task"]
    prepared = Path(taskrow["prepared_hdf5"])
    imported = json.loads(
        prepared.parent.parent.joinpath("raw", task + ".hdf5.import.json").read_text()
    )
    src = Path(imported["source_root"])
    meta = json.loads((src / "extras/episode_000000/ep_meta.json").read_text())
    for camera in ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]:
        video = next(src.glob("videos/*/observation.images." + camera + "/episode_000000.mp4"))
        with av.open(str(video)) as container:
            rgb = next(container.decode(video=0)).to_ndarray(format="rgb24")
        x = transform(Image.fromarray(rgb))
        assert rgb.shape == (256, 256, 3) and list(x.shape) == [3, 224, 224]
        native = np.zeros((224, 224, 3), np.uint8)
        y = transform(Image.fromarray(native))
        assert list(y.shape) == [3, 224, 224]
        sourcefacts.append(
            {
                "task": task,
                "camera": camera,
                "source_video_sha256": sha256(video),
                "source_resolution": [256, 256],
                "target_resolution": [224, 224],
                "source_transform": "RGB/255 -> ImageNet Normalize -> torchvision v2.Resize(224,BILINEAR,antialias=True); equal aspect ratio, no crop",
                "cf_transform": "same transform; native224 resize is identity",
                "camera_config": meta.get("cam_configs", {}).get(camera, "UNKNOWN"),
                "view_independence": "UNKNOWN",
                "visual_domain_limits": "MP4 Source compression and true replay/controller differences remain; no pixel identity or scene-disjoint claim",
            }
        )
model_files = {
    str(Path(m.__file__)): sha256(Path(m.__file__))
    for m in [
        __import__("cfact_models.data.temporal", fromlist=["x"]),
        __import__("cfact_models.data.interface_audit", fromlist=["x"]),
        __import__("cfact_models.data.media", fromlist=["x"]),
    ]
}
result = {
    "status": "PASS"
    if groups
    and all(
        any(g["task"] == t and g["layer"] == layer for g in starfacts)
        for t in ["OpenDrawer", "CoffeeServeMug"]
        for layer in ["Root", "Desc"]
    )
    else "NOT_VERIFIED",
    "recipe": "current declared H3/F5: context t-10,t-5,t; three 5x12 action60 blocks; future t+5; all intermediate control history real",
    "flat_windows": len(records),
    "flat_counts": [
        {"task": k[0], "layer": k[1], "segment": k[2], "parent_prefix": k[3], "count": v}
        for k, v in sorted(counts.items())
    ],
    "star_groups": starfacts,
    "star_parent_count": len(groups),
    "star_pair_count": sum(g["pairs"] for g in starfacts),
    "unpaired_flat_retained": len(unpaired),
    "all_qualified_flat_retained": True,
    "rejection_counts": dict(reject),
    "negative_contract_cases": negatives,
    "source_vs_cf_preprocessing": sourcefacts,
    "padding": padding,
    "model_interface_file_sha256": model_files,
    "training_updates": 0,
    "model_forward_calls": 0,
}
(a.output / "STAR_H3F5_VALIDATION.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False)
)
(a.output / "PHYSICAL_DIFFERENCES.json").write_text(json.dumps(physical, indent=2))
print(
    json.dumps(
        {
            "status": result["status"],
            "flat": len(records),
            "star_parents": len(groups),
            "star_pairs": result["star_pair_count"],
        },
        indent=2,
    )
)
