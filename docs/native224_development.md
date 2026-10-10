# Native224 generator development

The current code/config revision is `cfact-0.7.2.dev0-native224`. Historical
FREEZE, Git tag objects, Source archives and formal output remain untouched.

`resources/collection.json` declares `cell_width=224`, `cell_height=224`,
`feature_scale=1.0`, and `camera_schema=cfact.native-rgb224.v1` for all three
existing cameras: robot0_agentview_left/right and robot0_eye_in_hand. The
configuration flows through environment camera setup, direct `sim.render`,
control-history ledger, clean node observations, review writer, replay prefix,
edge/lineage videos and official by-task/by-tree exports. The camera schema
validator rejects a rectangular or scaled acquisition under the native224 name.

MuJoCo camera RGB receives its established vertical flip; no channel swap,
crop, letterbox or image stretch is introduced. Clean frames are JPEG q80 at
native size; history remains lossless RGB. Review panels combine three views
with text and are presentation only. Camera metadata records actual IDs,
extrinsics, vertical FOV, raw/feature size and transform receipt. With fixed
vertical FOV, changing wide rendering to square changes horizontal FOV; it is
not a pixel-equivalent rendering of the old view. The inspected official Source
camera MP4 is 256 square, uses the same named scene camera, and is explicitly
resized only by the model Source adapter.

Physics/candidate code uses simulator poses, contacts, controller/action codec
and task predicates. The environment emits image observations, but the
inspected generation/control path does not feed pixels to an image policy.
Rendering must not advance sim time. The bounded run checks clocks, action/state
alignment, snapshot identity, local/fixed boundaries and recursion; it does not
claim identical old/new trajectories across different run identities.

From this repository root, with the existing fixed runtime installed:

```bash
source .runtime/activate.sh
python -m cfact generate --config configs/pre_scale_native224.yaml \
  --prepared-dir .cache/cfact-20-sources.t0W1Cg/prepared --workers 1 --gpu-devices 1
"$CFACT_EXPORT_PYTHON" -m cfact export --run "$NEW_RUN" --media full --by-tree
"$CFACT_EXPORT_PYTHON" -m cfact verify --run "$NEW_RUN"
```

`NEW_RUN` must be the exact path printed by this invocation. The cached prepared
path names a local input; other installations must provide their own prepared
identity-bound official Sources. This configuration uses only development
OpenDrawer/CoffeeServeMug demo0, donors3/4 and the existing bounded recursion.
Do not rerun the example as a pilot expansion. The measured run already exists
under output/pre_scale_checks; its manifest, source snapshot and receipts bind
actual code/config. No historical output is re-exported in place.

The native224 tests cover actual render dimensions, vertical orientation, RGB
channel order, clean image/history agreement and schema rejection. End-to-end
receipts and the bounded next-stage proposal live in the model repository's
output/pre_scale_checks/lewm_h3f5_c1_c2_native224_v1 directory. Source splits and
formal training normalization require user review before freezing.
