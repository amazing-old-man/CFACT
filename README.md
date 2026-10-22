# CFACT

**Generate physical counterfactual action trees from robot demonstrations.**

[中文](README.zh-CN.md) · [Installation](#installation) · [Quick start](#quick-start) · [Data access](#read-and-visualize-data) · [Configuration](#configure-your-own-collection)

CFACT constructs action trees in RoboCasa simulation. Starting from a shared
physical state, it executes alternative robot actions and records what actually
happens. Eligible child states can become new branching points, yielding
trajectories with shared history and different physical outcomes.

Use it to generate datasets for action-conditioned prediction, compare outcomes
from the same state, and study how local action changes affect an interaction.

```text
Recorded demonstration → shared parent state
                          ├── reference action → observed child
                          ├── action variant 1 → observed child
                          └── action variant 2 → observed child
                                                   ├── next action → child
                                                   └── next action → child
```

## Features

- Execute alternatives from the same restored simulator snapshot.
- Preserve parent/child relationships and the recorded Source provenance.
- Save issued actions, measured robot/object states, task outcomes, and RGB observations.
- Reuse shared trajectory prefixes across multiple branches.
- Inspect native trees locally and export data through the official LeRobot API.

| Candidate family | Construction |
| --- | --- |
| **Reference** | Geometrically aligned execution of the recorded Source segment. |
| **G0** | Bounded local perturbation of the reference trajectory. |
| **G1** | Motion and event-timeline transfer from a compatible donor demonstration. |
| **G2** | Geometry-based task variants or local effect goals, depending on the generation profile. |

The Source is the recorded input; Reference is a newly executed candidate.
Candidate availability and G2 modes depend on the selected task and profile.
Requested goals are kept separate from measured physical outcomes.

## Requirements

The documented collection workflow targets **Linux x86_64, Python 3.10, and an
NVIDIA GPU with EGL support**. Start with one simulation worker. Windows and macOS
collection have not been validated by this workflow.

| Dependency | Setup |
| --- | --- |
| Python | 3.10 for the pinned simulation and export environments |
| Simulation | Pinned RoboCasa, robosuite, MimicGen, and robomimic source revisions |
| Rendering | Working NVIDIA driver and EGL libraries |
| Video | FFmpeg with `libx264`, plus `ffprobe` |
| Dataset export | Independent environment with LeRobot 0.4.0 |

Exact revisions are in [runtime.json](src/cfact/resources/runtime.json); Python
constraints are in [requirements/](requirements). Simulation and export use
separate environments to avoid dependency conflicts.

Disk and memory requirements depend on the scenes and collection size. Account
for asset downloads and extraction, Source archives, both environments, and RGB
recordings. Preview the asset inventory before downloading and check resource
admission before starting a larger collection.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/amazing-old-man/CFACT.git
cd CFACT
```

Run the following commands from this directory in Bash.

### 2. Install system dependencies

For Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y git python3.10 python3.10-venv ffmpeg libegl1 libgl1 libosmesa6
nvidia-smi
```

On other distributions, install equivalent packages and make Python 3.10
available. The NVIDIA driver must already work before configuring EGL.

### 3. Create the isolated runtimes

```bash
python3.10 scripts/bootstrap_sim.py --runtime-root .runtime --python python3.10
python3.10 scripts/bootstrap_export.py --runtime-root .runtime
source .runtime/activate.sh
```

The simulation installer downloads the pinned upstream source trees and creates
`.runtime/venv`. The export installer creates `.runtime/export-venv` independently.
`activate.sh` sets the interpreter, runtime paths, cache locations, rendering
backend, and export interpreter. Source it again in each new terminal.

To install on another disk, replace `.runtime` with your desired runtime path in
both installer commands and source the resulting `activate.sh` there.

### 4. Download assets and check rendering

```bash
cfact download-assets --plan
cfact download-assets

mkdir -p output
cfact doctor --render --gpu 0 --output output/doctor.json
```

Continue when the doctor reports `PASS`. It checks pinned dependencies, asset
presence, FFmpeg, the selected physical GPU, and a small real EGL render. Actual
task replay is checked by the quick start below.

## Quick start

This example downloads an official OpenDrawer Source bundle, imports three
demonstrations, prepares simulator annotations, collects a small physical tree,
and validates the exported data. It uses the packaged quick-start protocol.

### 1. Download a Source bundle

Choose a writable workspace with enough free disk space. The following uses a
directory next to the checkout; it can be replaced with a path on another disk.

```bash
export CFACT_WORKSPACE="$PWD/../CFACT-workspace"
mkdir -p "$CFACT_WORKSPACE"
export CFACT_WORKSPACE="$(cd "$CFACT_WORKSPACE" && pwd)"

cfact download --task OpenDrawer --split pretrain --source human \
  --output "$CFACT_WORKSPACE/sources/OpenDrawer.tar"
```

The downloader uses the official dataset registry in
[datasets.json](src/cfact/resources/datasets.json), resolves its revision, and
saves a SHA256 receipt. Dataset availability and access terms are controlled by
the provider. If authorization is required, obtain access and run `hf auth login`.

### 2. Run one end-to-end collection

```bash
cfact quickstart \
  --source "$CFACT_WORKSPACE/sources/OpenDrawer.tar" \
  --output "$CFACT_WORKSPACE/quickstart/OpenDrawer" \
  --task OpenDrawer --gpu 0 --max-episodes 3
```

The output directory must be new. The first imported episode is the root and the
remaining episodes are disjoint donors. The command runs real physics and checks
the resulting trajectories and official dataset readback.

Progress is logged inside the output directory. Check `acceptance.json` for the
final `PASS` or `FAIL` result, and the individual step logs for diagnostics.
Passing this example validates the selected inputs and environment; it does not
establish support or task success across every scene.

### 3. Inspect the result

```bash
export CFACT_RELEASE="$CFACT_WORKSPACE/quickstart/OpenDrawer/releases/pilot"
export CFACT_RAW="$CFACT_RELEASE/_work/collection"

cfact inspect --run "$CFACT_RELEASE"
cfact example --run "$CFACT_RELEASE" --task OpenDrawer
cfact view --run "$CFACT_RAW" --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765` while the viewer is running. For a remote server,
forward the port with SSH. Stop the viewer with `Ctrl+C`.

## Configure your own collection

For explicit task, root, and donor selection, use the configurable lifecycle
interface after the quick start works.

### Prepare replay inputs

```bash
cfact import-source \
  --source "$CFACT_WORKSPACE/sources/OpenDrawer.tar" \
  --output "$CFACT_WORKSPACE/sources/OpenDrawer.raw.hdf5" \
  --task OpenDrawer --max-episodes 3

cfact validate-source \
  --source "$CFACT_WORKSPACE/sources/OpenDrawer.raw.hdf5" \
  --task OpenDrawer \
  --output "$CFACT_WORKSPACE/sources/OpenDrawer.validation.json"

cfact prepare-source \
  --raw "$CFACT_WORKSPACE/sources/OpenDrawer.raw.hdf5" \
  --output "$CFACT_WORKSPACE/prepared/OpenDrawer.hdf5" \
  --task OpenDrawer --gpu 0 \
  --demo demo_0 --demo demo_1 --demo demo_2
```

Replay inputs must include environment metadata, scene XML, and simulator states.
The official replay `extras/` records provide these. Videos and actions alone
cannot restore a physical parent state. Import and preparation refuse to overwrite
completed files; reuse verified inputs or choose new output paths.

### Select tasks and budgets

Copy [quickstart.yaml](configs/quickstart.yaml) to a configuration of your own.
Its selection fields look like this:

```yaml
schema: cfact.run.v1
source_dataset: PhysicalAI-Robotics-Manipulation-Kitchen-Demos
seed: 20260922
workers: 1
maximum_decision_depth: 2
# Retain and adapt the resource block from configs/quickstart.yaml.
tasks:
  - task: OpenDrawer
    roots: [demo_0]
    donors: [demo_1, demo_2]
```

This excerpt is not a complete configuration: retain the `resource` block from
the supplied file. Root and donor IDs must exist in the prepared input and must
be disjoint. Add task rows only after preparing their corresponding HDF5 files.
Resource thresholds are configurable host policies; size them from measured
scene costs rather than treating them as universal hardware requirements.

### Check the plan, then generate

```bash
cfact generate --config configs/quickstart.yaml \
  --prepared-dir "$CFACT_WORKSPACE/prepared" \
  --artifact-root "$CFACT_WORKSPACE/generated" \
  --workers 1 --gpu-devices 0 --dry-run

cfact generate --config configs/quickstart.yaml \
  --prepared-dir "$CFACT_WORKSPACE/prepared" \
  --artifact-root "$CFACT_WORKSPACE/generated" \
  --workers 1 --gpu-devices 0
```

Replace the config path with your own file when customizing. `--dry-run` reports
the requested scope and resource admission without starting physics; it does not
validate replay or prove that a Source will yield a usable tree. The actual run
validates prepared identities and freezes inputs, configuration, and code identity.
The command prints its newly created run path.

For larger batches, increase independent root requests and worker count gradually.
`--gpu-devices` accepts comma-separated physical GPU indices. This lifecycle
interface accepts **1–6 workers**; advanced frozen-round orchestration has a
separate interface and configuration. Do not assume their flags or run layouts
are interchangeable.

### Status, pause, and resume

Set `CFACT_RUN` to the run directory printed by `cfact generate`:

```bash
export CFACT_RUN="/absolute/path/to/the/generated/run"
cfact status --run "$CFACT_RUN"
cfact pause --run "$CFACT_RUN"
# Wait for active roots to finish and the coordinator to exit.
cfact resume --run "$CFACT_RUN"
```

Pause stops new dispatch and drains active roots. Resume preserves Source
identities and skips completed roots. Keep the original code, configuration,
prepared files, and run path; create a new run when changing inputs.

To export the configurable run after collection:

```bash
cfact export --run "$CFACT_RUN" --media full
cfact verify --run "$CFACT_RUN"
```

For custom full collection parameters, `cfact plan --help` and `cfact collect
--help` expose the lower-level interfaces. Recursive effect-goal generation is
implemented in [stage_compiler.py](src/cfact/generation/stage_compiler.py) and
[frontier.py](src/cfact/generation/frontier.py); it requires an explicit matching
profile. The default quick start should not be interpreted as enabling every
historical or experimental protocol.

## What is saved?

The native tree format retains both intended plans and executed measurements:

| Record | Contents |
| --- | --- |
| Parent context and snapshot | Physical branching state, object roles, geometry, and history references |
| Candidate plan | Source family, proposed motion, gripper commands, and provenance |
| `execution.npz` | Issued actions and measured state trajectory |
| `local_result.json` | Stop reason, local events, physical progress, and official success |
| Child snapshot and context | Actual state available for further expansion |
| `tree.json` | Edges and recursive structure |
| Observation ledger and videos | Recorded RGB observations and review media |

Exports additionally contain LeRobot data, tree/episode mappings, lineage records,
and verification reports. The native camera profile uses three 224 × 224 RGB
views, including the wrist camera, at 20 Hz. For a segment with `K` executed
actions, measured state arrays contain `K + 1` states.

Keep these labels distinct:

- **Physical completion:** collection of the branch finished.
- **Local event attainment:** the branch met its local operation criterion.
- **Official task success:** the environment's full-task predicate was satisfied.
- **Horizon censored:** success was not observed within the recorded horizon;
  this is not proof of task failure.

Multiple branches can share ancestors. They should not be counted as independent
Source demonstrations or automatically converted into preference labels.

## Read and visualize data

The lightweight reader does not require a simulator installation:

```bash
python3 -m venv .venv-reader
source .venv-reader/bin/activate
python -m pip install -e .
```

For an existing native dataset:

```python
from cfact import TreeDataset

dataset = TreeDataset("/path/to/native/collection")
print(dataset.roots())
```

For a published release:

```python
from cfact import ReleaseDataset

dataset = ReleaseDataset("/path/to/published/release")
print(dataset.roots())
```

Use `cfact example --run /path/to/dataset` to inspect sibling outcomes and a
measured transition. To access the exported LeRobot data directly, use the
independent export environment and the official `LeRobotDataset` reader. Episode
offsets come from its metadata, not from assumptions about Parquet or MP4 filenames.

## Task coverage

The [task registry](src/cfact/tasks.py) includes drawer and rack sliding,
cabinet and oven doors, object placement, coffee-cup handling, buttons, faucets,
and knobs. Availability depends on the actual scene, simulator binding, Source
replay, contact conditions, and implemented continuation.

Free-body blender-lid grasp/release is explicitly unsupported by the recursive
effect compiler. A registered task does not guarantee a usable tree, every
candidate family, or official task success for every demonstration. General
regrasp and universal collision-free planning are not provided.

## Troubleshooting

| Issue | Suggested action |
| --- | --- |
| Missing simulator modules or wrong versions | Source the runtime `activate.sh` and use the pinned bootstrap environment. |
| EGL or GPU errors | Check `nvidia-smi`, EGL libraries, physical GPU selection, and `cfact doctor --render`. |
| Missing meshes, textures, or scene assets | Complete `cfact download-assets` and verify the Source includes replay metadata. |
| Dataset download returns 401/403 | Obtain provider access, then authenticate with `hf auth login`. |
| Xet-specific download failure | Retry with `HF_HUB_DISABLE_XET=1` if the network cannot reach the Xet service. |
| Resource admission blocked | Read the admission report, reduce workers, and adjust resource reserves to your measured host capacity. |
| Existing output path | Reuse verified inputs or select a new output directory; do not overwrite partial collection records. |
| No compatible donor or continuation | Inspect candidate rejection and stop reasons; missing branches are valid recorded outcomes. |
| Resume rejects changed inputs | Restore the original code and prepared files, or create a new run. |

## Development

In a development environment:

```bash
python -m pip install -e '.[test,dev]'
python -m pytest
```

Simulator integration checks need the pinned runtime, assets, and replay inputs.
Passing reader or synthetic-format tests alone does not validate physical
generation. When reporting a generated dataset, retain the code revision,
configuration, Source manifest, and verification results.

Historical experiment configurations and results are archived under
[research/](research/README.md). They are protocol/provenance references rather
than installation prerequisites; their server paths should not be copied into a
new deployment.

## License and attribution

CFACT code is released under the [MIT License](LICENSE). Upstream simulators,
assets, and datasets retain their own licenses and access requirements; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata is provided in
[CITATION.cff](CITATION.cff).
