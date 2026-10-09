# Third-party notices

The MIT license in this repository covers the original `cfact` package (formerly `mg-cf`),
tests, examples, and documentation only. It does not relicense dependencies,
datasets, simulator assets, or copied upstream code.

| Component | Pinned revision/version | Boundary |
|---|---|---|
| MimicGen (RoboCasa fork) | `b89c3b5e02927ceb8599401cc01551a8da21fbf8` | Retains its upstream NVIDIA license; source fetched separately |
| RoboCasa365 | `0c81ff9c39b771afc330349cbbe50be9776a74fa` | MIT code; assets/data have separate terms |
| robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` | MIT |
| robomimic | `271a76c2d55c8b0f94d3d589f26fcae0d47f64a1` | MIT |
| LeRobot | `0.4.0`, official Dataset v3 API | Apache-2.0; installed separately |

No upstream source, simulator assets, or downloaded datasets are bundled into the
public CFACT package. Runtime URLs and revisions are recorded in
`src/cfact/resources/runtime.json` and `datasets.json`. Downloaded archives retain
their source repository terms. Copies of user data created by publication do not
change those terms.

The installer uses clean pinned source checkouts. Historical external runtime
patches are not silently applied to new installations. CFACT's compatibility
adapters are explicit modules within this repository.

The exact public snapshots used by this acceptance run declare CC-BY-4.0 in their
upstream dataset cards (retrieved by the official client; see `evidence/data-terms.json`):

- [NVIDIA kitchen demonstrations](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos/blob/522e4ffa3bb9d9729f79b241c65fc9e589f2659c/README.md)
- [NVIDIA kitchen MJCF assets](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF/blob/420a04af939c34873e6839a586b70844baf28aab/README.md)
- [RoboCasa assets](https://huggingface.co/datasets/robocasa/robocasa-assets/blob/1b92c3d02ca4354984fec961357db0bff7b32166/README.md)

Preserve upstream attribution and applicable asset-specific notices when sharing
those inputs or derived media. This source release includes neither downloaded
assets/data nor generated media, and its MIT license does not replace their terms.
