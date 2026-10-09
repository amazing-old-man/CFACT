# CFACT handoff

Repository: `/mnt/ssd2/yanghc/CFACT` — an independent Git repository, with no public
remote and no automatic push. Entry point for a new execution session: **README.md,
section 1**. No private prepared data or old workspace is needed.

Frozen engineering version/tag: **0.7.1.dev0 / v0.7.1.dev0**. Resolve the immutable
release commit with `git rev-parse v0.7.1.dev0^{commit}`; its recorded value and the
post-tag installation result are in `dist/tag-verification.json`. This preserves
the development lineage label rather than relabeling the historical candidate as
a previously published stable release.

Input snapshot content SHA256:
`7f58d15b12889c8fd275429d539c23d2ea05dd2c7317005dd8b6ab3170398d6f`.
The 141-file source inventory and extraction mapping are in `provenance.json`.
The 25 scientific core files are byte-identical. All 299,962 previously inventoried
input entries retained their bytes or symbolic-link target.

T0–T5 passed; see `TEST_RESULTS.json`, `RELEASE_CHECK.json`, and `evidence/`.
Simulation contracts: 193 passed, 1 skipped. Independent clone/wheel tests:
178 passed, 15 skipped (simulator integrations absent from that base environment).
Synthetic official-API tests are recorded separately and do not count as physics.

The public Source is NVIDIA's pinned LeRobot v2.1 package with replay extras,
imported into raw HDF5 and prepared by the unchanged pinned MimicGen adapter.
The three fixed public roots used original episode 0; donors used episodes 1/2.
Three tasks produced 19 actual edges and three genuinely recursive trees:
OpenDrawer 6, PickPlaceCounterToCabinet 5, TurnOnSinkFaucet 8. No failed root was
replaced with a different demo. Source replay restores the initial state exactly;
recorded trajectory differences, including a 0.05-second clock difference and
missing recorded controller internals, remain explicitly reported.

Representative output:
`output/PhysicalAI-Robotics-Manipulation-Kitchen-Demos__20261009T154332Z__c066cb390beb/`.
Acquisition commit: `2c6086b9911c056b3f445962bdea9a4ba03b4a95`.
Official LeRobot 0.4.0 read every row of the three task datasets and all three
independent tree datasets: 29 episodes, 1,415 action rows, three cameras.
There are 16 full leaf-lineage MP4s (9,646,255 extra bytes); 38 review MP4s decoded.
Paused/resumed acquisition retained the original run; completed physics files
were unchanged. Deferred media completion is separately accounted for.

README-only acceptance used another clone with new dependency checkouts and
venvs, reusing only identified public download caches. It produced 6 edges,
9 episodes / 486 action rows, and 5 full lineages. Both README official reader
examples and complete-run resume returned exit code 0. See `evidence/readme-blind.json`.

Stable commands: `cfact doctor`, `download`, `import-source`, `prepare`,
`validate-source`, `generate`, `status`, `pause`, `resume`, `export`, `verify`.
The generated runtime activation file selects the simulation and official export
interpreters; README includes all required commands. Output and runtime caches
remain ignored by Git. Wheel, sdist, checksums and post-tag verification are in `dist/`.

Limits: no public Git URL has been tested; CPU software-rendered collection and
all twenty tasks are not claimed. Blender free-body lid grasping is unsupported.
Official task success, local events, censored horizons and unsupported continuation
remain distinct. Source/asset licenses are separate from the inherited package MIT
license. This delivery performs no full 20-task production and no training.
