# CFACT

**状态：验收进行中，尚未冻结，不是开源复现 READY。** 本仓库没有公开远程
地址。下文物理 quickstart 的最终验收结果以 `TEST_RESULTS.json` 为准；该文件
尚未生成时，不能把安装或合成夹具测试当作真实物理验收通过。

CFACT 从可回放的 RoboCasa Source 示范构建真实物理反事实动作树。在同一个
parent snapshot 上分别执行候选动作，保存实际 local child，再执行 fixed
continuation；递归仅从合格的实际 child 展开。Source 是录制输入，Reference
是实际执行的参考候选；CF-Root 和 CF-Desc 表示根层与递归层反事实，均不等于
把 Source 视频重命名。共享 ancestor 只执行一次，多个 lineage 可以引用它。

| 方法 | 保留的语义 |
| --- | --- |
| Reference | Source 操作片段的参考执行 |
| G0 | 同源片段的局部扰动，位置尺度 0.003 m |
| G1 | 独立 donor 示范的兼容片段迁移 |
| G2 | 几何目标候选，原比例 0.25 / 0.45 / 0.65 |

候选编译、控制器坐标解释、parent 恢复、local 160 步、fixed 150 步、官方
success、实际 child 与 frontier 顺序来自指定冻结输入。包版本
`0.7.1.dev0` 表示独立提取开发版，**不表示此前候选已是正式 0.7.1 release**。
`provenance.json` 记录来源内容 SHA256：
`7f58d15b12889c8fd275429d539c23d2ea05dd2c7317005dd8b6ab3170398d6f`。

支持边界是关节 articulation、button 和已实现的持物 placement 公共接口，
并非所有任务/场景都能形成非空树。`OpenBlenderLid`、`CloseBlenderLid` 的
free-body 抓取操作是 **UNSUPPORTED**，不会凭空给 Blender 添加关节。
`robocasa_20.yaml` 是预选覆盖请求，不是 20/20 支持承诺。

## 1. 软件、硬件与安装

物理采集使用 Linux x86_64、Python 3.10 和 NVIDIA EGL。CPU 可用于下载、
HDF5 导入、校验、Tree 读取、视频编码及官方 LeRobot 导出；本锁定后端尚未
验收 CPU 软件渲染采集。FFmpeg 必须有 `libx264` 和 `ffprobe`。

| 组件 | 精确版本或 commit |
| --- | --- |
| Python | 3.10；本机 3.10.12 |
| MuJoCo / NumPy / SciPy | 3.3.1 / 2.2.5 / 1.15.3 |
| RoboCasa | `0c81ff9c39b771afc330349cbbe50be9776a74fa` |
| robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` |
| MimicGen（RoboCasa fork） | `b89c3b5e02927ceb8599401cc01551a8da21fbf8` |
| robomimic | `271a76c2d55c8b0f94d3d589f26fcae0d47f64a1` |
| 官方 LeRobot writer/reader | 0.4.0，Dataset v3.0 |
| 导出 CPU Torch / torchvision / PyAV | 2.7.1 / 0.22.1 / 15.1.0 |

仿真与导出使用两个全新 venv，避免 Gymnasium 等依赖版本冲突。精确约束在
`requirements/simulation-lock.txt`、`requirements/export-lock.txt`。
下载的四个依赖源码和资产都保存在本仓库忽略的 `.runtime/`，不引用原工作区。

从空目录取得项目：公共地址尚未发布，`<UNPUBLISHED_CFACT_GIT_URL>` 是明确
占位符，不能执行，也未做外网 clone 验证。本机验收使用本地 Git clone：

```bash
git clone /mnt/ssd2/yanghc/CFACT CFACT-checkout
cd CFACT-checkout
```

其他机器将 clone 来源替换为实际取得的 CFACT Git 仓库。下列命令均从项目根
执行，不依赖 shell 中遗留的 `PYTHONPATH`：

```bash
unset PYTHONPATH
python3.10 scripts/bootstrap_sim.py --runtime-root .runtime --python /usr/bin/python3.10
python3.10 scripts/bootstrap_export.py --runtime-root .runtime
source .runtime/activate.sh
export HF_HUB_DISABLE_XET=1
cfact download-assets --runtime-root .runtime
cfact doctor --runtime-root .runtime --render --gpu 0 --output .cache/doctor.json
```

安装器下载并检查固定 Git revision，不应用实验室补丁。`activate.sh` 声明
本次安装的 `CFACT_RUNTIME_ROOT`、`CFACT_SOURCE_ROOT`、缓存位置、EGL 与
`CFACT_EXPORT_PYTHON`；这些均由安装位置生成。`HF_HUB_DISABLE_XET=1` 使用
官方 HTTP 传输，本机 Xet 连接失败时采用此方式，不使用个人隧道。

资产压缩包约 10.2 GiB，另需解压空间、两个 venv 和输出空间。三个本轮
Source tar 合计约 240 MiB。资源取决于场景和视频数量，不能把本机参数当作
所有机器的最低配置。本机约 188 GiB RAM、RTX PRO 6000 Blackwell 96 GiB
显存；验收配置保留 20 GiB RAM、每 worker 8192 MiB 预算（首轮 Source 回放峰值约 4774 MiB）、GPU 10 GiB
余量、磁盘 50 GiB 余量及 85°C 上限。应根据新主机实测峰值至少加 1.2 倍
余量调整 YAML，不能为了跑通而绕过保护。生成前和 worker 调度均检查资源；
其他用户的进程不被终止。

```bash
python --version
ffmpeg -version
ffmpeg -hide_banner -encoders
ffprobe -version
df -h .
free -h
nvidia-smi
cfact generate --config configs/quickstart.yaml --workers 1 --gpu-devices 0 --dry-run
```

`--dry-run` 只显示冻结请求和当前资源，不创建树，不生成假物理数据。

## 2. 官方 Source：下载、extras 与转换

固定官方代码可单独查看；实际运行用安装器创建的相同 commit：

```bash
git clone --filter=blob:none --no-checkout https://github.com/robocasa/robocasa.git .cache/official-robocasa
git -C .cache/official-robocasa checkout --detach 0c81ff9c39b771afc330349cbbe50be9776a74fa
```

该 commit 的官方命令为：

```bash
python -m robocasa.scripts.download_datasets --tasks OpenDrawer --split pretrain --source human
```

它会询问 `y/n`。本机实际执行后 **退出码 1 / HTTP 401**：该版本命令指向
`nvidia/PhysicalAI-Robotics-Kitchen-Sim-Demos`。不要把此路径描述为已公开下载
成功。默认复现改用同一官方 registry 路径在仍公开可取的 NVIDIA 官方
`nvidia/PhysicalAI-Robotics-Manipulation-Kitchen-Demos` 固定数据快照；CFACT
通过官方 Hugging Face 客户端下载，记录 repo、revision、文件名和内容 SHA。
需要授权的版本请在官方页面取得权限并使用 `hf auth login`，不要把令牌写入
命令、配置、Git 或报告。

```bash
cfact download --task OpenDrawer --split pretrain --source human \
  --revision 522e4ffa3bb9d9729f79b241c65fc9e589f2659c \
  --output .cache/sources/OpenDrawer.tar
cfact download --task OpenDrawer --split pretrain --source human \
  --output .cache/sources/OpenDrawer.tar --offline
cfact import-source --source .cache/sources/OpenDrawer.tar \
  --output .cache/sources/OpenDrawer.hdf5 --task OpenDrawer --max-episodes 3
cfact validate-source --source .cache/sources/OpenDrawer.hdf5 --task OpenDrawer \
  --output .cache/sources/OpenDrawer.validation.json
```

`download --offline` 校验已存在 tar 和 receipt 的身份及 SHA，不访问网络；
损坏或缺失时失败。`import-source` 只写新文件，拒绝覆盖；已导入时直接使用
原输出并重新 validate，不能用新 demo 暗中替换失败输入。

默认包的路径是 `pretrain/atomic/OpenDrawer/20250819/lerobot.tar`，SHA256 为
`9ec49a3e752b7ba7d12efa04a7e9e75d80fabfec4c572bbbe3b23e0c5d033fa9`。
导入后的 `demo_0/1/2` 对应原 `episode_000000/000001/000002`；root 固定
`demo_0`，独立 donors 固定 `demo_1`、`demo_2`。

导入器会解包到 `.cache/sources/OpenDrawer.source/` 并检查：

| 官方输入字段 | 检查及内部表示 |
| --- | --- |
| `meta/info.json`, `meta/modality.json` | 源 LeRobot v2.1、20 FPS、12 维动作切片与 gripper/base/control mode |
| `extras/dataset_meta.json` | 环境 task、机器人、controller 配置；缺失时拒绝 |
| `extras/episode_*/states.npz` | 真实 MuJoCo 状态、有限数、与动作条数一致；进入 `data/demo_N/states` |
| `extras/episode_*/model.xml.gz` | 真实 scene XML；进入 `model_file` 属性，资产由固定源码/下载身份解析 |
| `extras/episode_*/ep_meta.json` | layout/style/fixture/object 等元数据；进入 `ep_meta` 属性 |
| `data/*/episode_*.parquet` | 实际动作和原 reward/done；仅按 modality 重排动作，保留原始数值 |
| `videos/*/...mp4` | 原示范视频来源记录；不用于推断或补造状态 |

导入 receipt 保存每个消耗文件的 SHA。HDF5 中 `data.attrs.env_args` 和
`cfact_source_provenance` 保留环境与官方快照身份。单有普通训练视频/动作、
或者任意 HDF5 扩展名，不代表能恢复 MuJoCo 状态。

| RoboCasa 来源 | 导入格式 | 必需信息 | 已实测范围 |
| --- | --- | --- | --- |
| NVIDIA 官方快照 `522e4ff…` | LeRobot v2.1 tar/目录 → raw HDF5 | 上表 replay extras、Parquet action、controller、XML | 三个指定 task 下载与前三 demo 导入已通过；物理结果见验收表 |
| 相同 schema 的本地 RoboCasa 数据 | raw HDF5 | `data/env_args`、每 demo actions/states/model_file/ep_meta | schema 检查；不能代替公开来源验收 |
| 普通训练 LeRobot（无 extras） | 不支持 | 缺少状态、XML 或环境元数据 | 明确拒绝，不补造 |
| 任意源 LeRobot v3 | 未声明支持导入 | 必须按 episode offset 解析，不能按文件名猜 | 当前默认 Source 是上述 v2.1；目标导出才是 v3 |

可选本地导入（文件路径由使用者明确提供）：

```bash
cfact import-source --source /path/to/your/robocasa.hdf5 \
  --output .cache/sources/local.hdf5 --task OpenDrawer
cfact validate-source --source .cache/sources/local.hdf5 --task OpenDrawer
```

本地 HDF5 必须通过同样的 schema、动作、时钟和身份检查。它不自动获得官方
下载身份；使用该输入时配置中的 `source_dataset` 应与实际 provenance 一致。

## 3. Source 回放、准备和单 worker Quickstart

公开 Source 的首个存储状态间隔为 0.10 s，后续为 0.05 s。Source 未存储
控制器全部内部状态，`env_args` 中未写 `control_freq` 时使用固定 RoboCasa
构造器的 20 Hz 默认值。校验报告保留这些事实；不补插动作、状态或帧，不
宣称原录制轨迹或跨机器图像字节一致。

以下真实 replay 校验恢复原始初态，比较 qpos/qvel/time，验证同 snapshot
独立执行动作的重复性，并逐步记录实际 replay 与存储状态的差异及官方
success；`SCHEMA_VALID_REPLAY_NOT_TESTED` 仅表示结构有效，不能算物理 PASS。

```bash
cfact validate-source --source .cache/sources/OpenDrawer.hdf5 --task OpenDrawer \
  --replay --demo demo_0 --gpu 0 --output .cache/sources/OpenDrawer.replay.json
cfact prepare --source .cache/sources/OpenDrawer.hdf5 \
  --output .cache/prepared/OpenDrawer.hdf5 --task OpenDrawer \
  --demo demo_0 --demo demo_1 --demo demo_2 --gpu 0 --config configs/quickstart.yaml
cfact generate --config configs/quickstart.yaml --prepared-dir .cache/prepared \
  --workers 1 --gpu-devices 0 > .cache/quickstart-generation.log
```

`prepare` 使用固定 MimicGen 及冻结 CFACT adapter 顺序重建注释，保留
controller/codec/preparer/raw 文件身份；prepared schema 是
`cfact.prepared.v2`。重复准备只复用身份一致的文件。中断的 `.partial` 保留
诊断信息，显式 `prepare --resume` 才隔离旧的未完成文件后重新准备。

`generate` 首行 JSON 输出本次 `run` 路径。下面从这次命令自己的日志读取
**确切路径**，不会猜测“最新目录”：

```bash
export CFACT_RUN="$(python -c 'import json; print(json.loads(open(".cache/quickstart-generation.log").readline())["run"])')"
cfact status --run "$CFACT_RUN"
"$CFACT_EXPORT_PYTHON" -m cfact export --run "$CFACT_RUN" --media full --by-tree
"$CFACT_EXPORT_PYTHON" -m cfact verify --run "$CFACT_RUN"
cfact resume --run "$CFACT_RUN"
```

默认 export 是 full-media；`--by-tree` 同时生成完整的独立单树 LeRobot 数据集。`verify`
必须核对非空 edge、官方 reader 的所有 action/state/time 行和视频。
生成退出码 0 仍可能是合法零 edge root；只有实际非空树和读取证据才满足
Quickstart 的成功条件。被拒绝的 Source、不支持操作、失败原因和 root
身份必须保留，不能自动更换 demo 追求非空或成功。

## 4. 三种代表操作与 20 任务配置

有限代表验收选择 `OpenDrawer`、`PickPlaceCounterToCabinet`、
`TurnOnSinkFaucet`。后两者使用相同的 `download → import-source →
validate-source --replay → prepare` 步骤，把 `--task` 和文件名替换为实际
task；仍选择 `pretrain/human` 固定快照及原始 episode 0/1/2，不搜索成功 demo。
`configs/representative.yaml` 声明每任务 1 root、2 donors、相同 seed，
最多两层决策与每 parent 每方法 1 个实际候选预算。

`configs/robocasa_20.yaml` 声明预选 **20 任务 × 每任务 3 roots = 60 请求**：
每任务 roots 为 `demo_0/1/2`，donors 为 `demo_3/4`；须先下载、导入至少
5 个 demo 并准备这些明确 ID。默认 1 worker，seed 20260922，full-media。
本次交付不会执行全量 60-root 生产。覆盖应分别统计请求数、可回放数、非空
root/task、发生递归的 root 和真实 edge，不能以 episode 数冒充 edge。

```bash
cfact generate --config configs/robocasa_20.yaml --workers 1 --gpu-devices 0 --dry-run
cfact generate --config configs/robocasa_20.yaml --prepared-dir .cache/prepared \
  --task OpenDrawer --max-roots-per-task 1 --workers 1 --gpu-devices 0 --dry-run
```

实际运行时去掉 `--dry-run` 会启动真实采集，请按所需任务范围执行。可用
`--root-demo TASK=demo_N`、`--donor-demo TASK=demo_N` 显式覆盖选择，
`--seed` 固定种子，`--max-control-steps` 设置每 root/路径控制步预算。
这些只在创建新 run 前确定；resume 不接受重新选样或新配置。
`--gpu-devices` 接受物理 GPU 索引的逗号分隔列表；没有安全可用设备时报告
`NOT_RUN_RESOURCE_BLOCKED`，不启动采集。YAML 中 resource 是可配置的主机
验收策略，不是算法要求。

## 5. 输出、官方读取和完整视频

目录严格在本项目 `output/` 下，由真实 Source dataset 身份、安全 slug、
创建时 UTC 和随机唯一 run ID 组成：

```text
output/<dataset>__<YYYYMMDDTHHMMSSZ>__<run_id>/
  manifest.json, source_manifest.json, config.yaml
  trees/<task>/<tree_id>/tree.json, nodes/, edges/, lineage.json
  lerobot/by_task/<task>/meta/, data/, videos/
  lerobot/by_tree/<task>/<tree_id>/index.json
  lerobot/by_tree_exports/<task>/<tree_id>/meta/, data/, videos/   # --by-tree
  media/replay_prefix/, edges/, lineages/
  mappings/episode_tree_map.parquet, terminal_observation_map.parquet
           edge_segment_episode_lineage.json, lineage_paths.jsonl
  reports/official_readback.json, full_media.json, resource_admission.json
  runtime/plan/, collection.log, collection/_work/collection/
```

`manifest.json` 在初始化后不可变，记录 Git commit、包代码 hash、来源
hash、依赖版本、任务与 seed；可变状态另存 `runtime/state.json`。
native runtime 保留 parent/child backend snapshot、每步实际动作与状态、
候选、官方 first/ever/last success、local/fixed 和停止原因。

任务级官方 reader 示例，必须在导出环境执行：

```bash
"$CFACT_EXPORT_PYTHON" - <<'PY'
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = Path(os.environ["CFACT_RUN"])
dataset = LeRobotDataset("local/OpenDrawer", root=root / "lerobot/by_task/OpenDrawer",
                         video_backend="pyav", download_videos=False)
print(len(dataset), dataset.num_episodes, dataset[0]["action"])
PY
```

LeRobot v3 的 Parquet/MP4 文件可共享多个 episode，必须按官方 meta offset
读取，不能按文件夹或文件名拆分。每个 local/fixed 连续段是一条 episode，
行 `t` 对应动作前状态 `s[t]` 与实际动作 `a[t]`；`s[K]` 单独保存在终态映射。
零动作段没有动作行，但保留终态记录。

树目录索引本身**不是独立 LeRobot 数据集**：

```bash
"$CFACT_EXPORT_PYTHON" - <<'PY'
import json, os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
root = Path(os.environ["CFACT_RUN"])
index_path = next((root / "lerobot/by_tree/OpenDrawer").glob("*/index.json"))
index = json.loads(index_path.read_text())
print(index["tree_id"], len(index["edges"]), index["episodes"])
tree_root = root / "lerobot/by_tree_exports" / index["task"] / index["tree_id"]
dataset = LeRobotDataset("local/" + index["tree_id"], root=tree_root,
                         video_backend="pyav", download_videos=False)
print(len(dataset), dataset.num_episodes)
PY
```

首次 export 未指定 `--by-tree` 时，可以在同 run 再执行带 `--by-tree` 的
export 添加独立单树数据集，不重新采集物理数据。`--media playlist` 是明确的
轻量模式，仅保存共享片段和 playlist，不代表完整 lineage MP4；Quickstart
与正式演示必须用 `--media full`。

每个 leaf 的完整 MP4 在 `media/lineages/`，相邻 receipt 记录帧区间和 SHA。
replay prefix 来自本次真实 replay 历史，不是原 Source 视频的替代品。
`reports/full_media.json` 记录所有 leaf 数量、物化数量与额外磁盘字节数。
公共 ancestor 在多个完整视频中出现不会被统计成多次独立物理执行。

## 6. 暂停、恢复和诊断

```bash
cfact pause --run "$CFACT_RUN"
cfact status --run "$CFACT_RUN"
cfact resume --run "$CFACT_RUN"
```

pause 请求停止分派并排空当前 root。resume 始终写回原 run；完整 root 不
重复采集，不创建新的日期目录。配置、源码或 prepared SHA 变化时拒绝
resume。未完成 root 的旧数据保留在隔离目录，显式 resume 才重试同一个
root，且不计作新的独立请求。磁盘耗尽/温度/RAM/显存保护触发时保留诊断，
恢复安全资源后再 resume；不要删除不明 `.partial` 或伪造 completed 标记。

优先查看 `runtime/collection.log`、native runtime 的 `runtime/logs/`、
`reports/resource_admission.json` 与每 root 的 `root_result.json`。
`cfact verify` 核对 manifest、导出文件 hash 和官方读取。完整 export 的
重复运行校验已有内容，避免重新编码或重复 episode。

官方 task_success、candidate 局部 event、停止原因是不同字段：
`HORIZON_CENSORED`、`NO_SUPPORTED_CONTINUATION` 不是已确认最终任务失败；
局部 event 也不等于官方成功。跨机器浮点与图像不保证字节相同，但版本、
输入身份、parent/child 关系、动作和成功语义必须可以核验。

## 7. 许可证、测试和未发布状态

保留来源已附带的 MIT 许可证和原版权行，未擅自给依赖或数据重新授权。
四个固定源码仓库分别保留其上游许可证；MimicGen 使用 NVIDIA License。
下载的数据和资产有独立条款，见其官方仓库及 `THIRD_PARTY_NOTICES.md`。
本项目不打包依赖源码、原始数据、资产、缓存、venv 或生成输出，不自动向
GitHub/Hugging Face 推送。引用请用 `CITATION.cff` 并附确切 commit、tag
和来源 hash；没有虚构论文 DOI 或公开地址。

CPU 契约测试、固定仿真依赖测试和官方 API 合成夹具测试与真实物理 E2E
分别报告。合成夹具视频不计作物理数据。T0–T5 全部实证通过、许可/扫描无
阻塞之后才可以创建 release tag；本地完整复现成功但无公开地址时只能称
`LOCAL_RELEASE_READY`。当前尚未完成这些门槛，不能声称 READY。
