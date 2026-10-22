# CFACT

**从机器人示范生成经过物理执行的反事实动作树。**

[English](README.md)

CFACT 在 RoboCasa 仿真中，从同一个物理状态执行不同动作，记录实际产生的
轨迹、图像和任务结果，再从允许继续操作的子状态展开后续分支。
它适合构建动作条件预测数据，以及比较相同起点下不同动作造成的物理变化。

| 方法 | 候选来源 |
| --- | --- |
| Reference | 对齐到当前场景后的 Source 参考动作。 |
| G0 | 参考轨迹上的有界局部扰动。 |
| G1 | 兼容 donor 示范的运动与动作事件时序迁移。 |
| G2 | 根据生成协议构造几何目标变体或局部效果目标。 |

Source 是录制输入，Reference 是重新执行的候选。规划目标与执行结果分别保存。
仓库提供生成、读取和导出代码；示范数据、仿真资产、生成的数据和虚拟环境需要
单独下载或创建。

## 环境要求

推荐采集环境为 **Linux x86_64、Python 3.10、NVIDIA GPU 和 EGL**。
先用单 worker 跑通一个任务。这里的安装流程未验证 Windows、macOS 物理采集。

- 系统需要 Git、支持 `libx264` 的 FFmpeg、`ffprobe` 和可用的 NVIDIA/EGL 驱动。
- 仿真依赖的 Git revision 固定在 [runtime.json](src/cfact/resources/runtime.json)。
- Python 依赖约束见 [requirements/](requirements)。
- 仿真与 LeRobot 导出使用两个独立环境，避免依赖冲突。
- 内存、显存和存储需求随场景及数据规模变化，需要同时预留资产解压、环境、
  Source 和逐步 RGB 记录的空间。

## 安装

下面使用 Bash，从仓库根目录执行。Ubuntu 22.04 的系统安装示例如下；
其他发行版请安装对应软件包，并先准备 Python 3.10。

```bash
git clone https://github.com/amazing-old-man/CFACT.git
cd CFACT

sudo apt-get update
sudo apt-get install -y git python3.10 python3.10-venv ffmpeg libegl1 libgl1 libosmesa6
nvidia-smi

python3.10 scripts/bootstrap_sim.py --runtime-root .runtime --python python3.10
python3.10 scripts/bootstrap_export.py --runtime-root .runtime
source .runtime/activate.sh
```

安装器创建固定依赖的仿真环境和独立导出环境。每次打开新终端，都需要重新
`source .runtime/activate.sh`。如需放到其他磁盘，两个安装命令使用同一个
自定义 `--runtime-root`，然后加载该目录内的 `activate.sh`。

下载资产并检查真实渲染：

```bash
cfact download-assets --plan
cfact download-assets
mkdir -p output
cfact doctor --render --gpu 0 --output output/doctor.json
```

`--plan` 只列出资产清单。doctor 检查依赖版本、资产、FFmpeg、GPU 和小尺寸
EGL 实际渲染。检查通过后，再用下面的示例验证具体场景回放。

## 快速开始：生成一棵真实动作树

选择可写工作目录，可以位于其他磁盘：

```bash
export CFACT_WORKSPACE="$PWD/../CFACT-workspace"
mkdir -p "$CFACT_WORKSPACE"
export CFACT_WORKSPACE="$(cd "$CFACT_WORKSPACE" && pwd)"

cfact download --task OpenDrawer --split pretrain --source human \
  --output "$CFACT_WORKSPACE/sources/OpenDrawer.tar"

cfact quickstart \
  --source "$CFACT_WORKSPACE/sources/OpenDrawer.tar" \
  --output "$CFACT_WORKSPACE/quickstart/OpenDrawer" \
  --task OpenDrawer --gpu 0 --max-episodes 3
```

该流程会下载官方 Source，导入三条示范，准备仿真标注，进行真实物理采集，
并检查导出数据的官方读回。第一条示范作为 root，其余作为不重叠的 donors。
它使用仓库打包的 quick-start 协议，不默认启用所有实验协议。

输出目录必须是新目录。完成后查看其中的 `acceptance.json`，以及每个阶段
对应的日志。`PASS` 表示这个输入上的流程验收通过，不代表所有任务、场景都能成功。

数据由 [datasets.json](src/cfact/resources/datasets.json) 注册的官方来源下载，
保留 revision 和 SHA256 收据。如果来源要求授权，请先向提供方取得权限并执行
`hf auth login`。数据是否可下载取决于提供方的访问政策。

## 查看结果

```bash
export CFACT_RELEASE="$CFACT_WORKSPACE/quickstart/OpenDrawer/releases/pilot"
export CFACT_RAW="$CFACT_RELEASE/_work/collection"

cfact inspect --run "$CFACT_RELEASE"
cfact example --run "$CFACT_RELEASE" --task OpenDrawer
cfact view --run "$CFACT_RAW" --host 127.0.0.1 --port 8765
```

浏览器访问 `http://127.0.0.1:8765`，用 `Ctrl+C` 退出 viewer。
远程服务器可通过 SSH 转发端口。

原生数据保留父快照、候选计划、实际动作与状态、局部事件、官方成功信号、
子快照以及谱系。原生 RGB 配置包含三个 224 × 224 摄像头视角，包括腕部摄像头，
频率为 20 Hz。执行 `K` 个动作的连续段对应 `K+1` 个测量状态。

## 自定义任务和规模

完整的导入、准备、配置与采集命令见英文文档的
[Configure your own collection](README.md#configure-your-own-collection)。
入口包括：

| 步骤 | 命令或配置 |
| --- | --- |
| 导入官方数据 | `cfact import-source` |
| 检查回放字段 | `cfact validate-source` |
| 准备仿真标注 | `cfact prepare-source` |
| 选择任务、root、donor 和资源预算 | 复制并编辑 `configs/quickstart.yaml` |
| 检查请求及资源 | `cfact generate ... --dry-run` |
| 开始采集 | `cfact generate ...` |
| 查看、暂停、恢复 | `cfact status`、`cfact pause`、`cfact resume` |
| 导出和验证 | `cfact export`、`cfact verify` |

Replay 输入需要包含环境元数据、scene XML 和仿真状态，不能只提供视频与动作。
Root 与 donor 必须是不重叠的示范，且存在于对应 prepared HDF5 文件中。

此生命周期入口支持 1–6 个 worker；其他调度接口具有独立的协议与运行目录。
先通过单 worker 实测，再增加独立 root 数与并发。不要混用不同入口的状态、
恢复命令和输出目录。修改输入、代码或协议后，应创建新 run。

资源阈值是可调整的主机策略。应根据实际场景内存、显存和磁盘峰值设置预留，
而不是直接把示例阈值当成所有机器的最低要求。

## 独立读取数据

只读数据可以使用轻量环境，不需要安装模拟器：

```bash
python3 -m venv .venv-reader
source .venv-reader/bin/activate
python -m pip install -e .
```

```python
from cfact import TreeDataset, ReleaseDataset

native = TreeDataset("/path/to/native/collection")
print(native.roots())

release = ReleaseDataset("/path/to/published/release")
print(release.roots())
```

直接读取 LeRobot 导出时，使用独立导出环境中的官方 `LeRobotDataset`。
Episode 的边界应按官方元数据读取，不能仅凭 Parquet、MP4 文件名拆分。

## 支持范围和结果解释

任务注册表包括抽屉、机架、柜门、烤箱门、物体放置、咖啡杯、按钮、水龙头与
旋钮等操作。能否展开取决于具体场景、Source 回放、接触条件、实际关节绑定与
续接能力。递归效果编译器尚不支持自由体搅拌机盖子的抓取、释放流程。

需要区分：

- 物理采集完成，不等于官方任务成功。
- 局部目标达成，不等于完成整项任务。
- `horizon_censored` 表示记录时距内未观察到成功，不是确定失败。
- 共享祖先的多条分支，不是多条独立 Source，也不自动构成偏好标签。

历史实验材料保存在 [research/](research/README.md)，用于协议和来源追溯，
不是新电脑安装的必要步骤。其中服务器路径不应直接复制到新部署。

## 常见问题

| 问题 | 排查方式 |
| --- | --- |
| 缺模块或版本不一致 | 加载安装器生成的 `activate.sh`，使用固定依赖环境。 |
| EGL 无法渲染 | 检查 NVIDIA 驱动、EGL 库、物理 GPU 编号和 doctor 报告。 |
| 缺少资产 | 完成 `cfact download-assets`，确认 Source 包含 replay 元数据。 |
| 下载返回 401/403 | 取得提供方授权，并执行 `hf auth login`。 |
| Xet 连接失败 | 可尝试设置 `HF_HUB_DISABLE_XET=1` 后重试。 |
| 资源准入失败 | 查看报告，减少 worker，并依据主机实测调整资源预留。 |
| 输出已存在 | 复用已验证输入，或为新采集创建新目录。 |
| 某种分支缺失 | 检查兼容性拒绝、阶段和停止原因；不保证所有方法都可执行。 |

## 开发与许可证

开发环境可执行 `python -m pip install -e '.[test,dev]'`，随后运行
`python -m pytest`。物理集成验证还需要固定仿真环境、资产和真实回放输入；
只通过格式测试不代表物理采集通过。

代码采用 [MIT License](LICENSE)。上游代码、资产和数据各自遵守原许可证及
访问要求，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
引用元数据见 [CITATION.cff](CITATION.cff)。
