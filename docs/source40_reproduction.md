# 双任务 20+20 Source 与 native224 生成器复现

本文对应 `configs/source_train40_v1.json` 中冻结的身份请求。当前是否通过新冻结，
以本轮交接目录 `FROZEN_RELEASE.json`、`FINAL_STATUS.json` 为准。
只允许在用户下一次明确批准后运行正式 `generate`，本轮不运行正式生成或模型训练。

## 从独立 checkout 开始

```bash
git clone /mnt/ssd2/yanghc/CFACT /path/to/new/CFACT-checkout
cd /path/to/new/CFACT-checkout
# 本轮冻结 tag；完整 commit 和源码 SHA 在交接 FROZEN_RELEASE.json。
git checkout --detach v0.7.2.dev0-native224-source40
# 不应使用旧 v0.7.1.dev0 来生成 native224 数据。
unset PYTHONPATH
python3.10 scripts/bootstrap_sim.py --runtime-root .runtime --python /usr/bin/python3.10
python3.10 scripts/bootstrap_export.py --runtime-root .runtime
source .runtime/activate.sh
export HF_HUB_DISABLE_XET=1
python -m cfact download-assets --runtime-root .runtime
python -m cfact doctor --runtime-root .runtime --render --gpu 0 --output output/doctor40.json
```

新机器使用自己的实际 Git 来源、GPU 编号、下载凭证和足够的磁盘；未发布公共远程地址。
仿真依赖、资产和导出 reader 的固定版本由原有锁文件约束。完整安装步骤沿用 README
已实证的流程，不复制旧项目的源码、旧 prepared 或旧输出。

## 官方 Source 保存在 source/

```bash
export CFACT_SOURCE_DATA="$(python -c 'import json; print(json.load(open("configs/source_train40_v1.json"))["source_dir"])')"
mkdir -p "$CFACT_SOURCE_DATA/archives" "$CFACT_SOURCE_DATA/raw" "$CFACT_SOURCE_DATA/prepared"
python -m cfact download --task OpenDrawer --split pretrain --source human \
  --revision 522e4ffa3bb9d9729f79b241c65fc9e589f2659c \
  --output "$CFACT_SOURCE_DATA/archives/OpenDrawer.tar"
python -m cfact download --task CoffeeServeMug --split pretrain --source human \
  --revision 522e4ffa3bb9d9729f79b241c65fc9e589f2659c \
  --output "$CFACT_SOURCE_DATA/archives/CoffeeServeMug.tar"
```

若已有同 revision/SHA 的官方包和 receipt，复制到此新目录再用 `download --offline`
校验即可。只下载这两个 task 的完整归档，不通过连接器传输。SHA 分别为：

- OpenDrawer：`9ec49a3e752b7ba7d12efa04a7e9e75d80fabfec4c572bbbe3b23e0c5d033fa9`
- CoffeeServeMug：`7ee89a1caab69aa33d2d426f0ee57aace8368c626c9353b48a2b9a7d92510d9d`

下面读取冻结 JSON，构造实际 CLI 支持的逐 demo 参数，只导入并准备 20 roots+2 donors。
它不生成反事实树。已有有效输出时不要重复 import；先运行下一节的只读校验。

```bash
python - <<'PY'
import json, subprocess, sys
from pathlib import Path
m=json.load(open('configs/source_train40_v1.json'))
root=Path(m['source_dir'])
for task in m['tasks']:
    name=task['task']
    episodes=task['roots']+task['donors_by_group']['train_candidate']
    demos=['demo_'+str(int(ep[8:])) for ep in episodes]
    raw=root/'raw'/(name+'.hdf5')
    args=[sys.executable,'-m','cfact','import-source','--task',name,
          '--source',str(root/'archives'/(name+'.tar')),'--output',str(raw),
          '--extract-to',str(root/'extracted'/name)]
    for demo in demos: args+=['--demo',demo]
    subprocess.run(args,check=True)
    args=[sys.executable,'-m','cfact','prepare','--source',str(raw),
          '--output',str(root/'prepared'/(name+'.hdf5')),'--task',name,
          '--gpu','0','--config','configs/source_train40_v1.yaml']
    for demo in demos: args+=['--demo',demo]
    subprocess.run(args,check=True)
PY
```

每个新 Source 的 Parquet 时间/索引、状态有限值/时钟、XML/ep_meta 和三视频都应检查；
本轮逐行检查和解码证据见交接。数据损坏只能按已冻结备用顺序技术替换，需保存原请求、
理由、替代关系与时间。空树、未成功、Star 不足不允许换样本。

## 身份校验、技术回放与 dry-run

```bash
python scripts/verify_source_requests.py --manifest configs/source_train40_v1.json
python -m cfact generate --config configs/source_train40_v1.yaml \
  --workers 1 --gpu-devices 0 --dry-run
```

`verify_source_requests.py` 校验归档、冻结分组/身份、稀疏 demo 映射、三视频 SHA 和路径、
raw SHA、当前 preparer、prepared 所需 episodes 和控制器语义。它不执行物理。
`generate --dry-run` 只校验配置和资源，不验证 prepared 的完整内容；应组合使用两条命令。
真实技术回放入口如下，其中 demo 来自冻结 JSON，不以成功率筛选：

```bash
python -m cfact validate-source --source "$CFACT_SOURCE_DATA/raw/OpenDrawer.hdf5" \
  --task OpenDrawer --demo demo_8 --replay --gpu 0 \
  --config configs/source_train40_v1.yaml --output output/OpenDrawer_demo8_replay.json
```

原 Source 为 256 方形视频；模型接口显式采用 RGB/255 → ImageNet Normalize →
双线性 antialias 等比例 Resize224。CFACT 三相机直接真实 render224，原物理频率 20Hz、
动作与相机实体不变；没有把旧 224×398 帧拉伸成 native224。

## 仅在下一次用户明确批准后执行

```bash
python -m cfact generate --config configs/source_train40_v1.yaml \
  --workers 1 --gpu-devices 0 > output/formal40_generation.log
```

`source_dir/prepared` 是本配置实际读取路径。每 root 采用原公共生成接口和既有预算：
local160、fixed150、20Hz；当前公开入口保留每方法最多 1 候选、最多 2 decision parents、
既有前沿顺序及实际 child 递归。本清单扩大请求数，不扩大这些既有候选/前沿预算。
完整 Tree 是实际执行的全部节点、边、空根、状态和谱系，不能以结果替换请求。

先读取本次生成日志首行的确切 run 路径，避免误用旧输出：

```bash
export CFACT_RUN="$(python -c 'import json; print(json.loads(open("output/formal40_generation.log").readline())["run"])')"
python -m cfact status --run "$CFACT_RUN"
python -m cfact pause --run "$CFACT_RUN"
python -m cfact resume --run "$CFACT_RUN"
"$CFACT_EXPORT_PYTHON" -m cfact export --run "$CFACT_RUN" --media full --by-tree
"$CFACT_EXPORT_PYTHON" -m cfact verify --run "$CFACT_RUN"
```

最终结果在 `output/PhysicalAI-Robotics-Manipulation-Kitchen-Demos__<UTC>__<RunID>/`，
按任务、树独立定位。by_task 是官方 LeRobot 0.4.0 数据集；by_tree 是索引，
by_tree_exports 才是独立单树数据集。每个连续 local/fixed 段单独 episode，K actions/K+1
states 的末状态单独存终态映射；禁止拼接 restore 或 local/fixed 来制造 Star。

官方读回：

```bash
"$CFACT_EXPORT_PYTHON" - <<'PY'
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
run=Path(os.environ['CFACT_RUN'])
for task in ['OpenDrawer','CoffeeServeMug']:
    ds=LeRobotDataset('local/'+task,root=run/'lerobot/by_task'/task,
                     video_backend='pyav',download_videos=False)
    print(task,len(ds),ds.num_episodes,ds[0]['action'].shape)
PY
```

H3/F5 以 CFACT-Models 当前声明的数据合同只读验证：历史 t-10/t-5/t、每块 5×12=60
动作、真实未来 t+5、同父 canonical history。验证不用模型 forward、优化器或训练。
资源和时间估算见本轮交接，worker 必须根据未来实际资源准入重新确定，不默认并行扩张。


## 复现本轮数据接口验收

CFACT-Models 当前数据接口包含未提交文件，单独 checkout 其旧 HEAD 不足以复现。
本交接保留 `model_data_interface/` 的10个实际数据接口文件及 SHA；只有数据模块，
不含模型、训练器或 checkpoint。可以用这份副本做独立、只读验收：

```bash
export CFACT_INTERFACE_ROOT=/path/to/handoff/model_data_interface/src
export PYTHONPATH="$CFACT_INTERFACE_ROOT:$PWD/src"
"$CFACT_EXPORT_PYTHON" -m cfact_models.data.interface_audit \
  --run "$CFACT_RUN" --task OpenDrawer --task CoffeeServeMug \
  --output output/data_check/acquisition
"$CFACT_EXPORT_PYTHON" scripts/validate_h3f5_data.py \
  --run "$CFACT_RUN" --audit output/data_check/acquisition \
  --output output/data_check/h3f5
```

数据检查目录必须为新的 sibling 目录，不与 run 重叠。可复用已经完成的 acquisition
registry 做只读重查；不重跑物理生成。本轮实际使用原仓库的当前接口，副本字节 SHA
与其一致，副本不能解释为另一个模型实现。H3/F5 的每个 Star sample 返回 `branches`
列表，每个 branch 包含 pixels/action；interface_audit 的 layer 为 Root/Desc。
正式 train normalizer 未拟合，这些读取诊断使用固定 identity normalizer，不启动训练。

本机旧 editable 安装的分发 metadata 与当前 Source egg-info 并存，Smoke 的 dependencies
清单保留了旧 0.7.1 字符串；不能据该字符串推断实际执行代码。实际 0.7.2 模块路径、
完整 package SHA 与独立 wheel 身份在交接记录。独立 checkout bootstrap 安装当前版本；
正式启动前使用新的冻结校验脚本确认实际 imported package。
