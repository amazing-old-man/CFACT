from pathlib import Path
R=Path('/mnt/ssd1/yanghc/CFACT_Research');D=R/'docs';D.mkdir(exist_ok=True)
texts={
'CFACT_V2_METHOD_SPEC_ZH.md':'''# CFACT v2 方法与证据边界

本轮在独立 SSD1 checkout 中沿用真实 RoboCasa/MuJoCo、MimicGen waypoint、Reference/G0/G1/G2、完整快照和 ControlStepLedger。候选的 EffectIntent 仅为提议，实测 qpos、物体位置、fixture activation、接触与官方 success 才是效果事实。研究验收状态和最后选择见 ../05_validation/FINAL_RESEARCH_RESULTS_ZH.md；未达门槛的开发数据不可作为正式结果。

根来自预先锁定的 demo_0/demo_1，donors 为 demo_3/demo_4。初始五任务纵向切片只用 demo_0；不同方法重复同一 Source 不增加独立根数。当前 Source 导入 20/20 与真实递归覆盖分别统计，分母固定为 20。

根候选先冻结再执行；每个 sibling 恢复完整 physics/controller/gripper/RNG/fixture 快照。local 最后一步立即保存 actual child，fixed continuation 单独记录。下一层从 local child 快照恢复，重新读取当前 shadow observation 的 EE/object/fixture 几何。K 条实际控制对应 K+1 个实际状态。shadow observation 不执行 env.step；无任何 horizon 静态补齐。

事件锚来自已经执行的 local trace。阶段包括 APPROACH、CONTACT_ESTABLISHED、PARTIAL_TASK_EFFECT、HELD_TRANSPORT、PLACE_READY、RELEASE_AND_RETREAT、RECOVERABLE_MISS、TERMINAL_SUCCESS、UNSUPPORTED。持有验证要求任务接触、闭合夹爪和连续相对变换稳定；PLACE_READY 另需有界支持目标、实测目标距离 ≤3cm、最后四帧物体速度 ≤5cm/s。此标签仍不保证释放后支持。RECOVERABLE_MISS 只可在验证恢复几何后使用，本轮没有验证的恢复实例。自由体 Blender lid 没有已验证编译器，明确不支持，不创造关节。

三轮研究假设分别为实测效果多样性 frontier、Source/真实几何约束 G0 尺度、验证阶段驱动的通用 frontier。每轮独立 commit、开发物理收据、预注册端点和接受/回滚决定。第三轮按 depth/pool/step 限制循环真实 child；旧固定深度路径作为对照。Source suffix 耗尽后的独立新接近/恢复编译器本轮尚未完成，必须记录 NO_SUPPORTED_CONTINUATION，不宣称覆盖了该能力。

物理下限在方法物理实验前登记：slide 5mm，hinge 0.05rad，物体平移 10mm，activation 一次布尔改变。四个代表任务的完整 A/B/A 校准用于已验证任务族；不是所有 20 任务/所有 horizon 的噪声保证。0.25/0.5/1/2/4s 和首次真实接触两种对齐，全部 sibling pair 均保存，缺帧/未观察事件独立删失。全图 RGB 差异不等于目标 ROI 可观测；本轮 ROI/SSIM 未验证。控制器目标跟踪与真实任务效果分别测量。

主要端点是有效任务效果 sibling pair / 全部物理成本与非根状态覆盖 / 全部物理成本。独立 Source root 为统计单位，重复 horizon/边不是独立样本。正式 train/dev/test 场景隔离尚未建立，禁止跨场景泛化或新模型收益声称。
''',
'CFACT_V2_ARCHITECTURE.md':'''# 运行架构

`cfact.collection.worker` 建立 Source 根和唯一 live MuJoCo scene；`execution.snapshot` 维护完整可恢复 backend 状态，`execution.current` 从当前 shadow state 获取几何；`generation.context/options` 编译全部候选；`generation.task_adapters/event_anchor` 从 local 测量生成效果与阶段；`generation.frontier` 为通用预算队列。父级实际 trace 和完整快照只记录一次，由 lineage 引用连接后续节点。

`execution.history.ControlStepLedger` 捕获每步动作、状态、三相机原生 224 RGB、相机参数和 hashes。`storage.tree` 生成可移植树；直接 worker 的 clean camera 文件由 `storage.ledger_media` 从同一实际 ledger 帧按 integration/state/time/scene/camera/image hash 精确匹配后以无损 PNG 写出，不重渲染、不推进 physics。对缺帧或部分 spool 拒绝发布，原始 ledger NPZ 不修改。

`export.dataset` 使用固定 LeRobot 0.4.0 写出 Flat 数据；`data.release.ReleaseDataset` 用于发布视图，`data.reader.TreeDataset` 用于 raw data/ 视图。terminal map 显式保留 K+1 终态，所有 episode/edge 可双向追溯。发布哈希验证与研究准入独立：技术读回 PASS 不等于多数任务/H1/H2/训练 PASS。

新 `artifact_root/allowed_write_root/quota_root` 校验绝对路径及真实目标文件系统、symlink 逃逸、free bytes/inodes 和新增配额。外部 `scripts/research_run.py` 每 3 秒检查 RAM/GPU/磁盘、每 15 秒更新全工作区成本；只停止自己启动的进程组。所有新缓存与临时文件为 SSD1 实际路径。代码、数据和阈值每个 run 单独冻结；不混合作为 formal run。
''',
'TASK_ADAPTER_CONTRACTS.md':'''# 任务族合同

| 族 | 实际能力条件 | 实测主要效果 | 续接条件与限制 |
|---|---|---|---|
| articulated_contact | 唯一实际 slide/hinge joint，真实 axis/anchor/range，明确 task q target | 关节 m 或 rad，qvel，handle/EE/contact | 当前几何重锚，保留 controller/source cursor；source suffix 缺失不假造 |
| held_object_placement | 实际接触、闭合夹爪、连续相对变换稳定 | 物体实测世界位移，接触与官方 success | 有界 fixture/receptacle goal 才能判 PLACE_READY，释放须看到 opening commands |
| button_activation | 原生 fixture activation 可读且每帧状态 MEASURED | 布尔 turned_on 差异 | 初始已开启可能没有 task-effect 分叉，机械臂差异不能替代 |
| free_body_lid | 本轮无验证过的抓取/释放编译器 | UNSUPPORTED | 不添加假 hinge，保留拒绝理由与真实 Source 成本 |

受限接触不等于禁止所有碰撞。原始 motion validation/contact guard 记录仍保留；未验证的 attached-object collision/safety 不能标成 PASS。阶段分类器所用 3mm 关系漂移和 3cm/5cm/s 放置就绪阈值为操作条件，不能替代冻结的效果差异阈值。

20 项任务的精确 SOURCE_PRESENT/IMPORTABLE/REPLAYABLE/CANDIDATE/RECURSION/PHYSICS 状态由 ../05_validation/TASK20_COVERAGE.csv 提供。导入 prepare 的 reset_to 不产生新的机器人控制步；实际 replay/env.step 证据必须另外标记。
''',
'BASELINE_AND_ABLATION_PROTOCOL.md':'''# 开发基线与正式准入

五组为 SOURCE_ONLY、GENERIC_CF_SINGLE、GENERIC_CF_OFFDEMO、CFACT_EFFECT_SINGLE、CFACT_EFFECT_RECURSIVE。同一实际 Source/root/scene/seed/controller/三相机/官方 success 与上限匹配。Source-only 只读已存在示范，新增物理成本 0，既有 K 状态缺失 terminal 不补造 K+1。

强 Generic 包含 stop、0.5/1.5 缩放、反向、XY 轴交换、平移/旋转噪声、打开夹爪、10-step 延迟；动作范围和 base/control mode 保持原合同。Source 结束后的 stop 是明确 source index=-1 的实际 env.step，不是静态状态补齐。80 local +80 fixed 的所有真实步、前缀和失败均计费。Off-demo 6/8 起点来自 Source、2/8 来自固定 10-step 扰动的真实子状态；扰动前缀只计成本、不计递归方法收益。

当前阶段为开发单根对照，参见 ../02_baselines/BASELINE_DEVELOPMENT_PRE_REGISTRATION.json。实际预算消耗分别报告，不把相同上限说成实际步数/wall 完全相等。单根无法给跨独立根置信区间。三轮方法对照属于预注册的开发假设评估；与未执行的 G0/G1/G2-only、无事件/无恢复消融分别记录 NOT_TESTED。

正式准入需要完整真实快照/谱系/RGB/官方读回、至少 15/20 的两个独立根工程覆盖、正式 Source/场景 split、固定主端点与预算。任一缺失则 FORMAL_NOT_ADMITTED / TRAINING_NOT_STARTED。正式集从未冻结或查看 final test outcome；开发数据不能重命名为 formal。
''',
'PAPER_EXPERIMENT_TABLES.md':'''# 论文证据表索引

可复核的实际数值由 ../05_validation/ 下 TASK20_COVERAGE、EFFECT_DIVERSITY_BY_TASK、RECURSION_DEPTH_AND_FRONTIER、COST_NORMALIZED_BASELINES、ABLATION_RESULTS CSV 与逐对 Parquet 提供；每项带方法版本、实际控制步和独立 Source 单位。最终研究结论见 FINAL_RESEARCH_RESULTS_ZH.md。

表应分别列：20 项真实能力边界；根层/非根层有效效果 pair；效果状态 bins；完整 A/B/A 噪声与冻结下限；真实深度/未续接理由；全部成本包括技术失败和扰动前缀；五方案的实际执行状态；每轮接受或回滚；原生树与 Flat 完整读回。不能把 NOT_TESTED 写成 0，也不对单根重复 pair 给显著性。

OPENWAM 对比仅描述已存在同状态不同动作和 off-demo 起点机制，来源 https://openwam.stanford.edu/ 、https://arxiv.org/abs/2610.07922 。CFACT 新颖性须由 H1/H2 数据支持。已有 DINO-WM/LeWM 原报告的证据在 ../00_audit/LEGACY_MODEL_EVIDENCE.md；本轮没有新模型结果。
''',
'REPRODUCTION_GUIDE.md':'''# SSD1 复现与审计

原 SSD2/HDD 均只读。新 repo 位于 `/mnt/ssd1/yanghc/CFACT_Research/repo`，外层脚本/报告同属 CFACT_Research。manifest 保存原 main、实际最新 base、所有方法 commit、Source SHA/upstream revisions 和种子。runtime 为旧只读固定环境；新所有 HF/Torch/XDG/TMP/FFmpeg/OpenGL/Python 缓存均为 SSD1。

```bash
cd /mnt/ssd1/yanghc/CFACT_Research
source runtime/activate.sh
# 无新训练；每个命令必须用新的、明确位于 SSD1 的输出目录
/mnt/ssd2/yanghc/CFACT/.runtime/venv/bin/python -B scripts/quality_audit.py --input 01_development/DEV_ITERATION_01/physics --output 05_validation/recheck_iteration01
HF_HUB_OFFLINE=1 /mnt/ssd2/yanghc/CFACT/.runtime/export-venv/bin/python -B scripts/verify_development_release.py 04_datasets/development_iteration01
/mnt/ssd2/yanghc/CFACT/.runtime/venv/bin/python -B scripts/protection_scan.py --phase after
```

物理重跑须先看 BUDGET_POLICY、对应 CODE_FREEZE/HYPOTHESIS/spec/plan，并创建新 run/receipt 路径，不覆盖已有完整/失败收据。将 CFACT_SOURCE_ROOT 与 PYTHONPATH 设为相应 SSD1 frozen worktree，外层 research_run.py 用当前 guard，只有资源闸门 PASS 才启动一个 scene worker。CPU fallback 必须显式 MUJOCO_GL=osmesa、CUDA_VISIBLE_DEVICES=''、gpu=-1，且单独记录 backend，不能混成等 GPU 时间实验。

开发 release 在 `04_datasets/development_iteration01`：FULL_READBACK.json 的 all_rows_read/official/native/哈希验证为真实结果；scope 为 DEVELOPMENT_DIAGNOSTIC_RELEASE，training_admitted=false。原始物理记录、未采用的方法、失败导出日志均保留。回滚提交不会删除方法历史或重写其结果。最终 patch/bundle 仅供审阅与手动迁移，本轮不 push/merge 原仓库。
'''}
for name,text in texts.items():(D/name).write_text(text)
print('WROTE_METHOD_DOCUMENTS',len(texts))
