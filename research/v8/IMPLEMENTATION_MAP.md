# CFACT 递归生成器：实现与实际证据

本文件中的路径相对于本 round。正式生成使用不可变的 `runtime/v8` 物理 worker；调度和导出分别使用 `runtime/orchestrator_v9`、`runtime/publisher_v6`。三者的哈希及测试记录见 `manifests/DEVELOPMENT_V8_FREEZE.json`、`manifests/ORCHESTRATOR_V9_FREEZE.json` 和 `manifests/PUBLISHER_V6_FREEZE.json`。历史 v3 开发结果与 v8 正式结果分别统计。

| 实际能力 | 可运行入口 | 证据和边界 |
| --- | --- | --- |
| 在执行前提出 EffectGoal，按当前真实状态、阶段、接触与任务目标编译候选 | `runtime/v8/src/cfact/generation/stage_compiler.py` 的 `compile_goal`；`runtime/v8/src/cfact/collection/worker.py` | 每个 parent 的 `effect_goals.json` 和候选计划先于边的实际执行保存；任务身份、拒绝原因与测得效果分开保存。真实 v8 树见 `runs/formal_v8`，完整审计完成后见 `derived/formal_v8_audit/ROOT_AUDIT.json`。 |
| 从真实 child snapshot 继续展开多候选树，复用完整祖先历史 | `runtime/v8/src/cfact/collection/worker.py` 的 `_expand_child_pool`、实际前沿与执行模块 | `derived/source_exhaustion_v8_audit/ROOT_AUDIT.json` 全帧原生审计 PASS；正式首根 3 层、12 条独立边和 10 条共享祖先的叶谱系见 `manifests/FORMAL_V8_INITIAL_EXPORT_VERIFIED.json`。 |
| 当剩余 Source 后缀与当前接触入口不兼容时，生成当前状态 primitive | `runtime/v8/src/cfact/generation/stage_compiler.py` 与 `runtime/v8/src/cfact/collection/worker.py` | `manifests/CURRENT_STATE_PRIMITIVE_V8_PHYSICAL_PROOF.json`：OpenCabinet 等 4 个诊断根，21 条真实非根 primitive 边。原 Source 完整保留，后缀仍 AVAILABLE，但其入口几何超出事前声明的接触邻域；不声称 Source 自然耗尽或对一切策略不可行。 |
| 同父恢复、噪声校准和固定真实时距比较 | `runtime/v8/src/cfact/execution/snapshot.py`、`runtime/v8/src/cfact/collection/recursive_calibration.py`、`runtime/v8/src/cfact/collection/recursive_audit.py` | `derived/calibrations` 的 A/B/A；`config/EFFECT_COMPARISON_POLICY_V4.json` 冻结各族时距和阈值。审计要求非根 parent 的两条兄弟边在同一真实时刻有校准后的效果差异，才判 L3。 |
| 有配额、看门狗、失败记录、暂停/恢复和 GPU/CPU 后端身份的正式调度 | `runtime/orchestrator_v9/src/cfact/collection/round_runner.py`；`./cfact recursive generate/status/pause/resume` | `config/FORMAL_V8_FLEET8_QUEUE.json` 事前冻结 16×8=128 个独立 Source 请求；`logs/SCHEDULER_STATUS.json` 和 `logs/receipts` 记实时进度及每个 worker 的实际成本。最多 8 worker、每 GPU 最多 2；CPU 使用实测 OSMesa。GPU 92°C 为截止温度。 |
| 保持 SSD1 写入隔离的 Source 准备 | `runtime/v8/src/cfact/sources/isolated_prepare.py`、`runtime/v8/src/cfact/runtime/readonly_assets.py` | `manifests/SOURCE_AND_SPLITS.json` 与 `manifests/FORMAL_GENERATION_V8_INPUTS.json` 冻结来源及顺序；原始 Source 不改。验证和 final 各 20 个保留根见 `manifests/HELDOUT_V8_EVALUATION_PLAN.json`，候选尚未执行时即冻结。 |
| 原生树、LeRobot 官方行及 review 视频的双向发布与读回 | `runtime/publisher_v6/src/cfact/export/recursive.py`、`runtime/publisher_v6/src/cfact/export/dataset.py` | `datasets/formal_v8_initial_verified_root` 已完成：1 个 Source、12 条物理边、485 个实际动作行、13 个官方 episode，逐行官方读回、原生状态/RGB 全帧和 13 个 review 视频解码 PASS。128 个正式请求及原生审计已完成。publisher v6 已在实际发生缓存冲突的 CloseOven/demo_6 整根通过 457 行官方读回和 13 个视频完整解码；全量发布位于 datasets/formal_v8_complete_publisher6，release.json 为 COMPLETE/PASS，FULL_READBACK.json 已 PASS。历史首根发布使用 v5 保留。 |
| 独立验证、最终测试及公平对照 | `runtime/post_pipeline_v8_resource_r2.py`、`runtime/finalize_v8_parallel_r2.py`、`scripts/generic_baselines.py`、冻结的 baseline queue | 验证与 final 各 20 个根及全帧审计已完成，分别有 14 和 16 个 L3；状态见 `logs/POST_FORMAL_V8_STATUS.json`。公平对照计划见 `manifests/FAIR_BASELINES_V8_PLAN.json`，五种方法、两个相同 Source 的执行已完成，实际控制步预算曲线见 reports/FAIR_BASELINE_BUDGET_CURVES.csv；完整性与汇总已由最终报告生成器核验通过。 |

20 任务的**历史开发 v3** 能力登记见 `reports/development_v3_FINAL/TASK_CAPABILITY.csv`：16/20 达 L3，其中 11 项 L4。该表仅说明 v3 开发覆盖，不能替代 v8 正式和保留输入的审计。最终支持范围要以 v8 的正式、验证和 final 分开报告。

发布工程修复：`runtime/publisher_v6/src/cfact/storage/ledger_media.py` 为每条真实边隔离图像缓存，`storage/tree.py` 将派生 PNG 放在发布目录内，保留全部原始 RGB 哈希检查。原始 v8 物理数据、任务目标和效果阈值均未改变。失败的 v5 整批输出及日志保留，不计作完成发布。

最终入口复现：`config/FINAL_ENTRY_V8_E2E_QUEUE.json` 在新的独立采集目录使用 v8 真正执行一个既有 Source，随后实际调用统一 CLI 的 resume、verify 和 export。该重复 Source 不纳入正式或保留集覆盖，状态见 `logs/FINAL_ENTRY_V8_E2E_STATUS.json`。

按固定顺序选择的实际 L3 review 见 `reports/REVIEW_INDEX.html` 和 `reports/REVIEW_INDEX.csv`：覆盖正式或保留范围内的 17 项任务，68 个原生视频完整解码通过，根层与非根层分别同父并排查看，包含实际祖先边链接。没有新增编码、重渲染或改变原始视频。

最终交付已完成：125 个合格根、1,417 条物理边、57,726 行官方动作、1,590 个发布视频完整读回通过；安全暂停/恢复及重复 smoke 的新增控制步为 0。专用 tmux 与本轮所有采集/发布/收尾作业已退出，证据见 reports/FINAL_PROCESS_CHECK.json、reports/FINAL_CLI_CONTROL_CHECK.json 和 reports/VERIFY_RESULT.json。历史会话名称仅供追溯。logs/SCHEDULER_STATUS.json 记录最后一次单根幂等检查，完整批次状态保留在 derived/*FINAL_SCHEDULER_STATUS.json。
