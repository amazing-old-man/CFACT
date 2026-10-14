# CFACT 递归生成器最终交付

截至 2026-10-14T13:23:43.933182+00:00，冻结 v8 已执行全部 128 个正式请求和 20+20 个保留根。正式数据覆盖固定 20 项中的 16 项；16 项有 L3 证据，16 项有至少两个独立 Source 的 L3 证据（任务级 L4）。验证集达到 L3 的根为 14/20，final 为 16/20，三种输入都达到 L3 的任务为 12/20。全部原生完整性审计通过。

正式任务累计达到 L2、L3、L4 的数量分别为 16/20、16/20、16/20。正式 128 个独立 Source 根的精确层级为 L3=107、L2=5、L1=13、L0=3；L4 是跨 Source 的任务级别，不是单根深度。

正式请求的最终物理状态为 {'PHYSICS_COMPLETE': 125, 'NO_SUPPORTED_CONTINUATION': 1, 'BLOCKED_SOURCE_REPLAY': 2}；正式请求中因资源未执行和最终技术错误均为 0。Source 重放未达到前提 2 根、当前状态暂无支持续接 1 根，另有 18 根已完成物理执行但效果不足或比较时距删失，未达 L3。四项未进入正式请求的任务单列 NOT_REQUESTED，不计作资源失败；每项仍执行了验证和 final 输入。验证/final 的限制分类分别为 {'EFFECT_INSUFFICIENT_OR_CENSORED': 4, 'UNSUPPORTED_CURRENT_STATE': 2} / {'UNSUPPORTED_CURRENT_STATE': 2, 'EFFECT_INSUFFICIENT_OR_CENSORED': 2}。早期开发、资源准备和旧发布的技术错误保留在问题账本与历史日志中，不被当前最终状态的计数掩盖。

实际新增能力包括：执行前 EffectGoal 生成、当前状态阶段编译、从真实 child snapshot 的多候选递归、无适用 Source 后缀时的当前状态 primitive，以及有配额、看门狗和 GPU/CPU 后端记录的持久调度。入口和证据见 IMPLEMENTATION_MAP.md，正式物理入口为 runtime/v8/src/cfact/collection/worker.py。

正式批次使用 128 个不同 Source 请求，生成 1417 条独立物理边、323 个校准后的有效分叉 parent 实例、235 个非根 parent 实例和 1182 条共享祖先的终端路径。正式执行候选方法计数为 {'Reference': 343, 'G2': 1074}；登记 donor 路由 32 个，实际尝试读取 32 个不同 donor prior，其中 31 个成功加载；root 与尝试读取的 donor 共 160 个不同 Source 身份。实际 G1 donor 物理边 0 条，不能把它解释为未消费 donor 输入。共享谱系不计作独立 Source。

计入历史开发、正式、保留、校准和复现后，本轮共涉及 248 个不同原始 Source episode：root 输入 208 个、尝试消费的 donor prior 40 个，其中 39 个成功加载。实际 prepared 输入身份为 312 个；按任务及原始 episode 索引合并时，完整 states/actions 数组逐版本哈希一致，重复 prepared 版本没有计作额外独立示范。明细、每份 manifest 与 donor 记录的哈希见 SOURCE_CONSUMPTION.json；这些是输入来源统计，不把不同代码版本的物理输出混为正式 v8 数据。

官方发布位于 `/mnt/ssd1/yanghc/CFACT_Research/rounds/recursive_generator_20261014T042404Z_418365/datasets/formal_v8_complete_publisher6`：125 个合格根、1417 条独立边、57726 个真实动作行、1542 个官方 episode。全部 57726 行由 LeRobot 0.4.0 读取，1590 个 review 视频完整解码。未进入发布的 3 个正式请求仍保留原始 trace、状态和成本。保留集存放于独立目录，不混入正式训练发布。

直接查看 reports/REVIEW_INDEX.html，可并排播放 17 个实际 L3 任务的根层与非根层同父候选，68 个原生采集视频已完整解码。精确视频、祖先边与 Source 路径见 REVIEW_INDEX.csv。该 review 同时包括仅在保留输入取得 L3 的任务，未把它们混入正式训练发布。

输入拆分按 episode 及已知 layout/style 分组隔离，不宣称完整物理场景隔离。SOURCE_AND_SPLITS.json 中 final_never_consumed 是冻结时的事实；本报告中的 final 已按事前固定输入实际执行，未用其结果修改冻结物理方法。

历史旧协议的严格双独立根 0/20 结论保留不改写；历史开发 v3 为 16/20 L3、11 项 L4，与正式 v8 和保留输入分开统计。

真实 child 续接的具体证据见 manifests/CURRENT_STATE_PRIMITIVE_V8_PHYSICAL_PROOF.json：4 个诊断根完整审计通过，存在 21 条非根 CURRENT_STATE_PRIMITIVE 边。Source 文件与后缀没有删除，分类是剩余后缀入口与当前真实接触几何不兼容；该证据不等同于自然耗尽，也不宣称一切 Source 策略物理不可行。

| 任务 | 正式任务级别 | 正式 L3 根/根数 | 验证 | final |
| --- | --- | --- | --- | --- |
| PickPlaceCounterToCabinet | L4 | 8/8 | L3 | L3 |
| PickPlaceCounterToSink | L4 | 7/8 | L3 | L3 |
| OpenCabinet | L4 | 6/8 | L3 | L1 |
| OpenDrawer | L4 | 8/8 | L3 | L3 |
| OpenOven | L4 | 8/8 | L3 | L3 |
| CloseCabinet | L4 | 6/8 | L3 | L3 |
| CloseDrawer | L4 | 6/8 | L3 | L3 |
| CloseOven | L4 | 6/8 | L3 | L3 |
| OpenBlenderLid | L0 | 0/0 | L0 | L0 |
| CloseBlenderLid | L0 | 0/0 | L0 | L0 |
| CoffeeSetupMug | L4 | 8/8 | L3 | L3 |
| CoffeeServeMug | L4 | 8/8 | L3 | L3 |
| TurnOnMicrowave | L4 | 7/8 | L3 | L3 |
| StartCoffeeMachine | L4 | 8/8 | L3 | L3 |
| SlideDishwasherRack | L0 | 0/0 | L1 | L2 |
| SlideOvenRack | L4 | 6/8 | L3 | L3 |
| TurnOnSinkFaucet | L4 | 5/8 | L1 | L3 |
| TurnOnToaster | L0 | 0/0 | L3 | L3 |
| TurnOnStove | L4 | 7/8 | L1 | L3 |
| AdjustToasterOvenTemperature | L4 | 3/8 | L2 | L3 |

上述级别表示效果分叉证据，不是官方任务完成率。固定 20 项保留不删。未支持、Source 前提未达到、效果不足或时距截断详见 FAILURE_AND_PENDING.csv；未执行和未知不得解释成物理失败。当前不具备通用重新抓取或任意自由物体盖子的操纵能力。

正式分叉统计：根层有效 parent 112/123，非根有效 parent 211/228；有效 pair 1329/1832，另有 265 个时距不足的删失 pair。效果分箱和非根新增覆盖见 DIVERSITY_METRICS.csv，沿用已冻结的审计分箱规则。分箱不等于新物理机制，pair 不作为独立样本。Source 比较只在真实关节绑定或世界位姿对齐可核验时统计；缺少完整 Source activation 记录及无法确认位姿对齐的项明确为 UNKNOWN。

公平对照已在相同两个 OpenDrawer Source 和完整 parent snapshot 上执行五种方法，结果见 FAIR_BASELINES.csv；固定控制预算曲线见 FAIR_BASELINE_BUDGET_CURVES.csv。曲线从真实控制账本重建端点与完成边的可用时刻。总新增控制轴计入 CFACT 的付费 Source 重放；通用基线复用同一已存在的完整 parent，没有新增 Source 前缀步骤。扰动起点的实际 warmup 在总成本中付费，局部候选成本另列。每方法仅两个根，不作显著性或跨论文排名结论。通用基线按方法的 CPU 时间未独立计时，联合 worker 实测时间保留于 FAIR_BASELINES_VERIFY.json。

本轮实际控制步 102874；GPU 绑定 worker 4.951 小时，CPU worker（保守包含审计/发布）17.799 小时，日历 8.994 小时；SSD1 本 round 分配磁盘 256.66 GiB。完整收据及预算见 RESOURCE_AND_COST.json。

资源收据是 worker 进程占用时间，不是 GPU kernel 时间或全服务器 CPU 总时间；未独立计时的只读审计、测试和报告阶段另列。早期 v2 校准曾在旧资产目录创建并删除临时 XML，属于已记录的写入边界偏差；当前冻结运行时已重定向到 SSD1，并通过真实 A/B/A 复测。历史偏差、失败日志和修复证据均保留在 ISSUE_LEDGER.jsonl 的 E01 记录中。

最终统一入口已在新目录实际生成并验证 12 条边和 485 个官方动作行，完成后 resume 新增控制步为 0。该 Source 是复现输入，未重复纳入正式或保留集覆盖。状态和命令见 logs/FINAL_ENTRY_V8_E2E_STATUS.json。

本轮采集、导出和报告收尾作业均已退出，专用 tmux 服务已退出；真实服务器进程检查见 FINAL_PROCESS_CHECK.json。当前 logs/SCHEDULER_STATUS.json 是最后一次完成单根的幂等 CLI 检查；完整 128/20/20 请求状态分别保留在 derived/FORMAL_V8_FINAL_SCHEDULER_STATUS.json、derived/HELDOUT_VALIDATION_V8_FINAL_SCHEDULER_STATUS.json、derived/HELDOUT_FINAL_V8_FINAL_SCHEDULER_STATUS.json。剩余预算见 RESOURCE_AND_COST.json，查看、暂停和恢复命令见 REPRODUCTION.md。未开展世界模型训练。

任务书的本轮生成器与数据交付条件已满足；未支持的两项自由体盖子、洗碗机架的有限效果证据、仅保留输入通过的烤面包机，以及缺少完整 Source activation 记录等边界继续保留。没有通用重新抓取、全场景泛化或科研优越性结论。
