# CFACT v2 方法与证据边界

本轮在独立 SSD1 checkout 中沿用真实 RoboCasa/MuJoCo、MimicGen waypoint、Reference/G0/G1/G2、完整快照和 ControlStepLedger。候选的 EffectIntent 仅为提议，实测 qpos、物体位置、fixture activation、接触与官方 success 才是效果事实。研究验收状态和最后选择见 ../05_validation/FINAL_RESEARCH_RESULTS_ZH.md；未达门槛的开发数据不可作为正式结果。

根来自预先锁定的 demo_0/demo_1，donors 为 demo_3/demo_4。初始五任务纵向切片只用 demo_0；不同方法重复同一 Source 不增加独立根数。当前 Source 导入 20/20 与真实递归覆盖分别统计，分母固定为 20。

根候选先冻结再执行；每个 sibling 恢复完整 physics/controller/gripper/RNG/fixture 快照。local 最后一步立即保存 actual child，fixed continuation 单独记录。下一层从 local child 快照恢复，重新读取当前 shadow observation 的 EE/object/fixture 几何。K 条实际控制对应 K+1 个实际状态。shadow observation 不执行 env.step；无任何 horizon 静态补齐。

事件锚来自已经执行的 local trace。阶段包括 APPROACH、CONTACT_ESTABLISHED、PARTIAL_TASK_EFFECT、HELD_TRANSPORT、PLACE_READY、RELEASE_AND_RETREAT、RECOVERABLE_MISS、TERMINAL_SUCCESS、UNSUPPORTED。持有验证要求任务接触、闭合夹爪和连续相对变换稳定；PLACE_READY 另需有界支持目标、实测目标距离 ≤3cm、最后四帧物体速度 ≤5cm/s。此标签仍不保证释放后支持。RECOVERABLE_MISS 只可在验证恢复几何后使用，本轮没有验证的恢复实例。自由体 Blender lid 没有已验证编译器，明确不支持，不创造关节。

三轮研究假设分别为实测效果多样性 frontier、Source/真实几何约束 G0 尺度、验证阶段驱动的通用 frontier。每轮独立 commit、开发物理收据、预注册端点和接受/回滚决定。第三轮按 depth/pool/step 限制循环真实 child；旧固定深度路径作为对照。Source suffix 耗尽后的独立新接近/恢复编译器本轮尚未完成，必须记录 NO_SUPPORTED_CONTINUATION，不宣称覆盖了该能力。

物理下限在方法物理实验前登记：slide 5mm，hinge 0.05rad，物体平移 10mm，activation 一次布尔改变。四个代表任务的完整 A/B/A 校准用于已验证任务族；不是所有 20 任务/所有 horizon 的噪声保证。0.25/0.5/1/2/4s 和首次真实接触两种对齐，全部 sibling pair 均保存，缺帧/未观察事件独立删失。全图 RGB 差异不等于目标 ROI 可观测；本轮 ROI/SSIM 未验证。控制器目标跟踪与真实任务效果分别测量。

主要端点是有效任务效果 sibling pair / 全部物理成本与非根状态覆盖 / 全部物理成本。独立 Source root 为统计单位，重复 horizon/边不是独立样本。正式 train/dev/test 场景隔离尚未建立，禁止跨场景泛化或新模型收益声称。

最终保留的是通过开发门槛的 Source/当前几何约束 G0 尺度和工程完整性修复。通用 frontier 第三轮虽在 CoffeeServeMug 实际执行到 depth=3，但整体有效 sibling 产出不满足接受规则，已在 commit 642564b 实际回滚；通用路径仅保留在 iteration_03_final 历史 worktree 与开发数据中，当前主分支不宣称通用多阶段递归已验收。
