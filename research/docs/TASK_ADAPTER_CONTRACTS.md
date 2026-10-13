# 任务族合同

| 族 | 实际能力条件 | 实测主要效果 | 续接条件与限制 |
|---|---|---|---|
| articulated_contact | 唯一实际 slide/hinge joint，真实 axis/anchor/range，明确 task q target | 关节 m 或 rad，qvel，handle/EE/contact | 当前几何重锚，保留 controller/source cursor；source suffix 缺失不假造 |
| held_object_placement | 实际接触、闭合夹爪、连续相对变换稳定 | 物体实测世界位移，接触与官方 success | 有界 fixture/receptacle goal 才能判 PLACE_READY，释放须看到 opening commands |
| button_activation | 原生 fixture activation 可读且每帧状态 MEASURED | 布尔 turned_on 差异 | 初始已开启可能没有 task-effect 分叉，机械臂差异不能替代 |
| free_body_lid | 本轮无验证过的抓取/释放编译器 | UNSUPPORTED | 不添加假 hinge，保留拒绝理由与真实 Source 成本 |

受限接触不等于禁止所有碰撞。原始 motion validation/contact guard 记录仍保留；未验证的 attached-object collision/safety 不能标成 PASS。阶段分类器所用 3mm 关系漂移和 3cm/5cm/s 放置就绪阈值为操作条件，不能替代冻结的效果差异阈值。

20 项任务的精确 SOURCE_PRESENT/IMPORTABLE/REPLAYABLE/CANDIDATE/RECURSION/PHYSICS 状态由 ../05_validation/TASK20_COVERAGE.csv 提供。导入 prepare 的 reset_to 不产生新的机器人控制步；实际 replay/env.step 证据必须另外标记。
