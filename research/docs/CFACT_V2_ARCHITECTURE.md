# 运行架构

`cfact.collection.worker` 建立 Source 根和唯一 live MuJoCo scene；`execution.snapshot` 维护完整可恢复 backend 状态，`execution.current` 从当前 shadow state 获取几何；`generation.context/options` 编译全部候选；`generation.task_adapters/event_anchor` 从 local 测量生成效果与阶段；`generation.frontier` 为通用预算队列。父级实际 trace 和完整快照只记录一次，由 lineage 引用连接后续节点。

`execution.history.ControlStepLedger` 捕获每步动作、状态、三相机原生 224 RGB、相机参数和 hashes。`storage.tree` 生成可移植树；直接 worker 的 clean camera 文件由 `storage.ledger_media` 从同一实际 ledger 帧按 integration/state/time/scene/camera/image hash 精确匹配后以无损 PNG 写出，不重渲染、不推进 physics。对缺帧或部分 spool 拒绝发布，原始 ledger NPZ 不修改。

`export.dataset` 使用固定 LeRobot 0.4.0 写出 Flat 数据；`data.release.ReleaseDataset` 用于发布视图，`data.reader.TreeDataset` 用于 raw data/ 视图。terminal map 显式保留 K+1 终态，所有 episode/edge 可双向追溯。发布哈希验证与研究准入独立：技术读回 PASS 不等于多数任务/H1/H2/训练 PASS。

新 `artifact_root/allowed_write_root/quota_root` 校验绝对路径及真实目标文件系统、symlink 逃逸、free bytes/inodes 和新增配额。外部 `scripts/research_run.py` 每 3 秒检查 RAM/GPU/磁盘、每 15 秒更新全工作区成本；只停止自己启动的进程组。所有新缓存与临时文件为 SSD1 实际路径。代码、数据和阈值每个 run 单独冻结；不混合作为 formal run。

最终保留的是通过开发门槛的 Source/当前几何约束 G0 尺度和工程完整性修复。通用 frontier 第三轮虽在 CoffeeServeMug 实际执行到 depth=3，但整体有效 sibling 产出不满足接受规则，已在 commit 642564b 实际回滚；通用路径仅保留在 iteration_03_final 历史 worktree 与开发数据中，当前主分支不宣称通用多阶段递归已验收。
