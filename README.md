# CFACT v8 真实递归生成器

本分支包含实际生成数据的 v8 方法，以及已验证的 8 worker 调度和官方发布修复。源码基于 SSD1 当前 CFACT 历史增量开发，保留原始提交与旧实验记录。

新增能力：执行前提出 EffectGoal；按当前真实几何编译候选；恢复同一完整 parent 执行多个分支；从实际 child snapshot 再次分叉；剩余 Source 后缀不适用时执行有边界的当前状态 primitive。

- 物理 v8：`39bdf717dafbe93c43f8fdba4233af0614bc7e2d`。
- 8 worker 调度：`588f4f1372afd173d4cf3b1034c6058732ff83e9`。
- 官方发布修复：`dd8ec79365018be04a29311de8e8484e744cd4d9`。

完整源码位于 `src/cfact/`；核心入口为 `collection/worker.py`、`generation/stage_compiler.py`、`collection/round_runner.py`、`export/recursive.py`。物理 v8 的 worker、generation 和 execution 模块在后续调度/发布修复中保持不变。上述提交与冻结快照对应关系见 [CODE_FREEZE.json](research/v8/CODE_FREEZE.json)。

## 实际结果与范围

固定 20 项任务中，16 项在正式批次达到 L4。正式执行 128 个 Source 请求，125 个合格根生成 1,417 条独立物理边、57,726 个官方动作行。验证和 final 各 20 个根，分别 14 和 16 个达到 L3；12 项任务在三组输入都达到 L3。官方逐行读取、原生审计和 1,590 个发布视频完整解码通过。

L3 表示根层及真实非根层都有同父可区分效果；L4 表示至少两个不同 Source 分别达到 L3，不表示四层深度或官方任务成功率。自由体盖子、通用 regrasp 和完整跨场景泛化仍有明确限制，详见 [结果报告](research/v8/RESULTS_ZH.md) 与 [20 任务支持表](research/v8/TASK_CAPABILITY.csv)。五种基线仅比较两个相同 Source，不作显著性或科研优越性结论。

## 实际部署与复现

已验证服务器部署目录：

```text
/mnt/ssd1/yanghc/CFACT_Research/rounds/recursive_generator_20261014T042404Z_418365
```

在该目录执行已经实际测试的入口：

```bash
./cfact recursive status --round "$PWD"
./cfact recursive smoke --round "$PWD" --queue config/FINAL_ENTRY_V8_E2E_QUEUE.json
```

现有完成队列再次 smoke/resume 不新增控制步。完整生成、预检、暂停、恢复、验证和导出命令见 [服务器复现说明](research/v8/REPRODUCTION_SERVER.md)。`research/v8/configs/` 保存实际冻结配置；`research/v8/server/` 保存部署入口。这些文件包含已验证服务器的路径和输入身份；在其他机器部署时需要建立新 round、冻结新路径和输入，不能直接续写旧冻结 run。

正式物理配置明确启用 `task_effect_v2` 与 `recursive_goals_v2`；默认历史 quickstart 不等同于本轮 v8 正式配置。物理方法、调度器和发布器分别冻结，复现原始数据应使用记录的物理 v8 快照。仓库不包含原始数据、视频、资产或虚拟环境；它们保存在服务器目录。

依赖锁文件位于 `requirements/`；保留的 [旧版本安装与接口说明](docs/legacy/README_0_7_1.md) 对应历史 0.7.1 提取版本。其旧温度、后端和 local/tail 协议不替代 v8 本轮协议：最多 8 worker、92°C 截止、不设温度准入、已验证 EGL/OSMesa、固定效果时距及共享祖先存储。

原有回归为 258 passed、1 skipped（物理 venv 无 LeRobot 的合成集成测试）；真实 LeRobot 0.4.0 导出在独立 export venv 完成逐行验证。该私有发布分支还会运行当前 checkout 的回归，结果记录于 research/v8。
