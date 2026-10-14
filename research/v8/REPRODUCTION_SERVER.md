# 使用本轮冻结 CFACT 生成器

服务器目录：`/mnt/ssd1/yanghc/CFACT_Research/rounds/recursive_generator_20261014T042404Z_418365`。运行环境、缓存与新输出位置由该目录的 `activate.sh` 设置。物理 worker 固定为 `runtime/v8`，8 路调度为 `runtime/orchestrator_v9`，官方发布为 `runtime/publisher_v6`，身份见 `manifests/CODE_FREEZE.json`。

```bash
CFACT_ROUND=/mnt/ssd1/yanghc/CFACT_Research/rounds/recursive_generator_20261014T042404Z_418365
cd "$CFACT_ROUND"
./cfact recursive preflight --round "$CFACT_ROUND"
./cfact recursive status --round "$CFACT_ROUND"
```

正式队列入口已实际用于 128 个事前固定请求。下面的 generate/resume 命令须在本轮调度器退出或暂停收尾后调用，避免同时争用同一调度锁。已完成请求依据收据跳过，不重放控制步；不同代码或冻结输入不能续写同一 run。

```bash
./cfact recursive generate --round "$CFACT_ROUND" --queue config/FORMAL_V8_FLEET8_QUEUE.json
./cfact recursive pause --round "$CFACT_ROUND"
./cfact recursive resume --round "$CFACT_ROUND" --queue config/FORMAL_V8_FLEET8_QUEUE.json
```

pause 停止新增分派，当前根按正常路径收尾。用户授权的上限为 8 个 worker，每张 GPU 最多 2 个；额外或 GPU 繁忙的 worker 使用已验证 OSMesa。GPU 不设温度准入门槛，92°C 为运行截止。新的请求需要事先冻结任务、Source、配置、代码身份和新输出路径，不根据采集效果替换 Source。

单根入口使用 `config/REPRODUCTION_V8_SINGLE_ROOT_QUEUE.json`，指定 PickPlaceCounterToCabinet/demo_6、最终物理 v8 和相同冻结配置。该根已有真实 3 层树和官方读回记录；此队列引用其既有 run，完整收据下再次运行不会新增仿真步骤。

```bash
./cfact recursive smoke --round "$CFACT_ROUND" --queue config/REPRODUCTION_V8_SINGLE_ROOT_QUEUE.json
```

原生验证入口已实际执行，证据为 `tests/FINAL_CLI_NATIVE_VERIFY.log`：1 个 v8 正式根、L3、完整性错误 0。参数如下；重新验证应使用独立派生输出目录。

```bash
./cfact recursive verify --round "$CFACT_ROUND" \
  --input runs/formal_v8/formal_v8_PickPlaceCounterToCabinet_demo_6 \
  --output derived/final_cli_verified_root \
  --calibrations derived/calibrations
```

完整正式审计采用不可变的时距协议 `config/EFFECT_COMPARISON_POLICY_V4.json`。统一 verify 从冻结 run 配置读取同族比较时距；原始采集 trace 保留不变。

官方发布入口已实际用于首份完整读回的 v8 数据。整批正式发布由托管收尾流程调用同一 `cfact.export.recursive` 模块，输入为整个 `runs/formal_v8`，只发布 PHYSICS_COMPLETE 根；其余请求保留在失败表和成本账本。输入 fingerprint、物理代码和发布代码都匹配的 COMPLETE 输出可幂等读回；部分导出需要新目录，不能覆盖原始 trace。

```bash
./cfact export-recursive --input runs/formal_v8 \
  --output datasets/formal_v8_complete_publisher6 \
  --scope FORMAL_V8_FROZEN_128_REQUESTS_QUALIFIED_ROOTS
```

历史首份已验证数据（publisher v5）在 `datasets/formal_v8_initial_verified_root`；修复版的实际冲突根已在 `datasets/publisher_v6_verified_collision_root` 完成 457 行官方读回。全量完成状态以全量目录中的 `release.json` 和 `FULL_READBACK.json` 为准。数据包含共享 Source 前缀、独立物理边、父子 snapshot、真实动作/状态/RGB、LeRobot by-task 数据及 review 视频/播放索引。谱系共享祖先，不是额外独立 Source。

托管会话使用本轮自己的 tmux socket：

```bash
tmux -S "$CFACT_ROUND/tmp/tmux.sock" list-sessions
```

`formal_v8_fleet8` 完成批量及全量原生审计；`post_formal_v8_resource_r2` 随后执行事前冻结的 20 个验证根及 20 个 final 根；`finalize_v8_parallel_r2` 待两组完整审计完成后进行全量官方发布与五种小规模公平对照。实时阶段记录在 `logs/POST_FORMAL_V8_STATUS.json` 与 `logs/FINALIZE_V8_STATUS.json`。暂停标记也阻止后续阶段启动。只能停止本轮身份可核对的进程，不能使用 killall 或操作其他作业。

验证/final 的输入与公式在实际候选结果产生前冻结；这些数据位于各自独立的 raw 目录，排除在正式训练发布之外。最终支持表按正式、验证、final 分别报告，保留固定 20 项分母。最终表格生成程序为 `scripts/build_delivery_reports_v8.py`，只有完整发布与五种对照完成后才生成交付报告。

原保留集准备因八路保守内存预留中止，状态保留在 `logs/archive`，原日志没有覆盖。修订见 `manifests/POST_FORMAL_V8_RESOURCE_AMENDMENT_R2.json`：Source 准备按两批四路执行，实际物理队列仍保持 8 路上限并逐 worker 准入。全部 18 个准备请求已完成，验证队列已通过真实八路健康检查。

收尾并行调度见 `manifests/FINALIZE_V8_RESOURCE_AMENDMENT_R2.json`：保留集完整审计后，固定 CPU 发布器与每次一个物理对照 worker 并行运行，预留 3.5 CPU 占用小时和约 100 GiB 发布空间。实际占用以最终收据为准；原 publisher v5 发布因已确认的跨边图像缓存冲突退出，失败状态和部分输出均保留。该批发布由 `runtime/publish_v8_recovery_publisher6.py` 执行完成，使用实际整根读回已验证的独立 publisher v6；物理数据及基线不重跑。对照的墙钟成本受同期 CPU 发布影响，主要比较使用事前冻结的实际控制步预算曲线。

最终统一入口在独立目录的真实复现使用下面的命令，运行记录位于 `logs/FINAL_ENTRY_*`。实际物理 Source 为 PickPlaceCounterToCabinet/demo_6，物理代码仍为 v8；这个重复 Source 不增加正式或保留集覆盖。

```bash
./cfact recursive smoke --round "$CFACT_ROUND" --queue config/FINAL_ENTRY_V8_E2E_QUEUE.json
./cfact recursive resume --round "$CFACT_ROUND" --queue config/FINAL_ENTRY_V8_E2E_QUEUE.json
./cfact recursive verify --round "$CFACT_ROUND" \
  --input runs/reproduction_v8_final_entry/reproduction_v8_final_entry_PickPlaceCounterToCabinet_demo_6 \
  --output derived/final_entry_v8_full_audit --calibrations derived/calibrations
./cfact export-recursive \
  --input runs/reproduction_v8_final_entry/reproduction_v8_final_entry_PickPlaceCounterToCabinet_demo_6 \
  --output datasets/final_entry_v8_publisher6_verified \
  --scope FINAL_UNIFIED_ENTRY_REPRODUCTION_EXISTING_SOURCE
```

已完成的采集队列再次 smoke/resume 会读取完成收据并跳过实际控制步骤。新的原生 verify 应改用新的派生输出目录；完整且身份一致的官方发布可幂等重查。安全暂停针对新物理分派；发布恢复程序也检查暂停标记，运行中的整批发布不应仅为测试暂停而被打断。

按固定顺序选择的实际 L3 review 见 `reports/REVIEW_INDEX.html` 和 `reports/REVIEW_INDEX.csv`：覆盖正式或保留范围内的 17 项任务，68 个原生视频完整解码通过，根层与非根层分别同父并排查看，包含实际祖先边链接。没有新增编码、重渲染或改变原始视频。

最终交付已完成：125 个合格根、1,417 条物理边、57,726 行官方动作、1,590 个发布视频完整读回通过；安全暂停/恢复及重复 smoke 的新增控制步为 0。专用 tmux 与本轮所有采集/发布/收尾作业已退出，证据见 reports/FINAL_PROCESS_CHECK.json、reports/FINAL_CLI_CONTROL_CHECK.json 和 reports/VERIFY_RESULT.json。历史会话名称仅供追溯。logs/SCHEDULER_STATUS.json 记录最后一次单根幂等检查，完整批次状态保留在 derived/*FINAL_SCHEDULER_STATUS.json。
