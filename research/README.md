# 有界研发交付

本目录保存 SSD1 外层审计、采集保护、实验复现和报告生成脚本的版本化副本。它们默认使用 `/mnt/ssd1/yanghc/CFACT_Research` 以及只读 SSD2 固定环境。完整数据、失败日志、110 个 runner 收据与 Source 身份均留在该工作区，未打包大型数据或缓存。

科研结果未通过正式准入：严格两独立递归根覆盖 0/20；20/20 Source 当前导入，12/20 有真实控制步。三轮方法试验仅接受 Source 几何 G0，第一/三轮实际回滚；当前主路径保留旧固定 frontier。代表完整 A/B/A 与 1488 行官方 LeRobot 0.4 诊断读回通过，不能替代多数任务覆盖、独立场景划分或强基线优势。训练状态 TRAINING_NOT_STARTED。

只读复核入口见 docs/REPRODUCTION_GUIDE.md；物理脚本与 implement/fix/resume 文件保存历史行为，不能作为新预算下的自动重跑指令。新物理研发必须用新的预注册协议与输出目录。原盘工程未 merge/push。本目录追加不改变已经冻结的实验代码；实际采集 commit 始终见每个 CODE_FREEZE/runner receipt。
