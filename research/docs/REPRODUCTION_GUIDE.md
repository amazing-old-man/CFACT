# SSD1 复现与审计

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

最终实验 registry 为 ../05_validation/ALL_RUN_RECEIPTS.csv，主表/限制见 FINAL_RESEARCH_RESULTS_ZH.md；checksum 独立复验见 CHECKSUM_READBACK_RECEIPT.json。repo/research/scripts 保存外层工具的可审阅副本，运行时默认仍引用本轮 SSD1 的固定工作区。改用其他目录需要显式重新设定存储根/输入身份，不能随意在旧盘执行。本轮预算及三次科研试验已结束，原命令只能用于读回；新物理运行应先创建新的预注册轮次与预算。
