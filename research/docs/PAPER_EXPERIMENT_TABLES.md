# 论文证据表索引

可复核的实际数值由 ../05_validation/ 下 TASK20_COVERAGE、EFFECT_DIVERSITY_BY_TASK、RECURSION_DEPTH_AND_FRONTIER、COST_NORMALIZED_BASELINES、ABLATION_RESULTS CSV 与逐对 Parquet 提供；每项带方法版本、实际控制步和独立 Source 单位。最终研究结论见 FINAL_RESEARCH_RESULTS_ZH.md。

表应分别列：20 项真实能力边界；根层/非根层有效效果 pair；效果状态 bins；完整 A/B/A 噪声与冻结下限；真实深度/未续接理由；全部成本包括技术失败和扰动前缀；五方案的实际执行状态；每轮接受或回滚；原生树与 Flat 完整读回。不能把 NOT_TESTED 写成 0，也不对单根重复 pair 给显著性。

OPENWAM 对比仅描述已存在同状态不同动作和 off-demo 起点机制，来源 https://openwam.stanford.edu/ 、https://arxiv.org/abs/2610.07922 。CFACT 新颖性须由 H1/H2 数据支持。已有 DINO-WM/LeWM 原报告的证据在 ../00_audit/LEGACY_MODEL_EVIDENCE.md；本轮没有新模型结果。
