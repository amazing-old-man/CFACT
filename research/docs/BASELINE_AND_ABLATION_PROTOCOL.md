# 开发基线与正式准入

五组为 SOURCE_ONLY、GENERIC_CF_SINGLE、GENERIC_CF_OFFDEMO、CFACT_EFFECT_SINGLE、CFACT_EFFECT_RECURSIVE。同一实际 Source/root/scene/seed/controller/三相机/官方 success 与上限匹配。Source-only 只读已存在示范，新增物理成本 0，既有 K 状态缺失 terminal 不补造 K+1。

强 Generic 包含 stop、0.5/1.5 缩放、反向、XY 轴交换、平移/旋转噪声、打开夹爪、10-step 延迟；动作范围和 base/control mode 保持原合同。Source 结束后的 stop 是明确 source index=-1 的实际 env.step，不是静态状态补齐。80 local +80 fixed 的所有真实步、前缀和失败均计费。Off-demo 6/8 起点来自 Source、2/8 来自固定 10-step 扰动的真实子状态；扰动前缀只计成本、不计递归方法收益。

当前阶段为开发单根对照，参见 ../02_baselines/BASELINE_DEVELOPMENT_PRE_REGISTRATION.json。实际预算消耗分别报告，不把相同上限说成实际步数/wall 完全相等。单根无法给跨独立根置信区间。三轮方法对照属于预注册的开发假设评估；与未执行的 G0/G1/G2-only、无事件/无恢复消融分别记录 NOT_TESTED。

正式准入需要完整真实快照/谱系/RGB/官方读回、至少 15/20 的两个独立根工程覆盖、正式 Source/场景 split、固定主端点与预算。任一缺失则 FORMAL_NOT_ADMITTED / TRAINING_NOT_STARTED。正式集从未冻结或查看 final test outcome；开发数据不能重命名为 formal。

四组最终同父比较采用冻结 master snapshot SHA、Source ID/cursor、实际 prefix sim state 一致检查，恢复完整 master 后严格零差异；实际 CPU 两组通过，Generic 两组与其完整父快照比较均为零。所有 preflight 失败保留，见 ../manifests/PARENT_MATCH_ENGINEERING_RECEIPT.json。CFACT 为 OSMesa CPU，Generic 为 EGL GPU0，墙钟不可直接横比；Source-only 新增采集时间/步数为零。等实际成本的多根正式比较未完成。
