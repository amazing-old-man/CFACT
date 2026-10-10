# 双任务20+20 Source与native224冻结验收

本轮只准备正式Source并运行4个固定开发根，不启动正式40-root和模型训练。
交接路径：`output/preparation_40root__20261010T164033Z/HANDOFF.md`。

- 官方revision为522e4ffa3bb9d9729f79b241c65fc9e589f2659c；仅两个任务的pretrain/human包。
- 每任务20个冻结root+2个训练donor，保留原18条并按同一hash规则增加2条。
- 新source/路径完成52条导入/prepare/真实回放，包括8条开发Source；156个官方视频全解码。
- 两任务各一个worker并行，4根均非空，26条真实edge：Root14、Desc12；每根有1个actual-child递归parent。
- 官方LeRobotDataset0.4.0全部2109行、46episodes、两个任务dataset及4个独立tree dataset读回通过。
- 52个完整review/lineage视频全解码，native224三相机、K/K+1真实终态、所有文件SHA通过。
- 两任务demo_0的Root/Desc共4个A/B/A parent，状态/RGB重复差0，12个诊断控制步，无隐藏恢复步。
- 当前声明的H3/F5接口：Flat1526，Star8 parent/30 pair，1500个未配对合格Flat保留；零模型forward/优化器更新。
- 回归204passed/1skipped（官方reader集成模块在仿真venv缺lerobot；实际export环境reader完整通过），lint通过。
- Source/候选/执行/codec/interface的16个科学核心文件相对任务进入时SHA相同；native224进入时已有改造已保全。

四种机制保留原定义。OpenDrawer开发donor未满足原entry pose邻域兼容规则，本轮G1实际edge为0；
CoffeeServeMug有3条G1真实edge。未放宽门槛、改donor或按结果换Source。Reference/G0有1组前5步
精确近重复，原样保留。Source缺原控制器内部状态，不宣称原记录逐帧bit-exact；scene-disjoint未知。
Smoke小样本不保证正式40个请求非空或成功。未自然出现的空根/中断物理案例仅有公共回归证据。

新冻结建立独立commit/tag和源码副本，保留旧FREEZE.json、v0.7.1.dev0及旧输出；
新版用scripts/verify_generator_identity.py与交接FROZEN_RELEASE.json校验，不改旧冻结文件。
Smoke在未提交开发快照运行；新冻结源码完整package SHA与该实测快照完全一致。
只读H3/F5核验脚本的两次结构读取错误已修复，原失败日志保留，原模型API与物理数据未变。
原运行环境分发metadata残留旧版本字符串，以真实模块位置、__version__、完整package SHA和独立wheel识别新版。

正式40-root必须等待用户审查本次HANDOFF后的下一次明确批准。命令、环境与资源预算详见
`docs/source40_reproduction.md`及交接NEXT_RUN_COMMANDS.md、RESOURCE_ESTIMATE.md。
