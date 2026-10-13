"""Derive final reports from immutable physical receipts; never change outcomes."""
import csv,hashlib,json,os,subprocess,time
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
R=Path('/mnt/ssd1/yanghc/CFACT_Research');V=R/'05_validation';V.mkdir(exist_ok=True);M=R/'manifests';now=datetime.now(timezone.utc).isoformat();head=subprocess.check_output(['git','-C',str(R/'repo'),'rev-parse','HEAD'],text=True).strip()
def read(p,default=None):
 return json.loads(p.read_text()) if p.exists() else default
versions=[('fixed_latest','SMOKE1_BASE_CURRENT_GPU'),('iteration01','DEV_ITERATION_01'),('iteration02','DEV_ITERATION_02'),('iteration03','DEV_ITERATION_03'),('source20','SOURCE20_PHYSICS')];metrics=[];decisions={}
for version,folder in versions:
 D=R/'01_development'/folder;p=D/'audit_verified/ROOT_METRICS.csv'
 if p.exists():
  for row in pd.read_csv(p).to_dict('records'):row.update(method_version=version,audit_path=str(p.relative_to(R)));metrics.append(row)
 if (D/'DECISION.json').exists():decisions[version]=read(D/'DECISION.json')
# Baseline metrics are separate schemes, not additional independent Source roots.
for scheme in ['GENERIC_CF_SINGLE','GENERIC_CF_OFFDEMO','CFACT_EFFECT_SINGLE','CFACT_EFFECT_RECURSIVE_MATCHED']:
 p=R/'02_baselines'/('audit_CFACT_EFFECT_SINGLE_MATCHED' if scheme=='CFACT_EFFECT_SINGLE' else f'audit_{scheme}')/'ROOT_METRICS.csv'
 if p.exists():
  for row in pd.read_csv(p).to_dict('records'):row.update(method_version=scheme,audit_path=str(p.relative_to(R)));metrics.append(row)
pd.DataFrame(metrics).to_csv(V/'ALL_ROOT_METRICS.csv',index=False)
# Preserve every measured window, including negatives/censors, with method identity.
for filename in ['AUDIT_SIBLING_EFFECTS.parquet','AUDIT_ACTION_DIVERSITY.parquet']:
 tables=[]
 for version,folder in versions:
  pth=R/'01_development'/folder/'audit_verified'/filename
  if pth.exists():
   table=pd.read_parquet(pth);table['method_version']=version;table['audit_path']=str(pth.relative_to(R));tables.append(table)
 for scheme in ['GENERIC_CF_SINGLE','GENERIC_CF_OFFDEMO','CFACT_EFFECT_SINGLE','CFACT_EFFECT_RECURSIVE_MATCHED']:
  pth=R/'02_baselines'/('audit_CFACT_EFFECT_SINGLE_MATCHED' if scheme=='CFACT_EFFECT_SINGLE' else f'audit_{scheme}')/filename
  if pth.exists():
   table=pd.read_parquet(pth);table['method_version']=scheme;table['audit_path']=str(pth.relative_to(R));tables.append(table)
 if tables:pd.concat(tables,ignore_index=True).to_parquet(V/filename,index=False)

executed_tasks=set()
for pth in (R/'01_development').rglob('frames.jsonl'):
 if 'history' in pth.parts:
  last=pth.read_text().splitlines()[-1:]
  if last and json.loads(last[0]).get('control_step_total',0)>0:
   hi=pth.parts.index('history')
   if len(pth.parts)>hi+2:executed_tasks.add(pth.parts[hi+2])
   else:
    aba=read(pth.parent.parent/'SAME_PARENT_ABA.json',{})
    if aba.get('task'):executed_tasks.add(aba['task'])
lock=read(M/'SOURCE20_FROZEN_IDENTITY.json');requests=read(R/'01_development/SOURCE20_PHYSICS/RESULTS.json',[]);coverage=[]
for x in lock['tasks']:
 t=x['task'];rows=[r for r in metrics if r['task']==t];scan=[r for r in rows if r['method_version']=='source20'];eligible=[r for r in scan if r['lineage_pass'] and r['has_two_root_edges'] and r['has_recursive_parent_with_two_edges'] and r['calibrated_effective_pairs']>0 and r['nonroot_effect_bins']>0];ids={r['source_id'] for r in eligible};rr=[r for r in requests if r['task']==t];physical=[r for r in rows if r['charged_control_steps']>0] or ([{}] if t in executed_tasks else []);rootpairs=sum(r['root_effective_pairs'] for r in scan) if scan else None;childpairs=sum(r['nonroot_effective_pairs'] for r in scan) if scan else None;supported=len(ids)>=2
 coverage.append(dict(task=t,SOURCE_PRESENT=True,IMPORTABLE=True,REPLAYABLE='PHYSICALLY_TESTED_PREFIX' if physical else 'CURRENT_PREPARED_ONLY_FULL_ACTION_REPLAY_UNKNOWN',CANDIDATE_SUPPORTED='PHYSICS_EXECUTED' if any(r['physical_edges'] for r in rows) else 'NO_EXECUTED_BRANCH' if rows and physical else 'UNKNOWN_AFTER_PARTIAL_PREFIX' if physical else 'UNKNOWN',RECURSION_SUPPORTED='TWO_INDEPENDENT_ROOTS_PASS' if supported else 'ONE_OR_MORE_REAL_RECURSIVE_TREES' if any(r['has_recursive_parent_with_two_edges'] for r in rows) else 'NOT_ESTABLISHED',PHYSICS_TESTED=bool(physical),actual_source20_roots=len(scan),independent_source20_roots=len({r['source_id'] for r in scan}),eligible_source20_roots=len(ids),NONTRIVIAL_RECURSIVE_TREE_SUPPORTED=supported,root_effective_pairs=rootpairs,nonroot_effective_pairs=childpairs,source20_attempt_status=';'.join(r.get('runner_status',r.get('status','UNKNOWN')) for r in rr) or 'NOT_EXECUTED',reason='PASS' if supported else 'TWO_INDEPENDENT_CALIBRATED_RECURSIVE_ROOTS_NOT_ESTABLISHED'))
pd.DataFrame(coverage).to_csv(V/'TASK20_COVERAGE.csv',index=False);pd.DataFrame(coverage).to_csv(R/'00_audit/SOURCE20_CURRENT_VALIDATION_MATRIX.csv',index=False)
if metrics:pd.DataFrame(metrics)[['method_version','task','source_id','charged_control_steps','root_effective_pairs','nonroot_effective_pairs','calibrated_effective_pairs','effect_bins','nonroot_effect_bins','lineage_pass']].to_csv(V/'EFFECT_DIVERSITY_BY_TASK.csv',index=False);pd.DataFrame(metrics)[['method_version','task','source_id','depth','has_two_root_edges','has_recursive_parent_with_two_edges','nonroot_effect_bins','lineage_pass']].to_csv(V/'RECURSION_DEPTH_AND_FRONTIER.csv',index=False)
# Count actual ledger execution once. Published ledger copies are representations, not new physics.
ledgers=[];ledger_steps=0
for base in [R/'01_development',R/'02_baselines',R/'03_formal_runs']:
 for p in sorted(base.rglob('frames.jsonl')):
  if 'history' not in p.relative_to(R).parts:continue
  last=None
  for line in p.open():
   if line.strip():last=json.loads(line)
  if last:
   steps=int(last.get('control_step_total',0));ledger_steps+=steps;ledgers.append(dict(path=str(p.relative_to(R)),actual_control_steps=steps,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
extra_aba=[]
for folder in ['ABA_BASE','SNAPSHOT_FIX_01']:
 D=R/'01_development'/folder;p=D/'SAME_PARENT_ABA.json'
 if p.exists() and not list(D.rglob('frames.jsonl')):extra_aba.append(dict(path=str(p.relative_to(R)),actual_control_steps=read(p).get('actual_total_control_steps',0)))
receipts=[read(p) for p in R.rglob('runner_receipt.json') if '04_datasets' not in p.relative_to(R).parts];wall=sum(x.get('wall_seconds',0) for x in receipts);gpu=sum(x.get('wall_seconds',0) for x in receipts if x.get('compute_backend')=='EGL_GPU0');cpu=sum(x.get('wall_seconds',0) for x in receipts if x.get('compute_backend')=='OSMESA_CPU_NO_CUDA');unknown=sum(x.get('wall_seconds',0) for x in receipts if not x.get('compute_backend'));size=allocated=0;escaping=[]
for directory,dirs,names in os.walk(R):
 dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
 for name in names:
  p=Path(directory)/name
  if p.is_symlink():
   if not p.resolve().is_relative_to(R):escaping.append(str(p))
   continue
  st=p.stat();size+=st.st_size;allocated+=st.st_blocks*512
cost=dict(ledger_control_steps=ledger_steps,additional_aba_control_steps=sum(x['actual_control_steps'] for x in extra_aba),actual_total_control_steps=ledger_steps+sum(x['actual_control_steps'] for x in extra_aba),completed_runner_wall_seconds=wall,known_gpu_runner_seconds=gpu,known_cpu_runner_seconds=cpu,legacy_backend_unrecorded_runner_seconds=unknown,workspace_logical_bytes=size,workspace_allocated_bytes=allocated,ledgers=ledgers,additional_aba_receipts=extra_aba,duplicates_excluded='04_datasets published histories',collector_wall_caveat='Pre-dispatch waits in drivers and standalone export/readback are outside runner wall; backend-unspecified legacy receipts are not asserted GPU time',guards_preserved=True,created_at_utc=now)
(V/'ALL_PHYSICAL_COSTS.json').write_text(json.dumps(cost,indent=2));(V/'WRITE_ROOT_VALIDATION.json').write_text(json.dumps(dict(root=str(R),all_regular_new_files_actual_root=str(R.resolve()),external_symlinks=escaping,cache_contract=str(R/'runtime/activate.sh'),original_protection=read(R/'00_audit/original_protection/after_RECEIPT.json'),limitation='Authorized original inputs/runtime referenced by paths, not new copies; no whole-disk scan'),indent=2))
base_rows=[]
for scheme in ['SOURCE_ONLY','GENERIC_CF_SINGLE','GENERIC_CF_OFFDEMO','CFACT_EFFECT_SINGLE','CFACT_EFFECT_RECURSIVE']:
 version='CFACT_EFFECT_RECURSIVE_MATCHED' if scheme=='CFACT_EFFECT_RECURSIVE' else scheme
 rows=[r for r in metrics if r['method_version']==version and r['task']=='OpenDrawer'];matching=list((R/'02_baselines/MATCHED_DRAWER/runner'/scheme).glob('runner_receipt.json')) if scheme.startswith('GENERIC') else list((R/'02_baselines/MATCHED_DRAWER/runner'/scheme).glob('runner_receipt.json')) if scheme=='CFACT_EFFECT_SINGLE' else list((R/'01_development/DEV_ITERATION_02/physics/runner_cpu_final/OpenDrawer').glob('runner_receipt.json'))
 
 if scheme in ['CFACT_EFFECT_SINGLE','CFACT_EFFECT_RECURSIVE']:
  matching=list((R/'02_baselines/MATCHED_PARENT_CPU'/scheme/'runner').glob('runner_receipt.json'))
 rec=read(matching[0],{}) if matching and scheme!='SOURCE_ONLY' else {};steps=sum(r['charged_control_steps'] for r in rows);pairs=sum(r['root_effective_pairs']+r['nonroot_effective_pairs'] for r in rows);seconds=0.0 if scheme=='SOURCE_ONLY' else rec.get('wall_seconds')
 base_rows.append(dict(scheme=scheme,scope='DEVELOPMENT_SINGLE_DRAWER_ROOT',status='READ_ONLY_SOURCE' if scheme=='SOURCE_ONLY' else 'PHYSICS_AUDITED' if rows else rec.get('status','NOT_TESTED'),roots=len(rows),actual_control_steps=0 if scheme=='SOURCE_ONLY' else steps if rows else None,effective_pairs=0 if scheme=='SOURCE_ONLY' else pairs if rows else None,pairs_per_control_step=pairs/steps if steps else None,runner_wall_seconds=seconds,pairs_per_runner_hour=pairs/seconds*3600 if seconds and rows else None,renderer='READ_ONLY_SOURCE_INPUT' if scheme=='SOURCE_ONLY' else rec.get('compute_backend'),statistical_ci='NOT_ESTIMABLE_ONE_INDEPENDENT_SOURCE_ROOT',matched_actual_costs='NOT_ASSERTED_SAME_CEILING_ONLY',full_parent_match=read(V/'BASELINE_SAME_PARENT_RECEIPT.json',{}).get('status','UNKNOWN'),method_version=version))
pd.DataFrame(base_rows).to_csv(V/'COST_NORMALIZED_BASELINES.csv',index=False)
abl=[]
for k,d in decisions.items():abl.append(dict(ablation=k,status=d['status'],any_strict_primary_gain=d.get('any_strict_primary_gain'),aggregate_pair_yield_baseline=d.get('aggregate_effective_pair_yield_baseline'),aggregate_pair_yield_method=d.get('aggregate_effective_pair_yield_method'),evidence=f'01_development/DEV_ITERATION_{int(k[-2:]):02d}/DECISION.json'))
for k in ['G0_ONLY','G1_ONLY','G2_ONLY','NO_EVENT_ANCHOR','NO_RECOVERY_BRANCH']:abl.append(dict(ablation=k,status='NOT_TESTED_BOUNDED_BUDGET',evidence='No physics result; no zero imputation'))
pd.DataFrame(abl).to_csv(V/'ABLATION_RESULTS.csv',index=False)
aba={p.parent.name:read(p) for p in (R/'01_development/FAMILY_ABA').glob('*/SAME_PARENT_ABA.json')};(V/'SAME_PARENT_ABA.json').write_text(json.dumps(dict(status='PASS_REPRESENTATIVE_FAMILIES_ONLY' if len(aba)==4 and all(x['status']=='PASS' for x in aba.values()) else 'INCOMPLETE',representatives=aba,first_failed_nullable_roundtrip=read(R/'01_development/ABA_BASE/SAME_PARENT_ABA.json'),corrected_nullable_roundtrip=read(R/'01_development/SNAPSHOT_FIX_01/SAME_PARENT_ABA.json'),not_a_full_task20_noise_calibration=True),indent=2))
full=read(R/'04_datasets/development_iteration01/FULL_READBACK.json',{'status':'NOT_COMPLETE'})
checksum=read(V/'CHECKSUM_READBACK_RECEIPT.json',{})
if checksum.get('status')=='PASS':full.update(file_integrity_status='PASS',file_integrity_files_checked=checksum['files_checked'],checksum_receipt='CHECKSUM_READBACK_RECEIPT.json')
(V/'FULL_READBACK.json').write_text(json.dumps(full,indent=2));supported=sum(x['NONTRIVIAL_RECURSIVE_TREE_SUPPORTED'] for x in coverage);tested=sum(x['PHYSICS_TESTED'] for x in coverage);blockers=[f'NONTRIVIAL_RECURSIVE_TREE_SUPPORTED={supported}/20, required >=15/20 with two independent preselected roots','SCENE_DISJOINT_UNKNOWN: formal train/dev/test split not established','H1/H2 superiority not established by matched independent-root comparisons','New source-exhaustion/recovery stage compiler not physically validated','Only representative family ABA; ROI/SSIM observability and root-level significance not established'];summary=dict(status='BOUNDED_DEVELOPMENT_COMPLETE_FORMAL_NOT_ADMITTED',formal_admitted=False,training_status='TRAINING_NOT_STARTED',supported_tasks=supported,denominator=20,source_present=20,current_prepared_verified=20,physics_tested_tasks=tested,method_decisions={k:d['status'] for k,d in decisions.items()},development_readback=full,physical_costs=cost|{'ledgers':'ALL_PHYSICAL_COSTS.json','additional_aba_receipts':'ALL_PHYSICAL_COSTS.json'},blockers=blockers,final_repo_commit=head,created_at_utc=now,final_test_outcomes_accessed=False)
summary['baseline_complete_parent_match']=read(V/'BASELINE_SAME_PARENT_RECEIPT.json');summary['immutable_source_hashes']=read(V/'SOURCE20_IMMUTABLE_HASHES.json',{}).get('status');
(V/'FINAL_PHYSICS_AUDIT.json').write_text(json.dumps(summary,indent=2));lin=read(M/'CFACT_V2_REPO_LINEAGE.json');lin.update(final_commit=head,method_decisions={k:d['status'] for k,d in decisions.items()},patch='CFACT_V2.patch',bundle='CFACT_V2.bundle',formal_admitted=False);(M/'CFACT_V2_REPO_LINEAGE.json').write_text(json.dumps(lin,indent=2))
report=f'''# 本轮研究结果与阻塞\n\n本轮完成了 SSD1 独立源码、20/20 当前 Source 导入、真实 MuJoCo 开发采集、完整快照/谱系/效果审计、有限方法迭代和官方 LeRobot 全行读回。非平凡递归工程覆盖为 **{supported}/20**，物理实际测试任务 **{tested}/20**，低于至少 15/20 的正式目标。**正式集未生成，训练状态为 TRAINING_NOT_STARTED。**\n\n原始 base 为 5d5edccc，最终独立仓库 commit 为 `{head}`。所有新文件/缓存/临时数据实际位于 `{R}`。原盘保护见 ../00_audit/original_protection/after_RECEIPT.json；没有向原仓库 merge/push。\n\n开发迭代决定：{json.dumps({k:d['status'] for k,d in decisions.items()},ensure_ascii=False)}。各轮独立 HYPOTHESIS、CODE_FREEZE、DECISION、物理 trace 与回滚提交均保留。方法变更没有替换预选的坏 Source，也没有降低物理效果下限。第三轮的有限 generic frontier 不能代替未实现的 Source 耗尽后新恢复动作，未观察到的深度/阶段不宣称支持。\n\n开发发布读回：{full.get('status')}，官方 {full.get('official',{}).get('datasets')} 个数据集、{full.get('official',{}).get('episodes')} 个 episode、全部 {full.get('official',{}).get('all_rows_checked')} 行。原生 {full.get('native',{}).get('roots')} 个根、{full.get('native',{}).get('physical_edges')} 条实际边，K/K+1、节点、终态和双向映射验证，完整文件哈希通过。release 为 DEVELOPMENT_DIAGNOSTIC_RELEASE，技术 PASS 不代表方法通过科研准入。直接 worker RGB 从同一实际控制步 ledger 的原生 RGB 无损物化，哈希/时间/状态/相机完全匹配，不重渲染；失败导出日志保留。\n\n总新物理成本：{cost['actual_total_control_steps']} 个控制步，runner wall {wall:.1f}s，已明确记录 GPU {gpu:.1f}s / CPU {cpu:.1f}s，旧 backend 未记明 {unknown:.1f}s；新增工作区逻辑 {size/2**30:.2f} GiB / 实占 {allocated/2**30:.2f} GiB。每个实际 ledger 只计一次，发布副本不重计祖先。Driver 排队、独立编码/readback 另见日志，不虚称已包含在 runner wall。首次 guard 计到了导出的重复 ledger，后来限定真实采集目录修正；物理 root_result 中计费值未改变。\n\n已建立部分动作→物理效果差异：抽屉/烤箱/咖啡 placement 在代表独立根存在超冻结下限的 sibling pair，真实 root 和 child 层分开统计。按钮根已开启时可有动作差异而无 activation 分叉；自由体 lid 的绑定/编译器不支持，保留拒绝与 Source 成本。没有可靠 task ROI 和 SSIM，短时全图差异无法证实目标视觉可观测。没有新训练，因此无法从本轮判断模型是否学会，已有 LeWM/DINO 报告不能替代当前 H1/H2。\n\n五组开发对照与实际成本见 COST_NORMALIZED_BASELINES.csv；NOT_TESTED、启动前资源拒绝和实际删失不写成 0 个效果成功。仅同上限的开发单根对照不能给多任务 superiority 或跨场景 CI。宏/微统计、全部窗口和事件对齐见逐对 Parquet；独立 Source root 才是统计单位，重复候选和窗口不能扩充样本量。\n\n必须解决的阻塞：\n\n'''+''.join('- '+b+'\n' for b in blockers)+'''\n后续可执行工作是建立经过验证的实际按钮 pre-contact parent、自由体盖子 grasp/release 合同和 Source 耗尽后的当前几何 stage 编译器；新一轮先固定预算与独立开发根，再重新验证 >=15/20 和强 Generic 基线。场景隔离与训练预算、不可变数据 SHA 均通过后，才可按 ../06_training/TRAINING_PLAN.json 创建无人值守训练。不得直接继续本轮无限调参。\n'''

report += '\n开发单根五组实际表：\n\n' + pd.DataFrame(base_rows).to_csv(index=False) + '\n此表实际成本不同、部分 renderer 不同；只允许单位步数描述性比较。第一次 Generic run 的代码来源限制见 ../02_baselines/BASELINE_CODE_PROVENANCE_NOTE.json，最终报告注释与原预注册分别保存。历史 generator_revision 标签不是方法身份，权威身份为 Git/config/implementation/Source hashes。\n'
(V/'FINAL_RESEARCH_RESULTS_ZH.md').write_text(report);(R/'01_development/DEV_FINAL_REPORT.md').write_text(report.replace('# 本轮研究结果与阻塞','# 有限开发最终报告'));(M/'CHANGELOG_AUTONOMOUS.md').write_text('# 自主变更与实验收据\n\n'+subprocess.check_output(['git','-C',str(R/'repo'),'log','--format=- %h %s','5d5edccc7d2167f5aacc08dd9c9f53b031a5f605..HEAD'],text=True)+'\n各全局轮次的 HYPOTHESIS/DECISION 与物理/CPU/GPU/预算收据位于 ../01_development/DEV_ITERATION_01..03。失败假设保留并实际 git revert；snapshot nullable、当前几何和存储路径/RGB 发布为工程完整性修复。stage truth 为 held-placement 局部修复 1；不依据 final test。\n')
for p in [M/'FORMAL_PRE_REGISTRATION.json',M/'FORMAL_CODE_FREEZE.json',R/'06_training/RUN_STATUS.json']:
 x=read(p);x['blockers']=blockers;x['last_updated_utc']=now;p.write_text(json.dumps(x,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['physical_costs','development_readback','baseline_complete_parent_match']},indent=2))
