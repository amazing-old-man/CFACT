"""Add traceable per-task, per-run and uncertainty evidence to bounded final reports."""
import hashlib,json,os,subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
import pandas as pd
R=Path('/mnt/ssd1/yanghc/CFACT_Research'); V=R/'05_validation'
def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def md(rows,cols):
 return '| '+' | '.join(cols)+' |\n| '+' | '.join('---' for _ in cols)+' |\n'+''.join('| '+' | '.join(str(r.get(c,'' )).replace('|','/').replace('\n',' ') for c in cols)+' |\n' for r in rows)
def logical_bytes(p):
 return sum(q.stat().st_size for q in p.rglob('*') if q.is_file() and not q.is_symlink()) if p.exists() else None
metrics=pd.read_csv(V/'ALL_ROOT_METRICS.csv'); coverage=pd.read_csv(V/'TASK20_COVERAGE.csv'); requests=read(R/'01_development/SOURCE20_PHYSICS/RESULTS.json'); assert len(requests)==40
status_counts=dict(Counter(x['runner_status'] for x in requests))
proposal_rows=[]
for base in ['01_development','02_baselines']:
 for p in sorted((R/base).rglob('candidate_pool.json')):
  if 'data' not in p.parts:continue
  pool=read(p); relative=p.relative_to(R); root=p.parents[2].name; task=p.parts[p.parts.index('data')+2]
  for x in pool.get('rejected',[]):proposal_rows.append(dict(task=task,root=root,node=p.parent.name,candidate_id=x.get('candidate_id'),reason=x.get('reason'),pool_path=str(relative)))
pd.DataFrame(proposal_rows).to_csv(V/'PROPOSAL_REJECTIONS.csv',index=False)
# The registry includes pre-dispatch refusals, actual interrupted prefixes, repairs and all methods.
runrows=[]; times=[]; temps=[]; gpu_free=[]; host_free=[]
for p in sorted(R.rglob('runner_receipt.json')):
 if '04_datasets' in p.relative_to(R).parts:continue
 x=read(p); guard=p.parent/'guard.jsonl'; gs=[]
 if guard.exists():
  for line in guard.read_text().splitlines():
   g=json.loads(line);gs.append(g)
   if g.get('time_utc'):times.append(datetime.fromisoformat(g['time_utc']))
   if g.get('temperature_c') is not None:temps.append(g['temperature_c'])
   if g.get('gpu_free_mib') is not None:gpu_free.append(g['gpu_free_mib'])
   if g.get('available_ram_bytes') is not None:host_free.append(g['available_ram_bytes'])
 cmd=x.get('command',[]);cmd=[str(z) for z in cmd];control=0;runpath=None
 if '--run' in cmd:runpath=Path(cmd[cmd.index('--run')+1])
 elif '--output' in cmd:runpath=Path(cmd[cmd.index('--output')+1])
 if runpath and runpath.exists():
  for h in runpath.rglob('frames.jsonl'):
   if 'history' not in h.parts:continue
   last=h.read_text().splitlines()[-1:]
   if last:control+=int(json.loads(last[0]).get('control_step_total',0))
 runrows.append(dict(receipt=str(p.relative_to(R)),status=x['status'],pid=x.get('pid'),exit_code=x.get('exit_code'),runner_wall_seconds=x['wall_seconds'],backend=x.get('compute_backend','LEGACY_UNRECORDED'),executed_code_commit=x.get('executed_code_commit','LEGACY_UNRECORDED'),guard_sha256=x.get('guard_code_sha256','LEGACY_UNRECORDED'),actual_ledger_control_steps=control if runpath else None,run_path=str(runpath) if runpath else None,guard_samples=len(gs),maximum_observed_temperature_c=max((g['temperature_c'] for g in gs if 'temperature_c' in g),default=None),minimum_observed_available_host_ram_gib=min((g['available_ram_bytes']/2**30 for g in gs if 'available_ram_bytes' in g),default=None),minimum_observed_gpu_free_mib=min((g['gpu_free_mib'] for g in gs if 'gpu_free_mib' in g),default=None),own_process_peak_ram='NOT_MEASURED',own_process_peak_vram='NOT_MEASURED'))
pd.DataFrame(runrows).to_csv(V/'ALL_RUN_RECEIPTS.csv',index=False)
calendar=dict(first_guard_utc=min(times).isoformat(),last_guard_utc=max(times).isoformat(),guard_calendar_span_seconds=(max(times)-min(times)).total_seconds(),meaning='Calendar span between first and last guard samples; includes idle/waits, overlaps work, is not cumulative CPU/GPU time',maximum_observed_shared_gpu_temperature_c=max(temps),minimum_observed_shared_gpu_free_mib=min(gpu_free),minimum_observed_available_host_ram_gib=min(host_free)/2**30,own_process_peak_ram='NOT_MEASURED',own_process_peak_vram='NOT_MEASURED',source20_dispatch_requests=len(requests),source20_status_counts=status_counts,all_physical_dispatches_complete=True)
write(V/'ACQUISITION_CALENDAR_AND_RESOURCES.json',calendar)
# Keep reasons tied to executed physics or explicit resource refusal, not inferred simulator capability.
for i,row in coverage.iterrows():
 task=row['task']; group=metrics[metrics.task==task]; scan=group[group.method_version=='source20']; rr=[x for x in requests if x['task']==task]
 notes=[]
 if len(scan):
  if not bool(scan.has_recursive_parent_with_two_edges.any()):notes.append('SOURCE20_SINGLE_ROOT_HAS_NO_REAL_CHILD_WITH_TWO_SUCCESSORS')
  else:notes.append('SOURCE20_ONLY_ONE_COMPLETED_INDEPENDENT_ROOT')
 if any('BLOCKED_RESOURCE' in x['runner_status'] for x in rr):notes.append('SOURCE20_RESOURCE_REFUSAL: '+', '.join(sorted({x['runner_status'] for x in rr if 'BLOCKED_RESOURCE' in x['runner_status']})))
 if task=='OpenBlenderLid':notes.append('FREE_BODY_LID: operation_binding:bound_object_joint_ambiguous:[]; articulation binding/compiler unsupported; no artificial hinge')
 if task=='CloseBlenderLid':notes.append('ACTUAL_PREFIX_THEN_ERROR: observation_unsupported_binding:get_object_poses; current shadow binding unsupported')
 if task=='CoffeeSetupMug':notes.append('PARTIAL_PREFIX_INTERRUPTED_BY_GPU_TEMPERATURE; complete candidate/child capability UNKNOWN')
 if task=='TurnOnMicrowave':notes.append('DEVELOPMENT_PARENT_ALREADY_ACTIVATED; real commands differed but activation sibling difference was zero')
 if task=='CoffeeServeMug':notes.append('SEPARATE_METHOD_SMOKES_HAVE_REAL_DEPTH2_AND_REJECTED_DEPTH3; SOURCE20_PROFILE_HAS_NO_COMPLETED_ROOT')
 if not row.PHYSICS_TESTED:notes.append('NO_EXECUTED_CONTROL_LEDGER; presence/import cannot establish physical capability')
 coverage.at[i,'specific_blocker']='; '.join(notes)
 coverage.at[i,'evidence_path']='01_development/SOURCE20_PHYSICS/runner/'+task+'; 05_validation/ALL_ROOT_METRICS.csv; 05_validation/PROPOSAL_REJECTIONS.csv'
coverage.to_csv(V/'TASK20_COVERAGE.csv',index=False);coverage.to_csv(R/'00_audit/SOURCE20_CURRENT_VALIDATION_MATRIX.csv',index=False)
# Fixed horizons and variable-duration endpoints remain separate; never fill stopped windows.
siblings=pd.read_parquet(V/'AUDIT_SIBLING_EFFECTS.parquet'); actions=pd.read_parquet(V/'AUDIT_ACTION_DIVERSITY.parquet')
initial=bool(siblings.initial_measured_equal.all()) and bool(actions.initial_measured_equal.all())
assert initial, 'An actual same-parent pair did not share the initial measured state'
window_counts=siblings.groupby(['method_version','scope','alignment','horizon_s','status'],dropna=False).size().reset_index(name='measured_or_censored_pair_windows');window_counts.to_csv(V/'HORIZON_AND_CENSOR_COUNTS.csv',index=False)
write(V/'SAME_PARENT_MEASURED_STATE_READBACK.json',dict(status='PASS',all_measured_pair_initial_states_equal=initial,action_pair_count=len(actions),physical_pair_window_count=len(siblings),all_negative_and_censored_windows_retained=True,zero_local_action_edges_excluded_from_primary_pair_yield=True,full_parent_snapshot_comparison='BASELINE_SAME_PARENT_RECEIPT.json',temporal_windows='HORIZON_AND_CENSOR_COUNTS.csv'))
base=pd.read_csv(V/'COST_NORMALIZED_BASELINES.csv')
roots={s:R/'02_baselines/MATCHED_DRAWER/runs'/s for s in ['GENERIC_CF_SINGLE','GENERIC_CF_OFFDEMO']}
roots.update({s:R/'02_baselines/MATCHED_PARENT_CPU'/s/'run' for s in ['CFACT_EFFECT_SINGLE','CFACT_EFFECT_RECURSIVE']})
for i,row in base.iterrows():
 scheme=row.scheme
 if scheme=='SOURCE_ONLY':continue
 b=logical_bytes(roots[scheme]);pairs=int(row.effective_pairs);steps=int(row.actual_control_steps)
 base.at[i,'collector_artifact_bytes']=b
 base.at[i,'pairs_per_collector_GiB']=pairs/(b/2**30) if b else None
 base.at[i,'control_steps_per_effective_pair']=steps/pairs if pairs else None
 base.at[i,'runner_seconds_per_effective_pair']=row.runner_wall_seconds/pairs if pairs else None
 base.at[i,'collector_GiB_per_effective_pair']=(b/2**30)/pairs if pairs else None
 match=metrics[metrics.method_version==('CFACT_EFFECT_RECURSIVE_MATCHED' if scheme=='CFACT_EFFECT_RECURSIVE' else scheme)]
 base.at[i,'root_effective_pairs']=int(match.root_effective_pairs.sum());base.at[i,'nonroot_effective_pairs']=int(match.nonroot_effective_pairs.sum());base.at[i,'nonroot_effect_bins']=int(match.nonroot_effect_bins.sum())
base.to_csv(V/'COST_NORMALIZED_BASELINES.csv',index=False)
macro=[]
for version,g in metrics.groupby('method_version'):
 step=g.charged_control_steps;pair=g.root_effective_pairs+g.nonroot_effective_pairs
 macro.append(dict(method_version=version,completed_root_count=len(g),independent_source_ids=g.source_id.nunique(),tasks=g.task.nunique(),actual_control_steps=int(step.sum()),effective_pairs=int(pair.sum()),micro_pair_yield=float(pair.sum()/step.sum()),macro_mean_root_pair_yield=float((pair/step).mean()),macro_mean_nonroot_bins_per_step=float(g.nonroot_bins_per_charged_step.mean()),confidence_interval='NOT_ESTIMABLE_WITHIN_TASK_ONE_INDEPENDENT_ROOT',repeated_versions_are_independent=False))
pd.DataFrame(macro).to_csv(V/'MACRO_MICRO_DEVELOPMENT_METRICS.csv',index=False)
# Candidate effect dimensions may be representative-family calibrated, never all-task noise proof.
audit=read(V/'FINAL_PHYSICS_AUDIT.json');audit.update(all_same_parent_measured_states_equal=initial,source20_dispatch_requests=40,source20_completed_roots=8,source20_status_counts=status_counts,scene_disjoint_status='UNKNOWN',equal_realized_cost_baselines='NOT_TESTED_BOUNDED_DEVELOPMENT',equal_wall_baselines='NOT_TESTED_MIXED_RENDER_BACKENDS',only_representative_family_aba=True,calendar_and_resources='ACQUISITION_CALENDAR_AND_RESOURCES.json');write(V/'FINAL_PHYSICS_AUDIT.json',audit)
# Five single-source roots, selected during development; this is descriptive, not confirmatory.
dec=read(R/'01_development/DEV_ITERATION_02/DECISION.json'); boot=read(V/'EXPLORATORY_ROOT_BOOTSTRAP.json')
cols=['task','PHYSICS_TESTED','actual_source20_roots','eligible_source20_roots','RECURSION_SUPPORTED','specific_blocker']
append='\n逐任务验收及具体阻塞（分母固定 20；PHYSICS_TESTED 包括实际前缀，不能据此声称完整回放或递归支持）：\n\n'+md(coverage.to_dict('records'),cols)
append+='\n40 个预选 Source20 请求全部有最终收据：'+json.dumps(status_counts,ensure_ascii=False)+'。8 个完成的根包含 OpenBlenderLid 的 0 分支拒绝根；12 个任务执行过至少一个真实控制步，其余能力 UNKNOWN。不能用同一个 Source ID 的多个方法版本凑出第二个独立根。所有实际失败、前缀和同父复测成本都计入 ALL_PHYSICAL_COSTS.json。\n'
append+='\n被接受的唯一科学扩展是 Source/current geometry 限幅 G0：五个固定开发任务上微平均 pair yield 从 '+str(dec['aggregate_effective_pair_yield_baseline'])+' 到 '+str(dec['aggregate_effective_pair_yield_method'])+'，相对增幅 '+f"{dec['aggregate_yield_relative_gain']*100:.3f}%"+'，主要来自 CoffeeServeMug 子层多一个效果对。其余任务无严格增益；这是开发选择证据。事后探索性根级 bootstrap 95% 区间 '+str(boot['descriptive_percentile_95_ci'])+' 包含零，每任务 CI 无法由单根估计，不能作正式显著性或跨场景优势声明。微/宏均值见 MACRO_MICRO_DEVELOPMENT_METRICS.csv。\n'
append+='\n五组同 Source 完整父快照对照：\n\n'+md(base.to_dict('records'),['scheme','actual_control_steps','effective_pairs','root_effective_pairs','nonroot_effective_pairs','nonroot_effect_bins','control_steps_per_effective_pair','runner_seconds_per_effective_pair','collector_GiB_per_effective_pair','renderer'])
append+='\nH1：在这个抽屉根上 Generic single 的单位步效果对产出高于 CFACT single；本轮未建立任务效果候选的优势。H2：CFACT recursive 增加了真实非根状态和子层效果对，但总效果对仍为 5，单位步产出略低于 single；OffDemo 的实际偏离前缀和子层效果也保留，不能将偏离 Source 起点本身当作独有创新。实际步数 452/462/1396/1406 并未配平，OSMesa/EGL 的墙钟不可直接排名；同实际步数与同墙钟的多根正式比较为 NOT_TESTED。collector GiB 仅含相应采集 run 的文件，工作区还包括 prepared Source、源码、审计和诊断发布，不能用整个工作区重复分摊每组效果。\n'
append+='\n归因边界：事件/EffectIntent 第一轮与通用 frontier 第三轮没有通过冻结的聚合产出规则，已实际回滚；第三轮曾产生真实 CoffeeServeMug depth3，但未证明优势。当前接受策略沿用旧 fixed frontier；Source 耗尽后的新合法阶段编译器未完成物理验收。G0-only/G1-only/G2-only、无事件锚定和无恢复的独立消融均未执行，不能把组合变化归因于这些单项。H3 未测试：没有新 LeWM/DINO-WM 训练，不作超 Persistence、机会水平或跨 seeds CI 的结论。\n'
append+='\n主要负面解释：已排除把 issued action 当实际效果、把同一 parent restore 误当跨层连续、以及 shadow render 推进 live physics 等工程错误；仍可能受固定 Source 阶段、保守接触/候选限制和宽控制基线的实际候选数量差异影响，尚无等实际预算多根证据区分这些解释。微波炉父节点已激活和自由体 lid 合同不适配均有具体证据；其余 GPU 阻塞任务没有物理失败证据，不能归咎于算法。\n'
append+='\n时间和资源：首次/末次 guard 为 '+calendar['first_guard_utc']+' / '+calendar['last_guard_utc']+'，区间 '+f"{calendar['guard_calendar_span_seconds']:.1f}s"+'。冻结 7200s 阶段上限由 runner 累计墙钟执行；它不包含独立审计/导出或外层等待，研究日历耗时更长。所有资源触顶均保留拒绝或中断。host available RAM/GPU free memory 为共享服务器观测，未测量每个 worker 的峰值 RSS/VRAM。预算没有扩大，正式资源上限仅为未获准的 ceiling。\n'
append+='\n完整实验索引：ALL_RUN_RECEIPTS.csv 给出每次 guard 的 commit、SHA、backend、状态与进程退出；ALL_ROOT_METRICS.csv 给出实验版本、task、Source ID 和原审计路径；SOURCE20_FROZEN_IDENTITY.json 给出输入 SHA/upstream revision，各 run 的 root_manifest/config 保存派生种子。PROPOSAL_REJECTIONS.csv、HORIZON_AND_CENSOR_COUNTS.csv 和逐对 Parquet 保留拒绝/负例/删失。DELIVERABLE_SHA256.json 保存交付文件与代码补丁的哈希，原生及官方数据文件全量 SHA 在 FULL_READBACK.json 的证据文件中。\n'
for p in [V/'FINAL_RESEARCH_RESULTS_ZH.md',R/'01_development/DEV_FINAL_REPORT.md']:p.write_text(p.read_text()+append)
p=R/'06_training/RUN_STATUS.json';x=read(p);x['reason']='Coverage 0/20 under the frozen two-independent-root contract, scene-disjoint formal partition UNKNOWN, equal-realized-cost multi-root evidence NOT_TESTED; training/data admission failed.';write(p,x)
p=R/'06_training/HANDOFF.md';p.write_text(p.read_text().split('\n最终准入：')[0]+'\n最终准入：0/20 严格递归覆盖，12/20 有至少一个真实控制步，正式场景隔离 UNKNOWN。没有训练 PID/service。当前源码保留接受的 Source 几何 G0 与工程修复；全局科研迭代已达 3 次，第一/三轮实际回滚。本轮不会无限重试 GPU 或重新选择 Source。详见 ../05_validation/FINAL_RESEARCH_RESULTS_ZH.md、TASK20_COVERAGE.csv 与 ALL_RUN_RECEIPTS.csv。\n')
print(json.dumps(dict(status='FINAL_EVIDENCE_ENRICHED',source20_requests=40,complete_roots=8,source20_status_counts=status_counts,all_same_parent_measured_states_equal=initial,pair_windows=len(siblings)),indent=2))
