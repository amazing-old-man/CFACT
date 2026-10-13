import json
from pathlib import Path
import pandas as pd
from cfact.execution.snapshot import load_snapshot,snapshot_differences
R=Path('/mnt/ssd1/yanghc/CFACT_Research');paths={
'CFACT_EFFECT_RECURSIVE':R/'02_baselines/MATCHED_PARENT_CPU/CFACT_EFFECT_RECURSIVE/run',
'CFACT_EFFECT_SINGLE':R/'02_baselines/MATCHED_PARENT_CPU/CFACT_EFFECT_SINGLE/run',
'GENERIC_CF_SINGLE':R/'02_baselines/MATCHED_DRAWER/runs/GENERIC_CF_SINGLE',
'GENERIC_CF_OFFDEMO':R/'02_baselines/MATCHED_DRAWER/runs/GENERIC_CF_OFFDEMO'};parents={}
for s,p in paths.items():
 rp=next(p.glob('data/*/*/*/root_result.json'));parents[s]=load_snapshot(rp.parent/'nodes/n_root/parent_snapshot.json')
ref=parents['CFACT_EFFECT_RECURSIVE'];diff={s:snapshot_differences(ref,p) for s,p in parents.items()};receipt=dict(status='PASS' if all(not v for v in diff.values()) else 'FAIL',same_source_parent_complete_value_state=diff,scheme_count=4,offdemo_additional_start='Separate actual 10-step perturbed child, full snapshot retained',identical_realized_costs=False,primary_cost_unit='All actual control steps',wall_limitation='Both CFACT methods used OSMesa CPU, generic methods used EGL GPU0. Physics/controller/camera parameters and complete parent are matched; wall comparisons remain separate.',code_limitation='../02_baselines/BASELINE_CODE_PROVENANCE_NOTE.json');(R/'05_validation/BASELINE_SAME_PARENT_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(dict(status=receipt['status'],difference_counts={k:len(v) for k,v in diff.items()},scheme_count=4),indent=2))
