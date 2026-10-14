from cfact.collection.round_runner import choose_gpu, ledger_steps

def inventory(temp=70,util=10,free=12000):
    return {'gpu':{'gpus':[dict(uuid='verified',index=1,temperature_c=temp,
        utilization_pct=util,free_mib=free,total_mib=98000,compute_mode='Default')]}}

def test_shared_resources_require_verified_mapping_and_reserves_without_temperature_admission():
    assert choose_gpu(inventory(),[]) is None
    assert choose_gpu(inventory(temp=91),['verified'])['index']==1
    assert choose_gpu(inventory(util=81),['verified']) is None
    assert choose_gpu(inventory(free=5000),['verified']) is None
    assert choose_gpu(inventory(),['verified'])['index']==1

def test_incremental_cost_reads_actual_last_step_and_does_not_count_export(tmp_path):
    import json
    p=tmp_path/'history/development/OpenDrawer/request/frames.jsonl';p.parent.mkdir(parents=True)
    p.write_text('\n'.join(json.dumps({'control_step_total':v}) for v in [1,2,100])+'\n')
    exported=tmp_path/'published/history';exported.mkdir(parents=True)
    (exported/'frames.jsonl').write_text(json.dumps({'control_step_total':999})+'\n')
    assert ledger_steps(tmp_path,'OpenDrawer','request')==100
    assert ledger_steps(tmp_path,'OpenDrawer','unknown')==0
    formal=tmp_path/'history/formal/OpenDrawer/request/frames.jsonl';formal.parent.mkdir(parents=True)
    formal.write_text(json.dumps({'control_step_total':42})+'\n')
    assert ledger_steps(tmp_path,'OpenDrawer','request','formal')==42
