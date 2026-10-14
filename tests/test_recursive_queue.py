from cfact.collection.round_runner import choose_gpu, ledger_steps, fleet_lanes
import pytest


def test_eight_worker_fleet_bounds_each_gpu_and_partitions_requests_once():
    queue=dict(fleet_workers=8,verified_gpu_uuids=['gpu0','gpu1'],
               cpu_fallback_validated=True,jobs=list(range(128)))
    lanes=fleet_lanes(queue)
    assert len(lanes)==8
    assert sorted(job for lane in lanes for job in lane['jobs'])==queue['jobs']
    slots=[(lane['verified_gpu_uuids'][0],lane['gpu_slot'])
           for lane in lanes if not lane['force_cpu']]
    assert sorted(slots)==[('gpu0',0),('gpu0',1),('gpu1',0),('gpu1',1)]
    assert all(lane['gpu_concurrency']==2 for lane in lanes[:4])
    assert all(lane['force_cpu'] and lane['verified_gpu_uuids']==[]
               and lane['gpu_slot'] is None for lane in lanes[4:])


def test_extra_cpu_lanes_require_validation_and_gpu_ids_cannot_duplicate():
    queue=dict(fleet_workers=8,verified_gpu_uuids=['gpu0','gpu1'],jobs=[])
    with pytest.raises(ValueError,match='CPU_LANES_REQUIRE_VALIDATED_BACKEND'):
        fleet_lanes(queue)
    queue.update(cpu_fallback_validated=True,verified_gpu_uuids=['gpu0','gpu0'])
    with pytest.raises(ValueError,match='VERIFIED_PHYSICAL_GPU_MAPPING'):
        fleet_lanes(queue)


def test_two_gpu_slots_still_supported_for_the_one_gpu_proof_queue():
    lanes=fleet_lanes(dict(fleet_workers=2,verified_gpu_uuids=['gpu1'],jobs=[1,2,3]))
    assert [lane['gpu_slot'] for lane in lanes]==[0,1]
    assert all(not lane['force_cpu'] and lane['gpu_concurrency']==2 for lane in lanes)

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
