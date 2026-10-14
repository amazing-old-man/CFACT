import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from cfact.storage.ledger_media import clean_media_from_ledger,IDENTITY_KEYS
from cfact.execution.observation import array_sha
from cfact.io.review_video import QuadReviewWriter

def capture(tmp_path):
    history=tmp_path/'history';history.mkdir();edge=tmp_path/'edge';edge.mkdir()
    rgb=np.arange(3*224*224*3,dtype=np.uint8).reshape(3,224,224,3)
    np.savez_compressed(history/'frame.npz',rgb=rgb)
    row={**dict(zip(IDENTITY_KEYS,['a','b','c','d'])),'segment':str(edge),'sim_time_s':1.,'frame':'frame.npz','capture_integration_steps':0,'image_sha256':[array_sha(x) for x in rgb]}
    (history/'frames.jsonl').write_text(json.dumps(row)+'\n')
    media={'cameras':list(QuadReviewWriter.CAMERAS),'frame_sim_time_s':[1.],'frame_phase':['local'],'observation_captures':[row.copy()],'clean_camera_frame_paths':{c:[] for c in QuadReviewWriter.CAMERAS},'frame_spool':None,'feature_scale':1}
    return history,edge,rgb,media

def test_direct_media_is_losslessly_bound_to_actual_ledger_without_rerender(tmp_path):
    history,edge,rgb,media=capture(tmp_path);result=clean_media_from_ledger(tmp_path,edge,media,history)
    for i,camera in enumerate(result['cameras']):
        with Image.open(Path(result['frame_spool'])/result['clean_camera_frame_paths'][camera][0]) as image:assert np.array_equal(np.asarray(image),rgb[i])
    receipt=json.loads((Path(result['frame_spool'])/'RECONSTRUCTION_RECEIPT.json').read_text())
    assert receipt['new_env_steps']==receipt['new_render_calls']==0
    assert all(v==[] for v in media['clean_camera_frame_paths'].values())

def test_different_scene_or_corrupted_rgb_is_rejected(tmp_path):
    history,edge,rgb,media=capture(tmp_path);media['observation_captures'][0]['scene_sha256']='wrong'
    with pytest.raises(ValueError,match='NO_EXACT'):clean_media_from_ledger(tmp_path,edge,media,history)
    media['observation_captures'][0]['scene_sha256']='c';np.savez_compressed(history/'frame.npz',rgb=np.zeros_like(rgb))
    with pytest.raises(ValueError,match='IMAGE_HASH'):clean_media_from_ledger(tmp_path,edge,media,history)

def test_equal_physics_receipts_keep_distinct_edge_pixels_and_do_not_write_acquisition(tmp_path):
    run=tmp_path/'acquisition';run.mkdir()
    history,edge,rgb,media=capture(run)
    second=run/'other_edge';second.mkdir()
    different=np.bitwise_xor(rgb,1)
    np.savez_compressed(history/'second.npz',rgb=different)
    row=json.loads((history/'frames.jsonl').read_text())
    other={**row,'segment':str(second),'frame':'second.npz','image_sha256':[array_sha(x) for x in different]}
    with (history/'frames.jsonl').open('a') as stream:stream.write(json.dumps(other)+'\n')
    # Real capture receipts describe physics/cameras, not the RGB array hash.
    same_capture={key:row[key] for key in IDENTITY_KEYS}
    media['observation_captures']=[same_capture]
    before={str(p.relative_to(run)):p.read_bytes() for p in run.rglob('*') if p.is_file()}
    derived=tmp_path/'publication'/'.derived_camera_frames'
    first=clean_media_from_ledger(run,edge,media,history,derived_root=derived)
    second_media=clean_media_from_ledger(run,second,media,history,derived_root=derived)
    assert first['frame_spool']!=second_media['frame_spool']
    for result,expected in [(first,rgb),(second_media,different)]:
        for i,camera in enumerate(result['cameras']):
            with Image.open(Path(result['frame_spool'])/result['clean_camera_frame_paths'][camera][0]) as image:
                assert np.array_equal(np.asarray(image),expected[i])
    assert before=={str(p.relative_to(run)):p.read_bytes() for p in run.rglob('*') if p.is_file()}

def test_changed_cached_png_is_still_rejected(tmp_path):
    history,edge,rgb,media=capture(tmp_path)
    result=clean_media_from_ledger(tmp_path,edge,media,history,derived_root=tmp_path/'published_frames')
    camera=result['cameras'][0];target=Path(result['frame_spool'])/result['clean_camera_frame_paths'][camera][0]
    Image.fromarray(np.zeros_like(rgb[0])).save(target)
    with pytest.raises(ValueError,match='DERIVED_PNG_CHANGED'):
        clean_media_from_ledger(tmp_path,edge,media,history,derived_root=tmp_path/'published_frames')
