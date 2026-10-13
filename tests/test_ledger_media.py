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
