"""Expose originally captured RGB when a direct worker has no separate camera spool."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
from cfact.execution.observation import array_sha
from cfact.io.serialization import atomic_json
from cfact.io.review_video import QuadReviewWriter
from cfact.storage.layout import inside

IDENTITY_KEYS=('integration_state_sha256','state_sha256','scene_sha256','camera_config_sha256')

def clean_media_from_ledger(run: Path, edge: Path, media: dict, history: Path) -> dict:
    paths=media.get('clean_camera_frame_paths',{})
    if paths and all(len(v)==len(media.get('frame_sim_time_s',[])) for v in paths.values()):return media
    # Only the documented direct-stream case is recoverable; partial spools are errors.
    if any(paths.values()) or media.get('frame_spool'):
        raise ValueError('PARTIAL_CAMERA_SPOOL_CANNOT_BE_RECONSTRUCTED_SILENTLY')
    if media.get('feature_scale',1)!=1 or tuple(media['cameras'])!=QuadReviewWriter.CAMERAS:
        raise ValueError('LEDGER_NATIVE_CAMERA_SCHEMA_MISMATCH')
    times=media['frame_sim_time_s'];captures=media.get('observation_captures',[])
    if len(captures)!=len(times):raise ValueError('LEDGER_MEDIA_CAPTURE_RECEIPTS_MISSING')
    index={}
    for line in (history/'frames.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('segment')!=str(edge):continue
        key=tuple(row.get(k) for k in IDENTITY_KEYS)
        if any(v is None for v in key):raise ValueError('LEDGER_IDENTITY_INCOMPLETE')
        if key in index and index[key]['image_sha256']!=row['image_sha256']:
            raise ValueError('LEDGER_SAME_STATE_RGB_AMBIGUOUS')
        index[key]=row
    identity=hashlib.sha256(json.dumps({'captures':captures,'times':times},sort_keys=True).encode()).hexdigest()
    spool=inside(run,Path('derived_camera_frames')/identity)
    cleaned={camera:[] for camera in media['cameras']};receipt=[]
    for i,(timestamp,capture) in enumerate(zip(times,captures)):
        key=tuple(capture.get(k) for k in IDENTITY_KEYS)
        row=index.get(key)
        if row is None or row.get('capture_integration_steps')!=0 or not np.isclose(row['sim_time_s'],timestamp,atol=1e-9,rtol=0):
            raise ValueError('NO_EXACT_ACTUAL_LEDGER_FRAME_FOR_MEDIA')
        frame=inside(history,row['frame'])
        with np.load(frame,allow_pickle=False) as z:rgb=z['rgb']
        if rgb.shape!=(3,224,224,3) or rgb.dtype!=np.uint8:raise ValueError('LEDGER_RGB_NATIVE224_CONTRACT')
        hashes=[array_sha(image) for image in rgb]
        if hashes!=row['image_sha256']:raise ValueError('LEDGER_RGB_IMAGE_HASH_MISMATCH')
        for j,camera in enumerate(media['cameras']):
            target=spool/camera/f'{i:06d}.png';target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                with Image.open(target) as image:
                    if not np.array_equal(np.asarray(image),rgb[j]):raise ValueError('DERIVED_PNG_CHANGED')
            else:Image.fromarray(rgb[j]).save(target)
            cleaned[camera].append(str(target.relative_to(spool)))
        receipt.append({'media_index':i,'ledger_frame':str(frame.relative_to(run)),
                        'sim_time_s':timestamp,'identity':dict(zip(IDENTITY_KEYS,key)),
                        'image_sha256':hashes})
    atomic_json(spool/'RECONSTRUCTION_RECEIPT.json',{'status':'PASS','origin':'actual_recorded_control_step_RGB',
        'new_env_steps':0,'new_render_calls':0,'lossless_png':True,'frames':receipt})
    return {**media,'frame_spool':str(spool),'clean_camera_frame_paths':cleaned,
        'clean_camera_frame_encoding':'lossless_png_from_actual_recorded_ledger_rgb',
        'camera_observation_provenance':'actual_recorded_control_step_ledger_no_rerender',
        'ledger_reconstruction_receipt':str(spool/'RECONSTRUCTION_RECEIPT.json')}
