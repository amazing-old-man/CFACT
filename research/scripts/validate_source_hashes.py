import hashlib,json,time
from pathlib import Path
R=Path('/mnt/ssd1/yanghc/CFACT_Research');started=time.monotonic();lock=json.loads((R/'manifests/SOURCE20_FROZEN_IDENTITY.json').read_text());rows=[]
for t in lock['tasks']:
 p=Path(t['prepared_hdf5']);h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 rows.append(dict(task=t['task'],path=str(p),expected=t['prepared_sha256'],actual=h.hexdigest(),status='PASS' if h.hexdigest()==t['prepared_sha256'] else 'FAIL'))
record=dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'FAIL',scope='all twenty current development prepared inputs',rows=rows,wall_seconds=time.monotonic()-started);(R/'05_validation/SOURCE20_IMMUTABLE_HASHES.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(status=record['status'],files_checked=len(rows),wall_seconds=record['wall_seconds'])))
