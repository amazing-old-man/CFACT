import json,os,pathlib,subprocess,sys
r=pathlib.Path(os.environ['CFACT_WORKSPACE']).resolve();args=sys.argv[1:];snap=r/'runtime/orchestrator_v6';python='/mnt/ssd2/yanghc/CFACT/.runtime/venv/bin/python';module='cfact'
if args and args[0]=='prepare-isolated':args=args[1:];module='cfact.sources.isolated_prepare'
elif args and (args[0] in {'export-recursive','render-lineage'} or args[:2] in [['recursive','export'],['recursive','verify']]):
 snap=r/'runtime/publisher_v6'
 if args[0] in {'export-recursive','render-lineage'}:python='/mnt/ssd2/yanghc/CFACT/.runtime/export-venv/bin/python'
elif args[:2] in [['recursive','run'],['recursive','generate'],['recursive','resume'],['recursive','smoke']] and '--queue' in args:
 qpath=pathlib.Path(args[args.index('--queue')+1]).resolve()
 if not qpath.is_relative_to(r):raise ValueError('Queue must be inside round')
 q=json.loads(qpath.read_text());f=pathlib.Path(q['orchestrator_freeze']).resolve();snap=pathlib.Path(json.loads(f.read_text())['frozen_path']).resolve()
 if not f.is_relative_to(r) or not snap.is_relative_to(r):raise ValueError('Freeze must be inside round')
env=dict(os.environ,CFACT_SOURCE_ROOT=str(snap),PYTHONPATH=str(snap/'src'))
raise SystemExit(subprocess.call([python,'-B','-m',module,*args],env=env))
