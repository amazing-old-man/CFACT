"""Read every published physical trace, tree link, official episode and media file."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

from cfact.execution.snapshot import load_snapshot
from cfact.export.dataset import verify_rows
from cfact.export.lineage_video import _video_info
from cfact.io.serialization import atomic_json
from cfact.sources.manifest import sha256_file
from cfact.storage.integrity import verify_release
from cfact.storage.layout import inside


def _read(run, path):
    return json.loads(inside(run, path).read_text())


def _trace(run, path):
    with np.load(inside(run, path), allow_pickle=False) as values:
        result = {key: values[key].copy() for key in values.files}
    count = len(result['commanded_actions'])
    for key in ('sim_time', 'robot_qpos', 'robot_qvel', 'actual_eef_pose', 'task_success'):
        if len(result[key]) != count + 1 or not np.isfinite(result[key]).all():
            raise ValueError(f'Invalid measured state alignment: {path}:{key}')
    if not np.isfinite(result['commanded_actions']).all():
        raise ValueError('Nonfinite commanded actions')
    if count and not np.allclose(np.diff(result['sim_time']), .05, atol=1e-5, rtol=0):
        raise ValueError('Measured transition clock does not match 20 Hz')
    return result


def verify_run(run: Path) -> dict:
    record = _read(run, 'release.json')
    if record['status'] != 'COMPLETE':
        raise ValueError('Export is incomplete')
    verify_release(run, record)
    rows = pq.read_table(inside(run, record['episode_tree_map'])).to_pylist()
    official = verify_rows(run, rows)
    lineages = [json.loads(line) for line in inside(run, record['lineage_paths']).read_text().splitlines() if line]
    reverse = _read(run, 'mappings/edge_segment_episode_lineage.json')
    tree_reports, physical, transitions, leaves = [], 0, 0, set()
    for path in record['trees']:
        tree = _read(run, path)
        nodes = {n['node_id']: n for n in tree['nodes']}
        edges = {e['edge_id']: e for e in tree['edges']}
        node_states = {}
        for nid, node in nodes.items():
            with np.load(inside(run, node['state']), allow_pickle=False) as saved:
                state = saved['sim_state'].copy()
            context = _read(run, node['context'])
            snap = context['snapshot']
            snapshot = load_snapshot(inside(run, snap['path']))
            if sha256_file(inside(run, snap['path'])) != snap['sha256'] or not np.array_equal(snapshot['states'], state):
                raise ValueError('Saved backend snapshot differs from actual node')
            node_states[nid] = state
        starts = {}
        outcomes = []
        for eid, edge in edges.items():
            if edge['parent_node'] not in nodes or (edge['actual_child'] and edge['actual_child'] not in nodes):
                raise ValueError('Edge has a dangling parent/child')
            traces = {}
            for kind in ('local', 'fixed_continuation'):
                relative = edge['path'] + '/' + kind + '/execution.npz'
                if not inside(run, relative).is_file():
                    continue
                trace = _trace(run, relative)
                traces[kind] = trace
                count = len(trace['commanded_actions'])
                transitions += count
                matches = [r for r in rows if r['edge_id'] == eid and r['segment'] == kind]
                if len(matches) != bool(count):
                    raise ValueError('Measured segment does not map once to its official episode')
                flags = trace['task_success'].astype(bool)
                outcomes.append({'edge_id': eid, 'segment': kind, 'actions': count,
                    'official_success_first_state': next((i for i, x in enumerate(flags) if x), None),
                    'official_success_ever': bool(flags.any()), 'official_success_last': bool(flags[-1])})
            if 'local' in traces:
                physical += 1
                local = traces['local']
                parent = edge['parent_node']
                beginning = {k: local[k][0] for k in ('sim_time', 'robot_qpos', 'robot_qvel', 'actual_eef_pose')}
                if parent in starts and any(not np.allclose(beginning[k], starts[parent][k], atol=1e-8, rtol=0) for k in beginning):
                    raise ValueError('Sibling actions did not start at the same measured parent')
                starts[parent] = beginning
                if not np.isclose(local['sim_time'][0], node_states[parent][0], atol=1e-9, rtol=0):
                    raise ValueError('Local trace does not start at parent time')
                if edge['actual_child'] and not np.isclose(local['sim_time'][-1], node_states[edge['actual_child']][0], atol=1e-9, rtol=0):
                    raise ValueError('Actual child time differs from local terminal')
                if 'fixed_continuation' in traces:
                    for key in ('sim_time', 'robot_qpos', 'robot_qvel', 'actual_eef_pose'):
                        if not np.allclose(local[key][-1], traces['fixed_continuation'][key][0], atol=1e-8, rtol=0):
                            raise ValueError('Fixed continuation does not start at local child')
            if not any(e['parent_node'] == edge['actual_child'] for e in edges.values()):
                leaves.add(eid)
        index = _read(run, f"lerobot/by_tree/{tree['task']}/{tree['root_id']}/index.json")
        expected_rows = [r for r in rows if r['root_id'] == tree['root_id']]
        if index['episodes'] != expected_rows or index['edges'] != tree['edges'] or index['is_independent_dataset']:
            raise ValueError('Tree index differs from task dataset mapping')
        tree_readback = None
        if record['by_tree_exports'] and expected_rows:
            receipt = _read(run, f"lerobot/by_tree_exports/{tree['task']}/{tree['root_id']}.receipt.json")
            tree_readback = verify_rows(run, receipt['rows'])
        tree_reports.append({'task': tree['task'], 'root_id': tree['root_id'], 'status': tree['status'],
            'edges': len(edges), 'nodes': len(nodes), 'recursive_parents': sum(bool(n['parent_edge']) and n['decision_node'] for n in nodes.values()),
            'outcomes': outcomes, 'independent_tree_readback': tree_readback})
    if {line['edge_ids'][-1] for line in lineages} != leaves or len(lineages) != len(leaves):
        raise ValueError('Not every leaf has exactly one lineage')
    video_count, video_bytes = 0, 0
    for line in lineages:
        playlist = _read(run, line['playlist'])
        if playlist != line:
            raise ValueError('Lineage index differs from its playlist')
        for eid in line['edge_ids']:
            if line['lineage_id'] not in reverse[eid]['lineages']:
                raise ValueError('Reverse edge to lineage mapping is incomplete')
        if record['full_media']:
            path = inside(run, 'media/lineages/' + line['lineage_id'] + '.mp4')
            receipt = _read(run, str(path.relative_to(run)) + '.json')
            info = _video_info(path)
            expected = sum(round((s['end_s'] - s['start_s']) * receipt['fps']) for s in line['segments'])
            if info['frame_count'] != expected or receipt['frame_count'] != expected or sha256_file(path) != receipt['sha256']:
                raise ValueError('Full lineage video frame count or content mismatch')
            video_bytes += path.stat().st_size
    # Decode all actual prefix, edge and lineage media, including unused fixed tails.
    for path in sorted((run / 'media').rglob('*.mp4')):
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path), '-f', 'null', '-'], check=True)
        video_count += 1
    report = {'status': 'PASS' if physical else 'NO_EDGES', 'official': official, 'trees': tree_reports,
        'physical_edges': physical, 'measured_transitions': transitions, 'leaf_lineages': len(lineages),
        'media_mode': 'full' if record['full_media'] else 'playlist', 'decoded_review_videos': video_count,
        'full_lineage_extra_disk_bytes': video_bytes}
    atomic_json(run / 'reports/verification.json', report)
    return report
