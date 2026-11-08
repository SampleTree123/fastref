#!/usr/bin/env python3
"""Build public/private data for the editing harness; CPU, no media conversion.

Technical owner: datasets/editing_agent_v1/README.md. The original graph and
existing actual-ID tasks are read-only. No target or catalog attributes enter
public model inputs; current references are editing conditions only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
VARIANT = ROOT.parent / 'ltx2.5/data/uedata/outfit_camera_first_frame'
V3 = ROOT / 'datasets/history_selection_v3/test.jsonl'
AXES = ('identity', 'hair', 'hair_color', 'wardrobe', 'action', 'camera', 'site')
FORBIDDEN = {'source_history_index', 'source_node_id', 'target_node_id', 'target_video',
             'target_video_path', 'expected_source_video_id', 'attributes', 'supervision',
             'audit_only', 'ground_truth', 'selection_reference', 'selection_references'}


def canonical(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(path):
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        for value in values:
            f.write(canonical(value) + '\n')
    temp.replace(path)


def selection_kind(turn):
    if turn['source_history_index'] == turn['edit_round'] - 1:
        return 'normal_continue'
    text = turn['prompt']['en'].lower()
    if re.search(r'\bvideo\s+\d+\b', text):
        return 'explicit_id'
    if any(term in text for term in ('before ', 'input to ', 'prior to ')):
        return 'event_before'
    if any(term in text for term in ('result of ', 'after ', 'following the first', 'produced by ')):
        return 'event_after'
    return 'content_description_match'


def references(raw, package):
    result = []
    for i, ref in enumerate(raw, 1):
        path = Path(ref.get('absolute_path') or Path(ref.get('media_root', package)) / ref['path'])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f'Missing reference: {path}')
        result.append({'reference_id': f'ref_{i}', 'type': ref['type'], 'path': str(path),
                       'view': ref.get('view', '')})
    return result


def check_public(value, *, is_chain):
    def walk(obj):
        if isinstance(obj, dict):
            bad = FORBIDDEN.intersection(obj)
            if bad:
                raise ValueError(f'Private fields in public input: {sorted(bad)}')
            for x in obj.values():
                walk(x)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)
    walk(value)
    turns = value['turns'] if is_chain else [value['turn']]
    for i, turn in enumerate(turns, 1):
        if is_chain and turn['turn_id'] != i:
            raise ValueError('Nonsequential turn IDs')
        if not turn['user_prompt'].strip():
            raise ValueError('Empty prompt')
        rr = turn['references']
        if [r['reference_id'] for r in rr] != [f'ref_{j}' for j in range(1, len(rr)+1)]:
            raise ValueError('Reference IDs/order differ')
        if any(r.get('reference_purpose', 'editing') != 'editing' for r in rr):
            raise ValueError('Selection reference forbidden')
    if is_chain:
        if not 3 <= len(turns) <= 5:
            raise ValueError('Full chain must have 3–5 edits')
    else:
        history = value['history']
        if not 2 <= len(history) <= 5:
            raise ValueError('Understanding history outside 2–5 videos')
        for i, h in enumerate(history):
            if h['video_id'] != i:
                raise ValueError('Video IDs not contiguous')
            if (i == 0 and h['parent_id'] is not None) or (i > 0 and
                    (type(h['parent_id']) is not int or not 0 <= h['parent_id'] < i)):
                raise ValueError('History parent not an earlier available video')
        if value['turn']['turn_id'] != len(history):
            raise ValueError('Current turn index does not follow history')


def load_graph():
    path = ROOT / 'scripts/dataset/build_selection_v3.py'
    spec = importlib.util.spec_from_file_location('editing_source_graph', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    values = mod.load_graph()
    return mod, values


def chain_signature(chain):
    return tuple(chain['history_node_ids']), tuple(t['source_history_index'] for t in chain['turns'])


def choose_chains(chains, nodes, components, split, count, seed):
    pool = {}; skipped = Counter()
    for c in chains:
        if c['split'] != split or not 3 <= c['length'] <= 5:
            continue
        h = c['history_node_ids']
        if len({nodes[n]['sha256'] for n in h}) != len(h):
            skipped['repeated_video_in_chain'] += 1
            continue
        sig = chain_signature(c)
        if sig not in pool or c['chain_id'] < pool[sig]['chain_id']:
            pool[sig] = c
    remaining = list(pool.values())
    selected = []; axis_count = Counter(); kind_count = Counter(); length_count = Counter()
    component_count = Counter(); root_count = Counter(); select_count = Counter()
    # Greedy rarity coverage at every prefix, deterministic tie breaking. No
    # model answers or quality scores participate in ordering/selection.
    while remaining and len(selected) < count:
        def score(c):
            aa = set(t['axis'] for t in c['turns'])
            ss = set(selection_kind(t) for t in c['turns'])
            return (sum(4/(1+axis_count[a]) for a in aa)
                    + 5/(1+kind_count[c['kind']]) + 5/(1+length_count[c['length']])
                    + sum((7 if k == 'content_description_match' else 3)/(1+select_count[k]) for k in ss)
                    + 4/(1+component_count[components[c['root_node_id']]])
                    + 2/(1+root_count[c['root_node_id']]))
        best = max(remaining, key=lambda c: (score(c), digest([seed, c['chain_id']])))
        remaining.remove(best); selected.append(best)
        axis_count.update(set(t['axis'] for t in best['turns']))
        select_count.update(set(selection_kind(t) for t in best['turns']))
        kind_count[best['kind']] += 1; length_count[best['length']] += 1
        component_count[components[best['root_node_id']]] += 1; root_count[best['root_node_id']] += 1
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} eligible chains, requested {count}')
    return selected, dict(skipped)


def make_chain(chain, nodes, components, package, split, order):
    chain_id = f'{split}_chain_{chain["chain_id"]:05d}'
    public = {'schema': 'editing_agent_chain_v1', 'chain_id': chain_id, 'split': split,
              'root_video': nodes[chain['root_node_id']]['absolute_path'],
              'turns': [{'turn_id': t['edit_round'], 'user_prompt': t['prompt']['en'],
                         'references': references(t['references'], package)} for t in chain['turns']]}
    private = {'chain_id': chain_id, 'candidate_order': order, 'original_chain_id': chain['chain_id'],
               'original_split': chain['split'], 'component_id': components[chain['root_node_id']],
               'topology': chain['kind'], 'length': chain['length'],
               'history_node_ids': chain['history_node_ids'],
               'turns': [{'turn_id': t['edit_round'], 'edge_id': t['edge_id'], 'axis': t['axis'],
                          'selection_kind': selection_kind(t),
                          'expected_source_video_id': t['source_history_index'],
                          'source_node_id': t['source_node_id'], 'target_node_id': t['target_node_id'],
                          'target_video_path': nodes[t['target_node_id']]['absolute_path'],
                          'target_attributes': nodes[t['target_node_id']]['attributes'],
                          'original_prompt_zh': t['prompt']['zh'],
                          'expected_edit_prompt': t['current_edit_prompt'],
                          'sync_from_single_turn': t['sync_from_single_turn']} for t in chain['turns']],
               'route_metric': 'planned_route_agreement',
               'answer_basis': 'Original graph intended parent. Content-based requests require outcome-aware review if generated videos diverge.',
               'public_sha256': digest(public)}
    check_public(public, is_chain=True)
    return public, private


def make_history(history_nodes, parents, completed, nodes, package):
    history = [{'video_id': 0, 'parent_id': None, 'video_path': nodes[history_nodes[0]]['absolute_path'],
                'user_prompt': None, 'references': []}]
    for i, (parent, edge) in enumerate(zip(parents, completed), 1):
        assert edge['source_node_id'] == history_nodes[parent]
        assert edge['target_node_id'] == history_nodes[i]
        history.append({'video_id': i, 'parent_id': parent,
                        'video_path': nodes[history_nodes[i]]['absolute_path'],
                        'user_prompt': edge['prompt']['en'],
                        'references': references(edge['references'], package)})
    return history


def question_from_v3(row, nodes, edges, package):
    audit = row['audit_only']; h = audit['history_node_ids']; ps = audit['completed_parent_video_ids']
    assert not audit['selection_reference_paths']
    completed = [edges[e['edge_id']] for e in audit['completed_edges']]
    edge = edges[audit['current_edge_id']]
    texts = [b['text'] for m in row['model_input']['messages'] if isinstance(m['content'], list)
             for b in m['content'] if b.get('type') == 'text' and b['text'].startswith('Current request: ')]
    assert len(texts) == 1
    public = {'schema': 'editing_agent_understanding_v1', 'sample_id': row['sample_id'], 'split': 'test',
              'history': make_history(h, ps, completed, nodes, package),
              'turn': {'turn_id': len(h), 'user_prompt': audit['selection_prefix'] + edge['prompt']['en'],
                       'references': references(edge['references'], package)}}
    private = {'sample_id': row['sample_id'], 'original_split': row['split'],
               'origin': 'history_selection_v3_existing_text_only_selection',
               'component_id': audit['source_component_id'], 'selection_kind': audit['selection_kind'],
               'selection_axis': audit['selection_axis'], 'current_edit_axis': edge['axis'],
               'history_topology': audit['history_topology'], 'history_node_ids': h,
               'completed_edge_ids': [e['sample_id'] for e in completed],
               'current_edge_id': edge['sample_id'],
               'expected_source_video_id': row['supervision']['source_video_id'],
               'target_video_path': nodes[edge['target_node_id']]['absolute_path'],
               'target_node_id': edge['target_node_id'], 'current_source_node_id': edge['source_node_id'],
               'expected_edit_prompt': edge['prompt']['en'],
               'explicit_history_textual_clue': audit['explicit_history_textual_clue'],
               'history_recomposed': audit['history_recomposed'], 'event_round': audit.get('event_round'),
               'reference_purpose': 'editing_only', 'public_sha256': digest(public)}
    check_public(public, is_chain=False)
    return public, private


def explicit_id_candidates(chains, nodes, edges, components, package):
    result = {}; signatures = set()
    for chain in chains:
        if chain['split'] != 'test':
            continue
        for t in chain['turns'][1:]:
            n = t['edit_round']; h = chain['history_node_ids'][:n]
            if len({nodes[x]['sha256'] for x in h}) != n:
                continue
            parents = [x['source_history_index'] for x in chain['turns'][:n-1]]
            edge = edges[t['edge_id']]; pid = t['source_history_index']
            if edge['target_node_id'] in h:
                continue
            signature = (tuple(h), tuple(parents), pid, edge['sample_id'])
            if signature in signatures:
                continue
            signatures.add(signature)
            sid = f'actual_id_chain_{chain["chain_id"]:05d}_turn_{n}'
            public = {'schema': 'editing_agent_understanding_v1', 'sample_id': sid, 'split': 'test',
                      'history': make_history(h, parents, [edges[x['edge_id']] for x in chain['turns'][:n-1]], nodes, package),
                      'turn': {'turn_id': n, 'user_prompt': f'Please use video {pid} as the starting point. '+edge['prompt']['en'],
                               'references': references(edge['references'], package)}}
            for prior, original in zip(public['history'][1:], chain['turns'][:n-1]):
                prior['editing_prompt'] = prior['user_prompt']
                prior['user_prompt'] = original['prompt']['en']
            private = {'sample_id': sid, 'original_split': 'test', 'origin': 'original_chain_prefix_explicit_id_rewording',
                       'component_id': components[h[0]], 'selection_kind': 'explicit_id', 'selection_axis': None,
                       'current_edit_axis': edge['axis'], 'history_topology': 'fork' if any(p != i for i,p in enumerate(parents)) else 'linear',
                       'history_node_ids': h, 'completed_edge_ids': [x['edge_id'] for x in chain['turns'][:n-1]],
                       'current_edge_id': edge['sample_id'], 'expected_source_video_id': pid,
                       'target_video_path': nodes[edge['target_node_id']]['absolute_path'], 'target_node_id': edge['target_node_id'],
                       'current_source_node_id': edge['source_node_id'], 'expected_edit_prompt': edge['prompt']['en'],
                       'history_recomposed': False, 'event_round': None, 'reference_purpose': 'editing_only',
                       'public_sha256': digest(public)}
            check_public(public, is_chain=False); result[sid] = (public, private)
    return list(result.values())


def stratified_take(candidates, count, seed):
    buckets = defaultdict(list)
    for item in candidates:
        pub, lab = item
        key = (lab['history_topology'], lab['current_edit_axis'], len(pub['history']), lab['component_id'])
        buckets[key].append(item)
    for values in buckets.values():
        values.sort(key=lambda item: digest([seed, item[0]['sample_id']]))
    keys = sorted(buckets, key=lambda key: digest([seed, key])); selected = []
    while len(selected) < count and keys:
        newkeys = []
        for key in keys:
            if len(selected) < count and buckets[key]:
                selected.append(buckets[key].pop())
            if buckets[key]:
                newkeys.append(key)
        keys = newkeys
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} candidates, requested {count}')
    return selected


def chain_counts(labels):
    return {'chains': len(labels), 'turns': sum(x['length'] for x in labels),
            'topologies': dict(Counter(x['topology'] for x in labels)),
            'lengths': dict(Counter(str(x['length']) for x in labels)),
            'axes': dict(Counter(t['axis'] for x in labels for t in x['turns'])),
            'selection_kinds': dict(Counter(t['selection_kind'] for x in labels for t in x['turns'])),
            'components': len({x['component_id'] for x in labels})}


def build(output, debug_count=6, candidates_count=128, understanding_count=1000, seed=42):
    module, values = load_graph()
    chains, nodes, edges, pairs, outgoing, components, node_splits, anchors, attr = values
    sync_counts = Counter()
    for chain in chains:
        for turn in chain['turns']:
            edge = edges[turn['edge_id']]
            old_refs = references(turn['references'], module.PACKAGE)
            new_refs = references(edge['references'], module.PACKAGE)
            old_prompt = turn['prompt']['en']
            prefix = ''
            if turn['source_history_index'] != turn['edit_round'] - 1:
                if ':' not in old_prompt:
                    raise ValueError('Cannot isolate original history-selection prefix')
                prefix = old_prompt.split(':', 1)[0] + ': '
            turn['sync_from_single_turn'] = {'old_reference_paths':[r['path'] for r in old_refs],
                'new_reference_paths':[r['path'] for r in new_refs],
                'references_changed':old_refs!=new_refs, 'original_chain_prompt_en':old_prompt,
                'history_selection_prefix':prefix}
            if [r['path'] for r in old_refs] != [r['path'] for r in new_refs]:
                sync_counts[turn['axis']] += 1
            turn['references'] = edge['references']
            turn['current_edit_prompt'] = edge['prompt']['en']
            turn['prompt'] = {'en':prefix+edge['prompt']['en'], 'zh':turn['prompt']['zh']}
            if edge['axis']=='identity':
                assert all(r['attributes']['identity_id']==edge['target']['attributes']['identity_id'] for r in edge['references'])
    debug, debug_skips = choose_chains(chains, nodes, components, 'train', debug_count, seed)
    test, test_skips = choose_chains(chains, nodes, components, 'test', candidates_count, seed)
    outputs = {}
    for name, chosen in [('debug', debug), ('test', test)]:
        data = [make_chain(c,nodes,components,module.PACKAGE,name,i) for i,c in enumerate(chosen,1)]
        outputs[f'chains_{name}.jsonl'] = [x[0] for x in data]
        outputs[f'private/chains_{name}_labels.jsonl'] = [x[1] for x in data]
    pool = defaultdict(list)
    for row in rows(V3):
        au = row['audit_only']; kind = au['selection_kind']
        if au['selection_reference_paths'] or kind == 'content_reference_match':
            continue
        bucket = au['selection_axis'] if kind == 'content_description_match' else kind
        pool[bucket].append(question_from_v3(row,nodes,edges,module.PACKAGE))
    pool['explicit_id'] = explicit_id_candidates(chains,nodes,edges,components,module.PACKAGE)
    quotas = {'camera':250,'hair_color':250,'event_before':150,'event_after':150,
              'normal_continue':42,'explicit_latest':42,'explicit_id':116}
    if understanding_count != sum(quotas.values()):
        raise ValueError('This reviewed first version defines exactly 1000 questions; use a new dataset version for a different quota.')
    selected = []
    for key,count in quotas.items():
        selected.extend(stratified_take(pool[key],count,seed))
    selected.sort(key=lambda x:digest([seed,x[0]['sample_id']]))
    outputs['understanding_test.jsonl'] = [x[0] for x in selected]
    outputs['private/understanding_test_labels.jsonl'] = [x[1] for x in selected]
    # Independent cross-split, graph and visibility invariants.
    for chain in debug + test:
        h = chain['history_node_ids']
        for i, turn in enumerate(chain['turns'], 1):
            edge = edges[turn['edge_id']]
            assert turn['source_node_id'] == h[turn['source_history_index']] == edge['source_node_id']
            assert turn['target_node_id'] == h[i] == edge['target_node_id']
            assert len(h) == chain['length'] + 1
            assert [(r['type'], r['path']) for r in references(turn['references'], module.PACKAGE)] == [(r['type'], r['path']) for r in references(edge['references'], module.PACKAGE)]
    debug_nodes = {n for x in outputs['private/chains_debug_labels.jsonl'] for n in x['history_node_ids']}
    test_nodes = {n for x in outputs['private/chains_test_labels.jsonl'] for n in x['history_node_ids']}
    for _,lab in selected:
        test_nodes.update(lab['history_node_ids']); test_nodes.add(lab['target_node_id'])
        h=lab['history_node_ids']; pid=lab['expected_source_video_id']; e=edges[lab['current_edge_id']]
        assert e['source_node_id']==h[pid] and e['target_node_id']==lab['target_node_id']
    assert not debug_nodes & test_nodes
    assert not {components[n] for n in debug_nodes} & {components[n] for n in test_nodes}
    assert not {nodes[n]['sha256'] for n in debug_nodes} & {nodes[n]['sha256'] for n in test_nodes}
    assert all(node_splits[n]=='train' for n in debug_nodes)
    assert all(node_splits[n]=='test' for n in test_nodes)
    visible = [digest({k:v for k,v in x.items() if k not in ('sample_id','schema','split')}) for x in outputs['understanding_test.jsonl']]
    assert len(set(visible))==len(visible)
    paths=set()
    for name,rr in outputs.items():
        if name.startswith('private/'):
            continue
        for row in rr:
            if 'root_video' in row:
                paths.add(row['root_video']); turns=row['turns']
            else:
                paths.update(x['video_path'] for x in row['history']);turns=[row['turn'],*row['history']]
            for t in turns:
                paths.update(ref['path'] for ref in t.get('references',[]))
    assert all(Path(p).is_file() for p in paths)
    for name,rr in outputs.items():
        write_rows(output/name,rr)
    counts = {name: {'rows':len(rr),'bytes':(output/name).stat().st_size,'sha256':sha(output/name)} for name,rr in outputs.items()}
    testlabels=outputs['private/chains_test_labels.jsonl']
    audit={'schema':'editing_agent_dataset_audit_v1','status':'structural_checks_passed_no_new_visual_review',
           'created_date':'2026-11-08','seed':seed,'source_variant':'outfit_camera_first_frame',
           'source_files':{str(p):sha(p) for p in [VARIANT/'chains/multi_turn/chains.jsonl',V3,
              *[VARIANT/sp/'samples.jsonl' for sp in ('train','val','test','reserved')]]},
           'builder_sha256':sha(Path(__file__)), 'files':counts,
           'source_chain_reference_path_mismatches_by_axis':dict(sync_counts),
           'chain_sync_policy':'Keep graph parents and history-selection prefix; refresh editing prompt and refs from current same-edge single-turn manifest.',
           'debug':chain_counts(outputs['private/chains_debug_labels.jsonl']),
           'test_prefixes':{str(n):chain_counts(testlabels[:n]) for n in (30,60,100,candidates_count) if n<=candidates_count},
           'understanding':{'questions':len(selected),'quota':quotas,'available':{k:len(v) for k,v in pool.items()},
                            'selection_evidence_axes':dict(Counter(x[1]['selection_axis'] for x in selected if x[1]['selection_kind']=='content_description_match')),
                            'current_edit_axes':dict(Counter(x[1]['current_edit_axis'] for x in selected)),
                            'topologies':dict(Counter(x[1]['history_topology'] for x in selected)),
                            'unique_components':len({x[1]['component_id'] for x in selected}),
                            'unique_history_videos':len({x['video_path'] for p,_ in selected for x in p['history']})},
           'checks':{'debug_train_test_nodes_disjoint':True,'debug_train_test_media_sha_disjoint':True,
                     'debug_train_test_connected_components_disjoint':True,'all_current_and_completed_edges_real':True,
                     'no_private_gt_fields_in_public':True,'no_selection_references':True,
                     'no_duplicate_understanding_inputs':True,'all_public_media_exist':True,
                     'raw_media_or_old_datasets_modified':False},
           'unique_public_media':len(paths),'skips':{'debug':debug_skips,'test':test_skips},
           'limitations':['Content-based history selection is limited to existing textual hair-color and camera descriptions. Other existing v3 content questions required selection-reference media and are excluded.',
                          'All seven current editing axes are covered; this is different from seven selection-description axes.',
                          'Public histories contain actual prior user instructions and true parents, but no GT attribute cards. Generate cards by observing videos.',
                          'End-to-end chain intended parent is an original-graph label. Generated-content drift can make later content descriptions ambiguous; review separately.',
                          'Current camera first-frame images are authorized editing conditions derived from target frames; never inject the target video.',
                          'The understanding set and complete-chain test may reuse held-out components/media. They are two views, not independent datasets.',
                          'No new human visual review was performed during this CPU data build.']}
    write_json(output/'audit.json',audit)
    return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'datasets/editing_agent_v1')
    ap.add_argument('--debug-count',type=int,default=6);ap.add_argument('--candidates-count',type=int,default=128)
    ap.add_argument('--understanding-count',type=int,default=1000);ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args();result=build(args.output,args.debug_count,args.candidates_count,args.understanding_count,args.seed)
    print(json.dumps({k:result[k] for k in ('status','debug','test_prefixes','understanding','checks')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
