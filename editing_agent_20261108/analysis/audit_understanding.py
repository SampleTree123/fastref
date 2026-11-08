#!/usr/bin/env python3
"""Read saved understanding evidence; writes only an analysis report, never model inputs."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ANALYSIS = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def worker_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def require(condition, code):
    if not condition:
        raise ValueError(code)


def media_blocks(messages):
    return [b for m in messages if isinstance(m.get('content'), list)
            for b in m['content'] if b.get('type') in ('image', 'video')]


def check_media(call, expected):
    blocks = media_blocks(call['messages'])
    require([(b['type'], b.get('path'), b.get('media_role')) for b in blocks] ==
            [('video', p, 'history') for p in expected], 'unexpected_model_message_media')
    actual = call.get('media', [])
    require(len(actual) == len(expected), 'encoded_media_count_differs')
    require(all(m.get('kind') == 'video' and m.get('role') == 'history' for m in actual),
            'encoded_media_kind_or_role_differs')


def correction_messages(call):
    return call['messages']+[
        {'role':'assistant','content':call['raw_output'].replace('<|im_end|>','')},
        {'role':'user','content':'The previous tool call was not executed. Error: '+call['validation_error']+
         '. Correct it once using exactly one of the available tools and all required parameters. Do not invent a default ID.'}]


def parsed_attempt(call, contracts):
    if call['status']=='invalid':
        try:
            contracts.parse_tool_calls(call['raw_output'],call['tools'])
        except (ValueError,TypeError,KeyError) as error:
            require(str(error)==call['validation_error'],'recorded_format_error_differs')
            return None
        raise ValueError('invalid_observation_or_selection_was_actually_parseable')
    require(call['status']=='valid','unknown_call_status')
    parsed=contracts.parse_tool_calls(call['raw_output'],call['tools'])[0]
    require(parsed==call['parsed_call'],'saved_parse_differs_from_raw')
    return parsed


def public_history(items):
    return {'videos': [{k: h[k] for k in ('video_id', 'parent_id', 'observed', 'uncertain')} for h in items],
            'edits': [{'source_video_id': h['parent_id'], 'output_video_id': h['video_id'],
                       'user_prompt': h.get('user_prompt', ''),
                       'editing_prompt': h.get('editing_prompt', h.get('user_prompt', '')),
                       'review': {'note': 'Existing recorded history; no comparison review available.'}}
                      for h in items if h['video_id'] > 0]}


def validate_observation(h, observed, root, version, contracts, observe_prompt):
    video = Path(h['video_path']); stat = video.stat()
    key = worker_digest({'path': str(video), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                         'observation_version': version})
    cache = root / 'observed_cards' / key
    cached, card = read(cache/'cached_card.json'), read(cache/'card.json')
    require(cached == card, 'committed_observation_cards_differ')
    require(cached['video_path'] == h['video_path'], 'card_source_path_differs')
    for field in ('observed', 'uncertain', 'observation_identity'):
        require(observed.get(field) == cached.get(field), 'history_card_differs_from_observation_'+field)
    calls = [read(p) for p in sorted(cache.glob('observe_call_*.json'))]
    valid = []
    expected_messages=[{'role':'system','content':observe_prompt},{'role':'user','content':[
        {'type':'text','text':"Describe this video's actual visible contents."},
        {'type':'video','path':str(video.resolve()),'media_role':'history'}]}]
    require(1<=len(calls)<=2,'observation_exceeds_one_correction')
    for call in calls:
        require(not valid and call['messages']==expected_messages,'observation_messages_include_unexpected_context')
        require(call['messages'][0] == {'role':'system', 'content':observe_prompt}, 'observation_system_prompt_differs')
        require(call['messages'][1] == {'role':'user', 'content':[
            {'type':'text','text':"Describe this video's actual visible contents."},
            {'type':'video','path':str(video.resolve()),'media_role':'history'}]}, 'observation_not_video_only_instruction')
        check_media(call, [str(video.resolve())])
        require(call['tools'] == contracts.card_tools(), 'observation_tools_differ')
        parsed=parsed_attempt(call,contracts)
        if parsed is not None:
            valid.append((parsed, call))
        else:
            expected_messages=correction_messages(call)
    require(len(valid) == 1, 'observation_has_no_unique_valid_raw_call')
    parsed, call = valid[0]
    require(parsed['name'] == 'record_video_card', 'observation_wrong_tool')
    require(parsed['arguments'] == {'observed':cached['observed'], 'uncertain':cached['uncertain']},
            'card_not_equal_to_parsed_model_observation')
    return call['media'][0].get('content_sha256')


def audit_case(row, root, contracts, config, prompts):
    directory = root / 'cases' / row['sample_id']
    result = read(directory/'worker_result.json')
    require(result['status']=='ok', 'audit_case_requires_completed_case')
    identity = read(directory/'input_identity.json')
    require(identity['version'] == result['version'], 'worker_case_version_differs')
    require(identity['public_input_sha256'] == worker_digest(row), 'public_input_hash_differs')
    history = read(directory/'observed_history.json')
    require(len(history)==len(row['history']), 'history_length_differs')
    hashes = {}
    for h, observed in zip(row['history'], history):
        for key, value in h.items():
            require(observed.get(key)==value, 'public_history_field_changed_'+key)
        hashes[h['video_path']] = validate_observation(h, observed, root, result['version'], contracts, prompts['observe'])
    ids = [h['video_id'] for h in history]
    answer = result['result']
    require(type(answer['source_video_id']) is int and answer['source_video_id'] >= 0 and answer['source_video_id'] in ids,
            'selected_id_not_available')
    expected_refs = [r['reference_id'] for r in row['turn']['references']]
    require(answer['reference_ids']==expected_refs, 'editing_reference_ids_not_exact_order')
    require(read(directory/'decision/edit_request.json')==answer, 'saved_edit_request_differs_from_result')
    expected_text = 'History:\n'+contracts.canonical(public_history(history))+'\nCurrent user request:\n'+row['turn']['user_prompt']
    inspected, selected, n_calls, invalid_calls = [], [], 0, 0
    expected_messages=[{'role':'system','content':prompts['select']},{'role':'user','content':expected_text}]
    for path in sorted((directory/'decision').glob('select_call_*.json')):
        call=read(path); n_calls+=1
        require(not selected and call['messages']==expected_messages,'selection_messages_include_unexpected_context')
        require(call['messages'][0]=={'role':'system','content':prompts['select']}, 'selection_system_prompt_differs')
        require(call['messages'][1]=={'role':'user','content':expected_text}, 'selection_public_history_or_request_differs')
        media_paths = [history[i]['video_path'] for i in inspected]
        check_media(call, media_paths)
        for p, info in zip(media_paths, call.get('media', [])):
            require(info.get('content_sha256') == hashes[p], 'inspected_media_hash_differs_from_observed_video')
        tools = contracts.selection_tools(ids, allow_inspect=not inspected and bool(config['max_inspect_calls']),
                                           max_inspect_videos=config['max_inspect_videos'])
        require(call['tools']==tools, 'selection_tools_differ')
        parsed=parsed_attempt(call,contracts)
        if parsed is not None:
            if parsed['name']=='inspect_history':
                require(not inspected, 'multiple_inspections')
                inspected=parsed['arguments']['video_ids']
                require(all(type(i)is int and i in ids for i in inspected), 'inspected_id_not_available')
                content=[]
                for i in inspected:
                    content.extend([{'type':'text','text':f'Existing history video {i}:'},
                                    {'type':'video','path':history[i]['video_path'],'media_role':'history'}])
                content.append({'type':'text','text':'Select the source video now. No more inspection calls are available.'})
                expected_messages=call['messages']+[contracts.assistant_call(parsed),
                    {'role':'tool','name':'inspect_history','content':contracts.canonical({'viewed_video_ids':inspected})},
                    {'role':'user','content':content}]
            else:
                require(parsed['name']=='select_source_video', 'unexpected_selection_tool')
                selected.append(parsed['arguments']['source_video_id'])
        else:
            invalid_calls+=1
            require(invalid_calls<=1,'selection_exceeds_one_correction')
            expected_messages=correction_messages(call)
    require(n_calls>0 and selected==[answer['source_video_id']], 'final_selection_not_equal_to_model_call')
    require(inspected==answer['inspected_video_ids'], 'inspection_receipt_differs')
    require(answer['selection_model_calls']==n_calls and answer['selection_corrections']==invalid_calls,
            'selection_call_counters_differ')
    valid_edits=[]
    for path in sorted((directory/'decision').glob('edit_call_*.json')):
        call=read(path)
        tools=contracts.edit_tools(answer['source_video_id'],row['turn']['references'])
        require(call['tools']==tools, 'editing_tools_differ')
        if call['status']=='valid':
            parsed=contracts.parse_tool_calls(call['raw_output'],tools)[0]
            require(parsed==call['parsed_call'], 'editing_saved_parse_differs_from_raw')
            valid_edits.append(parsed['arguments'])
    require(valid_edits==[{k:answer[k] for k in ('source_video_id','prompt','reference_ids')}], 'final_edit_not_equal_to_model_call')
    return {'inspected':bool(inspected), 'history_videos':len(history), 'selection_calls':n_calls}


def context(run):
    frozen=run/'frozen_sources/qwen3.8'
    manifest=read(run/'freeze_manifest.json')['files']
    relatives=['scripts/editing_agent/contracts.py','configs/editing_agent/config.json',
               'configs/editing_agent/prompts/select.txt','configs/editing_agent/prompts/observe.txt',
               'datasets/editing_agent_v1/understanding_test.jsonl']
    for relative in relatives:
        matched=[value for path,value in manifest.items() if path.endswith('/qwen3.8/'+relative)]
        require(len(matched)==1 and hashlib.sha256((frozen/relative).read_bytes()).hexdigest()==matched[0],
                'frozen_audit_input_hash_differs_'+relative)
    path=frozen/'scripts/editing_agent/contracts.py'
    spec=importlib.util.spec_from_file_location('frozen_editing_audit_contracts',path)
    contracts=importlib.util.module_from_spec(spec);spec.loader.exec_module(contracts)
    config=read(frozen/'configs/editing_agent/config.json')
    prompts={name:(frozen/f'configs/editing_agent/prompts/{name}.txt').read_text().strip() for name in ('select','observe')}
    dataset=frozen/'datasets/editing_agent_v1/understanding_test.jsonl'
    return list(map(json.loads,dataset.read_text().splitlines())),contracts,config,prompts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ANALYSIS.parent/'formal_v1')
    args=parser.parse_args(); run=args.run.resolve()
    rows,contracts,config,prompts=context(run)
    require(len(rows)==1000 and len({r['sample_id'] for r in rows})==1000,'expected_1000_unique_questions')
    root=run/'understanding'; counts=Counter(expected=1000,ok=0,error=0,pending=0,audit_passed=0,audit_failed=0,
                                            completed_with_inspection=0,history_card_occurrences_checked=0); details=[]
    for row in rows:
        path=root/'cases'/row['sample_id']/'worker_result.json'
        status=read(path).get('status') if path.is_file() else 'pending'
        counts[status]+=1
        item={'sample_id':row['sample_id'],'status':status}
        if status=='ok':
            try:
                evidence=audit_case(row,root,contracts,config,prompts)
                item.update(audit='passed',**evidence);counts['audit_passed']+=1
                counts['completed_with_inspection']+=evidence['inspected']
                counts['history_card_occurrences_checked']+=evidence['history_videos']
            except (ValueError,KeyError,TypeError,OSError,IndexError) as exc:
                item.update(audit='failed',finding=str(exc));counts['audit_failed']+=1
        details.append(item)
    result={'created_unix':time.time(),'run':str(run),'scope':'1000 frozen public questions; runtime evidence, not answer accuracy',
            'counts':dict(counts),'checks':['Final selected ID is an existing nonnegative history ID.',
            'Reference IDs preserve the complete current reference order; final request equals parsed raw model call.',
            'Selection messages start with exactly allowed observed history/current text; no reference media before inspection.',
            'Inspection followups contain only requested existing history videos; logged media hashes match observation logs.',
            'Historical public IDs/parents/paths/prompts/references are unchanged; cards equal cached cards and parsed raw video-only observation outputs.'],
            'limitations':['This audit reads no private labels and does not measure source-selection accuracy.',
            'Failed/pending cases remain in the 1000-case denominator and are not certified by completed-case checks.',
            'Evidence is from saved messages, model raw outputs and receipts; this does not prove absence of unlogged process/file access.',
            'Video bytes are not rehashed or decoded; stat-based cache identity and recorded media hashes are checked for consistency.',
            'Observation wording being model-generated does not establish visual correctness or absence of hallucinations.',
            'The live run can advance during the audit; counts are a sequential read snapshot, not a globally locked instant.'],
            'cases':details}
    out=ANALYSIS/'understanding_runtime_audit.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'report':str(out),'findings':[d for d in details if d.get('audit')=='failed'][:10]},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
