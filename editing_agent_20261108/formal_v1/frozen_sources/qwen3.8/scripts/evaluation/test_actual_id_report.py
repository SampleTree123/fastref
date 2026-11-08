#!/usr/bin/env python3
"""Report boundaries: partial coverage, stale identities and uncertain RGB evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from build_actual_id_report import load_snapshot, choose_cases, render_report, render_cases
from evaluation_inputs_actual_id import task_prompt_text, tools_for_history, score_prediction
from media_inputs import sha_file
from run_model_eval_actual_id import digest


def xml(number):
    return f'<tool_call><function=select_source_video><parameter=source_video_id>{number}</parameter></function></tool_call>'


def row(sample='fixture', axis='identity'):
    return {'sample_id':sample,'audit':{'edit_round':2,'selection_kind':'content_reference_match',
        'original_split':'test','history_topology':'fork','selection_axis':axis},
        'supervision':{'source_video_id':0,'resolved_parent_video_id':0},
        'audit_only':{'explicit_history_textual_clue':False,'exact_selection_reference_in_completed_history':False,'singleton_only_candidate':False},
        'model_input':{'tools':tools_for_history(2),'messages':[{'role':'system','content':task_prompt_text()},
        {'role':'user','content':[{'type':'text','text':'Available history. Video 0 is the original video.'},
        {'type':'video','path':'/tmp/fixture_original.mp4','media_role':'history'},
        {'type':'text','text':'Completed edit 1: started from video 0. The user instruction was: Example edit.'},
        {'type':'video','path':'/tmp/fixture_result.mp4','media_role':'history'},
        {'type':'text','text':'Current request: Choose the original.'}]}]}}


class ActualIDReport(unittest.TestCase):
    def setup(self, folder):
        root=Path(folder);dataset=root/'eval.jsonl';rows=[row(),row('second')]
        dataset.write_text(''.join(json.dumps(r)+'\n' for r in rows));results=root/'results'
        for profile in ('low','high'):
            target=results/profile;target.mkdir(parents=True)
            identity={'dataset_sha256':sha_file(dataset),'config':{'name':profile,'selection_contract':'actual_id_v1'},'prepare_only':False}
            fingerprint=digest(identity)
            (target/'expected_run.json').write_text(json.dumps({'identity':identity,'run_fingerprint':fingerprint}))
            (target/'records').mkdir()
            for item in rows:
                raw=xml(0)
                record={'sample_id':item['sample_id'],'profile':profile,'row_sha256':digest(item),
                        'run_fingerprint':fingerprint,'status':'ok','raw_output':raw,'scores':score_prediction(raw,item)}
                (target/'records'/(item['sample_id']+'.json')).write_text(json.dumps(record))
        return dataset,results,rows

    def test_partial_and_incompatible_never_look_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset,results,rows=self.setup(folder)
            (results/'high/records/second.json').unlink()
            path=results/'low/records/fixture.json';record=json.loads(path.read_text());record['run_fingerprint']='old-run';path.write_text(json.dumps(record))
            snap=load_snapshot(dataset,results)
            self.assertFalse(snap['all_inference_succeeded'])
            self.assertEqual(snap['profiles']['high']['groups']['all']['missing'],1)
            self.assertEqual(snap['profiles']['low']['groups']['all']['incompatible'],1)
            self.assertEqual(snap['paired']['answered_in_both'],0)
            text=render_report(snap,False,dataset,results,choose_cases(snap))
            self.assertIn('当前为部分结果',text)
            self.assertNotIn('零运行错误。',text)

    def test_uncertain_rgb_and_first_observation_stay_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset,results,rows=self.setup(folder)
            flags=Path(folder)/'flags.json';flags.write_text(json.dumps({'schema':'qwen_selection_v3_independent_review_flags_v1',
                'default':{'rgb_reviewed':False,'rgb_review_status':'not_reviewed'},'samples':{'fixture':{'rgb_reviewed':True,
                'rgb_review_status':'unresolved_visual','initial_blind_match_gt':False,'followup_rgb_status':'still_uncertain','risks':['temporal_ambiguity']}}}))
            for profile in ('low','high'):
                p=results/profile/'records/fixture.json';record=json.loads(p.read_text());record['raw_output']=xml(1);record['scores']=score_prediction(record['raw_output'],rows[0]);p.write_text(json.dumps(record))
            snap=load_snapshot(dataset,results,flags)
            selected=choose_cases(snap)
            self.assertEqual(selected[0]['sample_id'],'fixture')
            text=render_cases(snap,selected,True)
            self.assertIn('画面尚无法可靠判定',text)
            self.assertIn('首次人工判断与标准答案：不一致',text)
            self.assertIn('still_uncertain',text)
            self.assertIn('不能替代正式全量结论',text)
            self.assertEqual(snap['profiles']['low']['groups']['review/rgb_review_status/not_reviewed']['expected'],1)

    def test_negative_id_does_not_gain_credit_from_stale_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset,results,rows=self.setup(folder)
            p=results/'low/records/fixture.json';record=json.loads(p.read_text());record['raw_output']=xml(-1);record['scores']=score_prediction(record['raw_output'],rows[0]);p.write_text(json.dumps(record))
            summary=results/'summary.json';summary.write_text(json.dumps({'dataset_sha256':'old-dataset','profiles':{}}))
            snap=load_snapshot(dataset,results,saved_summary=summary)
            self.assertEqual(snap['profiles']['low']['groups']['all']['strict_correct'],1)
            self.assertFalse(snap['records']['low']['fixture']['scores']['valid'])
            self.assertTrue(any('题目清单不同' in warning for warning in snap['warnings']))
            self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    unittest.main()
