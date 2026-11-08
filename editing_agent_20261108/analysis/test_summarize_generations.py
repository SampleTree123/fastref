import json
import os
from pathlib import Path
import tempfile
import unittest
from summarize_generations import summarize


class GenerationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'root.mp4';self.source.write_bytes(b'fixture')
        self.row={'chain_id':'a','root_video':str(self.source),'turns':[{'turn_id':1},{'turn_id':2}]}
        self.case=self.root/'cases/a'
    def tearDown(self): self.temp.cleanup()
    def put(self,path,value):
        p=self.case/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    def generate(self,tid,source):
        output=self.case/f'turn_{tid:02d}/generation/edited.mp4';output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(b'generated')
        self.put(f'turn_{tid:02d}/generation/editing_generation.json',{'status':'ok','video_path':str(output),'request':{'source_video':str(source)},'execution_sha256':str(tid),'timings':{'total_seconds':10,'encoding_seconds':2,'denoise_seconds':7,'decode_seconds':1},'peak_allocated_bytes':1024})
        return output
    def test_generated_but_not_reviewed(self):
        self.put('turn_01/edit_request.json',{'source_video_id':0});self.generate(1,self.source)
        d=summarize(self.root,[self.row],1);c=d['counts']
        self.assertEqual((c['selected_turns'],c['generated_turns'],c['recorded_completed_turns']),(1,1,0))
        self.assertEqual(c['generated_not_recorded_turns'],1);self.assertEqual(c['chains_in_progress'],1)
    def test_selection_before_generation_and_unstarted(self):
        self.put('turn_01/select_call_00.json',{'status':'valid','parsed_call':{'name':'select_source_video','arguments':{'source_video_id':0}}})
        other={**self.row,'chain_id':'b'};d=summarize(self.root,[self.row,other],2);c=d['counts']
        self.assertEqual((c['selected_turns'],c['generated_turns'],c['recorded_completed_turns']),(1,0,0))
        self.assertEqual(c['selected_not_generated_turns'],1);self.assertEqual(c['chains_not_started'],1)
    def test_completed_chain_receipts_survive_resumed_worker_seconds(self):
        first=self.generate(1,self.source);second=self.generate(2,first)
        for tid,src in [(1,0),(2,1)]:
            self.put(f'turn_{tid:02d}/progress.json',{'request':{'source_video_id':src},'review':{'card':{'observed':{}},'review':{'fulfillment':'failed'}}})
        self.put('state.json',{'status':'completed','completed_turns':2,'history':{'edits':[{'output_video_id':1},{'output_video_id':2}], 'videos':[{'video_id':0,'video_path':str(self.source)},{'video_id':1,'video_path':str(first)},{'video_id':2,'video_path':str(second)}]}})
        self.put('worker_result.json',{'status':'ok','seconds':.01})
        d=summarize(self.root,[self.row],1)
        self.assertEqual(d['counts']['chains_completed'],1);self.assertEqual(d['counts']['recorded_completed_turns'],2)
        self.assertEqual(d['source_counts_generated_turns'],{'root_video':1,'actual_generated_history':1})
        self.assertEqual(d['transition_counts_generated_turns'],{'first_edit':1,'continue_latest':1})
        self.assertEqual(d['native_timings_seconds']['total_seconds']['sum'],20)
        self.assertEqual(d['issues'],[])

    def test_failed_worker_and_newer_failed_state_remain_failed(self):
        worker=self.put('worker_result.json',{'status':'error','error':'review failed'})
        state=self.put('state.json',{'status':'failed','completed_turns':0})
        os.utime(worker, ns=(1000000000,1000000000))
        os.utime(state, ns=(2000000000,2000000000))
        d=summarize(self.root,[self.row],1)
        self.assertEqual(d['counts']['chains_failed'],1)
        self.assertEqual(d['counts']['chains_in_progress'],0)
        self.assertEqual(d['chains'][0]['state_status'],'failed')
        self.assertEqual(d['chains'][0]['last_worker_status'],'error')

if __name__=='__main__':unittest.main()
