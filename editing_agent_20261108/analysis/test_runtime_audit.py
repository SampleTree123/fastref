#!/usr/bin/env python3
"""Clone a completed runtime receipt under analysis, then forge one fact at a time."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import audit_understanding as audit


RUN=audit.ANALYSIS.parent/'formal_v1'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False))


class AuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows,cls.contracts,cls.config,cls.prompts=audit.context(RUN)
        for row in rows:
            p=RUN/'understanding/cases'/row['sample_id']/'worker_result.json'
            if p.is_file():
                result=audit.read(p)
                if result.get('status')=='ok' and result['result'].get('inspected_video_ids') and len(row['turn']['references'])>=2:
                    cls.row=row;cls.result=result;break
        else:
            raise RuntimeError('Need one completed inspection case with two references for forged evidence tests')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='runtime_fixture_',dir=audit.ANALYSIS)
        self.root=Path(self.tmp.name)
        original=RUN/'understanding/cases'/self.row['sample_id']
        self.case=self.root/'cases'/self.row['sample_id']
        shutil.copytree(original,self.case)
        self.caches=[]
        for h in self.row['history']:
            video=Path(h['video_path']);stat=video.stat()
            key=audit.worker_digest({'path':str(video),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                                     'observation_version':self.result['version']})
            cache=self.root/'observed_cards'/key
            shutil.copytree(RUN/'understanding/observed_cards'/key,cache)
            self.caches.append(cache)

    def tearDown(self):
        self.tmp.cleanup()

    def check(self):
        return audit.audit_case(self.row,self.root,self.contracts,self.config,self.prompts)

    def test_unmodified_real_evidence_passes(self):
        result=self.check()
        self.assertTrue(result['inspected'])

    def test_negative_source_is_rejected(self):
        p=self.case/'worker_result.json';v=audit.read(p);v['result']['source_video_id']=-1;write(p,v)
        with self.assertRaisesRegex(ValueError,'selected_id_not_available'):self.check()

    def test_reordered_editing_refs_are_rejected(self):
        p=self.case/'worker_result.json';v=audit.read(p);v['result']['reference_ids'].reverse();write(p,v)
        with self.assertRaisesRegex(ValueError,'editing_reference_ids_not_exact_order'):self.check()

    def test_current_reference_pixels_in_first_selection_are_rejected(self):
        p=sorted((self.case/'decision').glob('select_call_*.json'))[0]
        v=audit.read(p);v['messages'].append({'role':'user','content':[
            {'type':'image','path':self.row['turn']['references'][0]['path'],'media_role':'image_reference'}]});write(p,v)
        with self.assertRaisesRegex(ValueError,'selection_messages_include_unexpected_context'):self.check()

    def test_inspection_of_unrequested_media_is_rejected(self):
        p=sorted((self.case/'decision').glob('select_call_*.json'))[-1]
        v=audit.read(p)
        block=next(b for m in v['messages'] if isinstance(m.get('content'),list) for b in m['content'] if b.get('type')=='video')
        block['path']=self.row['turn']['references'][0]['path'];write(p,v)
        with self.assertRaisesRegex(ValueError,'selection_messages_include_unexpected_context'):self.check()

    def test_extra_answer_text_in_observation_input_is_rejected(self):
        p=sorted(self.caches[0].glob('observe_call_*.json'))[0]
        v=audit.read(p);v['messages'].append({'role':'user','content':'The expected attributes are black hair and a red shirt.'});write(p,v)
        with self.assertRaisesRegex(ValueError,'observation_messages_include_unexpected_context'):self.check()

    def test_card_changed_to_unobserved_claim_is_rejected_even_if_copies_agree(self):
        hpath=self.case/'observed_history.json';history=audit.read(hpath)
        claim='FORGED: guaranteed expected target attributes from an external answer'
        history[0]['observed']['hair_color']=claim;write(hpath,history)
        for name in ('cached_card.json','card.json'):
            p=self.caches[0]/name;v=audit.read(p);v['observed']['hair_color']=claim;write(p,v)
        with self.assertRaisesRegex(ValueError,'card_not_equal_to_parsed_model_observation'):self.check()

    def test_history_video_parent_mutation_is_rejected(self):
        p=self.case/'observed_history.json';v=audit.read(p);v[1]['parent_id']=1;write(p,v)
        with self.assertRaisesRegex(ValueError,'public_history_field_changed_parent_id'):self.check()


if __name__=='__main__':unittest.main(verbosity=2)
