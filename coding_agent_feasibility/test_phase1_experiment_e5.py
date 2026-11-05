#!/usr/bin/env python3
"""E5 client-policy unit tests using a deterministic fake 2-slot server."""
import pathlib
import csv
import json
import tempfile
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase1_experiment_e5 as e5


class FakeBackend:
    def __init__(self, b_time=.10, c_time=.10):
        self.lock=threading.Lock()
        self.previous={0:None,1:None}
        self.active={0:False,1:False}
        self.calls=[]
        self.b_time=b_time
        self.c_time=c_time

    def erase(self,base,n=2):
        with self.lock:
            self.previous={0:None,1:None}
            self.active={0:False,1:False}
            self.calls=[]
        return [{'id_slot':i,'n_erased':1} for i in range(n)]

    def execute(self,base,prompt,output_tokens,timeout,slot=0,cache=True):
        if 'Tool result:' in prompt:
            name='A2'
        elif prompt=='A_PROMPT':
            name='A1'
        elif prompt=='B_PROMPT':
            name='B'
        elif prompt=='C_PROMPT':
            name='C'
        else:
            raise AssertionError('unrecognized prompt '+prompt[:80])
        with self.lock:
            if self.active[slot]:
                raise AssertionError('overlapping requests on same slot')
            self.active[slot]=True
            previous=self.previous[slot]
            self.calls.append((name,slot,time.monotonic()))
        duration={'A1':.012,'A2':.012,'B':self.b_time,'C':self.c_time}[name]
        time.sleep(duration)
        with self.lock:
            self.previous[slot]=name
            self.active[slot]=False
        reuse = 100 if name=='A2' and cache and previous=='A1' else 0
        return {'ok':True,'seconds':duration,'completion':'a response',
                'timings':{'cache_n':reuse,'prompt_n':120-reuse,
                           'prompt_ms':1 if reuse else 20,'predicted_n':output_tokens},
                'tokens_cached':100, 'tokens_evaluated':120}


class Tests(unittest.TestCase):
    def arm(self, policy, tool_wait, b_time=.10,c_time=.10):
        fake=FakeBackend(b_time,c_time)
        with patch.object(e5.ex,'eraser',side_effect=fake.erase), \
             patch.object(e5.ex,'execute',side_effect=fake.execute):
            res=e5.run_arm('http://fake','A_PROMPT','B_PROMPT','C_PROMPT',
                           policy,tool_wait,4,8,2,1,'short_short',poll_s=.001)
        return res,fake.calls

    def test_preserve_keeps_cache_and_queues_c(self):
        r,calls=self.arm('preserve',.03)
        self.assertTrue(r['all_ok'])
        self.assertEqual(r['a2_cache_n'],100)
        self.assertGreater(r['c_queue_s'],.09)
        self.assertEqual(r['timings']['B']['slot'],1)
        self.assertEqual(r['timings']['C']['slot'],1)
        self.assertEqual(r['timings']['A2']['slot'],0)

    def test_reuse_overwrites_slot0_cache(self):
        r,calls=self.arm('reuse',.03)
        self.assertTrue(r['all_ok'])
        self.assertEqual(r['a2_cache_n'],0)
        self.assertEqual(r['timings']['B']['slot'],0)
        self.assertEqual(r['timings']['C']['slot'],1)
        self.assertGreater(r['a2_queue_s'],.05)

    def test_ttl_tool_finishes_before_threshold(self):
        r,calls=self.arm('ttl:0.10',.02)
        self.assertTrue(r['all_ok'])
        self.assertFalse(r['ttl_eviction_triggered'])
        self.assertEqual(r['a2_cache_n'],100)
        self.assertEqual(r['timings']['C']['slot'],1)
        self.assertEqual(r['timings']['A2']['slot'],0)

    def test_ttl_releases_slot0_only_when_competing_work_waits(self):
        r,calls=self.arm('ttl:0.02',.10,b_time=.12,c_time=.12)
        self.assertTrue(r['all_ok'])
        self.assertTrue(r['ttl_eviction_triggered'])
        self.assertEqual(r['timings']['C']['slot'],0)
        self.assertEqual(r['a2_cache_n'],0)
        self.assertGreaterEqual(r['timings']['A2']['start_s'],r['timings']['C']['end_s']-.002)
        self.assertGreater(r['a2_queue_s'],0)
        self.assertTrue(any(d['action']=='ttl_expired_reuse_slot0' for d in r['decisions']))

    def test_no_evict_if_b_done_before_ttl(self):
        r,calls=self.arm('ttl:0.10',.21,b_time=.02,c_time=.03)
        self.assertTrue(r['all_ok'])
        self.assertEqual(r['timings']['C']['slot'],1)
        self.assertFalse(r['ttl_eviction_triggered'])
        self.assertEqual(r['a2_cache_n'],100)

    def test_random_balanced_sample_hidden_from_decision_records(self):
        a=e5.draw_episode_delays([0,.5,3,8],12,42)
        self.assertEqual(sorted(a),[0]*3+[.5]*3+[3]*3+[8]*3)
        self.assertEqual(a,e5.draw_episode_delays([0,.5,3,8],12,42))
        self.assertNotEqual(a,e5.draw_episode_delays([0,.5,3,8],12,43))
        r,_=self.arm('ttl:0.02',.1)
        self.assertTrue(all('hidden_tool_delay_s' not in d for d in r['decisions']))
        self.assertTrue(all('tool_ready_future' not in d for d in r['decisions']))

    def test_cli_end_to_end_writes_all_tables(self):
        fake=FakeBackend(.015,.015)
        def fake_url(base,method,path,timeout=8):
            if path == '/health':
                return {'status':'ok'}
            if path == '/slots':
                return [{'id':0,'n_ctx':4096,'is_processing':False},
                        {'id':1,'n_ctx':4096,'is_processing':False}]
            raise AssertionError((method,path))
        def fake_prompt(base,label,target):
            if label=='E5_A':
                return 'A_PROMPT',1200,'tokenize'
            if label.startswith('E5_B_'):
                return 'B_PROMPT',target,'tokenize'
            if label.startswith('E5_C_'):
                return 'C_PROMPT',target,'tokenize'
            raise AssertionError(label)
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(e5.ex,'eraser',side_effect=fake.erase), \
             patch.object(e5.ex,'execute',side_effect=fake.execute), \
             patch.object(e5.ex,'check_local_server_context',return_value={
                'server_context_capacity_arg':8192}), \
             patch.object(e5.kp,'url_json',side_effect=fake_url), \
             patch.object(e5.kp,'build_prompt',side_effect=fake_prompt):
            out=pathlib.Path(temp)/'e5_out'
            retval=e5.main(['--episodes-per-mix','2','--wait-values','0','.025',
                            '--mixes','short_short','--ttls','.015',
                            '--bc-predict','8','--a-predict','4',
                            '--out-dir',str(out),'--seed','7'])
            self.assertEqual(retval,0)
            for filename in ('trials.csv','policy_summary.csv','episode_comparison.csv',
                             'trials.jsonl','summary.json'):
                self.assertTrue((out/filename).is_file())
            with (out/'trials.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),6)
            self.assertEqual({x['policy'] for x in rows},
                             {'preserve','reuse','ttl:0.015'})
            self.assertEqual({float(x['hidden_tool_delay_s']) for x in rows},
                             {0,.025})
            with (out/'episode_comparison.csv').open() as f:
                comparisons=list(csv.DictReader(f))
            self.assertEqual(len(comparisons),6)
            result=json.loads((out/'summary.json').read_text())
            self.assertEqual(result['experiment'],'E5')

    def test_group_and_static_hindsight_oracle(self):
        trials=[]
        for policy in ('preserve','reuse','ttl:0.02'):
            r,_=self.arm(policy,.10,b_time=.12,c_time=.12)
            trials.append(r)
        groups,episodes=e5.analyze(trials,['short_short'],
                                   ['preserve','reuse','ttl:0.02'])
        self.assertEqual(len(groups),3)
        self.assertEqual(len(episodes),3)
        self.assertEqual(sum(1 for g in groups if g['all_ok_arms']==1),3)
        self.assertTrue(all(x['delta_vs_best_static_s'] is not None for x in episodes))
        self.assertEqual(len(e5.flatten(trials[0])),22)


if __name__=='__main__':
    unittest.main()
