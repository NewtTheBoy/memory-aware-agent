#!/usr/bin/env python3
"""Tests for E4 client orchestration; no GPU or llama-server needed."""
import pathlib
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase1_experiment_e4 as e4


class E4Tests(unittest.TestCase):
    def setUp(self):
        self.cache = {0: None, 1: None}
        self.calls = []
        self.guard = threading.Lock()

    def fake_erase(self, base, n=2):
        with self.guard:
            self.cache = {0: None, 1: None}
        return [{'id_slot': i, 'n_erased': 0} for i in range(n)]

    def fake_execute(self, base, prompt, output_tokens, timeout, slot=0, cache=True):
        if 'Tool result:' in prompt:
            kind = 'A2'
        elif prompt == 'A_CONTEXT':
            kind = 'A1'
        elif prompt == 'B_CONTEXT':
            kind = 'B'
        elif prompt == 'C_CONTEXT':
            kind = 'C'
        else:
            raise AssertionError('unexpected prompt: '+prompt[:70])
        with self.guard:
            previous = self.cache[slot]
            self.calls.append((kind,slot,time.monotonic()))
        time.sleep(0.015 if kind in ('A1','A2') else 0.04)
        with self.guard:
            self.cache[slot] = kind
        hit = 100 if kind == 'A2' and previous == 'A1' and cache else 0
        return {'ok': True, 'seconds': 0.04,
                'timings': {'cache_n': hit, 'prompt_n': 120 - hit,
                            'prompt_ms': 1 if hit else 20, 'predicted_n': output_tokens},
                'completion': 'model answer', 'tokens_cached': 100,
                'tokens_evaluated': 120}

    def run_trial(self, policy, wait_s=0):
        with patch.object(e4.ex, 'eraser', side_effect=self.fake_erase), \
             patch.object(e4.ex, 'execute', side_effect=self.fake_execute):
            return e4.run_arm('http://fake', 'A_CONTEXT','B_CONTEXT','C_CONTEXT',
                              policy, wait_s, 4, 8, 4, 1, 'short_short')

    def test_preserve_runs_b_and_c_on_slot_one(self):
        out = self.run_trial('preserve')
        self.assertTrue(out['all_ok'])
        self.assertEqual(out['completed'],3)
        self.assertEqual({k:out['timings'][k]['slot'] for k in ('A1','B','C','A2')},
                         {'A1':0,'B':1,'C':1,'A2':0})
        self.assertGreaterEqual(out['timings']['C']['start_s'],
                                out['timings']['B']['end_s'] - 0.0001)
        self.assertEqual(out['a2_cache_n'],100)

    def test_reuse_runs_b_on_a_slot_and_blocks_resume(self):
        out = self.run_trial('reuse',wait_s=0.01)
        self.assertTrue(out['all_ok'])
        self.assertEqual({k:out['timings'][k]['slot'] for k in ('A1','B','C','A2')},
                         {'A1':0,'B':0,'C':1,'A2':0})
        self.assertGreaterEqual(out['timings']['A2']['start_s'],
                                out['timings']['B']['end_s'] - 0.0001)
        self.assertGreater(out['a2_queue_s'],0)
        self.assertEqual(out['a2_cache_n'],0)

    def test_wait_respected(self):
        out = self.run_trial('preserve',wait_s=0.08)
        self.assertGreaterEqual(out['timings']['A2']['start_s']-out['a1_end_s'],0.075)
        self.assertLess(out['a2_queue_s'],0.03)

    def test_median_pairing_and_csv_rows(self):
        a = self.run_trial('preserve')
        b = self.run_trial('reuse')
        grouped, paired = e4.aggregate([a,b],['short_short'],[0])
        self.assertEqual(len(grouped),2)
        self.assertEqual(paired[0]['paired_valid_n'],1)
        self.assertAlmostEqual(paired[0]['median_preserve_minus_reuse_makespan_s'],
                               a['total_makespan_s']-b['total_makespan_s'],places=5)
        cols, rows = e4.csv_rows([a,b])
        self.assertEqual(len(rows),2)
        self.assertIn('a2_cache_n',cols)
        self.assertNotIn('timings',rows[0])


if __name__=='__main__':
    unittest.main()
