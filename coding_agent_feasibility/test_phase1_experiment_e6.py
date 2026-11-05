"""No real GPU required; test E6's isolation, accounting and aggregation."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_experiment_e6 as e6


class FakeRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.cache = {0: None, 1: None}
        self.calls = []

    def erase(self, base, n=2):
        with self.lock:
            self.cache = {i: None for i in range(n)}
        return [{'id_slot': i, 'n_erased': 1} for i in range(n)]

    def url_json(self, base, method, path, payload=None, timeout=None):
        if path == '/slots':
            return [{'id': 0, 'n_ctx': 4096, 'is_processing': False},
                    {'id': 1, 'n_ctx': 4096, 'is_processing': False}]
        if path == '/health':
            return {'status': 'ok'}
        if path == '/slots/0?action=erase':
            with self.lock:
                self.cache[0] = None
            return {'id_slot': 0, 'n_erased': 100}
        raise AssertionError(f'unexpected {method} {path}')

    def exec(self, base, prompt, tokens, timeout, slot=0, cache=True):
        if 'Tool result:' in prompt:
            name = 'A2'
        elif prompt.startswith('PROMPT_E6_A_'):
            name = 'A1'
        elif prompt.startswith('PROMPT_E6_B_'):
            name = 'B'
        elif prompt.startswith('PROMPT_E6_C_'):
            name = 'C'
        else:
            raise AssertionError('unexpected prompt '+str(prompt))
        with self.lock:
            before = self.cache[slot]
            self.calls.append((name, slot, before))
        time.sleep(0.006 if name in ('A1', 'A2') else 0.01)
        with self.lock:
            self.cache[slot] = name
        hit = 110 if name == 'A2' and before == 'A1' and cache else 0
        return {'ok': True, 'seconds': 0.01, 'completion': 'done',
                'timings': {'cache_n': hit, 'prompt_n': 130-hit,
                            'prompt_ms': 2 if hit else 20, 'predicted_n': tokens}}

    def prompt(self, base, name, target):
        return f'PROMPT_{name}', target, 'tokenize'

    def patches(self):
        from unittest.mock import patch as p
        return [p.object(e6.ex, 'eraser', side_effect=self.erase),
                p.object(e6.ex, 'execute', side_effect=self.exec),
                p.object(e6.kp, 'url_json', side_effect=self.url_json),
                p.object(e6.kp, 'build_prompt', side_effect=self.prompt),
                p.object(e6.ex, 'check_local_server_context', return_value={
                    'server_context_capacity_arg': 4096,
                    'process_command': ('123 /project/llama-server -c 4096 -np 2 '
                                        '--kv-unified --kv-unified-per-slot 4096 '
                                        '--cache-ram 0 --slot-save-path /tmp/slots')}),
                ]


class E6Tests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeRuntime()
        self.ps = self.fake.patches()
        for p in self.ps:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.ps)])

    def test_idle_keep_reuses_a_prefix(self):
        arm = e6.run_idle_arm('fake', 'PROMPT_E6_A_2200', 'PROMPT_E6_B_LONG',
                              'keep', 0, 64, 128, 2, 1, 'A_2200', 'long')
        self.assertTrue(arm['all_ok'])
        self.assertEqual(arm['a2_cache_n'], 110)
        self.assertEqual([k[:2] for k in self.fake.calls],
                         [('A1',0),('B',1),('A2',0)])
        self.assertTrue(arm['a2_started_after_b'])

    def test_idle_evict_forces_reprefill_without_reassigning_slot(self):
        arm = e6.run_idle_arm('fake', 'PROMPT_E6_A_2200', 'PROMPT_E6_B_LONG',
                              'evict', 0, 64, 128, 2, 1, 'A_2200', 'long')
        self.assertTrue(arm['all_ok'])
        self.assertEqual(arm['a2_cache_n'], 0)
        self.assertEqual([k[:2] for k in self.fake.calls],
                         [('A1',0),('B',1),('A2',0)])
        self.assertEqual(arm['erase_a']['id_slot'],0)

    def test_wait_is_observed(self):
        arm = e6.run_idle_arm('fake', 'PROMPT_E6_A_2200', 'PROMPT_E6_B_LONG',
                              'keep', .055, 64, 128, 2, 1, 'A_2200', 'long')
        self.assertGreaterEqual(arm['a2_start_s'] - arm['a1_end_s'], .049)

    def test_failure_not_misreported_as_faster_success(self):
        a = {'budget': 4096, 'a_profile': 'A_2200', 'b_profile': 'long',
             'wait_s': 3, 'policy': 'keep', 'repeat': 1,
             'all_ok': False, 'completed': 1, 'total_makespan_s': .1,
             'b_capacity_error': True}
        b = {**a, 'policy': 'evict', 'all_ok': True, 'completed': 2,
             'b_capacity_error': False, 'total_makespan_s': 3.5}
        stats = e6.group_summary([a,b], 'idle')
        ka = next(s for s in stats if s['policy']=='keep')
        self.assertIsNone(ka['median_success_makespan_s'])
        self.assertEqual(ka['success_rate'],0)
        self.assertEqual(ka['capacity_error_arms'], 1)
        pair = e6.paired_summary([a,b], 'idle')[0]
        self.assertEqual(pair['paired_total'],1)
        self.assertEqual(pair['paired_both_success'],0)
        self.assertIsNone(pair['median_keep_minus_evict_makespan_s'])
        self.assertEqual(pair['sum_completed_keep'], 1)
        self.assertEqual(pair['sum_completed_evict'], 2)

    def test_capacity_classification(self):
        self.assertTrue(e6.is_capacity_error({'error':'HTTP 500: Context size has been exceeded.'}))
        self.assertFalse(e6.is_capacity_error({'error':'HTTP 404 Not found'}))

    def test_prompt_hash_check_aggregation(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            for budget in (4096, 6144):
                d = p/str(budget)
                d.mkdir()
                (d/'config.json').write_text(json.dumps({
                    'kv_budget': budget,
                    'measured_prompts': {'A_2200': {'sha256':'identical'}},
                }),encoding='utf-8')
                (d/'idle_trials.jsonl').write_text(json.dumps({
                    'policy':'keep', 'all_ok':True, 'completed':2,
                    'repeat':1, 'wait_s':0, 'a_profile':'A_2200',
                    'b_profile':'long', 'total_makespan_s':2,
                })+'\n', encoding='utf-8')
            result = e6.aggregate_many([p/'4096',p/'6144'],p/'all')
            self.assertEqual(result,0)
            self.assertTrue((p/'all'/'idle_by_budget.csv').exists())
            self.assertEqual(len((p/'all'/'idle_by_budget.csv').read_text().splitlines()),3)

    def test_end_to_end_quick_run(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)/'e6'
            rc = e6.main(['--expect-ctx','4096', '--mode','both',
                          '--waits','0', '--mixes','short_long',
                          '--b-profiles','short', '--a-tokens','1200',
                          '--repeats','1','--sample-interval','0',
                          '--out-dir',str(out)])
            self.assertEqual(rc,0)
            for name in ['idle_trials.csv','idle_groups.csv','idle_pairs.csv',
                         'sweep_trials.csv','sweep_groups.csv','sweep_pairs.csv',
                         'config.json']:
                self.assertTrue((out/name).exists(), name)
            self.assertEqual(len(e6.load_jsonl(out/'idle_trials.jsonl')),2)
            self.assertEqual(len(e6.load_jsonl(out/'sweep_trials.jsonl')),2)


if __name__ == '__main__':
    unittest.main()
