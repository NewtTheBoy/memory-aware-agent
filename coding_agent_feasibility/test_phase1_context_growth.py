"""Offline tests; no GPU or llama.cpp required."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_context_growth as e9


class FakeRuntime:
    def __init__(self):
        self.token_calls = 0
        self.erasures = []
        self.requests = []
        self.fail_above = None

    def url_json(self, base, method, path, payload=None, timeout=0):
        if path == '/tokenize':
            self.token_calls += 1
            return {'tokens': list(range(len(payload['content'].split())))}
        if path == '/health':
            return {'status': 'ok'}
        if path == '/slots':
            return [{'id': 0, 'n_ctx': 4096, 'is_processing': False},
                    {'id': 1, 'n_ctx': 4096, 'is_processing': False}]
        if '?action=erase' in path:
            self.erasures.append(path)
            return {'id_slot': int(path.split('/')[2].split('?')[0])}
        raise AssertionError(f'Unexpected endpoint {path}')

    def perform(self, base, prompt, n_predict, timeout,
                barrier=None, cache_prompt=False, id_slot=None):
        if barrier is not None:
            barrier.wait(timeout=10)
        count = len(prompt.split())
        self.requests.append((id_slot, count, cache_prompt))
        if self.fail_above is not None and count > self.fail_above and id_slot == 1:
            return {'ok': False, 'seconds': .05,
                    'error': 'HTTP 500: Context size has been exceeded'}
        return {'ok': True, 'seconds': .02,
                'timings': {'cache_n': min(count, 100), 'prompt_n': max(count - 100, 0),
                            'prompt_ms': 12., 'predicted_n': 16},
                'truncated': False}


class TestContextGrowth(unittest.TestCase):
    def setUp(self):
        self.fake = FakeRuntime()
        self.patch_url = patch.object(e9.kp, 'url_json', self.fake.url_json)
        self.patch_perform = patch.object(e9.kp, 'perform', self.fake.perform)
        self.patch_gpu = patch.object(e9.kp, 'gpu_query', return_value=[
            {'index': 0, 'used_mib': 3200, 'util_percent': 50}])
        self.patch_url.start()
        self.patch_perform.start()
        self.patch_gpu.start()
        self.addCleanup(self.patch_url.stop)
        self.addCleanup(self.patch_perform.stop)
        self.addCleanup(self.patch_gpu.stop)
        self.prompts = e9.make_prompts('http://localhost:8080', [180, 360, 540])

    def test_fixed_vs_growing_tokens_and_prefix(self):
        for agent in e9.AGENTS:
            grow = self.prompts[agent]['growing']
            fixed = self.prompts[agent]['fixed']
            self.assertTrue(grow[1]['text'].startswith(grow[0]['text']))
            self.assertTrue(grow[2]['text'].startswith(grow[1]['text']))
            for i, target in enumerate([180, 360, 540]):
                self.assertLess(abs(grow[i]['tokens'] - target), 25)
                self.assertEqual(fixed[i]['sha256'],
                                 e9.hashlib.sha256(fixed[i]['text'].encode()).hexdigest())
            self.assertTrue(all(125 <= x['tokens'] <= 250 for x in fixed))
            self.assertEqual(grow[0]['text'], fixed[0]['text'])
            self.assertNotEqual(fixed[1]['text'], fixed[0]['text'])

    def test_successful_pair_and_reports(self):
        t = e9.run_trial('http://localhost:8080', self.prompts, 8192,
                         'growing', repeat=0, wait_s=0,
                         n_predict=32, timeout=5, interval=0)
        self.assertEqual(t['completed_tasks'], 2)
        self.assertEqual(t['completed_steps'], 6)
        self.assertTrue(t['all_ok'])
        self.assertEqual(len(self.fake.erasures), 2)
        self.assertTrue(all(x[2] for x in self.fake.requests))
        self.assertEqual({x['slot'] for x in t['steps']}, {0, 1})
        summary = e9.summaries([t])[0]
        self.assertEqual(summary['tasks_completed'], 2)
        self.assertEqual(summary['requests_attempted'], 6)
        self.assertEqual(summary['task_completion_rate'], 1.0)

    def test_failure_not_counted_as_completed_work(self):
        self.fake.fail_above = 400
        t = e9.run_trial('http://localhost:8080', self.prompts, 4096,
                         'growing', repeat=0, wait_s=0,
                         n_predict=32, timeout=5, interval=0)
        self.assertEqual(t['completed_tasks'], 1)
        self.assertFalse(t['all_ok'])
        self.assertGreaterEqual(t['capacity_error_requests'], 1)
        row = e9.summaries([t])[0]
        self.assertIsNone(row['median_successful_pair_makespan_s'])
        self.assertEqual(row['task_completion_rate'], 0.5)


if __name__ == '__main__':
    unittest.main()
