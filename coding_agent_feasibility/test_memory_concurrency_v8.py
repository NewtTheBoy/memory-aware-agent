"""Unit tests for shared LLM admission, task overlap and timestamped monitoring."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import run_memory_concurrency as memory
import summarize_memory_concurrency as summary


class MemoryConcurrencyTests(unittest.TestCase):
    def test_interval_peak_half_open_and_quantile(self):
        self.assertEqual(memory.interval_peak([(0, 2), (2, 3), (1, 4)]), 2)
        self.assertEqual(memory.interval_peak([]), 0)
        self.assertEqual(memory.quantile([1, 2, 3, 4], .5), 2.5)
        self.assertIsNone(memory.quantile([], .95))

    def test_global_llm_semaphore_and_true_task_overlap(self):
        state = {'tasks': 0, 'peak_tasks': 0, 'llm': 0, 'peak_llm': 0,
                 'semaphore_ids': set()}

        class FakePilot:
            def __init__(self, repo, api, python, max_llm, timeout, followups):
                self.sem = asyncio.Semaphore(max_llm)
                self.model = 'fake'
                self.events = []

            async def run(self):
                state['tasks'] += 1
                state['peak_tasks'] = max(state['peak_tasks'], state['tasks'])
                state['semaphore_ids'].add(id(self.sem))
                ready = time.time()
                async with self.sem:
                    state['llm'] += 1
                    state['peak_llm'] = max(state['peak_llm'], state['llm'])
                    started = time.time()
                    await asyncio.sleep(0.025)
                    finished = time.time()
                    state['llm'] -= 1
                    self.events.append({'id': 'P0', 'kind': 'llm',
                                        'start_wall_epoch_s': started,
                                        'end_wall_epoch_s': finished,
                                        'queue_s': started - ready,
                                        'usage': {'prompt_tokens': 30}})
                state['tasks'] -= 1
                return {'fake': True}

        cases = [{'id': f'case_{i}', 'question': 'find it',
                  'investigation_profile': 'basic', 'expected_source': 'src/click/core.py',
                  'expected_symbol': 'Context.invoke',
                  'expected_test': 'tests/test_context.py'} for i in range(5)]
        grade = {'implementation_correct': True, 'focused_pytest_passed': True,
                 'correct': False, 'required_evidence_steps_complete': True}

        with patch.object(memory, 'CodingPilot', FakePilot), patch.object(memory, 'grade_case', return_value=grade):
            rows, duration = asyncio.run(memory.execute_cases(
                cases, repo=Path('/tmp/does-not-matter'), api='http://dummy/v1',
                max_tasks=3, max_llm=2, test_timeout=90, max_followups=1))
        self.assertEqual(state['peak_tasks'], 3)
        self.assertEqual(state['peak_llm'], 2)
        self.assertEqual(len(state['semaphore_ids']), 1)
        self.assertEqual([r['case_id'] for r in rows], [c['id'] for c in cases])
        s = memory.make_summary(rows, duration)
        self.assertEqual(s['observed_peak_active_tasks'], 3)
        self.assertEqual(s['observed_peak_client_inflight_llm'], 2)
        self.assertEqual(s['implementation_localization_correct'], 5)
        self.assertGreater(s['llm_client_queue_total_s'], 0)

    def test_slots_poll_uses_root_not_v1_and_handles_slot_fields(self):
        urls = []
        def fake_get(url, timeout):
            urls.append(url)
            return [{'id': 0, 'is_processing': True, 'n_ctx': 8192, 'n_past': 22},
                    {'id': 1, 'is_processing': False, 'n_ctx': 8192}]
        async def run(path):
            stop = asyncio.Event()
            async def soon():
                await asyncio.sleep(.05)
                stop.set()
            f = asyncio.create_task(soon())
            stats = await memory.sample_slots('http://localhost:8080/v1', path, .25, stop)
            await f
            return stats
        with tempfile.TemporaryDirectory() as td, patch.object(memory, 'http_json', side_effect=fake_get):
            path = Path(td) / 'slots.jsonl'
            result = asyncio.run(run(path))
            self.assertEqual(result['observed_peak_busy_slots'], 1)
            self.assertEqual(result['errors'], 0)
            self.assertEqual(urls[0], 'http://localhost:8080/slots')
            rec = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(rec['slots'][0]['n_past'], 22)

    def test_gpu_csv_time_filtering_and_baseline_delta(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'gpu.csv'
            path.write_text('wall_epoch_s,used_mib,gpu_util_percent\n'
                            '1,3333,1\n2,3433,50\n3,3533,70\n4,4000,20\n')
            path.with_suffix('.meta.json').write_text(json.dumps({'baseline_device_used_mib': 3333}))
            s = summary.gpu_stats(path, 2, 3)
            self.assertEqual(s['peak_gpu_used_mib'], 3533)
            self.assertEqual(s['peak_minus_baseline_mib'], 200)
            self.assertEqual(s['mean_gpu_util_percent'], 60)
            self.assertEqual(s['sample_count_during_tasks'], 2)

if __name__ == '__main__':
    unittest.main()
