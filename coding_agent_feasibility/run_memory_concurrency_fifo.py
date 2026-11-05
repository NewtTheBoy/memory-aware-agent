#!/usr/bin/env python3
"""v9: explicit FIFO task-admission baseline for the v7.1 Click agent.

Uses the unchanged CodingPilot, case labels, and grader. The two limits are
independent: --max-tasks controls complete agent tasks, --max-inflight-llm
controls ONE shared semaphore across every agent's LLM requests.

Example:
  python coding_agent_feasibility/run_memory_concurrency_fifo.py \
    --cases-file coding_agent_feasibility/cases_30_diverse_v7.json \
    --max-tasks 2 --max-inflight-llm 2 \
    --out results/task_c2.json --slots-out results/slots_c2.jsonl

For whole-device GPU measurements, wrap this command with profile_gpu_memory.py.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from coding_agent_correctness_5 import (
    CASES, PROFILE_INSTRUCTIONS, CodingPilot, grade_case, http_json,
    load_cases_file, preflight,
)


def quantile(numbers: list[float], p: float) -> float | None:
    """Linear interpolation, independent of optional scientific packages."""
    if not numbers:
        return None
    values = sorted(numbers)
    pos = (len(values) - 1) * p
    i = math.floor(pos)
    return round(values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i), 6)


def interval_peak(intervals: list[tuple[float, float]]) -> int:
    """Maximum overlapping half-open wall-time intervals (start included, end excluded)."""
    edges = []
    for start, end in intervals:
        if end > start:
            edges.extend([(start, 1), (end, -1)])
    edges.sort(key=lambda item: (item[0], item[1]))  # ending before starting at ties
    busy, peak = 0, 0
    for _, delta in edges:
        busy += delta
        peak = max(peak, busy)
    return peak


def make_summary(rows: list[dict], duration_s: float) -> dict:
    completed = [c for c in rows if c['status'] == 'completed']
    good = [c for c in completed if c['grade']['implementation_correct']]
    good_and_test = [c for c in good if c['grade']['focused_pytest_passed']]
    llms = [event for case in rows for event in case['events'] if event['kind'] == 'llm']
    successful_llms = [e for e in llms if 'error' not in e and not e.get('cancelled')]
    task_durations = [float(r['duration_s']) for r in rows]
    task_e2e = [float(r['end_to_end_latency_s']) for r in rows]
    task_queues = [float(r['task_queue_s']) for r in rows]
    llm_queues = [float(e.get('queue_s', 0)) for e in llms]
    llm_intervals = [(e['start_wall_epoch_s'], e['end_wall_epoch_s']) for e in llms
                     if e.get('start_wall_epoch_s') and e.get('end_wall_epoch_s')]
    task_intervals = [(c['case_start_wall_epoch_s'], c['case_end_wall_epoch_s']) for c in rows]
    prompt_counts = [int(e.get('usage', {}).get('prompt_tokens') or 0) for e in successful_llms]
    return {
        'attempted': len(rows),
        'agent_executed': len(completed),
        'implementation_localization_correct': len(good),
        'implementation_and_focused_pytest_passed': len(good_and_test),
        'exact_localization_correct': sum(r['grade']['correct'] for r in rows),
        'required_evidence_steps_complete': sum(r['grade'].get('required_evidence_steps_complete', False) for r in rows),
        'makespan_s': round(duration_s, 6),
        'completed_tasks_per_s': round(len(completed) / duration_s, 6) if duration_s > 0 else None,
        'implementation_correct_tasks_per_s': round(len(good) / duration_s, 6) if duration_s > 0 else None,
        'implementation_plus_pytest_tasks_per_s': round(len(good_and_test) / duration_s, 6) if duration_s > 0 else None,
        # End-to-end task latency includes waiting in the explicit FIFO admission queue.
        'task_latency_p50_s': quantile(task_e2e, .5),
        'task_latency_p95_s': quantile(task_e2e, .95),
        'task_queue_p50_s': quantile(task_queues, .5),
        'task_queue_p95_s': quantile(task_queues, .95),
        'task_execution_p50_s': quantile(task_durations, .5),
        'task_execution_p95_s': quantile(task_durations, .95),
        'llm_requests': len(llms),
        'llm_successful_requests': len(successful_llms),
        'llm_client_queue_p50_s': quantile(llm_queues, .5),
        'llm_client_queue_p95_s': quantile(llm_queues, .95),
        'llm_client_queue_total_s': round(sum(llm_queues), 6),
        'observed_peak_active_tasks': interval_peak(task_intervals),
        'observed_peak_client_inflight_llm': interval_peak(llm_intervals),
        'sum_prompt_tokens': sum(prompt_counts),
        'largest_single_prompt_tokens': max(prompt_counts, default=0),
    }


async def sample_slots(api: str, path: Path, interval_s: float, stop: asyncio.Event) -> dict:
    """Best-effort observational llama.cpp /slots sampling; does not control slots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count, errors, observed_busy_max = 0, 0, 0
    warned = False
    server_base = api.rstrip('/')
    if server_base.endswith('/v1'):
        server_base = server_base[:-3]
    with path.open('w', encoding='utf-8') as handle:
        while not stop.is_set():
            stamp = time.time()
            row: dict = {'wall_epoch_s': stamp}
            try:
                slots = await asyncio.to_thread(http_json, f'{server_base}/slots', timeout=4)
                if not isinstance(slots, list):
                    raise ValueError('The /slots response is not a JSON list')
                records = []
                for s in slots:
                    if not isinstance(s, dict):
                        continue
                    # Slot fields are backend/version-dependent. n_ctx is a
                    # CAPACITY, not actual used KV tokens.
                    records.append({key: s.get(key) for key in
                                    ('id', 'is_processing', 'n_ctx', 'n_past', 'n_tokens', 'state')
                                    if key in s})
                busy = sum(bool(s.get('is_processing')) for s in records)
                row.update({'n_slots': len(records), 'busy_slots': busy, 'slots': records})
                observed_busy_max = max(observed_busy_max, busy)
            except Exception as exc:
                errors += 1
                row['sample_error'] = str(exc)[:240]
                if not warned:
                    print(f'[slot monitor] /slots unavailable: {exc!s}. Agent run continues.',
                          file=sys.stderr, flush=True)
                    warned = True
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            count += 1
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
            except asyncio.TimeoutError:
                pass
    return {'path': str(path), 'samples': count, 'errors': errors,
            'observed_peak_busy_slots': observed_busy_max,
            'note': 'n_ctx is slot capacity; busy_slots are snapshots, not KV occupancy'}


async def execute_cases(selected: list[dict], *, repo: Path, api: str, max_tasks: int,
                        max_llm: int, test_timeout: int, max_followups: int) -> tuple[list[dict], float]:
    """FIFO task admission using one shared asyncio.Queue and max_tasks workers.

    All tasks are admitted to the pending queue in dataset order. Task latency is
    measured from enqueue -> completion; execution time is admission -> completion.
    A separate global LLM semaphore bounds in-flight model calls, NOT task starts.
    """
    shared_llm_sem = asyncio.Semaphore(max_llm)
    task_queue: asyncio.Queue[tuple[int, dict, float]] = asyncio.Queue()
    rows: list[dict | None] = [None] * len(selected)
    wall_start = time.time()
    for index, case in enumerate(selected):
        task_queue.put_nowait((index, case, time.time()))

    async def worker():
        while True:
            index, case, arrived = await task_queue.get()
            try:
                # FIFO get() order is explicit and independent of semaphore fairness.
                admitted = time.time()
                pilot = CodingPilot(repo, api, sys.executable, max_llm,
                                    test_timeout, max_followups)
                pilot.sem = shared_llm_sem
                pilot.investigation_profile = case.get('investigation_profile', 'basic')
                pilot.question = case['question']
                pilot.profile_instructions = PROFILE_INSTRUCTIONS[pilot.investigation_profile]
                print(f'\n[TASK START {index + 1}/{len(selected)}] {case["id"]} '
                      f'[{pilot.investigation_profile}] '
                      f'FIFO-queue={admitted - arrived:.3f}s', flush=True)
                status, result, err = 'completed', None, None
                try:
                    result = await pilot.run()
                except Exception as exc:
                    status, err = 'failed', repr(exc)
                    print(f'[TASK ERROR] {case["id"]}: {err}', file=sys.stderr, flush=True)
                end = time.time()
                grade = grade_case(case, result, status)
                row = {
                    'case_id': case['id'], 'question': case['question'],
                    'investigation_profile': pilot.investigation_profile,
                    'expected': {'source': case['expected_source'],
                                 'symbol': case['expected_symbol'],
                                 'test': case['expected_test']},
                    'status': status, 'error': err,
                    'case_enqueued_wall_epoch_s': arrived,
                    'case_admitted_wall_epoch_s': admitted,
                    'case_start_wall_epoch_s': admitted,
                    'case_end_wall_epoch_s': end,
                    'task_queue_s': round(admitted - arrived, 6),
                    'duration_s': round(end - admitted, 6),
                    'end_to_end_latency_s': round(end - arrived, 6),
                    'model': pilot.model, 'events': pilot.events,
                    'result': result, 'grade': grade,
                }
                rows[index] = row
                print(f'[TASK END] {case["id"]}: '
                      f'queue={row["task_queue_s"]:.2f}s, '
                      f'execution={row["duration_s"]:.2f}s, '
                      f'e2e={row["end_to_end_latency_s"]:.2f}s, '
                      f'impl={grade["implementation_correct"]}', flush=True)
            finally:
                task_queue.task_done()

    workers = [asyncio.create_task(worker())
               for _ in range(min(max_tasks, len(selected)))]
    try:
        await task_queue.join()
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    duration = time.time() - wall_start
    assert all(row is not None for row in rows)
    return rows, duration


def checked_cases(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if args.cases_file:
        try:
            cases, pinned = load_cases_file(args.cases_file)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        cases, pinned = CASES, None
    if args.cases is not None:
        if not args.cases or len(set(args.cases)) != len(args.cases):
            parser.error('--cases needs unique IDs, at least one')
        unknown = set(args.cases) - {c['id'] for c in cases}
        if unknown:
            parser.error(f'Unknown case IDs: {sorted(unknown)}')
        selected = [c for c in cases if c['id'] in set(args.cases)]
    else:
        selected = list(cases)
    repo = Path(args.repo).resolve()
    if not (repo / 'src/click').is_dir() or not (repo / 'tests').is_dir():
        parser.error(f'Not a Click repo: {repo}')
    for c in selected:
        preflight(c, repo)
    rev = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                         text=True, capture_output=True)
    commit = rev.stdout.strip() if rev.returncode == 0 else None
    if pinned and (not commit or pinned.lower() != commit.lower()):
        parser.error(f'Expected pinned Click commit {pinned}; actual {commit}')
    dirty_cmd = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'],
                               text=True, capture_output=True)
    dirty = bool(dirty_cmd.stdout.strip()) if dirty_cmd.returncode == 0 else None
    return cases, selected, pinned, repo, commit, dirty


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='coding-repos/click')
    parser.add_argument('--api', default='http://127.0.0.1:8080/v1')
    parser.add_argument('--cases-file', type=Path, default=None)
    parser.add_argument('--cases', nargs='*', default=None)
    parser.add_argument('--max-tasks', type=int, default=2,
                        help='Maximum concurrent complete agent tasks')
    parser.add_argument('--max-inflight-llm', type=int, default=2,
                        help='GLOBAL client-side LLM request limit, shared by all tasks')
    parser.add_argument('--max-followups', type=int, default=1)
    parser.add_argument('--test-timeout', type=int, default=90)
    parser.add_argument('--slots-out', type=Path, default=None,
                        help='Optional JSONL of llama.cpp /slots snapshots; not KV usage')
    parser.add_argument('--slot-poll-ms', type=int, default=500)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.max_tasks < 1 or args.max_inflight_llm < 1 or args.max_followups < 0:
        parser.error('max-tasks/max-inflight-llm >= 1, max-followups >= 0')
    if args.test_timeout < 1 or args.slot_poll_ms < 250:
        parser.error('test-timeout >= 1, slot-poll-ms >= 250')
    _, selected, pinned, repo, commit, dirty = checked_cases(args, parser)

    print(f'v9 FIFO memory pilot: {len(selected)} Click cases; max-tasks={args.max_tasks}; '
          f'GLOBAL max-inflight-llm={args.max_inflight_llm}; commit={commit}; dirty={dirty}',
          flush=True)
    print('Individual tool/event logs may interleave. Per-case events are kept separately in JSON.',
          flush=True)
    stop = asyncio.Event()
    monitor = (asyncio.create_task(sample_slots(args.api, args.slots_out,
                                                 args.slot_poll_ms / 1000.0, stop))
               if args.slots_out else None)
    run_start = time.time()
    try:
        rows, duration = await execute_cases(selected, repo=repo, api=args.api,
                                              max_tasks=args.max_tasks,
                                              max_llm=args.max_inflight_llm,
                                              test_timeout=args.test_timeout,
                                              max_followups=args.max_followups)
    finally:
        stop.set()
    slot_stats = await monitor if monitor else None
    summary = make_summary(rows, duration)
    print('\nSUMMARY ' + json.dumps(summary, ensure_ascii=False), flush=True)
    output = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation': 'coding_agent_fifo_task_concurrency_baseline_v9',
        'repo': str(repo), 'git_commit': commit, 'repo_dirty': dirty,
        'run_start_wall_epoch_s': run_start,
        'run_end_wall_epoch_s': run_start + duration,
        'config': {
            'api': args.api, 'max_tasks': args.max_tasks,
            'max_inflight_llm': args.max_inflight_llm,
            'max_followups': args.max_followups,
            'test_timeout_seconds': args.test_timeout,
            'artificial_tool_delay_seconds': 0,
            'llm_semaphore_scope': 'global_shared_across_tasks',
            'task_admission_policy': 'explicit_fifo_asyncio_queue_with_fixed_worker_pool',
        },
        'dataset': {
            'source': str(args.cases_file) if args.cases_file else 'built_in_5',
            'selected_cases': len(selected), 'selected_case_ids': [c['id'] for c in selected],
            'pinned_commit': pinned,
        },
        'summary': summary, 'slot_monitor': slot_stats, 'cases': rows,
        'measurement_limits': {
            'task_queue_s': 'wait in FIFO queue before task admission',
            'task_latency_s': 'enqueue to finish, includes FIFO task queue wait',
            'llm_queue_s': 'client semaphore wait, not llama.cpp server queue or GPU execution',
            'llm_duration_s': 'end-to-end HTTP request duration, not exclusive GPU compute',
            'gpu_used_mib': 'whole device allocation, not per-request KV residency',
            'correctness': 'reference symbol localization; focused pytest alone does not prove test relevance',
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Saved trace: {args.out}', flush=True)
    if slot_stats:
        print('Slot monitor: ' + json.dumps(slot_stats), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
