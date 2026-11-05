#!/usr/bin/env python3
"""E9: controlled two-agent context-growth vs fixed-window characterization.

Requires adjacent kv_memory_probe.py (from earlier E1/E3). Standard library only.
The workflow and tool pauses are SYNTHETIC, not Click code correctness tasks.
Two agents use slots 0/1 and each make 3 LLM requests. In fixed mode each
step replaces the prior observation chunk; in growing mode it accumulates it.
Both modes see the same step-specific synthetic observations, do the same
number of model steps, and use the same token generation cap.

Measure actual prompt tokenization, cache hits, server success/capacity errors,
workflow completions, request time, and device-wide GPU memory. GPU memory is
not direct live KV-cell usage, especially with preallocated KV buffers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import threading
import time

import kv_memory_probe as kp

AGENTS = ('A', 'B')
STEPS = 3
MODES = ('fixed', 'growing')


def now():
    return datetime.now(timezone.utc).isoformat()


def median(values):
    nums = [float(x) for x in values if x is not None]
    return round(statistics.median(nums), 6) if nums else None


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def tokenize_count(base, content):
    response = kp.url_json(base, 'POST', '/tokenize',
                           {'content': content, 'add_special': False}, timeout=30)
    if not isinstance(response, dict) or not isinstance(response.get('tokens'), list):
        raise RuntimeError(f'/tokenize unavailable: {str(response)[:200]}')
    return len(response['tokens'])


def synthetic_lines(agent, count=450):
    """Fixed deterministic external-tool-like records; not fabricated source code."""
    return [
        f'Tool observation {i:04d} agent {agent}: search source for function '
        f'resolve_{i % 51}, inspect call trace {i % 43}, compare reference '
        f'test_{i % 37}, and record code location evidence {i}.\n'
        for i in range(count)
    ]


def closest_prefix(base, header, lines, goal):
    def candidate(n):
        return header + ''.join(lines[:n])
    lo, hi = 0, len(lines)
    if tokenize_count(base, candidate(hi)) < goal:
        raise RuntimeError(f'Not enough observation lines to reach {goal} tokens')
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if tokenize_count(base, candidate(mid)) < goal:
            lo = mid
        else:
            hi = mid
    options = [(tokenize_count(base, candidate(n)), n) for n in (lo, hi)]
    return min(options, key=lambda v: abs(v[0] - goal))[1]


def make_prompts(base, targets):
    """Generate fixed and growing prompts sharing identical per-step observations.

    Growing: header + chunks 1, 2, 3 cumulatively (genuine text prefix).
    Fixed:   header + latest chunk, retaining a bounded context window.
    """
    if len(targets) != STEPS or not (150 <= targets[0] < targets[1] < targets[2]):
        raise ValueError('Expected three increasing context targets >=150')
    result = {}
    for agent in AGENTS:
        header = (f'You are background coding agent {agent}, analyzing a deterministic '
                  'synthetic source search. These are tool observations; do not run '
                  'tools. After reading them, answer in exactly one short sentence.\n')
        lines = synthetic_lines(agent)
        boundaries = [closest_prefix(base, header, lines, goal) for goal in targets]
        if any(a >= b for a, b in zip([0] + boundaries[:-1], boundaries)):
            raise RuntimeError(f'Prompt calibration cannot separate chunks for {agent}: {boundaries}')
        growing = [header + ''.join(lines[:n]) for n in boundaries]
        fixed = [header + ''.join(lines[start:end])
                 for start, end in zip([0] + boundaries[:-1], boundaries)]
        assert all(growing[i + 1].startswith(growing[i]) for i in range(2))
        result[agent] = {}
        for mode, prompts in [('fixed', fixed), ('growing', growing)]:
            result[agent][mode] = [
                {'text': p, 'tokens': tokenize_count(base, p),
                 'sha256': hashlib.sha256(p.encode('utf-8')).hexdigest()}
                for p in prompts
            ]
    return result


def server_config_check(expected, base):
    """Verify local process flags, not just a user-supplied budget label."""
    try:
        output = subprocess.check_output(['pgrep', '-af', 'llama-server'],
                                         text=True, timeout=6)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'Cannot inspect llama-server process: {exc}') from exc
    candidates = []
    for line in output.splitlines():
        try:
            argv = shlex.split(line)
        except ValueError:
            continue
        for pos, token in enumerate(argv):
            if token == 'llama-server' or token.endswith('/llama-server'):
                args = argv[pos + 1:]
                if '-c' in args:
                    candidates.append(args)
                break
    if len(candidates) != 1:
        raise RuntimeError('Expected exactly one running llama-server with identifiable -c; '
                           f'found {len(candidates)}')
    flags = candidates[0]
    def val(name):
        if name not in flags:
            raise RuntimeError(f'Server missing {name}; see E9_README.md')
        idx = flags.index(name)
        if idx + 1 >= len(flags):
            raise RuntimeError(f'Server missing value for {name}')
        return flags[idx + 1]
    if int(val('-c')) != expected:
        raise RuntimeError(f'Actual server -c {val("-c")} != --expect-ctx {expected}')
    if int(val('-np')) != 2:
        raise RuntimeError('Expected exactly -np 2')
    if not any(x in flags for x in ('--kv-unified', '-kv-unified')):
        raise RuntimeError('Expected --kv-unified')
    if int(val('--kv-unified-per-slot')) != 4096:
        raise RuntimeError('Expected --kv-unified-per-slot 4096')
    if int(val('--cache-ram')) != 0 or '--slot-save-path' not in flags:
        raise RuntimeError('Expected --cache-ram 0 and --slot-save-path for slot erase')
    health = kp.url_json(base, 'GET', '/health', timeout=10)
    if not isinstance(health, dict) or health.get('status') not in ('ok', 'no slot available'):
        raise RuntimeError(f'Unexpected server health {health}')
    raw = kp.url_json(base, 'GET', '/slots', timeout=10)
    slots = raw.get('slots') if isinstance(raw, dict) else raw
    if not isinstance(slots, list) or len(slots) != 2:
        raise RuntimeError('Expected exactly two /slots')
    if any(s.get('is_processing') for s in slots):
        raise RuntimeError('Server has active requests; stop other clients first')
    return {'process_flags': flags, 'health': health,
            'slot_context_sizes': [s.get('n_ctx') for s in slots]}


def erase_both(base):
    for slot in (0, 1):
        try:
            response = kp.url_json(base, 'POST', f'/slots/{slot}?action=erase', {}, timeout=15)
        except Exception as exc:
            raise RuntimeError(f'Cannot erase slot {slot}: {exc}. '
                               'Check --slot-save-path; do not run mixed-cache trials.') from exc
        if response is None:
            raise RuntimeError(f'Empty erase response for slot {slot}')


def slot_snapshot(base):
    try:
        rows = kp.url_json(base, 'GET', '/slots', timeout=8)
        rows = rows.get('slots') if isinstance(rows, dict) else rows
        return [{k: s.get(k) for k in ('id', 'is_processing', 'n_ctx', 'n_tokens',
                'n_tokens_cached', 'n_past', 'n_prompt_tokens') if k in s}
                for s in rows]
    except Exception as exc:
        return {'error': str(exc)[:200]}


def capacity_error(record):
    message = str((record or {}).get('error') or '').lower()
    return any(term in message for term in ('context size', 'kv cache', 'no kv slot',
                                            'out of memory', 'not enough memory', 'cuda error'))


def gpu_sampling(stop, samples, started, interval):
    while not stop.is_set():
        data = kp.gpu_query()
        if data:
            samples.append({'elapsed_s': round(time.monotonic() - started, 3),
                            'gpu': data})
        stop.wait(interval)


def invoke_pair(base, requests, n_predict, timeout, executor):
    """Submit both active agent requests together, one per pinned slot."""
    if len(requests) == 1:
        agent, prompt = requests[0]
        return {agent: kp.perform(base, prompt, n_predict, timeout,
                                  id_slot=AGENTS.index(agent), cache_prompt=True)}
    barrier = threading.Barrier(len(requests) + 1)
    futures = {agent: executor.submit(kp.perform, base, prompt, n_predict, timeout,
                                      barrier, True, AGENTS.index(agent))
               for agent, prompt in requests}
    barrier.wait(timeout=15)
    return {agent: fut.result() for agent, fut in futures.items()}


def run_trial(base, prompts, budget, mode, repeat, wait_s, n_predict,
              timeout, interval):
    erase_both(base)
    started = time.monotonic()
    record = {'budget': budget, 'mode': mode, 'repeat': repeat, 'created_utc': now(),
              'agent_completed': {}, 'steps': [], 'tool_delay_seconds': wait_s,
              'slot_start': slot_snapshot(base)}
    active = set(AGENTS)
    stopped = threading.Event()
    samples = []
    watcher = None
    if interval > 0:
        watcher = threading.Thread(target=gpu_sampling,
                                   args=(stopped, samples, started, interval), daemon=True)
        watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for stage in range(STEPS):
                if not active:
                    break
                reqs = [(agent, prompts[agent][mode][stage]['text']) for agent in AGENTS
                        if agent in active]
                before_s = time.monotonic() - started
                res = invoke_pair(base, reqs, n_predict, timeout, pool)
                after_s = time.monotonic() - started
                for agent, result in res.items():
                    timing = result.get('timings') or {}
                    entry = {'budget': budget, 'mode': mode, 'repeat': repeat,
                             'agent': agent, 'stage': stage + 1,
                             'slot': AGENTS.index(agent),
                             'prompt_tokens_calibrated': prompts[agent][mode][stage]['tokens'],
                             'prompt_sha256': prompts[agent][mode][stage]['sha256'],
                             'ok': bool(result.get('ok')),
                             'request_s': result.get('seconds'),
                             'cache_n': timing.get('cache_n'),
                             'prompt_n': timing.get('prompt_n'),
                             'prefill_ms': timing.get('prompt_ms'),
                             'predicted_n': timing.get('predicted_n'),
                             'truncated': result.get('truncated'),
                             'error': result.get('error'),
                             'capacity_error': capacity_error(result),
                             'step_started_s': round(before_s, 6),
                             'step_finished_s': round(after_s, 6)}
                    record['steps'].append(entry)
                    if not entry['ok']:
                        active.remove(agent)
                record[f'slot_after_step_{stage+1}'] = slot_snapshot(base)
                if stage < STEPS - 1 and active and wait_s:
                    time.sleep(wait_s)
    finally:
        stopped.set()
        if watcher:
            watcher.join(timeout=6)
    completed = sorted(active) if record['steps'] and any(s['stage'] == STEPS for s in record['steps']) else []
    record['agent_completed'] = {a: a in completed for a in AGENTS}
    record['completed_tasks'] = len(completed)
    record['attempted_tasks'] = len(AGENTS)
    record['completed_steps'] = sum(int(s['ok']) for s in record['steps'])
    record['attempted_steps'] = len(record['steps'])
    record['capacity_error_requests'] = sum(int(s['capacity_error']) for s in record['steps'])
    record['all_ok'] = len(completed) == 2
    record['makespan_s'] = round(time.monotonic() - started, 6)
    used = [sample['gpu'][0]['used_mib'] for sample in samples if sample.get('gpu')]
    record['gpu_used_mib_first'] = used[0] if used else None
    record['gpu_used_mib_peak'] = max(used) if used else None
    record['gpu_sample_count'] = len(used)
    record['gpu_samples'] = samples
    return record


def flatten_trials(trials):
    return [{k: trial.get(k) for k in ('budget', 'mode', 'repeat', 'all_ok',
             'completed_tasks', 'attempted_tasks', 'attempted_steps',
             'completed_steps', 'capacity_error_requests', 'makespan_s',
             'gpu_used_mib_first', 'gpu_used_mib_peak', 'gpu_sample_count')}
            for trial in trials]


def flatten_requests(trials):
    return [step for trial in trials for step in trial['steps']]


def summaries(trials):
    groups = {}
    for trial in trials:
        groups.setdefault((trial['budget'], trial['mode']), []).append(trial)
    rows = []
    for (budget, mode), group in sorted(groups.items()):
        allsteps = [step for t in group for step in t['steps']]
        step3 = [s for s in allsteps if s['stage'] == 3]
        successes = [t for t in group if t['all_ok']]
        row = {
            'budget': budget, 'mode': mode, 'repeats': len(group),
            'tasks_attempted': sum(t['attempted_tasks'] for t in group),
            'tasks_completed': sum(t['completed_tasks'] for t in group),
            'task_completion_rate': round(sum(t['completed_tasks'] for t in group) /
                                          sum(t['attempted_tasks'] for t in group), 6),
            'successful_pairs': len(successes),
            'requests_attempted': len(allsteps),
            'requests_failed': sum(not s['ok'] for s in allsteps),
            'capacity_error_requests': sum(s['capacity_error'] for s in allsteps),
            'median_successful_pair_makespan_s': median([t['makespan_s'] for t in successes]),
            'median_stage3_prefill_ms_success': median([s['prefill_ms'] for s in step3 if s['ok']]),
            'median_stage3_cache_n_success': median([s['cache_n'] for s in step3 if s['ok']]),
            'max_whole_gpu_used_mib': max((t['gpu_used_mib_peak'] for t in group
                                         if t['gpu_used_mib_peak'] is not None), default=None),
        }
        for stage in range(1, 4):
            row[f'step{stage}_median_prompt_tokens'] = median([
                s['prompt_tokens_calibrated'] for s in allsteps if s['stage'] == stage])
        rows.append(row)
    return rows


def save_run(out, config, trials):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n',
                                     encoding='utf-8')
    with (out / 'trials.jsonl').open('w', encoding='utf-8') as f:
        for t in trials:
            f.write(json.dumps(t, ensure_ascii=False)+'\n')
    write_csv(out / 'trials.csv', flatten_trials(trials))
    write_csv(out / 'requests.csv', flatten_requests(trials))
    write_csv(out / 'summary.csv', summaries(trials))


def aggregate(dirs, out):
    all_trials, configs, budgets = [], [], set()
    for folder in dirs:
        cfg = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        label = int(cfg['budget'])
        if label in budgets:
            raise RuntimeError(f'Duplicate -c budget {label}')
        budgets.add(label)
        configs.append(cfg)
        with (folder / 'trials.jsonl').open(encoding='utf-8') as f:
            all_trials.extend(json.loads(line) for line in f if line.strip())
    first = configs[0]
    for cfg in configs[1:]:
        for key in ('targets', 'n_predict', 'wait_s', 'prompt_hashes', 'modes'):
            if cfg[key] != first[key]:
                raise RuntimeError(f'Cannot aggregate incomparable runs: {key} differs')
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / 'all_summary.csv', summaries(all_trials))
    write_csv(out / 'all_trials.csv', flatten_trials(all_trials))
    write_csv(out / 'all_requests.csv', flatten_requests(all_trials))
    (out / 'aggregation.json').write_text(json.dumps({'budgets': sorted(budgets),
                                                      'source_dirs': [str(x) for x in dirs],
                                                      'configs': configs}, indent=2)+'\n', encoding='utf-8')
    print(f'Combined results: {out / "all_summary.csv"}', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--expect-ctx', type=int, choices=[4096, 8192])
    p.add_argument('--targets', type=int, nargs=3, default=[900, 1800, 2700])
    p.add_argument('--mode', choices=['both', *MODES], default='both')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--wait', type=float, default=1.0)
    p.add_argument('--n-predict', type=int, default=32)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--sample-interval', type=float, default=0.5)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--aggregate', type=Path, nargs='+', metavar='RUN_DIR')
    args = p.parse_args(argv)
    if args.aggregate:
        return args
    if args.expect_ctx is None:
        p.error('--expect-ctx required for a live experiment')
    if not (1 <= args.repeats <= 30 and 0 <= args.wait <= 30 and
            1 <= args.n_predict <= 128 and args.timeout > 1 and
            0 <= args.sample_interval <= 5):
        p.error('Invalid repeats, wait, n-predict, timeout or interval')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.aggregate:
        aggregate(args.aggregate, args.out_dir)
        return 0
    if args.out_dir.exists():
        raise RuntimeError(f'Output dir already exists: {args.out_dir}; use a new path')
    server = server_config_check(args.expect_ctx, args.url)
    prompts = make_prompts(args.url, args.targets)
    if any(prompts[a][m][i]['tokens'] + args.n_predict > 4096
           for a in AGENTS for m in MODES for i in range(STEPS)):
        raise RuntimeError('A single request exceeds the 4096 per-slot limit')
    modes = list(MODES) if args.mode == 'both' else [args.mode]
    prompt_hashes = {a: {m: [step['sha256'] for step in prompts[a][m]] for m in MODES}
                     for a in AGENTS}
    config = {'date_utc': now(), 'experiment': 'E9-context-growth-controlled',
              'budget': args.expect_ctx, 'url': args.url,
              'modes': modes, 'repeats': args.repeats, 'targets': args.targets,
              'wait_s': args.wait, 'n_predict': args.n_predict,
              'sample_interval': args.sample_interval, 'timeout': args.timeout,
              'server': server,
              'prompt_hashes': prompt_hashes,
              'prompt_tokens': {a: {m: [s['tokens'] for s in prompts[a][m]]
                                    for m in MODES} for a in AGENTS},
              'limitations': ['Synthetic tool records and sleep, not live Click tool execution',
                              'Successful synthetic task means all three LLM calls succeeded; '
                              'it does not establish semantic correctness',
                              '-c is a logical KV token pool budget, not physical available VRAM',
                              'GPU used is whole-device, not process-specific KV occupancy',
                              'Token counts measured by /tokenize; /completion prefill/cache '
                              'timings are runtime-dependent',
                              'Request output lengths capped by n_predict; record actual predicted_n']}
    print('Actual token counts (calibrated):', json.dumps(config['prompt_tokens']), flush=True)
    print(f'Budget {args.expect_ctx}; 2 agents x 3 steps; modes={modes}', flush=True)
    trials = []
    for repeat in range(args.repeats):
        order = modes if repeat % 2 == 0 else list(reversed(modes))
        for mode in order:
            record = run_trial(args.url, prompts, args.expect_ctx, mode, repeat,
                               args.wait, args.n_predict, args.timeout,
                               args.sample_interval)
            trials.append(record)
            print(f'[E9] repeat={repeat+1} mode={mode} '
                  f'completed={record["completed_tasks"]}/2 '
                  f'steps={record["completed_steps"]}/{record["attempted_steps"]} '
                  f'capacity_errors={record["capacity_error_requests"]} '
                  f'makespan={record["makespan_s"]:.2f}s', flush=True)
    save_run(args.out_dir, config, trials)
    print(f'Saved {args.out_dir / "summary.csv"} and detailed traces', flush=True)
    for row in summaries(trials):
        print('SUMMARY', json.dumps(row), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
