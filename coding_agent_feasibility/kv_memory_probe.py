#!/usr/bin/env python3
"""Standalone llama.cpp KV budget characterization. stdlib only.

Measures two concurrent native /completion requests, server /slots and /metrics,
and nvidia-smi whole-device memory. Does not claim /slots is direct KV occupancy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
from urllib import error, request


def url_json(base, method, path, payload=None, timeout=15):
    body = None if payload is None else json.dumps(payload).encode('utf-8')
    req = request.Request(base.rstrip('/') + path, data=body, method=method,
                          headers={'Content-Type': 'application/json'})
    try:
        with request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return json.loads(data) if data else None
    except error.HTTPError as exc:
        detail = exc.read(1500).decode('utf-8', errors='replace')
        raise RuntimeError(f'HTTP {exc.code} {path}: {detail}') from exc


def text_get(base, path, timeout=3):
    with request.urlopen(base.rstrip('/') + path, timeout=timeout) as r:
        return r.read().decode('utf-8', errors='replace')


def gpu_query():
    try:
        out = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=3, stderr=subprocess.DEVNULL)
        data = []
        for line in out.strip().splitlines():
            cols = [p.strip() for p in line.split(',')]
            if len(cols) >= 3:
                data.append({'index': int(cols[0]), 'used_mib': float(cols[1]),
                             'util_percent': float(cols[2])})
        return data
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


_METRIC = re.compile(r'^([^#\s{]+)(?:\{[^}]*\})?\s+([^\s]+)')


def parse_metrics(body):
    metrics = {}
    for line in body.splitlines():
        match = _METRIC.match(line)
        if not match:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if math.isfinite(value):
            metrics[match.group(1)] = value
    return metrics


def read_observation(base, started):
    o = {'s': round(time.monotonic() - started, 3)}
    try:
        slots = url_json(base, 'GET', '/slots', timeout=3)
        if isinstance(slots, dict) and isinstance(slots.get('slots'), list):
            slots = slots['slots']
        if isinstance(slots, list):
            o['slots'] = [
                {k: s.get(k) for k in (
                    'id', 'is_processing', 'id_task', 'n_ctx', 'n_past', 'n_decoded',
                    'n_prompt_tokens', 'n_tokens', 'n_tokens_cached', 'n_tokens_decoded')
                 if k in s}
                for s in slots if isinstance(s, dict)
            ]
            o['busy_slots'] = sum(bool(s.get('is_processing')) for s in o['slots'])
            if o['slots']:
                o['slot_keys'] = sorted(slots[0].keys())
    except Exception as exc:
        o['slots_error'] = str(exc)[:200]
    try:
        o['metrics'] = parse_metrics(text_get(base, '/metrics', timeout=3))
    except Exception as exc:
        o['metrics_error'] = str(exc)[:200]
    o['gpu'] = gpu_query()
    return o


def sampler(base, start, stop, interval, collected):
    while not stop.is_set():
        collected.append(read_observation(base, start))
        stop.wait(interval)


def build_prompt(base, label, target, timeout=20):
    """Fit a reproducible source-like prompt to ~target tokens using /tokenize.

    With unavailable /tokenize, use an approximate character heuristic and label it.
    """
    head = f'Investigate the following independent code trace, labelled {label}.\n'
    lines = [
        f'{i:04d}: Call Pipeline_{label}_{i % 71}.resolve_request; '
        f'check QueueEntry_{i % 103}, ResourceGraph_{i % 97}, '
        f'pending tool dependencies and memory residency for case {i}.\n'
        for i in range(1, 801)
    ]
    suffix = '\nExplain the memory scheduling consequence in one sentence.\n'

    def candidate(n):
        return head + ''.join(lines[:n]) + suffix

    def count(s):
        data = url_json(base, 'POST', '/tokenize', {'content': s, 'add_special': False}, timeout)
        if not isinstance(data, dict) or not isinstance(data.get('tokens'), list):
            raise RuntimeError(f'/tokenize unexpected shape: {str(data)[:150]}')
        return len(data['tokens'])

    try:
        # Fast monotone binary search. Ensures prompt has real token count.
        lo, hi = 0, len(lines)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if count(candidate(mid)) < target:
                lo = mid
            else:
                hi = mid
        choices = [(count(candidate(j)), j) for j in sorted({lo, hi})]
        actual, n = min(choices, key=lambda x: abs(x[0] - target))
        return candidate(n), actual, 'tokenize'
    except Exception as exc:
        # Estimated only; don't silently misrepresent exact tokens.
        n = max(1, min(len(lines), round(target / 40)))
        return candidate(n), None, f'approximate fallback (/tokenize failed: {str(exc)[:120]})'


def perform(base, prompt, n_predict, timeout, barrier=None, cache_prompt=False, id_slot=None):
    payload = {'prompt': prompt, 'n_predict': n_predict, 'temperature': 0,
               'cache_prompt': cache_prompt, 'stream': False}
    if id_slot is not None:
        payload['id_slot'] = id_slot
    if barrier is not None:
        barrier.wait(timeout=10)
    start = time.monotonic()
    try:
        res = url_json(base, 'POST', '/completion', payload, timeout=timeout)
        return {'ok': True, 'seconds': round(time.monotonic() - start, 3),
                'tokens_cached': res.get('tokens_cached'),
                'tokens_evaluated': res.get('tokens_evaluated'),
                'truncated': res.get('truncated'), 'stop_type': res.get('stop_type'),
                'timings': res.get('timings'), 'completion': str(res.get('content', ''))[:300],
                'response_keys': sorted(res.keys())}
    except Exception as exc:
        return {'ok': False, 'seconds': round(time.monotonic() - start, 3),
                'error': str(exc)[:800]}


def stats(samples):
    if not samples:
        return {}
    gpu = [x['gpu'][0]['used_mib'] for x in samples if x.get('gpu')]
    bslots = [x['busy_slots'] for x in samples if 'busy_slots' in x]
    kv_metrics = sorted({k for sample in samples for k in sample.get('metrics', {})
                         if any(s in k.lower() for s in ('kv', 'cache', 'memory'))})
    metric_ranges = {}
    for k in kv_metrics:
        vals = [x['metrics'][k] for x in samples if k in x.get('metrics', {})]
        if vals:
            metric_ranges[k] = {'min': min(vals), 'max': max(vals)}
    out = {'samples': len(samples), 'busy_slots_peak': max(bslots, default=None),
           'gpu_used_mib_first': gpu[0] if gpu else None,
           'gpu_used_mib_peak': max(gpu) if gpu else None,
           'gpu_delta_peak_minus_first_mib': round(max(gpu) - gpu[0], 1) if gpu else None,
           'candidate_kv_or_cache_metrics': metric_ranges}
    if samples:
        out['slot_fields'] = samples[0].get('slot_keys', [])
        out['slot_context_sizes'] = [s.get('n_ctx') for s in samples[0].get('slots', [])]
        out['slots_error'] = samples[0].get('slots_error')
        out['metrics_error'] = samples[0].get('metrics_error')
    return out


def run_pair(base, prompt1, prompt2, n_predict, timeout, interval):
    start = time.monotonic()
    stop = threading.Event()
    collected = []
    watcher = threading.Thread(target=sampler, args=(base, start, stop, interval, collected), daemon=True)
    watcher.start()
    # Ensure a baseline sample is collected before inference begins.
    deadline = time.monotonic() + 8
    while not collected and time.monotonic() < deadline:
        time.sleep(0.05)
    barrier = threading.Barrier(3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(perform, base, prompt1, n_predict, timeout, barrier)
        b = pool.submit(perform, base, prompt2, n_predict, timeout, barrier)
        barrier.wait(timeout=10)
        res = [a.result(), b.result()]
    # Observe whether slots stay allocated after completion. This is not a proof of KV lifetime.
    time.sleep(max(0.4, interval))
    stop.set()
    watcher.join(timeout=8)
    return {'wall_seconds': round(time.monotonic() - start, 3), 'requests': res,
            'observation_summary': stats(collected)}, collected


def run_reuse(base, prompt, timeout, wait_seconds, n_predict=32):
    # Pin sequential calls to slot 0; this is a controlled cache survival test.
    before = read_observation(base, time.monotonic())
    r1 = perform(base, prompt, n_predict, timeout, cache_prompt=True, id_slot=0)
    time.sleep(wait_seconds)
    during = read_observation(base, time.monotonic())
    # This imitates a model response followed by tool output / next-turn context.
    appended = '\nAssistant: ' + str(r1.get('completion', '')) + '\nTool result: inspection finished. Continue with one sentence.\n'
    r2 = perform(base, prompt + appended, n_predict, timeout, cache_prompt=True, id_slot=0)
    after = read_observation(base, time.monotonic())
    return {'wait_seconds': wait_seconds, 'first': r1, 'second': r2,
            'idle_slot_info': during.get('slots'),
            'slot_before': before.get('slots'), 'slot_after': after.get('slots'),
            'note': 'tokens_cached > 0 suggests prompt prefix reuse, not direct evidence of GPU KV bytes held during wait.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--mode', choices=['both', 'contention', 'reuse'], default='both')
    ap.add_argument('--low-tokens', type=int, default=1200)
    ap.add_argument('--high-tokens', type=int, default=2600)
    ap.add_argument('--n-predict', type=int, default=128)
    ap.add_argument('--interval', type=float, default=0.3)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--wait-seconds', type=float, default=3)
    ap.add_argument('--out-dir', type=Path, default=Path('results/kv_probe'))
    args = ap.parse_args()
    if not (128 <= args.low_tokens < args.high_tokens <= 12000):
        ap.error('require 128 <= low-tokens < high-tokens <= 12000')
    if not (1 <= args.n_predict <= 1024 and 0.1 <= args.interval <= 3):
        ap.error('n-predict must be 1..1024 and interval 0.1..3')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print('Checking llama.cpp server ...', flush=True)
    try:
        health = url_json(args.url, 'GET', '/health', timeout=8)
        if health != {'status': 'ok'}:
            print(f'Health response: {health!r}')
    except Exception as exc:
        ap.error(f'server unreachable at {args.url}: {exc}')
    at_rest = read_observation(args.url, time.monotonic())
    print('Slot context limits:', [x.get('n_ctx') for x in at_rest.get('slots', [])])
    if len(at_rest.get('slots', [])) < 2 and args.mode != 'reuse':
        print('WARNING: fewer than 2 slots found; two simultaneous server executions are not possible.')
    if not at_rest.get('metrics'):
        print('NOTE: /metrics missing/empty. Add --metrics when starting llama-server.')
    if not at_rest.get('gpu'):
        print('NOTE: nvidia-smi unavailable; GPU whole-device memory will be absent.')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'server_url': args.url,
              'parameters': vars(args) | {'out_dir': str(args.out_dir)},
              'idle_observation': at_rest, 'pairs': {}}
    if args.mode in ('both', 'contention'):
        for name, tokens in [('low', args.low_tokens), ('high', args.high_tokens)]:
            p1, count1, how1 = build_prompt(args.url, name+'A', tokens)
            p2, count2, how2 = build_prompt(args.url, name+'B', tokens)
            print(f'[{name}] prompt tokens per request: {count1}, {count2} ({how1}; {how2}); '
                  f'concurrent requests=2, n_predict={args.n_predict}', flush=True)
            s, samples = run_pair(args.url, p1, p2, args.n_predict, args.timeout, args.interval)
            s['target_tokens_each'] = tokens
            s['measured_tokens_each'] = [count1, count2]
            s['count_method'] = [how1, how2]
            result['pairs'][name] = s
            with (args.out_dir / f'{name}_samples.jsonl').open('w', encoding='utf-8') as f:
                for v in samples:
                    f.write(json.dumps(v, ensure_ascii=False) + '\n')
            print(f'[{name}] request OK = {[r["ok"] for r in s["requests"]]}, '
                  f'wall={s["wall_seconds"]:.2f}s, '
                  f'peak busy slots={s["observation_summary"]["busy_slots_peak"]}, '
                  f'GPU peak delta MiB={s["observation_summary"]["gpu_delta_peak_minus_first_mib"]}', flush=True)
        print('Metric candidates:', result['pairs']['high']['observation_summary']['candidate_kv_or_cache_metrics'])
    if args.mode in ('both', 'reuse'):
        prompt, count, how = build_prompt(args.url, 'REUSE', args.low_tokens)
        print(f'[reuse] common prompt tokens={count} ({how}), idle gap={args.wait_seconds}s', flush=True)
        result['reuse'] = run_reuse(args.url, prompt, args.timeout, args.wait_seconds)
        for phase in ('first', 'second'):
            r = result['reuse'][phase]
            print(f'[{phase}] success={r["ok"]}, tokens_cached={r.get("tokens_cached")}, '
                  f'tokens_evaluated={r.get("tokens_evaluated")}, elapsed={r["seconds"]}s', flush=True)
    path = args.out_dir / 'summary.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Saved:', path, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
