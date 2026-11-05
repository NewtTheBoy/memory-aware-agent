#!/usr/bin/env python3
"""E7-F: test *physical* GPU VRAM release versus logical KV cache clearing.

Two independent controls:
1. same llama-server: A1 -> optional slot erase -> A2; observe per-process VRAM,
   GPU free VRAM, cache_n, prefill time, and logical KV-cache metrics.
2. optional CUDA contender allocates+frees VRAM in *another process* during A2.

No claim of live KV-buffer resizing; this is a feasibility probe, not a scheduler.
Stdlib-only, requires running llama-server, nvidia-smi, NVIDIA CUDA driver.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import select
import subprocess
import sys
import time
from urllib import request, error


def api(base, method, path, payload=None, timeout=30):
    body = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(base.rstrip('/') + path, data=body, method=method,
                          headers={'Content-Type': 'application/json'})
    try:
        with request.urlopen(req, timeout=timeout) as res:
            raw = res.read()
        return json.loads(raw) if raw else None
    except error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code} {path}: {exc.read(600).decode(errors="replace")}') from exc


def token_count(base, content):
    data = api(base, 'POST', '/tokenize', {'content': content, 'add_special': False})
    if not isinstance(data, dict) or not isinstance(data.get('tokens'), list):
        raise RuntimeError('Unexpected /tokenize response; cannot label actual token count')
    return len(data['tokens'])


def prompt_for_tokens(base, approx_tokens):
    head = 'You are investigating a Python coding agent trace.\n'
    lines = [f'{i:04d} inspect function run_{i%43}, follow dependency chain {i%89}, check error handling.\n'
             for i in range(1, 1801)]
    tail = 'Explain the likely cause briefly.\n'
    lo, hi = 0, len(lines)
    def prompt(n):
        return head + ''.join(lines[:n]) + tail
    # Binary-search the number of lines using actual /tokenize.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if token_count(base, prompt(mid)) < approx_tokens:
            lo = mid
        else:
            hi = mid
    choices = [(token_count(base, prompt(n)), prompt(n)) for n in (lo, hi)]
    return min(choices, key=lambda x: abs(x[0] - approx_tokens))


def parse_nvidia_csv(text):
    records = []
    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        records.append([x.strip() for x in row])
    return records


def nvidia_stats(device=0):
    gpu = None
    processes = []
    try:
        res = subprocess.check_output(['nvidia-smi', '-i', str(device),
            '--query-gpu=memory.used,memory.free,memory.total',
            '--format=csv,noheader,nounits'], text=True, timeout=5, stderr=subprocess.DEVNULL)
        row = parse_nvidia_csv(res)[0]
        gpu = {'used_mib': float(row[0]), 'free_mib': float(row[1]), 'total_mib': float(row[2])}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        pass
    try:
        res = subprocess.check_output(['nvidia-smi', '-i', str(device),
            '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], text=True, timeout=5, stderr=subprocess.DEVNULL)
        for r in parse_nvidia_csv(res):
            try:
                processes.append({'pid': int(r[0]), 'used_mib': float(r[1])})
            except (IndexError, ValueError):
                continue
    except (OSError, subprocess.SubprocessError):
        pass
    return gpu, processes


def server_pids():
    try:
        data = subprocess.check_output(['pgrep', '-x', 'llama-server'], text=True, timeout=3)
        return [int(s) for s in data.split() if s.isdigit()]
    except (OSError, subprocess.SubprocessError):
        return []


def kv_metric_values(base):
    try:
        req = request.Request(base.rstrip('/') + '/metrics')
        with request.urlopen(req, timeout=4) as f:
            data = f.read().decode(errors='replace')
    except Exception:
        return {}
    out = {}
    for line in data.splitlines():
        if line.startswith('#') or 'kv_cache' not in line:
            continue
        m = re.match(r'^([^\s{]+)(?:\{[^}]*\})?\s+([-+\deE.]+)', line)
        if m:
            try:
                v = float(m.group(2))
                if math.isfinite(v):
                    out[m.group(1)] = v
            except ValueError:
                pass
    return out


def observe(label, base, device=0, pid=None):
    gpu, processes = nvidia_stats(device)
    ids = [pid] if pid else server_pids()
    llama = [p for p in processes if p['pid'] in ids]
    sample = {'label': label, 'timestamp': time.time(), 'gpu': gpu,
              'llama_pids': ids, 'llama_process_used_mib': sum(p['used_mib'] for p in llama) if llama else None,
              'all_compute_processes': processes,
              'kv_metrics': kv_metric_values(base)}
    g = sample['gpu'] or {}
    print(f'{label:>20}: GPU used={g.get("used_mib")} MiB, free={g.get("free_mib")} MiB, '
          f'llama process={sample["llama_process_used_mib"]} MiB, KV metrics={sample["kv_metrics"]}', flush=True)
    return sample


def completion(base, prompt, slot, n_predict, timeout):
    start = time.monotonic()
    payload = {'prompt': prompt, 'n_predict': n_predict, 'temperature': 0,
               'cache_prompt': True, 'stream': False, 'id_slot': slot}
    try:
        result = api(base, 'POST', '/completion', payload, timeout=timeout)
        timing = result.get('timings', {}) if isinstance(result, dict) else {}
        return {'ok': True, 'seconds': round(time.monotonic()-start, 4),
                'cache_n': timing.get('cache_n'), 'prompt_n': timing.get('prompt_n'),
                'prompt_ms': timing.get('prompt_ms'),
                'raw_tokens_cached': result.get('tokens_cached'),
                'timings': timing}
    except Exception as exc:
        return {'ok': False, 'seconds': round(time.monotonic()-start, 4), 'error': str(exc)}


def erase(base, slot):
    return api(base, 'POST', f'/slots/{slot}?action=erase', {}, timeout=20)


def trial(base, prompt, slot, predict, wait, mode, device, pid, timeout):
    erase(base, slot)
    snapshots = [observe(f'{mode}:initial', base, device, pid)]
    first = completion(base, prompt, slot, predict, timeout)
    if not first['ok']:
        return {'mode': mode, 'first': first, 'snapshots': snapshots}
    snapshots.append(observe(f'{mode}:after A1', base, device, pid))
    time.sleep(wait)
    erased = erase(base, slot) if mode == 'erase' else None
    time.sleep(0.6)  # allow per-process VRAM snapshot to settle
    snapshots.append(observe(f'{mode}:after wait', base, device, pid))
    second = completion(base, prompt+'\nTool output: tests finished. Continue analysis.\n', slot, predict, timeout)
    snapshots.append(observe(f'{mode}:after A2', base, device, pid))
    return {'mode': mode, 'first': first, 'second': second,
            'erase': erased, 'snapshots': snapshots}


def run_contender(base, prompt, slot, predict, mib, hold, device, pid, timeout):
    script = Path(__file__).with_name('cuda_vram_contender.py')
    erase(base, slot)
    a1 = completion(base, prompt, slot, predict, timeout)
    if not a1['ok']:
        return {'first': a1}
    snapshots = [observe('contender:before', base, device, pid)]
    proc = subprocess.Popen([sys.executable, str(script), '--mib', str(mib), '--hold', str(hold)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    messages = []
    try:
        # Wait for the contender to allocate, *before* issuing A2.
        ready, _, _ = select.select([proc.stdout], [], [], 20)
        if not ready:
            raise RuntimeError('Timed out waiting for CUDA contender allocation')
        message = json.loads(proc.stdout.readline())
        messages.append(message)
        if message.get('event') != 'allocated':
            raise RuntimeError(f'CUDA contender failed: {message}')
        snapshots.append(observe('contender:allocated', base, device, pid))
        a2 = completion(base, prompt+'\nTool output: tests finished. Continue analysis.\n',
                        slot, predict, timeout)
        snapshots.append(observe('contender:after A2', base, device, pid))
        stdout, stderr = proc.communicate(timeout=hold+20)
        for line in stdout.splitlines():
            try:
                messages.append(json.loads(line))
            except ValueError:
                messages.append({'raw': line})
        time.sleep(0.6)
        snapshots.append(observe('contender:freed', base, device, pid))
        return {'first': a1, 'second': a2, 'snapshots': snapshots,
                'contender_messages': messages, 'contender_exit_code': proc.returncode,
                'contender_stderr': stderr[-400:]}
    except Exception as exc:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        return {'first': a1, 'snapshots': snapshots, 'contender_messages': messages,
                'contender_error': str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://127.0.0.1:8080')
    p.add_argument('--slot', type=int, default=0)
    p.add_argument('--server-pid', type=int, default=None)
    p.add_argument('--gpu-index', type=int, default=0)
    p.add_argument('--tokens', type=int, default=2200)
    p.add_argument('--n-predict', type=int, default=64)
    p.add_argument('--wait', type=float, default=2)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--contender-mib', type=int, default=512)
    p.add_argument('--contender-hold', type=float, default=5)
    p.add_argument('--skip-contender', action='store_true')
    p.add_argument('--out', default='results/dynamic_gpu_feasibility.json')
    a = p.parse_args()
    print('Server:', api(a.url, 'GET', '/health'))
    actual_tokens, prompt = prompt_for_tokens(a.url, a.tokens)
    print(f'Calibrated prompt: {actual_tokens} tokens; slot={a.slot}')
    data = {'environment': {'requested_tokens': a.tokens, 'actual_tokens': actual_tokens,
                            'slot': a.slot, 'gpu_index': a.gpu_index,
                            'server_pid': a.server_pid, 'contender_mib': a.contender_mib},
            'trials': {}}
    for mode in ('keep','erase'):
        print('\n== Trial:', mode, '==')
        data['trials'][mode] = trial(a.url, prompt, a.slot, a.n_predict, a.wait,
                                     mode, a.gpu_index, a.server_pid, a.timeout)
        r = data['trials'][mode].get('second', {})
        print(f'  A2: ok={r.get("ok")} cache_n={r.get("cache_n")} '
              f'prompt_n={r.get("prompt_n")} prompt_ms={r.get("prompt_ms")}')
    if not a.skip_contender:
        print('\n== Trial: external GPU memory allocate -> free ==')
        data['trials']['contender'] = run_contender(a.url, prompt, a.slot, a.n_predict,
                      a.contender_mib, a.contender_hold, a.gpu_index, a.server_pid, a.timeout)
        r = data['trials']['contender'].get('second', {})
        print(f'  A2: ok={r.get("ok")} cache_n={r.get("cache_n")}, '
              f'contender_error={data["trials"]["contender"].get("contender_error")}')
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print('\nSaved:', target)
    return 0

if __name__ == '__main__':
    sys.exit(main())
