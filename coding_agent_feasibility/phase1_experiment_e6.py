#!/usr/bin/env python3
"""E6: KV-budget sensitivity of cache preservation / slot reuse.

This is an offline client-side *controlled experiment*, not a memory-aware
scheduler. Run against an otherwise idle, local two-slot llama.cpp server at
-c 4096 / 6144 / 8192 in separate processes. It has two independent tests:

  sweep: repeat E4 preserve-vs-reuse for a range of A-context lengths.
  idle:  isolate idle-KV retention: A1(slot0), B(slot1), A2(slot0), with
         slot0 either kept cached or explicitly erased before B arrives.

The idle test *never* uses A's idle slot for B: differences cannot be due to
client slot reservation/queueing. It may expose implicit llama.cpp cache
reclamation, but /slots counters are not physical KV occupancy.

Requires existing phase1_experiment_e4.py, phase1_experiments.py,
kv_memory_probe.py in the same directory. Python standard library only.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import threading
import time

import kv_memory_probe as kp
import phase1_experiments as ex
import phase1_experiment_e4 as e4

BUDGETS = (4096, 6144, 8192)
IDLE_POLICIES = ('keep', 'evict')
SWEEP_POLICIES = ('preserve', 'reuse')


def utc():
    return datetime.now(timezone.utc).isoformat()


def med(vals):
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return round(statistics.median(vals), 6) if vals else None


def write_csv(path, records, columns=None):
    if not records and not columns:
        return
    columns = columns or list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(records)


def is_capacity_error(response):
    err = str((response or {}).get('error') or '').lower()
    return ('context size has been exceeded' in err or 'context size exceeded' in err
            or 'no kv slot' in err or 'kv cache' in err or 'out of memory' in err)


def safe_slots(base):
    """Surface diagnostic slot counters without treating them as KV cell usage."""
    try:
        rows = kp.url_json(base, 'GET', '/slots', timeout=5)
        if isinstance(rows, dict):
            rows = rows.get('slots')
        if not isinstance(rows, list):
            return {'error': 'Unexpected /slots response'}
        return {str(s.get('id', i)): {key: s.get(key) for key in (
            'is_processing', 'n_ctx', 'n_tokens', 'n_prompt_tokens',
            'n_tokens_cached', 'n_past') if key in s}
                for i, s in enumerate(rows) if isinstance(s, dict)}
    except Exception as err:
        return {'error': str(err)[:250]}


def one_request(base, prompt, n_predict, timeout, slot, cache):
    return ex.proof_metrics(ex.execute(base, prompt, n_predict, timeout,
                                       slot=slot, cache=cache))


def run_idle_arm(base, a_prompt, b_prompt, policy, wait_s, a_predict,
                 b_predict, timeout, repeat, a_label, b_label):
    """No slot contention: B always uses slot1; A always uses slot0.

    A2 executes *after* B (same deliberate ordering in both arms) and after
    the designated tool wait. We measure loss of the *idle* A prefix, not
    throughput of a realistic online scheduler.
    """
    if policy not in IDLE_POLICIES:
        raise ValueError(f'unknown idle policy {policy}')
    ex.eraser(base, n=2)
    origin = time.monotonic()
    def elapsed():
        return time.monotonic() - origin

    snap_start = safe_slots(base)
    a1_start = elapsed()
    raw_a1 = ex.execute(base, a_prompt, a_predict, timeout, slot=0, cache=True)
    a1_end = elapsed()
    a1 = ex.proof_metrics(raw_a1)
    record = {
        'experiment': 'E6-idle-KV', 'created_utc': utc(), 'repeat': repeat,
        'a_profile': a_label, 'b_profile': b_label, 'wait_s': wait_s,
        'policy': policy, 'a1': a1, 'a1_end_s': round(a1_end, 6),
        'snap_before': snap_start,
    }
    if not a1['ok']:
        record.update({'all_ok': False, 'completed': 0,
                       'worker_errors': ['A1: ' + str(a1.get('error'))],
                       'total_makespan_s': round(elapsed(), 6)})
        return record

    next_prompt = ex.append_step(a_prompt, raw_a1)
    record['a2_prompt_sha256'] = hashlib.sha256(next_prompt.encode()).hexdigest()
    record['snap_after_a1'] = safe_slots(base)

    # Erasing only slot0 is the controlled manipulation; B still runs slot1.
    if policy == 'evict':
        try:
            record['erase_a'] = kp.url_json(base, 'POST', '/slots/0?action=erase', {}, timeout=12)
        except Exception as err:
            record.update({'all_ok': False, 'completed': 0,
                           'worker_errors': ['erase: ' + str(err)],
                           'total_makespan_s': round(elapsed(), 6)})
            return record
    record['snap_before_b'] = safe_slots(base)
    b_start = elapsed()
    b = one_request(base, b_prompt, b_predict, timeout, slot=1, cache=False)
    b_end = elapsed()
    record['b'] = b
    record['b_start_s'] = round(b_start, 6)
    record['b_end_s'] = round(b_end, 6)
    record['snap_after_b'] = safe_slots(base)

    # A2 begins after BOTH the planned pause and the B request. Never silently
    # turn an idle-KV capacity probe into a simultaneous A2/B execution test.
    remaining = a1_end + wait_s - elapsed()
    if remaining > 0:
        time.sleep(remaining)
    a2_start = elapsed()
    a2 = one_request(base, next_prompt, a_predict, timeout, slot=0, cache=True)
    a2_end = elapsed()
    record['a2'] = a2
    record['a2_start_s'] = round(a2_start, 6)
    record['a2_end_s'] = round(a2_end, 6)
    record['snap_after_a2'] = safe_slots(base)
    record['all_ok'] = bool(b['ok'] and a2['ok'])
    record['completed'] = int(bool(b['ok'])) + int(bool(a2['ok']))
    record['worker_errors'] = [f'{name}: {r.get("error")}' for name, r in [('B', b), ('A2', a2)]
                               if not r['ok']]
    record['b_capacity_error'] = is_capacity_error(b)
    record['a2_capacity_error'] = is_capacity_error(a2)
    record['a2_cache_n'] = a2.get('cache_n')
    record['a2_prompt_n'] = a2.get('prompt_n')
    record['a2_prompt_ms'] = a2.get('prompt_ms')
    record['b_request_s'] = b.get('request_s')
    record['a2_request_s'] = a2.get('request_s')
    record['total_makespan_s'] = round(a2_end, 6)
    record['a2_started_after_b'] = a2_start >= b_end - 0.001
    return record


def execute_with_optional_samples(func, base, sample_interval):
    if sample_interval <= 0:
        return func(), None
    origin = time.monotonic()
    stop = threading.Event()
    samples = []
    thread = threading.Thread(target=kp.sampler,
                              args=(base, origin, stop, sample_interval, samples),
                              daemon=True)
    thread.start()
    try:
        data = func()
    finally:
        stop.set()
        thread.join(timeout=8)
    return data, {'summary': kp.stats(samples), 'samples': samples}


def flatten_sweep(t, budget):
    keys = ['mix', 'wait_s', 'repeat', 'policy', 'all_ok', 'completed',
            'total_makespan_s', 'phase_makespan_s', 'a2_cache_n',
            'a2_prompt_n', 'a2_prompt_ms', 'a2_queue_s',
            'b_queue_s', 'c_queue_s', 'a_completion_s',
            'b_completion_from_arrival_s', 'c_completion_from_arrival_s']
    out = {k: t.get(k) for k in keys}
    out.update({'budget': budget, 'a_profile': t.get('a_profile'),
                'capacity_error': any(is_capacity_error(z.get('response'))
                 for z in t.get('timings', {}).values()),
                'worker_errors': ' | '.join(t.get('worker_errors', [])),
                'a2_prompt_sha256': t.get('a2_prompt_sha256')})
    obs = t.get('monitor') or {}
    for field in ('busy_slots_peak', 'gpu_used_mib_peak',
                  'gpu_delta_peak_minus_first_mib'):
        out[field] = obs.get('summary', {}).get(field)
    return out


def flatten_idle(t, budget):
    keys = ['wait_s', 'repeat', 'policy', 'all_ok', 'completed',
            'nominal_cached_a_plus_b_tokens', 'nominal_demand_over_budget',
            'total_makespan_s', 'a2_cache_n', 'a2_prompt_n',
            'a2_prompt_ms', 'a2_request_s', 'b_request_s',
            'b_capacity_error', 'a2_capacity_error']
    out = {k: t.get(k) for k in keys}
    out.update({'budget': budget, 'a_profile': t.get('a_profile'),
                'b_profile': t.get('b_profile'),
                'b_ok': (t.get('b') or {}).get('ok'),
                'a2_ok': (t.get('a2') or {}).get('ok'),
                'worker_errors': ' | '.join(t.get('worker_errors', [])),
                'a2_prompt_sha256': t.get('a2_prompt_sha256')})
    obs = t.get('monitor') or {}
    for field in ('busy_slots_peak', 'gpu_used_mib_peak',
                  'gpu_delta_peak_minus_first_mib'):
        out[field] = obs.get('summary', {}).get(field)
    return out


def group_summary(rows, kind):
    if kind == 'sweep':
        group_fields = ('budget', 'a_profile', 'mix', 'wait_s', 'policy')
    elif kind == 'idle':
        group_fields = ('budget', 'a_profile', 'b_profile', 'wait_s', 'policy')
    else:
        raise ValueError(kind)
    groups = {}
    for r in rows:
        key = tuple(r.get(field) for field in group_fields)
        groups.setdefault(key, []).append(r)
    output = []
    for key, recs in sorted(groups.items(), key=lambda v: tuple(str(x) for x in v[0])):
        valid = [r for r in recs if r.get('all_ok')]
        outcome = dict(zip(group_fields, key))
        outcome.update({
            'arms': len(recs), 'success_arms': len(valid),
            'success_rate': round(len(valid) / len(recs), 6),
            'mean_completed': round(sum(int(r.get('completed') or 0) for r in recs) / len(recs), 6),
            # Never report failed arms as faster successful makespans.
            'median_success_makespan_s': med([r.get('total_makespan_s') for r in valid]),
            'median_a2_cache_n': med([r.get('a2_cache_n') for r in valid]),
            'median_a2_cache_n_observed': med([r.get('a2_cache_n') for r in recs]),
            'median_a2_prefill_ms': med([r.get('a2_prompt_ms') for r in valid]),
            'capacity_error_arms': sum(bool(r.get('capacity_error') or
                                       r.get('b_capacity_error') or
                                       r.get('a2_capacity_error')) for r in recs),
            'peak_whole_gpu_used_mib': max((float(r['gpu_used_mib_peak']) for r in recs
                                         if r.get('gpu_used_mib_peak') is not None), default=None),
        })
        if kind == 'idle':
            outcome['b_success_rate'] = round(sum(bool(r.get('b_ok')) for r in recs) / len(recs), 6)
            outcome['a2_success_rate'] = round(sum(bool(r.get('a2_ok')) for r in recs) / len(recs), 6)
            outcome['median_nominal_demand_over_budget'] = med(
                [r.get('nominal_demand_over_budget') for r in recs])
        if kind == 'sweep':
            outcome['median_c_queue_s'] = med([r.get('c_queue_s') for r in valid])
            outcome['median_a2_queue_s'] = med([r.get('a2_queue_s') for r in valid])
        output.append(outcome)
    return output


def paired_summary(rows, kind):
    """Return ONLY success-conditioned time differences; expose all failures."""
    if kind == 'sweep':
        keyfields = ('budget', 'a_profile', 'mix', 'wait_s')
        pa, pb = SWEEP_POLICIES
    else:
        keyfields = ('budget', 'a_profile', 'b_profile', 'wait_s')
        pa, pb = IDLE_POLICIES
    grouped = {}
    for r in rows:
        k = (*[r.get(f) for f in keyfields], r.get('repeat'))
        grouped.setdefault(k, {})[r['policy']] = r
    aggregate = {}
    for key, pair in grouped.items():
        k = key[:-1]
        rec = aggregate.setdefault(k, {'pairs_total': 0, 'pairs_both_success': 0,
                                       'differences': [], 'sum_completed_a': 0,
                                       'sum_completed_b': 0, 'both_cache': []})
        if pa not in pair or pb not in pair:
            continue
        rec['pairs_total'] += 1
        a, b = pair[pa], pair[pb]
        rec['sum_completed_a'] += int(a.get('completed') or 0)
        rec['sum_completed_b'] += int(b.get('completed') or 0)
        if a.get('all_ok') and b.get('all_ok'):
            rec['pairs_both_success'] += 1
            rec['differences'].append(a['total_makespan_s'] - b['total_makespan_s'])
            rec['both_cache'].append((a.get('a2_cache_n'), b.get('a2_cache_n')))
    out = []
    for k, rec in sorted(aggregate.items(), key=lambda x: tuple(str(s) for s in x[0])):
        item = dict(zip(keyfields, k))
        delta = med(rec['differences'])
        item.update({
            'paired_total': rec['pairs_total'],
            'paired_both_success': rec['pairs_both_success'],
            f'sum_completed_{pa}': rec['sum_completed_a'],
            f'sum_completed_{pb}': rec['sum_completed_b'],
            f'median_{pa}_minus_{pb}_makespan_s': delta,
            'faster_when_both_succeed':
                pb if delta is not None and delta > 0 else
                pa if delta is not None and delta < 0 else 'unknown_or_tie',
            'comparison_note': 'Times comparable only for trials where both policies succeeded.',
        })
        out.append(item)
    return out


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_prompts(base, a_targets, short, long):
    specs = {**{f'A_{n}': n for n in a_targets},
             'B_SHORT': short, 'C_SHORT': short,
             'B_LONG': long, 'C_LONG': long}
    prompts, measured = {}, {}
    for name, target in specs.items():
        p, count, method = kp.build_prompt(base, 'E6_'+name, target)
        if method != 'tokenize' or count is None:
            raise RuntimeError(f'/tokenize failed for {name}: {method}')
        prompts[name] = p
        measured[name] = {'tokens': count, 'sha256': hashlib.sha256(p.encode()).hexdigest()}
    return prompts, measured


def verify_server(base, expected_ctx, max_prompt, max_predict):
    startup = ex.check_local_server_context(expected_ctx)
    line = startup['process_command']
    argv = shlex.split(line)[1:]
    for must in ('--kv-unified', '--kv-unified-per-slot', '--cache-ram', '--slot-save-path', '-np'):
        if must not in argv:
            raise RuntimeError(f'E6 requires {must} in the actual llama-server command: {line}')
    def val(flag):
        return argv[argv.index(flag) + 1]
    if val('-np') != '2' or int(val('--kv-unified-per-slot')) != 4096 or val('--cache-ram') != '0':
        raise RuntimeError('Expected -np 2 --kv-unified-per-slot 4096 --cache-ram 0')
    health = kp.url_json(base, 'GET', '/health', timeout=8)
    if health.get('status') != 'ok':
        raise RuntimeError('llama-server is not healthy: '+str(health))
    slots = kp.url_json(base, 'GET', '/slots', timeout=8)
    if isinstance(slots, dict):
        slots = slots.get('slots')
    if not isinstance(slots, list) or len(slots) != 2:
        raise RuntimeError('E6 requires exactly two accessible server slots')
    if any(s.get('is_processing') for s in slots):
        raise RuntimeError('Other clients are active; stop E5/other experiments first.')
    sizes = [s.get('n_ctx') for s in slots]
    if any(isinstance(x, int) and x < max_prompt + max_predict + 80 for x in sizes):
        raise RuntimeError(f'Per-slot context limits {sizes} smaller than maximum request; '
                           'check -c and --kv-unified-per-slot')
    ex.eraser(base, n=2)
    return {'startup': startup, 'slot_limits': sizes, 'health': health}


def run_budget(args):
    # We do not attempt to change -c on a running llama.cpp server.
    # Exact local command verification is mandatory for every arm group.
    conf = verify_server(args.url, args.expect_ctx,
                         max(args.a_tokens + [args.short_tokens, args.long_tokens]),
                         max(args.a_predict, args.bc_predict))
    prompts, measured = make_prompts(args.url, args.a_tokens, args.short_tokens, args.long_tokens)
    print('Server -c:', args.expect_ctx, 'slot limits:', conf['slot_limits'])
    print('Measured prompt tokens:', {k: v['tokens'] for k, v in measured.items()}, flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    needed = ([args.out_dir / 'sweep_trials.jsonl'] if args.mode in ('both', 'sweep') else []) + (
             [args.out_dir / 'idle_trials.jsonl'] if args.mode in ('both', 'idle') else [])
    for path in needed:
        if path.exists():
            raise RuntimeError(f'{path} already exists. Use a new --out-dir; do not merge repetitions.')
    meta = {
        'experiment': 'E6', 'created_utc': utc(), 'kv_budget': args.expect_ctx,
        'configuration': {**vars(args), 'out_dir': str(args.out_dir)},
        'server': conf, 'measured_prompts': measured,
        'limitations': [
            '-c is logical KV pool size in tokens, not physical free VRAM.',
            'Client-side preserve reserves slot0, not guaranteed KV pinning.',
            'Idle probe never overlaps model calls and is not a throughput benchmark.',
            '/slots token counters and whole-device nvidia-smi are not allocated KV-cell occupancy.',
            'Requests use synthetic reproducible contexts; tool waiting is sleep.',
        ],
    }
    (args.out_dir / 'config.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')

    idx = 0
    if args.mode in ('both', 'sweep'):
        count = len(args.a_tokens) * len(args.mixes) * len(args.waits) * args.repeats * 2
        print(f'E6 sweep: {count} arms; E4-like A/B/C scheduling', flush=True)
        sweep_path = args.out_dir / 'sweep_trials.jsonl'
        with sweep_path.open('w', encoding='utf-8') as f:
            for rep in range(1, args.repeats + 1):
                for aidx, target in enumerate(args.a_tokens):
                    aprompt = prompts[f'A_{target}']
                    for mi, mix in enumerate(args.mixes):
                        bp = prompts['B_LONG' if mix.startswith('long_') else 'B_SHORT']
                        cp = prompts['C_LONG' if mix.endswith('_long') else 'C_SHORT']
                        for wi, wait in enumerate(args.waits):
                            order = (SWEEP_POLICIES if (rep + aidx + mi + wi) % 2 == 0
                                     else SWEEP_POLICIES[::-1])
                            pair = {}
                            for policy in order:
                                idx += 1
                                run = lambda p=policy: e4.run_arm(
                                    args.url, aprompt, bp, cp, p, wait,
                                    args.a_predict, args.bc_predict,
                                    args.timeout, rep, mix)
                                arm, monitor = execute_with_optional_samples(run, args.url,
                                                                            args.sample_interval)
                                arm['experiment'] = 'E6-sweep'
                                arm['a_profile'] = f'A_{target}'
                                arm['kv_budget'] = args.expect_ctx
                                if monitor:
                                    arm['monitor'] = monitor
                                f.write(json.dumps(arm, ensure_ascii=False) + '\n')
                                f.flush()
                                pair[policy] = arm
                                print(f'  [sweep {idx}/{count}] A={target} mix={mix} wait={wait:g} '
                                      f'rep={rep} {policy}: {arm["completed"]}/3, '
                                      f'makespan={arm["total_makespan_s"]:.3f}s, '
                                      f'A2 cache={arm.get("a2_cache_n")}', flush=True)
                            if pair['preserve'].get('a2_prompt_sha256') != pair['reuse'].get('a2_prompt_sha256'):
                                print('WARNING: paired A2 text differs; examine a2_prompt_sha256', flush=True)
        rows = [flatten_sweep(t, args.expect_ctx) for t in load_jsonl(sweep_path)]
        write_csv(args.out_dir / 'sweep_trials.csv', rows)
        write_csv(args.out_dir / 'sweep_groups.csv', group_summary(rows, 'sweep'))
        write_csv(args.out_dir / 'sweep_pairs.csv', paired_summary(rows, 'sweep'))

    if args.mode in ('both', 'idle'):
        total = len(args.a_tokens) * len(args.b_profiles) * len(args.waits) * args.repeats * 2
        index = 0
        print(f'E6 idle-KV isolation probe: {total} arms', flush=True)
        idle_path = args.out_dir / 'idle_trials.jsonl'
        with idle_path.open('w', encoding='utf-8') as f:
            for rep in range(1, args.repeats + 1):
                for aidx, target in enumerate(args.a_tokens):
                    for bi, b_kind in enumerate(args.b_profiles):
                        for wi, wait in enumerate(args.waits):
                            order = (IDLE_POLICIES if (rep + aidx + bi + wi) % 2 == 0
                                     else IDLE_POLICIES[::-1])
                            pair = {}
                            for policy in order:
                                index += 1
                                run = lambda p=policy: run_idle_arm(
                                    args.url, prompts[f'A_{target}'],
                                    prompts['B_'+b_kind.upper()], p, wait,
                                    args.a_predict, args.bc_predict, args.timeout,
                                    rep, f'A_{target}', b_kind)
                                arm, monitor = execute_with_optional_samples(run, args.url,
                                                                            args.sample_interval)
                                arm['kv_budget'] = args.expect_ctx
                                nominal = (measured[f'A_{target}']['tokens'] + args.a_predict
                                           + measured['B_'+b_kind.upper()]['tokens'] + args.bc_predict)
                                arm['nominal_cached_a_plus_b_tokens'] = nominal
                                arm['nominal_demand_over_budget'] = round(nominal / args.expect_ctx, 4)
                                if monitor:
                                    arm['monitor'] = monitor
                                f.write(json.dumps(arm, ensure_ascii=False) + '\n')
                                f.flush()
                                pair[policy] = arm
                                print(f'  [idle {index}/{total}] A={target} B={b_kind} wait={wait:g} '
                                      f'rep={rep} {policy}: {arm["completed"]}/2, '
                                      f'B={arm.get("b",{}).get("ok")}, '
                                      f'A2 cache={arm.get("a2_cache_n")}, '
                                      f'B cap.err={arm.get("b_capacity_error")}', flush=True)
                            if pair['keep'].get('a2_prompt_sha256') != pair['evict'].get('a2_prompt_sha256'):
                                print('WARNING: paired A2 text differs', flush=True)
        rows = [flatten_idle(t, args.expect_ctx) for t in load_jsonl(idle_path)]
        write_csv(args.out_dir / 'idle_trials.csv', rows)
        write_csv(args.out_dir / 'idle_groups.csv', group_summary(rows, 'idle'))
        write_csv(args.out_dir / 'idle_pairs.csv', paired_summary(rows, 'idle'))

    print(f'Finished E6 c={args.expect_ctx}. Output: {args.out_dir}', flush=True)
    return 0


def aggregate_many(paths, output_dir):
    seen_ctx, prompt_reference = set(), None
    all_sweep, all_idle = [], []
    for root in paths:
        conf = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        ctx = conf['kv_budget']
        if ctx in seen_ctx:
            raise RuntimeError(f'Duplicate -c {ctx} in aggregation; only one run per budget')
        seen_ctx.add(ctx)
        known_prompts = {k: v['sha256'] for k, v in conf['measured_prompts'].items()}
        if prompt_reference is None:
            prompt_reference = known_prompts
        elif known_prompts != prompt_reference:
            raise RuntimeError(f'Prompt hashes differ across budgets: {root}. '
                               'Use the same A/B/C parameters and model.')
        for r in load_jsonl(root / 'sweep_trials.jsonl'):
            all_sweep.append(flatten_sweep(r, ctx))
        for r in load_jsonl(root / 'idle_trials.jsonl'):
            all_idle.append(flatten_idle(r, ctx))
    output_dir.mkdir(parents=True, exist_ok=True)
    if all_sweep:
        write_csv(output_dir / 'sweep_all.csv', all_sweep)
        write_csv(output_dir / 'sweep_by_budget.csv', group_summary(all_sweep, 'sweep'))
        write_csv(output_dir / 'sweep_policy_pairs.csv', paired_summary(all_sweep, 'sweep'))
    if all_idle:
        write_csv(output_dir / 'idle_all.csv', all_idle)
        write_csv(output_dir / 'idle_by_budget.csv', group_summary(all_idle, 'idle'))
        write_csv(output_dir / 'idle_policy_pairs.csv', paired_summary(all_idle, 'idle'))
    print(f'Aggregated budgets: {sorted(seen_ctx)}. Output: {output_dir}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--expect-ctx', type=int, choices=BUDGETS,
                        help='Required for live run: exact -c of local llama-server')
    parser.add_argument('--aggregate', nargs='+', type=Path,
                        help='Aggregate existing E6 result directories without running server')
    parser.add_argument('--mode', choices=('both', 'sweep', 'idle'), default='both')
    parser.add_argument('--waits', nargs='+', type=float, default=[0, 3],
                        help='Simulated tool waits, same values at every budget')
    parser.add_argument('--mixes', nargs='+',
                        choices=('short_short', 'short_long', 'long_short', 'long_long'),
                        default=['short_long', 'long_long'])
    parser.add_argument('--b-profiles', nargs='+', choices=('short', 'long'),
                        default=['short', 'long'])
    parser.add_argument('--a-tokens', type=int, nargs='+', default=[1200, 2200],
                        help='1200=E4-like; 2200=extra stress without crossing per-slot limit')
    parser.add_argument('--short-tokens', type=int, default=1200)
    parser.add_argument('--long-tokens', type=int, default=2600)
    parser.add_argument('--a-predict', type=int, default=64)
    parser.add_argument('--bc-predict', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--sample-interval', type=float, default=0.5,
                        help='Seconds between optional GPU /slots and /metrics samples; 0 disables')
    parser.add_argument('--out-dir', type=Path, default=Path('results/phase1_e6_c8192'))
    args = parser.parse_args(argv)
    if args.aggregate:
        return aggregate_many(args.aggregate, args.out_dir)
    if args.expect_ctx is None:
        parser.error('--expect-ctx is required for live experiments')
    if args.repeats < 1 or args.timeout <= 0 or not (0 <= args.sample_interval <= 5):
        parser.error('invalid repeats/timeout/sample interval')
    if (not args.waits or any(not 0 <= x <= 60 for x in args.waits)
            or len(args.waits) != len(set(args.waits))):
        parser.error('waits must be distinct and between 0 and 60 seconds')
    if (not args.a_tokens or len(set(args.a_tokens)) != len(args.a_tokens)
            or any(not 300 <= x <= 3500 for x in args.a_tokens)
            or not 300 <= args.short_tokens <= 3500
            or not 300 <= args.long_tokens <= 3500):
        parser.error('token targets must be unique and between 300 and 3500')
    if (args.short_tokens >= args.long_tokens or
            not (1 <= args.a_predict <= 512 and 1 <= args.bc_predict <= 512)):
        parser.error('require short < long and valid max prediction tokens')
    if len(args.mixes) != len(set(args.mixes)) or len(args.b_profiles) != len(set(args.b_profiles)):
        parser.error('mixes and b-profiles must be unique')
    return run_budget(args)


if __name__ == '__main__':
    raise SystemExit(main())
