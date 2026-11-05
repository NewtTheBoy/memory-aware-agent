#!/usr/bin/env python3
"""E4: compare two fixed slot-assignment policies around an agent tool pause.

Standard-library only, using the existing kv_memory_probe and phase1_experiments
helpers. This is a controlled *client-side policy comparison*, not KV pinning,
physical KV residency measurement, a scheduler, or an offloading implementation.

Work (same in both policies): A1 -> tool wait -> A2, with B and C arriving at
A1's finish. Preserve reserves slot 0 for A2 while B then C run on slot 1;
Reuse runs B on slot 0 and C on slot 1, so A2 waits for B and loses some prefix.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import threading
import time

import kv_memory_probe as kp
import phase1_experiments as ex


METRICS = (
    'all_ok', 'completed', 'total_makespan_s', 'phase_makespan_s',
    'throughput_completed_tasks_s', 'a_completion_s', 'a2_request_s',
    'a2_queue_s', 'a2_cache_n', 'a2_prompt_n', 'a2_prompt_ms',
    'b_completion_from_arrival_s', 'c_completion_from_arrival_s',
    'b_queue_s', 'c_queue_s',
)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def run_arm(base, a_prompt, b_prompt, c_prompt, policy, wait_s, a_predict, bc_predict,
            timeout, repeat, mix):
    """One isolated trial. All three tasks are counted, including failed ones.

    Arrival times: A starts at wall t=0. B and C arrive exactly when A1 ends.
    A2 becomes eligible wait_s after A1; in reuse it also waits for B/slot0.
    """
    if policy not in ('preserve', 'reuse'):
        raise ValueError('policy must be preserve or reuse')

    # Do not leak prefix or KV state between two policies or repeats.
    erase_responses = ex.eraser(base, n=2)
    origin = time.monotonic()

    def ts():
        return time.monotonic() - origin

    def timed_request(name, prompt, slot, tokens, cache, arrival_s):
        begun = ts()
        data = ex.execute(base, prompt, tokens, timeout, slot=slot, cache=cache)
        finished = ts()
        return {
            'name': name, 'slot': slot, 'arrival_s': round(arrival_s, 6),
            'start_s': round(begun, 6), 'end_s': round(finished, 6),
            'queue_s': round(max(0.0, begun - arrival_s), 6),
            'response': ex.proof_metrics(data),
            # Keep full A1 completion only as local scratch; never log huge text.
            '_raw': data,
        }

    a1 = timed_request('A1', a_prompt, 0, a_predict, True, 0.0)
    if not a1['response']['ok']:
        raise RuntimeError(f'Agent A Step1 failed in {mix}/{policy}: {a1["response"]["error"]}')
    a1_end = a1['end_s']
    eligible_a2_s = a1_end + wait_s
    a2_prompt = ex.append_step(a_prompt, a1['_raw'])
    a2_sha = hashlib.sha256(a2_prompt.encode('utf-8')).hexdigest()
    b_done = threading.Event()
    results = {}
    mutex = threading.Lock()

    def store(name, value):
        with mutex:
            results[name] = value
        return value

    def worker_b():
        try:
            slot = 1 if policy == 'preserve' else 0
            return store('B', timed_request('B', b_prompt, slot, bc_predict,
                                            False, a1_end))
        finally:
            b_done.set()

    def worker_c():
        if policy == 'preserve':
            # Same slot as B. Exact FIFO order enforced, not lock-race dependent.
            if not b_done.wait(timeout + 10):
                raise RuntimeError('B did not finish: cannot begin C on slot 1')
        return store('C', timed_request('C', c_prompt, 1, bc_predict,
                                        False, a1_end))

    def worker_a2():
        remaining = (origin + eligible_a2_s) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        if policy == 'reuse':
            # Wait for B to release slot 0, even if the tool is already ready.
            if not b_done.wait(timeout + 10):
                raise RuntimeError('B did not finish: cannot resume A on slot 0')
        return store('A2', timed_request('A2', a2_prompt, 0, a_predict,
                                         True, eligible_a2_s))

    # Distinct threads. preserve: A2 can overlap B/C on the other slot.
    # reuse: B and C run concurrently; A2 follows B on slot0.
    with ThreadPoolExecutor(max_workers=3) as pool:
        fb = pool.submit(worker_b)
        fc = pool.submit(worker_c)
        fa = pool.submit(worker_a2)
        # Collect exceptions without dropping partial trial results.
        worker_errors = []
        for name, fut in [('B', fb), ('C', fc), ('A2', fa)]:
            try:
                fut.result()
            except Exception as error:
                worker_errors.append(f'{name}: {error}')

    # Store response metrics only; avoid embedding a full generated text dump.
    timings = {name: {key: value for key, value in record.items() if key != '_raw'}
               for name, record in {'A1': a1, **results}.items()}
    all_ending_times = [rec['end_s'] for rec in timings.values()]
    makespan = max(all_ending_times)
    completed = sum(bool(timings.get(name, {}).get('response', {}).get('ok'))
                    for name in ('A2', 'B', 'C'))
    all_ok = completed == 3 and not worker_errors

    arm = {
        'experiment': 'E4', 'timestamp_utc': now_utc(), 'mix': mix,
        'policy': policy, 'wait_s': wait_s, 'repeat': repeat,
        'arm_description': (
            'preserve: A uses slot0, B->C use slot1, A2 at tool-ready time'
            if policy == 'preserve' else
            'reuse: B uses slot0, C uses slot1; A2 uses slot0 after B and tool-ready'
        ),
        'a2_prompt_sha256': a2_sha,
        'a1_end_s': round(a1_end, 6),
        'a2_eligible_s': round(eligible_a2_s, 6),
        'all_ok': all_ok, 'completed': completed,
        'worker_errors': worker_errors,
        'erase_responses': erase_responses,
        'timings': timings,
        'total_makespan_s': round(makespan, 6),
        'phase_makespan_s': round(makespan - a1_end, 6),
        'throughput_completed_tasks_s': round(completed / makespan, 6),
        'limitations': ('B/C are one-step model tasks; tool wait is simulated sleep. '
                        'Slot preservation is client-enforced idleness, not KV pinning.'),
    }
    if 'A2' in timings:
        a2 = timings['A2']
        rm = a2['response']
        arm.update({
            'a_completion_s': a2['end_s'],
            'a2_request_s': rm.get('request_s'),
            'a2_queue_s': a2['queue_s'],
            'a2_cache_n': rm.get('cache_n'),
            'a2_prompt_n': rm.get('prompt_n'),
            'a2_prompt_ms': rm.get('prompt_ms'),
        })
    for key, tag in [('B', 'b'), ('C', 'c')]:
        if key in timings:
            arm[f'{tag}_completion_from_arrival_s'] = round(
                timings[key]['end_s'] - a1_end, 6)
            arm[f'{tag}_queue_s'] = timings[key]['queue_s']
    return arm


def csv_rows(trials):
    columns = [
        'mix', 'wait_s', 'repeat', 'policy', 'all_ok', 'completed',
        'total_makespan_s', 'phase_makespan_s', 'throughput_completed_tasks_s',
        'a_completion_s', 'a2_request_s', 'a2_queue_s',
        'a2_cache_n', 'a2_prompt_n', 'a2_prompt_ms',
        'b_completion_from_arrival_s', 'c_completion_from_arrival_s',
        'b_queue_s', 'c_queue_s', 'worker_errors', 'a2_prompt_sha256',
    ]
    rows = [{key: (' | '.join(t.get(key, [])) if key == 'worker_errors'
                   else t.get(key)) for key in columns} for t in trials]
    return columns, rows


def median(values):
    numeric = [float(x) for x in values if x is not None and isinstance(x, (float, int))
               and math.isfinite(float(x))]
    return round(statistics.median(numeric), 6) if numeric else None


def aggregate(trials, mixes, waits):
    grouped = []
    paired = []
    for mix in mixes:
        for wait in waits:
            valid = {}
            for policy in ('preserve', 'reuse'):
                rows = [t for t in trials if t['mix'] == mix and t['wait_s'] == wait
                        and t['policy'] == policy and t['all_ok']]
                valid[policy] = rows
                group = {'mix': mix, 'wait_s': wait, 'policy': policy,
                         'valid_n': len(rows),
                         'failed_n': sum(1 for t in trials
                                         if t['mix'] == mix and t['wait_s'] == wait
                                         and t['policy'] == policy and not t['all_ok'])}
                for key in METRICS:
                    if key in ('all_ok', 'completed'):
                        continue
                    group['median_' + key] = median([r.get(key) for r in rows])
                grouped.append(group)
            # Paired arms (same replicate index; policy execution order alternates).
            reps = sorted({t['repeat'] for t in trials if t['mix'] == mix
                           and t['wait_s'] == wait})
            differences = []
            for rep in reps:
                a = next((t for t in valid['preserve'] if t['repeat'] == rep), None)
                b = next((t for t in valid['reuse'] if t['repeat'] == rep), None)
                if a and b:
                    differences.append(a['total_makespan_s'] - b['total_makespan_s'])
            delta = median(differences)
            paired.append({'mix': mix, 'wait_s': wait, 'paired_valid_n': len(differences),
                           'median_preserve_minus_reuse_makespan_s': delta,
                           'winner_by_median': (
                               'reuse' if delta is not None and delta > 0 else
                               'preserve' if delta is not None and delta < 0 else
                               'tie_or_no_data'
                           ),
                           'note': ('positive delta = reuse has shorter total makespan; '
                                    'not necessarily a statistically established advantage')})
    return grouped, paired


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--expect-ctx', type=int, default=8192,
                    help='Assert local server -c, default 8192; use isolated -np 2 server')
    ap.add_argument('--waits', nargs='+', type=float, default=[0, 3, 8])
    ap.add_argument('--mixes', nargs='+', choices=['short_short', 'short_long', 'long_short', 'long_long'],
                    default=['short_short', 'short_long'])
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--a-tokens', type=int, default=1200)
    ap.add_argument('--short-tokens', type=int, default=1200)
    ap.add_argument('--long-tokens', type=int, default=2600)
    ap.add_argument('--a-predict', type=int, default=64)
    ap.add_argument('--bc-predict', type=int, default=128)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--out-dir', type=Path, default=Path('results/phase1_e4_c8192'))
    args = ap.parse_args(argv)
    if not (args.repeats >= 1 and 1 <= args.a_predict <= 512
            and 1 <= args.bc_predict <= 512 and args.timeout > 0):
        ap.error('invalid repeats, prediction lengths or timeout')
    if any(not (0 <= w <= 60) for w in args.waits):
        ap.error('wait must be 0..60 seconds')
    if not (300 <= args.a_tokens <= 3500 and 300 <= args.short_tokens <= 3500
            and 300 <= args.long_tokens <= 3500):
        ap.error('prompt token targets must be 300..3500')
    # Keep unique scenarios and waits; duplicate values could overwrite aggregates.
    if len(args.mixes) != len(set(args.mixes)):
        ap.error('mixes must be unique')
    if len(args.waits) != len(set(args.waits)):
        ap.error('waits must be unique')
    startup = ex.check_local_server_context(args.expect_ctx)
    print(f'Verified local llama-server -c {startup["server_context_capacity_arg"]}', flush=True)
    health = kp.url_json(args.url, 'GET', '/health', timeout=8)
    print('Health:', health, flush=True)
    slots = kp.url_json(args.url, 'GET', '/slots', timeout=8)
    if isinstance(slots, dict):
        slots = slots.get('slots')
    if not isinstance(slots, list) or len(slots) != 2:
        ap.error('E4 requires exactly 2 visible server slots (-np 2)')
    if any(s.get('is_processing') for s in slots):
        ap.error('server is busy: stop external clients; E4 erases both caches')
    slot_limits = [s.get('n_ctx') for s in slots]
    # Confirm erase endpoint before generating prompts and consuming GPU work.
    ex.eraser(args.url, n=2)
    if any(isinstance(n, int) and n < (args.long_tokens + args.bc_predict + 64)
           for n in slot_limits):
        ap.error(f'per-slot context {slot_limits} too small for long prompt + output')

    kinds = [('A', args.a_tokens),
             ('B_SHORT', args.short_tokens), ('C_SHORT', args.short_tokens),
             ('B_LONG', args.long_tokens), ('C_LONG', args.long_tokens)]
    prompts = {}
    sizes = {}
    for name, target in kinds:
        p, actual, method = kp.build_prompt(args.url, 'E4_'+name, target)
        if method != 'tokenize' or actual is None:
            ap.error(f'/tokenize must work for controlled comparisons: {name}, {method}')
        prompts[name] = p
        sizes[name] = actual
    print('Measured prompt tokens:', sizes, flush=True)
    print('Server slot context limits:', slot_limits, flush=True)

    outfile = args.out_dir / 'trials.jsonl'
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if outfile.exists():
        ap.error(f'{outfile} already exists; choose a new --out-dir (avoid mixing runs)')
    trial_list = []
    combinations = len(args.waits) * len(args.mixes) * args.repeats * 2
    print(f'E4: {combinations} isolated arms, each A1 / B / C / A2', flush=True)
    with outfile.open('w', encoding='utf-8') as f:
        counter = 0
        for rep in range(1, args.repeats + 1):
            for mi, mix in enumerate(args.mixes):
                b_long = mix in ('long_short', 'long_long')
                c_long = mix in ('short_long', 'long_long')
                bp = prompts['B_LONG' if b_long else 'B_SHORT']
                cp = prompts['C_LONG' if c_long else 'C_SHORT']
                for wi, wait in enumerate(args.waits):
                    # Rotate order across repeats and conditions to mitigate thermal/drift effects.
                    policies = ('preserve', 'reuse') if (rep + mi + wi) % 2 == 0 else ('reuse', 'preserve')
                    pair = {}
                    for policy in policies:
                        counter += 1
                        arm = run_arm(args.url, prompts['A'], bp, cp, policy, wait,
                                      args.a_predict, args.bc_predict, args.timeout,
                                      rep, mix)
                        f.write(json.dumps(arm, ensure_ascii=False) + '\n')
                        f.flush()
                        trial_list.append(arm)
                        pair[policy] = arm
                        print(f'[{counter}/{combinations}] rep={rep} mix={mix} wait={wait:g}s '
                              f'{policy} OK={arm["completed"]}/3 '
                              f'makespan={arm["total_makespan_s"]:.3f}s '
                              f'A2-prefill={arm.get("a2_prompt_ms")}ms '
                              f'cache_n={arm.get("a2_cache_n")} '
                              f'Bqueue={arm.get("b_queue_s")} '
                              f'Cqueue={arm.get("c_queue_s")}', flush=True)
                    if pair['preserve']['a2_prompt_sha256'] != pair['reuse']['a2_prompt_sha256']:
                        print('WARNING: A2 exact prompts differ between paired policies; '
                              'inspect hashes in CSV and rerun if material.', flush=True)

    cols, rows = csv_rows(trial_list)
    with (args.out_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    grouped, paired = aggregate(trial_list, args.mixes, args.waits)
    write_csv(args.out_dir / 'group_medians.csv', grouped)
    write_csv(args.out_dir / 'policy_comparison.csv', paired)
    result = {
        'created_utc': now_utc(),
        'experiment': 'E4',
        'description': ('Client-enforced slot reservation vs reuse, controlled tool wait. '
                        'Not CPU/SSD swap, KV pinning, or an adaptive scheduler.'),
        'configuration': {**vars(args), 'out_dir': str(args.out_dir)},
        'server': startup,
        'slot_context_limits': slot_limits,
        'prompt_sizes': sizes,
        'group_medians': grouped,
        'policy_comparison': paired,
        'trials': trial_list,
    }
    (args.out_dir / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
    print('\nE4 paired comparison: positive delta means REUSE faster (smaller total makespan).')
    for row in paired:
        print(f'  {row["mix"]:>11} wait={row["wait_s"]:>4g}s '
              f'pairs={row["paired_valid_n"]} '
              f'preserve-minus-reuse={row["median_preserve_minus_reuse_makespan_s"]}s '
              f'({row["winner_by_median"]})')
    print(f'Saved: {args.out_dir / "summary.csv"}; '
          f'{args.out_dir / "policy_comparison.csv"}; '
          f'{args.out_dir / "group_medians.csv"}; '
          f'{args.out_dir / "summary.json"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
