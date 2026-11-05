#!/usr/bin/env python3
"""E5: online slot-reuse decisions when tool completion time is unknown.

A1 completes; an external tool begins; B/C arrive immediately. The scheduling
code can observe tool_ready.is_set() and elapsed time, but cannot read the actual
hidden delay. This is a *client-side* policy experiment with 2 llama.cpp slots.

Policies:
    preserve : reserve slot0 for A2, B->C use slot1
    reuse    : B uses slot0, C uses slot1, A2 uses slot0 after B
    ttl:<s>  : B uses slot1. C waits in a queue; if the tool is still running
               after s seconds, C may take slot0; otherwise C waits for slot1.
               A2 uses slot0 when tool finishes and slot0 becomes available.

TTL is a deliberately simple baseline; it is NOT expected to beat static
policies. There is no KV pinning, swap, offline training, or LLM prediction.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import threading
import time

import kv_memory_probe as kp
import phase1_experiments as ex


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def median(vals):
    values = [float(x) for x in vals if isinstance(x, (float, int))
              and not isinstance(x, bool) and math.isfinite(float(x))]
    return round(statistics.median(values), 6) if values else None


def mean(vals):
    values = [float(x) for x in vals if isinstance(x, (float, int))
              and not isinstance(x, bool) and math.isfinite(float(x))]
    return round(statistics.mean(values), 6) if values else None


def p95(vals):
    values = sorted(float(x) for x in vals if isinstance(x, (float, int))
                    and not isinstance(x, bool) and math.isfinite(float(x)))
    if not values:
        return None
    i = 0.95 * (len(values) - 1)
    lo = int(i)
    hi = min(lo + 1, len(values) - 1)
    return round(values[lo] * (hi-i) + values[hi] * (i-lo), 6)


def policy_name(ttl):
    return 'ttl:' + format(ttl, '.6g')


def parse_policy(policy):
    if policy in ('preserve', 'reuse'):
        return policy, None
    if policy.startswith('ttl:'):
        ttl = float(policy.split(':', 1)[1])
        if math.isfinite(ttl) and ttl >= 0:
            return 'ttl', ttl
    raise ValueError('invalid policy '+policy)


def draw_episode_delays(values, episodes, seed):
    """Balanced randomized delays. Runtime gets only tool_done event, never values."""
    if not values or episodes < 1:
        raise ValueError('need wait values and episodes>=1')
    rng = random.Random(seed)
    shuffled = (list(values) * (episodes // len(values))) + [rng.choice(values)
                for _ in range(episodes % len(values))]
    rng.shuffle(shuffled)
    return shuffled


def run_arm(base, a_prompt, b_prompt, c_prompt, policy, hidden_tool_delay_s,
            a_predict, bc_predict, timeout, episode, mix, poll_s=0.003):
    """Run one isolated arm. DO NOT pass hidden_tool_delay_s into the scheduler.

    The only recipient of hidden_tool_delay_s is the synthetic tool_worker.
    In real use this would be an externally completed tool Future/event.
    """
    kind, ttl_s = parse_policy(policy)
    if not (0 <= hidden_tool_delay_s <= 120 and poll_s > 0 and timeout > 0):
        raise ValueError('invalid delay/poll/timeout')
    erased = ex.eraser(base, n=2)
    origin = time.monotonic()
    now = lambda: time.monotonic() - origin

    def request_task(name, prompt, slot, limit, use_cache, arrival_s):
        start = now()
        try:
            raw = ex.execute(base, prompt, limit, timeout, slot=slot, cache=use_cache)
        except Exception as err:
            raw = {'ok': False, 'seconds': now() - start, 'error': str(err)}
        end = now()
        return {
            'name': name, 'slot': slot, 'arrival_s': round(arrival_s, 6),
            'start_s': round(start, 6), 'end_s': round(end, 6),
            'queue_s': round(max(0, start - arrival_s), 6),
            'response': ex.proof_metrics(raw), '_raw': raw,
        }

    a1 = request_task('A1', a_prompt, 0, a_predict, True, 0)
    if not a1['response']['ok']:
        raise RuntimeError('A1 failed: '+str(a1['response']['error']))
    a1_end = a1['end_s']
    a2_prompt = ex.append_step(a_prompt, a1['_raw'])
    a2_hash = hashlib.sha256(a2_prompt.encode('utf-8')).hexdigest()

    # The *controller* owns the true delay. Only worker_tool ever reads it.
    tool_ready = threading.Event()
    tool_stamps = {}
    def worker_tool():
        if hidden_tool_delay_s:
            time.sleep(hidden_tool_delay_s)
        tool_stamps['ready_s'] = now()
        tool_ready.set()
    tool_thread = threading.Thread(target=worker_tool, name='hidden-tool', daemon=True)
    tool_thread.start()

    started = {'A1': True, 'B': False, 'C': False, 'A2': False}
    finished = {'A1': True, 'B': False, 'C': False, 'A2': False}
    requests = {'A1': a1}
    futures = {}
    decisions = []
    # Only the dispatcher updates these slot assignments.
    busy = {0: None, 1: None}

    # Client-side scheduler sees only the tool-ready event and elapsed time.
    # It receives NO hidden delay, probability sample, or future tool timestamp.
    def online_dispatch(pool):
        def launch(name, prompt, slot, tokens, cache, arrival):
            assert busy[slot] is None, 'slot is already booked'
            assert not started[name], 'task started twice'
            started[name] = True
            busy[slot] = name
            futures[name] = pool.submit(request_task, name, prompt, slot, tokens,
                                        cache, arrival)
            decisions.append({'at_s': round(now(), 6), 'action': 'start_'+name,
                              'slot': slot,
                              'tool_ready_known': tool_ready.is_set()})

        if kind == 'reuse':
            launch('B', b_prompt, 0, bc_predict, False, a1_end)
            launch('C', c_prompt, 1, bc_predict, False, a1_end)
        else:
            launch('B', b_prompt, 1, bc_predict, False, a1_end)

        # TTL starts when A's tool begins, not at process/agent start.
        # The exact tool completion time is NOT used in this function.
        deadline = a1_end + ttl_s if ttl_s is not None else None
        watchdog = time.monotonic() + max(4*timeout + 125, 150)
        while not (finished['B'] and finished['C'] and finished['A2']):
            # Reclaim slots of completed requests, preserving precise request timestamps.
            for name in ('B', 'C', 'A2'):
                future = futures.get(name)
                if future and not finished[name] and future.done():
                    try:
                        record = future.result()
                    except Exception as err:
                        # Failure stays in metrics, but the dispatcher must still progress.
                        record = {'name': name, 'slot': next((slot for slot,n in busy.items()
                                    if n==name), -1), 'end_s': round(now(), 6),
                                  'start_s': None, 'arrival_s': a1_end, 'queue_s': None,
                                  'response': {'ok': False, 'error': repr(err)}}
                    requests[name] = record
                    finished[name] = True
                    slot = record['slot']
                    if slot in busy and busy[slot] == name:
                        busy[slot] = None
                    decisions.append({'at_s': round(now(), 6), 'action': 'finish_'+name,
                                      'slot': slot})

            if kind == 'preserve':
                if not started['C'] and finished['B'] and busy[1] is None:
                    launch('C', c_prompt, 1, bc_predict, False, a1_end)
                if not started['A2'] and tool_ready.is_set() and busy[0] is None:
                    launch('A2', a2_prompt, 0, a_predict, True, tool_stamps['ready_s'])
            elif kind == 'reuse':
                if not started['A2'] and tool_ready.is_set() and busy[0] is None:
                    launch('A2', a2_prompt, 0, a_predict, True, tool_stamps['ready_s'])
            else:
                # TTL policy: A2 (if ready) has priority over evicting its cache.
                # A2 can run only after any C already dispatched on slot0 finishes.
                if not started['A2'] and tool_ready.is_set() and busy[0] is None:
                    launch('A2', a2_prompt, 0, a_predict, True, tool_stamps['ready_s'])
                # C is queued until B releases slot1 OR the TTL expires while
                # tool is still pending. Eviction only happens to serve C.
                if not started['C']:
                    if finished['B'] and busy[1] is None:
                        launch('C', c_prompt, 1, bc_predict, False, a1_end)
                    elif (not tool_ready.is_set() and now() >= deadline
                          and busy[0] is None and not started['A2']):
                        launch('C', c_prompt, 0, bc_predict, False, a1_end)
                        decisions.append({'at_s': round(now(), 6),
                                          'action': 'ttl_expired_reuse_slot0',
                                          'ttl_s': ttl_s})
            if time.monotonic() > watchdog:
                raise TimeoutError('dispatcher timeout; check server or tool worker')
            if not (finished['B'] and finished['C'] and finished['A2']):
                time.sleep(poll_s)

    # Tool runs asynchronously; policy is blind to hidden_tool_delay_s.
    with ThreadPoolExecutor(max_workers=3) as pool:
        online_dispatch(pool)
    tool_thread.join(timeout=1)
    if not tool_ready.is_set():
        raise RuntimeError('tool never completed')
    logs = {name: {k: v for k, v in record.items() if k != '_raw'}
            for name, record in requests.items()}
    makespan = max(x['end_s'] for x in logs.values())
    completed = sum(bool(logs.get(name, {}).get('response', {}).get('ok'))
                    for name in ('B', 'C', 'A2'))
    a2 = logs['A2']
    b = logs['B']
    c = logs['C']
    elapsed_tool = round(tool_stamps['ready_s'] - a1_end, 6)
    ttl_reused = any(x['action'] == 'ttl_expired_reuse_slot0' for x in decisions)
    return {
        'experiment': 'E5', 'timestamp_utc': utcnow(), 'mix': mix,
        'episode': episode, 'policy': policy,
        # Evaluation labels are stored here only AFTER policy execution.
        'hidden_tool_delay_s': hidden_tool_delay_s,
        'observed_tool_delay_s': elapsed_tool,
        'tool_ready_s': round(tool_stamps['ready_s'], 6),
        'a1_end_s': a1_end,
        'completed': completed, 'all_ok': completed == 3,
        'total_makespan_s': round(makespan, 6),
        'phase_makespan_s': round(makespan - a1_end, 6),
        'throughput_completed_tasks_s': round(completed/makespan, 6),
        'a_completion_s': a2['end_s'],
        'a2_response_after_tool_s': round(a2['end_s'] - tool_stamps['ready_s'], 6),
        'a2_queue_s': a2['queue_s'],
        'a2_cache_n': a2['response'].get('cache_n'),
        'a2_prompt_n': a2['response'].get('prompt_n'),
        'a2_prompt_ms': a2['response'].get('prompt_ms'),
        'b_queue_s': b['queue_s'], 'c_queue_s': c['queue_s'],
        'b_completion_from_arrival_s': round(b['end_s']-a1_end, 6),
        'c_completion_from_arrival_s': round(c['end_s']-a1_end, 6),
        'ttl_eviction_triggered': ttl_reused,
        'a2_prompt_sha256': a2_hash,
        'erase_responses': erased,
        'decisions': decisions,
        'timings': logs,
        'limitations': ('Synthetic tool delay. TTL can choose a slot for queued C only. '
                        'Cache remains managed by llama.cpp; slot reservation is client-side.'),
    }


def flatten(t):
    fields = ('mix', 'episode', 'policy', 'hidden_tool_delay_s',
              'observed_tool_delay_s', 'completed', 'all_ok',
              'total_makespan_s', 'phase_makespan_s', 'throughput_completed_tasks_s',
              'a_completion_s', 'a2_response_after_tool_s', 'a2_queue_s',
              'a2_cache_n', 'a2_prompt_n', 'a2_prompt_ms',
              'b_queue_s', 'c_queue_s', 'b_completion_from_arrival_s',
              'c_completion_from_arrival_s', 'ttl_eviction_triggered',
              'a2_prompt_sha256')
    return {key:t.get(key) for key in fields}


def analyze(trials, mixes, policies):
    groups = []
    episodes = []
    for mix in mixes:
        sub = [t for t in trials if t['mix'] == mix]
        for policy in policies:
            ours = [t for t in sub if t['policy'] == policy]
            # Failed arms still count in completion rate, but successful-only
            # latency summaries exclude failures (not an implicit success).
            ok = [t for t in ours if t['all_ok']]
            gr = {
                'mix':mix, 'policy':policy, 'arms':len(ours), 'all_ok_arms':len(ok),
                'completed_total':sum(t['completed'] for t in ours),
                'submitted_total':3*len(ours),
                'completion_fraction':round(sum(t['completed'] for t in ours)/(3*len(ours)),6)
                    if ours else None,
                'total_completed_per_total_makespan_s':round(sum(t['completed'] for t in ours)/
                    sum(t['total_makespan_s'] for t in ours), 6) if ours else None,
                'mean_makespan_s': mean([t['total_makespan_s'] for t in ok]),
                'median_makespan_s': median([t['total_makespan_s'] for t in ok]),
                'p95_makespan_s': p95([t['total_makespan_s'] for t in ok]),
                'mean_a2_response_after_tool_s': mean([t['a2_response_after_tool_s'] for t in ok]),
                'mean_a2_prefill_ms': mean([t['a2_prompt_ms'] for t in ok]),
                'mean_a2_cache_n': mean([t['a2_cache_n'] for t in ok]),
                'mean_c_queue_s': mean([t['c_queue_s'] for t in ok]),
                'ttl_eviction_count':sum(bool(t['ttl_eviction_triggered']) for t in ours),
                'mean_regret_vs_best_static_s':None,
            }
            groups.append(gr)
        for episode in sorted({t['episode'] for t in sub}):
            these = {t['policy']:t for t in sub if t['episode']==episode}
            static = [these.get('preserve'),these.get('reuse')]
            valid = [t for t in static if t is not None and t['all_ok']]
            if not valid:
                continue
            best = min(t['total_makespan_s'] for t in valid)
            best_name = min(valid, key=lambda t:t['total_makespan_s'])['policy']
            for policy in policies:
                current = these.get(policy)
                if current is None:
                    continue
                regret = (round(current['total_makespan_s']-best,6)
                          if current['all_ok'] else None)
                episodes.append({'mix':mix, 'episode':episode,
                                 'evaluation_only_tool_delay_s':current['hidden_tool_delay_s'],
                                 'policy':policy, 'all_ok':current['all_ok'],
                                 'makespan_s':current['total_makespan_s'],
                                 'best_static_in_hindsight':best_name,
                                 'best_static_makespan_s':best,
                                 'delta_vs_best_static_s':regret,
                                 'ttl_eviction_triggered':current['ttl_eviction_triggered']})
    for g in groups:
        vals = [t['delta_vs_best_static_s'] for t in episodes
                if t['mix']==g['mix'] and t['policy']==g['policy']
                and t['delta_vs_best_static_s'] is not None]
        g['mean_regret_vs_best_static_s'] = mean(vals)
    return groups, episodes


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:8080')
    ap.add_argument('--expect-ctx', type=int, default=8192)
    ap.add_argument('--mixes', nargs='+', choices=['short_short','short_long','long_short','long_long'],
                    default=['short_short','short_long'])
    ap.add_argument('--wait-values', nargs='+', type=float, default=[0,0.5,3,8],
                    help='Hidden tool-delay possibilities; balanced shuffled across episodes')
    ap.add_argument('--episodes-per-mix', type=int, default=8)
    ap.add_argument('--ttls', nargs='+', type=float, default=[0.5,1.5],
                    help='TTL thresholds, relative to tool start (seconds)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--a-tokens', type=int, default=1200)
    ap.add_argument('--short-tokens', type=int, default=1200)
    ap.add_argument('--long-tokens', type=int, default=2600)
    ap.add_argument('--a-predict', type=int, default=64)
    ap.add_argument('--bc-predict', type=int, default=128)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--poll-ms', type=float, default=3)
    ap.add_argument('--out-dir', type=Path, default=Path('results/phase1_e5_c8192'))
    args = ap.parse_args(argv)
    if len(args.mixes)!=len(set(args.mixes)) or len(args.ttls)!=len(set(args.ttls)):
        ap.error('mixes and ttls must not contain duplicates')
    if (args.episodes_per_mix < 1 or args.timeout <= 0 or args.poll_ms <= 0
        or any(not math.isfinite(v) or v < 0 or v > 60 for v in args.wait_values)
        or any(not math.isfinite(t) or t < 0 or t > 60 for t in args.ttls)):
        ap.error('invalid episodes, tool waits, ttls or timeout')
    if (not all(300 <= v <= 3500 for v in (args.a_tokens,args.short_tokens,args.long_tokens))
        or not(1<=args.a_predict<=512 and 1<=args.bc_predict<=512)):
        ap.error('invalid prompt sizes or generation lengths')

    server = ex.check_local_server_context(args.expect_ctx)
    print('Verified local llama-server -c',server['server_context_capacity_arg'],flush=True)
    print('Health:',kp.url_json(args.url,'GET','/health',timeout=8),flush=True)
    slots = kp.url_json(args.url,'GET','/slots',timeout=8)
    if isinstance(slots,dict):
        slots = slots.get('slots')
    if not isinstance(slots,list) or len(slots)!=2:
        ap.error('requires exactly 2 llama-server slots (-np 2)')
    if any(s.get('is_processing') for s in slots):
        ap.error('server busy; no other clients allowed (will erase both slot caches)')
    slot_limits = [s.get('n_ctx') for s in slots]
    if any(isinstance(v,int) and v < args.long_tokens+args.bc_predict+64 for v in slot_limits):
        ap.error('per-slot context too small for longest request')
    ex.eraser(args.url,n=2)

    prompt_cfg = [('A',args.a_tokens),('B_SHORT',args.short_tokens),
                  ('C_SHORT',args.short_tokens),('B_LONG',args.long_tokens),
                  ('C_LONG',args.long_tokens)]
    prompts = {}
    sizes = {}
    for label,number in prompt_cfg:
        prompt,actual,source = kp.build_prompt(args.url,'E5_'+label,number)
        if source!='tokenize' or actual is None:
            ap.error(f'/tokenize required for exact lengths: {label} {source}')
        prompts[label] = prompt
        sizes[label] = actual
    print('Measured prompt tokens:',sizes,flush=True)
    print('Server slot limits:',slot_limits,flush=True)

    out = args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    trialfile = out/'trials.jsonl'
    if trialfile.exists():
        ap.error(f'{trialfile} already exists; choose a new --out-dir')
    policies = ['preserve','reuse'] + [policy_name(t) for t in args.ttls]
    schedules = {}
    for mix_idx,mix in enumerate(args.mixes):
        schedules[mix] = draw_episode_delays(args.wait_values,args.episodes_per_mix,
                                             args.seed + 1009*mix_idx)
    total = len(args.mixes)*args.episodes_per_mix*len(policies)
    print(f'E5: {total} arms, {len(policies)} policies, '
          f'{args.episodes_per_mix} paired episodes/mix; hidden, balanced delays.', flush=True)
    rng=random.Random(args.seed+101)
    trials=[]
    counter=0
    with trialfile.open('w',encoding='utf-8') as f:
        for mix in args.mixes:
            bp = prompts['B_LONG' if mix in ('long_short','long_long') else 'B_SHORT']
            cp = prompts['C_LONG' if mix in ('short_long','long_long') else 'C_SHORT']
            for episode, actual_wait in enumerate(schedules[mix],1):
                # Same hidden delay is replayed for each arm of the matched episode.
                # Runtime policies never get passed the delay; only worker_tool does.
                run_order = policies[:]
                rng.shuffle(run_order)
                prompt_hashes = set()
                for policy in run_order:
                    counter+=1
                    arm=run_arm(args.url,prompts['A'],bp,cp,policy,actual_wait,
                                args.a_predict,args.bc_predict,args.timeout,
                                episode,mix,args.poll_ms/1000)
                    f.write(json.dumps(arm,ensure_ascii=False)+'\n')
                    f.flush()
                    trials.append(arm)
                    prompt_hashes.add(arm['a2_prompt_sha256'])
                    print(f'[{counter}/{total}] mix={mix} episode={episode} '
                          f'wait(hidden during decision)={actual_wait:g}s '
                          f'{policy} completed={arm["completed"]}/3 '
                          f'makespan={arm["total_makespan_s"]:.3f}s '
                          f'A2 cache_n={arm["a2_cache_n"]} '
                          f'A2 response-after-tool={arm["a2_response_after_tool_s"]:.3f}s '
                          f'Cqueue={arm["c_queue_s"]:.3f}s '
                          f'TTL-reuse={arm["ttl_eviction_triggered"]}',flush=True)
                if len(prompt_hashes)!=1:
                    print(f'WARNING: A2 prompt hashes vary within episode {mix}/{episode}; '
                          'check trials.jsonl before relying on paired comparisons.',flush=True)
    groups,episode_stats=analyze(trials,args.mixes,policies)
    write_csv(out/'trials.csv',[flatten(x) for x in trials])
    write_csv(out/'policy_summary.csv',groups)
    write_csv(out/'episode_comparison.csv',episode_stats)
    metadata={'experiment':'E5','created_utc':utcnow(),
              'description':'Online TTL with unknown tool latency; no swap or KV scheduler.',
              'configuration':{**vars(args),'out_dir':str(out)},
              'server':server,'slot_context_limits':slot_limits,
              'prompt_token_counts':sizes,'policy_list':policies,
              'evaluation_only_hidden_delay_schedules':schedules,
              'summary':groups}
    (out/'summary.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',
                                    encoding='utf-8')
    print('\nE5 policy summary (lower mean makespan is better, compare success first):')
    for r in groups:
        print(f'  {r["mix"]:>11} {r["policy"]:>9}: '
              f'OK={r["all_ok_arms"]}/{r["arms"]} '
              f'mean makespan={r["mean_makespan_s"]}s '
              f'mean A2 prefill={r["mean_a2_prefill_ms"]}ms '
              f'TTL reuses={r["ttl_eviction_count"]} '
              f'mean Δ vs hindsight-static={r["mean_regret_vs_best_static_s"]}s')
    print('Saved:',out/'trials.csv',out/'policy_summary.csv',
          out/'episode_comparison.csv',out/'trials.jsonl',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
