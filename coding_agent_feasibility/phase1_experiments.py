#!/usr/bin/env python3
"""Phase 1 exploratory llama.cpp experiments, no scheduler implementation.

E1: controlled tool pause x prompt cache on/off.
E2: interleaved other requests x slot placement x prefix reuse.
E3: fixed pair concurrency vs serialized admission as a feasibility control.

Uses the existing kv_memory_probe.py. Never infers KV cell occupancy from GPU memory.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import shlex
import subprocess
from pathlib import Path
import statistics
import sys
import time

import kv_memory_probe as kp


def check_local_server_context(expected):
    """Refuse mislabeled -c sweeps. Assumes llama-server runs on this Linux host."""
    try:
        output = subprocess.check_output(['pgrep', '-af', 'llama-server'],
                                         text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'cannot inspect running llama-server: {exc}') from exc
    found = []
    for line in output.splitlines():
        try:
            argv = shlex.split(line)
        except ValueError:
            continue
        # pgrep prefixes a PID; reject helper commands and shells.
        for i,part in enumerate(argv):
            if part.endswith('/llama-server') or part == 'llama-server':
                args=argv[i+1:]
                try:
                    idx=args.index('-c')
                    found.append((int(args[idx+1]),line))
                except (ValueError, IndexError):
                    pass
                break
    if len(found)!=1:
        raise RuntimeError(f'expected one identifiable llama-server process, found {len(found)}: {found}')
    actual,command=found[0]
    if expected is not None and actual!=expected:
        raise RuntimeError(f'server -c={actual}, but --expect-ctx={expected}; '
                           'restart the correct llama-server before labeling this run')
    return {'server_context_capacity_arg':actual,'process_command':command}


def to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def eraser(base, n=2):
    """Hard isolation requirement: never quietly treat cache-contaminated trials as fresh."""
    records = []
    for i in range(n):
        try:
            r = kp.url_json(base, 'POST', f'/slots/{i}?action=erase', {}, timeout=10)
            if not isinstance(r, dict) or r.get('id_slot') not in (None, i):
                raise RuntimeError(f'unexpected response: {r!r}')
            records.append(r)
        except Exception as exc:
            raise RuntimeError(
                f'Could not erase slot {i}: {exc}. Cannot run controlled cache experiment; '
                'check llama.cpp /slots/{id}?action=erase endpoint and restart with idle slots.'
            ) from exc
    return records


def execute(base, prompt, output_tokens, timeout, slot=0, cache=True):
    return kp.perform(base, prompt, output_tokens, timeout,
                      id_slot=slot, cache_prompt=cache)


def append_step(prompt, step1):
    # Always a genuine extension, not an identical repeated request.
    text = str(step1.get('completion', '')).strip()[:256]
    return prompt + '\nAssistant: ' + text + '\nTool result: grep verified two related functions; continue the analysis.\n'


def proof_metrics(response):
    t = response.get('timings') or {}
    return {
        'ok': response.get('ok'),
        'request_s': response.get('seconds'),
        'cache_n': t.get('cache_n'),
        'prompt_n': t.get('prompt_n'),
        'prompt_ms': t.get('prompt_ms'),
        'predicted_n': t.get('predicted_n'),
        'cached_field': response.get('tokens_cached'),
        'evaluated_field': response.get('tokens_evaluated'),
        'error': response.get('error'),
    }


def do_e1(base, source_prompt, wait_values, n_predict, timeout, slot=0):
    records = []
    # Reverse order per delay between repetitions to mitigate drift.
    for wait in wait_values:
        for cache in (True, False):
            eraser(base)
            first = execute(base, source_prompt, n_predict, timeout, slot=slot, cache=True)
            if not first['ok']:
                raise RuntimeError('E1 first LLM step failed: '+str(first.get('error')))
            time.sleep(wait)
            second = execute(base, append_step(source_prompt, first), n_predict, timeout,
                             slot=slot, cache=cache)
            rec = {
                'experiment':'E1', 'wait_s':wait, 'cache_enabled':cache,
                'first':proof_metrics(first),'second':proof_metrics(second),
                'notes':'Controlled time.sleep simulates an external tool wait; no real tool was invoked.',
            }
            records.append(rec)
            print(f'[E1] gap={wait:g}s cache={"on" if cache else "off"} '
                  f'first={first["ok"]} second={second["ok"]} '
                  f'prefill_n={rec["second"]["prompt_n"]} '
                  f'cache_n={rec["second"]["cache_n"]} '
                  f'prefill_ms={rec["second"]["prompt_ms"]}', flush=True)
    return records


def do_e2(base, source_prompt, interferer_prompts, gap_s, n_predict, timeout):
    records = []
    # No-interference baseline plus same-slot vs other-slot, short/long B.
    scenarios = [('none', None, None)]
    scenarios.extend((f'{name}_{placement}', p, placement)
                     for name,p in interferer_prompts.items()
                     for placement in ('other_slot', 'same_slot'))
    for scenario,prompt_b,placement in scenarios:
        eraser(base)
        first = execute(base, source_prompt, n_predict, timeout, slot=0, cache=True)
        if not first['ok']:
            raise RuntimeError('E2 first LLM step failed: '+str(first.get('error')))
        begin_gap = time.monotonic()
        b = None
        if prompt_b is not None:
            bslot = 1 if placement == 'other_slot' else 0
            b = execute(base, prompt_b, n_predict, timeout, slot=bslot, cache=False)
        spent = time.monotonic()-begin_gap
        if spent < gap_s:
            time.sleep(gap_s-spent)
        real_gap = time.monotonic()-begin_gap
        second = execute(base, append_step(source_prompt, first), n_predict, timeout,
                         slot=0, cache=True)
        rec = {'experiment':'E2','scenario':scenario,'gap_target_s':gap_s,
               'gap_actual_s':round(real_gap,3),
               'step1':proof_metrics(first),
               'interferer':proof_metrics(b) if b else None,
               'step2':proof_metrics(second),
               'note':'same_slot replaces slot-0 context; other_slot can still compete for shared KV. '
                      'Cache reuse measured by timings cache_n/prompt_n, not allocated VRAM.'}
        records.append(rec)
        print(f'[E2] {scenario}: B={None if b is None else b["ok"]} '
              f'A2={second["ok"]} cache_n={rec["step2"]["cache_n"]} '
              f'prefill_n={rec["step2"]["prompt_n"]} '
              f'prefill_ms={rec["step2"]["prompt_ms"]} gap={real_gap:.2f}s', flush=True)
    return records


def do_e3(base, pairs, n_predict, timeout, interval, out_dir):
    records = []
    for kind,(p1,p2) in pairs.items():
        eraser(base)
        concurrency, samples = kp.run_pair(base, p1,p2,n_predict,timeout,interval)
        sample_path = out_dir / f'e3_{kind}_samples.jsonl'
        with sample_path.open('w',encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample,ensure_ascii=False)+'\n')
        eraser(base)
        start=time.monotonic()
        # Pin both requests to same slot to ensure strict sequential use.
        a=execute(base,p1,n_predict,timeout,slot=0,cache=False)
        b=execute(base,p2,n_predict,timeout,slot=0,cache=False)
        seq_wall=round(time.monotonic()-start,3)
        record={
            'experiment':'E3','pair':kind,
            'concurrent':{'wall_s':concurrency['wall_seconds'],
                          'requests':[proof_metrics(r) for r in concurrency['requests']],
                          'observations':concurrency['observation_summary']},
            'serial':{'wall_s':seq_wall,'requests':[proof_metrics(a),proof_metrics(b)]},
            'note':'Manual execution-order control, NOT a memory-aware scheduler. '
                   'Concurrent baseline disables cache_prompt; we erase slot caches before each arm.'
        }
        records.append(record)
        print(f'[E3] {kind}: concurrent {[r["ok"] for r in concurrency["requests"]]} '
              f'({concurrency["wall_seconds"]}s) / serial {[a["ok"],b["ok"]]} '
              f'({seq_wall}s) busy_peak={concurrency["observation_summary"].get("busy_slots_peak")}',flush=True)
    return records


def save(results, output_dir):
    target=output_dir/'summary.json'
    target.write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
    rows=[]
    for r in results['records']:
        if r['experiment']=='E1':
            rows.append({'exp':'E1','name':f'wait={r["wait_s"]}:cache={r["cache_enabled"]}',
                         **r['second']})
        if r['experiment']=='E2':
            rows.append({'exp':'E2','name':r['scenario'],**r['step2'],
                         'interferer_ok': None if r['interferer'] is None else r['interferer']['ok']})
        if r['experiment']=='E3':
            for typ in ('concurrent','serial'):
                req=r[typ]['requests']
                rows.append({'exp':'E3','name':r['pair']+':'+typ,
                             'ok':sum(bool(x['ok']) for x in req),
                             'request_s':r[typ]['wall_s'],
                             'error':' | '.join(str(x.get('error') or '')[:100] for x in req)})
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (output_dir/'summary.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)
    print(f'Saved: {target}; {output_dir/"summary.csv"}',flush=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url',default='http://127.0.0.1:8080')
    ap.add_argument('--expect-ctx',type=int,default=None,help='Assert running server has this -c; requires local Linux pgrep')
    ap.add_argument('--experiment',choices=('e1','e2','e3','all'),default='e1')
    ap.add_argument('--a-tokens',type=int,default=1200)
    ap.add_argument('--b-short-tokens',type=int,default=1200)
    ap.add_argument('--b-long-tokens',type=int,default=2600)
    ap.add_argument('--n-predict',type=int,default=64)
    ap.add_argument('--waits',type=float,nargs='+',default=[0,3,8])
    ap.add_argument('--gap',type=float,default=3)
    ap.add_argument('--timeout',type=float,default=180)
    ap.add_argument('--interval',type=float,default=0.3)
    ap.add_argument('--out-dir',type=Path,default=Path('results/phase1_e1'))
    args=ap.parse_args(argv)
    if not (300 <= args.a_tokens <= 3500 and 300 <= args.b_short_tokens <= 3500
            and 300 <= args.b_long_tokens <= 3500):
        ap.error('token targets must be between 300 and 3500')
    if not (0 <= args.gap <= 60 and all(0 <= x <= 60 for x in args.waits)
            and 1 <= args.n_predict <= 256 and .1 <= args.interval <= 3):
        ap.error('invalid gap/waits/n-predict/interval')
    startup=check_local_server_context(args.expect_ctx) if args.expect_ctx is not None else None
    if startup:
        print('Verified server -c:',startup['server_context_capacity_arg'],flush=True)
    status=kp.url_json(args.url,'GET','/health',timeout=8)
    print('Server health:',status)
    observation=kp.read_observation(args.url,time.monotonic())
    slots=observation.get('slots') or []
    if len(slots)<2:
        ap.error('requires two llama.cpp server slots; restart with -np 2')
    if any(s.get('is_processing') for s in slots):
        ap.error('server busy: stop other clients; experiments erase caches')
    # This erases slot caches, so never run against a server in use by others.
    eraser(args.url)
    p_a, ca, ha=kp.build_prompt(args.url, 'AGENT_A', args.a_tokens)
    p_bs, cs, hs=kp.build_prompt(args.url, 'AGENT_B_SHORT', args.b_short_tokens)
    p_bl, cl, hl=kp.build_prompt(args.url, 'AGENT_B_LONG', args.b_long_tokens)
    print('Prompt token counts:', {'A':ca,'B_short':cs,'B_long':cl}, flush=True)
    if any(v is None for v in (ca,cs,cl)):
        ap.error('require working /tokenize; approximate token counts are not adequate for controlled experiment')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    records=[]
    if args.experiment in ('e1','all'):
        records.extend(do_e1(args.url,p_a,args.waits,args.n_predict,args.timeout))
    if args.experiment in ('e2','all'):
        records.extend(do_e2(args.url,p_a,{'short':p_bs,'long':p_bl},
                             args.gap,args.n_predict,args.timeout))
    if args.experiment in ('e3','all'):
        records.extend(do_e3(args.url,{'short_short':(p_a,p_bs),
                                       'long_long':(p_bl,kp.build_prompt(args.url, 'AGENT_C_LONG',args.b_long_tokens)[0]),
                                       'short_long':(p_a,p_bl)},
                             args.n_predict,args.timeout,args.interval,args.out_dir))
    results={
        'date_utc':datetime.now(timezone.utc).isoformat(),
        'description':'Phase 1 exploratory characterization; no scheduler added',
        'params':{**vars(args),'out_dir':str(args.out_dir)},
        'measured_prompt_tokens':{'A':ca,'B_short':cs,'B_long':cl},
        'verified_local_server': startup,
        'slot_context_limits':[s.get('n_ctx') for s in slots],
        'records':records,
        'limitations':['nvidia-smi allocations do not measure live KV cells',
                       'cache_n/prompt_n are response metrics, not direct GPU KV occupancy',
                       'fixed sleep simulates tool waiting; it is not a real external tool',
                       'cache behavior may depend on llama.cpp version and KV budget',
                       'results depend on server startup flags; record pgrep -af llama-server']
    }
    save(results,args.out_dir)
    if not all((r.get('second') or r.get('step2') or {}).get('ok',True) for r in records):
        print('NOTE: some steps failed; see summary.json for errors')
    return 0


if __name__=='__main__':
    sys.exit(main())
