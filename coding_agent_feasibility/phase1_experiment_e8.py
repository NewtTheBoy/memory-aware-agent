#!/usr/bin/env python3
"""E8: Does clearing a llama.cpp slot reclaim physical GPU VRAM?

Runs independently launched llama-server instances across context sizes. Parses
startup KV allocation log, samples whole-device GPU VRAM in steady states,
compares KEEP / ERASE at fixed server configuration, and validates prompt reuse.

This is a *measurement* harness, NOT online KV-buffer resizing / offloading.
On WSL2, per-process VRAM is often unavailable via nvidia-smi; whole-device
changes cannot be conclusively attributed to one process without isolation.

Stdlib only. Requires llama-server binary, nvidia-smi, and locally working GPU.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

GPU_LOG_RE = re.compile(
    r'(?P<device>\S+)\s+KV\s+buffer\s+size\s*=\s*'
    r'(?P<size>[\d.]+)\s*(?P<unit>MiB|GiB|KiB)', re.IGNORECASE
)
KV_GENERIC_RE = re.compile(r'KV\s+buffer\s+size\s*=\s*[\d.]+\s*(?:MiB|GiB|KiB)', re.IGNORECASE)
MEM_RE = re.compile(r'^\s*([\d.]+)\s*(?:MiB)?\s*$', re.IGNORECASE)


def parse_kv_log(log: str) -> dict:
    """Return individual GPU KV allocation lines and MiB sum (only GPU entries)."""
    records = []
    for line in log.splitlines():
        if not KV_GENERIC_RE.search(line):
            continue
        match = GPU_LOG_RE.search(line)
        if match:
            unit = match.group('unit').lower()
            factor = {'kib': 1/1024, 'mib': 1.0, 'gib': 1024.0}[unit]
            device = match.group('device')
            size = round(float(match.group('size')) * factor, 4)
            records.append({'device': device, 'size_mib': size, 'line': line.strip(),
                            'gpu': bool(re.search(r'CUDA|VULKAN|ROCM|HIP|METAL', device, re.I))})
        else:
            records.append({'device': 'unparsed', 'size_mib': None,
                            'line': line.strip(), 'gpu': False})
    return {'records': records,
            'gpu_kv_mib_sum': round(sum(r['size_mib'] for r in records
                                        if r['gpu'] and r['size_mib'] is not None), 4)
            if any(r['gpu'] and r['size_mib'] is not None for r in records) else None,
            'gpu_kv_lines': [r['line'] for r in records if r['gpu']]}


def parse_nvidia_gpu(output: str) -> dict:
    """Parse `nvidia-smi --query-gpu=memory.used,... --format=csv,noheader,nounits`."""
    rows = [r for r in csv.reader(output.splitlines()) if r]
    if not rows or len(rows[0]) < 3:
        raise ValueError('nvidia-smi returned no usable GPU memory row')
    fields = []
    for raw in rows[0][:3]:
        m = MEM_RE.fullmatch(raw.strip())
        if not m:
            raise ValueError(f'invalid nvidia-smi field {raw!r}')
        fields.append(float(m.group(1)))
    return {'used_mib': fields[0], 'free_mib': fields[1], 'total_mib': fields[2]}


def gpu_read(device: int) -> dict:
    res = subprocess.check_output([
        'nvidia-smi', '-i', str(device),
        '--query-gpu=memory.used,memory.free,memory.total',
        '--format=csv,noheader,nounits'], text=True, timeout=10,
        stderr=subprocess.DEVNULL)
    return parse_nvidia_gpu(res)


def median(values):
    vals = [x for x in values if x is not None and math.isfinite(x)]
    return float(statistics.median(vals)) if vals else None


def delta(a, b):
    return round(a - b, 3) if a is not None and b is not None else None


def snapshot(label: str, budget: int | str, mode: str, rep: int,
             gpu: int, samples: int, gap: float, rows: list[dict]) -> dict:
    obs = []
    for i in range(samples):
        try:
            data = gpu_read(gpu)
            obs.append(data)
            rows.append({'budget': budget, 'mode': mode, 'rep': rep,
                         'phase': label, 'sample': i, 'utc': datetime.now(timezone.utc).isoformat(),
                         **data})
        except Exception as exc:
            raise RuntimeError(f'nvidia-smi read failed at {label}: {exc}') from exc
        if i < samples - 1:
            time.sleep(gap)
    used = [v['used_mib'] for v in obs]
    free = [v['free_mib'] for v in obs]
    point = {'used_mib': median(used), 'free_mib': median(free),
             'used_range_mib': round(max(used)-min(used),3)}
    print(f'      {label}: GPU used={point["used_mib"]:.1f} MiB '
          f'(within-phase range={point["used_range_mib"]:.1f} MiB)', flush=True)
    return point


def api(base, method, path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(base.rstrip('/') + path, data=data, method=method,
                          headers={'Content-Type': 'application/json'})
    try:
        with request.urlopen(req, timeout=timeout) as res:
            body = res.read()
        return json.loads(body) if body else None
    except error.HTTPError as exc:
        detail = exc.read(450).decode(errors='replace')
        raise RuntimeError(f'HTTP {exc.code} {path}: {detail}') from exc


def prompt_token_count(base, prompt):
    response = api(base, 'POST', '/tokenize', {'content': prompt, 'add_special': False}, timeout=30)
    if not isinstance(response, dict) or not isinstance(response.get('tokens'), list):
        raise RuntimeError('Unexpected /tokenize response')
    return len(response['tokens'])


def calibrated_prompt(base: str, tokens: int):
    header = 'You are a coding agent examining an issue in a Python repository.\n'
    lines = [f'{i:04d}: inspect function handle_{i%31}, inspect call path {i%73}, test edge case {i%47}.\n'
             for i in range(1, 2200)]
    footer = '\nExplain the likely bug in one sentence.\n'
    def make(n): return header + ''.join(lines[:n]) + footer
    lo, hi = 0, len(lines)
    while hi-lo>1:
        m=(lo+hi)//2
        if prompt_token_count(base, make(m)) < tokens: lo=m
        else: hi=m
    candidates=[(prompt_token_count(base, make(n)),make(n)) for n in (lo,hi)]
    return min(candidates, key=lambda x: abs(x[0]-tokens))


def completion(base, prompt, slot, predict, timeout):
    t0=time.monotonic()
    try:
        res=api(base,'POST','/completion',{
            'prompt':prompt,'n_predict':predict,'temperature':0,
            'cache_prompt':True,'stream':False,'id_slot':slot},timeout=timeout)
        if not isinstance(res,dict) or 'timings' not in res:
            raise ValueError(f'Unexpected completion response: {str(res)[:200]}')
        times=res['timings'] or {}
        return {'ok':True,'request_s':round(time.monotonic()-t0,4),
                'cache_n':times.get('cache_n'),'prompt_n':times.get('prompt_n'),
                'prompt_ms':times.get('prompt_ms'),'predicted_n':times.get('predicted_n'),
                'slot_id':res.get('id_slot')}
    except Exception as exc:
        return {'ok':False,'request_s':round(time.monotonic()-t0,4), 'error':str(exc)}


def erase(base, slot):
    response=api(base,'POST',f'/slots/{slot}?action=erase',{},timeout=30)
    if not isinstance(response,dict) or response.get('id_slot') not in (None,slot):
        raise RuntimeError(f'Unexpected erase response: {response}')
    return response


def server_is_busy(host, port):
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host,port))==0


def start_server(binary:Path, model:str, budget:int, args, logs_dir:Path, slot_dir:Path):
    logfile=logs_dir/f'llama_c{budget}.log'
    slot_dir.mkdir(parents=True,exist_ok=True)
    cmd=[str(binary.resolve()),'-hf',model,'-ngl',str(args.ngl),
         '-c',str(budget),'-np','2','--kv-unified',
         '--kv-unified-per-slot',str(args.per_slot),
         '--cache-ram','0','--slot-save-path',str(slot_dir.resolve()),
         '--metrics','-cb','--host','127.0.0.1','--port',str(args.port)]
    print('  Starting:', ' '.join(cmd), flush=True)
    loghandle=logfile.open('w',encoding='utf-8')
    try:
        proc=subprocess.Popen(cmd,stdout=loghandle,stderr=subprocess.STDOUT,
                              cwd=str(args.project_root.resolve()))
    finally:
        loghandle.close()
    url=f'http://127.0.0.1:{args.port}'
    deadline=time.monotonic()+args.startup_timeout
    try:
        while time.monotonic()<deadline:
            if proc.poll() is not None:
                raise RuntimeError(f'llama-server exited {proc.returncode}; inspect {logfile}')
            try:
                health=api(url,'GET','/health',timeout=2)
                if isinstance(health,dict) and health.get('status')=='ok':
                    return proc,url,logfile,cmd
            except Exception:
                pass
            time.sleep(0.75)
        raise TimeoutError(f'Server did not become healthy within {args.startup_timeout}s; see {logfile}')
    except Exception:
        stop_server(proc)
        raise


def stop_server(proc):
    if proc.poll() is None:
        proc.terminate()
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait(timeout=10)


def validate_server(base,args,budget):
    slots=api(base,'GET','/slots',timeout=10)
    if not isinstance(slots,list) or len(slots)!=2:
        raise RuntimeError(f'Expected exactly two slots; got {slots}')
    context=[s.get('n_ctx') for s in slots]
    if any(x is not None and x>args.per_slot for x in context):
        raise RuntimeError(f'Unexpected slot context limits {context}; expected <= {args.per_slot}')
    # Erase both slots up front and fail early on endpoint mismatch.
    erase(base,0);erase(base,1)
    return {'n_slots':len(slots),'slot_context_limits':context,'requested_c':budget}


def trial(base, mode, rep, budget, prompt, args, samples):
    erase(base,0);erase(base,1)
    initial=snapshot('initial_idle',budget,mode,rep,args.gpu,args.samples,args.sample_gap,samples)
    a1=completion(base,prompt,0,args.predict,args.request_timeout)
    if not a1['ok']:
        return {'budget':budget,'mode':mode,'rep':rep,'ok':False,'stage':'A1','error':a1.get('error')}
    active=snapshot('after_a1',budget,mode,rep,args.gpu,args.samples,args.sample_gap,samples)
    erase_response=None
    if mode=='erase':
        erase_response=erase(base,0)
    time.sleep(args.wait)
    idle=snapshot('before_a2',budget,mode,rep,args.gpu,args.samples,args.sample_gap,samples)
    a2=completion(base,prompt+'\nTool output: pytest finished. Continue investigating.\n',
                  0,args.predict,args.request_timeout)
    after=snapshot('after_a2',budget,mode,rep,args.gpu,args.samples,args.sample_gap,samples)
    record={'budget':budget,'mode':mode,'rep':rep,'ok':bool(a2['ok']),
            'a1':a1,'a2':a2,'erase_response':erase_response,
            'snapshots':{'initial_idle':initial,'after_a1':active,
                         'before_a2':idle,'after_a2':after},
            'gpu_used_delta_wait_mib':delta(idle['used_mib'],active['used_mib']),
            'gpu_used_delta_from_initial_mib':delta(idle['used_mib'],initial['used_mib'])}
    print(f'    {mode} rep={rep}: A2 ok={a2["ok"]}, '
          f'cache_n={a2.get("cache_n")}, prefill_ms={a2.get("prompt_ms")}, '
          f'VRAM(wait minus A1)={record["gpu_used_delta_wait_mib"]} MiB', flush=True)
    return record


def modes_for_repeat(rep):
    # ABBA pattern avoids systematic mode-order bias across repeats.
    return ['keep','erase'] if (rep%2)==1 else ['erase','keep']


def export(root:Path, records:list[dict], samples:list[dict], servers:list[dict], config:dict):
    root.mkdir(parents=True,exist_ok=True)
    def csv_dump(name,rows,cols):
        with (root/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
            writer.writeheader();writer.writerows(rows)
    flattened=[]
    for r in records:
        q={'budget':r['budget'],'mode':r['mode'],'rep':r['rep'],'ok':r['ok'],
           'stage':r.get('stage'),'error':r.get('error'),'gpu_used_delta_wait_mib':r.get('gpu_used_delta_wait_mib'),
           'gpu_used_delta_from_initial_mib':r.get('gpu_used_delta_from_initial_mib')}
        for point in ['initial_idle','after_a1','before_a2','after_a2']:
            q[f'{point}_used_mib']=(r.get('snapshots',{}).get(point) or {}).get('used_mib')
        for metric in ['cache_n','prompt_n','prompt_ms','request_s']:
            q[f'a2_{metric}']=(r.get('a2') or {}).get(metric)
        q['a1_request_s']=(r.get('a1') or {}).get('request_s')
        flattened.append(q)
    trial_cols=['budget','mode','rep','ok','stage','error','initial_idle_used_mib','after_a1_used_mib',
                'before_a2_used_mib','after_a2_used_mib','gpu_used_delta_wait_mib',
                'gpu_used_delta_from_initial_mib','a2_cache_n','a2_prompt_n','a2_prompt_ms',
                'a2_request_s','a1_request_s']
    csv_dump('trials.csv',flattened,trial_cols)
    csv_dump('gpu_samples.csv',samples,['budget','mode','rep','phase','sample','utc',
                                        'used_mib','free_mib','total_mib'])
    csv_dump('server_summary.csv',servers,['budget','kv_gpu_mib_log','n_slots','per_slot',
                                          'pre_server_used_mib','server_idle_used_mib','post_server_used_mib',
                                          'startup_delta_mib','shutdown_drop_mib','log_path'])
    summaries=[]
    for budget in sorted({r['budget'] for r in records}):
        a={mode:[r for r in records if r['budget']==budget and r['mode']==mode and r['ok']]
           for mode in ('keep','erase')}
        keep=a['keep']; evict=a['erase']
        def med(rs,field):return median([r.get(field) for r in rs])
        keep_prefill=median([(r.get('a2') or {}).get('prompt_ms') for r in keep])
        erase_prefill=median([(r.get('a2') or {}).get('prompt_ms') for r in evict])
        summaries.append({'budget':budget,'keep_ok':len(keep),'erase_ok':len(evict),
            'keep_cache_n_median':median([(r.get('a2') or {}).get('cache_n') for r in keep]),
            'erase_cache_n_median':median([(r.get('a2') or {}).get('cache_n') for r in evict]),
            'keep_prefill_median_ms':keep_prefill,'erase_prefill_median_ms':erase_prefill,
            'keep_vram_wait_delta_median_mib':med(keep,'gpu_used_delta_wait_mib'),
            'erase_vram_wait_delta_median_mib':med(evict,'gpu_used_delta_wait_mib'),
            'erase_minus_keep_vram_delta_mib':delta(med(evict,'gpu_used_delta_wait_mib'),
                                                     med(keep,'gpu_used_delta_wait_mib'))})
    csv_dump('comparison.csv',summaries,['budget','keep_ok','erase_ok',
            'keep_cache_n_median','erase_cache_n_median','keep_prefill_median_ms',
            'erase_prefill_median_ms','keep_vram_wait_delta_median_mib',
            'erase_vram_wait_delta_median_mib','erase_minus_keep_vram_delta_mib'])
    payload={'created_utc':datetime.now(timezone.utc).isoformat(),'settings':config,
             'servers':servers,'trials':records,'comparison':summaries,
             'interpretation_warning':
             'Whole-GPU memory under WSL2 is noisy. A slot erase invalidates KV tokens but does not by itself demonstrate freeing a CUDA allocation. Startup KV buffer sizes are logged allocation evidence, not proof of live resizing.'}
    (root/'results.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    return summaries


def get_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--project-root',type=Path,default=Path.cwd())
    p.add_argument('--server-binary',type=Path,default=Path('llama.cpp/build/bin/llama-server'))
    p.add_argument('--model-repo',default='Qwen/Qwen2.5-3B-Instruct-GGUF:Q4_K_M')
    p.add_argument('--budgets',type=int,nargs='+',default=[4096,8192,16384])
    p.add_argument('--per-slot',type=int,default=4096)
    p.add_argument('--ngl',type=int,default=99)
    p.add_argument('--port',type=int,default=8080)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--tokens',type=int,default=2200)
    p.add_argument('--predict',type=int,default=64)
    p.add_argument('--wait',type=float,default=2.0)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--samples',type=int,default=5)
    p.add_argument('--sample-gap',type=float,default=0.18)
    p.add_argument('--startup-timeout',type=float,default=180)
    p.add_argument('--request-timeout',type=float,default=180)
    p.add_argument('--settle-after-stop',type=float,default=1.5)
    p.add_argument('--out-dir',type=Path,default=Path('results/phase1_e8_physical_vram'))
    return p


def main(argv=None):
    args=get_parser().parse_args(argv)
    binary=args.server_binary
    if not binary.is_absolute():binary=args.project_root/binary
    if not binary.is_file():
        raise SystemExit(f'Cannot find server binary: {binary}. Set --server-binary.')
    if not 0<args.tokens<args.per_slot-args.predict-100:
        raise SystemExit('--tokens + --predict must fit per-slot context with safety margin')
    if args.repeats<1 or args.samples<1 or args.sample_gap<0 or args.wait<0:
        raise SystemExit('repeats/samples must be positive; sample-gap/wait nonnegative')
    if len(set(args.budgets))!=len(args.budgets) or any(x<args.per_slot for x in args.budgets):
        raise SystemExit('Require unique budgets, all >= --per-slot')
    if server_is_busy('127.0.0.1',args.port):
        raise SystemExit(f'Port {args.port} is occupied. STOP existing server yourself; '
                         'this script will not kill other processes.')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise SystemExit(f'Output dir {args.out_dir} is not empty; choose a fresh --out-dir.')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    try:
        _=gpu_read(args.gpu)
    except Exception as exc:
        raise SystemExit(f'nvidia-smi GPU memory read failed: {exc}')
    results=[]; samples=[]; servers=[]
    logs=args.out_dir/'server_logs';logs.mkdir(exist_ok=True)
    print('E8: PHYSICAL VRAM FEASIBILITY. Only this harness will launch/stop its servers.',flush=True)
    print('Never interpret slot erase as guaranteed GPU VRAM release.',flush=True)
    for budget in args.budgets:
        if server_is_busy('127.0.0.1',args.port):
            raise RuntimeError('Unexpected other process occupies port before startup')
        print(f'\n===== KV Context budget -c {budget} =====',flush=True)
        pre=snapshot('before_server',budget,'server',0,args.gpu,args.samples,args.sample_gap,samples)
        proc=None
        try:
            proc,base,log,cmd=start_server(binary,args.model_repo,budget,args,logs,
                        args.out_dir/f'slots_c{budget}')
            cfg=validate_server(base,args,budget)
            # Startup preallocations normally appear in log before /health reports OK.
            log_parsed=parse_kv_log(log.read_text(encoding='utf-8',errors='replace'))
            print('  Startup GPU KV buffer size:',log_parsed['gpu_kv_mib_sum'],'MiB',flush=True)
            if not log_parsed['gpu_kv_lines']:
                print('  WARNING: did not find CUDA KV buffer size; inspect server log manually.',flush=True)
            for line in log_parsed['gpu_kv_lines']:
                print('    '+line[:240],flush=True)
            # Stabilize startup allocations and warmup before idle measurement.
            time.sleep(1)
            idle=snapshot('server_idle',budget,'server',0,args.gpu,args.samples,args.sample_gap,samples)
            n,prompt=calibrated_prompt(base,args.tokens)
            print(f'  Actual prompt tokens: {n}; slot context limits: {cfg["slot_context_limits"]}',flush=True)
            if n+args.predict+100>args.per_slot:
                raise RuntimeError('Calibrated prompt too large for per-slot context')
            for rep in range(1,args.repeats+1):
                for mode in modes_for_repeat(rep):
                    r=trial(base,mode,rep,budget,prompt,args,samples)
                    results.append(r)
                    if not r['ok']:
                        print(f'  ERROR: trial failed: {r.get("error") or (r.get("a2") or {}).get("error")}',flush=True)
            servers.append({'budget':budget,'kv_gpu_mib_log':log_parsed['gpu_kv_mib_sum'],
                            'n_slots':cfg['n_slots'],'per_slot':args.per_slot,
                            'pre_server_used_mib':pre['used_mib'],
                            'server_idle_used_mib':idle['used_mib'],
                            'startup_delta_mib':delta(idle['used_mib'],pre['used_mib']),
                            'log_path':str(log),'cmd':cmd,
                            'kv_log_records':log_parsed['records'],'tokenized_prompt_count':n})
        finally:
            if proc is not None:
                print('  Stopping only the server launched by this harness...',flush=True)
                stop_server(proc)
        time.sleep(args.settle_after_stop)
        post=snapshot('after_server',budget,'server',0,args.gpu,args.samples,args.sample_gap,samples)
        if servers and servers[-1]['budget']==budget:
            servers[-1]['post_server_used_mib']=post['used_mib']
            servers[-1]['shutdown_drop_mib']=delta(servers[-1]['server_idle_used_mib'],post['used_mib'])
        export(args.out_dir,results,samples,servers,{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    summaries=export(args.out_dir,results,samples,servers,
              {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    print('\n===== E8 completed =====',flush=True)
    print('Startup KV log sizes:',[(x['budget'],x['kv_gpu_mib_log']) for x in servers],flush=True)
    for row in summaries:
        print(f'  c={row["budget"]}: cache keep={row["keep_cache_n_median"]}, '
              f'erase={row["erase_cache_n_median"]}, '
              f'prefill ms keep={row["keep_prefill_median_ms"]}, '
              f'erase={row["erase_prefill_median_ms"]}, '
              f'wait VRAM Δ keep={row["keep_vram_wait_delta_median_mib"]}, '
              f'erase={row["erase_vram_wait_delta_median_mib"]}',flush=True)
    print('Saved:',args.out_dir/'server_summary.csv',args.out_dir/'comparison.csv',
          args.out_dir/'trials.csv',args.out_dir/'gpu_samples.csv',flush=True)
    return 0

if __name__=='__main__':
    sys.exit(main())
