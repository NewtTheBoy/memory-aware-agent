#!/usr/bin/env python3
"""Compare v8 task-concurrency traces and optional device-wide GPU CSV samples.

Report client-side task/LLM overlaps, latency, throughput, and sampled GPU usage.
Never interpret nvidia-smi memory.used as per-request KV cache allocation.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def _num(text):
    try:
        v = float(text)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def gpu_stats(path: Path, start: float, end: float) -> dict:
    with path.open(encoding='utf-8', newline='') as f:
        samples = list(csv.DictReader(f))
    in_run = [r for r in samples if (t := _num(r.get('wall_epoch_s'))) is not None and start <= t <= end]
    used = [v for row in in_run if (v := _num(row.get('used_mib'))) is not None]
    utils = [v for row in in_run if (v := _num(row.get('gpu_util_percent'))) is not None]
    meta = path.with_suffix('.meta.json')
    metadata = json.loads(meta.read_text()) if meta.exists() else {}
    baseline = _num(metadata.get('baseline_device_used_mib'))
    peak = max(used, default=None)
    return {
        'sample_count_during_tasks': len(in_run),
        'peak_gpu_used_mib': peak,
        'baseline_gpu_used_mib': baseline,
        'peak_minus_baseline_mib': round(peak - baseline, 2) if peak is not None and baseline is not None else None,
        'mean_gpu_util_percent': round(statistics.mean(utils), 2) if utils else None,
        'gpu_csv': str(path),
    }


def to_row(trace: Path, gpu: Path | None) -> dict:
    data = json.loads(trace.read_text(encoding='utf-8'))
    conf = data['config']
    summary = data['summary']
    return {
        'trace': str(trace),
        'max_tasks': conf['max_tasks'],
        'max_llm': conf['max_inflight_llm'],
        'n': summary['attempted'],
        'seconds': summary['makespan_s'],
        'completed_per_s': summary['completed_tasks_per_s'],
        'impl_correct_per_s': summary['implementation_correct_tasks_per_s'],
        'impl_correct': summary['implementation_localization_correct'],
        'latency_p50_s': summary['task_latency_p50_s'],
        'latency_p95_s': summary['task_latency_p95_s'],
        'client_queue_p95_s': summary['llm_client_queue_p95_s'],
        'observed_max_llm': summary['observed_peak_client_inflight_llm'],
        'observed_max_tasks': summary['observed_peak_active_tasks'],
        'slot_busy_peak': (data.get('slot_monitor') or {}).get('observed_peak_busy_slots'),
        **(gpu_stats(gpu, data['run_start_wall_epoch_s'], data['run_end_wall_epoch_s']) if gpu else {}),
        'case_ids': data.get('dataset', {}).get('selected_case_ids'),
    }


def display(row: dict) -> None:
    print(f"c={row['max_tasks']}; global LLM={row['max_llm']}; "
          f"{row['n']} tasks; makespan={row['seconds']:.2f}s; "
          f"completed/s={row['completed_per_s']:.3f}; "
          f"implementation-correct/s={row['impl_correct_per_s']:.3f} "
          f"({row['impl_correct']}/{row['n']}); "
          f"task p50/p95={row['latency_p50_s']:.2f}/{row['latency_p95_s']:.2f}s; "
          f"LLM client queue p95={row['client_queue_p95_s']:.2f}s; "
          f"observed peak client LLM={row['observed_max_llm']}; "
          f"busy server slots peak={row['slot_busy_peak']}")
    if 'peak_gpu_used_mib' in row:
        print(f"   device used: baseline={row['baseline_gpu_used_mib']} MiB, "
              f"peak={row['peak_gpu_used_mib']} MiB, "
              f"delta={row['peak_minus_baseline_mib']} MiB; "
              f"mean GPU util={row['mean_gpu_util_percent']}%; "
              f"samples={row['sample_count_during_tasks']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces', nargs='+', required=True, type=Path)
    parser.add_argument('--gpu-csv', nargs='*', default=[], type=Path,
                        help='Optional GPU CSV, in same order as --traces')
    parser.add_argument('--json-out', type=Path)
    args = parser.parse_args()
    if args.gpu_csv and len(args.gpu_csv) != len(args.traces):
        parser.error('--gpu-csv requires one CSV per trace, same order')
    rows = [to_row(path, args.gpu_csv[i] if args.gpu_csv else None)
            for i, path in enumerate(args.traces)]
    target = rows[0]['case_ids']
    for row in rows[1:]:
        if row['case_ids'] != target:
            print('WARNING: Case IDs or their order differ across runs; throughput comparison may be misleading.')
    llm_caps = {row['max_llm'] for row in rows}
    if len(llm_caps) > 1:
        print('WARNING: max-inflight-llm changed between runs, so both task and request admission changed.')
    for row in rows:
        display(row)
    print('\nCAUTION: GPU memory.used measures whole-device allocations. A flat trace is '
          'consistent with a preallocated llama.cpp KV pool; it is not evidence of flat KV residency.')
    print('LLM client queue is semaphore waiting, not server-side queue or GPU compute. '
          'Passing pytest is not proof that the selected test module covers the target symbol.')
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(f'Saved comparison: {args.json_out}')


if __name__ == '__main__':
    main()
