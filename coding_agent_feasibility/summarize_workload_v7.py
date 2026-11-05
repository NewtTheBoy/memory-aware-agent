#!/usr/bin/env python3
"""Summarize real v6/v7 traces by workload profile; no inference server needed.

Reported prompt tokens are SUMS OF REQUESTS, not simultaneous KV cache use.
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def summarize_case(case):
    events = case.get('events', [])
    llm = [e for e in events if e.get('kind') == 'llm' and 'error' not in e]
    usage = [e.get('usage', {}) for e in llm]
    grade = case.get('grade') or {}
    return {
        'case_id': case['case_id'],
        'profile': case.get('investigation_profile', 'basic'),
        'status': case.get('status'),
        'duration': float(case.get('duration_s', 0)),
        'llm_calls': len(llm),
        'tool_calls': sum(e.get('kind') not in ('llm',) for e in events),
        'prompt_sum': sum(u.get('prompt_tokens', 0) or 0 for u in usage),
        'prompt_max': max((u.get('prompt_tokens', 0) or 0 for u in usage), default=0),
        'completion_sum': sum(u.get('completion_tokens', 0) or 0 for u in usage),
        'implementation_correct': bool(grade.get('implementation_correct')),
        'exact_correct': bool(grade.get('correct')),
        'evidence_complete': grade.get('required_evidence_steps_complete'),
        'source_steps': [e['id'] for e in events if e.get('id','').startswith('A3')],
        'test_evidence_steps': any(e.get('id') == 'B3E' for e in events),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text(encoding='utf-8'))
    rows = [summarize_case(c) for c in payload['cases']]
    by_profile = defaultdict(list)
    for row in rows: by_profile[row['profile']].append(row)
    print('PROFILE          N  LLM calls range    Task latency med/range (s)   Max prompt med/range     Impl correct')
    for name in ('basic','test_evidence','call_trace','deep'):
        subset = by_profile.get(name, [])
        if not subset: continue
        calls = [c['llm_calls'] for c in subset]
        durations = [c['duration'] for c in subset]
        prompts = [c['prompt_max'] for c in subset]
        print(f"{name:16}{len(subset):3d}  {min(calls):2d}-{max(calls):<2d}             "
              f"{statistics.median(durations):5.2f} [{min(durations):.2f},{max(durations):.2f}]      "
              f"{statistics.median(prompts):6.0f} [{min(prompts)},{max(prompts)}]       "
              f"{sum(c['implementation_correct'] for c in subset)}/{len(subset)}")
    print('\nCases (id, profile, LLM, tools, latency_s, prompt_max, impl_correct, evidence_complete):')
    for c in rows:
        print(f"{c['case_id']:24} {c['profile']:14} {c['llm_calls']:2d} "
              f"{c['tool_calls']:2d} {c['duration']:6.2f} {c['prompt_max']:5d} "
              f"{int(c['implementation_correct'])} {c['evidence_complete']}")
    print('\nNOTE: prompt_sum is cumulative across LLM calls, not peak simultaneous KV occupancy.')
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')

if __name__=='__main__': main()
