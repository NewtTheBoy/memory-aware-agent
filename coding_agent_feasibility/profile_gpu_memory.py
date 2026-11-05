#!/usr/bin/env python3
"""Run an existing workload while sampling whole-GPU memory/utilization.

This is a non-invasive *device-level* probe. It cannot measure per-request KV
cache residency. It never changes the llama.cpp server or the coding agent.

Usage:
 python coding_agent_feasibility/profile_gpu_memory.py \
     --out results/coding_memory_v5.csv --interval-ms 250 -- \
     python coding_agent_feasibility/coding_agent_correctness_5.py \
     --max-inflight-llm 2 --max-followups 1 --out results/coding_v5.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

QUERY_FIELDS = [
    "index", "name", "memory.total", "memory.used", "memory.free",
    "utilization.gpu", "utilization.memory",
]
OUTPUT_FIELDS = [
    "wall_epoch_s", "utc_iso", "elapsed_s", "gpu_index", "gpu_name",
    "total_mib", "used_mib", "free_mib", "gpu_util_percent",
    "memory_util_percent", "sample_error",
]


def as_number(s: str) -> float | None:
    try:
        val = float(s.strip())
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def sample(index: int, timeout: float = 5.0) -> dict:
    command = ["nvidia-smi", "-i", str(index),
               "--query-gpu=" + ",".join(QUERY_FIELDS),
               "--format=csv,noheader,nounits"]
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if process.returncode != 0:
        raise RuntimeError((process.stderr or process.stdout).strip()[:300])
    rows = list(csv.reader(process.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(QUERY_FIELDS):
        raise RuntimeError(f"Expected 1 GPU with {len(QUERY_FIELDS)} columns, got {rows!r}")
    parts = [part.strip() for part in rows[0]]
    return {
        "gpu_index": parts[0], "gpu_name": parts[1],
        "total_mib": as_number(parts[2]), "used_mib": as_number(parts[3]),
        "free_mib": as_number(parts[4]),
        "gpu_util_percent": as_number(parts[5]),
        "memory_util_percent": as_number(parts[6]),
        "sample_error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="CSV output path")
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command to run, preceded by --")
    args = parser.parse_args()
    if args.interval_ms < 100:
        parser.error("--interval-ms must be >= 100 (GPU polls have overhead)")
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        parser.error("Missing workload command after --")
    if shutil.which("nvidia-smi") is None:
        parser.error("nvidia-smi not found. Run this probe on the GPU machine.")
    try:
        initial = sample(args.gpu_index)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        parser.error(f"nvidia-smi preflight failed: {exc}")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(".meta.json")
    begin_monotonic = time.monotonic()
    begin_wall = time.time()
    used_values: list[float] = []
    errors = 0
    rows_written = 0
    child: subprocess.Popen | None = None
    exit_code = 1
    print(f"Memory sampling -> {path} every {args.interval_ms} ms (GPU {args.gpu_index})", flush=True)
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            def emit(gpu: dict) -> None:
                nonlocal rows_written, errors
                wall = time.time()
                row = {
                    "wall_epoch_s": round(wall, 6),
                    "utc_iso": datetime.fromtimestamp(wall, tz=timezone.utc).isoformat(),
                    "elapsed_s": round(time.monotonic() - begin_monotonic, 6),
                    **gpu,
                }
                writer.writerow({k: row.get(k, "") for k in OUTPUT_FIELDS})
                handle.flush()
                rows_written += 1
                if row.get("used_mib") is not None:
                    used_values.append(float(row["used_mib"]))
                if row.get("sample_error"):
                    errors += 1

            emit(initial)  # one GPU-memory baseline sample before child launch
            child = subprocess.Popen(cmd)
            while True:
                started = time.monotonic()
                try:
                    emit(sample(args.gpu_index))
                except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
                    emit({"gpu_index": args.gpu_index, "sample_error": repr(exc)[:300]})
                status = child.poll()
                if status is not None:
                    exit_code = status
                    break
                delay = args.interval_ms / 1000.0 - (time.monotonic() - started)
                time.sleep(max(0.0, delay))
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        exit_code = 130
    finally:
        end_wall = time.time()
        baseline = as_number(str(initial["used_mib"]))
        meta = {
            "command": cmd, "exit_code": exit_code,
            "gpu_index": args.gpu_index, "gpu_name": initial["gpu_name"],
            "sample_interval_ms": args.interval_ms, "samples": rows_written,
            "sampling_errors": errors,
            "started_wall_epoch_s": begin_wall,
            "ended_wall_epoch_s": end_wall,
            "baseline_device_used_mib": baseline,
            "observed_peak_device_used_mib": max(used_values) if used_values else None,
            "device_memory_scope": (
                "whole GPU, including model weights, other processes, and preallocated pools; "
                "not per-request KV residency or allocation"
            ),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Memory sampling complete ({rows_written} samples; {errors} sampling errors; "
              f"child exit={exit_code}). Metadata -> {meta_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
