#!/usr/bin/env python3
"""Join GPU device-memory samples with five-case JSON case/event timestamps."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_samples(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        try:
            if not row["used_mib"]:
                continue
            out.append({"t": float(row["wall_epoch_s"]),
                        "used": float(row["used_mib"]),
                        "util": float(row["gpu_util_percent"]) if row["gpu_util_percent"] else None})
        except (TypeError, KeyError, ValueError):
            continue
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True, help="Five-case results JSON from v5")
    p.add_argument("--samples", required=True, help="CSV from profile_gpu_memory.py")
    args = p.parse_args()
    cases = json.loads(Path(args.trace).read_text(encoding="utf-8"))["cases"]
    samples = read_samples(Path(args.samples))
    if not samples:
        p.error("No usable GPU samples found")
    print("case                    samples   GPU used min/peak MiB   avg GPU util %  LLM tokens (prompt/out)")
    for case in cases:
        start = case.get("case_start_wall_epoch_s")
        end = case.get("case_end_wall_epoch_s")
        if start is None or end is None:
            print(f"{case['case_id']:<24}  no wall timestamps (rerun with v5)")
            continue
        subset = [x for x in samples if start <= x["t"] <= end]
        if not subset:
            print(f"{case['case_id']:<24}  no samples in case interval")
            continue
        used = [x["used"] for x in subset]
        util = [x["util"] for x in subset if x["util"] is not None]
        usage = [e.get("usage") or {} for e in case.get("events", []) if e.get("kind") == "llm"]
        prompt = sum(u.get("prompt_tokens", 0) or 0 for u in usage)
        completion = sum(u.get("completion_tokens", 0) or 0 for u in usage)
        avg_gpu_util = f"{sum(util)/len(util):6.1f}" if util else "   n/a"
        print(f"{case['case_id']:<24}  {len(subset):>4}        "
              f"{min(used):7.0f}/{max(used):7.0f}           "
              f"{avg_gpu_util}           {prompt}/{completion}")
    print("NOTE: GPU memory is whole-device usage. Constant used MiB can reflect llama.cpp KV preallocation;")
    print("      it does NOT prove no per-request KV activity. Compare server logs/metrics next.")


if __name__ == "__main__":
    main()
