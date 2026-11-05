#!/usr/bin/env python3
"""Regrade saved v4/v5 results offline, without another llama.cpp run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_agent_correctness_5 import CASES, grade_case


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Existing results JSON")
    p.add_argument("--out", required=True, help="Destination for regraded JSON")
    args = p.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    labels = {case["id"]: case for case in CASES}
    rows = data["cases"]
    for row in rows:
        case = labels.get(row["case_id"])
        if case is None:
            p.error(f"Unknown case id: {row['case_id']}")
        row["grade"] = grade_case(case, row.get("result"), row.get("status", "failed"))
    summary = {
        "attempted": len(rows),
        "agent_executed": sum(row["status"] == "completed" for row in rows),
        "implementation_localization_correct": sum(
            row["grade"]["implementation_correct"] for row in rows),
        "exact_localization_correct": sum(row["grade"]["correct"] for row in rows),
        "exact_localization_and_test_passed": sum(
            row["grade"]["correct"] and row["grade"].get("focused_pytest_passed")
            for row in rows),
        "source_correct": sum(row["grade"]["source_correct"] for row in rows),
        "symbol_correct": sum(row["grade"]["symbol_correct"] for row in rows),
        "inspected_symbol_correct": sum(row["grade"]["inspected_symbol_correct"] for row in rows),
        "test_correct": sum(row["grade"]["test_correct"] for row in rows),
        "focused_pytest_passed": sum(row["grade"]["focused_pytest_passed"] for row in rows),
        "format_correct": sum(row["grade"]["format_correct"] for row in rows),
    }
    data["summary"] = summary
    data["regraded_from_evaluation"] = data.get("evaluation")
    data["evaluation"] = "five_case_code_investigation_offline_regrade_v5"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Regraded without running any model requests or pytest:")
    print(json.dumps(summary, indent=2))
    print(f"Saved results: {out}")


if __name__ == "__main__":
    main()
