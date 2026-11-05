"""Offline tests for JSON case-dataset integration; no llama.cpp needed."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).with_name("coding_agent_correctness_5.py")
spec = importlib.util.spec_from_file_location("coding_agent_runner", MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ExternalCasesTests(unittest.TestCase):
    def test_real_30_case_dataset_loads_with_original_five_unchanged(self):
        cases, commit = runner.load_cases_file(Path(__file__).with_name("cases_30.json"))
        self.assertEqual(len(cases), 30)
        self.assertEqual(len(set(c["id"] for c in cases)), 30)
        self.assertEqual(commit, "2247b35ea1c47c727d7a06e51fa280e12a863ff6")
        by_id = {c["id"]: c for c in cases}
        for original in runner.CASES:
            self.assertEqual(by_id[original["id"]], original)

    def test_invalid_duplicate_id_is_rejected(self):
        case = runner.CASES[0]
        self._expect_error({"cases": [case, case]}, "Duplicate case id")

    def test_path_traversal_is_rejected(self):
        case = dict(runner.CASES[0], expected_test="tests/../secrets.py")
        self._expect_error({"cases": [case]}, "Unsafe expected_test")

    def test_invalid_source_directory_is_rejected(self):
        case = dict(runner.CASES[0], expected_source="other/testing.py")
        self._expect_error({"cases": [case]}, "Unsafe expected_source")

    def test_missing_field_is_rejected(self):
        case = dict(runner.CASES[0]); case.pop("question")
        self._expect_error({"cases": [case]}, "missing 'question'")

    def test_invalid_commit_is_rejected(self):
        self._expect_error({"repo_commit": "main", "cases": [runner.CASES[0]]}, "repo_commit")

    def test_list_only_format_is_supported(self):
        items, commit = self._load([runner.CASES[0]])
        self.assertEqual(items[0], runner.CASES[0]); self.assertIsNone(commit)

    def _load(self, payload):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return runner.load_cases_file(path)

    def _expect_error(self, payload, text):
        with self.assertRaisesRegex(ValueError, text):
            self._load(payload)


if __name__ == "__main__":
    unittest.main()
