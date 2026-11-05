"""Focused offline checks for v7 workload heterogeneity and navigation reliability."""
import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "coding_v7", Path(__file__).with_name("coding_agent_correctness_5.py"))
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)


class V7WorkloadTests(unittest.TestCase):
    def test_30_profiled_cases_and_no_label_changes(self):
        base, _ = v7.load_cases_file(Path(__file__).with_name("cases_30.json"))
        new, _ = v7.load_cases_file(Path(__file__).with_name("cases_30_diverse_v7.json"))
        self.assertEqual(len(new), 30)
        base_by_id = {c["id"]: c for c in base}
        hist = {k: 0 for k in v7.PROFILE_INSTRUCTIONS}
        for item in new:
            hist[item["investigation_profile"]] += 1
            for key in ("question", "expected_source", "expected_symbol", "expected_test"):
                self.assertEqual(item[key], base_by_id[item["id"]][key])
        self.assertEqual(hist, {"basic": 10, "test_evidence": 7,
                                "call_trace": 7, "deep": 6})

    def test_full_symbol_allowlist_in_large_module(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d); path = repo / "src/click/core.py"; path.parent.mkdir(parents=True)
            path.write_text("\n".join(f"def helper_{i}(): pass" for i in range(90)) +
                            "\n\ndef important(): pass\n")
            runner = v7.CodingPilot(repo, "http://localhost/v1", "python", 1, 10)
            runner.question = "Investigate important"
            index = runner.source_symbol_index(path, max_entries=6)
            names = [x["symbol"] for x in index["symbols"]] + index["other_allowed_symbols"]
            self.assertEqual(len(names), 91)
            self.assertIn("important", names)
            self.assertEqual(len(index["symbols"]), 6)

    def test_question_fallback_no_gold_label(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d)
            path = repo / "src/click/types.py"; path.parent.mkdir(parents=True)
            path.write_text('class IntRange:\n    def _clamp(self, value):\n        return value\n')
            wrong = repo / "src/click/utils.py"; wrong.write_text('def open_file(): pass')
            runner = v7.CodingPilot(repo, "http://localhost/v1", "python", 1, 10)
            runner.question = "Investigate integer range clamping behavior"
            self.assertEqual(runner.grep_files('src/click', 'clickintrange'), [])
            ranked=runner.rank_files_by_question('src/click')
            self.assertEqual(ranked[0]['path'],'src/click/types.py')
            self.assertEqual(ranked[0]['discovery'],'question_lexical_fallback')

    def test_helper_trace_reads_actual_imported_function(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d); folder = repo / "src/click"; folder.mkdir(parents=True)
            (folder / "main.py").write_text('''from .utils import echo
class Engine:
    def execute(self):
        self.prepare()
        echo("hi")
    def prepare(self):
        return True
''')
            (folder / "utils.py").write_text('def echo(x):\n    return x\n')
            runner = v7.CodingPilot(repo, "http://localhost/v1", "python", 1, 10)
            runner.question = "Investigate how Engine.execute prepares and outputs"
            candidates = runner.find_related_symbols(folder / "main.py", "Engine.execute")
            names={(x['path'],x['symbol']) for x in candidates}
            self.assertIn(('src/click/main.py','Engine.prepare'),names)
            self.assertIn(('src/click/utils.py','echo'),names)
            runner.investigation_profile = 'call_trace'
            async def fake_llm(sid, deps, messages, max_tokens=240):
                if sid == 'A2': return '{"path":"src/click/main.py"}'
                if sid == 'A3S': return '{"symbol":"Engine.execute"}'
                if sid == 'A3H': return '{"path":"src/click/utils.py","symbol":"echo"}'
                if sid == 'A4':
                    self.assertIn('return x', messages[1]['content'])
                    return 'Engine.execute calls echo'
                raise AssertionError(sid)
            runner.llm=fake_llm
            source=asyncio.run(runner.source_branch('execute'))
            self.assertEqual(source['related_helper']['symbol'],'echo')
            self.assertTrue(source['related_helper_available'])

    def test_source_test_evidence_uses_relevant_late_function(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d); path = repo/'tests/test_flow.py'; path.parent.mkdir(parents=True)
            path.write_text('def test_unrelated(): pass\n'+('\n'*1000)+
                            'def test_forwarding_command_args():\n'
                            '    assert 7 == 7\n')
            pilot=v7.CodingPilot(repo, 'http://localhost/v1','python',1,10)
            pilot.question='Investigate forwarding command args'
            evidence=pilot.read_test_evidence(path, limit=1)
            self.assertIn('test_forwarding_command_args', evidence)
            self.assertIn('assert 7 == 7', evidence)

    def test_test_branch_runs_real_evidence_step_and_reasoning(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d); path=repo/'tests/test_echo.py'; path.parent.mkdir(parents=True)
            path.write_text('def test_echo():\n    assert 1 + 1 == 2\n')
            pilot=v7.CodingPilot(repo,'http://localhost/v1','python',2,10, max_followups=1)
            pilot.investigation_profile='test_evidence'; pilot.question='Investigate echo'
            async def fake_llm(sid, deps, messages, max_tokens=240):
                if sid=='B2': return '{"path":"tests/test_echo.py"}'
                if sid=='B3E':
                    self.assertIn('assert 1 + 1 == 2',messages[1]['content'])
                    return 'Test exists but not necessarily relevant'
                if sid=='D0': return '{"action":"finish","reason":"source inspected"}'
                if sid=='B5': return 'Limited test evidence'
                raise AssertionError(sid)
            async def fake_pytest(sid,deps,test_path):
                return {'exit_code':0,'output':'1 passed','timed_out':False}
            pilot.llm=fake_llm; pilot.pytest=fake_pytest
            result=asyncio.run(pilot.test_branch('echo'))
            self.assertTrue(result['inspected_test_file'])
            self.assertEqual(result['test_evidence_analysis'],'Test exists but not necessarily relevant')
            self.assertTrue(any(ev['id']=='B3I' for ev in pilot.events))

    def test_grader_evidence_steps_are_separate_from_correctness(self):
        case={**v7.CASES[0],'investigation_profile':'deep'}
        run={"source":{"source_path":case['expected_source'],
                       "source_symbol":case['expected_symbol'],
                       "related_helper_available":True,
                       "related_helper":None},
             "tests":{"test_path":case['expected_test'],"targeted":{"exit_code":0},
                      "inspected_test_file":True},
             "final_answer":json.dumps({'source_file':case['expected_source'],
                'symbol':case['expected_symbol'],'test_file':case['expected_test'],
                'explanation':'observed result'})}
        grade=v7.grade_case(case,run,'completed')
        self.assertTrue(grade['implementation_correct'])
        self.assertFalse(grade['required_evidence_steps_complete'])
        self.assertTrue(grade['test_source_inspected'])

if __name__ == '__main__': unittest.main()
