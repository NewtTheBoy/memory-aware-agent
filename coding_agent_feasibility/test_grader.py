import importlib.util
import asyncio
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name('coding_agent_correctness_5.py')
spec = importlib.util.spec_from_file_location('feasibility', SCRIPT)
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)

class GraderTests(unittest.TestCase):
    def test_exact_matches(self):
        c = f.CASES[0]
        r = {'source':{'source_path':c['expected_source'], 'source_symbol': c['expected_symbol']},
             'tests':{'test_path':c['expected_test'], 'targeted':{'exit_code':0}},
             'final_answer':'{"source_file":"src/click/testing.py","symbol":"CliRunner.invoke","test_file":"tests/test_testing.py","explanation":"uses runner"}'}
        g=f.grade_case(c,r,'completed')
        self.assertTrue(g['correct']); self.assertTrue(g['focused_pytest_passed'])
        self.assertTrue(g['implementation_correct'])
    def test_alternate_test_not_silently_accepted(self):
        c = next(x for x in f.CASES if x['id'] == 'option_decorator')
        r = {'source': {'source_path': c['expected_source'],
                        'source_symbol': c['expected_symbol']},
             'tests': {'test_path': 'tests/test_command_decorators.py',
                       'targeted': {'exit_code': 0, 'timed_out': False}},
             'final_answer': '{"source_file":"src/click/decorators.py",'
                             '"symbol":"option",'
                             '"test_file":"tests/test_command_decorators.py",'
                             '"explanation":"looked at implementation"}'}
        g = f.grade_case(c, r, 'completed')
        self.assertTrue(g['implementation_correct'])
        self.assertFalse(g['test_correct'])
        self.assertFalse(g['correct'])
        self.assertTrue(g['focused_pytest_passed'])

    def test_passing_pytest_cannot_replace_reading_the_symbol(self):
        c = f.CASES[0]
        r = {'source': {'source_path': c['expected_source'],
                        'source_symbol': 'CliRunner.helper'},
             'tests': {'test_path': c['expected_test'],
                       'targeted': {'exit_code': 0}},
             'final_answer': '{"source_file":"src/click/testing.py",'
                             '"symbol":"CliRunner.invoke",'
                             '"test_file":"tests/test_testing.py",'
                             '"explanation":"located"}'}
        g = f.grade_case(c, r, 'completed')
        self.assertTrue(g['correct'])  # retained v4 exact-match metric
        self.assertFalse(g['implementation_correct'])

    def test_usage_formatter_alternate_keeps_exact_metric_false(self):
        c = next(x for x in f.CASES if x['id'] == 'usage_formatter')
        r = {'source': {'source_path': c['expected_source'],
                        'source_symbol': c['expected_symbol']},
             'tests': {'test_path': 'tests/test_arguments.py',
                       'targeted': {'exit_code': 0}},
             'final_answer': '{"source_file":"src/click/formatting.py",'
                             '"symbol":"HelpFormatter.write_usage",'
                             '"test_file":"tests/test_arguments.py",'
                             '"explanation":"investigated"}'}
        g = f.grade_case(c, r, 'completed')
        self.assertTrue(g['implementation_correct'])
        self.assertFalse(g['test_correct'])
        self.assertTrue(g['focused_pytest_passed'])

    def test_wrong_tool_choice_does_not_pass(self):
        c = f.CASES[0]
        r = {'source':{'source_path':'src/click/core.py'},
             'tests':{'test_path':c['expected_test'],'targeted':{'exit_code':0}},
             'final_answer':'{"source_file":"src/click/testing.py","symbol":"CliRunner.invoke","test_file":"tests/test_testing.py","explanation":"guessed"}'}
        g=f.grade_case(c,r,'completed')
        self.assertFalse(g['correct']); self.assertFalse(g['source_correct'])
    def test_invalid_json_not_correct(self):
        c = f.CASES[0]
        self.assertFalse(f.grade_case(c,{'final_answer':'invalid'},'completed')['correct'])
    def test_expected_symbol_requires_exact_name(self):
        self.assertFalse(f.normalized_symbol_equal('Group.invoke','Group.resolve_command'))
        self.assertTrue(f.normalized_symbol_equal('click.core.Group.resolve_command','Group.resolve_command'))
    def test_preflight_checks_repo_version(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            case = f.CASES[0]
            a=repo/case['expected_source']; b=repo/case['expected_test']
            a.parent.mkdir(parents=True); b.parent.mkdir(parents=True)
            a.write_text('class CliRunner:\n    def invoke(self):\n        pass\n')
            b.write_text('def test_runner(): pass\n')
            f.preflight(case,repo)
            a.write_text('def wrong(): pass\n')
            with self.assertRaises(RuntimeError):
                f.preflight(case,repo)
    def test_excerpt_contains_relevant_late_lines(self):
        with tempfile.TemporaryDirectory() as d:
            repo=Path(d); p=repo/'src/click/core.py'
            p.parent.mkdir(parents=True)
            p.write_text('\n'.join(['x = 0']*2500+['def resolve_command(self):','  return None']+['x=1']*10))
            pilot=f.CodingPilot(repo,'http://localhost/v1','python',1,30)
            excerpt=pilot.relevant_source_excerpt(p,'resolve_command')
            self.assertIn('def resolve_command',excerpt)
            self.assertIn('2501:',excerpt)

class DiscoveryFallbackTests(unittest.TestCase):
    def test_source_file_listing_does_not_use_gold_labels(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "src/click").mkdir(parents=True)
            for name in ["core.py", "testing.py"]:
                (repo / "src/click" / name).write_text("# test fixture")
            pilot = f.CodingPilot(repo, "http://127.0.0.1:8080/v1", "python", 1, 10)
            names = [x["path"] for x in pilot.list_source_candidates()]
            self.assertEqual(names, ["src/click/core.py", "src/click/testing.py"])

    def test_zero_hit_keyword_falls_back_to_test_directory_listing(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "tests").mkdir(parents=True)
            (repo / "tests/test_testing.py").write_text("def test_run(): pass")
            pilot = f.CodingPilot(repo, "http://127.0.0.1:8080/v1", "python", 1, 10)
            pilot.question = "Find test helper behavior"
            pilot.grep_files = lambda root, term: []

            async def fake_llm(sid, deps, messages, max_tokens=240):
                if sid == "B2":
                    self.assertIn("tests/test_testing.py", messages[1]["content"])
                    return '{"path":"tests/test_testing.py"}'
                if sid == "D0":
                    return '{"action":"finish","reason":"test result observed"}'
                if sid == "B5":
                    return "The focused test passed."
                raise AssertionError(sid)

            async def fake_pytest(sid, deps, test_path):
                self.assertEqual(test_path, "tests/test_testing.py")
                return {"exit_code": 0, "output": "1 passed", "timed_out": False}

            pilot.llm = fake_llm
            pilot.pytest = fake_pytest
            outcome = asyncio.run(pilot.test_branch("testcommand"))
            self.assertEqual(outcome["test_path"], "tests/test_testing.py")
            self.assertEqual(outcome["targeted"]["exit_code"], 0)
            self.assertTrue(any(e["id"] == "B1L" for e in pilot.events))

class NavigationIndexTests(unittest.TestCase):
    def test_source_symbol_index_and_exact_read(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            p = repo / 'src/click/testing.py'
            p.parent.mkdir(parents=True)
            p.write_text('''class CliRunner:
    """Runs a command line app during tests."""

    def helper(self):
        pass

    def invoke(self, cli):
        """Invoke a command and capture results."""
        return cli()
''')
            pilot = f.CodingPilot(repo, 'http://127.0.0.1:8080/v1', 'python', 1, 10)
            pilot.question = 'Find helper that runs CLI and captures results'
            index = pilot.source_symbol_index(p)
            symbols = [o['symbol'] for o in index['symbols']]
            self.assertIn('CliRunner.invoke', symbols)
            content = pilot.read_source_symbol(p, 'CliRunner.invoke')
            self.assertIn('def invoke', content)
            self.assertIn('return cli()', content)
            self.assertNotIn('def helper', content)
            with self.assertRaises(ValueError):
                pilot.read_source_symbol(p, 'NoSuchMethod')

    def test_test_symbol_index_from_real_file_contents(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            p = repo / 'tests/test_testing.py'
            p.parent.mkdir(parents=True)
            p.write_text('''def test_runner(): pass

def test_invoke_result(): pass
''')
            pilot = f.CodingPilot(repo, 'http://127.0.0.1:8080/v1', 'python', 1, 10)
            pilot.question = 'Test CLI runner output capture'
            index = pilot.index_test_candidates([{'path':'tests/test_testing.py', 'count':0}])
            tests = index['files'][0]['example_tests']
            self.assertIn('test_runner', tests)
            self.assertIn('test_invoke_result', tests)
            self.assertEqual(index['files'][0]['test_count'],2)

    def test_source_branch_symbol_selected_by_model_then_read(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            p = repo / 'src/click/testing.py'
            p.parent.mkdir(parents=True)
            p.write_text('''class CliRunner:
    def invoke(self, command):
        return command()
''')
            pilot = f.CodingPilot(repo, 'http://127.0.0.1:8080/v1', 'python', 1, 10)
            pilot.question = 'Locate command runner'
            pilot.grep_files = lambda root,term: [{'path':'src/click/testing.py', 'count':1}]
            async def fake_llm(sid,deps,messages,max_tokens=240):
                if sid=='A2': return '{"path":"src/click/testing.py"}'
                if sid=='A3S':
                    self.assertIn('CliRunner.invoke',messages[1]['content'])
                    return '{"symbol":"CliRunner.invoke"}'
                if sid=='A4':
                    self.assertIn('def invoke',messages[1]['content'])
                    return 'CliRunner.invoke runs the command.'
                raise AssertionError(sid)
            pilot.llm=fake_llm
            result=asyncio.run(pilot.source_branch('runner'))
            self.assertEqual(result['source_symbol'],'CliRunner.invoke')
            self.assertIn('A3I',[e['id'] for e in pilot.events])
            self.assertIn('A3',[e['id'] for e in pilot.events])

class TestPathRecoveryTests(unittest.TestCase):
    def test_invalid_b2_path_gets_reprompt_and_valid_path_runs(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / 'tests').mkdir()
            (repo / 'tests/test_testing.py').write_text('def test_invoke_capture(): pass\n')
            (repo / 'tests/test_options.py').write_text('def test_option_default(): pass\n')
            pilot = f.CodingPilot(repo, 'http://127.0.0.1:8080/v1', 'python', 2, 10)
            pilot.question = 'Locate CLI testing runner and capture tests'
            pilot.grep_files = lambda root, term: []
            seen = []

            async def fake_llm(sid, deps, messages, max_tokens=240):
                seen.append((sid, deps))
                if sid == 'B2':
                    return '{"path":"tests/test_capture.py"}'
                if sid == 'B2R1':
                    self.assertIn('tests/test_testing.py', messages[-1]['content'])
                    self.assertIn('VALIDATION ERROR', messages[-1]['content'])
                    return '{"path":"tests/test_testing.py"}'
                if sid == 'D0':
                    return '{"action":"finish","reason":"pytest complete"}'
                if sid == 'B5':
                    return 'Focused test passed.'
                raise AssertionError(sid)

            async def fake_pytest(sid, deps, test_path):
                self.assertEqual((sid, deps, test_path),
                                 ('B3', ['B2R1'], 'tests/test_testing.py'))
                return {'exit_code': 0, 'output': '1 passed', 'timed_out': False}

            pilot.llm = fake_llm
            pilot.pytest = fake_pytest
            result = asyncio.run(pilot.test_branch('testclick'))
            self.assertEqual(result['test_path'], 'tests/test_testing.py')
            self.assertIn(('B2R1', ['B2']), seen)
            self.assertTrue(result['targeted']['exit_code'] == 0)

    def test_b2_retries_are_bounded_and_no_invalid_path_executes(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / 'tests').mkdir()
            (repo / 'tests/test_testing.py').write_text('def test_invoke_capture(): pass\n')
            pilot = f.CodingPilot(repo, 'http://127.0.0.1:8080/v1', 'python', 1, 10)
            pilot.grep_files = lambda root, term: []
            calls = []

            async def fake_llm(sid, deps, messages, max_tokens=240):
                calls.append(sid)
                return '{"path":"tests/test_capture.py"}'

            async def fake_pytest(*args):
                self.fail('pytest must never run on an invalid path')

            pilot.llm = fake_llm
            pilot.pytest = fake_pytest
            with self.assertRaisesRegex(ValueError, 'after 3 attempts'):
                asyncio.run(pilot.test_branch('testclick'))
            self.assertEqual(calls, ['B2', 'B2R1', 'B2R2'])

    def test_compact_index_retains_all_allowed_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            pilot = f.CodingPilot(repo, 'http://127.0.0.1:8080/v1', 'python', 1, 10)
            pilot.question = 'How does a CLI runner capture command output?'
            files = [{'path': f'tests/test_{i:02d}.py', 'test_count': 1,
                      'example_tests':['test_misc'], 'grep_count': 0}
                     for i in range(20)]
            files.append({'path': 'tests/test_runner.py', 'test_count': 2,
                          'example_tests': ['test_capture_output'], 'grep_count': 0})
            compact = pilot.compact_test_index({'files': files})
            self.assertEqual(compact['total_count'], 21)
            self.assertIn('tests/test_runner.py',
                          [x['path'] for x in compact['detailed_candidates']])
            all_paths = ([x['path'] for x in compact['detailed_candidates']]
                         + compact['other_allowed_paths'])
            self.assertCountEqual(all_paths, [x['path'] for x in files])

class MemoryJoinTests(unittest.TestCase):
    def test_memory_trace_column_parser(self):
        import importlib.util
        import io
        s = importlib.util.spec_from_file_location('memory', SCRIPT.with_name('profile_gpu_memory.py'))
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        self.assertEqual(m.as_number('3452'), 3452.0)
        self.assertIsNone(m.as_number('N/A'))
        self.assertIsNone(m.as_number('inf'))

if __name__ == '__main__':
    unittest.main()
