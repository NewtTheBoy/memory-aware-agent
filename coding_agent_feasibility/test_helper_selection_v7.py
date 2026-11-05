"""Regression tests for v7 Call Trace: rejecting invented/self helpers."""
import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'coding_v7_helper_fix', Path(__file__).with_name('coding_agent_correctness_5.py')
)
v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v7)


class HelperSelectionTests(unittest.TestCase):
    def make_pilot(self, root):
        repo = Path(root)
        folder = repo / 'src/click'
        folder.mkdir(parents=True)
        (folder / 'main.py').write_text('''from .utils import echo
class Engine:
    def execute(self):
        self.prepare()
        echo("hi")
    def prepare(self):
        return True
''', encoding='utf-8')
        (folder / 'utils.py').write_text('''def echo(x):
    return x
''', encoding='utf-8')
        pilot = v7.CodingPilot(repo, 'http://localhost/v1', 'python', 2, 10)
        pilot.question = 'Investigate how Engine.execute prepares and outputs'
        pilot.investigation_profile = 'call_trace'
        return pilot

    def test_self_selected_then_valid_helper_is_read(self):
        with tempfile.TemporaryDirectory() as root:
            pilot = self.make_pilot(root)
            asked = []
            async def fake_llm(sid, deps, messages, max_tokens=240):
                asked.append(sid)
                if sid == 'A2': return '{"path":"src/click/main.py"}'
                if sid == 'A3S': return '{"symbol":"Engine.execute"}'
                if sid == 'A3H':
                    self.assertNotIn('"symbol": "Engine.execute"', messages[1]['content'].split('Allowed direct helper candidates: ')[1].split('\\nPick one')[0])
                    return '{"path":"src/click/main.py","symbol":"Engine.execute"}'
                if sid == 'A3HR1':
                    self.assertIn('VALIDATION ERROR', messages[-1]['content'])
                    self.assertIn('NOT a helper', messages[-1]['content'])
                    return '{"path":"src/click/utils.py","symbol":"echo"}'
                if sid == 'A4':
                    self.assertIn('return x', messages[1]['content'])
                    return 'Engine.execute calls echo'
                raise AssertionError(sid)
            pilot.llm = fake_llm
            source = asyncio.run(pilot.source_branch('execute'))
            self.assertEqual(source['related_helper'],
                {'path':'src/click/utils.py','symbol':'echo'})
            self.assertEqual(asked.count('A3H'), 1)
            self.assertEqual(asked.count('A3HR1'), 1)
            self.assertEqual(sum(ev['id']=='A3R' for ev in pilot.events), 1)
            self.assertEqual(next(ev['depends_on'] for ev in pilot.events if ev['id']=='A3R'), ['A3HR1'])

    def test_repeated_invalid_choice_does_not_fake_helper_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            pilot = self.make_pilot(root)
            asked = []
            async def fake_llm(sid, deps, messages, max_tokens=240):
                asked.append(sid)
                if sid == 'A2': return '{"path":"src/click/main.py"}'
                if sid == 'A3S': return '{"symbol":"Engine.execute"}'
                if sid in ('A3H','A3HR1'):
                    return '{"path":"src/click/main.py","symbol":"Engine.execute"}'
                if sid == 'A4':
                    self.assertNotIn('TOOL RESULT related helper:',messages[1]['content'])
                    return 'No helper was inspected'
                raise AssertionError(sid)
            pilot.llm = fake_llm
            source = asyncio.run(pilot.source_branch('execute'))
            self.assertTrue(source['related_helper_available'])
            self.assertIsNone(source['related_helper'])
            self.assertNotIn('A3R', asked)
            self.assertFalse(any(ev['id']=='A3R' for ev in pilot.events))
            self.assertEqual(asked.count('A3HR1'),1)

    def test_no_direct_callee_does_not_add_fabricated_steps(self):
        with tempfile.TemporaryDirectory() as root:
            pilot = self.make_pilot(root)
            (Path(root) / 'src/click/main.py').write_text('''class Engine:
    def execute(self):
        return 42
''', encoding='utf-8')
            async def fake_llm(sid, deps, messages, max_tokens=240):
                if sid == 'A2': return '{"path":"src/click/main.py"}'
                if sid == 'A3S': return '{"symbol":"Engine.execute"}'
                if sid == 'A4': return 'No callee is present'
                raise AssertionError(sid)
            pilot.llm = fake_llm
            source = asyncio.run(pilot.source_branch('execute'))
            self.assertFalse(source['related_helper_available'])
            self.assertIsNone(source['related_helper'])
            self.assertFalse(any(ev['id'] in ('A3H','A3HR1','A3R') for ev in pilot.events))


if __name__ == '__main__':
    unittest.main()
