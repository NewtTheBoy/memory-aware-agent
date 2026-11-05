import unittest
from unittest.mock import patch
import dynamic_gpu_feasibility as p

class TestProbe(unittest.TestCase):
    def test_parse_nvidia_csv(self):
        self.assertEqual(p.parse_nvidia_csv('123, 299 MiB\n456, 400 MiB\n'),
                         [['123', '299 MiB'], ['456', '400 MiB']])
    def test_kv_metrics(self):
        body = 'llamacpp:kv_cache_usage_ratio 0.52\n# ignored\nllamacpp:requests_processing 1\n'
        class Obj:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self): return body.encode()
        with patch.object(p.request, 'urlopen', return_value=Obj()):
            self.assertEqual(p.kv_metric_values('http://x'), {'llamacpp:kv_cache_usage_ratio': 0.52})
    def test_prompt(self):
        def fake_tok(base, content): return int(len(content) / 7)
        with patch.object(p, 'token_count', fake_tok):
            n, value = p.prompt_for_tokens('http://x', 2200)
            self.assertLess(abs(n - 2200), 60)
            self.assertGreater(len(value), 5000)
    def test_completion(self):
        with patch.object(p, 'api', return_value={'timings': {'cache_n': 21, 'prompt_n': 5, 'prompt_ms': 4}}):
            a = p.completion('http://localhost', 'abc', 0, 8, 10)
            self.assertEqual(a['cache_n'], 21)
            self.assertTrue(a['ok'])

if __name__ == '__main__': unittest.main()
