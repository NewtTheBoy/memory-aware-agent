import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kv_memory_probe as kp


class Handler(BaseHTTPRequestHandler):
    processing = 0
    cache_seen = False
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def send_json(self, x):
        d = json.dumps(x).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(d)))
        self.end_headers()
        self.wfile.write(d)

    def do_GET(self):
        if self.path == '/health':
            return self.send_json({'status': 'ok'})
        if self.path == '/slots':
            with self.lock:
                n = type(self).processing
            return self.send_json([{'id': i, 'n_ctx': 4096, 'is_processing': i < n} for i in range(2)])
        if self.path == '/metrics':
            b = b'# metrics\nllamacpp:kv_cache_usage_ratio 0.45\nllamacpp:requests_processing 2\n'
            self.send_response(200)
            self.send_header('Content-Length', str(len(b)))
            self.end_headers()
            return self.wfile.write(b)
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', '0'))
        payload = json.loads(self.rfile.read(n))
        if self.path == '/tokenize':
            # Deterministic fake tokenizer; exactly split into words.
            return self.send_json({'tokens': list(range(len(payload['content'].split())))})
        if self.path == '/completion':
            with self.lock:
                type(self).processing += 1
            time.sleep(.25)
            with self.lock:
                type(self).processing -= 1
            cached = len(payload['prompt'].split()) - 20 if payload.get('id_slot') == 0 and type(self).cache_seen else 0
            if payload.get('id_slot') == 0:
                type(self).cache_seen = True
            return self.send_json({'content': 'DONE', 'tokens_cached': max(cached, 0),
                                   'tokens_evaluated': len(payload['prompt'].split()),
                                   'truncated': False, 'timings': {}})
        self.send_error(404)


class TestProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_token_calibration(self):
        _, count, method = kp.build_prompt(self.url, 'SMOKE', 200)
        self.assertLess(abs(count-200), 30)
        self.assertEqual(method, 'tokenize')

    def test_probe_records_overlap(self):
        p1, _, _ = kp.build_prompt(self.url, 'AA', 160)
        p2, _, _ = kp.build_prompt(self.url, 'BB', 160)
        r, samples = kp.run_pair(self.url, p1, p2, 10, 5, .1)
        self.assertTrue(all(x['ok'] for x in r['requests']))
        self.assertEqual(r['observation_summary']['busy_slots_peak'], 2)
        self.assertTrue(r['observation_summary']['candidate_kv_or_cache_metrics'])

    def test_reuse_prompt(self):
        p, _, _ = kp.build_prompt(self.url, 'REUSE', 160)
        res = kp.run_reuse(self.url, p, 5, .1)
        self.assertTrue(res['first']['ok'] and res['second']['ok'])
        self.assertGreater(res['second']['tokens_cached'], 0)

    def test_complete_cli(self):
        p = Path('/tmp/kv-probe-fake-test')
        r = subprocess.run([sys.executable, str(Path(kp.__file__)), '--url', self.url,
                            '--mode', 'contention', '--low-tokens', '160', '--high-tokens', '280',
                            '--n-predict', '10', '--interval', '0.1', '--out-dir', str(p)],
                           text=True, capture_output=True, timeout=30)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((p/'summary.json').exists())
        data = json.loads((p/'summary.json').read_text())
        self.assertEqual(len(data['pairs']), 2)
        self.assertEqual(data['pairs']['high']['observation_summary']['busy_slots_peak'], 2)


if __name__ == '__main__':
    unittest.main()
