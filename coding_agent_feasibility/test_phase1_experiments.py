import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase1_experiments as ex


class FakeServer(BaseHTTPRequestHandler):
    cache = {0:'',1:''}
    active = 0
    lock = threading.Lock()
    def log_message(self,*args): pass
    def response(self, data, status=200):
        b=json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(b)))
        self.end_headers();self.wfile.write(b)
    def do_GET(self):
        if self.path=='/health':return self.response({'status':'ok'})
        if self.path=='/slots':
            return self.response([{'id':i,'n_ctx':4096,'is_processing': False} for i in range(2)])
        if self.path=='/metrics':return self.response({})
        return self.response({'error':'404'},404)
    def do_POST(self):
        n=int(self.headers.get('Content-Length','0'))
        obj=json.loads(self.rfile.read(n))
        if self.path=='/tokenize':
            return self.response({'tokens':list(range(len(obj.get('content','').split())))})
        if self.path.startswith('/slots/') and self.path.endswith('?action=erase'):
            idx=int(self.path.split('/')[2].split('?')[0])
            with self.lock:
                old=len(type(self).cache[idx]);type(self).cache[idx]=''
            return self.response({'id_slot':idx,'n_erased':old})
        if self.path=='/completion':
            idx=int(obj.get('id_slot',0))
            prompt=obj['prompt'];token_count=len(prompt.split())
            with self.lock:
                old=type(self).cache[idx]
                oldw=old.split();noww=prompt.split()
                prefix=0
                for a,b in zip(oldw,noww):
                    if a!=b: break
                    prefix+=1
                cached=prefix if obj.get('cache_prompt') else 0
                type(self).cache[idx]=prompt+' canned answer'
            time.sleep(.025)
            return self.response({'content':'canned answer','tokens_cached':cached,
                                  'tokens_evaluated':token_count,'truncated':False,
                                  'timings':{'cache_n':cached,'prompt_n':token_count-cached,
                                             'prompt_ms':(token_count-cached)*.01,
                                             'predicted_n':obj['n_predict']}})
        return self.response({'error':'404'},404)


class Phase1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(('127.0.0.1',0),FakeServer)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url=f'http://127.0.0.1:{cls.server.server_address[1]}'
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(2)
    def setUp(self):
        ex.eraser(self.url)
    def test_erase_clears_old_context(self):
        p='some words in a test prompt'
        ex.execute(self.url,p,4,10)
        a=ex.eraser(self.url)
        self.assertGreater(a[0]['n_erased'],0)
    def test_e1_on_off_and_wait(self):
        rows=ex.do_e1(self.url,'alpha beta gamma delta', [0],4,10)
        self.assertEqual(len(rows),2)
        self.assertGreater(rows[0]['second']['cache_n'],0)
        self.assertEqual(rows[1]['second']['cache_n'],0)
    def test_e2_interference(self):
        rows=ex.do_e2(self.url,'alpha beta gamma delta',
                      {'short':'separate unrelated prompt'},0,4,10)
        self.assertEqual(len(rows),3)
        self.assertGreater(rows[0]['step2']['cache_n'],0)
        self.assertGreater(rows[1]['step2']['cache_n'],0)
        self.assertEqual(rows[2]['step2']['cache_n'],0)
    def test_e3_pair(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td, patch.object(ex.kp,'gpu_query',return_value=None):
            rows=ex.do_e3(self.url,{'mixed':('one two three','completely other words')},4,10,.1,Path(td))
            self.assertEqual(len(rows),1)
            self.assertEqual(len(rows[0]['concurrent']['requests']),2)
            self.assertEqual(len(rows[0]['serial']['requests']),2)
            self.assertTrue((Path(td)/'e3_mixed_samples.jsonl').exists())
    def test_save(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            ex.save({'records':[{'experiment':'E1','wait_s':0,'cache_enabled':True,
                                  'second':{'ok':True,'prompt_n':2,'cache_n':8}}]},Path(td))
            self.assertEqual(json.loads((Path(td)/'summary.json').read_text())['records'][0]['experiment'],'E1')
            self.assertIn('prompt_n',(Path(td)/'summary.csv').read_text())

    def test_server_context_check(self):
        txt='123 ./build/bin/llama-server -c 8192 -np 2'
        with patch.object(ex.subprocess, 'check_output', return_value=txt):
            self.assertEqual(ex.check_local_server_context(8192)['server_context_capacity_arg'],8192)
            with self.assertRaisesRegex(RuntimeError,'server -c=8192'):
                ex.check_local_server_context(4096)

if __name__=='__main__':unittest.main()
