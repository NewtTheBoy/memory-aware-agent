#!/usr/bin/env python3
"""Run from project root: python -m unittest discover -s coding_agent_feasibility -p 'test_phase1_experiment_e8.py' -v"""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('phase1_experiment_e8', HERE/'phase1_experiment_e8.py')
e8=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(e8)


class E8Tests(unittest.TestCase):
    def test_parse_kv_memory_log(self):
        raw='''llama_kv_cache_unified:     CUDA0 KV buffer size =   576.00 MiB
llama_kv_cache_unified:       CPU KV buffer size =     2.00 MiB
llama_kv_cache_unified:     CUDA1 KV buffer size =   0.50 GiB
llama_context: CUDA0 compute buffer size = 123.00 MiB'''
        p=e8.parse_kv_log(raw)
        self.assertEqual(p['gpu_kv_mib_sum'],1088.0)
        self.assertEqual(len(p['records']),3)
        self.assertEqual(len(p['gpu_kv_lines']),2)

    def test_parse_nonstandard_log(self):
        p=e8.parse_kv_log('unexpected KV buffer size = 99 MiB')
        self.assertIsNone(p['gpu_kv_mib_sum'])
        self.assertEqual(len(p['records']),1)

    def test_parse_nvidia_smi(self):
        self.assertEqual(e8.parse_nvidia_gpu(' 3285 MiB, 4526 MiB, 8151 MiB\n')['free_mib'],4526.)
        with self.assertRaises(ValueError):
            e8.parse_nvidia_gpu('N/A, 300, 8151')

    def test_order(self):
        self.assertEqual(e8.modes_for_repeat(1),['keep','erase'])
        self.assertEqual(e8.modes_for_repeat(2),['erase','keep'])

    def test_snapshot_repeat_median(self):
        with patch.object(e8,'gpu_read',side_effect=[
            {'used_mib':100,'free_mib':900,'total_mib':1000},
            {'used_mib':120,'free_mib':880,'total_mib':1000},
            {'used_mib':110,'free_mib':890,'total_mib':1000}]):
            rows=[]
            with patch.object(e8.time,'sleep'):
                d=e8.snapshot('before_a2',4096,'keep',1,0,3,0.02,rows)
        self.assertEqual(d['used_mib'],110)
        self.assertEqual(d['used_range_mib'],20)
        self.assertEqual(len(rows),3)

    def test_calibrate(self):
        with patch.object(e8,'prompt_token_count',side_effect=lambda b,p: len(p)//5):
            n,p=e8.calibrated_prompt('http://x',1800)
        self.assertLess(abs(n-1800),20)
        self.assertIn('Python repository',p)

    def test_trial_keep_erase_behavior(self):
        args=SimpleNamespace(gpu=0,samples=1,sample_gap=0.,predict=8,request_timeout=10.,wait=0.)
        state={'cached':False,'requests':0}
        def erase(base,slot):
            state['cached']=False
            return {'id_slot':slot,'n_erased':10}
        def completion(base,prompt,slot,predict,timeout):
            old=state['cached']; state['cached']=True;state['requests']+=1
            return {'ok':True,'prompt_ms':12 if old else 400,
                    'cache_n':500 if old else 0,'prompt_n':20 if old else 520,
                    'request_s':0.1}
        def snap(label,*a,**kw):
            return {'used_mib':3400.0,'free_mib':4000.0,'used_range_mib':0.}
        with patch.object(e8,'erase',side_effect=erase),\
             patch.object(e8,'completion',side_effect=completion),\
             patch.object(e8,'snapshot',side_effect=snap):
            k=e8.trial('http://x','keep',1,4096,'prompt',args,[])
            v=e8.trial('http://x','erase',1,4096,'prompt',args,[])
        self.assertEqual(k['a2']['cache_n'],500)
        self.assertEqual(v['a2']['cache_n'],0)
        self.assertTrue(k['ok'] and v['ok'])

    def test_export_grouping_success_only(self):
        def r(b,m,rep,ok=True):
            return {'budget':b,'mode':m,'rep':rep,'ok':ok,
                    'a1':{'request_s':0.3},
                    'a2':{'cache_n':123 if m=='keep' else 0,
                          'prompt_n':10 if m=='keep' else 150,
                          'prompt_ms':20 if m=='keep' else 240,'request_s':0.7},
                    'gpu_used_delta_wait_mib':-5 if m=='keep' else -8,
                    'gpu_used_delta_from_initial_mib':0,
                    'snapshots':{k:{'used_mib':3200} for k in ['initial_idle','after_a1','before_a2','after_a2']}}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            s=e8.export(root,[r(4096,'keep',1),r(4096,'erase',1),r(8192,'keep',1)],
                        [],[],{'test':True})
            self.assertEqual(len(s),2)
            self.assertEqual(s[0]['erase_minus_keep_vram_delta_mib'],-3.)
            self.assertTrue((root/'trials.csv').exists())
            self.assertEqual(json.loads((root/'results.json').read_text())['settings'],{'test':True})
            with (root/'comparison.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),2)

    def test_cli_validation_wont_kill_existing_server(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td); (p/'llama-server').write_text('binary')
            with patch.object(e8,'server_is_busy',return_value=True):
                with self.assertRaises(SystemExit) as caught:
                    e8.main(['--project-root',str(p),'--server-binary','llama-server',
                             '--out-dir',str(p/'out')])
            self.assertIn('STOP existing server yourself',str(caught.exception))


if __name__=='__main__':unittest.main()
