"""Offline validation catches missing final-row absorbs, restored cold runs and missing physical observations."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
TOOL = Path(__file__).resolve().parents[1] / 'cuda_prefill_ledger.py'
spec = importlib.util.spec_from_file_location('prefill_ledger', TOOL)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
N = 4096

def row(stage, sequence, **extra):
    return dict(schema=module.SCHEMA, sequence=sequence, parent=None, request=1, stage=stage,
                complete=True, dropped_stages=0, event_status='not_launched', cuda_event_ns=None,
                graph_mode='eager', launches=0,
                restoration=dict(kind='fresh',tokens_restored=0,kv_restored=False,recurrent_restored=False,restore_failed=False),
                memory=dict(rss_bytes=1024,physical_peak_bytes=2048), **extra)

def sequence(drafted=True):
    out = [row('restore',0), row('target_chunk',1,start_token=0,rows=N,layers_run=64,
                               shape_bucket_rows=N,shape_bucket_segments=1)]
    out[1].update(event_status='complete',cuda_event_ns=1000,launches=1)
    if drafted:
        out += [row('drafter_absorb',2,start_token=0,rows=N-1),row('trailing_absorb',3,start_token=N-1,rows=1)]
    return out

def encode(rows): return '\n'.join(map(json.dumps,rows))

class Ledger(unittest.TestCase):
    def rejected(self,rows):
        with self.assertRaises((ValueError,AssertionError,KeyError)):
            module.ledger(encode(rows),N,True)
    def test_good_plain_and_real_final_row(self):
        for drafted in (False,True):
            self.assertEqual(sequence(drafted),module.ledger(encode(sequence(drafted)),N,drafted))
    def test_ordinary_absorb_cannot_impersonate_final_row(self):
        rows=sequence()[:-1];rows[-1]['stage']='trailing_absorb';self.rejected(rows)
    def test_restore_record_must_be_fresh_independently_of_chunk(self):
        rows=sequence();rows[0]['restoration'].update(kind='snapshot',tokens_restored=1,kv_restored=True,recurrent_restored=True);self.rejected(rows)
    def test_physical_rss_is_required_even_with_peak(self):
        rows=sequence();rows[1]['memory']['rss_bytes']=None;self.rejected(rows)
    def test_physical_peak_is_required_even_with_rss(self):
        rows=sequence();rows[1]['memory']['physical_peak_bytes']=None;self.rejected(rows)
    def test_existing_bad_controls_stay_rejected(self):
        variants=[]
        rows=sequence();rows[1]['complete']=False;variants.append(rows)
        variants.append(sequence()[:-1])
        rows=sequence();rows[1]['rows']=N-1;rows[1]['shape_bucket_rows']=N-1;variants.append(rows)
        rows=sequence();rows[1]['event_status']='unavailable';rows[1]['cuda_event_ns']=None;variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows): self.rejected(rows)
    def test_offline_cli_emits_reproducible_expected_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'controls.json'
            run=subprocess.run([sys.executable,str(TOOL),'--self-test','--controls-output',str(output)],capture_output=True,text=True)
            self.assertEqual(0,run.returncode,run.stdout+run.stderr)
            doc=json.loads(output.read_text())
            self.assertFalse(doc['network_or_device_run'])
            controls=doc['controls']
            self.assertGreaterEqual(len(controls),8)
            self.assertIn('restored_restore',{x['name'] for x in controls})
            self.assertIn('missing_rss',{x['name'] for x in controls})
            for case in controls:
                self.assertEqual('rejected',case['outcome'])
                self.assertTrue(case['reason'])
                self.rejected(json.loads(case['input_json']))

if __name__=='__main__': unittest.main()
