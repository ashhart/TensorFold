"""Offline client-arm controls: declared drafter, request flag and observed proposals must agree."""
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import test_cuda_prefill_ledger as fixture
module = fixture.module

class DrafterArm(unittest.TestCase):
    def exercise(self, drafted, source, proposals=None, extra_stage=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'native'; binary.write_bytes(b'fixture binary')
            output = root / 'receipts'; output.mkdir()
            args = argparse.Namespace(binary=binary, model_dir=root/'model', drafter=root/'drafter',
                                      output=output, lengths=[fixture.N], server_arg=[], start_timeout=1,
                                      prompts_from=None, expected_drafter='DFlash2')
            rows = fixture.sequence(drafted)
            if drafted:
                rows.insert(2, fixture.row('drafter_open',2))
                for number,row in enumerate(rows): row['sequence']=number
            if extra_stage:
                extra = fixture.row(extra_stage, len(rows))
                if extra_stage == 'drafter_absorb': extra.update(start_token=0, rows=fixture.N-1)
                if extra_stage == 'trailing_absorb': extra.update(start_token=fixture.N-1, rows=1)
                rows.append(extra)
            log = f'[tensorfold] serving m (sampling: greedy; drafts: {source}; context: 4352; loaded in 1s)\n'
            log += fixture.encode(rows)+'\n'
            seen = {}
            process = mock.Mock(); process.poll.return_value=None; process.wait.return_value=0
            def spawn(command, **options):
                seen['command']=command
                options['stdout'].write(log.encode()); options['stdout'].flush()
                return process
            def answer(base, route, body=None):
                if route=='/health': return {'status':'ok'}
                if route=='/v1/tokenize': return {'tokens':[3]}
                if route=='/v1/completions':
                    seen['body']=body
                    amount=(8 if drafted else 0) if proposals is None else proposals
                    return {'usage':{'prompt_tokens':fixture.N,'completion_tokens':64,
                                     'prompt_tokens_details':{'cached_tokens':0}},
                            'tensorfold':{'token_sha256':'a'*64},
                            'speculative':{'rounds':63,'drafted':amount,'accepted':min(amount,4)}}
                raise AssertionError(route)
            with mock.patch.object(module.socket,'socket') as sock, mock.patch.object(module.subprocess,'Popen',side_effect=spawn), mock.patch.object(module,'request',side_effect=answer):
                sock.return_value.__enter__.return_value.getsockname.return_value=('127.0.0.1',12345)
                result=module.run(args,fixture.N,drafted,1,0)
            self.assertIs(seen['body']['draft'],drafted)
            option=seen['command'].index('--drafter')+1
            self.assertEqual(str(args.drafter.resolve()) if drafted else 'none',seen['command'][option])
            process.terminate.assert_called_once()
            return result
    def test_plain_and_loaded_drafter_keep_flags_and_arguments_aligned(self):
        self.assertFalse(self.exercise(False,'copies only')['draft'])
        self.assertTrue(self.exercise(True,'DFlash2 (example/drafter) + copies')['draft'])
    def test_drafted_tag_cannot_hide_copies_only_startup(self):
        with self.assertRaises(ValueError): self.exercise(True,'copies only')
    def test_loaded_drafter_with_no_proposals_is_not_a_drafted_run(self):
        with self.assertRaises(ValueError): self.exercise(True,'DFlash2 (example/drafter)',0)
    def test_plain_rejects_drafter_open(self):
        with self.assertRaisesRegex(ValueError, 'plain arm emitted drafter prompt work'):
            self.exercise(False, 'copies only', extra_stage='drafter_open')
    def test_plain_rejects_tap_upload(self):
        with self.assertRaisesRegex(ValueError, 'plain arm emitted drafter prompt work'):
            self.exercise(False, 'copies only', extra_stage='tap_upload')
    def test_plain_rejects_ordinary_drafter_absorb(self):
        with self.assertRaisesRegex(ValueError, 'plain arm emitted drafter prompt work'):
            self.exercise(False, 'copies only', extra_stage='drafter_absorb')
    def test_plain_rejects_trailing_absorb(self):
        with self.assertRaisesRegex(ValueError, 'plain arm emitted drafter prompt work'):
            self.exercise(False, 'copies only', extra_stage='trailing_absorb')
    def test_drafter_only_selector_is_available_without_a_device(self):
        result=subprocess.run([sys.executable,str(fixture.TOOL),'--help'],capture_output=True,text=True)
        self.assertEqual(0,result.returncode,result.stderr)
        self.assertIn('--draft-arms',result.stdout)

if __name__=='__main__': unittest.main()
