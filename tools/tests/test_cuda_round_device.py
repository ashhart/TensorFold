"""Round recipe coverage and exact token joins using synthetic receipts; no socket, server or GPU."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).parents[1]))
import cuda_round_device as client
import cuda_round_attribution as trace
from test_cuda_round_attribution import fixture


def sample(streams=1):
    header,_,_,r = fixture()
    header.update(quantization=dict(bits=4,group_size=64,mode='affine'),sm_minor=1,
                  geometry=dict(kinds=dict(modelopt=False)),drafter=dict(identity=dict(checkpoint_sha256='d'*64)))
    r['requests'] = []
    windows = []
    replies = []
    template = r['windows'][0]
    for index in range(streams):
        prompt = client.sha(f'prompt-{index}'.encode())
        ids = [101+index,37]
        r['requests'].append(dict(stream=str(index),prompt_sha256=prompt,prompt_tokens=8,cached=8,
                                  sampling_json=json.dumps(dict(sampling=dict(seed=7),penalties={},drafts=True))))
        windows.append(dict(template,stream=str(index),committed_ids=ids))
        replies.append(dict(prompt_sha256=prompt,prompt_tokens=8,completion_tokens=2,cached_tokens=7,
                            token_sha256=client.token_hash(ids),speculative=dict(rounds=1,drafted=4,accepted=2)))
    r['windows'] = windows
    r['frames'][0].update(live_rows=5*streams,windows=streams,shape_bucket=32)
    q = dict(m=5*streams,n=5120,k=5120,sk=4,cluster=[1,1,1],reduce=False,
             mode='mlx_affine4_g64',reduction='cta_k_order')
    r['device'][0].update(op='affine_g64',qmmf=[],affine_g64=[q])
    r['budget'] = trace.calculate(r)
    return header,r,replies


def encode(header, r):
    raw = json.dumps(header,separators=(',',':'))
    key = hashlib.sha256(raw.encode()).hexdigest()
    r['identity'] = key
    return f'TF_ROUND_IDENTITY {key} {raw}\nTF_ROUND '+json.dumps(r)+'\n'


class Device(unittest.TestCase):
    def exercise_arm(self, streams, failure=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); binary=root/'native'; binary.write_bytes(b'synthetic executable')
            args=argparse.Namespace(output=root,binary=binary,model_dir=root/'model',drafter=root/'draft',
                                    max_tokens=2,cache_gib=24,server_arg=[],conditions='fixture',
                                    start_timeout=1,stream_timeout=1,prompts_from=None,source='a'*40+':'+'b'*40)
            prompts=[[i]+[1]*31 for i in range(streams)]
            h,r,replies=sample(streams)
            h['binary_sha256']=client.sha(binary.read_bytes())
            for i in range(streams):
                key=client.token_hash(prompts[i]);r['requests'][i]['prompt_sha256']=key
                r['requests'][i]['prompt_tokens']=32
                replies[i].update(prompt_sha256=key,prompt_tokens=32,cached_tokens=31)
            text='[tensorfold] serving m (drafts: DFlash2; context: 256)\n'+encode(h,r)
            process=mock.Mock();process.poll.return_value=None
            def spawn(command,**options):
                self.assertIn('--no-update-check',command)
                self.assertIn('127.0.0.1',command)
                options['stdout'].write(text.encode());options['stdout'].flush()
                return process
            def completion(base,ids,tokens,timeout,barrier=None):
                if failure: raise ValueError('injected completion failure')
                if tokens==1:return {'preparation':'synthetic'}
                barrier.wait(timeout=1)
                return replies[ids[0]]
            with mock.patch.object(client.socket,'socket') as sock, \
                    mock.patch.object(client.subprocess,'Popen',side_effect=spawn), \
                    mock.patch.object(client,'request',return_value={}), \
                    mock.patch.object(client,'prompts',return_value=prompts), \
                    mock.patch.object(client,'completion',side_effect=completion):
                sock.return_value.__enter__.return_value.getsockname.return_value=('127.0.0.1',19001)
                if failure:
                    with self.assertRaisesRegex(ValueError,'injected'):client.arm(args,'code',32,streams,0,1)
                else:
                    result=client.arm(args,'code',32,streams,0,1)
                    self.assertEqual(streams,len(result['replies']))
                    self.assertTrue(json.loads((root/f'code-n32-s{streams}-r0-t1/rounds.json').read_text())
                                    ['trace_commit_ids_equal'])
                    self.assertIn('host_commit_publication_union_ns',
                                  (root/f'code-n32-s{streams}-r0-t1/rounds.csv').read_text())
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=120)

    def test_arm_prepares_then_submits_one_or_four_and_joins_receipts(self):
        self.exercise_arm(1)
        self.exercise_arm(4)

    def test_arm_failure_still_stops_only_its_started_process(self):
        self.exercise_arm(1,True)

    def test_actual_request_windows_and_trace_commit_ids(self):
        for streams in (1,4):
            h,r,replies = sample(streams)
            result = client.decode_rounds(encode(h,r),'c'*64,replies,streams)
            self.assertTrue(result['trace_commit_ids_equal'])
            self.assertTrue(result['actual_stream_window_seen'])
            self.assertEqual(1,len(result['per_round']))
            with self.assertRaisesRegex(ValueError,'source attestation'):
                client.decode_rounds(encode(h,r),'c'*64,replies,streams,'f'*40+':'+'f'*40)

    def test_negative_device_evidence(self):
        for bad in ('binary','target','drafter','missing_projection','tokens','count'):
            h,r,replies = sample()
            if bad == 'binary': h['binary_sha256']='a'*64
            if bad == 'target': h['geometry']['kinds']['modelopt']=True
            if bad == 'drafter': h['drafter']=None
            if bad == 'missing_projection':
                r['device'][0].update(op='other',affine_g64=[])
                r['budget']=trace.calculate(r)
            if bad == 'tokens': r['windows'][0]['committed_ids'][0]+=1
            if bad == 'count': replies[0]['completion_tokens']+=1
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                client.decode_rounds(encode(h,r),'c'*64,replies,1)

    def test_four_http_requests_without_a_four_window_frame_are_refused(self):
        h,r,replies = sample(4)
        frame = r['frames'][0]
        frame.update(windows=2,live_rows=10,shape_bucket=16)
        r['frames'].append(copy.deepcopy(frame))
        with self.assertRaisesRegex(ValueError,'not concurrent enough'):
            client.decode_rounds(encode(h,r),'c'*64,replies,4)

    def test_prompt_preparation_rounds_are_excluded_from_decode_statistics(self):
        h,r,replies = sample()
        measured = encode(h,r)
        prep = copy.deepcopy(r)
        prep['round']=2
        prep['requests'][0]['prompt_sha256']='a'*64
        text = measured+'TF_ROUND '+json.dumps(prep)+'\n'
        result = client.decode_rounds(text,'c'*64,replies,1)
        self.assertEqual(1,result['preparation_rounds_excluded'])
        self.assertEqual(1,result['measured_rounds'])

    def test_pair_validation_rejects_token_and_counter_changes_under_optimization(self):
        _,_,replies = sample()
        off = dict(binary_sha256='c'*64,source_hashes={},source='a'*40+':'+'b'*40,kind='code',length=8,streams=1,cache_gib=24,
                   max_tokens=2,replies=replies)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); a=root/'off.json'; b=root/'on.json'
            a.write_text(json.dumps(off))
            for mode in ('same','token_sha256','speculative','cached_tokens'):
                on=copy.deepcopy(off)
                if mode=='token_sha256': on['replies'][0][mode]='a'*64
                if mode=='speculative': on['replies'][0][mode]['accepted']+=1
                if mode=='cached_tokens': on['replies'][0][mode]-=1
                b.write_text(json.dumps(on))
                code=('import json,sys; sys.path.insert(0,sys.argv[1]); import cuda_round_device as c; '
                      'c.compare(json.load(open(sys.argv[2])),json.load(open(sys.argv[3])))')
                result=subprocess.run([sys.executable,'-B','-O','-c',code,str(Path(client.__file__).parent),
                                       str(a),str(b)],capture_output=True,text=True)
                with self.subTest(mode=mode):
                    self.assertEqual(mode=='same',result.returncode==0,result.stderr)
                    if mode!='same': self.assertIn(mode+' differs',result.stderr)


if __name__ == '__main__':
    unittest.main()
