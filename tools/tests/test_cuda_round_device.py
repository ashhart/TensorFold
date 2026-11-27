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
            policy=dict(config=dict(timed=True),events=[dict(kind='done',key=0,mark='cost',ms=0)])
            policy_path=root/'policy.json';client.write(policy_path,policy)
            h['policy_clock_sha256']=client.sha(policy_path.read_bytes())
            for i in range(streams):
                key=client.token_hash(prompts[i]);r['requests'][i]['prompt_sha256']=key
                r['requests'][i]['prompt_tokens']=32
                replies[i].update(prompt_sha256=key,prompt_tokens=32,cached_tokens=31)
            text=('[tensorfold] serving m (drafts: DFlash2; context: 256)\n'
                  +'TF_POLICY_CONFIG '+json.dumps(policy['config'])+'\n'
                  +'TF_POLICY '+json.dumps(policy['events'][0])+'\n'
                  +'TF_POLICY_END complete=true events=1\n'+encode(h,r))
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
                    with self.assertRaisesRegex(ValueError,'injected'):client.arm(args,'code',32,streams,0,1,policy=policy_path)
                else:
                    result=client.arm(args,'code',32,streams,0,1,policy=policy_path)
                    self.assertEqual(streams,len(result['replies']))
                    self.assertTrue(json.loads((root/f'code-n32-s{streams}-r0-t1/rounds.json').read_text())
                                    ['trace_commit_ids_equal'])
                    self.assertIn('host_commit_publication_union_ns',
                                  (root/f'code-n32-s{streams}-r0-t1/rounds.csv').read_text())
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=120)

    def test_zero_counts_and_prompt_id_are_valid_but_impossible_counts_fail(self):
        class Observed:
            def __init__(self, _):
                self.end = 'done'
                self.usage = dict(prompt_tokens=1,completion_tokens=1,prompt_tokens_details=dict(cached_tokens=0))
                self.runtime = dict(token_sha256=client.token_hash([0]))
                self.speculative = dict(rounds=0,drafted=0,accepted=0)
            def read(self, _): pass
            def receipt(self): return dict(first_output_s=0.01)
        with mock.patch.object(client,'Collector',Observed), \
                mock.patch.object(client,'response') as response, mock.patch.object(client,'lines'):
            response.return_value.__enter__.return_value=(None,None)
            result=client.completion('http://127.0.0.1:1',[0],1,1)
            self.assertEqual(0,result['cached_tokens'])
            self.assertEqual(0,result['speculative']['accepted'])
            old=Observed.__init__
            for mode in ('cache','spec'):
                def bad(self, arg):
                    old(self,arg)
                    if mode=='cache':self.usage['prompt_tokens_details']['cached_tokens']=2
                    else:self.speculative.update(drafted=0,accepted=1)
                with mock.patch.object(Observed,'__init__',bad), self.assertRaises(ValueError):
                    client.completion('http://127.0.0.1:1',[0],1,1)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(client,'request',return_value=dict(tokens=[0,1])):
            ids=client.prompts('',Path(folder),'code',32,1)
            self.assertEqual(0,ids[0][0])
            self.assertEqual(ids,client.prompts('',Path(folder),'code',32,1))

    def test_listener_override_is_refused_before_process_or_output_creation(self):
        with mock.patch.object(client.subprocess,'Popen') as spawn:
            for override in (['--host','0.0.0.0'],['--port','80'],['--unix-socket','s'],['--listen-fd','3'],['--config','c']):
                args=argparse.Namespace(server_arg=override)
                with self.assertRaisesRegex(ValueError,'server-arg is disabled'):
                    client.arm(args,'code',32,1,0,1)
            spawn.assert_not_called()

    def test_policy_coverage_workload_and_cost_validation(self):
        config=dict(timed=True)
        event=dict(kind='done',key=0,mark='cost',ms=0)
        def encode_event(e=event,ending='TF_POLICY_END complete=true events=1',c=config):
            return 'TF_POLICY_CONFIG '+json.dumps(c)+'\nTF_POLICY '+json.dumps(e)+'\n'+ending+'\n'
        self.assertEqual(dict(config=config,events=[event]),client.policy_receipt(encode_event()))
        for text in (encode_event(ending='TF_POLICY_END complete=false events=1'),
                     encode_event(ending='TF_POLICY_END complete=true events=2'),
                     encode_event(c=dict(timed=False)), encode_event(dict(event,ms=float('nan'))),
                     encode_event(dict(event,kind='elapsed')), encode_event(dict(event,ms=-1)),
                     encode_event()+'TF_POLICY_ERROR mismatch'):
            with self.assertRaises(ValueError):client.policy_receipt(text)

    def test_real_sse_cold_zero_counts_are_preserved(self):
        body=dict(choices=[dict(index=0,text='a',finish_reason='length')],
                  usage=dict(prompt_tokens=1,completion_tokens=1,prompt_tokens_details=dict(cached_tokens=0)),
                  tensorfold=dict(token_sha256=client.token_hash([0])),
                  speculative=dict(rounds=0,drafted=0,accepted=0))
        with mock.patch.object(client,'response') as response, \
                mock.patch.object(client,'lines',return_value=[b'data: '+json.dumps(body).encode(),b'data: [DONE]']):
            response.return_value.__enter__.return_value=(None,None)
            reply=client.completion('http://127.0.0.1:1',[0],1,1)
            self.assertEqual(0,reply['cached_tokens'])
            self.assertEqual(0,reply['speculative']['accepted'])

    def test_matrix_records_once_then_replays_same_cost_tape_in_both_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'matrix'
            calls=[]
            def arm(args,kind,n,streams,rep,enabled,policy=None,calibration=False):
                calls.append((enabled,calibration,policy))
                return dict(policy_file=str(root/'costs.json'),trace=enabled)
            argv=['cuda_round_device.py','--binary','native','--model-dir','model','--drafter','draft',
                  '--output',str(root),'--conditions','fixture','--source','a'*40+':'+'b'*40,
                  '--lengths','32','--streams','1','--kinds','code','--reps','2']
            with mock.patch.object(sys,'argv',argv), mock.patch.object(client,'arm',side_effect=arm), \
                    mock.patch.object(client,'compare') as compare:
                client.main()
            self.assertEqual([(0,True),(0,False),(1,False),(0,True),(1,False),(0,False)],
                             [(enabled,cal) for enabled,cal,_ in calls])
            self.assertEqual(4,compare.call_count)
            self.assertTrue(all(policy==root/'costs.json' for _,cal,policy in calls if not cal))

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
                   max_tokens=2,replies=replies,policy_sha256='e'*64)
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
