#!/usr/bin/env python3
"""Lead-run MLX+DFlash2 decode matrix with prepared prefixes; no device is accessed by offline tests."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import struct
import subprocess
import threading
import time
from bench_stream import Collector
import cuda_round_attribution as trace
from cuda_trace_http import request, response, lines, remaining


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(data): return hashlib.sha256(data).hexdigest()
def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def count(value): return type(value) is int and value >= 0
def token_hash(ids): return sha(b''.join(struct.pack('<I', x) for x in ids))


def completion(base, prompt, tokens, timeout, barrier=None):
    body = dict(model='round-check', prompt=prompt, max_tokens=tokens, temperature=0, seed=7,
                draft=True, ignore_eos=True, stream=True, stream_options=dict(include_usage=True))
    if barrier is not None:
        barrier.wait(timeout=30)
    began = time.perf_counter()
    observed = Collector(began)
    with response(base, '/v1/completions', body, timeout=timeout) as values:
        observed.read(lines(*values, timeout=timeout))
    require(observed.end == 'done', 'incomplete/error SSE response; no partial result accepted')
    timing = observed.receipt()
    require(timing['first_output_s'] is not None, 'response has no output')
    usage = observed.usage
    for key, expected in [('prompt_tokens',len(prompt)), ('completion_tokens',tokens)]:
        require(count(usage.get(key)) and usage[key] == expected,
                f'{key} differs: expected {expected}, observed {usage.get(key)!r}')
    cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    fingerprint = observed.runtime.get('token_sha256')
    require(count(cached) and cached <= len(prompt), 'missing or impossible observed cached-token count')
    require(trace.sha(fingerprint), 'missing output token SHA-256')
    counters = {k:observed.speculative.get(k) for k in ('rounds','drafted','accepted')}
    require(all(count(v) for v in counters.values()), 'missing speculative counters')
    require(counters['accepted'] <= counters['drafted'], 'accepted count exceeds drafted count')
    return dict(prompt_tokens=usage['prompt_tokens'], prompt_sha256=token_hash(prompt),
                completion_tokens=usage['completion_tokens'], cached_tokens=cached,
                token_sha256=fingerprint, speculative=counters, timing=timing)


def compare(off, on):
    for key in ('binary_sha256','source_hashes','source','kind','length','streams','cache_gib','max_tokens','policy_sha256'):
        require(off[key] == on[key], 'arm identity differs: '+key)
    require(len(off['replies']) == len(on['replies']) == off['streams'], 'missing stream replies')
    for index, (a,b) in enumerate(zip(off['replies'], on['replies'])):
        for key in ('prompt_tokens','prompt_sha256','completion_tokens','cached_tokens','token_sha256','speculative'):
            require(a[key] == b[key], f'stream {index} {key} differs: off {a[key]!r}, on {b[key]!r}')


def decode_rounds(text, binary, replies, streams, expected_source=None, expected_policy=None):
    headers, all_rounds = trace.read(text.splitlines(keepends=True))
    expected = {r['prompt_sha256']:r for r in replies}
    require(len(expected) == streams, 'measurement prompts must have distinct token identities')
    selected = [r for r in all_rounds if all(q['prompt_sha256'] in expected for q in r['requests'])]
    require(selected, 'no rounds bound to the measured requests')
    committed = {key:[] for key in expected}
    affine = 0
    for r in selected:
        h = headers[r['identity']]
        require(h['binary_sha256'] == binary, 'trace executable digest differs from served binary')
        require(expected_source is None or h['source'] == expected_source, 'trace source attestation differs')
        require(expected_policy is None or h.get('policy_clock_sha256') == expected_policy, 'trace policy attestation differs')
        require((h['sm_major'],h['sm_minor']) == (12,1), 'this recipe requires GB10 SM121')
        require(h['geometry'].get('kinds',{}).get('modelopt') is False, 'trace target is not MLX')
        require(isinstance(h.get('drafter'),dict) and trace.sha(h['drafter']['identity']['checkpoint_sha256']),
                'trace lacks loaded drafter content identity')
        requests = {q['stream']:q for q in r['requests']}
        for q in requests.values():
            require(q['prompt_tokens'] == expected[q['prompt_sha256']]['prompt_tokens'], 'trace prompt count differs')
        for w in r['windows']:
            committed[requests[w['stream']]['prompt_sha256']] += w['committed_ids']
        affine += sum(d['op'] == 'affine_g64' for d in r['device'])
    require(affine > 0, 'no decoded MLX affine-g64 projection work')
    require(any(f['windows'] == streams for r in selected for f in r['frames']),
            'requested stream count never appeared in one actual frame; workload was not concurrent enough')
    for key, reply in expected.items():
        require(len(committed[key]) == reply['completion_tokens'], 'trace committed-token coverage differs')
        require(token_hash(committed[key]) == reply['token_sha256'], 'trace committed IDs differ from response tokens')
    result = trace.summary((headers,selected))
    result.update(measured_rounds=len(selected), preparation_rounds_excluded=len(all_rounds)-len(selected),
                  trace_commit_ids_equal=True, actual_stream_window_seen=True)
    return result


def prompts(base, output, kind, n, streams, from_dir=None):
    output.mkdir(exist_ok=True)
    result = []
    for index in range(streams):
        path = output/f'{kind}-{n}-{index}.json'
        if not path.exists() and from_dir:
            path.write_bytes((from_dir/'prompts'/path.name).read_bytes())
        if not path.exists():
            seed = (f'Case {index}. Explain and improve this program.\nfor i in range(100):\n    print(i*i)\n'
                    if kind == 'code' else f'Case {index}. User: Help plan a day of walking, food and museums.\n'
                    'Assistant: What matters most?\nUser: Compare several choices and explain the tradeoffs.\n')
            unit = request(base,'/v1/tokenize',dict(prompt=seed,add_special_tokens=False))['tokens']
            require(unit and all(count(x) and x < 2**32 for x in unit), 'invalid tokenization')
            write(path,(unit*((n+len(unit)-1)//len(unit)))[:n])
        ids = json.loads(path.read_text())
        require(len(ids) == n and all(count(x) and x < 2**32 for x in ids), 'invalid saved prompt IDs')
        result.append(ids)
    return result


def policy_receipt(text):
    require('TF_POLICY_ERROR' not in text, 'policy clock failed')
    configs = [json.loads(line[len('TF_POLICY_CONFIG '):]) for line in text.splitlines()
               if line.startswith('TF_POLICY_CONFIG ')]
    events = [json.loads(line[len('TF_POLICY '):]) for line in text.splitlines() if line.startswith('TF_POLICY ')]
    ends = re.findall(r'^TF_POLICY_END complete=true events=([0-9]+)$',text,re.M)
    require(len(configs) == len(ends) == 1 and events and int(ends[0]) == len(events),
            'missing, partial or unconsumed policy tape')
    require(events[-1]['kind'] == 'done', 'policy tape ends inside a round')
    for event in events:
        require(event['kind'] in ('round','start','elapsed','verify','frame','commit','done')
                and event['mark'] in ('cost','overhead') and type(event['key']) is int
                and 0 <= event['key'] < 2**64 and type(event['ms']) in (int,float)
                and math.isfinite(event['ms']) and event['ms'] >= 0, 'invalid policy event')
    require(configs[0]['timed'] is True, 'adaptive timing must remain enabled')
    return dict(config=configs[0],events=events)


def arm(args, kind, n, streams, repetition, enabled, policy=None, calibration=False):
    require(not args.server_arg, 'server-arg is disabled for this fixed loopback recipe')
    require(not enabled or policy is not None, 'tracing requires the untraced policy tape')
    require(calibration == (policy is None), 'choose calibration or replay explicitly')
    dest = args.output/f'{kind}-n{n}-s{streams}-r{repetition}-t{enabled}{"-cal" if calibration else ""}'
    dest.mkdir()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    cmd = [str(args.binary.resolve()),'serve',str(args.model_dir.resolve()),'--backend','cuda',
           '--host','127.0.0.1','--port',str(port),'--name','round-check','--no-update-check','--no-thinking',
           '--context',str(n+args.max_tokens+256),'--parallel',str(streams),
           '--prompt-cache-gib',str(args.cache_gib),'--drafter',str(args.drafter.resolve())]
    nonce = secrets.token_hex(32)
    env = {**os.environ,'TF_CUDA_ROUND_TRACE':str(enabled),'TF_CUDA_ROUND_CONDITIONS':args.conditions,'TF_CUDA_ROUND_HEALTH_NONCE':nonce}
    env['TF_QWEN_KV_PAGES'] = '16'
    env['TF_QWEN_KV_SHARE'] = str(4 * (streams + 1))
    env.pop('TF_CUDA_ROUND_POLICY_RECORD',None);env.pop('TF_CUDA_ROUND_POLICY_REPLAY',None)
    env['TF_CUDA_ROUND_POLICY_RECORD' if calibration else 'TF_CUDA_ROUND_POLICY_REPLAY'] = '1' if calibration else str(policy.resolve())
    log = dest/'server.log'
    with log.open('wb') as output:
        process = subprocess.Popen(cmd,env=env,stdout=output,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+args.start_timeout
            while True:
                remaining(deadline)
                require(process.poll() is None,'server exited before readiness')
                try:
                    health = request(base,'/health',deadline=deadline,timeout=min(2,remaining(deadline)))
                except (OSError,ValueError):
                    time.sleep(min(.25,remaining(deadline)))
                    continue
                observed = health.get('nonce') if isinstance(health,dict) else None
                require(type(observed) is str and observed.isascii() and secrets.compare_digest(observed,nonce),
                        'spawned server ownership nonce mismatch')
                break
            ids = prompts(base,args.output/'prompts',kind,n,streams,args.prompts_from)
            prepared = [completion(base,p[:-1],1,args.stream_timeout) for p in ids]
            write(dest/'prepared.json',prepared)
            barrier = threading.Barrier(streams)
            with ThreadPoolExecutor(max_workers=streams) as pool:
                jobs = [pool.submit(completion,base,p,args.max_tokens,args.stream_timeout,barrier) for p in ids]
                replies = [job.result(timeout=args.stream_timeout+30) for job in jobs]
            require(all(r['cached_tokens'] >= n-16 for r in replies),
                    'prepared prefixes were not restored; do not call this the prepared-context decode cell')
            require(all(r['speculative']['drafted'] > 0 for r in replies)
                    and sum(r['speculative']['accepted'] for r in replies) > 0, 'no active accepted DFlash2 proposals')
            result = dict(kind=kind,length=n,streams=streams,trace=enabled,repetition=repetition,
                          cache_gib=args.cache_gib,max_tokens=args.max_tokens,replies=replies,command=cmd,
                          binary_sha256=sha(args.binary.read_bytes()),source=args.source,
                          source_hashes={name:sha((Path(__file__).parent/name).read_bytes()) for name in
                                         ('cuda_round_device.py','cuda_round_attribution.py','cuda_trace_http.py',
                                          'bench_stream.py','bench_evidence.py')})
        finally:
            process.terminate()
            try: process.wait(timeout=120)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
    text = log.read_text(errors='replace')
    observed_policy = policy_receipt(text)
    if calibration:
        policy = dest/'policy.json'
        write(policy,observed_policy)
    else:
        require(observed_policy == json.loads(policy.read_text()), 'policy observations or graph/window workloads differ')
    result.update(policy_sha256=sha(policy.read_bytes()),policy_file=str(policy))
    require(re.search(r'drafts: [^\n]*DFlash2',text) is not None,'server did not identify DFlash2')
    if enabled:
        analysis = decode_rounds(text,result['binary_sha256'],replies,streams,args.source,result['policy_sha256'])
        write(dest/'rounds.json',analysis)
        table = []
        for row in analysis['per_round']:
            values = dict(round=row['round'],complete_round_wall_ns=row['budget']['complete_round_wall_ns'],
                          functional_return_ns=row['budget']['functional_return_ns'],
                          host_outside_named_spans_ns=row['host_outside_named_spans_ns'],
                          device_envelope_gap_ns=row['budget']['device_envelope_gap_ns'],
                          attribution_drain_union_ns=row['budget']['attribution_drain_union_ns'])
            for domain in ('device','host'):
                values.update({f'{domain}_{key}_union_ns':value for key,value in row[f'{domain}_stage_union_ns'].items()})
            table.append(values)
        with (dest/'rounds.csv').open('w') as output:
            writer = csv.DictWriter(output,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    else:
        require('TF_ROUND ' not in text and 'TF_ROUND_IDENTITY ' not in text and 'TF_ROUND_ERROR ' not in text,
                'trace-off emitted attribution output')
    write(dest/'receipt.json',result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True); p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--drafter',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--conditions',required=True)
    p.add_argument('--source',required=True,help='expected full COMMIT:TREE embedded in this binary')
    p.add_argument('--lengths',type=lambda s:list(map(int,s.split(','))),default=[4096,32768])
    p.add_argument('--streams',type=lambda s:list(map(int,s.split(','))),default=[1,4])
    p.add_argument('--kinds',type=lambda s:s.split(','),default=['code','chat'])
    p.add_argument('--max-tokens',type=int,default=128); p.add_argument('--reps',type=int,default=2)
    p.add_argument('--cache-gib',type=int,default=24); p.add_argument('--prompts-from',type=Path)
    p.add_argument('--start-timeout',type=float,default=1800); p.add_argument('--stream-timeout',type=float,default=3600)
    p.add_argument('--server-arg',action='append',default=[])
    args = p.parse_args()
    require(re.fullmatch('[0-9a-f]{40}:[0-9a-f]{40}',args.source) is not None,'invalid expected source')
    require(args.lengths and all(n >= 32 for n in args.lengths),'invalid context lengths')
    require(args.streams and set(args.streams) <= {1,4},'this matrix covers1/4 streams')
    require(args.kinds and set(args.kinds) <= {'code','chat'},'invalid prompt kinds')
    require(args.max_tokens > 0 and args.reps > 0 and args.cache_gib > 0,'invalid generation/cache/repetitions')
    require(all(math.isfinite(v) and v > 0 for v in (args.start_timeout,args.stream_timeout)),'invalid deadline')
    require(not args.server_arg, 'server-arg is disabled for this fixed loopback recipe')
    args.output.mkdir(parents=True,exist_ok=True)
    require(not any(args.output.iterdir()),'use a new output directory')
    pairs = []
    for kind in args.kinds:
        for n in args.lengths:
            for streams in args.streams:
                for rep in range(args.reps):
                    order = (0,1) if rep%2 == 0 else (1,0)
                    calibration = arm(args,kind,n,streams,rep,0,calibration=True)
                    policy = Path(calibration['policy_file'])
                    arms = {v:arm(args,kind,n,streams,rep,v,policy=policy) for v in order}
                    compare(calibration,arms[0])
                    compare(arms[0],arms[1])
                    pairs.append(dict(kind=kind,length=n,streams=streams,repetition=rep,token_counter_equal=True,policy_clock_equal=True))
                    write(args.output/'pairs.json',pairs)
    print('PASS: requested prepared-context decode cells, token/counter pairs and actual round receipts checked')


if __name__ == '__main__':
    main()
