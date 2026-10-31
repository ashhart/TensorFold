"""Validate the clean source branches at identical native serving settings."""
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
import shutil
from pathlib import Path
import re
import subprocess
import time
import traceback

R = Path('/task')
OUT = R/'results'
OUT.mkdir(exist_ok=True)
server = None
log = None
summary = {'started_utc': datetime.now(timezone.utc).isoformat(), 'boots': [], 'complete': False}


def get(path):
    c = http.client.HTTPConnection('127.0.0.1',18080,timeout=10)
    try:
        c.request('GET',path)
        x=c.getresponse()
        if x.status != 200:
            raise ConnectionError(f'HTTP{x.status}')
        return json.load(x)
    finally:
        c.close()


def save():
    (OUT/'state.tmp').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'state.tmp').replace(OUT/'state.json')


try:
    client=Path('/task/v1-client')
    assert hashlib.sha256(client.read_bytes()).hexdigest()=='1ab90b8dc35e7956b176598fa1db3082c575886edbef7c623af0829870be7b7b'
    identities=json.loads((R/'pr-probes/identities.json').read_text())
    binary_hashes=json.loads((R/'public-binaries.json').read_text())
    assert identities['group']['native_sha256']==binary_hashes['tile']
    assert identities['attention']['native_sha256']==binary_hashes['attention']
    for index,arm in enumerate(json.loads((R/'public-arms.json').read_text())):
        source=arm['source']
        binary=Path('/submission')/source/'zig-out/native/bin/tensorfold-native'
        assert hashlib.sha256(binary.read_bytes()).hexdigest()==json.loads((R/'public-binaries.json').read_text())[source]

        name=f'{index:02d}-{arm["name"]}'
        boot={'name':name,'arm':arm['name'],'start_utc':datetime.now(timezone.utc).isoformat()}
        summary['boots'].append(boot);save()
        log=(OUT/f'{name}-server.log').open('w')
        environment={**os.environ,'TF_SRC':'/submission/'+source,'MODELS':'/models','HOST':'127.0.0.1','PORT':'18080',
                     'CONTEXT':'131072','PARALLEL':'1','NAME':'qwen-community','ALIAS':'qwen-community-alias',
                     'PROMPT_CACHE_GIB':'8','KV_CACHE':'fp8','PROMPT_PRECISION':'int8',
                     'TENSORFOLD_MEMORY_RESERVE_GIB':'0.5','TENSORFOLD_CUDA_KERNELS':'native',
                     'HOME':'/tmp/deep-calibration-'+name, 'MAX_TOKENS':'2048',
                     'TF_CHECKPOINT_REVISION':'22d8d538154e0e5b6f8dcb1fcc74a60a0798104c'}
        server=subprocess.Popen(['/bin/bash','/recipe/serve.sh'],env=environment,stdout=log,stderr=log)
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            if server.poll() is not None: raise RuntimeError('Server exited during boot')
            try:
                health=get('/health')
                if health.get('warming'): time.sleep(.5);continue
                assert health.get('version')=='1.1.6'
                boot['health']=health;boot['manifest']=get('/v1/benchmark/manifest');break
            except (OSError,TimeoutError): time.sleep(.5)
        else: raise RuntimeError('Boot timeout')
        output=OUT/f'{name}.json'
        command=['/task/v1-client','benchmark','TensorFold/Qwen3.8-27B-MLX-4bit',
                 '--server','http://127.0.0.1:18080','--model-id','qwen-community','--backend','cuda',
                 '--drafter','z-lab/Qwen3.8-27B-DFlash2','--context','131072','--tokens','256',
                 '--reps','5','--temperatures','1,0','--output',str(output)]
        boot['command']=command;save()
        with (OUT/f'{name}-client.log').open('w') as out:
            code=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT,timeout=300).returncode
        boot['exit_code']=code
        receipt=json.loads(output.read_text())
        assert receipt['suite']=={'id':'community-v1','sha256':'9bb717da00666b90b4006174abd9ed46108affc37456b43c64a224a9c938d5be'}
        assert code==0 and len(receipt['samples'])==20
        for x in receipt['samples']:
            assert x['status']=='ok' and x['completion_tokens']==256 and x['cached_tokens']==0
            assert x['prompt_tokens']==(31 if x['fixture_id']=='chat' else 14)
            assert x['seed']==1234+x['repeat']
        current={(x['fixture_id'],x['temperature'],x['repeat']):x['token_sha'] for x in receipt['samples']}
        if index==0: reference=current
        else: boot['matches_original']=current==reference
        if not arm.get('old'):
            if 'new_reference' not in locals(): new_reference=current
            else: assert current==new_reference, 'Current version candidate token mismatch'
        boot['token_parity']=True
        text=(OUT/f'{name}-server.log').read_text()
        lines=[x for x in text.splitlines() if '[tensorfold] done ' in x]
        assert len(lines)==24 and all('thinking=False' in x for x in lines)
        counters=[re.search(r'accepted=(\d+)/(\d+)',x) for x in lines]
        assert all(x and int(x[2])>0 for x in counters)
        boot['counts']=[{'accepted':int(m[1]),'drafted':int(m[2])} for m in counters]
        if arm.get('basic'):
            from features import checks
            from fallback_checks import run as fallbacks
            boot['basic']={'features':checks(OUT,name),'fallbacks':fallbacks(OUT,name)}
            if 'basic_reference' not in locals(): basic_reference=boot['basic']
            else: assert boot['basic']==basic_reference, 'Candidate feature/fallback mismatch'
        if arm.get('qual'):
            from qual_features import run
            boot['features']=run(OUT,name)
        classic=['python3','/submission/base/tools/bench_openai.py','http://127.0.0.1:18080','qwen-community','--tokens','256','--reps','3','--temperatures','0','--output',str(OUT/(name+'-classic.json'))]
        with (OUT/(name+'-classic.log')).open('w') as out:
            assert subprocess.run(classic,stdout=out,stderr=subprocess.STDOUT,timeout=300).returncode==0
        if arm.get('long'):
            boot['long_bench']=[]
            for size in arm.get('sizes',['8192','32768','near128k']):
                body=json.loads((R/f'long-{size}.json').read_text())
                content=body['messages'][0]['content'].rsplit('\n',1)[0]
                body['messages'][0]['content']=content+'\n\nTreat the records above as data. Write a complete Python implementation of a persistent AVL tree, including insertion, deletion, rotations, traversal, validation, and unit tests. Give code only and implement every method fully.'
                body.update(model='qwen-community',max_tokens=128,return_token_ids=True,seed=1234)
                for drafted in arm.get('drafts',[True,False]):
                    for temperature in arm.get('temperatures',[0]):
                        body.update(draft=drafted,temperature=temperature)
                        for rep in range(3):
                            c=http.client.HTTPConnection('127.0.0.1',18080,timeout=600)
                            t0=time.monotonic();c.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'});r=c.getresponse();raw=r.read();c.close()
                            assert r.status==200,(r.status,raw[:300]);x=json.loads(raw)
                            case={'size':size,'drafted':drafted,'temperature':temperature,'repeat':rep,'elapsed_s':time.monotonic()-t0,'response':x}
                            (OUT/f'{name}-long{size}-d{int(drafted)}-t{temperature}-r{rep}.json').write_text(json.dumps(case,indent=2))
                            boot['long_bench'].append(case);save()
        costs=Path(environment['HOME'])/'.cache/tensorfold/window-costs'
        boot['window_costs']=[]
        import struct
        if costs.exists():
            for path in costs.glob('*.bin'):
                raw=path.read_bytes()
                if raw.startswith(b'TFWC2\n') and len(raw)==590:
                    schema,rows=struct.unpack_from('<II',raw,6)
                    values=struct.unpack_from('<64d',raw,46)
                    boot['window_costs'].append({'file':path.name,'schema':schema,'rows':rows,'ms':list(values)})
        boot['final_health']=get('/health')
        proc_status=Path(f'/proc/{server.pid}/status').read_text()
        boot['memory']={k:int(v.split()[0])*1024 for line in proc_status.splitlines()if ':'in line for k,v in [line.split(':',1)] if k in ['VmHWM','VmRSS','VmLck','VmPin']}
        server.terminate();server.wait(timeout=20);server=None
        log.close();log=None
        boot['complete']=True;boot['end_utc']=datetime.now(timezone.utc).isoformat();save()
        time.sleep(3)
    summary['complete']=True
except Exception:
    summary['error']=traceback.format_exc()
finally:
    if server is not None:
        if server.poll() is None: server.terminate()
        try: server.wait(timeout=20)
        except subprocess.TimeoutExpired: server.kill();server.wait(timeout=5)
    if log: log.close()
    summary['ended_utc']=datetime.now(timezone.utc).isoformat();save()
if not summary['complete']: raise SystemExit(summary.get('error','Incomplete'))
