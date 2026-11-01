import concurrent.futures as cf
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(os.environ.get('TF_QUAL_OUT', '/tmp/tf1394-evidence'))
OUT.mkdir(exist_ok=True)
HUB = Path.home()/'.cache/huggingface/hub'
MODEL = Path(os.environ.get('TF_QUAL_MODEL', HUB/'models--LiquidAI--LFM2.5-24B-A2B-MLX-4bit/snapshots/667408d494bde191c2cb9dd4f4549827816b9275'))
DRAFT = Path(os.environ.get('TF_QUAL_DRAFTER', Path.home()/'models/LiquidAI--LFM2.5-24B-A2B-DSpark'))
CAND = Path(os.environ.get('TF_QUAL_CANDIDATE', ROOT/'zig-out/native/bin/tensorfold-native'))
MAIN = Path(os.environ.get('TF_QUAL_MAIN', '/tmp/tf1394-main-base/zig-out/native/bin/tensorfold-native'))
BASE = 'http://127.0.0.1:18194'
PROMPTS = json.loads((ROOT/'tools/bench_prompts.json').read_text())[:8]

def emit(kind, **data):
    print(json.dumps({'event':kind, **data}), flush=True)

def save(name, data):
    (OUT/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')

def post(body, stream=False, cancel=None):
    body = {'model':'review', 'max_tokens':128, 'temperature':0, 'seed':1234,
            'ignore_eos':True, 'return_token_ids':True, **body, 'stream':stream}
    if stream: body['stream_options']={'include_usage':True}
    chat='messages' in body
    if chat: body['chat_template_kwargs']={'enable_thinking':False}
    endpoint='/v1/chat/completions' if chat else '/v1/completions'
    req = urllib.request.Request(BASE+endpoint, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    sent = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        if not stream:
            x=json.load(r)
            content=x['choices'][0].get('message',{}).get('content',x['choices'][0].get('text',''))
            return {'usage':x['usage'], 'runtime':x['tensorfold'], 'speculative':x.get('speculative',{}), 'content':content, 'wall':time.perf_counter()-sent}
        usage={}; runtime={}; spec={}; first=None; last=None; chunks=0; done=False
        for raw in r:
            line=raw.decode().strip()
            if line=='data: [DONE]': done=True; break
            if not line.startswith('data:'): continue
            x=json.loads(line[5:])
            if 'error' in x: raise RuntimeError(str(x['error']))
            usage=x.get('usage') or usage; runtime=x.get('tensorfold') or runtime; spec=x.get('speculative') or spec
            if any(c.get('text') or c.get('delta',{}).get('content') for c in x.get('choices',[])):
                now=time.perf_counter(); first=first or now; last=now; chunks+=1
                if cancel and chunks>=cancel: return {'cancelled':True}
        if not done or not runtime.get('token_sha256'): raise RuntimeError('incomplete stream')
        return {'usage':usage, 'runtime':runtime, 'speculative':spec, 'sent':sent, 'first':first, 'last':last, 'chunks':chunks}

class Server:
    def __init__(self, label, binary=CAND, mode='bf16', context=16384, parallel=16, cache=0, validation=False, model=MODEL, draft=DRAFT):
        self.label=label; self.binary=binary; self.mode=mode; self.context=context; self.parallel=parallel
        self.cache=cache; self.validation=validation; self.model=model; self.draft=draft
    def __enter__(self):
        processes=subprocess.check_output(['ps','-Ao','pid=,comm='],text=True)
        busy=[line for line in processes.splitlines() if line.split() and Path(line.split()[-1]).name in ('tensorfold-native','tf-lfm2-exact','tf-lfm2-dspark')]
        if busy: raise RuntimeError('another model process is already running')
        args=[str(self.binary),'serve',str(self.model),'--name','review','--host','127.0.0.1','--port','18194','--parallel',str(self.parallel),'--context',str(self.context),'--prompt-cache-gib',str(self.cache),'--keep-warm','900']
        if self.mode!='plain': args+=['--drafter',str(self.draft)]
        if self.mode=='8': args+=['--drafter-bits','8']
        env=dict(os.environ)
        if self.validation: env['MTL_DEBUG_LAYER']='1'
        self.log=open(OUT/(self.label+'.server.log'),'w')
        self.proc=subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        start=time.monotonic()
        while time.monotonic()-start<600:
            if self.proc.poll() is not None: raise RuntimeError(self.label+' server exited')
            try:
                with urllib.request.urlopen(BASE+'/v1/models',timeout=1) as r: json.load(r)
                emit('ready', label=self.label, seconds=time.monotonic()-start)
                return self
            except Exception: time.sleep(.5)
        self.__exit__(None,None,None)
        raise TimeoutError('load timeout')
    def __exit__(self,*args):
        started=time.monotonic(); forced=False
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try: self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                forced=True; os.killpg(self.proc.pid,signal.SIGKILL); self.proc.wait()
        self.log.close()
        save(self.label+'.shutdown',{'forced_kill':forced,'seconds':time.monotonic()-started,'returncode':self.proc.returncode})
        if forced: raise RuntimeError(self.label+' required forced shutdown')

def digest(r): return r['runtime']['token_sha256']

def functional():
    results={}
    first_messages=[{'role':'user','content':'Public cache fixture.\n'+('alpha beta gamma delta\n'*700)+'\nExplain matrix multiplication in one paragraph.'}]
    with Server('functional-cache',cache=1,validation=True):
        first=post({'messages':first_messages,'max_tokens':64})
        messages=first_messages+[{'role':'assistant','content':first['content']},{'role':'user','content':'Give a small numerical example and explain each step.'}]
        resumed=post({'messages':messages,'max_tokens':64})
        if resumed['usage'].get('prompt_tokens_details',{}).get('cached_tokens',resumed['usage'].get('cached_tokens',0))<=0:
            raise AssertionError('resume did not report reused tokens: '+str(resumed['usage']))
        results['resumed']=resumed; results['resume_request']={'messages':messages,'max_tokens':64}
        refs=[post({'prompt':p['prompt'],'max_tokens':64}) for p in PROMPTS]
        results['stress_reference_hashes']=[{'prompt':p['name'],'sha256':digest(r)} for p,r in zip(PROMPTS,refs)]
        results['stress']=[]
        for slots,seconds in [(6,60),(16,90)]:
            end=time.monotonic()+seconds
            def worker(i):
                rng=random.Random(4242+i); counts={'completed':0,'cancelled':0,'wrong':0,'errors':[]}
                while time.monotonic()<end:
                    j=rng.randrange(len(PROMPTS)); cancel=rng.randint(1,12) if rng.random()<.35 else None
                    try:
                        r=post({'prompt':PROMPTS[j]['prompt'],'max_tokens':64},True,cancel)
                        if r.get('cancelled'): counts['cancelled']+=1
                        else:
                            counts['completed']+=1
                            counts['wrong']+=digest(r)!=digest(refs[j])
                    except Exception as e: counts['errors'].append(type(e).__name__+': '+str(e))
                return counts
            with cf.ThreadPoolExecutor(slots) as ex: rows=list(ex.map(worker,range(slots)))
            summary={'slots':slots,'seconds':seconds,**{k:sum(x[k] for x in rows) for k in ['completed','cancelled','wrong']},'errors':[e for x in rows for e in x['errors']]}
            results['stress'].append(summary); save('functional',results); emit('stress',**summary)
            assert summary['wrong']==0 and not summary['errors'] and summary['cancelled']>0
        sampled=[]
        for n in [1,4,8,16]:
            bodies=[{'prompt':PROMPTS[i%len(PROMPTS)]['prompt'],'temperature':.7,'top_p':.95,'seed':4242+i,'max_tokens':64} for i in range(n)]
            plain=[post({**b,'draft':False}) for b in bodies]
            solo=[post(b) for b in bodies]
            gate=threading.Barrier(n)
            def run(b): gate.wait(); return post(b)
            with cf.ThreadPoolExecutor(n) as ex: together=list(ex.map(run,bodies))
            row={'streams':n,'hashes':[[digest(p),digest(s),digest(t)] for p,s,t in zip(plain,solo,together)]}
            sampled.append(row); emit('sampled',streams=n,equal=sum(len(set(x))==1 for x in row['hashes']))
            assert all(len(set(x))==1 for x in row['hashes'])
        results['sampled']=sampled; save('functional',results)
    with Server('functional-fresh'):
        fresh=post({'messages':messages,'max_tokens':64})
        assert digest(fresh)==digest(results['resumed'])
        results['fresh']=fresh; results['resume_equal']=True; save('functional',results); emit('resume',equal=True)

def long_context():
    results=[]
    with Server('long-context',parallel=1,context=131072):
        tail=[825,264,15352,4829,9944,13,271,279,854,22373]
        for length in [4096,16384,65536,130816]:
            ids=(tail*((length+9)//10))[:length]
            plain=post({'prompt':ids,'max_tokens':64,'draft':False})
            drafted=post({'prompt':ids,'max_tokens':64})
            row={'length':length,'plain':plain,'drafted':drafted,'equal':digest(plain)==digest(drafted)}
            results.append(row); save('long-context',results); emit('long-context',length=length,equal=row['equal'])
            assert row['equal']
    for length in [65536,130816]:
        emit('long-reference-start',length=length)
        with open(OUT/('dspark-reference-'+str(length)+'.log'),'w') as log:
            r=subprocess.run([str(ROOT/'zig-out/bin/tf-lfm2-dspark'),str(MODEL),str(DRAFT),str(length)],stdout=log,stderr=subprocess.STDOUT,timeout=3600)
        emit('long-reference-done',length=length,exit=r.returncode)
        assert r.returncode==0

def performance(family='lfm2', compare=False):
    model=MODEL; draft=DRAFT; widths=[1,4,8,16]
    if family=='qwen':
        model=HUB/'models--TensorFold--Qwen3.8-27B-MLX-4bit/snapshots/22d8d538154e0e5b6f8dcb1fcc74a60a0798104c'
        draft=HUB/'models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4'
        widths=[1,4]
    prompts=[{'name':'code','prompt':'Write a short Python function that computes the Fibonacci sequence and explain it.'},{'name':'chat','prompt':'Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.'}]
    arms={'main-plain':(MAIN,'plain'),'candidate-plain':(CAND,'plain'),'candidate-drafted':(CAND,'bf16')}
    if family=='qwen': arms={'main-drafted':(MAIN,'bf16'),'candidate-drafted':(CAND,'bf16')}
    if compare:
        published=os.environ.get('TF_QUAL_PUBLISHED')
        if not published: raise SystemExit('compare requires TF_QUAL_PUBLISHED pointing to the earlier PR binary')
        arms={'main':(MAIN,'plain'),'published':(Path(published),'bf16'),'candidate':(CAND,'bf16')}
    results=[]; reference={}
    order=list(arms)
    for round_ in range(3):
        rotated=order[round_%len(order):]+order[:round_%len(order)]
        for arm in rotated:
            binary,mode=arms[arm]
            with Server(f'perf-{family}-{round_}-{arm}',binary,mode,4096,max(widths),0,False,model,draft):
                for p in prompts: post({'prompt':p['prompt'],'max_tokens':32})
                for p in prompts:
                    for width in widths:
                        gate=threading.Barrier(width)
                        def run(i): gate.wait(); return post({'prompt':p['prompt'],'max_tokens':256},True)
                        with cf.ThreadPoolExecutor(width) as ex: rows=list(ex.map(run,range(width)))
                        key=p['name']; hashes=[digest(x) for x in rows]
                        if key not in reference: reference[key]=hashes[0]
                        assert all(h==reference[key] for h in hashes),(family,arm,key,'token mismatch')
                        first=min(x['first'] for x in rows); last=max(x['last'] for x in rows)
                        row={'round':round_,'arm':arm,'prompt':key,'streams':width,'aggregate_decode_tps':width*255/(last-first),'ttft_max':max(x['first']-x['sent'] for x in rows),'requests':rows}
                        results.append(row); save(('comparison-' if compare else 'performance-')+family,results)
                        emit('performance',family=family,round=round_,arm=arm,prompt=key,streams=width,tps=round(row['aggregate_decode_tps'],2))

def cold_prefill():
    results=[]
    arms={'main-plain':(MAIN,'plain'),'candidate-plain':(CAND,'plain'),'candidate-drafted':(CAND,'bf16')}
    order=list(arms); cycle=[825,264,15352,4829,9944,13,271,279,854,22373]
    for round_ in range(3):
        for arm in order[round_:]+order[:round_]:
            binary,mode=arms[arm]
            with Server(f'cold-{round_}-{arm}',binary,mode,131072,1):
                for length in [2048,8192,32768,65536]:
                    ids=(cycle*((length+9)//10))[:length]
                    r=post({'prompt':ids,'max_tokens':1})
                    assert r['usage']['prompt_tokens']==length
                    assert r['usage']['prompt_tokens_details']['cached_tokens']==0
                    row={'round':round_,'arm':arm,'prompt_tokens':length,**r}
                    results.append(row); save('cold-prefill',results)
                    emit('cold-prefill',round=round_,arm=arm,length=length,ttft=r['runtime']['time_to_first_token'])

if __name__=='__main__':
    phase=sys.argv[1]
    with open('/tmp/tensorfold-model.lock','a') as lock:
        emit('waiting-for-model-lock',phase=phase)
        fcntl.flock(lock,fcntl.LOCK_EX)
        emit('starting',phase=phase)
        if phase=='functional': functional()
        elif phase=='long': long_context()
        elif phase=='performance': performance(sys.argv[2] if len(sys.argv)>2 else 'lfm2')
        elif phase=='compare': performance(compare=True)
        elif phase=='cold': cold_prefill()
