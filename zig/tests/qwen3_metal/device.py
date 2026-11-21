#!/usr/bin/env python3
"""Qwen3 embedding server qualification, executed only in the operator's device queue."""

import argparse, base64, concurrent.futures, hashlib, http.client, json, math, os, socket, struct, subprocess, time
from pathlib import Path
from reference import Reference
TEXTS=[
    'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: What is photosynthesis?',
    'The quick brown fox jumps over the lazy dog.',
    '地球绕太阳公转。Water freezes at zero degrees Celsius. Café ☕',
]
COSINE_DISTANCE=0.001
class Server:
    def __init__(self,binary,checkpoint,out):
        self.out=out;self.port=18090
        with socket.socket() as s:assert s.connect_ex(('127.0.0.1',self.port))!=0,'port occupied'
        self.argv=[str(binary),'serve',str(checkpoint),'--host','127.0.0.1','--port',str(self.port),'--parallel','1','--context','4096','--prompt-cache-gib','0','--no-update-check']
        self.log=(out/'server.log').open('w');self.p=subprocess.Popen(self.argv,stdout=self.log,stderr=subprocess.STDOUT)
    def __enter__(self):
        for _ in range(600):
            if self.p.poll() is not None:break
            try:
                status,reply=self.ask('GET','/v1/models');assert status==200
                self.model=reply['data'][0]['id'];return self
            except (OSError,ValueError,AssertionError,KeyError):time.sleep(1)
        self.__exit__(None,None,None);raise RuntimeError('server startup failed; read '+str(self.out/'server.log'))
    def __exit__(self,*_):
        if self.p.poll() is None:
            self.p.terminate()
            try:self.p.wait(timeout=30)
            except subprocess.TimeoutExpired:self.p.kill();self.p.wait(timeout=10)
        self.log.close()
    def ask(self,method,path,body=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.port,timeout=600)
        try:
            conn.request(method,path,None if body is None else json.dumps(body),{'Content-Type':'application/json'})
            reply=conn.getresponse();return reply.status,json.loads(reply.read())
        finally:conn.close()
    def post(self,path,body):
        status,data=self.ask('POST',path,body)
        assert status==200,(path,status,data)
        return data
    def embed(self,inputs,dimensions=None,encoding='base64'):
        body={'model':self.model,'input':inputs,'encoding_format':encoding}
        if dimensions is not None:body['dimensions']=dimensions
        t=time.monotonic();reply=self.post('/v1/embeddings',body);seconds=time.monotonic()-t
        rows=[base64.b64decode(x['embedding'],validate=True) if encoding=='base64' else struct.pack('<'+'f'*len(x['embedding']),*x['embedding']) for x in reply['data']]
        return rows,seconds,reply

def vector(raw):return struct.unpack('<'+'f'*(len(raw)//4),raw)
def metrics(got,want):
    assert len(got)==len(want) and all(math.isfinite(x) for x in got+want)
    dot=sum(a*b for a,b in zip(got,want));cos=dot/math.sqrt(sum(x*x for x in got)*sum(x*x for x in want))
    return {'maxAbs':max(abs(a-b) for a,b in zip(got,want)),'cosine':cos,'cosineDistance':max(0,1-cos)}
def qualify(binary,checkpoint,library,out):
    out.mkdir(exist_ok=False);receipt=json.loads((checkpoint/'VERIFIED.json').read_text());config=json.loads((checkpoint/'config.json').read_text());hidden=config['hidden_size']
    result={'checkpoint':checkpoint.name,'revision':receipt['revision'],'dimensions':hidden,'cosineDistanceLimit':COSINE_DISTANCE,'status':'RUNNING','tests':[]}
    def save():(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        with Server(binary,checkpoint,out) as server:
            result['serverArgv']=server.argv
            ids=[server.post('/v1/tokenize',{'prompt':t,'add_special_tokens':True})['tokens'] for t in TEXTS]
            (out/'text-ids.json').write_text(json.dumps(ids)+'\n')
            for text,tokens in zip(TEXTS,ids):
                a,_,_=server.embed(text);b,_,_=server.embed(tokens);assert a==b,'text/ID mismatch'
            # Cross all row-kernel and attention cutovers, and one full prompt chunk plus its last row.
            shapes=[1,16,17,32,33,129,2049]
            cuts=[[(i*7919+17)%140000 for i in range(n-1)]+[151643] for n in shapes]
            cases=ids+cuts
            full,elapsed,reply=server.embed(cases);assert len(full)==len(cases)
            assert reply['usage']['prompt_tokens']==sum(map(len,cases))
            result['batchSeconds']=elapsed;result['batchTokensPerSecond']=sum(map(len,cases))/elapsed
            for i,tokens in enumerate(cases):
                one,secs,_=server.embed(tokens);assert one[0]==full[i],('batch/single',i,len(tokens))
                again,_,_=server.embed(tokens);assert again==one,('repeat',i)
                flt,_,_=server.embed(tokens,encoding='float');assert flt==one,('float/base64',i)
                norm=math.sqrt(sum(x*x for x in vector(one[0])));assert abs(norm-1)<1e-5,norm
                result['tests'].append({'tokens':len(tokens),'singleSeconds':secs,'tokensPerSecond':len(tokens)/secs,'batchSingleExact':True,'repeatExact':True,'floatEncodingExact':True,'sha256':hashlib.sha256(one[0]).hexdigest()})
                save()
            for width in sorted(set([32,128,hidden//2,hidden])):
                batch,_,_=server.embed(cases,width)
                for i,tokens in enumerate(cases):
                    one,_,_=server.embed(tokens,width);assert one[0]==batch[i],('MRL batch/single',width,i)
                    got=vector(one[0]);expect=list(vector(full[i]))[:width];scale=math.sqrt(sum(v*v for v in expect));expect=[v/scale for v in expect]
                    assert max(abs(a-b) for a,b in zip(got,expect))<2e-7,('MRL crop',width,i)
                result.setdefault('mrlWidths',[]).append(width);save()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                together=list(pool.map(lambda tokens:server.embed(tokens)[0][0],cases[:4]))
            assert together==full[:4];result['concurrentSingleExact']=True
            refusals=[{'input':''},{'input':' '},{'input':[]},{'input':[-1]},{'input':[config['vocab_size']]},{'input':[[1],[]]},
                      {'input':[[1]]*129},{'input':[1]*4097},{'input':'x','dimensions':31},{'input':'x','dimensions':hidden+1},
                      {'input':'x','encoding_format':'invalid'},{'input':'x','images':['invalid']}]
            for body in refusals:
                status,why=server.ask('POST','/v1/embeddings',body);assert 400<=status<500,(status,why)
                result.setdefault('refusals',[]).append({'case':{k:(len(v) if isinstance(v,list) else v) for k,v in body.items()},'status':status,'reply':why})
            for route,body in [('/v1/completions',{'prompt':'x','max_tokens':1}),('/v1/rerank',{'query':'x','documents':['x']})]:
                status,why=server.ask('POST',route,body);assert status>=400,(route,status,why)
                result.setdefault('unsupportedRoutes',[]).append({'route':route,'status':status,'reply':why})
            # The short texts and two non-tile row counts have independent FP32 answers at every size.
            ref_indices=[0,1,2,7,8] # text cases; 33 and 129 synthetic tokens
            if hidden==1024:ref_indices.append(9) # 2048+1 checks the retained KV boundary too
            ref=Reference(checkpoint,library)
            for i in ref_indices:
                t=time.monotonic();want=ref.forward(cases[i]);m=metrics(list(vector(full[i])),want)
                (out/f'fp32-{i}.json').write_text(json.dumps({'tokens':cases[i],'vector':want,'seconds':time.monotonic()-t})+'\n')
                result.setdefault('fp32',[]).append({'case':i,'tokens':len(cases[i]),**m});save()
                assert m['cosineDistance']<=COSINE_DISTANCE,('FP32 mismatch',i,m)
            (out/'full-vectors.json').write_text(json.dumps([list(vector(x)) for x in full])+'\n')
            result['status']='PASS';save();return result
    except BaseException as e:result['status']='FAIL';result['error']=repr(e);save();raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('checkpoints',nargs='+',type=Path);o=p.parse_args()
    o.output.mkdir(exist_ok=False)
    for checkpoint in o.checkpoints:
        result=qualify(o.binary,checkpoint,o.library,o.output/checkpoint.name);print(json.dumps({'checkpoint':checkpoint.name,'status':result['status'],'fp32':result['fp32']}),flush=True)
if __name__=='__main__':main()
