import os,importlib.util,fcntl,subprocess,json,urllib.request,hashlib,threading,concurrent.futures as cf
from pathlib import Path
root=Path(os.environ.get('TF_RESTACK_ROOT',Path.cwd()))
s=importlib.util.spec_from_file_location('q',root/'docs/receipts/lfm2-dspark-review/qualification.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
q.OUT=Path(os.environ.get('TF_RESTACK_OUT','/tmp/tf-chain-restack-m4'));q.OUT.mkdir(exist_ok=True)
def pressure(label):
 if 'memory pressure:' in (q.OUT/(label+'.server.log')).read_text(errors='replace'):raise RuntimeError('memory pressure invalidates admission/cache qualification')
with open('/tmp/tensorfold-model.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 q.save('binaries',{str(n):hashlib.sha256((root.parent/('tf-dspark-'+str(n))/'zig-out/native/bin/tensorfold-native').read_bytes()).hexdigest() for n in [1392,1393,1394]})
 for n in [1392,1393]:
  with q.Server('load-'+str(n),root.parent/('tf-dspark-'+str(n))/'zig-out/native/bin/tensorfold-native','bf16',4096,1):
   pressure('load-'+str(n))
   with urllib.request.urlopen(q.BASE+'/memory') as response:q.save('load-'+str(n),json.load(response))
 q.emit('load-peaks-done')
 with open(q.OUT/'kernel.log','w') as log:subprocess.run([str(root/'zig-out/bin/tf-kernel-contract'),'--only','lfm2'],stdout=log,stderr=subprocess.STDOUT,check=True)
 q.emit('kernel-done')
 with open(q.OUT/'reference.log','w') as log:subprocess.run([str(root/'zig-out/bin/tf-lfm2-dspark'),str(q.MODEL),str(q.DRAFT),'256'],stdout=log,stderr=subprocess.STDOUT,check=True)
 q.emit('reference-done')
 # An explicit M4 serving scope: 4096 tokens, rather than the helper's default 16384.
 defaults=list(q.Server.__init__.__defaults__);defaults[2]=4096;q.Server.__init__.__defaults__=tuple(defaults)
 q.functional()
 q.emit('functional-done')
 for mode in ['8']:
  with q.Server('eight-bit',q.CAND,mode,4096,16):
   pressure('eight-bit');cells=[]
   for name,prompt in [('code','Write a short Python function that computes the Fibonacci sequence and explain it.'),('prose','Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.')]:
    oracle=q.post({'prompt':prompt,'max_tokens':64,'draft':False});solo=q.post({'prompt':prompt,'max_tokens':64})
    assert q.digest(oracle)==q.digest(solo)
    for width in [1,4,8,16]:
     gate=threading.Barrier(width)
     def run(i):gate.wait();return q.post({'prompt':prompt,'max_tokens':64})
     with cf.ThreadPoolExecutor(width) as ex:rows=list(ex.map(run,range(width)))
     assert all(q.digest(r)==q.digest(oracle) for r in rows)
     cells.append({'prompt':name,'streams':width,'oracle':oracle,'solo':solo,'concurrent':rows});q.save('eight-bit-equality',cells)
     q.emit('eight-bit',prompt=name,streams=width,equal=True)
    pressure('eight-bit')
 q.emit('done')
