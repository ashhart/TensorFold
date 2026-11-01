import importlib.util,os,fcntl,threading,concurrent.futures as cf,hashlib,re,json
from pathlib import Path
s=importlib.util.spec_from_file_location('q',os.environ['TF_QUAL_HELPER']);q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
q.OUT=Path(os.environ.get('TF_REPORT_OUT','/tmp/tf1381-device-report'));q.OUT.mkdir(exist_ok=True)
arms={'main':(Path(os.environ['TF_REPORT_MAIN']),'plain'),'standalone':(Path(os.environ['TF_REPORT_STANDALONE']),'plain'),'dspark-integration':(Path(os.environ['TF_REPORT_INTEGRATION']),'bf16')}
prompts={'code':'Write a short Python function that computes the Fibonacci sequence and explain it.','prose':'Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.'}
results=[];refs={}
def batch(prompt,width):
 gate=threading.Barrier(width)
 def run(i):gate.wait();return q.post({'prompt':prompt,'max_tokens':256},True)
 with cf.ThreadPoolExecutor(width) as ex:return list(ex.map(run,range(width)))
def verify(name,rows):
 ref=refs.setdefault(name,q.digest(rows[0]));assert all(q.digest(r)==ref for r in rows)
def pressure(label):
 if 'memory pressure:' in (q.OUT/(label+'.server.log')).read_text(errors='replace'):raise RuntimeError('memory pressure invalidates this cell')
with open('/tmp/tensorfold-model.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 q.save('binaries',{a:hashlib.sha256(b.read_bytes()).hexdigest() for a,(b,_) in arms.items()})
 for rep in range(4):
  names=list(arms);offset=rep%len(names);names=names[offset:]+names[:offset]
  for arm in names:
   binary,mode=arms[arm];label=f'{rep}-{arm}'
   with q.Server(label,binary,mode,4096,16):
    pressure(label)
    for name,prompt in prompts.items():
     q.post({'prompt':prompt,'max_tokens':32})
     if rep == 0 and arm == 'dspark-integration':
      oracle=q.post({'prompt':prompt,'max_tokens':256,'draft':False});verify(name,[oracle]);q.save(name+'-draft-false-oracle',oracle)
     for width in [4,8,16]:
      rows=batch(prompt,width);verify(name,rows);pressure(label)
      item={'round':rep,'arm':arm,'prompt':name,'streams':width,'tps':width*255/(max(r['last'] for r in rows)-min(r['first'] for r in rows)),'requests':rows};results.append(item);q.save('comparison',results);q.emit('cell',**{k:v for k,v in item.items() if k!='requests'})
 diagnostic=Path(os.environ['TF_REPORT_DIAGNOSTIC'])
 q.save('diagnostic-binary',{'sha256':hashlib.sha256(diagnostic.read_bytes()).hexdigest()})
 counters=[];label='hold-retention'
 with q.Server(label,diagnostic,'bf16',4096,16):
  path=q.OUT/(label+'.server.log')
  for name,prompt in prompts.items():
   q.post({'prompt':prompt,'max_tokens':32})
   for width in [4,8,16]:
    start=path.stat().st_size;rows=batch(prompt,width);verify(name,rows);pressure(label)
    text=path.read_bytes()[start:].decode(errors='replace')
    events=[dict(zip(['streams','offered','kept'],map(int,m))) for m in re.findall(r'TF_HOLD_PRICE streams=(\d+) offered=(\d+) kept=(\d+)',text)]
    offered=sum(x['offered'] for x in events);kept=sum(x['kept'] for x in events)
    assert events and offered>0
    item={'prompt':name,'requested_streams':width,'offered':offered,'kept':kept,'fraction':kept/offered,'events':events,'requests':rows};counters.append(item);q.save('hold-retention',counters);q.emit('retention',**{k:v for k,v in item.items() if k not in ('requests','events')})
 q.emit('done',cells=len(results),retention_cells=len(counters))
