# Four-arm Metal serving comparison; reserve the GPU exclusively before running.
import importlib.util,os,fcntl,threading,concurrent.futures as cf,hashlib
from pathlib import Path
s=importlib.util.spec_from_file_location('q',Path(__file__).with_name('qualification.py'));q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
root=Path(os.environ.get('TF_QUAL_CANDIDATE_ROOT',str(q.ROOT)));q.OUT=Path(os.environ.get('TF_QUAL_OUT','/tmp/tf-lfm2-wide-matrix'));q.OUT.mkdir(exist_ok=True)
with open('/tmp/tensorfold-model.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 q.save('source-hashes',{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['zig/src/core/lanes/depth.zig','zig/src/core/lanes/drafted_chances_test.zig','zig/src/families/lfm2_moe/backend.zig','zig/src/families/lfm2_moe/dspark.zig','zig/src/families/lfm2_moe/dspark_block.zig']})
 arms={'main':(Path(os.environ['TF_QUAL_MAIN']),'plain'),'published':(Path(os.environ['TF_QUAL_PUBLISHED']),'bf16'),'candidate-plain':(q.CAND,'plain'),'candidate-drafted':(q.CAND,'bf16')}
 q.save('binaries',{name:hashlib.sha256(binary.read_bytes()).hexdigest() for name,(binary,_) in arms.items()})
 results=[];refs={}
 prompts=['Write a short Python function that computes the Fibonacci sequence and explain it.','Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example.']
 for round_ in range(4):
  order=list(arms);offset=round_%len(order);order=order[offset:]+order[:offset]
  for arm in order:
   binary,mode=arms[arm]
   label=f'final-{round_}-{arm}'
   with q.Server(label,binary,mode,4096,16):
    if 'memory pressure:' in (q.OUT/(label+'.server.log')).read_text(errors='replace'):raise RuntimeError('memory pressure prevents a valid concurrency qualification')
    for prompt in prompts:
     q.post({'prompt':prompt,'max_tokens':32})
     for width in [1,4,8,16]:
      gate=threading.Barrier(width)
      def run(i):gate.wait();return q.post({'prompt':prompt,'max_tokens':256},True)
      with cf.ThreadPoolExecutor(width) as ex:rows=list(ex.map(run,range(width)))
      ref=refs.setdefault(prompt,q.digest(rows[0]));assert all(q.digest(r)==ref for r in rows)
      if 'memory pressure:' in (q.OUT/(label+'.server.log')).read_text(errors='replace'):raise RuntimeError('memory pressure changed stream admission during qualification')
      rate=width*255/(max(r['last'] for r in rows)-min(r['first'] for r in rows))
      item={'round':round_,'arm':arm,'prompt':'code' if prompt.startswith('Write') else 'prose','streams':width,'tps':rate,'requests':rows};results.append(item);q.save('comparison',results);q.emit('comparison',**{k:v for k,v in item.items() if k!='requests'})
 q.emit('comparison-done',cells=len(results))
