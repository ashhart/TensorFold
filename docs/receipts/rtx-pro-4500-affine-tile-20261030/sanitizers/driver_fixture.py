import ctypes as C,json,os,math,gc
from pathlib import Path
import numpy as np
R=Path(os.environ.get('TF_RECEIPT_ROOT','/task'));O=R/'results'/os.environ['SAN_TOOL'];O.mkdir(parents=True,exist_ok=True)
scope=os.environ.get('TF_SAN_SCOPE','both');assert scope in ['both','tile','attention']
d=C.CDLL('libcuda.so.1');vp=C.c_void_p;u64=C.c_uint64;ui=C.c_uint;sz=C.c_size_t

def api(name,args):
 f=getattr(d,name);f.argtypes=args;f.restype=C.c_int;return f
def ok(code):
 if code:raise RuntimeError('CUDA status '+str(code))
init=api('cuInit',[ui]);retain=api('cuDevicePrimaryCtxRetain',[C.POINTER(vp),C.c_int]);setctx=api('cuCtxSetCurrent',[vp]);alloc=api('cuMemAlloc_v2',[C.POINTER(u64),sz]);free=api('cuMemFree_v2',[u64]);h2d=api('cuMemcpyHtoD_v2',[u64,vp,sz]);d2h=api('cuMemcpyDtoH_v2',[vp,u64,sz]);fill=api('cuMemsetD8_v2',[u64,C.c_ubyte,sz]);sc=api('cuStreamCreate',[C.POINTER(vp),ui]);sync=api('cuStreamSynchronize',[vp]);load=api('cuModuleLoad',[C.POINTER(vp),C.c_char_p]);get=api('cuModuleGetFunction',[C.POINTER(vp),vp,C.c_char_p]);attr=api('cuFuncSetAttribute',[vp,C.c_int,C.c_int]);launch=api('cuLaunchKernel',[vp,ui,ui,ui,ui,ui,ui,ui,vp,C.POINTER(vp),vp]);begin=api('cuStreamBeginCapture_v2',[vp,C.c_int]);end=api('cuStreamEndCapture',[vp,C.POINTER(vp)]);instantiate=api('cuGraphInstantiateWithFlags',[C.POINTER(vp),vp,u64]);graphlaunch=api('cuGraphLaunch',[vp,vp]);graphdestroy=api('cuGraphDestroy',[vp]);execdestroy=api('cuGraphExecDestroy',[vp])
class Attr(C.Structure):_fields_=[('id',ui),('value',u64*8)]
class Config(C.Structure):_fields_=[('gx',ui),('gy',ui),('gz',ui),('bx',ui),('by',ui),('bz',ui),('smem',ui),('stream',vp),('attrs',C.POINTER(Attr)),('count',ui)]
ex=api('cuLaunchKernelEx',[C.POINTER(Config),vp,C.POINTER(vp),vp]);pdl=Attr();pdl.id=6;pdl.value[0]=1
class Geo(C.Structure):_fields_=[(s,C.c_int) for s in ['hid','nk','nv','qh','kvh','nl']]
class Seg(C.Structure):_fields_=[(s,u64) for s in ['rec','conv','kv','tap','log','before']]+[(s,C.c_int) for s in ['cap','len','row0','rows','tap_rows','taps','delta']]+[(s,u64) for s in ['pt','ps']]+[('ptok',C.c_int)]
class Part(C.Structure):_fields_=[(s,vp) for s in ['w','s','b','out']]+[(s,C.c_int) for s in ['n','npad','sk','tiles','first']]
class Parts(C.Structure):_fields_=[('p',Part*4),('count',C.c_int)]
ok(init(0));ctx=vp();ok(retain(C.byref(ctx),0));ok(setctx(ctx));stream=vp();ok(sc(C.byref(stream),1));modules=[];checks=[];keep=[];rng=np.random.default_rng(492)
class Mem:
 def __init__(self,data):
  self.initial=data.tobytes() if isinstance(data,np.ndarray) else bytes(data);self.n=len(self.initial);base=u64();ok(alloc(C.byref(base),self.n+512));self.base=base.value;self.ptr=self.base+256;ok(fill(self.base,165,self.n+512));self.reset();keep.append(self)
 def reset(self):
  raw=C.create_string_buffer(self.initial);ok(h2d(self.ptr,C.cast(raw,vp),self.n))
 def read(self):
  raw=C.create_string_buffer(self.n);ok(d2h(C.cast(raw,vp),self.ptr,self.n));return raw.raw
 def guards(self):
  for ptr_ in [self.base,self.ptr+self.n]:
   raw=C.create_string_buffer(256);ok(d2h(C.cast(raw,vp),ptr_,256));assert raw.raw==bytes([165])*256,'buffer guard'
 def close(self):
  self.guards();ok(free(self.base))
def save():
 (O/'validation.json').write_text(json.dumps(checks,indent=2));(O/'failures.json').write_text('[]')
def exact(label,a,b):
 equal=a==b;checks.append({'case':label,'bit_exact':equal});save();assert equal,label
def readonly(ms):
 for m in ms:m.guards();assert m.read()==m.initial,'input changed'
def fn(file,name,smem=0):
 if os.environ.get('TF_CUDA_FIXTURE_NVCC')=='13.0':file=file.replace('.fatbin','-13.0.fatbin')
 m=vp();ok(load(C.byref(m),str(R/'pr-probes'/file).encode()));f=vp();ok(get(C.byref(f),m,name.encode()));modules.append(m)
 if smem:ok(attr(f,8,smem))
 return f
def call(f,grid,block,shared,args,pd=False):
 vals=[vp(x.ptr) if isinstance(x,Mem) else C.c_int(x) if isinstance(x,int) else x for x in args];ptrs=(vp*len(vals))(*[C.cast(C.byref(x),vp) for x in vals]);keep.append((vals,ptrs));block=(block,1,1) if isinstance(block,int) else block
 if pd:
  config=Config(*grid,*block,shared,stream,C.pointer(pdl),1);ok(ex(C.byref(config),f,ptrs,None))
 else:ok(launch(f,*grid,*block,shared,stream,ptrs,None))
class Graph:
 def __init__(self,f):
  ok(begin(stream,1));f();g=vp();ok(end(stream,C.byref(g)));self.e=vp();ok(instantiate(C.byref(self.e),g,0));ok(graphdestroy(g))
 def run(self):ok(graphlaunch(self.e,stream));ok(sync(stream))
 def close(self):ok(execdestroy(self.e))
def bf(x):
 a=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32);return ((a+np.uint32(32767)+((a>>16)&1))>>16).astype(np.uint16)
def normal(n):return rng.standard_normal(n).astype(np.float32)
def release(memories):
 for m in memories:m.close()
 keep.clear();gc.collect()
sym=lambda bn,st:f'_ZN12tf_qmm_group12group_kernelILi64ELi16ELi{bn}ELi1ELi{bn//16}ELi{st}ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii'
group={64:fn('group.fatbin',sym(64,8),35328),128:fn('group.fatbin',sym(128,4),26880)} if scope!='attention' else {}
attention_form='flat'
attn={2:fn('attention.fatbin','tf_q35_attn_rows8_mt2'),4:fn('attention.fatbin','tf_q35_attn_rows8')} if scope!='tile' else {}

def mlp():
 n=17408;k=5120;xraw=bf(normal(16*k)).reshape(16,k);xf=(xraw.astype(np.uint32)<<16).view(np.float32).reshape(16,k//64,64);sums=np.zeros((16,k//64),np.float32)
 for i in range(64):sums=np.add(sums,xf[:,:,i],dtype=np.float32)
 weights=[Mem(rng.integers(0,256,n*k//2,dtype=np.uint8)),Mem(bf(rng.random(n*k//64,dtype=np.float32)*.003+.001)),Mem(bf(-rng.random(n*k//64,dtype=np.float32)*.03))];x=Mem(xraw);xs=Mem(sums);outputs=[Mem(np.zeros(16*n,np.uint16)) for _ in range(2)]
 for rows in [1,8,16]:
  graphs=[]
  for bn,out in zip([64,128],outputs):
   p=Parts();p.count=1;p.p[0]=Part(*(w.ptr for w in weights),out.ptr,n,n,1,n//bn,0)
   graphs.append(Graph(lambda bn=bn,out=out,p=p:call(group[bn],(n//bn,1,1),bn*2,35328 if bn==64 else 26880,[x,xs,p,rows,k,k,1,1],bn==128)))
  for repeat in range(5):
   for g in graphs:g.run()
   exact(f'mlp/{rows}/repeat{repeat}',outputs[0].read()[:rows*n*2],outputs[1].read()[:rows*n*2])
  for g in graphs:g.close()
 readonly(weights+[x,xs]);release(weights+[x,xs]+outputs)

def attention(paged=False):
 geo=Geo(5120,16,48,24,4,1)
 for length in [8192,32768,122000]:
  for m in [1,8,16]:
   cap=length+m+32;raw=rng.integers(0,120,2*4*cap*(256+4),dtype=np.uint8);raw[2*4*cap*256:].view(np.float32).fill(.001)
   table=None;page_bytes=2*4*256*(256+4)
   if paged:
    count=(cap+255)//256;order=rng.permutation(count).astype(np.int32);arena=np.zeros((count,page_bytes),np.uint8)
    codes=raw[:2*4*cap*256].reshape(2,4,cap,256);scales=raw[2*4*cap*256:].view(np.float32).reshape(2,4,cap)
    for logical,physical in enumerate(order):
     lo=logical*256;hi=min(cap,lo+256);n=hi-lo
     arena[physical,:2*4*256*256].reshape(2,4,256,256)[:,:,:n]=codes[:,:,lo:hi]
     arena[physical,2*4*256*256:].view(np.float32).reshape(2,4,256)[:,:,:n]=scales[:,:,lo:hi]
    kv=Mem(arena);table=Mem(order)
   else:kv=Mem(raw)
   q=Mem(bf(normal(m*24*256)*.1));rows=Mem(np.zeros(m,np.int32));anc=Mem(b''.join((0).to_bytes(8,'little')+i.to_bytes(4,'little')+(1).to_bytes(4,'little') for i in range(m)));params=Mem(np.array([1,m,m,1,1],np.int32));sg=Mem(bytes(Seg(0,0,kv.ptr,0,0,0,cap,length,0,m,0,0,0,table.ptr if table else 0,page_bytes if table else 0,256 if table else 0)));outs=[];graphs=[]
   for mt in [4,2]:
    pm=Mem(np.full(m*24*64,-123,np.float32));pl=Mem(np.full(m*24*64,-123,np.float32));po=Mem(np.full(m*24*64*256,-123,np.float32));outs.append([pm,pl,po]);rb=16*mt//6
    graphs.append(Graph(lambda mt=mt,rb=rb,pm=pm,pl=pl,po=po:call(attn[mt],((m+rb-1)//rb,64,4),mt*64,0,[geo,q,sg,rows,anc,0,params,C.c_float(1/16),m,pm,pl,po])))
   reference=None
   for repeat in range(5):
    for g in graphs:g.run()
    values=[[a.read() for a in out] for out in outs]
    for i,label in enumerate(['PM','PL','PO']):exact(f'attn/{attention_form}/{length}/M{m}/{label}/repeat{repeat}',values[0][i],values[1][i])
    if reference is None:reference=values[1]
    else:
     for i,label in enumerate(['PM','PL','PO']):exact(f'attn-repeat/{length}/M{m}/{label}/{repeat}',reference[i],values[1][i])
   for g in graphs:g.close()
   inputs=[kv,q,rows,anc,params,sg]+([table] if table else [])
   readonly(inputs);release(inputs+outs[0]+outs[1])

try:
 if scope!='attention':mlp()
 if scope!='tile':
  attention()
  attention_form='paged-table'
  attention(paged=True)
  attention_form='whole'
  attn={2:fn('attention.fatbin','tf_q35_attn_rows8_mt2_whole'),4:fn('attention.fatbin','tf_q35_attn_rows8_whole')}
  attention()
 (O/'complete.json').write_text(json.dumps({'complete':True,'scope':'Selected real MLP geometry, MT2 FP8 attention rows1/8/16 at8K/32K/122K, five graph replays, byte equality, unchanged inputs and256-byte guards. Direct CUDA driver and CPU numpy fixtures; no PyTorch runtime.'}))
finally:save()
