#!/usr/bin/env python3
"""Independent FP32 Qwen3 embedding oracle from the checkpoint files; no model-library imports."""

import argparse, ctypes as C, hashlib, json, math, mmap, struct, time
from pathlib import Path
F=C.c_float; U=C.c_uint16; P=C.POINTER(F); W=C.POINTER(U)
def array(n): return (F*n)()
class Reference:
    def __init__(self, directory, library):
        self.dir=Path(directory); self.config=json.loads((self.dir/'config.json').read_text())
        self.receipt=json.loads((self.dir/'VERIFIED.json').read_text()); self.maps=[]; self.tensors={}
        self.lib=C.CDLL(str(Path(library).resolve()))
        self.lib.q3_linear.argtypes=[P,P,W,C.c_int,C.c_int,C.c_int]; self.lib.q3_linear.restype=C.c_int
        self.lib.q3_norm.argtypes=[P,P,W,C.c_int,C.c_int,F]
        self.lib.q3_rope.argtypes=[P,C.c_int,C.c_int,C.c_double]
        self.lib.q3_attention.argtypes=[P,P,P,P,C.c_int,C.c_int,C.c_int]; self.lib.q3_attention.restype=C.c_int
        self.lib.q3_add.argtypes=[P,P,C.c_size_t]; self.lib.q3_silu.argtypes=[P,P,C.c_size_t]
        self.lib.q3_widen.argtypes=[P,W,C.c_size_t]
        index=self.dir/'model.safetensors.index.json'
        files=sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
        verified={r['file']:r for r in self.receipt['files']}
        for name in files:
            path=self.dir/name; assert path.stat().st_size==verified[name]['size'], name
            with path.open('rb') as f:
                mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_COPY)
            self.maps.append(mm); size=struct.unpack('<Q',mm[:8])[0]; header=json.loads(mm[8:8+size])
            for key,t in header.items():
                if key=='__metadata__': continue
                assert t['dtype']=='BF16', (key,t['dtype'])
                lo,hi=t['data_offsets']; assert hi-lo==2*math.prod(t['shape'])
                raw=(U*math.prod(t['shape'])).from_buffer(mm,8+size+lo)
                self.tensors[key]=(raw,t['shape'])
        self.scope='model.' if 'model.embed_tokens.weight' in self.tensors else ''
    def weight(self,key,shape):
        raw,got=self.tensors[self.scope+key]; assert got==list(shape),(key,got,shape)
        return raw
    def linear(self,x,key,rows,n,k):
        y=array(rows*n)
        assert self.lib.q3_linear(y,x,self.weight(key,(n,k)),rows,n,k)==0
        return y
    def norm(self,x,key,rows,width):
        y=array(rows*width);self.lib.q3_norm(y,x,self.weight(key,(width,)),rows,width,self.config['rms_norm_eps']);return y
    def forward(self,ids):
        c=self.config;r=len(ids);h=c['hidden_size'];a=c['num_attention_heads'];kv=c['num_key_value_heads'];inter=c['intermediate_size'];qw=a*128;kw=kv*128
        assert r>0 and all(0<=t<c['vocab_size'] for t in ids)
        emb=self.weight('embed_tokens.weight',(c['vocab_size'],h)); x=array(r*h)
        for row,t in enumerate(ids):self.lib.q3_widen(C.cast(C.byref(x,row*h*4),P),C.cast(C.byref(emb,t*h*2),W),h)
        for layer in range(c['num_hidden_layers']):
            p=f'layers.{layer}.';z=self.norm(x,p+'input_layernorm.weight',r,h)
            q=self.linear(z,p+'self_attn.q_proj.weight',r,qw,h);k=self.linear(z,p+'self_attn.k_proj.weight',r,kw,h);v=self.linear(z,p+'self_attn.v_proj.weight',r,kw,h)
            q=self.norm(q,p+'self_attn.q_norm.weight',r*a,128);k=self.norm(k,p+'self_attn.k_norm.weight',r*kv,128)
            self.lib.q3_rope(q,r,a,c['rope_theta']);self.lib.q3_rope(k,r,kv,c['rope_theta'])
            att=array(r*qw);assert self.lib.q3_attention(att,q,k,v,r,a,kv)==0
            y=self.linear(att,p+'self_attn.o_proj.weight',r,h,qw);self.lib.q3_add(x,y,r*h)
            z=self.norm(x,p+'post_attention_layernorm.weight',r,h)
            gate=self.linear(z,p+'mlp.gate_proj.weight',r,inter,h);up=self.linear(z,p+'mlp.up_proj.weight',r,inter,h)
            self.lib.q3_silu(gate,up,r*inter);y=self.linear(gate,p+'mlp.down_proj.weight',r,h,inter);self.lib.q3_add(x,y,r*h)
        z=self.norm(x,'norm.weight',r,h);vec=list(z[(r-1)*h:r*h]);norm=math.sqrt(sum(v*v for v in vec))
        return [v/norm for v in vec]
def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('inputs');p.add_argument('output');p.add_argument('--library',required=True);o=p.parse_args()
    ref=Reference(o.checkpoint,o.library);data={'revision':ref.receipt['revision'],'arithmetic':'FP32 from stored BF16 weights; no intermediate BF16 casts','inputs':[]}
    for ids in json.loads(Path(o.inputs).read_text()):
        t=time.monotonic();vec=ref.forward(ids);data['inputs'].append({'tokens':ids,'vector':vec,'seconds':time.monotonic()-t})
        Path(o.output).write_text(json.dumps(data)+'\n')
if __name__=='__main__':main()
