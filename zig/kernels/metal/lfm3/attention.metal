// Fixed contiguous key partitions keep one query's arithmetic independent of neighboring rows and lane width.
#include <metal_stdlib>
using namespace metal;

#ifndef L3_RISK_FAST
#define L3_RISK_FAST 0
#endif

constant constexpr uint L3CHUNKS = 64;
constant constexpr uint L3PER = 4096;
constant constexpr uint L3PART = 1024;
struct lfm3_attn_args { uint len, cap; float scale; uint window, chunk_base; };

#ifndef L3_ATTN_ACC
#define L3_ATTN_ACC float
#endif

inline float lfm3_attention_dot(const thread float* q, const device bfloat* k) {
  L3_ATTN_ACC dot = 0.0f;
  for (uint j = 0; j < 4; ++j) dot += q[j] * float(k[32 * j]);
  return simd_sum(float(dot));
}

// Rare-path range extension only: FP32 mantissas, integer exponents, no precision reduction.
#ifdef L3_RANGE_SKIP_REPAIR
constant constexpr bool L3_RANGE_REPAIR_ENABLED=false;
#else
constant constexpr bool L3_RANGE_REPAIR_ENABLED=true;
#endif
struct l3_x { float m; int e; };
inline l3_x l3_x_bits(uint u) {
 const uint sign=u&0x80000000u,be=(u>>23)&255u,fr=u&0x7fffffu;
 if(be==255||((u&0x7fffffffu)==0))return {as_type<float>(u),0};
 if(be==0){const uint shift=clz(fr)-8u;return {as_type<float>(sign|0x3f800000u|((fr<<shift)&0x7fffffu)),-126-int(shift)};}
 return {as_type<float>(sign|0x3f800000u|fr),int(be)-127};
}
inline l3_x l3_x_float(float a){return l3_x_bits(as_type<uint>(a));}
inline l3_x l3_x_norm(float m,int e){l3_x a=l3_x_float(m);if(isfinite(a.m)&&a.m!=0)a.e+=e;return a;}
inline l3_x l3_x_bf(ushort a){return l3_x_bits(uint(a)<<16);}
inline l3_x l3_x_mul(l3_x a,l3_x b){return l3_x_norm(a.m*b.m,a.e+b.e);}
inline l3_x l3_x_add(l3_x a,l3_x b){
 if(!isfinite(a.m)||!isfinite(b.m))return l3_x_float(a.m+b.m);
 if(a.m==0&&b.m==0)return l3_x_float(a.m+b.m);
 if(a.m==0)return b;if(b.m==0)return a;
 l3_x hi=a.e>=b.e?a:b,lo=a.e>=b.e?b:a;const int d=hi.e-lo.e;
 if(d>=26)return hi;
 return l3_x_norm(hi.m+ldexp(lo.m,-d),hi.e);
}
inline l3_x l3_x_div(l3_x a,float denominator){const l3_x b=l3_x_float(denominator);return l3_x_norm(a.m/b.m,a.e-b.e);}
inline l3_x l3_x_exp(float gap,float ordinary){
 if(gap==-INFINITY)return {0,0};
 // Correct zero class of FP64 exp on the discrete FP32-gap grid; also safe for nonfinite V.
 if(gap<=-745.13323974609375f)return {0,0};
 if(!isfinite(ordinary)||((as_type<uint>(ordinary)&0x7f800000u)!=0))return l3_x_float(ordinary);
 const float nf=floor(gap*0x1.715476p+0f);const int n=int(nf);
 float residual=fma(-nf,0x1.62e430p-1f,gap);
 residual=fma(-nf,-0x1.05c610p-29f,residual);
 return l3_x_norm(metal::precise::exp(residual),n);
}
inline ushort l3_x_pack(l3_x a){
 const uint u=as_type<uint>(a.m),sign=(u>>16)&0x8000u;
 if(isnan(a.m))return ushort(sign|0x7fc0u);if(isinf(a.m))return ushort(sign|0x7f80u);
 if(a.m==0||a.e<=-135)return ushort(sign);
 const uint M=(u&0x7fffffu)|0x800000u;int e=a.e;
 const uint q=e>=-126?16u:uint(-e-110),hi=M>>q,tail=M&((1u<<q)-1u),halfway=1u<<(q-1);
 uint rounded=hi+uint(tail>halfway||(tail==halfway&&(hi&1u)));
 if(e<-126)return ushort(sign|rounded);
 if(rounded==256){++e;rounded=128;}
 if(e>=128)return ushort(sign|0x7f80u);
 return ushort(sign|(uint(e+127)<<7)|(rounded-128));
}
inline bool l3_finite_bits(uint a){return (a&0x7f800000u)!=0x7f800000u;}
inline bool l3_nonzero_bits(uint a){return (a&0x7fffffffu)!=0;}
inline bool l3_product_risk(uint a,uint b){
 const bool live=l3_finite_bits(a)&l3_finite_bits(b)&l3_nonzero_bits(a)&l3_nonzero_bits(b);
 const uint ae=(a>>23)&255u,be=(b>>23)&255u,s=ae+be;
 return live&((ae==0)|(be==0)|(s<=129u)|(s>=380u));
}
inline bool l3_sum_risk(uint a,uint b,uint c,uint d,uint result){
 const bool finite=l3_finite_bits(a)&l3_finite_bits(b)&l3_finite_bits(c)&l3_finite_bits(d);
 const uint e=(result>>23)&255u;
 const bool any=(l3_nonzero_bits(a)&l3_nonzero_bits(b))|(l3_nonzero_bits(c)&l3_nonzero_bits(d));
 return finite&((e>=253u)|((e==0)&any));
}
inline bool l3_exp_risk(float gap,float value){
 return isfinite(gap)&(gap<0)&((as_type<uint>(value)&0x7f800000u)==0);}
inline bool l3_div_risk(uint numerator,uint denominator){
 const bool live=l3_finite_bits(numerator)&l3_finite_bits(denominator)&l3_nonzero_bits(numerator)&l3_nonzero_bits(denominator);
 const int a=int((numerator>>23)&255u),b=int((denominator>>23)&255u);
 return live&((a==0)|(b==0)|(a-b<=-125)|(a-b>=126));
}

struct l3_risk_factor { uint e; bool finite, nonzero; };
inline l3_risk_factor l3_risk_decode(uint bits){
 const uint e=(bits>>23)&255u;
 return {e,e!=255u,(bits&0x7fffffffu)!=0u};
}
inline bool l3_pv_risk(uint before,uint value,uint after,l3_risk_factor a,l3_risk_factor b){
 const l3_risk_factor x=l3_risk_decode(before),y=l3_risk_decode(value);
 const uint e=(after>>23)&255u;
 const bool xf=x.finite&a.finite,yf=y.finite&b.finite;
 const bool xl=xf&x.nonzero&a.nonzero,yl=yf&y.nonzero&b.nonzero;
 const bool xr=(x.e==0u)|(a.e==0u)|((x.e+a.e-130u)>=250u);
 const bool yr=(y.e==0u)|(b.e==0u)|((y.e+b.e-130u)>=250u);
 return (xl&(xr|(yf&(e==0u))))|(yl&(yr|(xf&(e==0u))))|(xf&yf&(e>=253u));
}

inline void l3_x_merge(float part_max,float part_total,thread l3_x* part,
 thread float& maximum,thread float& total,thread l3_x* result){
 if(part_total==0)return;
 const float next=max(maximum,part_max),ga=maximum-next,gb=part_max-next;
 const float a=maximum==-INFINITY?0:metal::precise::exp(ga),b=metal::precise::exp(gb);
 const l3_x ae=maximum==-INFINITY?l3_x{0,0}:l3_x_exp(ga,a),be=l3_x_exp(gb,b);
 total=total*a+part_total*b;
 for(uint j=0;j<4;++j)result[j]=l3_x_add(l3_x_mul(result[j],ae),l3_x_mul(part[j],be));
 maximum=next;
}
inline void l3_attention_replay(const device bfloat* queries,const device bfloat* keys,const device bfloat* values,
 constant lfm3_attn_args& p,uint head,uint row,uint lane,thread l3_x* result){
 const uint count=p.len+row+1,cut=p.window==0?0:count>p.window?count-p.window:0,kh=head/4;
 float q[4];for(uint j=0;j<4;++j)q[j]=float(queries[(size_t(row)*16+head)*128+lane+32*j]);
 float maximum=-INFINITY,total=0;for(uint j=0;j<4;++j)result[j]={0,0};
 for(uint chunk=cut/4096;chunk<(count+4095)/4096;++chunk){
  float cm=-INFINITY,cl=0;l3_x co[4];for(uint j=0;j<4;++j)co[j]={0,0};
  for(uint part=0;part<4;++part){
   const uint first=max(chunk*4096+part*1024,cut),end=min(chunk*4096+(part+1)*1024,count);
   float pm=-INFINITY,pl=0;l3_x po[4];for(uint j=0;j<4;++j)po[j]={0,0};
   for(uint key=first;key<end;++key){
    const size_t base=(size_t(kh)*p.cap+key)*128+lane;
    const float score=lfm3_attention_dot(q,keys+base)*p.scale,next=max(pm,score),ga=pm-next,gp=score-next;
    const float a=pm==-INFINITY?0:metal::precise::exp(ga),v=score==-INFINITY?0:metal::precise::exp(gp);
    const l3_x ae=pm==-INFINITY?l3_x{0,0}:l3_x_exp(ga,a),pe=score==-INFINITY?l3_x{0,0}:l3_x_exp(gp,v);
    pl=pl*a+v;
    for(uint j=0;j<4;++j){
     const l3_x value=l3_x_bf(as_type<ushort>(values[base+32*j]));
     po[j]=l3_x_add(l3_x_mul(po[j],ae),l3_x_mul(pe,value));
    }
    pm=next;
   }
   l3_x_merge(pm,pl,po,cm,cl,co);
  }
  l3_x_merge(cm,cl,co,maximum,total,result);
 }
 for(uint j=0;j<4;++j)result[j]=l3_x_div(result[j],total);
}

inline void lfm3_attention_partial(const device bfloat* queries, const device bfloat* keys,
    const device bfloat* values, constant lfm3_attn_args& p, device float* pm, device float* pl, device float* po,
    uint head, uint row, uint partition, uint sg, uint lane,
    threadgroup float* maxima, threadgroup float* totals, threadgroup float* outputs,device uint* range_flags,threadgroup uint* range_parts) {
  const uint chunk = partition + p.chunk_base, count = p.len + row + 1;
  const uint cutoff = p.window == 0 ? 0 : count > p.window ? count - p.window : 0;
  const uint first = max(chunk * L3PER + sg * L3PART, cutoff);
  const uint end = min(chunk * L3PER + (sg + 1) * L3PART, count);
  const uint kh = head / 4;
  float q[4];
  L3_ATTN_ACC result[4] = {0, 0, 0, 0};
  for (uint j = 0; j < 4; ++j) q[j] = float(queries[(size_t(row) * 16 + head) * 128 + lane + 32 * j]);
  uint range_risk=0;
  float maximum = -INFINITY;
  L3_ATTN_ACC total = 0.0f;
  for (uint key = first; key < end; ++key) {
    const size_t base = (size_t(kh) * p.cap + key) * 128 + lane;
    const float score = lfm3_attention_dot(q, keys + base) * p.scale;
    const float next = max(maximum, score);
    const float alpha = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
    const float probability = score == -INFINITY ? 0.0f : metal::precise::exp(score - next);
    range_risk|=uint(l3_exp_risk(maximum-next,alpha)|l3_exp_risk(score-next,probability));
    total = total * alpha + probability;
    const l3_risk_factor ar=l3_risk_decode(as_type<uint>(alpha)),br=l3_risk_decode(as_type<uint>(probability));
    for (uint j = 0; j < 4; ++j) {
      const uint before=as_type<uint>(float(result[j])),a=as_type<uint>(alpha),b=as_type<uint>(probability),v=uint(as_type<ushort>(values[base+32*j]))<<16;
      if(!L3_RISK_FAST)range_risk|=uint(l3_product_risk(before,a)|l3_product_risk(b,v));
      result[j] = result[j] * alpha + probability * float(values[base + 32 * j]);
      range_risk|=uint(L3_RISK_FAST?l3_pv_risk(before,v,as_type<uint>(float(result[j])),ar,br):
        l3_sum_risk(before,a,b,v,as_type<uint>(float(result[j]))));
    }
    maximum = next;
  }
  const uint part_risk=simd_or(range_risk);
  if (lane == 0) { maxima[sg] = maximum; totals[sg] = total; range_parts[sg]=part_risk; }
  for (uint j = 0; j < 4; ++j) outputs[sg * 128 + lane + 32 * j] = result[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const size_t target = (size_t(row) * 16 + head) * L3CHUNKS + chunk;
  range_risk=range_parts[0]|range_parts[1]|range_parts[2]|range_parts[3];
  float merged_max = -INFINITY, merged_total = 0.0f, merged[4] = {0, 0, 0, 0};
  for (uint part = 0; part < 4; ++part) {
    if (totals[part] == 0.0f) continue;
    const float next = max(merged_max, maxima[part]);
    const float a = merged_max == -INFINITY ? 0.0f : metal::precise::exp(merged_max - next);
    const float b = metal::precise::exp(maxima[part] - next);
    range_risk|=uint(l3_exp_risk(merged_max-next,a)|l3_exp_risk(maxima[part]-next,b));
    merged_total = merged_total * a + totals[part] * b;
    const l3_risk_factor ar=l3_risk_decode(as_type<uint>(a)),br=l3_risk_decode(as_type<uint>(b));
    for (uint j = 0; j < 4; ++j) {
      const uint before=as_type<uint>(merged[j]),aa=as_type<uint>(a),bb=as_type<uint>(b),v=as_type<uint>(outputs[part*128+lane+32*j]);
      if(!L3_RISK_FAST)range_risk|=uint(l3_product_risk(before,aa)|l3_product_risk(v,bb));
      merged[j] = merged[j] * a + outputs[part * 128 + lane + 32 * j] * b;
      range_risk|=uint(L3_RISK_FAST?l3_pv_risk(before,v,as_type<uint>(merged[j]),ar,br):
        l3_sum_risk(before,aa,v,bb,as_type<uint>(merged[j])));
    }
    merged_max = next;
  }
  const uint chunk_risk=simd_or(range_risk);
  if (lane == 0) { pm[target] = merged_max; pl[target] = merged_total; range_flags[target]=chunk_risk;
#ifdef L3_RANGE_CLEAR_RISK
  range_flags[target]=0u;
#endif
 }
  for (uint j = 0; j < 4; ++j) po[target * 128 + lane + 32 * j] = merged[j];
}

[[kernel, max_total_threads_per_threadgroup(128)]] void lfm3_attn_partial(const device bfloat* queries [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                 const device bfloat* values [[buffer(2)]], constant lfm3_attn_args& p [[buffer(3)]],
                                 device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], device float* po [[buffer(6)]],device uint* range_flags [[buffer(7)]],
                                 uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float maxima[4], totals[4], outputs[4 * 128];threadgroup uint range_parts[4];
  lfm3_attention_partial(queries, keys, values, p, pm, pl, po, tg.x, tg.y, tg.z, sg, lane, maxima, totals, outputs,range_flags,range_parts);
}

inline float lfm3_attention_merge(const device float* pm, const device float* pl, const device float* po,
                                  constant lfm3_attn_args& p, uint head, uint row, uint lane,
                                  thread float* result,const device uint* range_flags=nullptr,thread bool* risk=nullptr) {
  const uint count = p.len + row + 1;
  const uint cutoff = p.window == 0 ? 0 : count > p.window ? count - p.window : 0;
  const uint first = cutoff / L3PER, last = (count + L3PER - 1) / L3PER;
  const size_t base = (size_t(row) * 16 + head) * L3CHUNKS;
  float maximum = -INFINITY;
  L3_ATTN_ACC total = 0.0f;
  for (uint chunk = first; chunk < last; ++chunk) {
    const size_t i = base + chunk;
    if(risk!=nullptr&&range_flags!=nullptr)*risk|=range_flags[i]!=0;
    if (pl[i] == 0.0f) continue;
    const float next = max(maximum, pm[i]);
    const float a = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
    const float b = metal::precise::exp(pm[i] - next);
    if(risk!=nullptr)*risk|=l3_exp_risk(maximum-next,a)|l3_exp_risk(pm[i]-next,b);
    total = total * a + pl[i] * b;
    const l3_risk_factor ar=l3_risk_decode(as_type<uint>(a)),br=l3_risk_decode(as_type<uint>(b));
    for (uint j = 0; j < 4; ++j) {
      const uint before=as_type<uint>(float(result[j])),aa=as_type<uint>(a),bb=as_type<uint>(b),v=as_type<uint>(po[i*128+lane+32*j]);
      if(risk!=nullptr&&!L3_RISK_FAST)*risk|=l3_product_risk(before,aa)|l3_product_risk(v,bb);
      result[j] = float(L3_ATTN_ACC(result[j] * a + po[i * 128 + lane + 32 * j] * b));
      if(risk!=nullptr)*risk|=L3_RISK_FAST?l3_pv_risk(before,v,as_type<uint>(float(result[j])),ar,br):
        l3_sum_risk(before,aa,v,bb,as_type<uint>(float(result[j])));
    }
    maximum = next;
  }
  return float(total);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attn_merge(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                               const device float* po [[buffer(2)]], constant lfm3_attn_args& p [[buffer(3)]],
                               device bfloat* y [[buffer(4)]],const device bfloat* queries [[buffer(5)]],const device bfloat* keys [[buffer(6)]],const device bfloat* values [[buffer(7)]],const device uint* range_flags [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
  float result[4] = {0, 0, 0, 0};
  bool range_risk=false;
  const float total = lfm3_attention_merge(pm, pl, po, p, tg.x, tg.y, lane, result,range_flags,&range_risk);
  for(uint j=0;j<4;++j)range_risk|=l3_div_risk(as_type<uint>(float(result[j])),as_type<uint>(total));
  if(L3_RANGE_REPAIR_ENABLED&&simd_any(range_risk)){
    l3_x repaired[4];l3_attention_replay(queries,keys,values,p,tg.x,tg.y,lane,repaired);
    for(uint j=0;j<4;++j)reinterpret_cast<device ushort*>(y)[(size_t(tg.y)*16+tg.x)*128+lane+32*j]=l3_x_pack(repaired[j]);
    return;
  }
  for (uint j = 0; j < 4; ++j) y[(size_t(tg.y) * 16 + tg.x) * 128 + lane + 32 * j] = bfloat(result[j] / total);
}

// The merge tap invokes the production chunk loop before division and BF16 output rounding.
[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attention_merge_sums(
    const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]], const device float* po [[buffer(2)]],
    constant lfm3_attn_args& p [[buffer(3)]], device float* out [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float result[4] = {0, 0, 0, 0};
  const float total = lfm3_attention_merge(pm, pl, po, p, tg.x, tg.y, lane, result);
  const size_t base = (size_t(tg.y) * 16 + tg.x) * 129;
  for (uint j = 0; j < 4; ++j) out[base + lane + 32 * j] = result[j];
  if (lane == 0) out[base + 128] = total;
}

// The contract observes the production dot before softmax can mask its accumulation error.
[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attention_dot_sums(
    const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], device float* out [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float values[4];
  for (uint j = 0; j < 4; ++j) values[j] = float(q[size_t(row) * 128 + lane + 32 * j]);
  const float total = lfm3_attention_dot(values, k + size_t(row) * 128 + lane);
  if (lane == 0) out[row] = total;
}

struct lfm3_attn_segment { ulong keys, values; uint base, rows; lfm3_attn_args args; uint pad; };

[[kernel, max_total_threads_per_threadgroup(128)]] void lfm3_attn_segments(
    const device bfloat* queries [[buffer(0)]], constant lfm3_attn_segment* segments [[buffer(1)]],
    const device uint* row_segments [[buffer(2)]],device uint* range_flags [[buffer(3)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    device float* po [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constant lfm3_attn_segment& seg = segments[row_segments[tg.y]];
  const uint row = tg.y - seg.base;
  const uint last = (seg.args.len + row + 1 + L3PER - 1) / L3PER;
  if (tg.z + seg.args.chunk_base >= last) return;
  threadgroup float maxima[4], totals[4], outputs[4 * 128];threadgroup uint range_parts[4];
  const size_t offset = size_t(seg.base) * 16 * L3CHUNKS;
  lfm3_attention_partial(queries + size_t(seg.base) * 16 * 128,
    reinterpret_cast<const device bfloat*>(seg.keys), reinterpret_cast<const device bfloat*>(seg.values),
    seg.args, pm + offset, pl + offset, po + offset * 128, tg.x, row, tg.z, sg, lane, maxima, totals, outputs,range_flags+offset,range_parts);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attn_segments_merge(
    const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]], const device float* po [[buffer(2)]],
    constant lfm3_attn_segment* segments [[buffer(3)]], device bfloat* y [[buffer(4)]],
    const device uint* row_segments [[buffer(5)]],const device bfloat* queries [[buffer(6)]],const device uint* range_flags [[buffer(7)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constant lfm3_attn_segment& seg = segments[row_segments[tg.y]];
  const size_t offset = size_t(seg.base) * 16 * L3CHUNKS;
  float result[4] = {0, 0, 0, 0};
  bool range_risk=false;
  const float total = lfm3_attention_merge(pm + offset, pl + offset, po + offset * 128,
    seg.args, tg.x, tg.y - seg.base, lane, result,range_flags+offset,&range_risk);
  for(uint j=0;j<4;++j)range_risk|=l3_div_risk(as_type<uint>(float(result[j])),as_type<uint>(total));
  if(L3_RANGE_REPAIR_ENABLED&&simd_any(range_risk)){
    l3_x repaired[4];l3_attention_replay(queries+size_t(seg.base)*16*128,reinterpret_cast<const device bfloat*>(seg.keys),reinterpret_cast<const device bfloat*>(seg.values),seg.args,tg.x,tg.y-seg.base,lane,repaired);
    for(uint j=0;j<4;++j)reinterpret_cast<device ushort*>(y)[(size_t(tg.y)*16+tg.x)*128+lane+32*j]=l3_x_pack(repaired[j]);
    return;
  }
#ifdef L3_SEGMENT_BAD_TAIL
  if (tg.y + 1 == seg.base + seg.rows) return;
#endif
  for (uint j = 0; j < 4; ++j) y[(size_t(tg.y) * 16 + tg.x) * 128 + lane + 32 * j] = bfloat(result[j] / total);
}
