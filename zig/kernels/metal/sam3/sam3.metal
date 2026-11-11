#include <metal_stdlib>
using namespace metal;

kernel void im_sam_copy(const device float *x, device float *y, constant uint &count, uint i [[thread_position_in_grid]]) {
    if (i < count) y[i] = x[i];
}

kernel void im_sam_bias(device float *x, const device float *bias, constant uint2 &p, uint i [[thread_position_in_grid]]) {
    if (i < p.x) x[i] += bias[i % p.y];
}

struct SamVisionLinear { uint rows,cols,depth,activate; };
template<typename Out, bool RESIDUAL=false>
inline void sam_vision_linear(const device half *x, const device half *w, const device float *bias, device Out *out,
    constant SamVisionLinear &p, uint3 group, uint sg, uint lane, const device float *residual=nullptr) {
    const uint row=group.x*64+32*(sg/4), col=group.y*128+32*(sg%4);
    const short2 home=tfp::frag_home(lane);
    tfp::frag<float> acc[2][2];
    tfp::gemm_block<half,false,true,true,256>(acc,x+row*p.depth,w+col*p.depth,p.depth,p.depth,p.depth,p.depth/256,32,32,home);
    TF_UNROLL
    for(short m=0;m<2;m++) {
        TF_UNROLL
        for(short n=0;n<2;n++) {
            TF_UNROLL
            for(short e=0;e<8;e++) {
                const uint c=col+16*n+home.x+TF_COL(e);
                const float value=acc[m][n][e]+bias[c];
                if(RESIDUAL) {
                    const uint r=row+16*m+home.y+(e>>2)*8;
                    const uint win=r/576,local=r%576;
                    const uint dest=p.activate ? (win/3*24+local/24)*72+win%3*24+local%24 : r;
                    out[dest*p.cols+c]=residual[dest*p.cols+c]+value;
                } else acc[m][n][e]=p.activate ? 0.5f*value*(1.0f+qwv_erf(value*0.70710678118654752f)) : value;
            }
        }
    }
    if(!RESIDUAL)tfp::gemm_store<Out,true>(acc,out+row*p.cols+col,p.cols,32,32,home);
}
#define IM_SAM_VISION_LINEAR(NAME,OUT) \
[[max_total_threads_per_threadgroup(256)]] kernel void NAME(const device half *x, const device half *w, const device float *bias, device OUT *out, \
    constant SamVisionLinear &p, uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    sam_vision_linear(x,w,bias,out,p,group,sg,lane); \
}
IM_SAM_VISION_LINEAR(im_sam_vision_linear,float)
IM_SAM_VISION_LINEAR(im_sam_vision_gelu,half)

[[max_total_threads_per_threadgroup(256)]]
kernel void im_sam_vision_residual(const device half *x, const device half *w, const device float *bias, device float *out,
    const device float *residual, constant SamVisionLinear &p, uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    sam_vision_linear<float,true>(x,w,bias,out,p,group,sg,lane,residual);
}

struct SamAttention { uint rows,keys,heads,batch; };
template<typename T, short DIM, short BK>
inline void sam_attention(const device T *q, const device T *k, const device T *v, device T *out,
    constant SamAttention &p, uint3 group, uint sg, uint lane) {
    const uint width=p.heads*DIM, row=group.x*64+sg*16;
    q+=group.z*p.rows*width+row*width+group.y*DIM;
    k+=group.z*p.keys*width+group.y*DIM;
    v+=group.z*p.keys*width+group.y*DIM;
    out+=group.z*p.rows*width+row*width+group.y*DIM;
    const short2 home=tfp::frag_home(lane);
    tfp::frag<T> query[DIM/16];
    tfp::frag<float> acc[DIM/16];
    TF_UNROLL
    for(short d=0;d<DIM/16;d++) {
        tfp::frag_get_in(query[d],q,width,0,16*d,home,int(p.rows)-int(row),DIM);
        acc[d]=tfp::frag<float>(0);
    }
    float top[2]={tfp::kMasked,tfp::kMasked},total[2]={0,0};
    const float scale2=1.44269504089f*metal::precise::rsqrt(float(DIM));
    for(uint first=0;first<p.keys;first+=BK) {
        tfp::frag<float> s[BK/16];
        TF_UNROLL
        for(short n=0;n<BK/16;n++)s[n]=tfp::frag<float>(0);
        TF_UNROLL
        for(short d=0;d<DIM/16;d++) {
            TF_UNROLL
            for(short n=0;n<BK/16;n+=2) {
                tfp::frag<T> k0,k1;
                if(first+BK<=p.keys) {
                    tfp::frag_get(k0,k,width,first+16*n,16*d,home);
                    tfp::frag_get(k1,k,width,first+16*n+16,16*d,home);
                } else {
                    tfp::frag_get_in(k0,k,width,first+16*n,16*d,home,p.keys,DIM);
                    tfp::frag_get_in(k1,k,width,first+16*n+16,16*d,home,p.keys,DIM);
                }
                tfp::mma_16x32<false,true>(s[n],s[n+1],query[d],k0,k1);
            }
        }
        float next[2]={top[0],top[1]};
        TF_UNROLL
        for(short n=0;n<BK/16;n++) {
            TF_UNROLL
            for(short e=0;e<8;e++)s[n][e]=first+16*n+home.x+TF_COL(e)<p.keys ? s[n][e]*scale2 : tfp::kMasked;
            tfp::row_fold<true>(s[n],next);
        }
        float previous[2];
        TF_UNROLL
        for(short h=0;h<2;h++) {
            previous[h]=fast::exp2(top[h]-next[h]);
            total[h]*=previous[h];
            top[h]=next[h];
        }
        TF_UNROLL
        for(short n=0;n<BK/16;n++) {
            TF_UNROLL
            for(short e=0;e<8;e++)s[n][e]=fast::exp2(s[n][e]-next[e>>2]);
            tfp::row_fold<false>(s[n],total);
        }
        TF_UNROLL
        for(short d=0;d<DIM/16;d++) {
            TF_UNROLL
            for(short e=0;e<8;e++)acc[d][e]*=previous[e>>2];
        }
        TF_UNROLL
        for(short d=0;d<DIM/16;d+=2) {
            TF_UNROLL
            for(short n=0;n<BK/16;n++) {
                tfp::frag<T> v0,v1;
                if(first+BK<=p.keys) {
                    tfp::frag_get(v0,v,width,first+16*n,16*d,home);
                    tfp::frag_get(v1,v,width,first+16*n,16*d+16,home);
                } else {
                    tfp::frag_get_in(v0,v,width,first+16*n,16*d,home,p.keys,DIM);
                    tfp::frag_get_in(v1,v,width,first+16*n,16*d+16,home,p.keys,DIM);
                }
                tfp::frag<T> probability;
                TF_UNROLL
                for(short e=0;e<8;e++)probability[e]=T(s[n][e]);
                tfp::mma_16x32<false,false>(acc[d],acc[d+1],probability,v0,v1);
            }
        }
    }
    TF_UNROLL
    for(short d=0;d<DIM/16;d++) {
        TF_UNROLL
        for(short e=0;e<8;e++)acc[d][e]/=total[e>>2];
        tfp::frag_put_in(acc[d],out,width,0,16*d,home,int(p.rows)-int(row),DIM);
    }
}
#define IM_SAM_ATTENTION(NAME,TYPE,DIM,BK) \
[[max_total_threads_per_threadgroup(128)]] kernel void NAME(const device TYPE *q, const device TYPE *k, const device TYPE *v, device TYPE *out, \
    constant SamAttention &p, uint3 group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    sam_attention<TYPE,DIM,BK>(q,k,v,out,p,group,sg,lane); \
}
IM_SAM_ATTENTION(im_sam_attention_32_32,float,32,32)
IM_SAM_ATTENTION(im_sam_attention_64_32,float,64,32)
IM_SAM_ATTENTION(im_sam_attention_32_64_h,half,32,64)
IM_SAM_ATTENTION(im_sam_attention_64_32_h,half,64,32)

[[max_total_threads_per_threadgroup(256)]]
kernel void im_sam_vision_norm(const device float *x, const device float *weight, const device float *bias, device half *out,
    constant uint &windowed, uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[33];
    const uint win=row/576, local=row%576;
    const uint src=windowed ? (win/3*24+local/24)*72+win%3*24+local%24 : row;
    float sum=0;
    for(uint c=tid;c<1024;c+=256)sum+=x[src*1024+c];
    const float mean=qwv_sum(sum,partial,256,lane,sg)/1024;
    float squares=0;
    for(uint c=tid;c<1024;c+=256){const float v=x[src*1024+c]-mean;squares+=v*v;}
    const float inv=metal::precise::rsqrt(qwv_sum(squares,partial,256,lane,sg)/1024+1e-5f);
    for(uint c=tid;c<1024;c+=256){const float v=(x[src*1024+c]-mean)*inv*weight[c];out[row*1024+c]=half(v+bias[c]);}
}

kernel void im_sam_columns(const device float *x, device float *y, constant uint4 &p, uint i [[thread_position_in_grid]]) {
    if (i >= p.x*p.w) return;
    y[i] = x[(i/p.w)*p.y+p.z+i%p.w];
}

kernel void im_sam_transpose(const device float *x, device float *y, constant uint2 &p, uint i [[thread_position_in_grid]]) {
    if (i < p.x*p.y) y[(i%p.y)*p.x+i/p.y] = x[i];
}

kernel void im_sam_unary(const device float *x, device float *y, constant uint2 &p, uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    float v=x[i];
    y[i] = p.y==0 ? max(v,0.0f) : 1/(1+exp(-v));
}

kernel void im_sam_scale(const device float *x, device float *y, constant float2 &p, uint i [[thread_position_in_grid]]) {
    if (i < as_type<uint>(p.x)) y[i]=x[i]*p.y;
}

kernel void im_sam_mask_values(const device float *x, device float *out, constant uint &count, uint i [[thread_position_in_grid]]) {
    if(i<count)out[i]=2/(1+metal::precise::exp(-x[i]))-1;
}

kernel void im_sam_mask_downsample(const device float *x, const device float *w, const device float *bias, device float *out,
    constant uint &conditioning, uint i [[thread_position_in_grid]]) {
    if(i>=576*576*16)return;
    const uint channel=i%16,row=i/16/576,col=i/16%576;
    float sum=0;
    for(uint ky=0;ky<3;ky++) {
        for(uint kx=0;kx<3;kx++) {
            const int y=int(row*2+ky)-1,z=int(col*2+kx)-1;
            if(y>=0 && z>=0 && y<1152 && z<1152) {
                const uint tap=ky*3+kx;
                sum=fma(x[y*1152+z],w[channel*32*9+tap],sum);
                if(conditioning)sum+=w[(channel*32+16)*9+tap];
            }
        }
    }
    out[i]=sum+bias[channel];
}

// NHWC activations, OIHW checkpoint kernels.
struct SamConv { uint h,w,c,n,k,oh,ow,stride,pad,first,rows; };
kernel void im_sam_deconv_weight(const device float *x, device float *y, constant uint3 &p, uint i [[thread_position_in_grid]]) {
    if(i>=p.x*p.y*4)return;
    const uint c=i%p.y,o=(i/p.y)%p.x,offset=i/(p.y*p.x);
    y[i]=x[(c*p.x+o)*4+offset];
}
kernel void im_sam_deconv_shuffle(const device float *x, const device float *bias, device float *out, constant SamConv &p, uint i [[thread_position_in_grid]]) {
    if(i>=p.oh*p.ow*p.n)return;
    const uint c=i%p.n,y=i/p.n/p.ow,z=i/p.n%p.ow;
    const uint src=((y/2)*p.w+z/2)*(4*p.n)+(y%2*2+z%2)*p.n+c;
    out[i]=x[src]+bias[c];
}
kernel void im_sam_conv_weight(const device float *x, device float *y, constant uint3 &p, uint i [[thread_position_in_grid]]) {
    if (i >= p.x*p.y*p.z) return;
    uint c=i%p.y, s=(i/p.y)%p.z, o=i/(p.y*p.z);
    y[i]=x[(o*p.y+c)*p.z+s];
}
kernel void im_sam_im2col(const device float *x, device float *y, constant SamConv &p, uint i [[thread_position_in_grid]]) {
    uint width=p.k*p.k*p.c;
    if (i>=p.rows*width) return;
    uint row=p.first+i/width, c=i%p.c, s=(i/ p.c)%(p.k*p.k);
    int yy=int(row/p.ow*p.stride+s/p.k)-int(p.pad), xx=int(row%p.ow*p.stride+s%p.k)-int(p.pad);
    y[i]=(yy<0 || xx<0 || yy>=int(p.h) || xx>=int(p.w)) ? 0 : x[(yy*p.w+xx)*p.c+c];
}
kernel void im_sam_deconv(const device float *x, const device float *w, const device float *b, device float *y, constant SamConv &p, uint i [[thread_position_in_grid]]) {
    if (i>=p.oh*p.ow*p.n) return;
    uint o=i%p.n, yy=i/p.n/p.ow, xx=i/p.n%p.ow;
    float sum=b[o];
    for(uint c=0;c<p.c;c++) sum+=x[((yy/2)*p.w+xx/2)*p.c+c]*w[((c*p.n+o)*2+yy%2)*2+xx%2];
    y[i]=sum;
}
kernel void im_sam_depthwise(const device float *x, const device float *w, const device float *b, device float *y, constant SamConv &p, uint i [[thread_position_in_grid]]) {
    if(i>=p.h*p.w*p.c) return;
    uint c=i%p.c, yy=i/p.c/p.w, xx=i/p.c%p.w;
    float sum=b[c];
    for(uint ky=0;ky<p.k;ky++) for(uint kx=0;kx<p.k;kx++) {
        int sy=int(yy+ky)-int(p.pad), sx=int(xx+kx)-int(p.pad);
        if(sy>=0 && sx>=0 && sy<int(p.h) && sx<int(p.w)) sum+=x[(sy*p.w+sx)*p.c+c]*w[(c*p.k+ky)*p.k+kx];
    }
    y[i]=sum;
}
kernel void im_sam_resize(const device float *x, device float *y, constant uint4 &p, constant uint2 &extra, uint i [[thread_position_in_grid]]) {
    uint oh=extra.x, ow=extra.y;
    if(i>=oh*ow*p.z) return;
    uint c=i%p.z, yy=i/p.z/ow, xx=i/p.z%ow;
    if(p.w==0) { y[i]=x[((yy*p.x/oh)*p.y+xx*p.y/ow)*p.z+c]; return; }
    float sy=max(0.0f,(yy+0.5f)*p.x/oh-0.5f), sx=max(0.0f,(xx+0.5f)*p.y/ow-0.5f);
    uint y0=min(uint(sy),p.x-1), x0=min(uint(sx),p.y-1), y1=min(y0+1,p.x-1), x1=min(x0+1,p.y-1);
    float fy=sy-y0, fx=sx-x0;
    y[i]=mix(mix(x[(y0*p.y+x0)*p.z+c],x[(y0*p.y+x1)*p.z+c],fx),mix(x[(y1*p.y+x0)*p.z+c],x[(y1*p.y+x1)*p.z+c],fx),fy);
}
kernel void im_sam_group_norm(const device float *x, const device float *w, const device float *b, device float *y, constant uint3 &p, uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sums[256], squares[256];
    uint gc=p.y/p.z, count=p.x*gc;
    float sum=0,sq=0;
    for(uint j=lane;j<count;j+=256) {float v=x[(j/gc)*p.y+group*gc+j%gc];sum+=v;sq+=v*v;}
    sums[lane]=sum; squares[lane]=sq; threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint s=128;s;s>>=1) {if(lane<s){sums[lane]+=sums[lane+s];squares[lane]+=squares[lane+s];}threadgroup_barrier(mem_flags::mem_threadgroup);}
    float mean=sums[0]/count, inv=rsqrt(max(0.0f,squares[0]/count-mean*mean)+1e-5f);
    for(uint j=lane;j<count;j+=256){uint c=group*gc+j%gc, at=(j/gc)*p.y+c;y[at]=(x[at]-mean)*inv*w[c]+b[c];}
}
kernel void im_sam_window(const device float *x, device float *y, constant uint3 &p, uint i [[thread_position_in_grid]]) {
    if(i>=72*72*p.x) return;
    uint c=i%p.x, row=i/p.x, win=row/576, local=row%576;
    uint src=((win/3*24+local/24)*72+win%3*24+local%24)*p.x+c;
    if(p.y==0)y[i]=x[src];else y[src]=x[i];
}
struct SamRope {uint rows,heads,dim,spatial,side;float scale;};
kernel void im_sam_rope_table(device float *out, constant SamRope &p, uint i [[thread_position_in_grid]]) {
    if(i>=p.side*p.side*p.dim/2)return;
    const uint pair=i%(p.dim/2), pos=i/(p.dim/2), quarter=p.dim/4;
    const float coord=pair<quarter ? pos%p.side : pos/p.side;
    const float angle=coord*p.scale*pow(10000.0f,-float(pair%quarter)/quarter);
    out[2*i]=cos(angle);out[2*i+1]=sin(angle);
}
kernel void im_sam_rope_cached(const device float *x, const device float *table, device float *y, constant SamRope &p, uint i [[thread_position_in_grid]]) {
    const uint width=p.heads*p.dim;
    if(i>=p.rows*width)return;
    const uint row=i/width,d=i%p.dim;
    if(row>=p.spatial){y[i]=x[i];return;}
    const uint at=(row%(p.side*p.side))*p.dim+(d/2)*2;
    y[i]=x[i]*table[at]+x[i^1]*table[at+1]*((d&1)?1.0f:-1.0f);
}
template<typename T>
inline void sam_qkv_rope(const device float *x, const device float *table, device T *q, device T *k, device T *v, constant uint &side, uint i) {
    if(i>=5184*1024)return;
    const uint row=i/1024,c=i%1024,d=c%64,src=row*3072+c,at=(row%(side*side))*64+(d/2)*2;
    const float cosine=table[at], sine=table[at+1]*((d&1)?1.0f:-1.0f);
    q[i]=x[src]*cosine+x[src^1]*sine;
    k[i]=x[src+1024]*cosine+x[(src+1024)^1]*sine;
    v[i]=x[src+2048];
}
#define IM_SAM_QKV(NAME,TYPE) \
kernel void NAME(const device float *x, const device float *table, device TYPE *q, device TYPE *k, device TYPE *v, constant uint &side, uint i [[thread_position_in_grid]]) { \
    sam_qkv_rope(x,table,q,k,v,side,i); \
}
IM_SAM_QKV(im_sam_qkv_rope,float)
IM_SAM_QKV(im_sam_qkv_rope_h,half)
kernel void im_sam_rope(const device float *x, device float *y, constant SamRope &p, uint i [[thread_position_in_grid]]) {
    uint width=p.heads*p.dim;
    if(i>=p.rows*width)return;
    uint row=i/width, d=i%p.dim, pair=d/2, quarter=p.dim/4;
    if(row>=p.spatial) {y[i]=x[i];return;}
    uint pos=row%(p.side*p.side);
    float coord=pair<quarter ? pos%p.side : pos/p.side;
    float angle=coord*p.scale*pow(10000.0f,-float(pair%quarter)/quarter);
    uint other=(d&1) ? i-1 : i+1;
    y[i]=x[i]*cos(angle)+x[other]*sin(angle)*((d&1)?1.0f:-1.0f);
}
// Decoder attention includes a learned box bias, and uses head dimensions 16/32.
kernel void im_sam_attention(const device float *q,const device float *k,const device float *v,const device float *bias,device float *out,constant uint4 &p,constant uint &has_bias,uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    uint row=group.x,head=group.y,dim=p.w;
    float top=-INFINITY,total=0,acc[8]={0,0,0,0,0,0,0,0};
    for(uint j=0;j<p.y;j++) {
        float dot=0;
        for(uint d=lane;d<dim;d+=32)dot+=q[(row*p.z+head)*dim+d]*k[(j*p.z+head)*dim+d];
        dot=simd_sum(dot)*rsqrt(float(dim));
        if(has_bias)dot+=bias[(head*p.x+row)*p.y+j];
        float next=max(top,dot),old=exp(top-next),prob=exp(dot-next);
        for(uint t=0;t<(dim+31)/32;t++){uint d=lane+t*32;if(d<dim)acc[t]=acc[t]*old+prob*v[(j*p.z+head)*dim+d];}
        total=total*old+prob;top=next;
    }
    for(uint t=0;t<(dim+31)/32;t++){uint d=lane+t*32;if(d<dim)out[(row*p.z+head)*dim+d]=acc[t]/total;}
}
struct SamMemoryMasks { uint pixels, channels, active, conditioning, hard; float scale, bias; };
kernel void im_sam_memory_masks(const device float* logits [[buffer(0)]],
    device float* out [[buffer(1)]], constant SamMemoryMasks& p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= p.pixels * p.channels) return;
  uint c = i % p.channels, pixel = i / p.channels, slot = c % 16;
  if (!(p.active & (1u << slot))) { out[i] = 0; return; }
  if (c >= 16) { out[i] = (p.conditioning & (1u << slot)) ? 1.0f : 0.0f; return; }
  float value = logits[pixel * (p.channels == 1 ? 1 : 16) + slot];
  float prob = p.hard ? float(value > 0) : 1.0f / (1.0f + exp(-value));
  out[i] = prob * p.scale + p.bias;
}
