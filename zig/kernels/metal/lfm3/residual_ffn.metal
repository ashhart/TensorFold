// Development-only residual-weight prompt FFN; finite tiles plus public NF fallback.
struct LfmResidualArgs { int K, N, R; float one; };
using namespace tfp;
inline void l3_nax_mma(thread frag<float>& lo,thread frag<float>& hi,
 const thread frag<bfloat>& x,const thread frag<bfloat>& w0,const thread frag<bfloat>& w1){
 using namespace mpp::tensor_ops;
 constexpr auto shape=matmul2d_descriptor(16,32,16,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
 matmul2d<shape,execution_simdgroup> op;
 auto a=op.template get_left_input_cooperative_tensor<bfloat,bfloat,float>();
 auto b=op.template get_right_input_cooperative_tensor<bfloat,bfloat,float>();
 auto c=op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
 TF_UNROLL for(short e=0;e<8;++e){a[e]=x[e];b[e]=w0[e];b[e+8]=w1[e];c[e]=lo[e];c[e+8]=hi[e];}
 op.run(a,b,c);
 TF_UNROLL for(short e=0;e<8;++e){lo[e]=c[e];hi[e]=c[e+8];}
}
template<int BITS> inline uint l3_nax_code(const device uchar* w,int k){
 const uint bit=uint(k)*BITS,byte=bit/8,shift=bit%8;
 uint v=uint(w[byte]);if(shift+BITS>8)v|=uint(w[byte+1])<<8;
 return(v>>shift)&((1u<<BITS)-1);
}
template<int BITS>
[[kernel,max_total_threads_per_threadgroup(128)]] void lfm3_nax_prompt(
 const device bfloat* X [[buffer(0)]],const device uchar* W [[buffer(1)]],
 const device bfloat* S [[buffer(2)]],const device bfloat* B [[buffer(3)]],
 constant LfmResidualArgs& a [[buffer(4)]],device bfloat* Y [[buffer(5)]],const device float* XS [[buffer(6)]],device uint* tileflags [[buffer(7)]],
 uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint2 tg [[threadgroup_position_in_grid]]){
 const int rb=int(tg.y)*64,nb=int(tg.x)*64,G=a.K/64;
 const int mr=int(sg/2)*32,nc=int(sg%2)*32;const short2 home=frag_home(ushort(lane));
 threadgroup bfloat weights[64*72],residual[64*72];threadgroup uint flags[4];bool local_bad=false;
 frag<float> low[2][2];TF_UNROLL for(short m=0;m<2;++m)TF_UNROLL for(short j=0;j<2;++j)low[m][j]=frag<float>(0);
 frag<float> acc[2][2];TF_UNROLL for(short m=0;m<2;++m)TF_UNROLL for(short j=0;j<2;++j)acc[m][j]=frag<float>(0);
 for(int g=0;g<G;++g){
  if(BITS==4){
   const int col=int(tid)/2,blockhalf=int(tid)%2;
   const device uchar* w=W+size_t(nb+col)*(a.K/2)+g*32+blockhalf*16;
   const uint4 packed=*((const device uint4*)w);
   const float scale=float(S[size_t(nb+col)*G+g]),bias=float(B[size_t(nb+col)*G+g]);
   TF_UNROLL for(int z=0;z<32;++z){uint code=(packed[z/8]>>(4*(z%8)))&15u;
#ifdef L3_NAX_BAD_PACK
    code^=1u;
#endif
#ifdef L3_NAX_BAD_BIAS
    const bfloat value=bfloat(float(code)*scale);
#else
    const bfloat value=bfloat(fma(float(code),scale,bias));
#endif
    weights[col*72+blockhalf*32+z]=value;
#ifdef L3_NAX_BAD_BIAS
    residual[col*72+blockhalf*32+z]=bfloat(float(code)*scale-float(value));
#else
    residual[col*72+blockhalf*32+z]=bfloat(fma(float(code),scale,bias)-float(value));
#endif
    local_bad|=!isfinite(float(value));
   }
  }else{
  for(int at=int(tid);at<4096;at+=128){
   const int col=at/64,k=at%64;const device uchar* w=W+size_t(nb+col)*(a.K*BITS/8);
   const float scale=float(S[size_t(nb+col)*G+g]),bias=float(B[size_t(nb+col)*G+g]);
#ifdef L3_NAX_BAD_BIAS
   weights[col*72+k]=bfloat(float(l3_nax_code<BITS>(w,g*64+k))*scale);
#else
   weights[col*72+k]=bfloat(fma(float(l3_nax_code<BITS>(w,g*64+k)),scale,bias));
#endif
   residual[col*72+k]=bfloat(fma(float(l3_nax_code<BITS>(w,g*64+k)),scale,bias)-float(weights[col*72+k]));
   local_bad|=!isfinite(float(weights[col*72+k]));
  }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  TF_UNROLL for(short q=0;q<4;++q){
   frag<bfloat> w0,w1;frag_get_tg(w0,weights,72,nc,q*16,home);frag_get_tg(w1,weights,72,nc+16,q*16,home);
   frag<bfloat> r0,r1;frag_get_tg(r0,residual,72,nc,q*16,home);frag_get_tg(r1,residual,72,nc+16,q*16,home);
   TF_UNROLL for(short m=0;m<2;++m){frag<bfloat>x;frag_get_in(x,X,a.K,rb+mr+m*16,g*64+q*16,home,a.R,a.K);TF_UNROLL for(short z=0;z<8;++z)local_bad|=!isfinite(float(x[z]));l3_nax_mma(acc[m][0],acc[m][1],x,w0,w1);l3_nax_mma(low[m][0],low[m][1],x,r0,r1);}
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 TF_UNROLL for(short m=0;m<2;++m)TF_UNROLL for(short j=0;j<2;++j)TF_UNROLL for(short z=0;z<8;++z)acc[m][j][z]+=low[m][j][z];
 TF_UNROLL for(short m=0;m<2;++m)TF_UNROLL for(short j=0;j<2;++j)TF_UNROLL for(short z=0;z<8;++z)local_bad|=!isfinite(float(bfloat(acc[m][j][z])));
 const uint any_bad=uint(simd_any(local_bad));if(lane==0)flags[sg]=any_bad;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0){uint flag=flags[0]|flags[1]|flags[2]|flags[3];
#ifdef L3_NAX_BAD_FLAG
 if(tg.x==0&&tg.y==0)flag=0;
#endif
 tileflags[size_t(tg.y)*(a.N/64)+tg.x]=flag;}
 TF_UNROLL for(short m=0;m<2;++m)TF_UNROLL for(short j=0;j<2;++j)
  frag_put_in(acc[m][j],Y,a.N,rb+mr+m*16,nb+nc+j*16,home,a.R,a.N);
}
// NF input/parameter masks are produced on-device; no CPU input scans in timed path.
[[kernel]] void lfm3_nax_mask(const device bfloat* X [[buffer(0)]],const device bfloat* S [[buffer(1)]],const device bfloat* B [[buffer(2)]],constant LfmResidualArgs& a [[buffer(3)]],device uint* mask [[buffer(4)]],uint i [[thread_position_in_grid]]){
 uint bad=0;
 if(i<uint(a.R)){for(int k=0;k<a.K;++k)bad|=uint(!isfinite(float(X[size_t(i)*a.K+k])));mask[i]=bad;}
 else if(i<uint(a.R+a.N)){const int c=int(i)-a.R;for(int g=0;g<a.K/64;++g)bad|=uint(!isfinite(float(S[size_t(c)*(a.K/64)+g]))||!isfinite(float(B[size_t(c)*(a.K/64)+g])));mask[i]=bad;}
}
template<int BITS>
[[kernel,max_total_threads_per_threadgroup(64)]] void lfm3_nax_nf_repair(
 const device bfloat* X [[buffer(0)]],const device uint* W [[buffer(1)]],const device bfloat* S [[buffer(2)]],const device bfloat* B [[buffer(3)]],constant LfmResidualArgs& a [[buffer(4)]],device bfloat* Y [[buffer(5)]],const device uint* mask [[buffer(6)]],uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],uint2 tg [[threadgroup_position_in_grid]]){
 const int rowbase=int(tg.x)*64,colbase=int(tg.y)*64;
 if(mask[size_t(tg.x)*(a.N/64)+tg.y]==0)return;
 for(int rp=0;rp<32;++rp)for(int cp=0;cp<8;++cp){
 const int first=rowbase+rp*2,col=colbase+cp*8+int(sg)*4;
 if(first>=a.R)continue;
#ifdef L3_NAX_BAD_NF
 if(lane==0)for(int r=0;r<2&&first+r<a.R;++r)for(int j=0;j<4;++j)Y[size_t(first+r)*a.N+col+j]=bfloat(0);
 continue;
#endif

 const device uchar* wr=(const device uchar*)W+size_t(col)*(a.K*BITS/8);const device bfloat* sr=S+size_t(col)*(a.K/64),*br=B+size_t(col)*(a.K/64);
 if(first+1<a.R){float acc[2][4];tf_rp_rows<4,2,BITS,false>(wr,sr,br,X+size_t(first)*a.K,a.K,64,lane,acc);if(lane==0)for(int r=0;r<2;++r)for(int j=0;j<4;++j)Y[size_t(first+r)*a.N+col+j]=tf_rp_out<false>(acc[r][j]);}
 else{float acc[1][4];tf_rp_rows<4,1,BITS,false>(wr,sr,br,X+size_t(first)*a.K,a.K,64,lane,acc);if(lane==0)for(int j=0;j<4;++j)Y[size_t(first)*a.N+col+j]=tf_rp_out<false>(acc[0][j]);}
}
}

#define L3_NAX_ENTRY(B) template [[host_name("lfm3_nax_prompt_q" #B)]] [[kernel]] decltype(lfm3_nax_prompt<B>) lfm3_nax_prompt<B>; template [[host_name("lfm3_nax_nf_repair_q" #B)]] [[kernel]] decltype(lfm3_nax_nf_repair<B>) lfm3_nax_nf_repair<B>;
L3_NAX_ENTRY(4)
