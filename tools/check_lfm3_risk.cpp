#include <cstdint>
#include <cstdio>
#include <cstdlib>
using uint=uint32_t;
#include "predicate.inc"
bool finite(uint a){return(a&0x7f800000u)!=0x7f800000u;}
bool nz(uint a){return(a&0x7fffffffu)!=0;}
bool product(uint a,uint b){uint ae=a>>23&255,be=b>>23&255,s=ae+be;return(finite(a)&finite(b)&nz(a)&nz(b))&((ae==0)|(be==0)|(s<=129)|(s>=380));}
bool sum(uint a,uint b,uint c,uint d,uint r){uint e=r>>23&255;return(finite(a)&finite(b)&finite(c)&finite(d))&((e>=253)|((e==0)&((nz(a)&nz(b))|(nz(c)&nz(d)))));}
bool original(uint x,uint a,uint y,uint b,uint r){return product(x,a)|product(y,b)|sum(x,a,y,b,r);}
void check(uint x,uint a,uint y,uint b,uint r){if(original(x,a,y,b,r)!=l3_pv_risk(x,y,r,l3_risk_decode(a),l3_risk_decode(b))){printf("mismatch %08x %08x %08x %08x %08x\n",x,a,y,b,r);abort();}}
int main(){
 uint64_t pairs=0,classes=0,random=0;uint frac[]={0,1,0x400000,0x7fffff};
 // All 256 exponent classes, both signs, and exact zero/nonzero mantissa categories.
 for(uint ae=0;ae<256;++ae)for(uint be=0;be<256;++be)for(uint sa=0;sa<2;++sa)for(uint sb=0;sb<2;++sb)for(uint af:frac)for(uint bf:frac){
  uint a=sa<<31|ae<<23|af,b=sb<<31|be<<23|bf;auto x=l3_risk_decode(a),y=l3_risk_decode(b);
  if(x.e!=ae||y.e!=be||x.finite!=finite(a)||y.finite!=finite(b)||x.nonzero!=nz(a)||y.nonzero!=nz(b))abort();
  bool p=(x.finite&y.finite&x.nonzero&y.nonzero)&((x.e==0)|(y.e==0)|((x.e+y.e-130u)>=250u));
  if(p!=product(a,b))abort();check(a,b,0,0,0);++pairs;
 }
 // The complete Boolean identity factoring the three original predicates, including infeasible states.
 for(uint bits=0;bits<256;++bits){bool xf=bits&1,yf=bits&2,xn=bits&4,yn=bits&8,xr=bits&16,yr=bits&32,zero=bits&64,high=bits&128;
  bool old=(xf&xn&xr)|(yf&yn&yr)|(xf&yf&(high|(zero&(xn|yn))));
  bool now=((xf&xn)&(xr|(yf&zero)))|((yf&yn)&(yr|(xf&zero)))|(xf&yf&high);if(old!=now)abort();
 }
 // Full fused source over Cartesian zero/subnormal/normal/infinity/NaN operand classes and every output exponent.
 uint cls[]={0,0x80000000,1,0x80000001,0x3f800000,0xbf800000,0x7f800000,0xff800000,0x7fc00000,0xffc00000};
 for(uint x:cls)for(uint a:cls)for(uint y:cls)for(uint b:cls)for(uint e=0;e<256;++e)for(uint sign=0;sign<2;++sign)for(uint f:frac){check(x,a,y,b,sign<<31|e<<23|f);++classes;}
 uint seed=1502;auto next=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;};
 for(uint i=0;i<10000000;++i){uint x=next(),a=next(),y=next(),b=next(),r=next();check(x,a,y,b,r);++random;}
 // Predicate controls must distinguish a missing high-result check and a missing subnormal-product check.
 if(!original(0x3f800000,0x3f800000,0x3f800000,0x3f800000,253u<<23))abort();
 if(!original(1,0x3f800000,0,0,0x3f800000))abort();
 printf("{\"pair_class_checks\":%llu,\"fused_operand_result_classes\":%llu,\"random_fused_words\":%llu,\"boolean_truth_table_rows\":256,\"control_witnesses\":2,\"mismatches\":0,\"scope\":\"Exhaustive predicate category identity plus deterministic random sample; not exhaustive floating-point arithmetic or GPU qualification\"}\n",pairs,classes,random);
}
