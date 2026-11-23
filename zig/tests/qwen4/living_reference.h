// Binary64 formula replay propagates absolute FP32 roundoff through the scalar expressions and BF16 store intervals.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
struct LwTerm {LwTerm* previous;double value;};
struct LwChunk {LwChunk* previous;unsigned used;LwTerm nodes[4096];};
static LwChunk* lw_chunks=nullptr;
static unsigned lw_accumulator=0,lw_order=0;
static bool lw_memory_failure=false;
static double lw_bf64(double);
static double lw_acc_round(double x){return lw_accumulator==1?double(float(x)):lw_accumulator==2?double((_Float16)x):lw_bf64(x);}
static double lw_bf64(double);
static LwTerm* lw_term(LwTerm* previous,double value){
    if(!lw_chunks||lw_chunks->used==4096){auto* c=(LwChunk*)malloc(sizeof(LwChunk));if(!c){lw_memory_failure=true;return previous;}c->previous=lw_chunks;c->used=0;lw_chunks=c;}
    auto* n=&lw_chunks->nodes[lw_chunks->used++];n->previous=previous;n->value=value;return n;
}
static void lw_terms_clear(){while(lw_chunks){auto* previous=lw_chunks->previous;free(lw_chunks);lw_chunks=previous;}}
static double lw_pair(const double* x,unsigned n){if(!n)return 0;if(n==1)return lw_acc_round(x[0]);const unsigned m=n/2;return lw_acc_round(lw_pair(x,m)+lw_pair(x+m,n-m));}
static double lw_terms_value(LwTerm* head){
    unsigned n=0;for(auto* p=head;p;p=p->previous)n++;
    auto* values=(double*)malloc(n*sizeof(double));if(!values){lw_memory_failure=true;return NAN;}
    unsigned at=n;for(auto* p=head;p;p=p->previous)values[--at]=p->value;
    double value=0;
    if(lw_order==1)value=lw_pair(values,n);
    else if(lw_order==2){double lanes[32]={};for(unsigned lane=0;lane<32;lane++)for(unsigned i=lane;i<n;i+=32)lanes[lane]=lw_acc_round(lanes[lane]+lw_acc_round(values[i]));value=lw_pair(lanes,32);}
    else for(unsigned i=0;i<n;i++)value=lw_acc_round(value+lw_acc_round(values[i]));
    free(values);return value;
}
struct LwReference {
    double value,error;
    unsigned depth;
    LwTerm* terms;
    LwReference()=default;
    LwReference(double v):value(v),error(0),depth(0),terms(nullptr){}
    LwReference(double v,double e,unsigned d):value(v),error(__builtin_isfinite(v)?e:0),depth(d),terms(nullptr){}
    LwReference resolved() const {return terms?LwReference(lw_terms_value(terms),0,depth):*this;}
};
static LwReference lw_interval(double v,double error,unsigned depth){
    if(!__builtin_isfinite(v))return {v,0,depth};
    const double low=double(float(nextafter(v-error,-INFINITY)));
    const double high=double(float(nextafter(v+error,INFINITY)));
    return {v,fmax(fabs(v-low),fabs(high-v)),depth};
}
static unsigned lw_depth(LwReference a,LwReference b){return (a.depth>b.depth?a.depth:b.depth)+1;}
static constexpr double lw_u=0x1p-24;
static LwReference operator+(LwReference a,LwReference b){a=a.resolved();b=b.resolved();const double v=a.value+b.value;return lw_interval(v,a.error+b.error,lw_depth(a,b));}
static LwReference operator-(LwReference a,LwReference b){a=a.resolved();b=b.resolved();const double v=a.value-b.value;return lw_interval(v,a.error+b.error,lw_depth(a,b));}
static LwReference operator-(LwReference a){a=a.resolved();return {-a.value,a.error,a.depth};}
static LwReference operator*(LwReference a,LwReference b){a=a.resolved();b=b.resolved();const double v=a.value*b.value;return lw_interval(v,fabs(a.value)*b.error+fabs(b.value)*a.error+a.error*b.error,lw_depth(a,b));}
static LwReference operator/(LwReference a,LwReference b){a=a.resolved();b=b.resolved();const double v=a.value/b.value;const double den=fabs(b.value)-b.error;return lw_interval(v,den>0?(a.error+fabs(v)*b.error)/den:INFINITY,lw_depth(a,b));}
static LwReference& operator+=(LwReference& a,LwReference b){
    if(lw_accumulator){b=b.resolved();if(!a.terms&&a.value!=0)a.terms=lw_term(nullptr,a.value);a.terms=lw_term(a.terms,b.value);a.value+=b.value;return a;}
    a=a+b;return a;
}
static LwReference& operator*=(LwReference& a,LwReference b){a=a*b;return a;}
static bool operator==(LwReference a,LwReference b){a=a.resolved();b=b.resolved();return a.value==b.value;}
static bool operator<(LwReference a,LwReference b){a=a.resolved();b=b.resolved();return a.value<b.value;}
static bool operator>(LwReference a,LwReference b){a=a.resolved();b=b.resolved();return a.value>b.value;}
static LwReference lw_abs(LwReference a){a=a.resolved();return {fabs(a.value),a.error,a.depth};}
static LwReference lw_exp(LwReference a){a=a.resolved();if(a.error==0&&a.value==0)return {1,0,a.depth+1};const double v=exp(a.value);return lw_interval(v,v*expm1(a.error)+4*lw_u*fabs(v),a.depth+4);}
static LwReference lw_log(LwReference a){a=a.resolved();if(a.error==0&&a.value==1)return {0,0,a.depth+1};const double v=log(a.value);return lw_interval(v,a.value>a.error?-log1p(-a.error/a.value)+4*lw_u*fabs(v):INFINITY,a.depth+4);}
static LwReference lw_sqrt(LwReference a){a=a.resolved();const double v=sqrt(a.value);return lw_interval(v,a.value>=a.error?v-sqrt(a.value-a.error):INFINITY,a.depth+1);}
static LwReference lw_sin(LwReference a){a=a.resolved();if(a.error==0&&a.value==0)return {0,0,a.depth+1};const double v=sin(a.value);return lw_interval(v,a.error+4*lw_u,a.depth+4);}
static LwReference lw_cos(LwReference a){a=a.resolved();if(a.error==0&&a.value==0)return {1,0,a.depth+1};const double v=cos(a.value);return lw_interval(v,a.error+4*lw_u,a.depth+4);}
static LwReference lw_fma(LwReference a,LwReference b,LwReference c){const auto v=a*b+c;return v;}
static double lw_bf64(double x){
    uint64_t bits;memcpy(&bits,&x,8);
    const uint64_t sign=bits>>63;
    if(__builtin_isnan(x))return NAN;
    if(__builtin_isinf(x))return x;
    const double ax=fabs(x);
    if(ax==0)return x;
    const double quantum=ax<0x1p-126?0x1p-133:ldexp(1.0,ilogb(ax)-7);
    const double rounded=nearbyint(ax/quantum)*quantum;
    const double out=rounded>=0x1.ffp127?INFINITY:rounded;
    return sign?-out:out;
}
static LwReference lw_bf_ref(LwReference a){a=a.resolved();
    const double v=lw_bf64(a.value);
    if(!__builtin_isfinite(a.value))return {v,0,a.depth};
    const double lo=lw_bf64(a.value-a.error),hi=lw_bf64(a.value+a.error);
    return {v,fmax(fabs(v-lo),fabs(hi-v)),a.depth};
}
static float lw_ref_bits(unsigned x){float v;memcpy(&v,&x,4);return v;}
static unsigned lw_ref_float_bits(LwReference x){const float v=float(x.value);unsigned bits;memcpy(&bits,&v,4);return bits;}
