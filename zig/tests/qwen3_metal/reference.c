// Independent FP32 Qwen3 formulas. BLAS supplies only the matrix product, never model or tokenizer code.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static float widen(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }
void q3_widen(float *y, const uint16_t *x, size_t n) { for (size_t i=0;i<n;i++) y[i]=widen(x[i]); }
int q3_linear(float *y, const float *x, const uint16_t *stored, int rows, int n, int k) {
  float *w=malloc((size_t)n*k*sizeof(float)); if (!w) return 1;
  q3_widen(w,stored,(size_t)n*k);
  cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,n,k,1,x,k,w,k,0,y,n);
  free(w); return 0;
}
void q3_norm(float *y, const float *x, const uint16_t *w, int rows, int width, float eps) {
  for (int r=0;r<rows;r++) {
    float sum=0; for (int i=0;i<width;i++) sum+=x[(size_t)r*width+i]*x[(size_t)r*width+i];
    float inv=1.0f/sqrtf(sum/(float)width+eps);
    for (int i=0;i<width;i++) y[(size_t)r*width+i]=x[(size_t)r*width+i]*inv*widen(w[i]);
  }
}
void q3_rope(float *x, int rows, int heads, double theta) {
  for (int r=0;r<rows;r++) for (int h=0;h<heads;h++) for (int d=0;d<64;d++) {
    float angle=(float)r*(float)(1.0/pow(theta,(double)(2*d)/128.0));
    float c=cosf(angle),s=sinf(angle); size_t a=((size_t)r*heads+h)*128+d;
    float lo=x[a],hi=x[a+64]; x[a]=lo*c-hi*s; x[a+64]=hi*c+lo*s;
  }
}
int q3_attention(float *out, const float *q, const float *k, const float *v, int rows, int heads, int kv) {
  float *scores=malloc((size_t)rows*rows*sizeof(float)); if (!scores) return 1;
  for (int h=0;h<heads;h++) {
    int kh=h/(heads/kv);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,rows,rows,128,1/sqrtf(128),q+h*128,heads*128,k+kh*128,kv*128,0,scores,rows);
    for (int r=0;r<rows;r++) {
      float *p=scores+(size_t)r*rows,top=-INFINITY,sum=0;
      for (int j=0;j<=r;j++) if (p[j]>top) top=p[j];
      for (int j=0;j<=r;j++) { p[j]=expf(p[j]-top); sum+=p[j]; }
      for (int j=0;j<=r;j++) p[j]/=sum;
      for (int j=r+1;j<rows;j++) p[j]=0;
    }
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,rows,128,rows,1,scores,rows,v+kh*128,kv*128,0,out+h*128,heads*128);
  }
  free(scores); return 0;
}
void q3_add(float *x, const float *y, size_t n) { for (size_t i=0;i<n;i++) x[i]+=y[i]; }
void q3_silu(float *g, const float *u, size_t n) { for (size_t i=0;i<n;i++) g[i]=(g[i]/(1+expf(-g[i])))*u[i]; }
