#include <math.h>
#include <stdint.h>
#include <string.h>
#define LW_DEV
#define LW_INLINE inline
static float lw_bits_float(unsigned x) {float y;memcpy(&y,&x,4);return y;}
static unsigned lw_float_bits(float x) {unsigned y;memcpy(&y,&x,4);return y;}
#define LW_BITS_FLOAT(x) lw_bits_float(x)
#define LW_FLOAT_BITS(x) lw_float_bits(x)
#define LW_TANH(x) tanhf(x)
#define LW_EXP(x) expf(x)
#define LW_ABS(x) fabsf(x)
#define LW_SQRT(x) sqrtf(x)
#define LW_LOG(x) logf(x)
#define LW_SIN(x) sinf(x)
#define LW_COS(x) cosf(x)
#define LW_INFINITY INFINITY
#define LW_FMA(a,b,c) fmaf(a,b,c)
#define LW_ISNAN(x) __builtin_isnan(x)
#include "../../kernels/common/living_rows.h"
extern "C" void tf_living_rows_cpu(unsigned op,LwParams p,unsigned char** views) {
    for(unsigned id=0;id<p.u[15];id++)lw_rows(op,id,p,views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7],views[8],views[9],views[10]);
}
#include "../../kernels/common/living_gdn.h"
#include "../../kernels/common/living_qsa.h"
extern "C" void tf_living_gdn_cpu(unsigned op,LwParams p,unsigned char** views) {
    for(unsigned id=0;id<p.u[15];id++)lw_gdn(op,id,p,views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7],views[8],views[9],views[10]);
}
extern "C" void tf_living_qsa_cpu(unsigned op,LwParams p,unsigned char** views) {
    for(unsigned id=0;id<p.u[15];id++)lw_qsa(op,id,p,views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7],views[8],views[9],views[10]);
}

#include "../../kernels/common/living_gemma.h"
extern "C" void tf_living_gemma_cpu(unsigned op,LwParams p,unsigned char** views) {
    for(unsigned id=0;id<p.u[15];id++)lw_gemma(op,id,p,views[0],views[1],views[2],views[3],views[4],views[5],views[6],views[7],views[8],views[9],views[10]);
}
