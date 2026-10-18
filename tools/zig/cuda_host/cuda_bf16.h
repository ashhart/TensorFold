#pragma once
#include "cuda_fp16.h"
struct __nv_bfloat16 { unsigned short x; };
struct __nv_bfloat162 { __nv_bfloat16 x,y; };
__device__ float __bfloat162float(__nv_bfloat16); __device__ __nv_bfloat16 __float2bfloat16(float);
__device__ __nv_bfloat16 __float2bfloat16_rn(float); __device__ __nv_bfloat16 __ushort_as_bfloat16(unsigned short);
__device__ unsigned short __bfloat16_as_ushort(__nv_bfloat16);
__device__ float2 __bfloat1622float2(__nv_bfloat162);
__device__ __nv_bfloat162 __floats2bfloat162_rn(float,float);
__device__ __nv_bfloat162 __halves2bfloat162(__nv_bfloat16,__nv_bfloat16);
__device__ __nv_bfloat162 __hadd2(__nv_bfloat162,__nv_bfloat162);
__device__ __nv_bfloat162 __hsub2(__nv_bfloat162,__nv_bfloat162);
__device__ __nv_bfloat162 __hmul2(__nv_bfloat162,__nv_bfloat162);
__device__ __nv_bfloat162 __hfma2(__nv_bfloat162,__nv_bfloat162,__nv_bfloat162);
__device__ float __low2float(__nv_bfloat162); __device__ float __high2float(__nv_bfloat162);
