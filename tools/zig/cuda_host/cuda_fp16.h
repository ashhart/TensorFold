#pragma once
#include "cuda_runtime.h"
struct __half_raw { unsigned short x; };
struct __half { unsigned short x; __half()=default; __host__ __device__ explicit __half(float); __host__ __device__ __half(__half_raw); };
struct __half2_raw { unsigned short x,y; };
struct __half2 { __half x,y; __half2()=default; __host__ __device__ __half2(__half2_raw); };
using half=__half; using half2=__half2;
__device__ float __half2float(half); __device__ half __ushort_as_half(unsigned short);
__device__ unsigned short __half_as_ushort(half); __device__ half __float2half(float);
__device__ half __float2half_rn(float); __device__ half __uint2half_rn(unsigned);
__device__ float2 __half22float2(half2); __device__ half2 __half2half2(half);
__device__ half2 __float2half2_rn(float); __device__ half2 __floats2half2_rn(float,float);
__device__ half2 __halves2half2(half,half); __device__ half __low2half(half2); __device__ half __high2half(half2);
__device__ float __low2float(half2); __device__ float __high2float(half2);
__device__ half2 __hadd2(half2,half2); __device__ half2 __hsub2(half2,half2); __device__ half2 __hmul2(half2,half2);
__device__ half2 __hfma2(half2,half2,half2); __device__ half __hadd(half,half);
