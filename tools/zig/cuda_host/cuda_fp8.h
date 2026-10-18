#pragma once
#include "cuda_fp16.h"
using __nv_fp8_storage_t=unsigned char; using __nv_fp8x2_storage_t=unsigned short;
enum { __NV_SATFINITE, __NV_E4M3 };
struct __nv_fp8_e4m3 { unsigned char __x; __nv_fp8_e4m3()=default; __device__ explicit __nv_fp8_e4m3(float); __device__ explicit operator float() const; };
__device__ __nv_fp8_storage_t __nv_cvt_float_to_fp8(float,int,int);
__device__ __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t,int);
__device__ __half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t,int);
