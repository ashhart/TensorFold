#pragma once
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ inline __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __grid_constant__
#define __CUDACC__ 1
#include <cstddef>
#include <cstdint>
#include <math.h>
#include <algorithm>
template<class T> __host__ __device__ T min(T,T);
template<class T> __host__ __device__ T max(T,T);
struct dim3 { unsigned x, y, z; __host__ __device__ constexpr dim3(unsigned a=1, unsigned b=1, unsigned c=1):x(a),y(b),z(c){} };
extern __device__ const dim3 threadIdx, blockIdx, blockDim, gridDim;
struct uint2 { unsigned x,y; }; struct uint3 { unsigned x,y,z; }; struct uint4 { unsigned x,y,z,w; };
struct int2 { int x,y; }; struct int4 { int x,y,z,w; };
struct float2 { float x,y; }; struct float4 { float x,y,z,w; };
struct double2 { double x,y; }; struct ulonglong2 { unsigned long long x,y; };
__device__ float2 make_float2(float,float); __device__ float4 make_float4(float,float,float,float);
__device__ uint2 make_uint2(unsigned,unsigned); __device__ uint4 make_uint4(unsigned,unsigned,unsigned,unsigned);
template<class T> __device__ T __ldg(const T*); template<class T> __device__ T __ldcg(const T*);
template<class T> __device__ T __ldcv(const T*);
template<class T> __device__ T __shfl_sync(unsigned,T,int,int=32);
template<class T> __device__ T __shfl_xor_sync(unsigned,T,int,int=32);
template<class T> __device__ T __shfl_up_sync(unsigned,T,unsigned,int=32);
template<class T> __device__ T __shfl_down_sync(unsigned,T,unsigned,int=32);
__device__ void __syncthreads(); __device__ int __syncthreads_count(int); __device__ void __syncwarp(unsigned=~0u);
__device__ void __threadfence(); __device__ void __threadfence_system(); __device__ void __trap();
__device__ void __nanosleep(unsigned); __device__ unsigned long long clock64();
__device__ unsigned __ballot_sync(unsigned,int); __device__ int __any_sync(unsigned,int);
__device__ unsigned __match_any_sync(unsigned,unsigned long long);
__device__ unsigned __reduce_add_sync(unsigned,unsigned); __device__ int __reduce_add_sync(unsigned,int);
__device__ unsigned __reduce_max_sync(unsigned,unsigned); __device__ int __reduce_max_sync(unsigned,int);
__device__ unsigned __reduce_min_sync(unsigned,unsigned); __device__ int __reduce_min_sync(unsigned,int);
__device__ unsigned __funnelshift_l(unsigned,unsigned,unsigned);
__device__ unsigned __funnelshift_r(unsigned,unsigned,unsigned);
__device__ unsigned __dp4a(unsigned,unsigned,unsigned); __device__ int __dp4a(int,int,int);
__device__ unsigned __byte_perm(unsigned,unsigned,unsigned); __device__ unsigned __brev(unsigned);
__device__ int __clz(unsigned); __device__ int __popc(unsigned); __device__ int __popcll(unsigned long long);
__device__ int __ffs(int); __device__ int __ffsll(long long);
__device__ unsigned __umulhi(unsigned,unsigned); __device__ size_t __cvta_generic_to_shared(const void*);
__device__ int __float_as_int(float);
__device__ unsigned __float_as_uint(float); __device__ float __int_as_float(int); __device__ float __uint_as_float(unsigned);
__device__ double __longlong_as_double(long long); __device__ long long __double2ll_rz(double);
__device__ float __double2float_rn(double);
__device__ float __fadd_rn(float,float); __device__ float __fsub_rn(float,float);
__device__ float __fmul_rn(float,float); __device__ float __fdiv_rn(float,float); __device__ float __fsqrt_rn(float);
__device__ float __fmaf_rn(float,float,float); __device__ float __expf(float);
__device__ double __dadd_rn(double,double); __device__ double __dsub_rn(double,double);
__device__ double __dmul_rn(double,double); __device__ double __ddiv_rn(double,double);
template<class T> __device__ T atomicAdd(T*,T); template<class T> __device__ T atomicMax(T*,T);
template<class T> __device__ T atomicMin(T*,T); template<class T> __device__ T atomicExch(T*,T);
template<class T> __device__ T atomicCAS(T*,T,T); template<class T> __device__ T atomicOr(T*,T);
__device__ unsigned atomicInc(unsigned*,unsigned);
using cudaStream_t=void*;
using cudaError_t=int;
enum { cudaSuccess, cudaFuncAttributeMaxDynamicSharedMemorySize };
template<class T> int cudaFuncSetAttribute(T,int,int);
template<class T> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*,T,int,size_t);
extern "C" int cudaConfigureCall(dim3,dim3,size_t=0,cudaStream_t=nullptr);
extern "C" int cudaSetupArgument(const void*,size_t,size_t);
extern "C" int cudaLaunch(const void*);
__device__ int2 make_int2(int,int); __device__ uint3 make_uint3(unsigned,unsigned,unsigned);
int cudaGetLastError();
enum { cudaErrorInvalidValue, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared,
       cudaLaunchAttributeClusterDimension, cudaLaunchAttributeProgrammaticStreamSerialization };
struct cudaLaunchAttribute { int id; struct { dim3 clusterDim; int programmaticStreamSerializationAllowed; } val; };
struct cudaLaunchConfig_t { dim3 gridDim,blockDim; size_t dynamicSmemBytes; cudaStream_t stream;
                           cudaLaunchAttribute* attrs; unsigned numAttrs; };
template<class F,class... A> int cudaLaunchKernelEx(const cudaLaunchConfig_t*,F,A...);
__host__ __device__ long min(long,int); __host__ __device__ long min(int,long);
__host__ __device__ long max(long,int); __host__ __device__ long max(int,long);
