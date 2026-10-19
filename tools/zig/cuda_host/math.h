#pragma once
#define INFINITY (__builtin_inff())
#define NAN (__builtin_nanf(""))
__host__ __device__ float fminf(float,float); __host__ __device__ float fmaxf(float,float);
__host__ __device__ float expf(float); __host__ __device__ float exp2f(float); __host__ __device__ float logf(float);
__host__ __device__ float log2f(float); __host__ __device__ float log1pf(float); __host__ __device__ float sqrtf(float);
__host__ __device__ float rsqrtf(float); __host__ __device__ float sinf(float); __host__ __device__ float cosf(float);
__host__ __device__ float tanhf(float); __host__ __device__ float fabsf(float); __host__ __device__ float floorf(float);
__host__ __device__ float ceilf(float); __host__ __device__ float roundf(float); __host__ __device__ float nearbyintf(float);
__host__ __device__ float powf(float,float); __host__ __device__ float fmaf(float,float,float);
__host__ __device__ double fma(double,double,double); __host__ __device__ double sqrt(double);
__host__ __device__ double exp(double); __host__ __device__ double log(double); __host__ __device__ double fabs(double);
__host__ __device__ double floor(double); __host__ __device__ double nearbyint(double);
__host__ __device__ double fmin(double,double); __host__ __device__ double fmax(double,double);
__host__ __device__ void sincosf(float,float*,float*);
template<class T> __host__ __device__ bool isnan(T); template<class T> __host__ __device__ bool isinf(T);
template<class T> __host__ __device__ bool isfinite(T);
#define M_SQRT2 1.41421356237309504880
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT1_2 0.70710678118654752440
__host__ __device__ float erff(float); __host__ __device__ float rintf(float);
__host__ __device__ double pow(double,double); __host__ __device__ double sin(double); __host__ __device__ double cos(double);
template<class T> __host__ __device__ bool signbit(T);
