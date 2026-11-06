// The reference's gathered quantised matmul for our routed shapes.
// Ported from ml-explore/mlx v0.32.4 (MIT): mlx/backend/metal/kernels/quantized.h and steel/gemm/**, flattened
// into mlx_steel.h with attribution. These entries are TensorFold's: a kernel function cannot call another, so
// each is the reference template instantiated at one geometry (the form MLX itself uses).
#include "mlx_steel.h"

template [[host_name("tf_gather_qmm_f32_gs64_b4")]] [[kernel]] decltype(affine_gather_qmm_rhs<float, 64, 4, 64, 64, 32, 2, 2, true>) affine_gather_qmm_rhs<float, 64, 4, 64, 64, 32, 2, 2, true>;
template [[host_name("tf_gather_qmm_f32_gs64_b4_bn64")]] [[kernel]] decltype(affine_gather_qmm_rhs<float, 64, 4, 64, 64, 64, 2, 2, true>) affine_gather_qmm_rhs<float, 64, 4, 64, 64, 64, 2, 2, true>;

// Wider warp grids: more MMA issue slots per threadgroup. WM/WN are the warp grid, so 4x2 = 256 threads,
// 4x4 = 512. These are the geometries that could raise the tensor rate above the 2x2 baseline.
template [[host_name("tf_gather_qmm_w4x2")]] [[kernel]] decltype(affine_gather_qmm_rhs<float, 64, 4, 128, 64, 32, 4, 2, true>) affine_gather_qmm_rhs<float, 64, 4, 128, 64, 32, 4, 2, true>;
template [[host_name("tf_gather_qmm_w4x4")]] [[kernel]] decltype(affine_gather_qmm_rhs<float, 64, 4, 128, 128, 32, 4, 4, true>) affine_gather_qmm_rhs<float, 64, 4, 128, 128, 32, 4, 4, true>;
template [[host_name("tf_gather_qmm_bm128")]] [[kernel]] decltype(affine_gather_qmm_rhs<float, 64, 4, 128, 64, 32, 2, 2, true>) affine_gather_qmm_rhs<float, 64, 4, 128, 64, 32, 2, 2, true>;
