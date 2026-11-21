// Inkling 2 Mini's soft-capped NVFP4 entries: included at the end of nvfp4_experts.cu, whose templates they launch.
#pragma once

// Soft-capped SwiGLU gate-up (Inkling 2 Mini): nvfp4_expert_kernel<2, 2, 4> decode and prompt tiles, epilogue 4.
extern "C" __global__ void __launch_bounds__(128)
    tf_nvfp4_softcap(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                     const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                     const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                     float gate_cap, float up_cap, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * 4 + (threadIdx.x >> 5); unit < units; unit += gridDim.x * 4)
    tf_nvfp4_experts::unit_full<2, 4, 4>(unit, lane, gq, t, threadIdx.x >> 5, X, x_stride, slots, W, scale, KG, NB,
                                         items, members, out, N, gate_cap, skip, up_cap);
}

extern "C" __global__ void __launch_bounds__(128)
    tf_nvfp4_prompt_softcap(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                            const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                            const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out,
                            int N, float gate_cap, float up_cap, int skip) {
  tf_nvfp4_experts::prompt_body<2, 4, 4, 1, 4>(X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N,
                                               gate_cap, skip, nullptr, nullptr, 0, up_cap);
}
