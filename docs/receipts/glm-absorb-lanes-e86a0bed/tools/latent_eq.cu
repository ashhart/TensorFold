// main's tf_glm_absorb_q / tf_glm_expand_v against the branch's (renamed new_*): bits at many row counts, and time.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "glm_glue_latent.cuh"
#include "new_latent.cuh"

int main() {
  const int64_t heads = 32, D = 256, LW = 512;
  int bad = 0;
  std::vector<int64_t> sizes;
  for (int64_t r = 1; r <= 70; ++r) sizes.push_back(r);
  for (int64_t r : {100, 127, 128, 129, 1000, 2047, 2048, 4096}) sizes.push_back(r);
  for (int64_t rows : sizes) {
    std::vector<__nv_bfloat16> hq(rows * heads * D), hw(heads * D * LW), hol(rows * heads * LW);
    srand(rows);
    for (auto& x : hq) x = __float2bfloat16((rand() / (float)RAND_MAX - 0.5f) * 4.0f);
    for (auto& x : hw) x = __float2bfloat16((rand() / (float)RAND_MAX - 0.5f) * 0.2f);
    for (auto& x : hol) x = __float2bfloat16((rand() / (float)RAND_MAX - 0.5f) * 4.0f);
    __nv_bfloat16 *q, *w, *ol, *a1, *a2, *e1, *e2;
    cudaMalloc(&q, hq.size() * 2); cudaMalloc(&w, hw.size() * 2); cudaMalloc(&ol, hol.size() * 2);
    cudaMalloc(&a1, rows * heads * LW * 2); cudaMalloc(&a2, rows * heads * LW * 2);
    cudaMalloc(&e1, rows * heads * D * 2); cudaMalloc(&e2, rows * heads * D * 2);
    cudaMemcpy(q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(ol, hol.data(), hol.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(a2, 0xFF, rows * heads * LW * 2); cudaMemset(e2, 0xFF, rows * heads * D * 2);
    const int64_t rb = rows >= 64 ? 64 : rows;  // cuda_latent.zig's rowBlocks
    dim3 ga(heads, LW / 32, (rows + rb - 1) / rb), ge(heads, D / 16, (rows + rb - 1) / rb);
    cudaEvent_t t0, t1, t2; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2);
    cudaEventRecord(t0);
    tf_glm_absorb_q<<<ga, 128>>>(q, w, a1, rows, heads, rb); tf_glm_expand_v<<<ge, 128>>>(ol, w, e1, rows, heads, rb);
    cudaEventRecord(t1);
    new_absorb_q<<<ga, 128>>>(q, w, a2, rows, heads, rb); new_expand_v<<<ge, 128>>>(ol, w, e2, rows, heads, rb);
    cudaEventRecord(t2); cudaEventSynchronize(t2);
    float m1, m2; cudaEventElapsedTime(&m1, t0, t1); cudaEventElapsedTime(&m2, t1, t2);
    std::vector<char> x1(rows * heads * LW * 2), x2(x1.size()), y1(rows * heads * D * 2), y2(y1.size());
    cudaMemcpy(x1.data(), a1, x1.size(), cudaMemcpyDeviceToHost); cudaMemcpy(x2.data(), a2, x2.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(y1.data(), e1, y1.size(), cudaMemcpyDeviceToHost); cudaMemcpy(y2.data(), e2, y2.size(), cudaMemcpyDeviceToHost);
    const bool sa = !memcmp(x1.data(), x2.data(), x1.size()), se = !memcmp(y1.data(), y2.data(), y1.size());
    bad += !sa + !se;
    if (!sa || !se || rows >= 64 && (rows % 64 == 0 || rows > 100)) printf("rows %5ld absorb %s expand %s  main %.3f ms branch %.3f ms\n", rows, sa ? "same" : "DIFFER", se ? "same" : "DIFFER", m1, m2);
    cudaFree(q); cudaFree(w); cudaFree(ol); cudaFree(a1); cudaFree(a2); cudaFree(e1); cudaFree(e2);
  }
  printf("%zu sizes, %d differ; %s\n", sizes.size(), bad, cudaGetErrorString(cudaGetLastError()));
  return bad != 0;
}
