// Full-vocabulary normalization, deterministic argmax and selected logits for up to four independent rows.
extern "C" __global__ void __launch_bounds__(256) tf_q35_score_stats(
    const float* logits, int vocab, float* stats, const unsigned* labels) {
    const int t = threadIdx.x;
    const float* x = logits + (size_t)blockIdx.x * vocab;
    if (t < 16) stats[8 + blockIdx.x * 16 + t] =
        t < labels[blockIdx.x * 17] ? x[labels[blockIdx.x * 17 + 1 + t]] : 0.f;
    __shared__ float maxima[256];
    __shared__ int indices[256];
    __shared__ float sums[256];
    float m = -INFINITY;
    int best = 0;
    for (int i = t; i < vocab; i += 256) {
        if (x[i] > m || (x[i] == m && i < best)) { m = x[i]; best = i; }
    }
    maxima[t] = m;
    indices[t] = best;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) {
            const float b = maxima[t + stride];
            const int j = indices[t + stride];
            if (b > maxima[t] || (b == maxima[t] && j < indices[t])) {
                maxima[t] = b;
                indices[t] = j;
            }
        }
        __syncthreads();
    }
    float sum = 0;
    for (int i = t; i < vocab; i += 256) sum += expf(x[i] - maxima[0]);
    sums[t] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    if (t == 0) {
        stats[blockIdx.x * 2] = maxima[0] + logf(sums[0]);
        reinterpret_cast<int*>(stats)[blockIdx.x * 2 + 1] = indices[0];
    }
}
