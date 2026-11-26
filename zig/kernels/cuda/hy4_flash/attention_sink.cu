#include "../common/softmax_identity.cuh"
// Finish a head's online softmax with its learned, zero-value sink.
extern "C" __global__ void tf_hy4_finish_attention(
    const float *maxima,
    const float *mass,
    const float *numerator,
    const float *sinks,
    float *output,
    unsigned int rows,
    unsigned int heads,
    unsigned int width) {
    const unsigned int row_head = blockIdx.x;
    const unsigned int column = blockIdx.y * blockDim.x + threadIdx.x;
    if (column >= width || row_head >= static_cast<unsigned long long>(rows) * heads) return;

    const float sink = sinks[row_head % heads];
    const float top = fmaxf(maxima[row_head], sink);
    const bool same = TF_CUDA_SOFTMAX_IDENTITY_GUARD && isfinite(top) && top == maxima[row_head]; const float prior = same ? 1.0f : expf(maxima[row_head] - top);
    const float denominator = (same && isfinite(mass[row_head] + expf(sink - top))) ? mass[row_head] + expf(sink - top) : mass[row_head] * prior + expf(sink - top);
    const unsigned long long offset = static_cast<unsigned long long>(row_head) * width + column;
    output[offset] = ((same && isfinite(numerator[offset])) ? numerator[offset] : numerator[offset] * prior) / denominator;
}
