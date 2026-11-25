// Pointwise activations several families share, in fp32 with torch's bf16 rounding points where the name says so.
#include <metal_stdlib>
using namespace metal;
// bf16(silu(x)): torch's SiLU on a bf16 tensor, fp32 math and one rounding.
inline float tf_silu_bf(float x) { return float(bfloat(x / (1.0f + metal::precise::exp(-x)))); }
// tanh(sqrt(2 / pi) (x + 0.044715 x^3)), gelu_pytorch_tanh's inner term; a macro so the IR stays the callers' own.
#define TF_GELU_TANH_INNER(x) metal::precise::tanh(0.79788456080286536f * ((x) + 0.044715f * (x) * (x) * (x)))
