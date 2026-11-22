#if TF_FP_NATIVE
TF_NVFP4_PROMPT(tf_nvfp4_prompt_relu2_native, 1, 1, 8, 0, true)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_swiglu_native, 2, 2, 4, 0, true)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_down_native, 1, 3, 8, 0, true)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_relu2_native_s3, 1, 1, 8, 3, true)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_swiglu_native_s3, 2, 2, 4, 3, true)
TF_NVFP4_PROMPT(tf_nvfp4_prompt_down_native_s3, 1, 3, 8, 3, true)

// Native versus software through the prompt helpers: all fp4 bytes and both scale selectors for every scale byte.
extern "C" __global__ void tf_nvfp4_native_fp_check(uint32_t* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 512) return;
  const uint32_t c = i & 255, b = i >> 8;
  const uint32_t word = (c & 15u) | ((c >> 4) << 4) | ((c & 15u) << 16) | ((c >> 4) << 20);
  uint32_t x = 0xA5B6C7D8u;
  if (b) x = (x & 0x00FF00FFu) | (c << 8) | (c << 24);
  else x = (x & 0xFF00FF00u) | c | (c << 16);
  const uint32_t pick = b | 0x4040u | (b << 8);
  uint32_t n0, n1, s0, s1;
  tf_nvfp4_experts::fp4_pairs<true>(word, 0, n0, n1);
  tf_nvfp4_experts::fp4_pairs<false>(word, 0, s0, s1);
  out[6 * i] = n0;
  out[6 * i + 1] = n1;
  out[6 * i + 2] = tf_nvfp4_experts::scale_pair<true>(x, pick);
  out[6 * i + 3] = s0;
  out[6 * i + 4] = s1;
  out[6 * i + 5] = tf_nvfp4_experts::scale_pair<false>(x, pick);
}
#endif
