// Optional CPU diagnostic accelerator for the independent Metal row-reduction replay, never native inference.
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

// Compile with -ffp-contract=off: only the explicitly named affine fmaf is contracted.
int lfm3_row_replay(const float *x, const uint8_t *codes, const float *scales, const float *biases,
                    float *output, int rows, int n, int k, int group, int bits) {
  float *placed = malloc((int64_t)rows * k * sizeof(float));
  float *sums = malloc((int64_t)rows * (k / 16) * sizeof(float));
  if (!placed || !sums) { free(placed); free(sums); return 1; }
  for (int row = 0; row < rows; ++row) {
    for (int start = 0; start < k; start += 16) {
      float sum = 0.0f;
      for (int i = 0; i < 16; ++i) {
        const int64_t at = (int64_t)row * k + start + i;
        sum = sum + x[at];
        const float place = bits == 4 ? (float)(1u << (4 * (i % 4))) : 1.0f;
        placed[at] = x[at] / place;
      }
      sums[(int64_t)row * (k / 16) + start / 16] = sum;
    }
  }
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < n; ++column) {
      float lanes[32] = {0};
      for (int base = 0; base < k; base += 512) {
        for (int lane = 0; lane < 32; ++lane) {
          const int start = base + lane * 16;
          const float *input = placed + (int64_t)row * k + start;
          const uint8_t *weight = codes + (int64_t)column * k + start;
          const float sum = sums[(int64_t)row * (k / 16) + start / 16];
          float dot = 0.0f;
          for (int i = 0; i < 16; i += 4) {
            float products[4];
            for (int j = 0; j < 4; ++j) {
              const float place = bits == 4 ? (float)(1u << (4 * j)) : 1.0f;
              const float placed_input = input[i + j];
              const float placed_code = (float)weight[i + j] * place;
              products[j] = placed_input * placed_code;
            }
            float four = products[0] + products[1];
            four = four + products[2];
            four = four + products[3];
            dot = dot + four;
          }
          const int64_t at = (int64_t)column * (k / group) + start / group;
          const float bias_product = sum * biases[at];
          lanes[lane] = lanes[lane] + fmaf(scales[at], dot, bias_product);
        }
      }
      for (int distance = 1; distance < 32; distance *= 2)
        for (int lane = 0; lane < 32; lane += 2 * distance)
          lanes[lane] = lanes[lane] + lanes[lane + distance];
      output[(int64_t)row * n + column] = lanes[0];
    }
  }
  free(placed);
  free(sums);
  return 0;
}

// BF16 operands' dense projection uses one FP32 accumulator per lane and the same adjacent butterfly.
void lfm3_dense_replay(const float *x, const float *weight, float *output, int rows, int n, int k) {
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < n; ++column) {
      float lanes[32] = {0};
      for (int lane = 0; lane < 32; ++lane)
        for (int i = lane; i < k; i += 32)
          lanes[lane] = fmaf(x[(int64_t)row * k + i], weight[(int64_t)column * k + i], lanes[lane]);
      for (int distance = 1; distance < 32; distance *= 2)
        for (int lane = 0; lane < 32; lane += 2 * distance)
          lanes[lane] = lanes[lane] + lanes[lane + distance];
      output[(int64_t)row * n + column] = lanes[0];
    }
  }
}
