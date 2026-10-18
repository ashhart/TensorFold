// H3 BF16 block library, ABI 1. The host supplies complete device or host sequences and owns sampling and decoders.
#ifndef TF_H3_H
#define TF_H3_H
#include <stdint.h>

typedef struct tf_h3_model tf_h3_model;
typedef struct {
    uint64_t x, tables, line, cos, sin, slot, sizes;
    uint32_t rows, lines, rot, tiles, prefix_tiles, keep, blocks;
    uint64_t attention_out;
} tf_h3_inputs;

#ifdef __cplusplus
extern "C" {
#endif
uint32_t tf_h3_abi_version(void);
tf_h3_model *tf_h3_open(const char *path);
// Failed completion retains the complete handle; stderr reports failure. Retry only after context recovery.
void tf_h3_close(tf_h3_model *model);
void tf_h3_shape(const tf_h3_model *model, uint32_t out[7]);
double tf_h3_forward(tf_h3_model *model, const tf_h3_inputs *in);
// Every nonzero input address, including attention_out, is host memory.
double tf_h3_forward_host(tf_h3_model *model, const tf_h3_inputs *in);
int tf_h3_set_weight(tf_h3_model *model, uint32_t block, int projection, uint64_t weight, uint32_t rows, uint32_t cols);
void tf_h3_profile(tf_h3_model *model, int on);
int tf_h3_spent(const tf_h3_model *model, double *out, const char **names);
void *tf_h3_host_alloc(tf_h3_model *model, uint64_t bytes, uint64_t *device);
// Only a matching allocation from this model can be freed; failed completion retains it.
void tf_h3_host_free(tf_h3_model *model, void *ptr);
#ifdef __cplusplus
}
#endif
#endif

// A failed open can retain an incomplete loader. Retrying open drains it in its owner context before new work.
