// DFly2 5-layer speculative drafter glue shared constants and ABI structs (`dfly2_glue.cu`).
#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

#define HY4D_H 3072  // hidden size

#ifdef __cplusplus
extern "C" {
#endif

// Row set of 3072 bf16s (`base + i * ld`, `i = ids ? ids[r * ids_stride] : r`; out-of-range id reads zero).
typedef struct {
    const __nv_bfloat16* base;
    const int* ids;
    long long ld;
    int ids_stride;
    int nrows;
} Hy4DflyRows;

// Sparse top-k history + local window + causal block index expansion (`out[l, b*qw+q, 0:k_pad]`).
enum { HY4D_SRC_TOK = 0, HY4D_SRC_LIST = 1 };
typedef struct {
    int mode;
    const int* src;             // tok [slots][src_rows_cap][target_topk] or list [slots][src_rows_cap][512]
    long long src_slot_stride;  // elements
    long long src_row_stride;   // elements (multiple of 4)
    int src_rows_cap;           // source rows outside [0, cap) give no history
    int dense_below;            // LIST: 2048
    const int* layer_slot;      // [num_layers]
    int num_layers;
    const int* active;          // [B] 0/1
    const int* source_rows;     // [B]
    const int* anchors;         // [B]
    int B, qw, target_topk, local_window, k_pad;
    int* out;                   // [num_layers][out_rows][k_pad]
    long long out_layer_stride; // elements
    // slots (optional: slot_out null skips)
    long long* slot_out;        // [B * qw]
    int* row_pos_out;           // [B * qw] optional: anchor + q (active) else q
    const int* block_table;     // [B][bt_stride] or null (contiguous)
    long long bt_stride;
    int blocks_per_req, block_size, max_model_len;
    const int* req_stream;      // contiguous mode: stream of request b (null: b)
    long long ctx;              // contiguous mode: rows a stream
    long long pad_slot;
} Hy4DflyTopkArgs;

#ifdef __cplusplus
}
#endif
