// DFly2 5-layer speculative drafter glue declarations and shared struct layouts (`sm_121` / multi-SM).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define HY4D_H 3072        // hidden size
#define HY4D_KL 512        // kv_lora_rank: normed latent
#define HY4D_RD 64         // qk_rope_head_dim
#define HY4D_KW 576        // one cache row = latent + roped key
#define HY4D_PAD_SLOT (-1LL)

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

// split: CTAs (a thread-block cluster) per row: 1, 2, 3, 4 or 6 (384 / split threads each); 0 picks by R.

// (1) Dual RMSNorm + concat into `out[r, 0:6144]` (`mtp_fusion` and `hidden_correction`).
cudaError_t hy4_dfly_dual_rmsnorm_cat(Hy4DflyRows a, const __nv_bfloat16* wa, Hy4DflyRows b,
                                      const __nv_bfloat16* wb, float eps, __nv_bfloat16* out, long long ld_out,
                                      int R, int split, cudaStream_t stream);

// (2) Query-block `mtp_fusion` gated add on anchor rows (`r % qw == 0`) + layer-0 `input_layernorm`.
cudaError_t hy4_dfly_fusion_gated_add(Hy4DflyRows e, const float* eh, long long eh_ld, long long eh_world_stride,
                                      int eh_world, const __nv_bfloat16* gate, const __nv_bfloat16* w_norm, float eps,
                                      int qw, __nv_bfloat16* residual_out, long long ld_res, __nv_bfloat16* norm_out,
                                      long long ld_norm, int N, int split, cudaStream_t stream);

// (3) Sublayer boundary: rank-order fp32 sum, optional sandwich RMSNorm, residual add, and optional next RMSNorm.
cudaError_t hy4_dfly_sublayer_boundary(const float* part, long long part_ld, long long world_stride, int world,
                                       const __nv_bfloat16* sandwich_w, const __nv_bfloat16* residual,
                                       long long ld_res_in, const __nv_bfloat16* next_w,
                                       __nv_bfloat16* residual_out, long long ld_res_out, __nv_bfloat16* next_out,
                                       long long ld_next, float eps, int R, int split, cudaStream_t stream);

// (4) Sparse top-k history + local window + causal block index expansion (`out[l, b*qw+q, 0:k_pad]`).
enum { HY4D_SRC_TOK = 0, HY4D_SRC_LIST = 1 };
typedef struct {
    int mode;
    const int* src;            // tok [slots][src_rows_cap][target_topk] or list [slots][src_rows_cap][512]
    long long src_slot_stride; // elements
    long long src_row_stride;  // elements (multiple of 4)
    int src_rows_cap;          // source rows outside [0, cap) give no history
    int dense_below;           // LIST: 2048
    const int* layer_slot;     // [num_layers]
    int num_layers;
    const int* active;         // [B] 0/1
    const int* source_rows;    // [B]
    const int* anchors;        // [B]
    int B, qw, target_topk, local_window, k_pad;
    int* out;                  // [num_layers][out_rows][k_pad]
    long long out_layer_stride; // elements
    // slots (optional: slot_out null skips)
    long long* slot_out;       // [B * qw]
    int* row_pos_out;          // [B * qw] optional: anchor + q (active) else q
    const int* block_table;    // [B][bt_stride] or null (contiguous)
    long long bt_stride;
    int blocks_per_req, block_size, max_model_len;
    const int* req_stream;     // contiguous mode: stream of request b (null: b)
    long long ctx;             // contiguous mode: rows a stream
    long long pad_slot;
} Hy4DflyTopkArgs;
cudaError_t hy4_dfly_extend_topk(const Hy4DflyTopkArgs* args, cudaStream_t stream);

// (5) Guarded scatter of `[N, 576]` draft cache rows (`kv_c` + `k_pe`) into target MLA cache slots.
cudaError_t hy4_dfly_kv_overlay_write(const __nv_bfloat16* kv_c, long long ld_c, const __nv_bfloat16* k_pe,
                                      long long ld_pe, const long long* slots, int N, __nv_bfloat16* cache,
                                      long long cache_ld, long long num_slots, cudaStream_t stream);

// The split extend_topk / the norm kernels pick for R when split == 0.
int hy4_dfly_auto_split(int R);

#ifdef __cplusplus
}
#endif
