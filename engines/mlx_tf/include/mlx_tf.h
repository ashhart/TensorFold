/* TensorFold mlx_tf — Bare-Metal Apple Silicon LLM Inference C ABI (Zero Apple MLX) */
#ifndef TENSORFOLD_MLX_TF_H
#define TENSORFOLD_MLX_TF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MlxTfEngine MlxTfEngine;

typedef struct MlxTfGenOptions {
    uint32_t max_tokens;
    float temperature;
    float top_p;
    uint32_t top_k;
    float min_p;
    uint64_t seed;
    uint8_t enable_thinking;
    uint8_t use_chat_template;
    uint8_t stop_on_eos;
    uint8_t _pad;
    float repetition_penalty;
    float frequency_penalty;
    float presence_penalty;
    uint32_t _reserved;
} MlxTfGenOptions;

typedef struct MlxTfStats {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    uint32_t rounds;
    uint32_t drafts_accepted;
    uint32_t drafts_proposed;
    uint32_t cached_tokens;
    double prefill_ms;
    double decode_ms;
    double total_ms;
    double tok_per_sec;
} MlxTfStats;

typedef uint8_t (*MlxTfTokenCb)(
    void *user_data,
    uint32_t token_id,
    const char *utf8,
    size_t utf8_len
);

typedef uint8_t (*MlxTfWaveTokenCb)(
    void *user_data,
    uint32_t stream_idx,
    uint32_t token_id,
    const char *utf8,
    size_t utf8_len
);

const char *mlxtf_version(void);

MlxTfEngine *mlxtf_load(
    const char *model_dir,
    const char *dflash_dir,
    const char *cross_draft_dir,
    uint32_t max_context,
    uint32_t max_streams
);

MlxTfEngine *mlxtf_load_ex(
    const char *model_dir,
    const char *dflash_dir,
    const char *cross_draft_dir,
    uint32_t max_context,
    uint32_t max_streams,
    uint8_t force_simd,
    uint8_t keep_bf16
);

void mlxtf_free(MlxTfEngine *eng);
const char *mlxtf_device_name(MlxTfEngine *eng);
const char *mlxtf_arch_name(MlxTfEngine *eng);
uint8_t mlxtf_uses_m5_tensor_ops(MlxTfEngine *eng);
uint8_t mlxtf_has_vision(MlxTfEngine *eng);

int32_t mlxtf_encode_image(
    MlxTfEngine *eng,
    const char *path_or_uri,
    float *out_f32,
    size_t max_elems,
    uint32_t *out_num_tokens,
    uint32_t *out_hidden
);

int32_t mlxtf_encode_patches(
    MlxTfEngine *eng,
    const uint16_t *patches_bf16,
    size_t num_elems,
    uint32_t grid_h,
    uint32_t grid_w,
    float *out_f32,
    size_t max_elems,
    uint32_t *out_num_tokens,
    uint32_t *out_hidden
);

int32_t mlxtf_generate(
    MlxTfEngine *eng,
    const char *prompt_ptr,
    size_t prompt_len,
    const MlxTfGenOptions *opts,
    MlxTfTokenCb cb,
    void *user_data,
    MlxTfStats *out_stats
);

int32_t mlxtf_generate_with_images(
    MlxTfEngine *eng,
    const char *prompt_ptr,
    size_t prompt_len,
    const char *const *image_paths,
    uint32_t num_images,
    const MlxTfGenOptions *opts,
    MlxTfTokenCb cb,
    void *user_data,
    MlxTfStats *out_stats
);

int32_t mlxtf_stream_parallel(
    MlxTfEngine *eng,
    const char *shared_ptr,
    size_t shared_len,
    uint32_t n_streams,
    const char *const *deltas_ptrs,
    const size_t *delta_lens,
    const uint32_t *max_tokens_arr,
    const float *temps_arr,
    MlxTfWaveTokenCb cb,
    void *user_data,
    MlxTfStats *out_stats
);

int32_t mlxtf_serve(MlxTfEngine *eng, uint16_t port, const char *model_name);
int32_t mlxtf_serve_host(MlxTfEngine *eng, const char *host, uint16_t port, const char *model_name);

int32_t mlxtf_convert(
    const char *src_dir,
    const char *dst_dir,
    uint32_t bits,
    uint32_t group_size
);

int32_t mlxtf_convert_tfq3(
    const char *src_dir,
    const char *dst_dir,
    uint32_t bits,
    const char *profile,
    uint32_t head_bits,
    uint8_t quantize_head
);

#ifdef __cplusplus
}
#endif

#endif /* TENSORFOLD_MLX_TF_H */
