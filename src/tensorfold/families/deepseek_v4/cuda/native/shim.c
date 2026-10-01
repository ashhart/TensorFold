/* TensorFold's small, versioned boundary over the pinned MIT ds4 engine.
 * Include the donor translation unit so the ABI smoke can call its IQ2 primitive.
 * The donor implementation and its arithmetic are otherwise unchanged. */
#include "ds4.c"

typedef struct {
    ds4_engine *engine;
    ds4_session *session;
    ds4_session_snapshot snapshot;
    int *snapshot_prefix;
    uint64_t snapshot_limit;
    int snapshot_tokens, snapshot_target, cached;
    bool snapshot_failed;
    char snapshot_error[1024];
} tf_ds4;

int tf_ds4_abi(void) { return 2; }
const char *tf_ds4_revision(void) { return TF_DS4_REVISION; }
int tf_ds4_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#else
    return DS4_BACKEND_CUDA;
#endif
}

void tf_ds4_close(tf_ds4 *ctx) {
    if (!ctx) return;
    ds4_session_snapshot_free(&ctx->snapshot);
    free(ctx->snapshot_prefix);
    if (ctx->session) ds4_session_free(ctx->session);
    if (ctx->engine) ds4_engine_close(ctx->engine);
    free(ctx);
}

#ifndef DS4_NO_GPU
static uint64_t tf_ds4_inactive_primary_bytes(int context) {
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        const uint64_t rows = (uint64_t)(context / ratio + 2u);
        if (!DS4_GPU_ATTN_COMP_CACHE_F16 && ds4_cuda_fp8_kv_enabled())
            bytes += rows * DS4_N_HEAD_DIM * sizeof(float);
        if (ratio == 4 && ds4_cuda_fp4_index_enabled())
            bytes += rows * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    }
    return bytes;
}

/* Upper bound from donor capacities: packed rows, compressor state, both
 * histories and staging. A fixed-capacity snapshot avoids realloc's peak. */
static uint64_t tf_ds4_snapshot_bytes(int context) {
    ds4_context_memory m = ds4_context_memory_estimate(DS4_BACKEND_CUDA, context);
    const uint64_t inactive = tf_ds4_inactive_primary_bytes(context);
    if (inactive > m.compressed_bytes) return 0;
    ds4_gpu_graph geometry = {0};
    geometry.raw_cap = m.raw_cap;
    geometry.raw_window = DS4_N_SWA;
    const uint32_t raw_live = session_raw_live_rows(&geometry, (uint32_t)context);
    uint64_t bytes = (uint64_t)DS4_N_LAYER * raw_live * DS4_N_HEAD_DIM * sizeof(float)
        + m.compressed_bytes - inactive;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        bytes += 2 * layer_attn_state_bytes(ratio);
        if (ratio == 4) bytes += 2 * layer_index_state_bytes(ratio);
    }
    return bytes + 2ull * context * sizeof(uint32_t) + DS4_N_VOCAB * sizeof(float)
        + (DS4_SESSION_PAYLOAD_U32_FIELDS + 2 * DS4_N_LAYER + 1) * sizeof(uint32_t)
        + DS4_SESSION_PAYLOAD_MTP_TAIL_BYTES + DS4_SESSION_IO_CHUNK;
}

static void tf_ds4_save_boundary(void *ud, const char *event, int current, int total) {
    (void)total;
    tf_ds4 *ctx = ud;
    if (strcmp(event, "prefill_chunk") || current <= 0 || current != ctx->snapshot_target
        || current == ctx->snapshot_tokens || ctx->snapshot_failed) return;
    uint64_t bytes = ds4_session_payload_bytes(ctx->session);
    if (current > ds4_session_ctx(ctx->session) || current != ctx->session->checkpoint.len
        || !bytes || bytes > ctx->snapshot_limit) {
        snprintf(ctx->snapshot_error, sizeof(ctx->snapshot_error), "canonical snapshot exceeds admitted capacity");
        ctx->snapshot_failed = true;
        return;
    }
    if (!ctx->snapshot.ptr) {
        ctx->snapshot.ptr = malloc(ctx->snapshot_limit);
        if (!ctx->snapshot.ptr) {
            snprintf(ctx->snapshot_error, sizeof(ctx->snapshot_error), "canonical snapshot allocation failed");
            ctx->snapshot_failed = true;
            return;
        }
        ctx->snapshot.cap = ctx->snapshot_limit;
    }
    if (ds4_session_save_snapshot(ctx->session, &ctx->snapshot,
                                 ctx->snapshot_error, sizeof(ctx->snapshot_error))) {
        ctx->snapshot_failed = true;
        return;
    }
    memcpy(ctx->snapshot_prefix, ctx->session->checkpoint.v, (size_t)current * sizeof(int));
    ctx->snapshot_tokens = current;
}
#endif

int tf_ds4_open(const char *path, int context, int threads, tf_ds4 **out,
                char *err, size_t cap) {
    if (out) *out = NULL;
    if (!err || !cap) return 1;
    if (!path || !out || context < 1 || threads < 1 || access(path, R_OK)) {
        snprintf(err, cap, "invalid native model path/context/threads");
        return 1;
    }
    tf_ds4 *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { snprintf(err, cap, "native allocation failed"); return 1; }
    ds4_engine_options opt = {0};
    opt.model_path = path;
    opt.backend = (ds4_backend)tf_ds4_backend();
    opt.n_threads = threads;
    opt.power_percent = 100;
    /* No MTP, DSpark or distributed execution.
     * Keep the donor's bounded boot prewarm before accepting requests. */
    if (ds4_engine_open(&ctx->engine, &opt) ||
        ds4_session_create(&ctx->session, ctx->engine, context)) {
        snprintf(err, cap, "native engine/session initialization failed");
        tf_ds4_close(ctx);
        return 1;
    }
#ifndef DS4_NO_GPU
    uint64_t budget = tf_ds4_snapshot_bytes(context);
    uint64_t prefix_bytes = (uint64_t)context * sizeof(int);
    if (budget <= prefix_bytes + DS4_SESSION_IO_CHUNK || budget > SIZE_MAX) {
        snprintf(err, cap, "invalid canonical snapshot capacity");
        tf_ds4_close(ctx);
        return 1;
    }
    ctx->snapshot_limit = budget - prefix_bytes - DS4_SESSION_IO_CHUNK;
    ctx->snapshot_prefix = malloc((size_t)prefix_bytes);
    if (!ctx->snapshot_prefix) {
        snprintf(err, cap, "canonical prefix allocation failed");
        tf_ds4_close(ctx);
        return 1;
    }
#endif
    *out = ctx;
    return 0;
}

int tf_ds4_vocab(tf_ds4 *ctx) { return ctx ? ds4_engine_vocab_size(ctx->engine) : 0; }
int tf_ds4_eos(tf_ds4 *ctx) { return ctx ? ds4_token_eos(ctx->engine) : -1; }
int tf_ds4_context(tf_ds4 *ctx) { return ctx ? ds4_session_ctx(ctx->session) : 0; }
void tf_ds4_reset(tf_ds4 *ctx) {
    if (!ctx) return;
    ctx->snapshot_tokens = ctx->cached = 0;
    ctx->snapshot.len = 0;
    ds4_session_invalidate(ctx->session);
}
int tf_ds4_cached(tf_ds4 *ctx) { return ctx ? ctx->cached : 0; }
int tf_ds4_sync(tf_ds4 *ctx, const int *ids, int count, char *err, size_t cap) {
    if (!err || !cap) return 1;
    if (!ctx || !ids || count <= 0 || count >= ds4_session_ctx(ctx->session)) {
        snprintf(err, cap, "invalid native prompt length"); return 1;
    }
    for (int i = 0; i < count; i++) {
        if (ids[i] < 0 || ids[i] >= tf_ds4_vocab(ctx)) {
            snprintf(err, cap, "invalid native prompt token"); return 1;
        }
    }
    ds4_tokens tokens = {(int *)ids, count, count};
    ctx->cached = 0;
#ifndef DS4_NO_GPU
    if (!ds4_session_is_cpu(ctx->session)) {
        /* Decode mutates the unfinished chunk. Restore the last complete
         * prefill boundary, then rebuild its tail with the cold chunk shape.
         * A token-count rewind cannot restore rings and compressor state. */
        if (ctx->snapshot_tokens > 0 && count >= ctx->snapshot_tokens
            && !memcmp(ids, ctx->snapshot_prefix, (size_t)ctx->snapshot_tokens * sizeof(int))) {
            if (ds4_session_load_snapshot(ctx->session, &ctx->snapshot, err, cap)) return 1;
            ctx->cached = ctx->snapshot_tokens;
        } else {
            tf_ds4_reset(ctx);
        }
        ctx->snapshot_target = count / ctx->session->prefill_cap * ctx->session->prefill_cap;
        ctx->snapshot_failed = false;
        ds4_session_set_progress(ctx->session, tf_ds4_save_boundary, ctx);
        int rc = ds4_session_sync(ctx->session, &tokens, err, cap);
        ds4_session_set_progress(ctx->session, NULL, NULL);
        /* A one-chunk cold sync does not emit the chunked progress callback. */
        if (!rc && count == ctx->snapshot_target && ctx->snapshot_tokens != count)
            tf_ds4_save_boundary(ctx, "prefill_chunk", count, count);
        if (ctx->snapshot_failed) {
            snprintf(err, cap, "%s", ctx->snapshot_error);
            return 1;
        }
        return rc;
    }
#endif
    if (ctx->session->checkpoint_valid && ds4_tokens_starts_with(&tokens, &ctx->session->checkpoint))
        ctx->cached = ctx->session->checkpoint.len;
    return ds4_session_sync(ctx->session, &tokens, err, cap);
}
int tf_ds4_eval(tf_ds4 *ctx, int id, char *err, size_t cap) {
    if (!err || !cap) return 1;
    if (!ctx || id < 0 || id >= tf_ds4_vocab(ctx) ||
        ds4_session_pos(ctx->session) >= ds4_session_ctx(ctx->session)) {
        snprintf(err, cap, "invalid native token or exhausted context"); return 1;
    }
    return ds4_session_eval(ctx->session, id, err, cap);
}
int tf_ds4_logits(tf_ds4 *ctx, float *out, int count) {
    if (!ctx || !out || count < tf_ds4_vocab(ctx)) return 0;
    return ds4_session_copy_logits(ctx->session, out, count);
}
int tf_ds4_encode(tf_ds4 *ctx, const char *text, int rendered, int **out) {
    if (!out) return -1;
    *out = NULL;
    if (!ctx || !text) return -1;
    ds4_tokens tokens = {0};
    if (rendered) ds4_tokenize_rendered_chat(ctx->engine, text, &tokens);
    else ds4_tokenize_text(ctx->engine, text, &tokens);
    *out = tokens.v;
    return tokens.len;
}
void tf_ds4_free(void *ptr) { free(ptr); }
/* Metadata-only estimator. CPU inspect binds tensor descriptors without touching
 * payloads; the CUDA allocator's own estimate is then queried, not reimplemented. */
int tf_ds4_estimate(const char *path, int context, uint64_t *out, int count,
                    char *err, size_t cap) {
    if (!err || !cap) return 1;
#ifdef DS4_NO_GPU
    snprintf(err, cap, "CUDA estimate requires the CUDA library"); return 1;
#else
    if (!path || context < 1 || !out || count != 7 || access(path, R_OK)) {
        snprintf(err, cap, "invalid estimate inputs"); return 1;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {0};
    opt.model_path = path; opt.backend = DS4_BACKEND_CPU;
    opt.inspect_only = true; opt.defer_boot_prewarm = true;
    if (ds4_engine_open(&engine, &opt)) {
        snprintf(err, cap, "metadata inspection failed"); return 1;
    }
    ds4_context_memory m = ds4_context_memory_estimate(DS4_BACKEND_CUDA, context);
    out[5] = ds4_engine_session_graph_bytes_estimate(engine, context);
    out[0] = out[5];
    /* The donor estimate includes F32 cache shells even when all readers and
     * writers use packed primaries. Those VMM shells remain uncommitted. Keep
     * every packed row and all scratch in the budget; remove only inactive
     * primary storage under the exact compiled/selected policy. */
    uint64_t inactive = tf_ds4_inactive_primary_bytes(context);
    if (inactive >= out[0]) {
        ds4_engine_close(engine);
        snprintf(err, cap, "invalid resident graph estimate"); return 1;
    }
    out[0] -= inactive;
    out[1] = m.raw_bytes; out[2] = m.compressed_bytes;
    out[3] = m.scratch_bytes; out[4] = m.prefill_cap;
    out[6] = tf_ds4_snapshot_bytes(context);
    ds4_engine_close(engine);
    return out[0] ? 0 : 1;
#endif
}
const char *tf_ds4_token_text(tf_ds4 *ctx, int token, size_t *len) {
    if (!len) return NULL;
    *len = 0;
    if (!ctx || token < 0 || token >= tf_ds4_vocab(ctx)) return NULL;
    return ds4_token_text(ctx->engine, token, len);
}

/* CPU-only packed IQ2 ABI sanity; no model/GPU/context is initialized. */
int tf_ds4_iq2_dot(const void *packed, int bytes, const int8_t *activation,
                   int count, float *out) {
    if (!packed || !activation || !out || bytes != 66 || count != 256) return 1;
    block_iq2_xxs block;
    block_q8_K rhs = {0};
    memcpy(&block, packed, sizeof(block));
    rhs.d = 1.0f;
    memcpy(rhs.qs, activation, 256);
    pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);
    ds4_vec_dot_iq2_xxs_q8_K(256, out, &block, &rhs);
    return 0;
}
