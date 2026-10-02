/* TensorFold's small, versioned boundary over the pinned MIT ds4 engine.
 * Include the donor translation unit so the ABI smoke can call its IQ2 primitive.
 * The donor implementation and its arithmetic are otherwise unchanged. */
typedef int (*tf_ds4_sample_fn)(const float *, int, void *);
static __thread tf_ds4_sample_fn tf_ds4_sampler;
static __thread void *tf_ds4_sampler_ud;
static __thread void (*tf_ds4_checkpoint)(void *, int);
static __thread void *tf_ds4_checkpoint_ud;
#include "ds4-sampling.inc"

typedef struct {
    ds4_engine *engine;
    ds4_session *session;
    ds4_batch_ctx *batch;
    int context;
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
    if (ctx->batch) ds4_batch_ctx_destroy(ctx->batch);
    if (ctx->engine) ds4_engine_close(ctx->engine);
    free(ctx);
}

#ifndef DS4_NO_GPU
static uint32_t tf_ds4_draft_prefill_cap(int context) {
    return context < 1024 ? (uint32_t)context : 1024;
}
static int tf_ds4_draft_prefix_cap(int context) {
    return context < 131072 ? context : 131072;
}

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

static uint64_t tf_ds4_draft_snapshot_bytes(int context) {
    uint32_t raw_cap = metal_graph_raw_cap_for_context(context, tf_ds4_draft_prefill_cap(context));
    /* Retention is bounded independently of the admitted request context. */
    int retained = tf_ds4_draft_prefix_cap(context);
    return tf_ds4_snapshot_bytes(retained)
        + (uint64_t)DS4_DSPARK_N_LAYER * raw_cap * DS4_N_HEAD_DIM * sizeof(float);
}

/* Save only canonical prefill boundaries, never a decode-built frontier.
 * The existing bank wire format owns target state; the injected DSpark rings
 * travel with it so restored requests retain their drafter's prefix as well. */
static void tf_ds4_draft_save_boundary(void *ud, int current) {
    tf_ds4 *ctx = ud;
    if (current != ctx->snapshot_target || current <= 0
        || current == ctx->snapshot_tokens || ctx->snapshot_failed) return;
    ctx->snapshot_tokens = 0;
    uint64_t bytes = ds4_cont_bank_payload_bytes(ctx->batch, 0);
    uint64_t ring = (uint64_t)ctx->batch->g.raw_cap * DS4_N_HEAD_DIM * sizeof(float);
    if (!bytes || bytes + DS4_DSPARK_N_LAYER * ring >= ctx->snapshot_limit) goto failed;
    FILE *fp = fmemopen(ctx->snapshot.ptr, ctx->snapshot_limit, "w+b");
    if (!fp) goto failed;
    int rc = ds4_cont_bank_save_payload(ctx->batch, 0, fp,
        ctx->snapshot_error, sizeof(ctx->snapshot_error));
    if (!rc && (fflush(fp) || ftello(fp) != (off_t)bytes)) rc = 1;
    if (fclose(fp)) rc = 1;
    if (rc) goto failed;
    for (uint32_t il = 0; il < DS4_DSPARK_N_LAYER; il++) {
        if (!ds4_gpu_tensor_read(ctx->batch->dsl.multi_raw[il], 0,
            (char *)ctx->snapshot.ptr + bytes + il * ring, ring)) goto failed;
    }
    const int *history = NULL;
    if (ds4_batch_ctx_bank_committed(ctx->batch, 0, &history) != current || !history) goto failed;
    memcpy(ctx->snapshot_prefix, history, (size_t)current * sizeof(int));
    ctx->snapshot.len = bytes;
    ctx->snapshot_tokens = current;
    return;
failed:
    ctx->snapshot.len = 0;
    ctx->snapshot_failed = true;
    fprintf(stderr, "tensorfold: canonical bank checkpoint unavailable: %s\n",
        *ctx->snapshot_error ? ctx->snapshot_error : "capacity or tensor copy failed");
}

static int tf_ds4_draft_restore(tf_ds4 *ctx, char *err, size_t cap) {
    FILE *fp = fmemopen(ctx->snapshot.ptr, ctx->snapshot.len, "rb");
    if (!fp) { snprintf(err, cap, "canonical bank checkpoint open failed"); return 1; }
    int rc = ds4_cont_bank_restore_payload(ctx->batch, 0, fp, ctx->snapshot.len, err, cap);
    if (fclose(fp)) rc = 1;
    uint64_t ring = (uint64_t)ctx->batch->g.raw_cap * DS4_N_HEAD_DIM * sizeof(float);
    for (uint32_t il = 0; !rc && il < DS4_DSPARK_N_LAYER; il++) {
        if (!ds4_gpu_tensor_write(ctx->batch->dsl.multi_raw[il], 0,
            (char *)ctx->snapshot.ptr + ctx->snapshot.len + il * ring, ring)) rc = 1;
    }
    if (rc && !*err) snprintf(err, cap, "canonical bank checkpoint restore failed");
    return rc;
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

int tf_ds4_draft_estimate(const char *path, int context, uint64_t *out,
                char *err, size_t cap) {
    if (!err || !cap) return 1;
#ifdef DS4_NO_GPU
    snprintf(err, cap, "DSpark requires CUDA"); return 1;
#else
    if (!path || access(path, R_OK) || context < 1 || !out) {
        snprintf(err, cap, "invalid DSpark estimate inputs"); return 1;
    }
    ds4_engine_options opt = {0};
    opt.model_path = path; opt.backend = DS4_BACKEND_CPU;
    opt.inspect_only = true; opt.defer_boot_prewarm = true;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt)) return 1;
    uint32_t pc = tf_ds4_draft_prefill_cap(context);
    uint32_t rc = metal_graph_raw_cap_for_context(context, pc);
    uint64_t inactive = tf_ds4_inactive_primary_bytes(context);
    uint64_t graph = metal_graph_alloc_bytes_estimate(&engine->weights,
        &engine->weights.layer[0], rc, context, pc, false, true);
    /* Price commons and exactly one full-depth bank with the donor's own
     * ledger math. Inactive F32 VMM shells have no physical pages under the
     * packed policy, just as in the serial admission estimator. */
    ds4_gpu_graph g = {0}; g.raw_cap = rc;
    /* The ledger only tests tensor presence; use a non-null metadata anchor.
     * No device tensor is allocated or dereferenced by this inspection. */
    ds4_gpu_tensor *present = (ds4_gpu_tensor *)engine->model.map;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        g.layer_comp_cap[il] = context / ratio + 2u;
        g.layer_attn_comp_cache[il] = present;
        g.layer_comp_cache_fp8[il] = !DS4_GPU_ATTN_COMP_CACHE_F16 && ds4_cuda_fp8_kv_enabled() ? present : NULL;
        if (ratio == 4) {
            g.layer_index_comp_cache[il] = present;
            g.layer_index_comp_cache_fp4[il] = ds4_cuda_fp4_index_enabled() ? present : NULL;
        }
    }
    uint64_t bank = ds4_batch_slabs_bank_bytes(&g, false, false, true);
    if (graph <= inactive || bank <= inactive) {
        ds4_engine_close(engine);
        snprintf(err, cap, "invalid packed DSpark estimate"); return 1;
    }
    *out = graph - inactive + bank - inactive + tf_ds4_draft_snapshot_bytes(context);
    ds4_engine_close(engine);
    return *out ? 0 : 1;
#endif
}

int tf_ds4_draft_open(const char *path, const char *drafter, int context,
                int threads, tf_ds4 **out, char *err, size_t cap) {
    if (out) *out = NULL;
    if (!err || !cap) return 1;
    if (!out || !path || !drafter || access(path, R_OK) || access(drafter, R_OK)
        || context < 1 || threads < 1) {
        snprintf(err, cap, "invalid DSpark model/context/threads"); return 1;
    }
    tf_ds4 *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { snprintf(err, cap, "DSpark allocation failed"); return 1; }
    ds4_engine_options opt = {0};
    opt.model_path = path; opt.dspark_path = drafter;
    opt.backend = (ds4_backend)tf_ds4_backend();
    opt.n_threads = threads; opt.power_percent = 100;
    ctx->context = context;
    int pc = context < 512 ? context : 512;
#ifndef DS4_NO_GPU
    pc = (int)tf_ds4_draft_prefill_cap(context);
#endif
    if (ds4_engine_open(&ctx->engine, &opt)
        || ds4_batch_ctx_create_fit(ctx->engine, context, 1, pc, &ctx->batch, err, cap)) {
        if (!*err) snprintf(err, cap, "DSpark startup failed");
        tf_ds4_close(ctx); return 1;
    }
#ifndef DS4_NO_GPU
    uint64_t budget = tf_ds4_draft_snapshot_bytes(context);
    uint64_t prefix_bytes = (uint64_t)tf_ds4_draft_prefix_cap(context) * sizeof(int);
    if (budget <= prefix_bytes + DS4_SESSION_IO_CHUNK || budget > SIZE_MAX) {
        snprintf(err, cap, "invalid canonical bank checkpoint capacity");
        tf_ds4_close(ctx); return 1;
    }
    ctx->snapshot_limit = budget - prefix_bytes - DS4_SESSION_IO_CHUNK;
    ctx->snapshot.ptr = malloc(ctx->snapshot_limit);
    ctx->snapshot.cap = ctx->snapshot_limit;
    ctx->snapshot_prefix = malloc((size_t)prefix_bytes);
    if (!ctx->snapshot.ptr || !ctx->snapshot_prefix) {
        snprintf(err, cap, "canonical bank checkpoint allocation failed");
        tf_ds4_close(ctx); return 1;
    }
#endif
    *out = ctx;
    return 0;
}

typedef int (*tf_ds4_emit_fn)(int token, void *ud);
typedef struct {
    tf_ds4 *ctx;
    const int *ids;
    int count, budget, eos, admitted, cached, generated, emitted, failed;
    tf_ds4_emit_fn emit;
    void *ud;
} tf_ds4_generation;

static int tf_ds4_admitted(void *ud, void *user, int cached, int computed, int bank) {
    (void)user; (void)computed; (void)bank;
    ((tf_ds4_generation *)ud)->cached = cached;
    return 1;
}
static int tf_ds4_admit(void *ud, ds4_cont_request *req) {
    tf_ds4_generation *g = ud;
    if (g->admitted++) return 0;
    memset(req, 0, sizeof(*req));
    req->tokens = g->ids; req->n = g->count; req->max_new = g->budget;
    req->eos = g->eos; req->on_admitted = tf_ds4_admitted;
    const int *history = NULL;
    int n = ds4_batch_ctx_bank_committed(g->ctx->batch, 0, &history);
    /* generate restores a matching canonical boundary before admission. */
    if (n > 0 && n <= g->count && history
        && !memcmp(g->ids, history, (size_t)n * sizeof(int))) {
        req->place_bank = 1; req->n_cached = n;
    }
    return 1;
}
static int tf_ds4_emit(void *ud, void *user, int token) {
    (void)user;
    tf_ds4_generation *g = ud;
    g->emitted++;
    return g->emit(token, g->ud);
}
static void tf_ds4_done(void *ud, void *user, const int *tokens, int n, int finish) {
    (void)user; (void)finish;
    tf_ds4_generation *g = ud;
    if (!tokens || n < g->emitted || n > g->budget) { g->failed = 1; return; }
    /* Native streaming omits EOS; TensorFold callbacks receive it once. */
    for (int i = g->emitted; i < n; i++) g->emit(tokens[i], g->ud);
    g->generated = n;
}
int tf_ds4_draft_generate(tf_ds4 *ctx, const int *ids, int count, int budget,
                int draft, int stop_eos, tf_ds4_sample_fn sample,
                tf_ds4_emit_fn emit, void *ud, int *cached,
                char *err, size_t cap) {
    if (!err || !cap) return -1;
#ifdef DS4_NO_GPU
    snprintf(err, cap, "DSpark requires CUDA"); return -1;
#else
    if (!ctx || !ctx->batch || !ids || count < 1 || budget < 1
        || count > ctx->context - budget || !sample || !emit || !cached || tf_ds4_sampler) {
        snprintf(err, cap, "invalid DSpark generation inputs"); return -1;
    }
    for (int i = 0; i < count; i++) {
        if (ids[i] < 0 || ids[i] >= ds4_engine_vocab_size(ctx->engine)) {
            snprintf(err, cap, "invalid DSpark prompt token"); return -1;
        }
    }
    tf_ds4_generation g = {0};
    if (ctx->snapshot_tokens > 0 && count > ctx->snapshot_tokens
        && !memcmp(ids, ctx->snapshot_prefix, (size_t)ctx->snapshot_tokens * sizeof(int))) {
        if (tf_ds4_draft_restore(ctx, err, cap)) {
            bank_hist_invalidate_all(ctx->batch);
            ctx->snapshot_tokens = 0;
            return -1;
        }
    } else {
        bank_hist_invalidate_all(ctx->batch);
        ctx->snapshot_tokens = 0;
    }
    ctx->snapshot_target = (count - 1) / ctx->batch->prefill_cap * ctx->batch->prefill_cap;
    int retained = tf_ds4_draft_prefix_cap(ctx->context);
    if (ctx->snapshot_target > retained) ctx->snapshot_target = retained;
    ctx->snapshot_failed = false;
    *ctx->snapshot_error = '\0';
    g.ctx = ctx; g.ids = ids; g.count = count; g.budget = budget;
    g.eos = stop_eos ? ds4_token_eos(ctx->engine) : ds4_engine_vocab_size(ctx->engine);
    g.emit = emit; g.ud = ud;
    /* Keep the same target/sampler path for --no-drafts and draft:false. */
    int previous_mode = ctx->batch->mtp_draft_mode;
    ctx->batch->mtp_draft_mode = draft ? 2 : 0;
    tf_ds4_sampler = sample; tf_ds4_sampler_ud = ud;
    tf_ds4_checkpoint = tf_ds4_draft_save_boundary; tf_ds4_checkpoint_ud = ctx;
    int rc = ds4_engine_continuous_generate(ctx->batch, tf_ds4_admit,
        tf_ds4_emit, tf_ds4_done, &g, err, cap);
    tf_ds4_sampler = NULL; tf_ds4_sampler_ud = NULL;
    tf_ds4_checkpoint = NULL; tf_ds4_checkpoint_ud = NULL;
    ctx->batch->mtp_draft_mode = previous_mode;
    *cached = g.cached;
    if (rc || g.failed) {
        if (!*err) snprintf(err, cap, "DSpark request refused or failed");
        return -1;
    }
    return g.generated;
#endif
}

int tf_ds4_vocab(tf_ds4 *ctx) { return ctx ? ds4_engine_vocab_size(ctx->engine) : 0; }
int tf_ds4_eos(tf_ds4 *ctx) { return ctx ? ds4_token_eos(ctx->engine) : -1; }
int tf_ds4_context(tf_ds4 *ctx) { return ctx ? (ctx->batch ? ctx->context : ds4_session_ctx(ctx->session)) : 0; }
void tf_ds4_reset(tf_ds4 *ctx) {
    if (!ctx) return;
#ifndef DS4_NO_GPU
    if (ctx->batch) { bank_hist_invalidate_all(ctx->batch); ctx->snapshot_tokens = 0; return; }
#endif
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
/* Explicit byte length preserves NUL in ordinary text, including tool output.
 * Ordinary spans use the donor's unchanged pre-tokenizer and BPE merge logic. */
int tf_ds4_encode_bytes(tf_ds4 *ctx, const char *text, size_t bytes, int rendered, int **out) {
    if (!out) return -1;
    *out = NULL;
    if (!ctx || !text || bytes > INT_MAX) return -1;
    ds4_tokens tokens = {0};
    const ds4_vocab *vocab = &ctx->engine->vocab;
    if (!rendered) {
        tf_ds4_bpe_bytes(vocab, text, bytes, &tokens);
    } else {
        size_t span = 0, pos = 0;
        while (pos < bytes) {
            int token = -1;
            size_t length = 0;
            if (tf_ds4_special_bytes(vocab, text + pos, bytes - pos, &token, &length)) {
                tf_ds4_bpe_bytes(vocab, text + span, pos - span, &tokens);
                token_vec_push(&tokens, token);
                pos += length;
                span = pos;
            } else pos++;
        }
        tf_ds4_bpe_bytes(vocab, text + span, bytes - span, &tokens);
    }
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
