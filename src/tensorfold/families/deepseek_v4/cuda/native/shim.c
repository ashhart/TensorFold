/* TensorFold's small, versioned boundary over the pinned MIT ds4 engine.
 * Include the donor translation unit so the ABI smoke can call its IQ2 primitive.
 * The donor implementation and its arithmetic are otherwise unchanged. */
#include "ds4.c"

typedef struct {
    ds4_engine *engine;
    ds4_session *session;
} tf_ds4;

int tf_ds4_abi(void) { return 1; }
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
    if (ctx->session) ds4_session_free(ctx->session);
    if (ctx->engine) ds4_engine_close(ctx->engine);
    free(ctx);
}

int tf_ds4_open(const char *path, int context, int threads, tf_ds4 **out,
                char *err, size_t cap) {
    if (!path || !out || context < 1 || threads < 1 || access(path, R_OK)) {
        snprintf(err, cap, "invalid native model path/context/threads");
        return 1;
    }
    *out = NULL;
    tf_ds4 *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) { snprintf(err, cap, "native allocation failed"); return 1; }
    ds4_engine_options opt = {0};
    opt.model_path = path;
    opt.backend = (ds4_backend)tf_ds4_backend();
    opt.n_threads = threads;
    opt.power_percent = 100;
    /* No MTP, DSpark, distributed execution, graph snapshots or boot warmup. */
    opt.defer_boot_prewarm = true;
    if (ds4_engine_open(&ctx->engine, &opt) ||
        ds4_session_create(&ctx->session, ctx->engine, context)) {
        snprintf(err, cap, "native engine/session initialization failed");
        tf_ds4_close(ctx);
        return 1;
    }
    *out = ctx;
    return 0;
}

int tf_ds4_vocab(tf_ds4 *ctx) { return ds4_engine_vocab_size(ctx->engine); }
int tf_ds4_eos(tf_ds4 *ctx) { return ds4_token_eos(ctx->engine); }
int tf_ds4_context(tf_ds4 *ctx) { return ds4_session_ctx(ctx->session); }
void tf_ds4_reset(tf_ds4 *ctx) { ds4_session_invalidate(ctx->session); }
int tf_ds4_sync(tf_ds4 *ctx, const int *ids, int count, char *err, size_t cap) {
    if (!ctx || !ids || count <= 0 || count > ds4_session_ctx(ctx->session)) {
        snprintf(err, cap, "invalid native prompt length"); return 1;
    }
    ds4_tokens tokens = {(int *)ids, count, count};
    return ds4_session_sync(ctx->session, &tokens, err, cap);
}
int tf_ds4_eval(tf_ds4 *ctx, int id, char *err, size_t cap) {
    if (!ctx || id < 0 || id >= tf_ds4_vocab(ctx) ||
        ds4_session_pos(ctx->session) >= ds4_session_ctx(ctx->session)) {
        snprintf(err, cap, "invalid native token or exhausted context"); return 1;
    }
    return ds4_session_eval(ctx->session, id, err, cap);
}
int tf_ds4_logits(tf_ds4 *ctx, float *out, int count) {
    return ds4_session_copy_logits(ctx->session, out, count);
}
int tf_ds4_encode(tf_ds4 *ctx, const char *text, int rendered, int **out) {
    ds4_tokens tokens = {0};
    if (rendered) ds4_tokenize_rendered_chat(ctx->engine, text, &tokens);
    else ds4_tokenize_text(ctx->engine, text, &tokens);
    *out = tokens.v;
    return tokens.len;
}
void tf_ds4_free(void *ptr) { free(ptr); }
const char *tf_ds4_token_text(tf_ds4 *ctx, int token, size_t *len) {
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
