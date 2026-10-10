// A C ABI over xgrammar 0.2.8's C++ core; failing calls return NULL or -1 with the message in xgr_last_error().
#ifndef TF_XGR_C_H_
#define TF_XGR_C_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xgr_tok xgr_tok;
typedef struct xgr_compiler xgr_compiler;
typedef struct xgr_compiled xgr_compiled;
typedef struct xgr_matcher xgr_matcher;

enum xgr_kind {
    XGR_JSON_SCHEMA = 0,   // any_whitespace, strict_mode, max_whitespace_cnt (< 0: none)
    XGR_REGEX = 1,
    XGR_EBNF = 2,          // root rule "root"
    XGR_STRUCTURAL_TAG = 3 // the structural tag's JSON
};

// The calling thread's last error message (empty when none); valid until the thread's next failing call.
const char *xgr_last_error(void);

// {"vocab_type": int, "add_prefix_space": bool} for a tokenizer.json into out (NUL-terminated); its length or -1.
int64_t xgr_detect_metadata(const char *backend, size_t len, char *out, size_t cap);

// TokenizerInfo(encoded_vocab, vocab_type, vocab_size, stop_token_ids, add_prefix_space).
xgr_tok *xgr_tok_new(const char *const *vocab, const size_t *lens, int32_t n, int32_t vocab_type, int32_t vocab_size,
                     const int32_t *stops, int32_t n_stops, int32_t add_prefix_space);
void xgr_tok_free(xgr_tok *t);
// special_token_ids into out (at most cap); the count.
int32_t xgr_tok_special(const xgr_tok *t, int32_t *out, int32_t cap);
int32_t xgr_tok_vocab_type(const xgr_tok *t);
int32_t xgr_tok_add_prefix_space(const xgr_tok *t);
// The bitmask's int32 words for a vocabulary of vocab_size tokens.
int32_t xgr_bitmask_words(int32_t vocab_size);

xgr_compiler *xgr_compiler_new(const xgr_tok *t, int32_t max_threads, int64_t cache_bytes);
void xgr_compiler_free(xgr_compiler *c);
xgr_compiled *xgr_compile(xgr_compiler *c, int32_t kind, const char *text, size_t len, int32_t max_whitespace);
void xgr_compiled_free(xgr_compiled *g);

xgr_matcher *xgr_matcher_new(const xgr_compiled *g);
void xgr_matcher_free(xgr_matcher *m);
// 1 accepted, 0 rejected, -1 failed.
int32_t xgr_matcher_accept(xgr_matcher *m, int32_t token);
// The next token's allowed bits (token t: bit t % 32 of word t / 32); 1 some token masked, 0 none, -1 failed.
int32_t xgr_matcher_fill(xgr_matcher *m, int32_t *words, int32_t n_words);
int32_t xgr_matcher_rollback(xgr_matcher *m, int32_t n);
int32_t xgr_matcher_terminated(const xgr_matcher *m);

#ifdef __cplusplus
}
#endif
#endif
