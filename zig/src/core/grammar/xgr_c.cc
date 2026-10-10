// The C ABI of xgr_c.h over xgrammar's C++ API, with the defaults of xgrammar 0.2.8's Python bindings.
#include "xgr_c.h"

#include <xgrammar/xgrammar.h>

#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

template <typename F>
auto guard(F&& f, decltype(f()) failed) -> decltype(f()) {
  try {
    last_error.clear();
    return f();
  } catch (const std::exception& e) {
    last_error = e.what();
  } catch (...) {
    last_error = "unknown xgrammar error";
  }
  return failed;
}
}  // namespace

struct xgr_tok {
  xgrammar::TokenizerInfo info;
};
struct xgr_compiler {
  xgrammar::GrammarCompiler c;
};
struct xgr_compiled {
  xgrammar::CompiledGrammar g;
};
struct xgr_matcher {
  xgrammar::GrammarMatcher m;
};

extern "C" {

const char* xgr_last_error(void) { return last_error.c_str(); }

int64_t xgr_detect_metadata(const char* backend, size_t len, char* out, size_t cap) {
  return guard(
      [&]() -> int64_t {
        std::string s = xgrammar::TokenizerInfo::DetectMetadataFromHF(std::string(backend, len));
        if (s.size() + 1 > cap) throw std::runtime_error("metadata buffer too small");
        std::memcpy(out, s.data(), s.size());
        out[s.size()] = 0;
        return static_cast<int64_t>(s.size());
      },
      -1
  );
}

xgr_tok* xgr_tok_new(const char* const* vocab, const size_t* lens, int32_t n, int32_t vocab_type, int32_t vocab_size,
                     const int32_t* stops, int32_t n_stops, int32_t add_prefix_space) {
  return guard(
      [&]() -> xgr_tok* {
        std::vector<std::string> enc;
        enc.reserve(n);
        for (int32_t i = 0; i < n; ++i) enc.emplace_back(vocab[i], lens[i]);
        std::vector<int32_t> st(stops, stops + n_stops);
        return new xgr_tok{xgrammar::TokenizerInfo(enc, static_cast<xgrammar::VocabType>(vocab_type), vocab_size, st,
                                                   add_prefix_space != 0)};
      },
      nullptr
  );
}

void xgr_tok_free(xgr_tok* t) { delete t; }

int32_t xgr_tok_special(const xgr_tok* t, int32_t* out, int32_t cap) {
  const auto& ids = t->info.GetSpecialTokenIds();
  int32_t n = static_cast<int32_t>(ids.size());
  for (int32_t i = 0; i < n && i < cap; ++i) out[i] = ids[i];
  return n;
}

int32_t xgr_tok_vocab_type(const xgr_tok* t) { return static_cast<int32_t>(t->info.GetVocabType()); }
int32_t xgr_tok_add_prefix_space(const xgr_tok* t) { return t->info.GetAddPrefixSpace() ? 1 : 0; }
int32_t xgr_bitmask_words(int32_t vocab_size) { return xgrammar::GetBitmaskSize(vocab_size); }

xgr_compiler* xgr_compiler_new(const xgr_tok* t, int32_t max_threads, int64_t cache_bytes) {
  return guard([&]() -> xgr_compiler* { return new xgr_compiler{xgrammar::GrammarCompiler(t->info, max_threads, true, cache_bytes)}; },
               nullptr);
}

void xgr_compiler_free(xgr_compiler* c) { delete c; }

xgr_compiled* xgr_compile(xgr_compiler* c, int32_t kind, const char* text, size_t len, int32_t max_whitespace) {
  return guard(
      [&]() -> xgr_compiled* {
        std::string s(text, len);
        switch (kind) {
          case XGR_JSON_SCHEMA: {
            std::optional<int> ws = max_whitespace < 0 ? std::nullopt : std::optional<int>(max_whitespace);
            return new xgr_compiled{c->c.CompileJSONSchema(s, true, std::nullopt, std::nullopt, true, ws, false)};
          }
          case XGR_REGEX:
            return new xgr_compiled{c->c.CompileRegex(s)};
          case XGR_EBNF:
            return new xgr_compiled{c->c.CompileGrammar(s, "root")};
          case XGR_STRUCTURAL_TAG:
            return new xgr_compiled{c->c.CompileStructuralTag(s)};
        }
        throw std::runtime_error("unknown grammar kind");
      },
      nullptr
  );
}

void xgr_compiled_free(xgr_compiled* g) { delete g; }

xgr_matcher* xgr_matcher_new(const xgr_compiled* g) {
  return guard([&]() -> xgr_matcher* { return new xgr_matcher{xgrammar::GrammarMatcher(g->g)}; }, nullptr);
}

void xgr_matcher_free(xgr_matcher* m) { delete m; }

int32_t xgr_matcher_accept(xgr_matcher* m, int32_t token) {
  return guard([&]() -> int32_t { return m->m.AcceptToken(token) ? 1 : 0; }, -1);
}

int32_t xgr_matcher_fill(xgr_matcher* m, int32_t* words, int32_t n_words) {
  return guard(
      [&]() -> int32_t {
        int64_t shape[2] = {1, n_words};
        DLTensor t;
        t.data = words;
        t.device = DLDevice{kDLCPU, 0};
        t.ndim = 2;
        t.dtype = xgrammar::GetBitmaskDLType();
        t.shape = shape;
        t.strides = nullptr;
        t.byte_offset = 0;
        return m->m.FillNextTokenBitmask(&t, 0) ? 1 : 0;
      },
      -1
  );
}

int32_t xgr_matcher_rollback(xgr_matcher* m, int32_t n) {
  return guard(
      [&]() -> int32_t {
        m->m.Rollback(n);
        return 0;
      },
      -1
  );
}

int32_t xgr_matcher_terminated(const xgr_matcher* m) { return m->m.IsTerminated() ? 1 : 0; }

}  // extern "C"
