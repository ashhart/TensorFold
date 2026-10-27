// libtfgrammar: a C ABI over xgrammar's C++ library for the native server (bitmask words: token t at bit t % 32 of t / 32).
#include <dlpack/dlpack.h>
#include <picojson.h>
#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tf_budget.h"

namespace {

// xgrammar's error without its timestamp and source location, first line only (grammar.py _message).
void message(char* err, size_t len, const char* text) {
    if (!err || len == 0) return;
    std::string t(text);
    t = t.substr(0, t.find('\n'));
    if (!t.empty() && t[0] == '[') {
        size_t close = t.find("] ");
        size_t colon = close == std::string::npos ? close : t.find(": ", close + 2);
        if (colon != std::string::npos) {
            // "[time] path/file.cc:123: words": the location is one token ending in ":<digits>"
            std::string where = t.substr(close + 2, colon - close - 2);
            size_t last = where.rfind(':');
            bool line = last != std::string::npos && last + 1 < where.size() && where.find(' ') == std::string::npos;
            for (size_t i = last + 1; line && i < where.size(); i++) line = where[i] >= '0' && where[i] <= '9';
            if (line) t = t.substr(colon + 2);
        }
    }
    std::strncpy(err, t.c_str(), len - 1);
    err[len - 1] = 0;
}

// Each token's decoded text to its id (the lowest id for a repeated text), for spelling a grammar's forced text.
struct Spellings {
    std::unordered_map<std::string, int32_t> ids;
    size_t longest = 0;
};

std::shared_ptr<const Spellings> spellings(const xgrammar::TokenizerInfo& info) {
    auto out = std::make_shared<Spellings>();
    const auto& vocab = info.GetDecodedVocab();
    std::vector<bool> skip(vocab.size(), false);
    for (int32_t t : info.GetSpecialTokenIds()) if (t >= 0 && (size_t)t < skip.size()) skip[t] = true;
    for (int32_t t : info.GetStopTokenIds()) if (t >= 0 && (size_t)t < skip.size()) skip[t] = true;
    for (size_t t = 0; t < vocab.size(); t++) {
        if (skip[t] || vocab[t].empty()) continue;
        out->ids.emplace(vocab[t], (int32_t)t);  // emplace keeps the first (lowest) id
        out->longest = std::max(out->longest, vocab[t].size());
    }
    return out;
}

struct Compiler {
    xgrammar::TokenizerInfo info;
    xgrammar::GrammarCompiler compiler;
    int vocab;
    std::shared_ptr<const Spellings> spell;
    tfb::Cache budgets;
};

struct Compiled {
    xgrammar::CompiledGrammar g;
    std::shared_ptr<const Spellings> spell;
    Compiler* owner;
};

struct Matcher {
    xgrammar::GrammarMatcher m;
    std::shared_ptr<const Spellings> spell;
    Compiler* owner;
    xgrammar::CompiledGrammar g;
    std::unique_ptr<tfb::Follower> budget;  // armed by tfg_budget before the first token (grammar_budget)
};

constexpr int BLANKS = 32;                      // grammar.py BLANKS
constexpr int64_t CACHE_BYTES = 256ll << 20;    // grammar.py CACHE_BYTES
const char* OBJECT = "{\"type\": \"object\"}";  // grammar.py OBJECT: json_object

}  // namespace

extern "C" {

// tokenizer.json's text; vocab: the logits' width; stops: the model's eos ids. NULL with ``err`` on failure.
void* tfg_open(const char* tokenizer_json, size_t len, int vocab, const int32_t* stops, int nstops, char* err, size_t errlen) {
    try {
        std::string text(tokenizer_json, len);
        picojson::value v;
        std::string perr = picojson::parse(v, text);
        if (!perr.empty() || !v.is<picojson::object>()) { message(err, errlen, "tokenizer.json is not a JSON object"); return nullptr; }
        const auto& o = v.get<picojson::object>();
        std::vector<std::string> encoded(vocab);
        auto model = o.find("model");
        if (model == o.end() || !model->second.is<picojson::object>()) { message(err, errlen, "tokenizer.json has no model"); return nullptr; }
        const auto& mo = model->second.get<picojson::object>();
        auto vocab_it = mo.find("vocab");
        if (vocab_it == mo.end() || !vocab_it->second.is<picojson::object>()) { message(err, errlen, "tokenizer.json's model has no vocab"); return nullptr; }
        for (const auto& kv : vocab_it->second.get<picojson::object>()) {
            int64_t id = (int64_t)kv.second.get<double>();
            if (id >= 0 && id < vocab) encoded[id] = kv.first;
        }
        auto added = o.find("added_tokens");
        if (added != o.end() && added->second.is<picojson::array>())
            for (const auto& t : added->second.get<picojson::array>()) {
                if (!t.is<picojson::object>()) continue;
                const auto& to = t.get<picojson::object>();
                auto id = to.find("id"), content = to.find("content");
                if (id == to.end() || content == to.end() || !content->second.is<std::string>()) continue;
                int64_t i = (int64_t)id->second.get<double>();
                if (i >= 0 && i < vocab) encoded[i] = content->second.get<std::string>();
            }
        std::string metadata = xgrammar::TokenizerInfo::DetectMetadataFromHF(text);
        picojson::value mv;
        picojson::parse(mv, metadata);
        const auto& mo2 = mv.get<picojson::object>();
        auto type = (xgrammar::VocabType)(int)mo2.at("vocab_type").get<double>();
        bool prefix = mo2.at("add_prefix_space").get<bool>();
        std::vector<int32_t> stop(stops, stops + nstops);
        xgrammar::TokenizerInfo info(encoded, type, vocab, stop, prefix);
        return new Compiler{info, xgrammar::GrammarCompiler(info, 8, true, CACHE_BYTES), vocab, spellings(info), tfb::Cache(info)};
    } catch (const std::exception& e) {
        message(err, errlen, e.what());
    } catch (...) {
        message(err, errlen, "xgrammar failed");
    }
    return nullptr;
}

void tfg_close(void* c) { delete static_cast<Compiler*>(c); }

// Words a row's bitmask holds.
int tfg_words(void* c) { return (static_cast<Compiler*>(c)->vocab + 31) / 32; }

// kind: 0 json (any object), 1 json_schema, 2 regex, 3 choice (JSON strings), 4 EBNF, 5 structural tag (xgrammar JSON).
void* tfg_compile(void* cp, int kind, const char* text, size_t len, char* err, size_t errlen) {
    auto* c = static_cast<Compiler*>(cp);
    try {
        std::string t(text, len);
        xgrammar::CompiledGrammar g = [&] {
            switch (kind) {
                case 0: return c->compiler.CompileJSONSchema(OBJECT, true, std::nullopt, std::nullopt, true, BLANKS);
                case 1: return c->compiler.CompileJSONSchema(t, true, std::nullopt, std::nullopt, true, BLANKS);
                case 2: return c->compiler.CompileRegex(t);
                case 3: {
                    picojson::value v;
                    std::string perr = picojson::parse(v, t);
                    if (!perr.empty() || !v.is<picojson::array>()) throw std::runtime_error("choices must be a JSON array");
                    std::string ebnf = "root ::= ";
                    bool first = true;
                    for (const auto& x : v.get<picojson::array>()) {
                        if (!first) ebnf += " | ";
                        first = false;
                        ebnf += x.serialize();
                    }
                    return c->compiler.CompileGrammar(ebnf);
                }
                case 5: return c->compiler.CompileStructuralTag(t);
                default: return c->compiler.CompileGrammar(t);
            }
        }();
        return new Compiled{g, c->spell, c};
    } catch (const std::exception& e) {
        message(err, errlen, e.what());
    } catch (...) {
        message(err, errlen, "xgrammar failed");
    }
    return nullptr;
}

void tfg_free(void* g) { delete static_cast<Compiled*>(g); }

void* tfg_matcher(void* g) {
    try {
        auto* c = static_cast<Compiled*>(g);
        return new Matcher{xgrammar::GrammarMatcher(c->g), c->spell, c->owner, c->g, nullptr};
    } catch (...) {
        return nullptr;
    }
}

void tfg_matcher_free(void* m) { delete static_cast<Matcher*>(m); }

// 1 taken, 0 rejected, -1 failed.
int tfg_accept(void* m, int32_t token) {
    try {
        auto* w = static_cast<Matcher*>(m);
        if (!w->m.AcceptToken(token)) return 0;
        if (w->budget) w->budget->Accept(token);
        return 1;
    } catch (...) {
        return -1;
    }
}

int tfg_rollback(void* m, int n) {
    try {
        auto* w = static_cast<Matcher*>(m);
        w->m.Rollback(n);
        if (w->budget) w->budget->Rollback(n);
        return 0;
    } catch (...) {
        return -1;
    }
}

// grammar_budget's tables for a compiled grammar, explored once: 1 covered, 0 not, -1 failed.
int tfg_budget_ready(void* g) {
    try {
        auto* c = static_cast<Compiled*>(g);
        return tfb::Ready(*c->owner->budgets.Get(c->g)) ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

// n == 0 arms a fresh matcher, else clears bits that leave no completion in left - 1 more: 0, 1 not covered, -1 failed.
int tfg_budget(void* m, int32_t* words, int n, int left) {
    try {
        auto* w = static_cast<Matcher*>(m);
        if (!w->budget) {
            if (n > 0) return 1;
            w->budget = std::make_unique<tfb::Follower>(w->owner->budgets.Get(w->g));
            return 0;
        }
        return n == 0 ? 0 : w->budget->Mask(words, n, left);
    } catch (...) {
        return -1;
    }
}

int tfg_terminated(void* m) { return static_cast<Matcher*>(m)->m.IsTerminated() ? 1 : 0; }

// The next token's allowed bits into ``words`` (tfg_words of them). 0, or -1 when it failed.
int tfg_fill(void* m, int32_t* words, int n) {
    try {
        int64_t shape[2] = {1, n};
        DLTensor t{};
        t.data = words;
        t.device = {kDLCPU, 0};
        t.ndim = 2;
        t.dtype = {kDLInt, 32, 1};
        t.shape = shape;
        static_cast<Matcher*>(m)->m.FillNextTokenBitmask(&t, 0);
        return 0;
    } catch (...) {
        return -1;
    }
}

// Up to n tokens spelling the grammar's forced text, longest first, never its last character (models join it on).
int tfg_drafts(void* mp, int32_t* out, int n) {
    auto* w = static_cast<Matcher*>(mp);
    int got = 0;
    bool failed = false;
    try {
        while (got < n && !w->m.IsTerminated()) {
            std::string forced = w->m.FindJumpForwardString();
            int32_t pick = -1;
            for (size_t len = std::min(forced.size() - (forced.empty() ? 0 : 1), w->spell->longest); len > 0 && pick < 0; len--) {
                auto it = w->spell->ids.find(forced.substr(0, len));
                if (it != w->spell->ids.end()) pick = it->second;
            }
            if (pick < 0 || !w->m.AcceptToken(pick)) break;
            out[got++] = pick;
        }
    } catch (...) {
        failed = true;
    }
    try {
        if (got > 0) w->m.Rollback(got);
    } catch (...) {
        return -1;
    }
    return failed ? -1 : got;
}

}  // extern "C"
