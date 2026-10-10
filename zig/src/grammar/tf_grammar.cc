// libtfgrammar: structured output for the native server, through xgrammar's C++ library (zig/build/grammar.zig's
// pinned tag) as tensorfold/engine/grammar.py drives it: the tokenizer's vocabulary from tokenizer.json
// (its model vocabulary and added tokens: PreTrainedTokenizerFast.get_vocab), the vocabulary size the logits have,
// the model's stop tokens; json / json_schema through CompileJSONSchema (at most 32 blank characters between
// tokens), regex, a choice as an EBNF of JSON string literals, EBNF; a matcher per reply that takes tokens, rolls
// back and fills next-token bitmasks (int32 words, token t is bit t % 32 of word t / 32).
#include <dlpack/dlpack.h>
#include <picojson.h>
#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

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

struct Compiler {
    xgrammar::TokenizerInfo info;
    xgrammar::GrammarCompiler compiler;
    int vocab;
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
        return new Compiler{info, xgrammar::GrammarCompiler(info, 8, true, CACHE_BYTES), vocab};
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

// kind: 0 json (any object), 1 json_schema, 2 regex, 3 choice (a JSON array of strings), 4 EBNF grammar.
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
                default: return c->compiler.CompileGrammar(t);
            }
        }();
        return new xgrammar::CompiledGrammar(g);
    } catch (const std::exception& e) {
        message(err, errlen, e.what());
    } catch (...) {
        message(err, errlen, "xgrammar failed");
    }
    return nullptr;
}

void tfg_free(void* g) { delete static_cast<xgrammar::CompiledGrammar*>(g); }

void* tfg_matcher(void* g) {
    try {
        return new xgrammar::GrammarMatcher(*static_cast<xgrammar::CompiledGrammar*>(g));
    } catch (...) {
        return nullptr;
    }
}

void tfg_matcher_free(void* m) { delete static_cast<xgrammar::GrammarMatcher*>(m); }

// 1 taken, 0 rejected, -1 failed.
int tfg_accept(void* m, int32_t token) {
    try {
        return static_cast<xgrammar::GrammarMatcher*>(m)->AcceptToken(token) ? 1 : 0;
    } catch (...) {
        return -1;
    }
}

int tfg_rollback(void* m, int n) {
    try {
        static_cast<xgrammar::GrammarMatcher*>(m)->Rollback(n);
        return 0;
    } catch (...) {
        return -1;
    }
}

int tfg_terminated(void* m) { return static_cast<xgrammar::GrammarMatcher*>(m)->IsTerminated() ? 1 : 0; }

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
        static_cast<xgrammar::GrammarMatcher*>(m)->FillNextTokenBitmask(&t, 0);
        return 0;
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
