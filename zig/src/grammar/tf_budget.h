// libtfgrammar's budget masks: the fewest tokens from a grammar's parser state to a reply it completes.
#pragma once
#include "tf_compile.h"

#include <xgrammar/xgrammar.h>

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <utility>

namespace tfb {

struct Trie;
class Tables;

// What exploring one grammar may take, in counted steps (no clock): past it the grammar is refused, never part-masked.
struct Limits {
  int64_t work = int64_t{1} << 28;  // counted parser/table steps, plus the compile-thread deadline
  int64_t edges = int64_t{1} << 20;   // total token successors retained by one grammar
  int64_t bytes = int64_t{128} << 20; // charged state keys, byte tables, names and edge storage
};

// A completed asynchronous exploration; owning the compiled grammar keeps its cache key unique.
struct ReadyResult {
  std::shared_ptr<Tables> tables;
  std::shared_ptr<const xgrammar::CompiledGrammar> owner;
};

// One grammar's exploration, bounded by work and tfc's compile-thread deadline.
int Ready(Tables& t, const std::atomic<bool>* cancel = nullptr);

// One tokenizer's tables for the grammars it compiled lately, found by the compiled grammar.
class Cache {
 public:
  explicit Cache(xgrammar::TokenizerInfo info) : info_(std::move(info)) {}
  size_t Active() const { return compiles_.Active(); }
  std::shared_ptr<Tables> Get(const xgrammar::CompiledGrammar& g);
  // The limits of every grammar first looked up after this.
  void Limit(Limits limits);

 private:
  xgrammar::TokenizerInfo info_;
  std::once_flag once_;
  std::shared_ptr<const Trie> trie_;
  std::mutex mutex_;
  std::list<std::pair<xgrammar::CompiledGrammar, std::shared_ptr<Tables>>> recent_;
  Limits limits_;
  tfc::Cache<ReadyResult> compiles_;
};

// Explore an already prepared grammar's tables once: 1 ready, 0 uncovered, 2 over the work limit.

// A matcher's place mirrored for its budget masks: armed before its first token, told of every token after.
class Follower {
 public:
  explicit Follower(std::shared_ptr<Tables> t);
  ~Follower();
  void Accept(int32_t token);
  void Rollback(int n);
  // Clear the bits of tokens after which no reply completes within `left - 1` more: 0, 1 not covered, -1 failed.
  int Mask(int32_t* words, int n, int left);

 private:
  struct State;
  std::unique_ptr<State> s_;
};

}  // namespace tfb
