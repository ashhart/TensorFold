// Budget masks: every parser state a grammar reaches at a token boundary, its token steps, and its fewest to complete.
#include "tf_budget.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

#include "earley_parser.h"

namespace tfb {
namespace {

using xgrammar::EarleyParser;
using xgrammar::Grammar;
using xgrammar::ParserState;
using Expr = Grammar::Impl::GrammarExprType;

constexpr int32_t kInf = 1 << 29;
constexpr int32_t kMaxStates = 16384;  // parser states the tables name; a token reaching past them is never masked
constexpr int32_t kRoot = -2;          // the parent set of the root rule's states
constexpr int32_t kUnknown = -2;       // a successor past kMaxStates: taken to complete at once
constexpr uint16_t kOpen = 0xFFFF, kDead = 0xFFFE, kPast = 0xFFFD;  // a transition not yet probed, refused, unnamed

struct VecHash {
  size_t operator()(const std::vector<int32_t>& v) const {
    uint64_t h = 1469598103934665603ull;
    for (int32_t x : v) h = (h ^ static_cast<uint32_t>(x)) * 1099511628211ull;
    return static_cast<size_t>(h);
  }
};

using Names = std::unordered_map<std::vector<int32_t>, int32_t, VecHash>;

// The chart's parent sets loop (left recursion), or a replayed path was refused: no budget mask then.
struct Uncovered {};

int32_t Find(Names& names, std::vector<int32_t>&& key, bool add) {
  auto it = names.find(key);
  if (it != names.end()) return it->second;
  if (!add) return -1;
  int32_t id = static_cast<int32_t>(names.size());
  names.emplace(std::move(key), id);
  return id;
}

uint16_t Encode(int32_t y) { return y == -1 ? kDead : y == kUnknown ? kPast : static_cast<uint16_t>(y); }

int32_t Decode(uint16_t v) { return v == kDead ? -1 : v == kPast || v == kOpen ? kUnknown : v; }

}  // namespace

// The tokenizer's text tokens (no special or stop tokens) in byte order, walked as a trie.
struct Trie {
  xgrammar::TokenizerInfo info;
  std::vector<uint16_t> shared;  // bytes each token shares with the one before it
  size_t longest = 0;

  explicit Trie(xgrammar::TokenizerInfo i) : info(std::move(i)) {
    const auto& v = info.GetSortedDecodedVocab();
    shared.assign(v.size(), 0);
    for (size_t k = 0; k < v.size(); k++) {
      const std::string& s = v[k].second;
      longest = std::max(longest, s.size());
      if (k == 0) continue;
      const std::string& p = v[k - 1].second;
      size_t n = 0, m = std::min(s.size(), p.size());
      while (n < m && s[n] == p[n]) n++;
      shared[k] = static_cast<uint16_t>(std::min<size_t>(n, 0xFFFF));
    }
  }

  // The first token after `i` that does not start with token i's first `len` bytes.
  size_t Skip(size_t i, size_t len) const {
    const auto& v = info.GetSortedDecodedVocab();
    const std::string& p = v[i].second;
    size_t lo = i + 1, hi = v.size();
    while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (v[mid].second.compare(0, len, p, 0, len) == 0) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  // Each token from state c: `step(y, byte, token, depth)` moves (-1 refused, kUnknown past), `end(token, state)`.
  template <class Step, class End>
  void Walk(int32_t c, Step&& step, End&& end) const {
    const auto& v = info.GetSortedDecodedVocab();
    std::vector<int32_t> at(longest + 1);
    at[0] = c;
    size_t valid = 0;
    for (size_t i = 0; i < v.size();) {
      const std::string& s = v[i].second;
      size_t d = std::min<size_t>(valid, shared[i]);
      int32_t y = c;
      for (; d < s.size(); d++) {
        y = step(at[d], static_cast<uint8_t>(s[d]), i, d);
        if (y < 0) break;
        at[d + 1] = y;
      }
      if (d < s.size()) {  // every token sharing these d + 1 bytes stops here too
        size_t j = Skip(i, d + 1);
        if (y == kUnknown)
          for (size_t k = i; k < j; k++) end(k, kUnknown);
        valid = d;
        i = j;
        continue;
      }
      end(i, at[s.size()]);
      valid = s.size();
      i++;
    }
  }
};

class Probe;

// One grammar's states: named by their chart, their byte steps, token successors and fewest tokens to complete.
class Tables {
 public:
  Tables(Grammar g, std::shared_ptr<const Trie> trie) : grammar(std::move(g)), trie(std::move(trie)) {}

  Grammar grammar;
  std::shared_ptr<const Trie> trie;
  Names items, parents, states;
  std::vector<uint8_t> done, explored;          // per state: the reply is complete there; its tokens were walked
  std::vector<uint16_t> next;                   // per state and byte
  std::vector<std::pair<int32_t, uint8_t>> from;  // the state and byte each state was first reached from
  std::vector<std::vector<int32_t>> succ;       // an explored state's successors after one token
  std::vector<int32_t> need, worst;             // fewest tokens to complete; the most any one token leaves
  int32_t most = 0;
  bool ok = false;
  std::once_flag once;

  // The state named by `key` ({complete, items...}), -1 unknown, kUnknown past kMaxStates.
  int32_t State(std::vector<int32_t>&& key, bool add, int32_t prev = -1, uint8_t byte = 0) {
    auto it = states.find(key);
    if (it != states.end()) return it->second;
    if (!add) return -1;
    if (static_cast<int32_t>(states.size()) >= kMaxStates) return kUnknown;
    int32_t id = static_cast<int32_t>(states.size());
    done.push_back(key[0] != 0);
    states.emplace(std::move(key), id);
    explored.push_back(0);
    next.resize(next.size() + 256, kOpen);
    from.emplace_back(prev, byte);
    succ.emplace_back();
    return id;
  }

  void Explore();

 private:
  bool Covered() const;
  void Place(Probe& p, int32_t c);
  void Need();
};

// The Earley parser with its chart in reach: each row's parent sets are named once, and popped with the row.
class Probe : public EarleyParser {
 public:
  explicit Probe(const Grammar& g) : EarleyParser(g), memo_(1) {}

  bool Step(uint8_t b) {
    if (!Advance(b)) return false;
    memo_.emplace_back();
    path_.push_back(b);
    return true;
  }

  void Back(size_t n) {
    if (n == 0) return;
    PopLastStates(static_cast<int32_t>(n));
    memo_.resize(memo_.size() - n);
    path_.resize(path_.size() - n);
  }

  const std::vector<uint8_t>& Path() const { return path_; }

  // The latest row's key: whether the reply is complete, then its states' names; false when one is unknown.
  bool Key(Tables& t, bool add, std::vector<int32_t>& key) {
    key.assign(1, IsCompleted() ? 1 : 0);
    for (const auto& s : scanable_state_history_[scanable_state_history_.size() - 1]) {
      int32_t id = Item(t, add, s);
      if (id < 0) return false;
      key.push_back(id);
    }
    std::sort(key.begin() + 1, key.end());
    key.erase(std::unique(key.begin() + 1, key.end()), key.end());
    return true;
  }

 private:
  int32_t Item(Tables& t, bool add, const ParserState& s) {
    int32_t p = Parents(t, add, s.rule_start_pos, s.rule_id);
    if (p == -1) return -1;
    return Find(t.items, {s.rule_id, s.sequence_id, s.element_id, s.sub_element_id, s.repeat_count, s.partial_codepoint, p}, add);
  }

  // The states that wait at row `pos` for `rule`, as a completion of it reads them.
  int32_t Parents(Tables& t, bool add, int32_t pos, int32_t rule) {
    if (pos == ParserState::kNoPrevInputPos) return kRoot;
    auto& memo = memo_[pos];
    auto it = memo.find(rule);
    if (it != memo.end()) {
      if (it->second == -3) throw Uncovered{};
      return it->second;
    }
    memo[rule] = -3;
    std::vector<int32_t> key;
    for (const auto& [ref, st] : rule_id_to_completable_states_[pos]) {
      if (ref != rule) continue;
      int32_t id = Item(t, add, st);
      if (id < 0) return memo_[pos][rule] = -1;
      key.push_back(id);
    }
    std::sort(key.begin(), key.end());
    key.erase(std::unique(key.begin(), key.end()), key.end());
    return memo_[pos][rule] = Find(t.parents, std::move(key), add);
  }

  std::vector<std::unordered_map<int32_t, int32_t>> memo_;  // row: rule to its parent set (-1 unknown, -3 naming)
  std::vector<uint8_t> path_;
};

// Rules the byte-level parser does not follow as the matcher does (token edges, budgets, captures, lazy, temperature).
bool Tables::Covered() const {
  for (int32_t r = 0; r < grammar->NumRules(); r++) {
    const auto& rule = grammar->GetRule(r);
    if (rule.max_tokens >= 0 || rule.max_chars >= 0 || !rule.capture_name.empty() || rule.is_lazy ||
        rule.temperature.has_value() || grammar->GetSuffixStopInfo(r) != nullptr)
      return false;
  }
  for (int32_t e = 0; e < grammar->NumGrammarExprs(); e++) {
    Expr type = grammar->GetGrammarExpr(e).type;
    if (type == Expr::kToken || type == Expr::kExcludeToken || type == Expr::kTokenTagDispatch) return false;
  }
  return true;
}

// Put the probe at state c by the bytes it was first reached with.
void Tables::Place(Probe& p, int32_t c) {
  std::vector<uint8_t> want;
  for (int32_t x = c; from[x].first >= 0; x = from[x].first) want.push_back(from[x].second);
  std::reverse(want.begin(), want.end());
  const auto& have = p.Path();
  size_t keep = 0;
  while (keep < have.size() && keep < want.size() && have[keep] == want[keep]) keep++;
  p.Back(have.size() - keep);
  for (size_t k = keep; k < want.size(); k++)
    if (!p.Step(want[k])) throw Uncovered{};
}

// Breadth first from the root: each state's tokens walked once, its byte steps probed as the walk first needs them.
void Tables::Explore() {
  if (!Covered()) return;
  try {
    Probe probe(grammar);
    std::vector<int32_t> key;
    if (!probe.Key(*this, true, key)) return;
    std::deque<int32_t> queue{State(std::move(key), true)};
    std::vector<uint8_t> queued(1, 1);
    while (!queue.empty()) {
      int32_t c = queue.front();
      queue.pop_front();
      bool placed = false;
      size_t base = 0;
      const auto& vocab = trie->info.GetSortedDecodedVocab();
      auto step = [&](int32_t y, uint8_t b, size_t i, size_t d) -> int32_t {
        size_t at = static_cast<size_t>(y) * 256 + b;
        if (next[at] != kOpen) return Decode(next[at]);
        if (!placed) {
          Place(probe, c);
          placed = true;
          base = probe.Path().size();
        }
        const std::string& s = vocab[i].second;
        size_t have = probe.Path().size() - base, keep = 0;
        while (keep < have && keep < d && probe.Path()[base + keep] == static_cast<uint8_t>(s[keep])) keep++;
        probe.Back(have - keep);
        for (size_t k = keep; k < d; k++)
          if (!probe.Step(static_cast<uint8_t>(s[k]))) throw Uncovered{};
        int32_t out = -1;
        if (probe.Step(b)) {
          if (!probe.Key(*this, true, key)) throw Uncovered{};
          out = State(std::move(key), true, y, b);
        }
        next[at] = Encode(out);
        return out;
      };
      std::vector<int32_t> out;
      trie->Walk(c, step, [&](size_t, int32_t y) {
        out.push_back(y);
        if (y < 0) return;
        if (static_cast<size_t>(y) >= queued.size()) queued.resize(y + 1, 0);
        if (!queued[y]) {
          queued[y] = 1;
          queue.push_back(y);
        }
      });
      std::sort(out.begin(), out.end());
      out.erase(std::unique(out.begin(), out.end()), out.end());
      succ[c] = std::move(out);
      explored[c] = 1;
    }
    Need();
    ok = true;
  } catch (const Uncovered&) {
  }
}

// Fewest tokens to complete, backwards from complete states; an unexplored state counts as one token from it.
void Tables::Need() {
  size_t n = states.size();
  need.assign(n, kInf);
  std::vector<std::vector<int32_t>> back(n), level(2);
  for (size_t c = 0; c < n; c++) {
    if (done[c]) need[c] = 0;
    else if (!explored[c]) need[c] = 1;
    else
      for (int32_t y : succ[c]) {
        if (y == kUnknown) need[c] = 1;
        else back[y].push_back(static_cast<int32_t>(c));
      }
    if (need[c] <= 1) level[need[c]].push_back(static_cast<int32_t>(c));
  }
  for (size_t e = 0; e < level.size(); e++) {
    for (size_t k = 0; k < level[e].size(); k++) {
      int32_t y = level[e][k];
      if (need[y] != static_cast<int32_t>(e)) continue;
      for (int32_t c : back[y]) {
        if (done[c] || need[c] <= static_cast<int32_t>(e) + 1) continue;
        need[c] = static_cast<int32_t>(e) + 1;
        if (level.size() <= e + 1) level.emplace_back();
        level[e + 1].push_back(c);
      }
    }
  }
  worst.assign(n, 0);
  for (size_t c = 0; c < n; c++) {
    if (!explored[c]) continue;
    for (int32_t y : succ[c]) worst[c] = std::max(worst[c], y == kUnknown ? 0 : need[y]);
    most = std::max(most, worst[c]);
  }
}

bool Ready(Tables& t) {
  std::call_once(t.once, [&] { t.Explore(); });
  return t.ok;
}

std::shared_ptr<Tables> Cache::Get(const xgrammar::CompiledGrammar& g) {
  std::call_once(once_, [&] { trie_ = std::make_shared<const Trie>(info_); });
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = recent_.begin(); it != recent_.end(); ++it) {
    if (it->first.ImplPtr() != g.ImplPtr()) continue;
    recent_.splice(recent_.begin(), recent_, it);
    return it->second;
  }
  auto t = std::make_shared<Tables>(g.GetGrammar(), trie_);
  recent_.emplace_front(g, t);
  if (recent_.size() > 8) recent_.pop_back();
  return t;
}

struct Follower::State {
  std::shared_ptr<Tables> t;
  Probe probe;
  std::vector<int32_t> taken;    // the matcher's tokens (-1: a stop token)
  std::vector<int32_t> applied;  // the tokens the probe has read
  std::vector<int32_t> key;

  explicit State(std::shared_ptr<Tables> tables) : t(std::move(tables)), probe(t->grammar) {}

  // The probe at the matcher's place; false once it terminated or a token was refused.
  bool Sync() {
    const auto& vocab = t->trie->info.GetDecodedVocab();
    size_t k = 0;
    while (k < applied.size() && k < taken.size() && applied[k] == taken[k]) k++;
    size_t pop = 0;
    for (size_t j = k; j < applied.size(); j++) pop += vocab[applied[j]].size();
    probe.Back(pop);
    applied.resize(k);
    for (; k < taken.size(); k++) {
      if (taken[k] < 0) return false;
      const std::string& s = vocab[taken[k]];
      for (size_t b = 0; b < s.size(); b++) {
        if (probe.Step(static_cast<uint8_t>(s[b]))) continue;
        probe.Back(b);
        return false;
      }
      applied.push_back(taken[k]);
    }
    return true;
  }
};

Follower::Follower(std::shared_ptr<Tables> t) : s_(std::make_unique<State>(std::move(t))) {}

Follower::~Follower() = default;

void Follower::Accept(int32_t token) {
  const auto& stops = s_->t->trie->info.GetStopTokenIds();
  bool stop = std::find(stops.begin(), stops.end(), token) != stops.end();
  s_->taken.push_back(stop ? -1 : token);
}

void Follower::Rollback(int n) { s_->taken.resize(s_->taken.size() - std::min<size_t>(n, s_->taken.size())); }

int Follower::Mask(int32_t* words, int n, int left) {
  Tables& t = *s_->t;
  if (left < 1) return 0;
  if (!Ready(t)) return 1;
  if (left - 1 >= t.most) return 0;
  if (!s_->Sync()) return 1;
  if (!s_->probe.Key(t, false, s_->key)) return 0;
  int32_t c = t.State(std::move(s_->key), false);
  if (c < 0 || !t.explored[c] || left - 1 >= t.worst[c]) return 0;
  std::vector<int32_t> keep(words, words + n);
  auto set = [&](int32_t id) { return id / 32 < n && (keep[id / 32] >> (id % 32) & 1) != 0; };
  bool any = false;  // a stop token, or a text token left set: else the reply cannot complete and xgrammar's bits stay
  for (int32_t id : t.trie->info.GetStopTokenIds()) any = any || set(id);
  const auto& vocab = t.trie->info.GetSortedDecodedVocab();
  auto step = [&](int32_t y, uint8_t b, size_t, size_t) { return Decode(t.next[static_cast<size_t>(y) * 256 + b]); };
  t.trie->Walk(c, step, [&](size_t i, int32_t y) {
    int32_t id = vocab[i].first;
    if (!set(id)) return;
    if (y >= 0 && t.need[y] > left - 1) keep[id / 32] &= ~static_cast<int32_t>(1u << (id % 32));
    else any = true;
  });
  if (any) std::memcpy(words, keep.data(), sizeof(int32_t) * n);
  return 0;
}

}  // namespace tfb
