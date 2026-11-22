// glm_index_*.cuh on the host, a block's threads at once, against plain models (zig build test-glm-index-emulate)
#define GLM_INDEX_HOST
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#include "cuda_bf16.h"
#include "cuda_runtime.h"

namespace glm_index {
inline void cp16(void* d, const void* s, bool ok) { ok ? (void)std::memcpy(d, s, 16) : (void)std::memset(d, 0, 16); }
inline void cp_commit() {}
template <int N> inline void cp_wait() {}
inline void ldm4(uint32_t* r, const void* p) {  // matrix i, row j of the 8 x 8 b16 tiles at the address of lane 8 i + j
  const int t = (int)threadIdx.x, l = t & 31, base = t - l;
  emu::pbuf[t] = p;
  warp_sync();
  for (int i = 0; i < 4; ++i) std::memcpy(&r[i], (const char*)emu::pbuf[base + 8 * i + l / 4] + 4 * (l % 4), 4);
  warp_sync();
}
inline float bf_at(uint32_t w, int half) { return __bfloat162float({(uint16_t)(half ? w >> 16 : w & 0xFFFF)}); }
inline void mma16816(float* c, const uint32_t* a, const uint32_t* b) {  // m16n8k16, c += a b as one double sum
  static uint32_t xa[1024][6];
  const int t = (int)threadIdx.x, l = t & 31, base = t - l, g = l / 4, q = l % 4;
  for (int i = 0; i < 4; ++i) xa[t][i] = a[i];
  xa[t][4] = b[0];
  xa[t][5] = b[1];
  warp_sync();
  float out[4];
  for (int j = 0; j < 4; ++j) {
    const int row = g + 8 * (j / 2), col = 2 * q + j % 2;
    double sum = 0;
    for (int k = 0; k < 16; ++k) {
      const uint32_t aw = xa[base + (row % 8) * 4 + (k % 8) / 2][(row < 8 ? 0 : 1) + (k < 8 ? 0 : 2)];
      const uint32_t bw = xa[base + col * 4 + (k % 8) / 2][4 + (k < 8 ? 0 : 1)];
      sum += (double)bf_at(aw, k % 2) * (double)bf_at(bw, k % 2);
    }
    out[j] = (float)((double)c[j] + sum);
  }
  warp_sync();
  for (int j = 0; j < 4; ++j) c[j] = out[j];
}
}  // namespace glm_index

#include "../../zig/kernels/cuda/glm_index_pools.cuh"
#include "../../zig/kernels/cuda/glm_index_scores.cuh"
#include "../../zig/kernels/cuda/glm_index_decode.cuh"
// hc_post uses bf/rbf and explicit rounding intrinsics, not these unused host-incompatible PTX helpers.
#define asm(...)
#include "../../zig/kernels/cuda/glm_glue_rows.cuh"
#undef asm
// every block of the grid in turn, its threads at once
static void run_grid(unsigned gx, unsigned gy, int threads, const std::function<void()>& kernel) {
  for (unsigned by = 0; by < gy; by++)
    for (unsigned bx = 0; bx < gx; bx++) {
      std::barrier<> block(threads);
      std::vector<std::barrier<>*> warps;
      for (int w = 0; w < threads / 32; w++) warps.push_back(new std::barrier<>(32));
      emu::block_bar = &block;
      for (int w = 0; w < threads / 32; w++) emu::warp_bar[w] = warps[w];
      std::vector<std::thread> ts;
      for (int t = 0; t < threads; t++)
        ts.emplace_back([=, &kernel] {
          blockIdx = {bx, by, 0};
          threadIdx = {(unsigned)t, 0, 0};
          blockDim = {(unsigned)threads, 1, 1};
          kernel();
        });
      for (auto& t : ts) t.join();
      for (auto* w : warps) delete w;
    }
}

static int fails = 0;
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
// sparse.select_tokens' row model: expanded pool tokens, the causal tail and the populated count.
static void token_rows(const int64_t* pools, int pos, int K, int* tokens, int* counts, int rows, const int* pt,
                       int64_t page_rows, int pool) {
  for (int r = 0; r < rows; r++) {
    const int64_t q = (int64_t)pos + r, npool = (q + 1) / pool;
    const int width = pool * K + pool - 1;
    auto at = [&](int64_t t) { return (int)(pt ? pt[t / 512] * page_rows + t % 512 : t); };
    for (int i = 0; i < pool * K; i++) tokens[r * width + i] = at(pools[(int64_t)r * K + i / pool] * pool + i % pool);
    for (int i = 0; i < pool - 1; i++) tokens[r * width + pool * K + i] = npool * pool + i <= q ? at(npool * pool + i) : -1;
    int tail = 0;
    for (int i = 0; i < pool - 1; i++) tail += npool * pool + i <= q;
    counts[r] = npool > K ? pool * K + tail : 0;
  }
}

struct Case { int64_t pos, rows, np; int loop, heads; bool paged; int pool = 4, rope = 0; };

// tf_glm_scores' result for one (row, pool), the model of the mma: a chain of double sums per k16 step, then its rules
static float model_score(const std::vector<__nv_bfloat16>& q, const std::vector<__nv_bfloat16>& w, int64_t w_stride,
                         const __nv_bfloat16* key, int64_t r, int heads, float scale, float wscale) {
  float x[32] = {}, wv[32] = {};
  for (int h = 0; h < heads; h++) {
    float acc = 0;
    for (int kq = 0; kq < 8; kq++) {
      double sum = 0;
      for (int k = 0; k < 16; k++) sum += (double)::bf(q[(r * heads + h) * 128 + kq * 16 + k]) * (double)::bf(key[kq * 16 + k]);
      acc = (float)((double)acc + sum);
    }
    x[h] = std::fmax(acc * scale, 0.0f);
    wv[h] = ::bf(w[r * w_stride + h]) * wscale;
  }
  float s[8];
  for (int g = 0; g < 8; g++) {
    float v = wv[g + 8] * x[g + 8];
    v = std::fma(wv[g], x[g], v);
    v = std::fma(wv[g + 16], x[g + 16], v);
    s[g] = std::fma(wv[g + 24], x[g + 24], v);
  }
  float t[8], u[8];
  for (int g = 0; g < 8; g++) t[g] = s[g] + s[g ^ 4];
  for (int g = 0; g < 8; g++) u[g] = t[g] + t[g ^ 2];
  return u[0] + u[1];
}

static void scores_case(std::mt19937_64& g, const Case& c, bool check_model) {
  const int D = 128, W = 40;  // head weights in a row of W columns
  const int64_t visible = (c.pos + c.rows) / c.pool, pp = 512 / c.pool;  // pp: a page's pools
  const int64_t nb = (std::min(c.np, std::max<int64_t>(512, visible)) + 63) / 64;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<__nv_bfloat16> q(c.rows * c.heads * D), w(c.rows * W);
  for (auto& x : q) x = __float2bfloat16_rn(nd(g));
  for (auto& x : w) x = __float2bfloat16_rn(nd(g));
  const int64_t pool_rows = (c.pos + c.rows) / c.pool + 1;
  std::vector<__nv_bfloat16> keys(std::max<int64_t>(c.np, pool_rows + 200) * D);
  for (auto& x : keys) x = __float2bfloat16_rn(nd(g) * 2);
  // pages of pp pools, shuffled, when paged: the table maps a logical page to its place
  const int64_t npages = (keys.size() / D + pp - 1) / pp, page_elems = pp * D;
  std::vector<int> pt(npages), order(npages);
  for (int i = 0; i < npages; i++) order[i] = i;
  std::shuffle(order.begin(), order.end(), g);
  std::vector<__nv_bfloat16> store(npages * page_elems);
  for (int64_t p = 0; p < (int64_t)(keys.size() / D); p++) {
    pt[p / pp] = order[p / pp];
    std::memcpy(&store[(int64_t)order[p / pp] * page_elems + (p % pp) * D], &keys[p * D], D * 2);
  }
  const __nv_bfloat16* pk = c.paged ? store.data() : keys.data();
  const float scale = 0.0883883461f, wscale = 0.176776692f;
  const int pos = (int)c.pos;
  // rope: the model turns q's pairs below c.rope itself (row r at pos + r), the kernel its fragments
  std::vector<float> inv(64);
  for (int i = 0; i < 64; i++) inv[i] = (float)std::pow(8.0e6, -2.0 * i / 64);
  std::vector<__nv_bfloat16> qm = q;
  for (int64_t r = 0; r < c.rows; r++)
    for (int h = 0; h < c.heads; h++)
      for (int i = 0; i < c.rope / 2; i++) {
        __nv_bfloat16* x = &qm[(r * c.heads + h) * D + 2 * i];
        const float x0 = ::bf(x[0]), x1 = ::bf(x[1]);
        float sn, cs;
        sincosf((float)(c.pos + r) * inv[i], &sn, &cs);
        x[0] = __float2bfloat16_rn(std::fma(x0, cs, -(x1 * sn)));
        x[1] = __float2bfloat16_rn(std::fma(x1, cs, x0 * sn));
      }
  std::vector<float> out(c.rows * c.np);
  std::vector<uint32_t> raw(out.size(), 0x7FC0DEAD);
  std::memcpy(out.data(), raw.data(), out.size() * 4);
  run_grid((c.rows + 7) / 8, (nb + c.loop - 1) / c.loop, 256, [&] {
    tf_glm_index_scores(q.data(), w.data(), W, pk, out.data(), &pos, c.rows, c.np, nb, scale, wscale, c.heads, c.loop,
                        c.paged ? pt.data() : nullptr, c.paged ? page_elems * 2 : 0, c.pool, c.rope, inv.data());
  });
  if (check_model) {
    int64_t bad = 0, unread = 0;
    for (int64_t r = 0; r < c.rows; r++)
      for (int64_t p = 0; p < c.np; p++) {
        const uint32_t have = bits(out[r * c.np + p]);
        if (p >= nb * 64) {
          unread += have != 0x7FC0DEAD;  // never written
          continue;
        }
        const int64_t npool = (c.pos + r + 1) / c.pool;
        const float want = p < npool ? model_score(qm, w, W, &keys[p * D], r, c.heads, scale, wscale) : -INFINITY;
        bad += have != bits(want);
      }
    printf("scores pos %lld rows %lld np %lld loop %d heads %d pool %d rope %d %s: %lld differ, %lld untouched columns written\n",
           (long long)c.pos, (long long)c.rows, (long long)c.np, c.loop, c.heads, c.pool, c.rope, c.paged ? "paged" : "plain",
           (long long)bad, (long long)unread);
    fails += bad != 0 || unread != 0;
  }
  // TP=2 whole rows versus rank blocks: score, pick and token bytes must survive rank-order concatenation.
  std::vector<float> part_scores(out.size());
  std::memcpy(part_scores.data(), raw.data(), raw.size() * 4);
  const int K = std::min<int64_t>(c.np, c.pool == 1 ? 2048 : 512);
  const int cap = c.np <= 12288 ? 1024 : c.np <= 32768 ? 2048 : 4096;
  const int target = c.np <= 12288 ? 724 : c.np <= 32768 ? 1024 : 2048;
  const int width = c.pool * K + c.pool - 1;
  std::vector<int64_t> pools((size_t)c.rows * K), part_pools(pools.size());
  std::vector<int> tokens((size_t)c.rows * width), part_tokens(tokens.size()), counts(c.rows), part_counts(c.rows);
  run_grid(c.rows, 1, glm_index::T, [&] { tf_glm_index_pools(out.data(), pools.data(), c.np, &pos, K, cap, target, nullptr, c.pool); });
  token_rows(pools.data(), pos, K, tokens.data(), counts.data(), c.rows, c.paged ? pt.data() : nullptr, page_elems, c.pool);
  const int64_t per = (c.rows + 1) / 2;
  for (int rank = 0; rank < 2; rank++) {
    const int64_t lo = std::min<int64_t>(c.rows, rank * per), hi = std::min<int64_t>(c.rows, lo + per), rows = hi - lo;
    if (rows == 0) continue;
    const int at = (int)(c.pos + lo);
    const int64_t part_nb = (std::min(c.np, std::max<int64_t>(512, (c.pos + hi) / c.pool)) + 63) / 64;
    run_grid((rows + 7) / 8, (part_nb + c.loop - 1) / c.loop, 256, [&] {
      tf_glm_index_scores(q.data() + lo * c.heads * D, w.data() + lo * W, W, pk, part_scores.data() + lo * c.np, &at, rows,
                          c.np, part_nb, scale, wscale, c.heads, c.loop, c.paged ? pt.data() : nullptr,
                          c.paged ? page_elems * 2 : 0, c.pool, c.rope, inv.data());
    });
    run_grid(rows, 1, glm_index::T, [&] {
      tf_glm_index_pools(part_scores.data() + lo * c.np, part_pools.data() + lo * K, c.np, &at, K, cap, target, nullptr, c.pool);
    });
    token_rows(part_pools.data() + lo * K, at, K, part_tokens.data() + lo * width, part_counts.data() + lo, rows,
               c.paged ? pt.data() : nullptr, page_elems, c.pool);
  }
  int score_diff = 0; // unused score tails are intentionally not written by a smaller rank block
  for (int64_t r = 0; r < c.rows; r++) {
    const int64_t used = std::min<int64_t>(c.np, std::max<int64_t>((c.pos + r + 1) / c.pool, K));
    score_diff += std::memcmp(out.data() + r * c.np, part_scores.data() + r * c.np, used * 4) != 0;
  }
  const int pool_diff = std::memcmp(pools.data(), part_pools.data(), pools.size() * 8) != 0;
  const int token_diff = std::memcmp(tokens.data(), part_tokens.data(), tokens.size() * 4) != 0 ||
                         std::memcmp(counts.data(), part_counts.data(), counts.size() * 4) != 0;
  printf("TP2 local/split pos %lld rows %lld: scores %d pools %d tokens %d\n", (long long)c.pos, (long long)c.rows,
         score_diff, pool_diff, token_diff);
  fails += score_diff + pool_diff + token_diff;
}

// REV25 E8: a copied gather room versus direct own partials, including x/xout aliasing and a poisoned own slot.
static void hc_post_own_case(std::mt19937_64& g) {
  constexpr int R = 3, D = 4096, S = 4, WORLD = 2, N = R * D;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<__nv_bfloat16> x(R * S * D);
  std::vector<float> post(R * S), comb(R * S * S), own[WORLD];
  for (auto& v : x) v = __float2bfloat16_rn(nd(g));
  for (auto& v : post) v = nd(g);
  for (auto& v : comb) v = nd(g);
  for (auto& p : own) {
    p.resize(N);
    for (auto& v : p) v = nd(g);
  }
  const float edge[] = {0.0f, -0.0f, 1.0f, -1.0f, 16777216.0f, 0.5f, std::numeric_limits<float>::denorm_min(),
                        std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN(), 65504.0f, -65504.0f};
  for (int me = 0; me < WORLD; me++) {
    for (int i = 0; i < (int)(sizeof(edge) / sizeof(edge[0])); i++) own[me][i] = edge[(i + me) % (sizeof(edge) / sizeof(edge[0]))];
    std::vector<float> room(WORLD * N), peers(WORLD * N, std::numeric_limits<float>::quiet_NaN());
    for (int rank = 0; rank < WORLD; rank++) {
      std::memcpy(room.data() + rank * N, own[rank].data(), N * 4);
      if (rank != me) std::memcpy(peers.data() + rank * N, own[rank].data(), N * 4);
    }
    std::vector<__nv_bfloat16> copied(x.size()), direct(x.size()), alias_a = x, alias_b = x;
    run_grid(R, D / 1024, 128, [&] { tf_glm_hc_post(x.data(), copied.data(), room.data(), post.data(), comb.data(), N, D, WORLD); });
    run_grid(R, D / 1024, 128, [&] {
      tf_glm_hc_post_own(x.data(), direct.data(), peers.data(), post.data(), comb.data(), N, D, WORLD, own[me].data(), me);
    });
    run_grid(R, D / 1024, 128, [&] { tf_glm_hc_post(alias_a.data(), alias_a.data(), room.data(), post.data(), comb.data(), N, D, WORLD); });
    run_grid(R, D / 1024, 128, [&] {
      tf_glm_hc_post_own(alias_b.data(), alias_b.data(), peers.data(), post.data(), comb.data(), N, D, WORLD, own[me].data(), me);
    });
    const int plain_diff = std::memcmp(copied.data(), direct.data(), copied.size() * 2) != 0;
    const int alias_diff = std::memcmp(alias_a.data(), alias_b.data(), alias_a.size() * 2) != 0;
    printf("REV25 E8 hc_post rank %d: copied/direct %d, aliased %d (own gather slot poisoned)\n", me, plain_diff, alias_diff);
    fails += plain_diff + alias_diff;
  }
}

static int tile_cap(int np) { return np <= 12288 ? 1024 : np <= 32768 ? 2048 : 4096; }
static int tile_target(int np) { return np <= 12288 ? 724 : np <= 32768 ? 1024 : 2048; }
static uint32_t key(float s) { return glm_index::order_key(s); }

static std::vector<int64_t> reference(const float* row, int n, int k) {
  std::vector<int> o(n);
  for (int i = 0; i < n; i++) o[i] = i;
  std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return key(row[a]) > key(row[b]); });
  std::vector<int64_t> best(o.begin(), o.begin() + k);
  std::sort(best.begin(), best.end());
  return best;
}

enum Kind { normal, ties, equal, zeros, skewed, decoy };

static void fill(std::mt19937_64& g, float* row, int np, int limit, Kind kind) {
  std::normal_distribution<float> nd(0.0f, 3.0f);
  const int vis = limit;  // the row's visible pools; the rest read as stale 1e9, which the kernel must never touch
  for (int i = 0; i < np; i++) {
    float v = nd(g);
    if (kind == ties) v = (float)(int)(g() % 9) * 0.5f;
    if (kind == equal) v = 1.25f;
    if (kind == zeros) v = g() % 3 == 0 ? 0.0f : g() % 2 ? -0.0f : (float)(g() % 1000) * 1e-3f - 0.6f;
    if (kind == skewed) v = i % 997 == 0 ? 100.0f + (float)(g() % 5) : -INFINITY;
    if (kind == decoy) v = (float)(g() % 1000) * 1e-3f;
    row[i] = i < vis ? v : 1e9f;
  }
  if (kind == decoy && limit > 4096)  // the floor's sample runs high: fewer than K pools at or above it
    for (int s = 0; s < glm_index::NS * 8; s++) {
      const int64_t j = (int64_t)(s / 8) * limit / glm_index::NS / 8 * 8 + s % 8;
      if (j < limit) row[j] = 10.0f + (float)(g() % 100000) * 1e-4f;
    }
}

int main(int argc, char** argv) {
  const bool rev25 = argc == 2 && std::strcmp(argv[1], "--rev25") == 0;
  if (argc != (rev25 ? 2 : 1)) return 2;
  std::mt19937_64 g(0x101);
  if (rev25) {
    for (const Case& c : {Case{3000, 24, 1024, 16, 32, true}, Case{12000, 8, 4096, 3, 32, false},
                          Case{32767, 8, 8192, 4, 32, true}, Case{3000, 9, 4096, 4, 32, true, 1, 64},
                          Case{32767, 8, 8192, 4, 32, true, 1, 64}})
      scores_case(g, c, false);
  } else {
    for (const Case& c : {Case{0, 1, 1024, 2, 32, false}, Case{1000, 19, 1024, 4, 32, false}, Case{3000, 24, 1024, 16, 32, true},
                          Case{3900, 17, 1000, 3, 32, false}, Case{5000, 9, 2048, 5, 20, true},
                          Case{6000, 24, 2048, 8, 32, false}, Case{12000, 8, 4096, 3, 32, true},
                          // full GLM-5.3's: every key scored (pools of 1, 512 a page), queries roped on 64 dims
                          Case{3000, 9, 4096, 4, 32, true, 1, 64}, Case{2100, 17, 2560, 3, 32, false, 1, 64},
                          Case{5000, 8, 8192, 16, 32, true, 1, 0}})
      scores_case(g, c, true);
  }
  hc_post_own_case(g);
  if (rev25) {
    printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
  }

  const int rows = 2;
  int paths[3] = {0, 0, 0};
  // GLM-5.3-Flash's 512 of pools of 4 at every bucket, then full GLM-5.3's 2,048 of single keys
  struct Pick { int np, pool, K; };
  std::vector<Pick> picks;
  for (int np : {1024, 4097, 12289, 32769, 70001}) picks.push_back({np, 4, 512});
  for (int np : {4097, 12289}) picks.push_back({np, 1, 2048});
  for (const Pick& pk : picks)
    for (int mode = 0; mode < 3; mode++)
      for (Kind kind : {normal, ties, equal, zeros, skewed, decoy}) {
        const int np = pk.np, pool = pk.pool, K = pk.K;
        // the first row's position: few visible pools (limit K), about half, or nearly all
        const int pos = mode == 0 ? 300 : mode == 1 ? pool * np / 2 : pool * np - rows - 5;
        std::vector<float> s((size_t)rows * np + 4);
        float* base = s.data() + (np % 4 ? 1 : 0);  // an unaligned row now and then: the scalar loads
        std::vector<int> limit(rows);
        for (int r = 0; r < rows; r++) {
          limit[r] = (int)std::min<int64_t>(np, std::max<int64_t>((pos + r + 1) / pool, K));
          fill(g, base + (size_t)r * np, np, limit[r], kind);
        }
        std::vector<int64_t> out((size_t)rows * K, -7);
        std::vector<int> path(rows, -1);
        run_grid(rows, 1, glm_index::T, [&] {
          tf_glm_index_pools(base, out.data(), np, &pos, K, tile_cap(np), tile_target(np), path.data(), pool);
        });
        for (int r = 0; r < rows; r++) {
          paths[path[r]]++;
          const auto want = reference(base + (size_t)r * np, limit[r], K);
          if (!std::equal(want.begin(), want.end(), out.begin() + (size_t)r * K)) {
            fails++;
            printf("pools np %d pool %d mode %d kind %d row %d (path %d) differs\n", np, pool, mode, (int)kind, r, path[r]);
          }
        }
      }
  printf("pools paths: whole row %d, candidates %d, whole row after candidates %d\n", paths[0], paths[1], paths[2]);
  if (paths[1] == 0 || paths[2] == 0) { fails++; printf("a path was never taken\n"); }

  // glm_index_decode.cuh: a decode window's rows marked on many CTAs, then merged, against the same plain top-K
  int dpaths[4] = {0, 0, 0, 0};
  const int drows = 3;
  std::vector<Pick> dpicks;
  for (int np : {4097, 12289, 70001}) dpicks.push_back({np, 4, 512});
  dpicks.push_back({12289, 1, 2048});
  for (const Pick& pk : dpicks)
    for (int mode = 0; mode < 3; mode++)
      for (Kind kind : {normal, ties, equal, zeros, skewed, decoy}) {
        const int np = pk.np, pool = pk.pool, K = pk.K;
        const int pos = mode == 0 ? 300 : mode == 1 ? pool * np / 2 : pool * np - drows - 5;
        std::vector<float> s((size_t)drows * np + 4);
        float* base = s.data() + (np % 4 ? 1 : 0);
        std::vector<int> limit(drows);
        for (int r = 0; r < drows; r++) {
          limit[r] = (int)std::min<int64_t>(np, std::max<int64_t>((pos + r + 1) / pool, K));
          fill(g, base + (size_t)r * np, np, limit[r], kind);
        }
        const int64_t words = (np + 31) / 32;
        std::vector<uint32_t> bitw((size_t)drows * 2 * words, 0xA5A5A5A5u);  // words a scan does not write stay unread
        const int last = (int)std::min<int64_t>(np, std::max<int64_t>((pos + drows) / pool, K));
        const unsigned ctas = last <= glm_dsel::CAP ? 0 : (unsigned)((last + glm_dsel::SPAN - 1) / glm_dsel::SPAN);
        if (ctas > 0)
          run_grid(ctas, drows, glm_dsel::T1, [&] { tf_glm_dsel_scan(base, np, &pos, K, 2048, bitw.data(), words, pool); });
        std::vector<int64_t> out((size_t)drows * K, -7);
        std::vector<int> path(drows, -1);
        run_grid(drows, 1, glm_dsel::T2, [&] { tf_glm_dsel_merge(base, out.data(), np, &pos, K, bitw.data(), words, path.data(), pool); });
        for (int r = 0; r < drows; r++) {
          if (path[r] < 0 || path[r] > 3) {
            fails++;
            printf("dsel np %d mode %d kind %d row %d: path %d\n", np, mode, (int)kind, r, path[r]);
            continue;
          }
          dpaths[path[r]]++;
          const auto want = reference(base + (size_t)r * np, limit[r], K);
          if (!std::equal(want.begin(), want.end(), out.begin() + (size_t)r * K)) {
            fails++;
            printf("dsel np %d pool %d mode %d kind %d row %d (path %d) differs\n", np, pool, mode, (int)kind, r, path[r]);
          }
        }
      }
  printf("dsel paths: whole row %d, candidates %d, whole row after candidates %d, ties at the floor %d\n", dpaths[0],
         dpaths[1], dpaths[2], dpaths[3]);
  if (dpaths[0] == 0 || dpaths[1] == 0 || dpaths[2] == 0 || dpaths[3] == 0) { fails++; printf("a dsel path was never taken\n"); }
  printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails != 0;
}
