// glm_index_scores.cuh and glm_index_pools.cuh run on the host, every thread of a block at once with real barriers and
// warp exchanges: scores against a plain model of tf_glm_scores (mma as a double sum per k16 step), pools against a
// plain top-K. Build: clang++ -std=c++20 -I tools/glm_index_emulate tools/glm_index_emulate/emulate.cpp -o <out>
#define GLM_INDEX_HOST
#include <cstdio>
#include <functional>
#include <random>
#include <thread>
#include <vector>
#include "cuda_bf16.h"
#include "cuda_runtime.h"

namespace glm_index {
inline void cp16(void* d, const void* s, bool ok) { ok ? (void)std::memcpy(d, s, 16) : (void)std::memset(d, 0, 16); }
inline void cp_commit() {}
inline void cp_wait() {}
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
          kernel();
        });
      for (auto& t : ts) t.join();
      for (auto* w : warps) delete w;
    }
}

static int fails = 0;
static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float bf(__nv_bfloat16 v) { return __bfloat162float(v); }

struct Case { int64_t pos, rows, np; int loop, heads; bool paged; };

// tf_glm_scores' result for one (row, pool), the model of the mma: a chain of double sums per k16 step, then its rules
static float model_score(const std::vector<__nv_bfloat16>& q, const std::vector<__nv_bfloat16>& w, int64_t w_stride,
                         const __nv_bfloat16* key, int64_t r, int heads, float scale, float wscale) {
  float x[32] = {}, wv[32] = {};
  for (int h = 0; h < heads; h++) {
    float acc = 0;
    for (int kq = 0; kq < 8; kq++) {
      double sum = 0;
      for (int k = 0; k < 16; k++) sum += (double)bf(q[(r * heads + h) * 128 + kq * 16 + k]) * (double)bf(key[kq * 16 + k]);
      acc = (float)((double)acc + sum);
    }
    x[h] = std::fmax(acc * scale, 0.0f);
    wv[h] = bf(w[r * w_stride + h]) * wscale;
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

static void scores_case(std::mt19937_64& g, const Case& c) {
  const int D = 128, W = 40;  // head weights in a row of W columns
  const int64_t visible = (c.pos + c.rows) / 4;
  const int64_t nb = (std::min(c.np, std::max<int64_t>(512, visible)) + 63) / 64;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<__nv_bfloat16> q(c.rows * c.heads * D), w(c.rows * W);
  for (auto& x : q) x = __float2bfloat16_rn(nd(g));
  for (auto& x : w) x = __float2bfloat16_rn(nd(g));
  const int64_t pool_rows = (c.pos + c.rows) / 4 + 1;
  std::vector<__nv_bfloat16> keys(std::max<int64_t>(c.np, pool_rows + 200) * D);
  for (auto& x : keys) x = __float2bfloat16_rn(nd(g) * 2);
  // pages of 128 pools, shuffled, when paged: the table maps a logical page to its place
  const int64_t npages = (keys.size() / D + 127) / 128, page_elems = 128 * D;
  std::vector<int> pt(npages), order(npages);
  for (int i = 0; i < npages; i++) order[i] = i;
  std::shuffle(order.begin(), order.end(), g);
  std::vector<__nv_bfloat16> store(npages * page_elems);
  for (int64_t p = 0; p < (int64_t)(keys.size() / D); p++) {
    pt[p / 128] = order[p / 128];
    std::memcpy(&store[(int64_t)order[p / 128] * page_elems + (p % 128) * D], &keys[p * D], D * 2);
  }
  const __nv_bfloat16* pk = c.paged ? store.data() : keys.data();
  const float scale = 0.0883883461f, wscale = 0.176776692f;
  const int pos = (int)c.pos;
  std::vector<float> out(c.rows * c.np);
  std::vector<uint32_t> raw(out.size(), 0x7FC0DEAD);
  std::memcpy(out.data(), raw.data(), out.size() * 4);
  run_grid((c.rows + 7) / 8, (nb + c.loop - 1) / c.loop, 256, [&] {
    tf_glm_index_scores(q.data(), w.data(), W, pk, out.data(), &pos, c.rows, c.np, nb, scale, wscale, c.heads, c.loop,
                        c.paged ? pt.data() : nullptr, c.paged ? page_elems * 2 : 0);
  });
  int64_t bad = 0, unread = 0;
  for (int64_t r = 0; r < c.rows; r++)
    for (int64_t p = 0; p < c.np; p++) {
      const uint32_t have = bits(out[r * c.np + p]);
      if (p >= nb * 64) {
        unread += have != 0x7FC0DEAD;  // never written
        continue;
      }
      const int64_t npool = (c.pos + r + 1) / 4;
      const float want = p < npool ? model_score(q, w, W, &keys[p * D], r, c.heads, scale, wscale) : -INFINITY;
      bad += have != bits(want);
    }
  printf("scores pos %lld rows %lld np %lld loop %d heads %d %s: %lld differ, %lld untouched columns written\n",
         (long long)c.pos, (long long)c.rows, (long long)c.np, c.loop, c.heads, c.paged ? "paged" : "plain",
         (long long)bad, (long long)unread);
  fails += bad != 0 || unread != 0;
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

int main() {
  std::mt19937_64 g(0x101);
  for (const Case& c : {Case{0, 1, 1024, 2, 32, false}, Case{1000, 19, 1024, 4, 32, false}, Case{3000, 24, 1024, 16, 32, true},
                        Case{3900, 17, 1000, 3, 32, false}, Case{5000, 9, 2048, 5, 20, true},
                        Case{6000, 24, 2048, 8, 32, false}, Case{12000, 8, 4096, 3, 32, true}})
    scores_case(g, c);

  const int K = 512, rows = 2;
  int paths[3] = {0, 0, 0};
  for (int np : {1024, 4097, 12289, 32769, 70001})
    for (int mode = 0; mode < 3; mode++)
      for (Kind kind : {normal, ties, equal, zeros, skewed, decoy}) {
        // the first row's position: few visible pools (limit K), about half, or nearly all
        const int pos = mode == 0 ? 300 : mode == 1 ? 2 * np : 4 * np - rows - 5;
        std::vector<float> s((size_t)rows * np + 4);
        float* base = s.data() + (np % 4 ? 1 : 0);  // an unaligned row now and then: the scalar loads
        std::vector<int> limit(rows);
        for (int r = 0; r < rows; r++) {
          limit[r] = (int)std::min<int64_t>(np, std::max<int64_t>((pos + r + 1) / 4, K));
          fill(g, base + (size_t)r * np, np, limit[r], kind);
        }
        std::vector<int64_t> out((size_t)rows * K, -7);
        std::vector<int> path(rows, -1);
        run_grid(rows, 1, glm_index::T, [&] {
          tf_glm_index_pools(base, out.data(), np, &pos, K, tile_cap(np), tile_target(np), path.data());
        });
        for (int r = 0; r < rows; r++) {
          paths[path[r]]++;
          const auto want = reference(base + (size_t)r * np, limit[r], K);
          if (!std::equal(want.begin(), want.end(), out.begin() + (size_t)r * K)) {
            fails++;
            printf("pools np %d mode %d kind %d row %d (path %d) differs\n", np, mode, (int)kind, r, path[r]);
          }
        }
      }
  printf("pools paths: whole row %d, candidates %d, whole row after candidates %d\n", paths[0], paths[1], paths[2]);
  if (paths[1] == 0 || paths[2] == 0) { fails++; printf("a path was never taken\n"); }
  printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails != 0;
}
