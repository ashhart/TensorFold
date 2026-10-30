// Ling's group-limited route: keep_groups then top-k, one simdgroup a row. Matches zig/src/families/ling/route.zig.
#include <metal_stdlib>
using namespace metal;

constant constexpr int MAXE = 16;

struct Args {
  int rows, ne, topk, ng, keep;
  float scale;
};

inline float tf_sigmoid_precise(float x) {
  float e = metal::precise::exp(metal::abs(x));
  float y = 1.0f / (1.0f + e);
  return (x < 0) ? y : (1.0f - y);
}

// -inf on every expert outside the `keep` groups of the best top-2 sums (ties to the lower group).
inline void keep_groups(thread float* choice, int lane, int ne, int gs, int ng, int keep) {
  float gscore = -INFINITY;
  const bool whole = gs % 32 == 0 && ne % 32 == 0;
  if (whole) {
    const int per = gs / 32, ni = ne / 32;
    float a = -INFINITY, b = -INFINITY;
    int g = 0, left = per;
    for (int i = 0; i < MAXE; ++i) {
      if (i >= ni) break;
      const float c = choice[i];
      if (c > a) {
        b = a;
        a = c;
      } else if (c > b) {
        b = c;
      }
      if (--left == 0) {
        for (int o = 16; o > 0; o >>= 1) {
          const float oa = simd_shuffle_xor(a, o), ob = simd_shuffle_xor(b, o);
          b = fmax(fmin(a, oa), fmax(b, ob));
          a = fmax(a, oa);
        }
        if (lane == g) gscore = a + b;
        a = b = -INFINITY;
        ++g;
        left = per;
      }
    }
  }
  for (int g = 0; g < (whole ? 0 : ng); ++g) {
    float a = -INFINITY, b = -INFINITY;
    for (int i = 0; i < MAXE; ++i) {
      const int e = i * 32 + lane;
      if (e < ne && e / gs == g) {
        const float c = choice[i];
        if (c > a) {
          b = a;
          a = c;
        } else if (c > b) {
          b = c;
        }
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float oa = simd_shuffle_xor(a, o), ob = simd_shuffle_xor(b, o);
      b = fmax(fmin(a, oa), fmax(b, ob));
      a = fmax(a, oa);
    }
    if (lane == g) gscore = a + b;
  }
  bool live = lane < ng;
  uint kept = 0u;
  for (int k = 0; k < keep; ++k) {
    float mx = live ? gscore : -INFINITY;
    mx = simd_max(mx);
    int id = (live && gscore == mx) ? lane : 32;
    for (int o = 16; o > 0; o >>= 1) id = min(id, simd_shuffle_xor(id, o));
    if (id == 32) break;
    kept |= 1u << id;
    if (lane == id) live = false;
  }
  if (whole) {
    const int per = gs / 32, ni = ne / 32;
    int g = 0, left = per;
    for (int i = 0; i < MAXE; ++i) {
      if (i >= ni) break;
      if (!((kept >> g) & 1u)) choice[i] = -INFINITY;
      if (--left == 0) {
        ++g;
        left = per;
      }
    }
    return;
  }
  for (int i = 0; i < MAXE; ++i) {
    const int e = i * 32 + lane;
    if (e < ne && !((kept >> (e / gs)) & 1u)) choice[i] = -INFINITY;
  }
}

// glue.select: top_k of sigmoid + bias in the kept groups, renormalized, times scale. One simdgroup a row.
[[kernel]] void tf_ling_route(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                              constant Args& a [[buffer(2)]], device int* PICK [[buffer(3)]],
                              device float* WTS [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const int r = int(tg);
  if (r >= a.rows) return;
  const int ne = a.ne, topk = a.topk;
  float score[MAXE], choice[MAXE];
  for (int i = 0; i < MAXE; ++i) {
    const int e = i * 32 + int(lane);
    score[i] = e < ne ? tf_sigmoid_precise(LOGITS[r * ne + e]) : 0.0f;
    choice[i] = e < ne ? score[i] + BIAS[e] : -INFINITY;
  }
  if (a.ng > 1) keep_groups(choice, int(lane), ne, ne / a.ng, a.ng, a.keep);
  float total = 0.0f, mine = 0.0f;
  int mine_id = 0;
  for (int k = 0; k < topk; ++k) {
    float mx = choice[0];
    for (int i = 1; i < MAXE; ++i) mx = fmax(mx, choice[i]);
    mx = simd_max(mx);
    int id = 1 << 30;
    for (int i = MAXE - 1; i >= 0; --i)
      if (choice[i] == mx) id = i * 32 + int(lane);
    for (int o = 16; o > 0; o >>= 1) id = min(id, simd_shuffle_xor(id, o));
    float sk = 0.0f;
    for (int i = 0; i < MAXE; ++i)
      if (i * 32 + int(lane) == id) {
        sk = score[i];
        choice[i] = -INFINITY;
      }
    sk = simd_shuffle(sk, uint(id & 31));
    total = total + sk;
    if (int(lane) == k) {
      mine = sk;
      mine_id = id;
    }
  }
  if (int(lane) < topk) {
    PICK[r * topk + int(lane)] = mine_id;
    WTS[r * topk + int(lane)] = (mine / (total + 1e-20f)) * a.scale;
  }
}
