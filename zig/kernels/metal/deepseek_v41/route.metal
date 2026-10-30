// DeepSeek-V4.1's MoE route: sqrt(softplus(logit)) scores, the top-k of score plus bias. Variants via TF_DSV41_E/TOPK.
#include <metal_stdlib>
using namespace metal;

#ifndef TF_DSV41_PER
#define TF_DSV41_PER ((TF_DSV41_E + 31) / 32)
#endif

// The reference's softplus: the linear tail above 20 and the quadratic one below -10, as the CUDA router computes it.
inline float tf_dsv41_softplus(float z) {
    if (z != z) return z;
    const float e = metal::precise::exp(fmin(z, 20.0f));
    if (z > 20.0f) return z;
    if (z < -10.0f) return e - 0.5f * e * e;
    return metal::precise::log(1.0f + e);
}

// A candidate beats the incumbent when its key is larger, or the keys tie and its id is lower; NaN keys never win.
inline bool tf_dsv41_better(float as, int ai, float bs, int bi) {
    if (ai >= TF_DSV41_E) return false;
    if (bi >= TF_DSV41_E) return true;
    if (as != as) return false;
    if (bs != bs) return true;
    if (as > bs) return true;
    if (as < bs) return false;
    return ai < bi;
}

struct Dsv41RouteArgs {
    int rows;
    int topk;
    float scale;
    float eps;
};

// One simdgroup a row: the top-k of score plus bias (ties to the lower id), then each picked score over the sum.
[[kernel]] void tf_dsv41_route_select(const device float* LOGITS [[buffer(0)]], const device float* BIAS [[buffer(1)]],
                                      const device float* BIAS_VL [[buffer(2)]], const device uchar* IMAGE [[buffer(3)]],
                                      constant Dsv41RouteArgs& a [[buffer(4)]], device int* PICK [[buffer(5)]],
                                      device float* WTS [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]]) {
    const int r = int(row);
    if (r >= a.rows || a.topk > TF_DSV41_TOPK) return;
    const device float* bias = IMAGE[r] != 0 ? BIAS_VL : BIAS;
    float s[TF_DSV41_PER], ch[TF_DSV41_PER];
    int eid[TF_DSV41_PER];
    for (int j = 0; j < TF_DSV41_PER; j++) {
        const int e = j * 32 + int(lane);
        if (e < TF_DSV41_E) {
            s[j] = metal::precise::sqrt(tf_dsv41_softplus(LOGITS[size_t(r) * TF_DSV41_E + e]));
            ch[j] = s[j] + bias[e];
            eid[j] = e;
        } else {
            s[j] = 0.0f;
            ch[j] = -INFINITY;
            eid[j] = -1;
        }
    }
    float w[TF_DSV41_TOPK];
    for (int k = 0; k < a.topk; k++) {
        // The lane's own best: its ids ascend with j, so a strict > keeps the lower id on a tie.
        float best = -INFINITY;
        int bid = TF_DSV41_E; // nothing yet
        for (int j = 0; j < TF_DSV41_PER; j++) {
            if (eid[j] >= 0 && (ch[j] > best || (ch[j] == best && eid[j] < bid))) { best = ch[j]; bid = eid[j]; }
        }
        // The butterfly: key descending then id ascending is a total order, so every lane ends with the same pick.
        for (int off = 16; off > 0; off /= 2) {
            const float ob = simd_shuffle_xor(best, off);
            const int oi = simd_shuffle_xor(bid, off);
            if (tf_dsv41_better(ob, oi, best, bid)) { best = ob; bid = oi; }
        }
        // The winner's own unbiased score: its lane holds it, and the other lanes take it from there.
        float mine = 0.0f;
        if (bid < TF_DSV41_E) {
            for (int j = 0; j < TF_DSV41_PER; j++) {
                if (eid[j] == bid) { mine = s[j]; eid[j] = -1; ch[j] = -INFINITY; } // the slot is spent
            }
        }
        w[k] = bid < TF_DSV41_E ? simd_shuffle(mine, uint(bid & 31)) : 0.0f;
        if (lane == 0) PICK[size_t(r) * a.topk + k] = bid < TF_DSV41_E ? bid : -1;
    }
    if (lane == 0) {
        float total = w[0];
        for (int k = 1; k < a.topk; k++) total = total + w[k];
        for (int k = 0; k < a.topk; k++) WTS[size_t(r) * a.topk + k] = (w[k] / (total + a.eps)) * a.scale;
    }
}
