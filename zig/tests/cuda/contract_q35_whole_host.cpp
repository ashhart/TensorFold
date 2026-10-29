// Exercise the production tile loaders with poisoned page metadata and every head's physical row offset.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#define __device__
using __nv_bfloat16 = uint16_t;
struct uint4 { uint32_t x, y, z, w; };
static uint4 make_uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return {a, b, c, d}; }
static struct { int x; } threadIdx;
namespace qmm_frag {
static void cp16z(uint4* dst, const void* src, bool in) {
    if (in) std::memcpy(dst, src, 16);
    else *dst = {};
}
static void commit() {}
template <int> static void wait() {}
}
#include "qwen35_page_run.h"
namespace tf_q35_tc {
constexpr int HD = 256, TK = 32, CH = HD / 8;
struct Seg { uintptr_t kv, pt; uint64_t ps; int cap, ptok; };
struct TileStore { uint4 *kt, *vt; float *ksc, *vsc; };
static int swz(int u, int chunk) { return u * CH + (chunk ^ (u & 7)); }
static void widen16(uint4, uint4*, uint4*) {}
#include "qwen35_attn_tiles.cuh"
#include "qwen35_attn_whole.cuh"

template <int THREADS> static bool compare() {
    constexpr int CAP = 512, KVH = 4, LAYERS = 2, TILE = TK * CH;
    const size_t bytes = LAYERS * 2 * KVH * CAP * (HD + 4) * 2;
    auto* data = static_cast<unsigned char*>(std::malloc(bytes));
    if (!data) return false;
    for (size_t i = 0; i < bytes; i++) data[i] = static_cast<unsigned char>((i * 31 + i / 17) & 255);
    uint4 left[4 * TILE], right[4 * TILE];
    TileStore ls{left, nullptr, nullptr, nullptr}, rs{right, nullptr, nullptr, nullptr};
    // A whole instance must never inspect pt/ptok, even when a nonzero invalid pointer would fault.
    Seg poison{reinterpret_cast<uintptr_t>(data), 1, UINT64_MAX, CAP, 0};
    Seg plain{reinterpret_cast<uintptr_t>(data), 0, 0, CAP, 48};
    bool ok = true;
    for (int layer = 0; layer < LAYERS; layer++) for (int h = 0; h < KVH; h++) {
        for (int first : {0, 1, 31, 127, 128, 479}) for (int tail : {1, 17, 32}) {
            WholeBf16Tiles<THREADS> a(ls, poison, layer, KVH);
            Bf16Tiles<THREADS> b(rs, plain, layer, KVH, h);
            std::memset(left, 0xa5, sizeof(left));
            std::memset(right, 0xa5, sizeof(right));
            for (int tid = 0; tid < THREADS; tid++) {
                threadIdx.x = tid;
                a.fetch(first, [&](int t) { return t < first + tail ? h * CAP + t : -1; });
                b.fetch(first, [&](int t) { return t < first + tail ? t : -1; });
            }
            ok &= std::memcmp(left, right, sizeof(left)) == 0;
            for (int tid = 0; tid < THREADS; tid++) {
                threadIdx.x = tid;
                WholeFp8Tiles<THREADS> x(ls, poison, layer, KVH);
                Fp8Tiles<THREADS> y(rs, plain, layer, KVH, h);
                x.fetch(first, [&](int t) { return t < first + tail ? h * CAP + t : -1; });
                y.fetch(first, [&](int t) { return t < first + tail ? t : -1; });
                ok &= std::memcmp(x.kr, y.kr, sizeof(x.kr)) == 0;
                ok &= std::memcmp(x.vr, y.vr, sizeof(x.vr)) == 0;
                ok &= std::memcmp(&x.sr, &y.sr, sizeof(x.sr)) == 0;
                const float v = x.key_scale(h * CAP + first), w = y.key_scale(first);
                ok &= std::memcmp(&v, &w, sizeof(v)) == 0;
                // Dropping the head's physical offset is the regression control.
                if (h > 0 && tid == 0) {
                    WholeFp8Tiles<THREADS> bad(ls, poison, layer, KVH);
                    bad.fetch(first, [](int t) { return t; });
                    ok &= std::memcmp(bad.kr, x.kr, sizeof(x.kr)) != 0;
                }
            }
        }
    }
    std::free(data);
    return ok;
}
}
extern "C" int tf_test_q35_whole() { return tf_q35_tc::compare<128>() && tf_q35_tc::compare<256>(); }
