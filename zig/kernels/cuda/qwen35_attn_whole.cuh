#pragma once

template <int THREADS>
struct WholeFp8Tiles {
    static constexpr bool paged_layout = false;
    static constexpr int PER = TK * 16 / THREADS;  // a thread's 16-code parts of a tile's keys (and of its values)
    TileStore s;
    const unsigned char *K, *V;
    const float *KS, *VS;
    uint4 kr[PER], vr[PER];
    float sr;
    __device__ WholeFp8Tiles(TileStore store, const Seg& sg, int layer, int kvh) : s(store) {
        K = reinterpret_cast<const unsigned char*>(sg.kv) + (size_t)layer * 2 * kvh * sg.cap * (HD + 4);
        V = K + (size_t)kvh * sg.cap * HD;
        KS = reinterpret_cast<const float*>(V + (size_t)kvh * sg.cap * HD);
        VS = KS + (size_t)kvh * sg.cap;
    }
    // keys t0 .. t0 + TK - 1, key t from cache row row(t) (-1: zeros)
    template <class Row>
    __device__ void fetch(int t0, Row row) {
        const int tid = threadIdx.x, part = tid & 15;
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const int r = row(t0 + (tid >> 4) + THREADS / 16 * i);
            kr[i] = vr[i] = make_uint4(0u, 0u, 0u, 0u);
            if (r >= 0) {
                kr[i] = *reinterpret_cast<const uint4*>(K + (size_t)r * HD + part * 16);
                vr[i] = *reinterpret_cast<const uint4*>(V + (size_t)r * HD + part * 16);
            }
        }
        const int r = row(t0 + (tid & 31));
        sr = 0.0f;
        if (tid < 64 && r >= 0) sr = (tid < 32 ? KS : VS)[r];
    }
    __device__ void put(bool keys) {
        const int tid = threadIdx.x, part = tid & 15;
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const int u = (tid >> 4) + THREADS / 16 * i;
            if (keys) widen16(kr[i], &s.kt[swz(u, 2 * part)], &s.kt[swz(u, 2 * part + 1)]);
            widen16(vr[i], &s.vt[swz(u, 2 * part)], &s.vt[swz(u, 2 * part + 1)]);
        }
        if (keys && tid < 32) s.ksc[tid] = sr;
        else if (tid >= 32 && tid < 64) s.vsc[tid - 32] = sr;
    }
    __device__ const uint4* keys() const { return s.kt; }
    __device__ const uint4* values() const { return s.vt; }
    __device__ float kscale(int u) const { return s.ksc[u]; }
    __device__ float vscale(int u) const { return s.vsc[u]; }
    __device__ float key_scale(int r) const { return KS[r]; }
};

// bf16 rows (qwen35.cu's layout): a tile copied by cp.async into the free one of two buffers, a tile ahead.
template <int THREADS>
struct WholeBf16Tiles {
    static constexpr bool paged_layout = false;
    static constexpr int TILE = TK * CH;  // uint4 of one tile's keys (or values)
    TileStore s;
    const __nv_bfloat16 *K, *V;
    int cur = 1;
    __device__ WholeBf16Tiles(TileStore store, const Seg& sg, int layer, int kvh) : s(store) {
        K = reinterpret_cast<const __nv_bfloat16*>(sg.kv) + (size_t)layer * 2 * kvh * sg.cap * HD;
        V = K + (size_t)kvh * sg.cap * HD;
    }
    template <class Row>
    __device__ void fetch(int t0, Row row) {
        uint4* dst = s.kt + (cur ^ 1) * 2 * TILE;
#pragma unroll
        for (int i = 0; i < TILE / THREADS; i++) {
            const int at = threadIdx.x + THREADS * i, u = at / CH, chunk = at % CH;
            const int r = row(t0 + u);
            const size_t from = (size_t)(r < 0 ? 0 : r) * HD + chunk * 8;
            qmm_frag::cp16z(dst + swz(u, chunk), K + from, r >= 0);
            qmm_frag::cp16z(dst + TILE + swz(u, chunk), V + from, r >= 0);
        }
        qmm_frag::commit();
    }
    __device__ void put(bool) {
        qmm_frag::wait<0>();
        cur ^= 1;
    }
    __device__ const uint4* keys() const { return s.kt + cur * 2 * TILE; }
    __device__ const uint4* values() const { return s.kt + cur * 2 * TILE + TILE; }
    __device__ float kscale(int) const { return 1.0f; }
    __device__ float vscale(int) const { return 1.0f; }
    __device__ float key_scale(int) const { return 1.0f; }
};
