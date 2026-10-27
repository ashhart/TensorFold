#pragma once

template <int THREADS>
struct Fp8Tiles {
    static constexpr bool paged_layout = true;
    static constexpr int PER = TK * 16 / THREADS;  // a thread's 16-code parts of a tile's keys (and of its values)
    TileStore s;
    const unsigned char *K, *V;
    const float *KS, *VS;
    uint4 kr[PER], vr[PER];
    float sr;
    // the key head's rows: a slab of cap rows, or (pt set) pages of ptok rows in qwen35_kv8.cu's layout
    const unsigned char* arena;
    TfQ35PageRun page;
    size_t lblk, hoff, vgap, sgap;  // a page's layer block (bytes), the head's first row in it, keys to values, codes to scales
    int ptok, kvh;
    bool paged;
    __device__ Fp8Tiles(TileStore store, const Seg& sg, int layer, int kvh_, int h) : s(store) {
        const unsigned char* slab = reinterpret_cast<const unsigned char*>(sg.kv) + (size_t)layer * 2 * kvh_ * sg.cap * (HD + 4);
        K = slab + (size_t)h * sg.cap * HD;
        V = K + (size_t)kvh_ * sg.cap * HD;
        KS = reinterpret_cast<const float*>(slab + (size_t)2 * kvh_ * sg.cap * HD) + (size_t)h * sg.cap;
        VS = KS + (size_t)kvh_ * sg.cap;
        arena = reinterpret_cast<const unsigned char*>(sg.kv);
        page = TfQ35PageRun{reinterpret_cast<const int*>(sg.pt), sg.ptok, -1, sg.ps, 0};
        lblk = (size_t)layer * 2 * kvh_ * sg.ptok * (HD + 4);
        hoff = (size_t)h * sg.ptok;
        vgap = (size_t)kvh_ * sg.ptok * HD;
        sgap = (size_t)2 * kvh_ * sg.ptok * HD;
        ptok = sg.ptok;
        kvh = kvh_;
        paged = sg.pt != 0;
    }
    // cache position pos's codes, values and scales (pos >= 0)
    __device__ const unsigned char* kp(int pos) { return paged ? arena + tf_q35_page_base(&page, pos) + lblk + (hoff + pos % ptok) * HD : K + (size_t)pos * HD; }
    __device__ const unsigned char* vp(int pos) { return paged ? kp(pos) + vgap : V + (size_t)pos * HD; }
    __device__ const float* ksp(int pos) { return paged ? reinterpret_cast<const float*>(arena + tf_q35_page_base(&page, pos) + lblk + sgap) + hoff + pos % ptok : KS + pos; }
    __device__ const float* vsp(int pos) { return paged ? ksp(pos) + (size_t)kvh * ptok : VS + pos; }
    // keys t0 .. t0 + TK - 1, key t from cache row row(t) (-1: zeros)
    template <class Row>
    __device__ void fetch(int t0, Row row) {
        const int tid = threadIdx.x, part = tid & 15;
#pragma unroll
        for (int i = 0; i < PER; i++) {
            const int r = row(t0 + (tid >> 4) + THREADS / 16 * i);
            kr[i] = vr[i] = make_uint4(0u, 0u, 0u, 0u);
            if (r >= 0) {
                kr[i] = *reinterpret_cast<const uint4*>(kp(r) + part * 16);
                vr[i] = *reinterpret_cast<const uint4*>(vp(r) + part * 16);
            }
        }
        const int r = row(t0 + (tid & 31));
        sr = 0.0f;
        if (tid < 64 && r >= 0) sr = *(tid < 32 ? ksp(r) : vsp(r));
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
    __device__ float key_scale(int pos) { return *ksp(pos); }
};

// bf16 rows (qwen35.cu's layout): a tile copied by cp.async into the free one of two buffers, a tile ahead.
template <int THREADS>
struct Bf16Tiles {
    static constexpr bool paged_layout = true;
    static constexpr int TILE = TK * CH;  // uint4 of one tile's keys (or values)
    TileStore s;
    const __nv_bfloat16 *K, *V;
    int cur = 1;
    // the key head's rows: a slab of cap rows, or (pt set) pages of ptok rows in qwen35.cu's layout
    const __nv_bfloat16* arena;
    TfQ35PageRun page;
    size_t lblk, hoff, vgap;  // a page's layer block (elements), the head's first row in it, keys to values
    int ptok;
    bool paged;
    __device__ Bf16Tiles(TileStore store, const Seg& sg, int layer, int kvh, int h) : s(store) {
        K = reinterpret_cast<const __nv_bfloat16*>(sg.kv) + (size_t)layer * 2 * kvh * sg.cap * HD + (size_t)h * sg.cap * HD;
        V = K + (size_t)kvh * sg.cap * HD;
        arena = reinterpret_cast<const __nv_bfloat16*>(sg.kv);
        page = TfQ35PageRun{reinterpret_cast<const int*>(sg.pt), sg.ptok, -1, sg.ps, 0};
        lblk = (size_t)layer * 2 * kvh * sg.ptok * HD;
        hoff = (size_t)h * sg.ptok;
        vgap = (size_t)kvh * sg.ptok * HD;
        ptok = sg.ptok;
        paged = sg.pt != 0;
    }
    // cache position pos's row of keys and of values (pos >= 0)
    __device__ const __nv_bfloat16* kp(int pos) { return paged ? arena + tf_q35_page_base(&page, pos) + lblk + (hoff + pos % ptok) * HD : K + (size_t)pos * HD; }
    __device__ const __nv_bfloat16* vp(int pos) { return paged ? kp(pos) + vgap : V + (size_t)pos * HD; }
    template <class Row>
    __device__ void fetch(int t0, Row row) {
        uint4* dst = s.kt + (cur ^ 1) * 2 * TILE;
#pragma unroll
        for (int i = 0; i < TILE / THREADS; i++) {
            const int at = threadIdx.x + THREADS * i, u = at / CH, chunk = at % CH;
            const int r = row(t0 + u), pos = r < 0 ? 0 : r;  // a key past the chunk reads row 0 and lands zeros
            qmm_frag::cp16z(dst + swz(u, chunk), kp(pos) + chunk * 8, r >= 0);
            qmm_frag::cp16z(dst + TILE + swz(u, chunk), vp(pos) + chunk * 8, r >= 0);
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
    __device__ float key_scale(int) { return 1.0f; }
};
