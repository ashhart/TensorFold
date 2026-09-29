// ROCm attention for head size 256 on WMMA (v_wmma_f32_16x16x16_bf16), prompt and tree alike. Keys fold into a
// query 16 at a time by absolute position, in ``fold``: S^T = K Q^T puts a query's 16 scores in its lane pair, P is
// then the A operand of P V, and a query's softmax reads no other query, so no query's bits depend on the others in
// its launch, block or wave.
// ``prompt_kernel``: a block per 16 RB prompt rows and KV head, a compute wave per 16 rows and query head; four
// loader waves keep the next two 16-key tiles of K and V in flight and fill a double buffer, so a compute wave only
// folds. A row's keys stop at its own position, so any chunking of a prompt gives a row the same bits.
// Tree attention: each 512-key chunk of a verify window's keys becomes a partial (o, m, l) per (row, head), merged
// in key order by attention.py's Triton ``_merge``. ``shared_kernel``: a block per full committed chunk and KV head,
// CW compute waves of 16 (row, head) pairs each, so all of a stream's pairs read the chunk once; with few chunks
// loader waves feed it, with many every wave loads a tile, then the compute waves fold it (twice the blocks fit).
// ``tail_kernel``: a block per row, KV head and tail chunk (the last committed keys, then the row's own path). A chunk
// a serial row reads whole as committed keys gets the bits a drafted row's tail gives it.

#ifndef __HIPCC__
#error "attention_rocm.cu is ROCm's attention; NVIDIA builds use the Triton kernels"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <type_traits>

namespace {

typedef short short8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int D = 256;
constexpr int CH = 512;                     // keys a chunk, at fixed absolute positions (attention.CHUNK)
constexpr int MAXD = 128;                   // a path's most rows (attention.MAX_NODES)
constexpr int KPAD = 8;                     // bf16 padding a K row: the 16 rows a step reads miss each other's banks
constexpr float NEG = -__builtin_huge_valf();

__device__ __forceinline__ unsigned short bf16_round(float f) {
    const unsigned u = __float_as_uint(f);
    return static_cast<unsigned short>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ unsigned pack2(float a, float b) {
    return static_cast<unsigned>(bf16_round(a)) | (static_cast<unsigned>(bf16_round(b)) << 16);
}

__device__ __forceinline__ float8 wmma(uint4 a, uint4 b, float8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(short8, a),
                                                              __builtin_bit_cast(short8, b), c);
}

constexpr int LOADERS = 4;                  // loader waves a block
constexpr int NL = 32 * LOADERS;            // loader threads
constexpr int ROW8 = D + 16;                // a packed FP8 row: 256 e4m3 bytes, the int8 exponent, padding

struct Tile {
    unsigned short k[16][D + KPAD];         // keys by row
    unsigned short vt[D][16];               // values transposed: B operand rows are keys
};

// Piece i of a tile into shared memory: K by rows (key i / 32), V keys first (key i % 16) so a wave's transposed
// stores hit consecutive keys.
__device__ __forceinline__ void put_k(Tile& t, int i, uint4 x) {
    *reinterpret_cast<uint4*>(&t.k[i / 32][(i % 32) * 8]) = x;
}

__device__ __forceinline__ void put_v(Tile& t, int i, uint4 x) {
    const int key = i % 16, col = (i / 16) * 8;
    const unsigned e[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        t.vt[col + 2 * u][key] = static_cast<unsigned short>(e[u]);
        t.vt[col + 2 * u + 1][key] = static_cast<unsigned short>(e[u] >> 16);
    }
}

// A 16-value piece of a packed row as bf16: e4m3 to fp32, times 2^e (exact), the top 16 bits (exact: an e4m3
// value has four significant bits). ``kv8.unpack`` in Python gives the same values.
__device__ __forceinline__ void widen(uint4 raw, int e, uint4& lo, uint4& hi) {
    const float sc = __builtin_ldexpf(1.0f, e);
    const unsigned w[4] = {raw.x, raw.y, raw.z, raw.w};
    unsigned out[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const auto a = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w[k]), false);
        const auto b = __builtin_amdgcn_cvt_pk_f32_fp8(static_cast<int>(w[k]), true);
        out[2 * k] = (__float_as_uint(a[0] * sc) >> 16) | (__float_as_uint(a[1] * sc) & 0xFFFF0000u);
        out[2 * k + 1] = (__float_as_uint(b[0] * sc) >> 16) | (__float_as_uint(b[1] * sc) & 0xFFFF0000u);
    }
    lo = make_uint4(out[0], out[1], out[2], out[3]);
    hi = make_uint4(out[4], out[5], out[6], out[7]);
}

// A 16-value piece into the tile: K by rows (key i / 16), V keys first (key i % 16).
__device__ __forceinline__ void put_k16(Tile& t, int i, uint4 lo, uint4 hi) {
    uint4* row = reinterpret_cast<uint4*>(&t.k[i / 16][(i % 16) * 16]);
    row[0] = lo;
    row[1] = hi;
}

__device__ __forceinline__ void put_v16(Tile& t, int i, uint4 lo, uint4 hi) {
    const int key = i % 16, col = (i / 16) * 16;
    const unsigned e[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        t.vt[col + 2 * u][key] = static_cast<unsigned short>(e[u]);
        t.vt[col + 2 * u + 1][key] = static_cast<unsigned short>(e[u] >> 16);
    }
}

__device__ __forceinline__ uint4 load(const unsigned short* p) {
    return p != nullptr ? *reinterpret_cast<const uint4*>(p) : make_uint4(0, 0, 0, 0);
}

struct Raw8 {                               // a packed piece in registers: its 16 bytes and its row's exponent
    uint4 b;
    int e;
};

__device__ __forceinline__ Raw8 load8(const unsigned char* row, int col) {
    return row != nullptr ? Raw8{*reinterpret_cast<const uint4*>(row + col), static_cast<signed char>(row[D])}
                          : Raw8{make_uint4(0, 0, 0, 0), 0};
}

struct Wide {                               // a 16-value piece already bf16
    uint4 lo, hi;
};

// A stage names a tile's pieces (``P`` of K and as many of V), loads one into registers (``k``, ``v``) and puts it in
// the bf16 tile (``put``); ``B`` pieces a thread load before any is put. The tile, and so every fold, is the same
// whether the cache holds bf16 or packed FP8 rows of the same values.
template <typename Src>
struct Stage16 {                            // bf16 rows (``Src``: a key's row, or none), 8 values a piece
    static constexpr int P = 16 * D / 8, B = 4;
    using Raw = uint4;
    Src src;
    __device__ Raw k(int key0, int i) const {
        const unsigned short* r = src.k(key0 + i / 32);
        return load(r != nullptr ? r + (i % 32) * 8 : nullptr);
    }
    __device__ Raw v(int key0, int i) const {
        const unsigned short* r = src.v(key0 + i % 16);
        return load(r != nullptr ? r + (i / 16) * 8 : nullptr);
    }
    __device__ static void put(Tile& t, int i, const Raw& kr, const Raw& vr) {
        put_k(t, i, kr);
        put_v(t, i, vr);
    }
};

template <typename Src>
struct Stage8 {                             // packed FP8 rows (``Src``: a key's packed row, or none), 16 values a piece
    static constexpr int P = 16 * D / 16, B = 2;
    using Raw = Raw8;
    Src src;
    __device__ Raw k(int key0, int i) const { return load8(src.k(key0 + i / 16), (i % 16) * 16); }
    __device__ Raw v(int key0, int i) const { return load8(src.v(key0 + i % 16), (i / 16) * 16); }
    __device__ static void put(Tile& t, int i, const Raw& kr, const Raw& vr) {
        uint4 lo, hi;
        widen(kr.b, kr.e, lo, hi);
        put_k16(t, i, lo, hi);
        widen(vr.b, vr.e, lo, hi);
        put_v16(t, i, lo, hi);
    }
};

template <typename S>
struct Pieces {                             // a loader thread's share of a tile
    typename S::Raw k[S::P / NL], v[S::P / NL];
};

// Every thread of the block stages one tile, ``S::B`` pieces of K and of V loaded before any is put.
template <typename S>
__device__ __forceinline__ void stage(Tile& t, const S& s, int key0) {
    for (int b = 0; b < S::P; b += S::B * static_cast<int>(blockDim.x)) {
        typename S::Raw kr[S::B], vr[S::B];
#pragma unroll
        for (int j = 0; j < S::B; ++j) {
            const int i = b + threadIdx.x + j * blockDim.x;
            if (i < S::P) {
                kr[j] = s.k(key0, i);
                vr[j] = s.v(key0, i);
            }
        }
#pragma unroll
        for (int j = 0; j < S::B; ++j) {
            const int i = b + threadIdx.x + j * blockDim.x;
            if (i < S::P) S::put(t, i, kr[j], vr[j]);
        }
    }
}

template <typename S>
__device__ __forceinline__ void fetch(Pieces<S>& r, const S& s, int key0, int lt) {
#pragma unroll
    for (int j = 0; j < S::P / NL; ++j) {
        r.k[j] = s.k(key0, lt + j * NL);
        r.v[j] = s.v(key0, lt + j * NL);
    }
}

template <typename S>
__device__ __forceinline__ void place(Tile& t, const Pieces<S>& r, int lt) {
#pragma unroll
    for (int j = 0; j < S::P / NL; ++j) S::put(t, lt + j * NL, r.k[j], r.v[j]);
}

// ``nt`` tiles from ``key0`` through the double buffer: tile kt sits in buffer kt % 2 while the loaders place tile
// kt + 1 in the other and fetch tile kt + 3 (tile kt + 2 is in flight), one barrier a tile. Loaders and compute
// waves run separate loops with the same barriers (a wave's role is fixed; each loop's registers are its own).
// Loader registers alternate in a two-tile unroll so their indices are fixed.
template <typename S>
__device__ __forceinline__ void load_tiles(Tile (&tb)[2], const S& s, int key0, int nt, int lt) {
    Pieces<S> r0, r1;
    fetch(r0, s, key0, lt);
    if (nt > 1) fetch(r1, s, key0 + 16, lt);
    place(tb[0], r0, lt);
    if (nt > 2) fetch(r0, s, key0 + 32, lt);
    __syncthreads();
    for (int kt = 0; kt < nt; kt += 2) {
        if (kt + 1 < nt) {
            place(tb[1], r1, lt);
            if (kt + 3 < nt) fetch(r1, s, key0 + 16 * (kt + 3), lt);
        }
        __syncthreads();
        if (kt + 1 >= nt) break;
        if (kt + 2 < nt) {
            place(tb[0], r0, lt);
            if (kt + 4 < nt) fetch(r0, s, key0 + 16 * (kt + 4), lt);
        }
        __syncthreads();
    }
}

template <typename Fold>
__device__ __forceinline__ void fold_tiles(const Tile (&tb)[2], int nt, Fold&& fold_tile) {
    __syncthreads();
    for (int kt = 0; kt < nt; kt += 2) {
        fold_tile(tb[0], kt);
        __syncthreads();
        if (kt + 1 >= nt) break;
        fold_tile(tb[1], kt + 1);
        __syncthreads();
    }
}

// This lane's query as the B operand of step t: dims 16 t + 8 half .. + 7 (zeros for no query).
__device__ __forceinline__ void query(const unsigned short* row, int half, uint4 (&qb)[16]) {
#pragma unroll
    for (int t = 0; t < 16; ++t)
        qb[t] = row != nullptr ? reinterpret_cast<const uint4*>(row)[2 * t + half] : make_uint4(0, 0, 0, 0);
}

// One 16-key tile into a wave's 16 queries: Triton ``_tile``'s online softmax, a query at a time, keys outside
// ``valid`` (bit j: key j of the tile) at -inf. A query's scores sit in lanes c and c + 16 (keys 8 half + i).
__device__ __forceinline__ void fold(const Tile& t, const uint4 (&qb)[16], float8 (&o)[16], float& m, float& l,
                                     unsigned valid, float scale, int c, int half) {
    float8 s = float8{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int d = 0; d < 16; ++d)
        s = wmma(*reinterpret_cast<const uint4*>(&t.k[c][16 * d + 8 * half]), qb[d], s);
    float mt = NEG;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        s[i] = (valid >> (8 * half + i)) & 1u ? s[i] * scale : NEG;
        mt = fmaxf(mt, s[i]);
    }
    mt = fmaxf(mt, __shfl_xor(mt, 16));
    const bool active = mt != NEG;
    const float next = active ? fmaxf(m, mt) : m;
    const float alpha = active ? (m == NEG ? 0.0f : __expf(m - next)) : 1.0f;
    float p[8];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        p[i] = active && s[i] != NEG ? __expf(s[i] - next) : 0.0f;
        sum = sum + p[i];
    }
    sum = sum + __shfl_xor(sum, 16);                            // the query's two halves (addition commutes)
    l = l * alpha + sum;
    m = next;
    if (__ballot(alpha != 1.0f) != 0) {                         // O rows 8 half + i: rescale those whose factor moved
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float a = __shfl(alpha, 8 * half + i);
            if (a != 1.0f) {
#pragma unroll
                for (int n = 0; n < 16; ++n) o[n][i] = o[n][i] * a;
            }
        }
    }
    const uint4 pa = make_uint4(pack2(p[0], p[1]), pack2(p[2], p[3]), pack2(p[4], p[5]), pack2(p[6], p[7]));
#pragma unroll
    for (int n = 0; n < 16; ++n)
        o[n] = wmma(pa, *reinterpret_cast<const uint4*>(&t.vt[16 * n + c][8 * half]), o[n]);
}

// A wave's partials: O rows are queries 8 half + i (``at(i)``: the pair's (chunk, row, head) index, or -1), m and l
// sit with query c.
template <typename At>
__device__ __forceinline__ void store(const float8 (&o)[16], float m, float l, At at, int c, int half, float* po,
                                      float* pm, float* pl) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const long long base = at(8 * half + i);
        if (base >= 0) {
#pragma unroll
            for (int n = 0; n < 16; ++n) po[base * D + 16 * n + c] = o[n][i];
        }
    }
    const long long mine = at(c);
    if (half == 0 && mine >= 0) {
        pm[mine] = m;
        pl[mine] = l;
    }
}

template <typename T>
struct Committed {                          // a full committed chunk: every key's row from the cache
    const T *kc, *vc;
    size_t stride;
    __device__ const T* k(int key) const { return kc + key * stride; }
    __device__ const T* v(int key) const { return vc + key * stride; }
};

template <typename T>
struct Bounded {                            // prompt keys: the cache's rows below ``keys``, none past them
    const T *kc, *vc;
    size_t stride;
    int keys;
    __device__ const T* k(int key) const { return key < keys ? kc + key * stride : nullptr; }
    __device__ const T* v(int key) const { return key < keys ? vc + key * stride : nullptr; }
};

struct Tail16 {                             // keys [0, p) from the cache, [p, end) the row's path, none past it
    const unsigned short *kc, *vc, *kn, *vn;
    const int* path;
    size_t stride;
    int p, end, vs, hk;
    __device__ const unsigned short* k(int key) const {
        if (key < p) return kc + key * stride;
        return key < end ? kn + static_cast<size_t>(path[key - p]) * stride + hk * D : nullptr;
    }
    __device__ const unsigned short* v(int key) const {
        if (key < p) return vc + key * stride;
        return key < end ? vn + static_cast<size_t>(path[key - p]) * vs + hk * D : nullptr;
    }
};

// A tail over packed caches: committed keys' pieces widened from their rows, the path's own keys (the window's
// FP8-rounded bf16 rows) read as they are.
struct TailStage8 {
    static constexpr int P = 16 * D / 16, B = 2;
    using Raw = Wide;
    const unsigned char *kc, *vc;
    const unsigned short *kn, *vn;
    const int* path;
    size_t stride8, nstride;
    int p, end, vs, hk;
    __device__ Raw piece(const unsigned char* cache, const unsigned short* nodes, size_t row_stride, int key,
                         int col) const {
        Wide w{make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if (key < p) {
            const Raw8 r = load8(cache + key * stride8, col);
            widen(r.b, r.e, w.lo, w.hi);
        } else if (key < end) {
            const unsigned short* row = nodes + static_cast<size_t>(path[key - p]) * row_stride + hk * D + col;
            w.lo = *reinterpret_cast<const uint4*>(row);
            w.hi = *reinterpret_cast<const uint4*>(row + 8);
        }
        return w;
    }
    __device__ Raw k(int key0, int i) const { return piece(kc, kn, nstride, key0 + i / 16, (i % 16) * 16); }
    __device__ Raw v(int key0, int i) const { return piece(vc, vn, vs, key0 + i % 16, (i / 16) * 16); }
    __device__ static void put(Tile& t, int i, const Raw& kr, const Raw& vr) {
        put_k16(t, i, kr.lo, kr.hi);
        put_v16(t, i, vr.lo, vr.hi);
    }
};

// The stage over one KV head's rows of a stream's caches, bf16 or packed: ``base`` plus ``offs`` counts bf16
// elements for bf16 caches and bytes for packed ones (``attention.offsets``).
template <bool KV8>
__device__ __forceinline__ auto committed(const unsigned short* base, const int64_t* offs, int s, int hk,
                                          int hk_count) {
    if constexpr (KV8) {
        const auto* b8 = reinterpret_cast<const unsigned char*>(base);
        return Stage8<Committed<unsigned char>>{{b8 + offs[2 * s] + hk * ROW8, b8 + offs[2 * s + 1] + hk * ROW8,
                                                 static_cast<size_t>(hk_count) * ROW8}};
    } else {
        return Stage16<Committed<unsigned short>>{{base + offs[2 * s] + hk * D, base + offs[2 * s + 1] + hk * D,
                                                   static_cast<size_t>(hk_count) * D}};
    }
}

// Item (stream, first pair, chunk) of full committed chunks, grid (KV heads, items): a block of CW compute waves
// takes the items whose first pair starts a run of CW tiles (16 CW pairs), the rest return. Pair r of a stream is
// row r / G, head hk G + r % G. KV heads fastest: a chunk's heads run together and read each key's 2 KiB at once.
// PIPE: four more waves load through the double buffer (fastest with few chunks); else every wave loads each tile,
// then the compute waves fold it (more blocks fit: fastest with many). Either way a pair folds the same tiles.
// KV8: the caches hold packed FP8 rows.
template <bool PIPE, bool KV8>
__global__ void __launch_bounds__(PIPE ? 384 : 256) shared_kernel(
        const unsigned short* __restrict__ q, const unsigned short* __restrict__ base,
        const int64_t* __restrict__ offs, const int* __restrict__ streams, const int* __restrict__ items,
        float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl, int w, int h, int hk_count, int g,
        float scale, int item0) {
    __shared__ __align__(16) Tile tb[PIPE ? 2 : 1];
    const int cw = (blockDim.x >> 5) - (PIPE ? LOADERS : 0), item = item0 + blockIdx.y, hk = blockIdx.x;
    const int s = items[3 * item], first = items[3 * item + 1], chunk = items[3 * item + 2];
    const int start = streams[4 * s], rows = streams[4 * s + 1], p = streams[4 * s + 2];
    if (first % (16 * cw) != 0 || (chunk + 1) * CH > p)       // another block's tiles; a padded plan's missing chunk
        return;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, c = lane & 15, half = lane >> 4;
    const bool loader = wave >= cw;
    const int pairs = rows * g, first_pair = first + 16 * wave;
    const bool live = !loader && first_pair < pairs;           // wave-uniform
    const auto src = committed<KV8>(base, offs, s, hk, hk_count);
    if constexpr (PIPE) {
        if (loader) {
            load_tiles(tb, src, chunk * CH, CH / 16, threadIdx.x - 32 * cw);
            return;
        }
    }
    const int mine = first_pair + c;
    uint4 qb[16];
    query(live && mine < pairs ? q + (static_cast<size_t>(start + mine / g) * h + hk * g + mine % g) * D : nullptr,
          half, qb);
    float8 o[16];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    float m = NEG, l = 0.0f;
    if constexpr (PIPE) {
        fold_tiles(tb, CH / 16, [&](const Tile& t, int) {
            if (live) fold(t, qb, o, m, l, 0xFFFFu, scale, c, half);
        });
    } else {
        for (int kt = 0; kt < CH / 16; ++kt) {
            __syncthreads();                                    // the previous tile is consumed
            stage(tb[0], src, chunk * CH + 16 * kt);
            __syncthreads();
            if (live) fold(tb[0], qb, o, m, l, 0xFFFFu, scale, c, half);
        }
    }
    if (live) {
        auto at = [&](int x) -> long long {
            const int r = first_pair + x;
            return r < pairs ? (static_cast<long long>(chunk) * w + start + r / g) * h + hk * g + r % g : -1;
        };
        store(o, m, l, at, c, half, po, pm, pl);
    }
}

// Row, KV head, tail chunk (grid (W, KV heads, tails)): keys from the chunk's start to the last committed one from
// the cache, then the row's path from the window's own keys and values. Four waves load a tile in one batch, wave 0
// folds it (the row's G heads are its queries). (A loader/compute split here spilled registers.)
template <bool KV8>
__global__ void __launch_bounds__(NL) tail_kernel(
        const unsigned short* __restrict__ q, const unsigned short* __restrict__ kn,
        const unsigned short* __restrict__ vn, const unsigned short* __restrict__ base,
        const int64_t* __restrict__ offs, const int* __restrict__ streams, const int* __restrict__ row_stream,
        const int* __restrict__ paths, const int* __restrict__ depths, float* __restrict__ po,
        float* __restrict__ pm, float* __restrict__ pl, int w, int vs, int h, int hk_count, int g, float scale) {
    __shared__ __align__(16) Tile t;
    const int node = blockIdx.x, hk = blockIdx.y;
    const int s = row_stream[node], p = streams[4 * s + 2], nch = streams[4 * s + 3];
    const int chunk = p / CH + blockIdx.z;
    if (chunk >= nch) return;
    const int lane = threadIdx.x & 31, c = lane & 15, half = lane >> 4;
    const bool folds = threadIdx.x < 32;                        // wave 0
    const int end = p + depths[node];                           // keys [0, p) committed, [p, end) the row's path
    const int key0 = chunk * CH;
    const int nt = max(0, min(CH / 16, (end - key0 + 15) / 16));   // tiles past the row's keys fold nothing
    const int* path = paths + static_cast<size_t>(node) * MAXD;
    const auto src = [&] {
        if constexpr (KV8) {
            const auto* b8 = reinterpret_cast<const unsigned char*>(base);
            return TailStage8{b8 + offs[2 * s] + hk * ROW8, b8 + offs[2 * s + 1] + hk * ROW8, kn, vn, path,
                              static_cast<size_t>(hk_count) * ROW8, static_cast<size_t>(hk_count) * D, p, end, vs,
                              hk};
        } else {
            const size_t stride = static_cast<size_t>(hk_count) * D;
            return Stage16<Tail16>{{base + offs[2 * s] + hk * D, base + offs[2 * s + 1] + hk * D, kn, vn, path,
                                    stride, p, end, vs, hk}};
        }
    }();
    uint4 qb[16];
    query(folds && c < g ? q + (static_cast<size_t>(node) * h + hk * g + c) * D : nullptr, half, qb);
    float8 o[16];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    float m = NEG, l = 0.0f;
    for (int kt = 0; kt < nt; ++kt) {
        const int k0 = key0 + 16 * kt;
        __syncthreads();                                        // the previous tile is consumed
        Pieces<std::remove_cv_t<decltype(src)>> r;
        fetch(r, src, k0, threadIdx.x);
        place(t, r, threadIdx.x);
        __syncthreads();
        if (folds) fold(t, qb, o, m, l, k0 + 16 <= end ? 0xFFFFu : (1u << (end - k0)) - 1u, scale, c, half);
    }
    if (folds) {
        auto at = [&](int x) -> long long {
            return x < g ? (static_cast<long long>(chunk) * w + node) * h + hk * g + x : -1;
        };
        store(o, m, l, at, c, half, po, pm, pl);
    }
}

// Prompt attention: q (W, H, D), caches (T, HK, D) holding keys [0, p0 + W), out (W, H, D); G query heads a KV
// head. A block per 16 RB query rows and KV head, a compute wave per 16 rows and query head; PIPE: four loader waves
// through the double buffer, else every wave loads each tile; KV8: packed FP8 caches (T, HK, ROW8). None of these
// change a row's bits.
template <int G, int RB, bool PIPE, bool KV8>
__global__ void __launch_bounds__(32 * (G * RB + (PIPE ? LOADERS : 0))) prompt_kernel(
        const unsigned short* __restrict__ q, const void* __restrict__ kc, const void* __restrict__ vc,
        unsigned short* __restrict__ out, int p0, int w, int h, int hk_count, float scale) {
    constexpr int CW = G * RB;
    __shared__ __align__(16) Tile tb[PIPE ? 2 : 1];
    const int r0 = (gridDim.x - 1 - blockIdx.x) * 16 * RB;    // longest causal blocks first
    const int kvh = blockIdx.y, wave = threadIdx.x >> 5;
    const int tiles = (p0 + min(r0 + 16 * RB, w) - 1) / 16 + 1;
    const auto src = [&] {
        if constexpr (KV8) {
            const auto* k8 = static_cast<const unsigned char*>(kc);
            const auto* v8 = static_cast<const unsigned char*>(vc);
            return Stage8<Bounded<unsigned char>>{{k8 + kvh * ROW8, v8 + kvh * ROW8,
                                                   static_cast<size_t>(hk_count) * ROW8, p0 + w}};
        } else {
            const auto* k16 = static_cast<const unsigned short*>(kc);
            const auto* v16 = static_cast<const unsigned short*>(vc);
            return Stage16<Bounded<unsigned short>>{{k16 + kvh * D, v16 + kvh * D,
                                                     static_cast<size_t>(hk_count) * D, p0 + w}};
        }
    }();
    if constexpr (PIPE) {
        if (wave >= CW) {
            load_tiles(tb, src, 0, tiles, threadIdx.x - 32 * CW);
            return;
        }
    }
    const int lane = threadIdx.x & 31, c = lane & 15, half = lane >> 4;
    const int rows0 = r0 + 16 * (wave / G), head = kvh * G + wave % G;
    const bool live = rows0 < w;                                // wave-uniform
    const int row = rows0 + c, pos = p0 + row;
    uint4 qb[16];
    query(live ? q + (static_cast<size_t>(min(row, w - 1)) * h + head) * D : nullptr, half, qb);
    float8 o[16];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    float m = NEG, l = 0.0f;
    auto valid = [&](int key0) -> unsigned {                   // keys at or before this lane's row, rows before w
        if (row >= w || pos < key0) return 0u;
        return pos >= key0 + 15 ? 0xFFFFu : (2u << (pos - key0)) - 1u;
    };
    if constexpr (PIPE) {
        fold_tiles(tb, tiles, [&](const Tile& t, int kt) {
            if (live) fold(t, qb, o, m, l, valid(16 * kt), scale, c, half);
        });
    } else {
        for (int kt = 0; kt < tiles; ++kt) {
            __syncthreads();                                    // the previous tile is consumed
            stage(tb[0], src, 16 * kt);
            __syncthreads();
            if (live) fold(tb[0], qb, o, m, l, valid(16 * kt), scale, c, half);
        }
    }
    if (!live) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float li = __shfl(l, 8 * half + i);
        const int r = rows0 + 8 * half + i;
        if (r < w) {
#pragma unroll
            for (int n = 0; n < 16; ++n)
                out[(static_cast<size_t>(r) * h + head) * D + 16 * n + c] = bf16_round(o[n][i] / li);
        }
    }
}

}  // namespace

bool tree_supported(int heads, int kv_heads, int dim) {
    return dim == D && kv_heads > 0 && heads % kv_heads == 0 && heads / kv_heads <= 16;
}

// attention.py's ``_shared`` (committed chunks) and ``_tail`` on WMMA; tensors as ``attention`` passes them;
// ``kv8``: the caches hold packed FP8 rows and ``offs`` counts bytes.
void tree_shared(const at::Tensor& q, const at::Tensor& base, const at::Tensor& offs, const at::Tensor& streams,
                 const at::Tensor& items, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int hk, int cw, bool pipe,
                 double scale, bool kv8) {
    const int w = q.size(0), h = q.size(1), n = items.size(0);
    TORCH_CHECK(tree_supported(h, hk, q.size(2)) && 1 <= cw && cw <= 8, "ROCm tree attention: head size 256");
    const auto kernel = pipe ? (kv8 ? shared_kernel<true, true> : shared_kernel<true, false>)
                             : (kv8 ? shared_kernel<false, true> : shared_kernel<false, false>);
    for (int item0 = 0; item0 < n; item0 += 65535)             // a grid's y extent: 65,535 items a launch
        hipLaunchKernelGGL(kernel, dim3(hk, std::min(65535, n - item0)), dim3(32 * (cw + (pipe ? LOADERS : 0))), 0,
                           at::hip::getCurrentHIPStream(), reinterpret_cast<const unsigned short*>(q.data_ptr()),
                           reinterpret_cast<const unsigned short*>(base.data_ptr()), offs.data_ptr<int64_t>(),
                           streams.data_ptr<int>(), items.data_ptr<int>(), po.data_ptr<float>(),
                           pm.data_ptr<float>(), pl.data_ptr<float>(), w, h, hk, h / hk, static_cast<float>(scale),
                           item0);
}

void tree_tail(const at::Tensor& q, const at::Tensor& kn, const at::Tensor& vn, const at::Tensor& base,
               const at::Tensor& offs, const at::Tensor& streams, const at::Tensor& rows, const at::Tensor& paths,
               const at::Tensor& depths, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int tails, double scale,
               bool kv8) {
    const int w = q.size(0), h = q.size(1), hk = kn.size(1);
    TORCH_CHECK(tree_supported(h, hk, q.size(2)) && paths.size(1) == MAXD, "ROCm tree attention: head size 256");
    hipLaunchKernelGGL(kv8 ? tail_kernel<true> : tail_kernel<false>, dim3(w, hk, tails), dim3(NL), 0,
                       at::hip::getCurrentHIPStream(), reinterpret_cast<const unsigned short*>(q.data_ptr()),
                       reinterpret_cast<const unsigned short*>(kn.data_ptr()),
                       reinterpret_cast<const unsigned short*>(vn.data_ptr()),
                       reinterpret_cast<const unsigned short*>(base.data_ptr()), offs.data_ptr<int64_t>(),
                       streams.data_ptr<int>(), rows.data_ptr<int>(), paths.data_ptr<int>(), depths.data_ptr<int>(),
                       po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), w,
                       static_cast<int>(vn.stride(0)), h, hk, h / hk, static_cast<float>(scale));
}

bool attention_supported(int heads, int kv_heads, int dim) {
    const int g = kv_heads > 0 && heads % kv_heads == 0 ? heads / kv_heads : 0;
    return dim == D && (g == 1 || g == 2 || g == 4 || g == 6 || g == 8);
}

// q (W, H, 256), k_cache and v_cache (T, HK, 256) bf16 or (T, HK, 272) packed FP8 rows (uint8), holding keys through
// p0 + W - 1, out (W, H, 256); contiguous. ``rb``: 16-row tiles a block (1 or 2), ``pipe``: loader waves.
void prompt_attention(const at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache, at::Tensor& out,
                      int p0, double scale, int rb, bool pipe) {
    const int w = q.size(0), h = q.size(1), hk = k_cache.size(1);
    const bool kv8 = k_cache.scalar_type() == at::kByte;
    TORCH_CHECK(attention_supported(h, hk, q.size(2)) && (rb == 1 || rb == 2) &&
                k_cache.size(2) == (kv8 ? ROW8 : D) && v_cache.scalar_type() == k_cache.scalar_type(),
                "ROCm prompt attention: head size 256, 1-8 heads a KV head, bf16 or packed FP8 caches");
    if (w == 0) return;
    const dim3 grid((w + 16 * rb - 1) / (16 * rb), hk);
    auto stream = at::hip::getCurrentHIPStream();
    const auto* qp = reinterpret_cast<const unsigned short*>(q.data_ptr());
    const void* kp = k_cache.data_ptr();
    const void* vp = v_cache.data_ptr();
    auto* op = reinterpret_cast<unsigned short*>(out.data_ptr());
    const float sc = static_cast<float>(scale);
#define LAUNCH(G, RB, PIPE, KV8)                                                                                  \
    hipLaunchKernelGGL((prompt_kernel<G, RB, PIPE, KV8>), grid, dim3(32 * ((G) * (RB) + ((PIPE) ? LOADERS : 0))), \
                       0, stream, qp, kp, vp, op, p0, w, h, hk, sc)
#define FORMAT(G, RB, PIPE)                                                                      \
    if (kv8) LAUNCH(G, RB, PIPE, true); else LAUNCH(G, RB, PIPE, false);
#define ROWS(G)                                                                                  \
    if (rb == 1) {                                                                               \
        if (pipe) { FORMAT(G, 1, true) } else { FORMAT(G, 1, false) }                            \
    } else {                                                                                     \
        if (pipe) { FORMAT(G, 2, true) } else { FORMAT(G, 2, false) }                            \
    }
    switch (h / hk) {
        case 1: ROWS(1); break;
        case 2: ROWS(2); break;
        case 4: ROWS(4); break;
        case 6: ROWS(6); break;
        default: ROWS(8); break;
    }
#undef ROWS
#undef FORMAT
#undef LAUNCH
}
