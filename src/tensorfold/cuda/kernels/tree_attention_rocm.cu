// ROCm tree attention for head size 256 on WMMA (v_wmma_f32_16x16x16_bf16): each 512-key chunk of a window's keys
// becomes a partial (o, m, l) per (row, head), merged in key order by attention.py's Triton ``_merge``.
// ``shared_kernel``: a block per full committed chunk and KV head, CW compute waves of 16 (row, head) pairs each, so
// all of a stream's pairs read the chunk once (the Triton kernel read it once per 16 pairs). With few chunks four
// loader waves keep the next two 16-key tiles in flight and fill a double buffer, so a compute wave only folds; with
// many, every wave loads a tile, then the compute waves fold it (twice the blocks fit). ``tail_kernel``: a block per
// row, KV head and tail chunk (the last committed keys, then the row's own path), four waves loading, one folding.
// Both fold a 16-key tile with ``fold``: S^T = K Q^T puts a pair's 16 scores in its lane pair, P is then the A
// operand of P V, and a pair's softmax reads no other pair. A chunk a serial row reads whole as committed keys gets
// the bits a drafted row's tail gives it, and no pair's bits depend on the pairs beside it, the block or its waves.

#ifndef __HIPCC__
#error "tree_attention_rocm.cu is ROCm's tree attention; NVIDIA builds use the Triton kernel"
#endif

#include <ATen/ATen.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>

#include <algorithm>

namespace {

typedef short short8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr int D = 256;
constexpr int CH = 512;                     // keys a chunk, at fixed absolute positions (attention.CHUNK)
constexpr int MAXD = 128;                   // a path's most rows (attention.MAX_NODES)
constexpr int KPAD = 8;                     // bf16 padding a K row: the 16 rows a step reads miss each other's banks
constexpr int TILE = 16 * D / 8;            // 16-byte pieces of one 16-key tile of K (or V)
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
constexpr int PL = TILE / NL;               // pieces of K (and of V) a loader thread carries for a tile

struct Tile {
    unsigned short k[16][D + KPAD];         // keys by row
    unsigned short vt[D][16];               // values transposed: B operand rows are keys
};

struct Pieces {
    uint4 k[PL], v[PL];
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

__device__ __forceinline__ uint4 load(const unsigned short* p) {
    return p != nullptr ? *reinterpret_cast<const uint4*>(p) : make_uint4(0, 0, 0, 0);
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

// A loader thread's pieces of the tile at ``key0`` (``src.k``/``src.v``: a key's row, or none), then into shared
// memory: K by rows (key i / 32), V keys first (key i % 16).
template <typename Src>
__device__ __forceinline__ void fetch(Pieces& r, const Src& src, int key0, int lt) {
#pragma unroll
    for (int j = 0; j < PL; ++j) {
        const int i = lt + j * NL;
        const unsigned short* kp = src.k(key0 + i / 32);
        const unsigned short* vp = src.v(key0 + i % 16);
        r.k[j] = load(kp != nullptr ? kp + (i % 32) * 8 : nullptr);
        r.v[j] = load(vp != nullptr ? vp + (i / 16) * 8 : nullptr);
    }
}

__device__ __forceinline__ void place(Tile& t, const Pieces& r, int lt) {
#pragma unroll
    for (int j = 0; j < PL; ++j) {
        put_k(t, lt + j * NL, r.k[j]);
        put_v(t, lt + j * NL, r.v[j]);
    }
}

// ``nt`` tiles from ``key0`` through the double buffer: tile kt sits in buffer kt % 2 while the loaders place tile
// kt + 1 in the other and fetch tile kt + 3 (tile kt + 2 is in flight), one barrier a tile. Loaders and compute
// waves run separate loops with the same barriers (a wave's role is fixed; each loop's registers are its own).
// Loader registers alternate in a two-tile unroll so their indices are fixed.
template <typename Src>
__device__ __forceinline__ void load_tiles(Tile (&tb)[2], const Src& src, int key0, int nt, int lt) {
    Pieces r0, r1;
    fetch(r0, src, key0, lt);
    if (nt > 1) fetch(r1, src, key0 + 16, lt);
    place(tb[0], r0, lt);
    if (nt > 2) fetch(r0, src, key0 + 32, lt);
    __syncthreads();
    for (int kt = 0; kt < nt; kt += 2) {
        if (kt + 1 < nt) {
            place(tb[1], r1, lt);
            if (kt + 3 < nt) fetch(r1, src, key0 + 16 * (kt + 3), lt);
        }
        __syncthreads();
        if (kt + 1 >= nt) break;
        if (kt + 2 < nt) {
            place(tb[0], r0, lt);
            if (kt + 4 < nt) fetch(r0, src, key0 + 16 * (kt + 4), lt);
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

struct Committed {                          // a full committed chunk: every key from the cache
    const unsigned short *kc, *vc;
    size_t stride;
    __device__ const unsigned short* k(int key) const { return kc + key * stride; }
    __device__ const unsigned short* v(int key) const { return vc + key * stride; }
};

struct Tail {                               // keys [0, p) from the cache, [p, end) the row's path, none past it
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

// Item (stream, first pair, chunk) of full committed chunks, grid (KV heads, items): a block of CW compute waves
// takes the items whose first pair starts a run of CW tiles (16 CW pairs), the rest return. Pair r of a stream is
// row r / G, head hk G + r % G. KV heads fastest: a chunk's heads run together and read each key's 2 KiB at once.
// PIPE: four more waves load through the double buffer (fastest with few chunks); else every wave loads each tile,
// then the compute waves fold it (more blocks fit: fastest with many). Either way a pair folds the same tiles.
template <bool PIPE>
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
    const size_t stride = static_cast<size_t>(hk_count) * D;
    const Committed src{base + offs[2 * s] + hk * D, base + offs[2 * s + 1] + hk * D, stride};
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
            for (int b = 0; b < TILE; b += 4 * blockDim.x) {    // every load of a batch before its stores
                uint4 kr[4], vr[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int i = b + threadIdx.x + j * blockDim.x;
                    kr[j] = load(i < TILE ? src.k(chunk * CH + 16 * kt + i / 32) + (i % 32) * 8 : nullptr);
                    vr[j] = load(i < TILE ? src.v(chunk * CH + 16 * kt + i % 16) + (i / 16) * 8 : nullptr);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int i = b + threadIdx.x + j * blockDim.x;
                    if (i < TILE) {
                        put_k(tb[0], i, kr[j]);
                        put_v(tb[0], i, vr[j]);
                    }
                }
            }
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
    const size_t stride = static_cast<size_t>(hk_count) * D;
    const Tail src{base + offs[2 * s] + hk * D, base + offs[2 * s + 1] + hk * D, kn, vn,
                   paths + static_cast<size_t>(node) * MAXD, stride, p, end, vs, hk};
    uint4 qb[16];
    query(folds && c < g ? q + (static_cast<size_t>(node) * h + hk * g + c) * D : nullptr, half, qb);
    float8 o[16];
#pragma unroll
    for (int n = 0; n < 16; ++n) o[n] = float8{0, 0, 0, 0, 0, 0, 0, 0};
    float m = NEG, l = 0.0f;
    for (int kt = 0; kt < nt; ++kt) {
        const int k0 = key0 + 16 * kt;
        __syncthreads();                                        // the previous tile is consumed
        Pieces r;
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

}  // namespace

bool tree_supported(int heads, int kv_heads, int dim) {
    return dim == D && kv_heads > 0 && heads % kv_heads == 0 && heads / kv_heads <= 16;
}

// attention.py's ``_shared`` (committed chunks) and ``_tail`` on WMMA; tensors as ``attention`` passes them.
void tree_shared(const at::Tensor& q, const at::Tensor& base, const at::Tensor& offs, const at::Tensor& streams,
                 const at::Tensor& items, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int hk, int cw, bool pipe,
                 double scale) {
    const int w = q.size(0), h = q.size(1), n = items.size(0);
    TORCH_CHECK(tree_supported(h, hk, q.size(2)) && 1 <= cw && cw <= 8, "ROCm tree attention: head size 256");
    for (int item0 = 0; item0 < n; item0 += 65535)             // a grid's y extent: 65,535 items a launch
        hipLaunchKernelGGL(pipe ? shared_kernel<true> : shared_kernel<false>, dim3(hk, std::min(65535, n - item0)),
                           dim3(32 * (cw + (pipe ? LOADERS : 0))), 0,
                           at::hip::getCurrentHIPStream(), reinterpret_cast<const unsigned short*>(q.data_ptr()),
                           reinterpret_cast<const unsigned short*>(base.data_ptr()), offs.data_ptr<int64_t>(),
                           streams.data_ptr<int>(), items.data_ptr<int>(), po.data_ptr<float>(),
                           pm.data_ptr<float>(), pl.data_ptr<float>(), w, h, hk, h / hk, static_cast<float>(scale),
                           item0);
}

void tree_tail(const at::Tensor& q, const at::Tensor& kn, const at::Tensor& vn, const at::Tensor& base,
               const at::Tensor& offs, const at::Tensor& streams, const at::Tensor& rows, const at::Tensor& paths,
               const at::Tensor& depths, at::Tensor& po, at::Tensor& pm, at::Tensor& pl, int tails, double scale) {
    const int w = q.size(0), h = q.size(1), hk = kn.size(1);
    TORCH_CHECK(tree_supported(h, hk, q.size(2)) && paths.size(1) == MAXD, "ROCm tree attention: head size 256");
    hipLaunchKernelGGL(tail_kernel, dim3(w, hk, tails), dim3(NL), 0, at::hip::getCurrentHIPStream(),
                       reinterpret_cast<const unsigned short*>(q.data_ptr()),
                       reinterpret_cast<const unsigned short*>(kn.data_ptr()),
                       reinterpret_cast<const unsigned short*>(vn.data_ptr()),
                       reinterpret_cast<const unsigned short*>(base.data_ptr()), offs.data_ptr<int64_t>(),
                       streams.data_ptr<int>(), rows.data_ptr<int>(), paths.data_ptr<int>(), depths.data_ptr<int>(),
                       po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(), w,
                       static_cast<int>(vn.stride(0)), h, hk, h / hk, static_cast<float>(scale));
}
