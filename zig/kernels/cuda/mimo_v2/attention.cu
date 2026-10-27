// MiMo-V2's attention: q/k heads of 192 (64 rotated), v heads of 128, sinks on sliding layers; built with glue flags.

// Caches: keys [slots, LEN, HK, 192], values [slots, LEN, HK, 128], key p at row p % LEN (a sliding layer's: a ring).

// A row's keys: chunks of 256 at absolute positions, tiles of 32; chunk partials fold in order, the sink last.

// Decode rows run chunks in parallel (rows, merge); full layers' prompt rows run on tensor cores (mimo_attn_prompt).

#include <cstdint>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace {

constexpr int QK = 192;      // q/k head dim
constexpr int VD = 128;      // v head dim
constexpr int RD = 64;       // rotated dims
constexpr int CK = 256;      // keys a chunk
constexpr int TILE = 32;     // keys a tile
constexpr int KW = QK / 2 + 1;  // K tile row stride in 32-bit words (96 + 1: conflict-free row-per-lane reads)
constexpr int QW = QK / 2;   // a query in half2 words
#define NEG_INF (-CUDART_INF_F)

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// A lane's share of a (row, head) state: m and l (every lane's), o dims 2 lane, 2 lane + 1, 64 + 2 lane, 65 + 2 lane.
struct State {
    float m, l, o[4];
    __device__ __forceinline__ void clear() {
        m = NEG_INF;
        l = 0.f;
        o[0] = o[1] = o[2] = o[3] = 0.f;
    }
};

// Load keys [t0, t0 + 32) of one key head into shared memory; rows at or past `end` are zero.
__device__ __forceinline__ void load_tile(uint32_t* ks, __half2* vs, const __half* kc, const __half* vc, int64_t base,
                                          int LEN, int HK, int kh, int t0, int end) {
    for (int i = threadIdx.x; i < TILE * (QW + VD / 2); i += blockDim.x) {
        const int row = i / (QW + VD / 2), w = i % (QW + VD / 2);
        const int key = t0 + row;
        uint32_t val = 0;
        if (key < end) {
            const int64_t at = (base + key % LEN) * HK + kh;
            val = w < QW ? ((const uint32_t*)(kc + at * QK))[w] : ((const uint32_t*)(vc + at * VD))[w - QW];
        }
        if (w < QW)
            ks[row * KW + w] = val;
        else
            *(uint32_t*)(vs + row * (VD / 2) + (w - QW)) = val;
    }
}

// A tile into a row's chunk state: lane j scores key t0 + j (valid in [lo, pos]); no valid key: state unchanged.
__device__ __forceinline__ void tile_step(State& st, const uint32_t* ks, const __half2* vs, const __half2* q, int t0,
                                         int lo, int pos, float scale) {
    const int lane = threadIdx.x & 31;
    const int key = t0 + lane;
    const bool valid = key >= lo && key <= pos;
    float s = NEG_INF;
    if (valid) {
        const uint32_t* kr = ks + lane * KW;
        float acc = 0.f;
#pragma unroll 8
        for (int w = 0; w < QW; ++w) {
            const uint32_t u = kr[w];
            const float2 kf = __half22float2(*(const __half2*)&u);
            const float2 qf = __half22float2(q[w]);
            acc = __fmaf_rn(qf.x, kf.x, acc);
            acc = __fmaf_rn(qf.y, kf.y, acc);
        }
        s = acc * scale;
    }
    const float mt = warp_max(s);
    if (mt == NEG_INF) return;
    const float mn = fmaxf(st.m, mt);
    const float corr = st.m == NEG_INF ? 0.f : expf(st.m - mn);
    const float pj = valid ? expf(s - mn) : 0.f;
    st.l = st.l * corr + warp_sum(pj);
#pragma unroll
    for (int i = 0; i < 4; ++i) st.o[i] = st.o[i] * corr;
    for (int j = 0; j < TILE; ++j) {
        const float pb = __shfl_sync(0xffffffffu, pj, j);
        const float2 a = __half22float2(vs[j * (VD / 2) + lane]);
        const float2 b = __half22float2(vs[j * (VD / 2) + 32 + lane]);
        st.o[0] = __fmaf_rn(pb, a.x, st.o[0]);
        st.o[1] = __fmaf_rn(pb, a.y, st.o[1]);
        st.o[2] = __fmaf_rn(pb, b.x, st.o[2]);
        st.o[3] = __fmaf_rn(pb, b.y, st.o[3]);
    }
    st.m = mn;
}

// A chunk's partial folded into the running state (chunk order); an empty chunk (m = -inf) changes nothing.
__device__ __forceinline__ void fold(State& run, float m, float l, const float* o) {
    if (m == NEG_INF) return;
    const float mn = fmaxf(run.m, m);
    const float a = run.m == NEG_INF ? 0.f : expf(run.m - mn);
    const float b = expf(m - mn);
    run.l = run.l * a + l * b;
#pragma unroll
    for (int i = 0; i < 4; ++i) run.o[i] = run.o[i] * a + o[i] * b;
    run.m = mn;
}

// The sink logit last, then o / l to fp16 at out (a head's 128 values).
__device__ __forceinline__ void finish(State run, const float* sinks, int h, __half* out) {
    const int lane = threadIdx.x & 31;
    if (sinks) {
        const float sink = sinks[h];
        const float mn = fmaxf(run.m, sink);
        const float a = run.m == NEG_INF ? 0.f : expf(run.m - mn);
        run.l = run.l * a + expf(sink - mn);
#pragma unroll
        for (int i = 0; i < 4; ++i) run.o[i] = run.o[i] * a;
    }
    *(__half2*)(out + 2 * lane) = __floats2half2_rn(run.o[0] / run.l, run.o[1] / run.l);
    *(__half2*)(out + 64 + 2 * lane) = __floats2half2_rn(run.o[2] / run.l, run.o[3] / run.l);
}

// A row's first key: pos - win + 1 on a sliding layer (win > 0), else 0.
__device__ __forceinline__ int first_key(int pos, int win) { return win > 0 ? max(pos - win + 1, 0) : 0; }

}  // namespace

// Grid (rows, H + 2 HK), 64 threads: q, k heads (RoPE on dims 0-63, fp32), v heads (128 of 192); k, v to the cache.
extern "C" __global__ void mimo_qkv(const __half* __restrict__ q, const __half* __restrict__ k,
                                    const __half* __restrict__ v, const float* __restrict__ cos_t,
                                    const float* __restrict__ sin_t, const int32_t* __restrict__ pos,
                                    const int32_t* __restrict__ slot, __half* __restrict__ qo, __half* __restrict__ kc,
                                    __half* __restrict__ vc, int LEN, int H, int HK) {
    const int64_t r = blockIdx.x;
    const int hd = blockIdx.y, t = threadIdx.x;
    const int p = pos[r];
    const int64_t at = (int64_t)slot[r] * LEN + p % LEN;
    if (hd < H + HK) {
        const bool is_q = hd < H;
        const __half* src = is_q ? q + (r * H + hd) * QK : k + (r * HK + (hd - H)) * QK;
        __half* dst = is_q ? qo + (r * H + hd) * QK : kc + (at * HK + (hd - H)) * QK;
        constexpr int half = RD / 2;
        const float x = __half2float(src[t]);
        const float xp = __half2float(src[(t + half) % RD]);
        const int j = t % half;
        const float c = cos_t[(int64_t)p * half + j], s = sin_t[(int64_t)p * half + j];
        const float y = x * c + (t < half ? -xp : xp) * s;
        dst[t] = __float2half_rn(y);
        dst[RD + t] = src[RD + t];
        dst[RD + 64 + t] = src[RD + 64 + t];
    } else {
        const int kh = hd - H - HK;
        const __half* src = v + (r * HK + kh) * QK;
        __half* dst = vc + (at * HK + kh) * VD;
        dst[t] = src[t];
        dst[64 + t] = src[64 + t];
    }
}

// Grid (rows, HK, NCH), a warp a query head: row r's chunk c (from its window's first) into [(r NCH + c) H + h].
extern "C" __global__ void mimo_attn_rows(const __half* __restrict__ qo, const __half* __restrict__ kc,
                                          const __half* __restrict__ vc, const int32_t* __restrict__ pos,
                                          const int32_t* __restrict__ slot, float* __restrict__ po,
                                          float* __restrict__ pm, float* __restrict__ pl, int LEN, int NCH, int H,
                                          int HK, int win, float scale) {
    extern __shared__ uint32_t smem[];
    uint32_t* ks = smem;
    __half2* vs = (__half2*)(ks + TILE * KW);
    __half2* qs = vs + TILE * (VD / 2);
    const int64_t r = blockIdx.x;
    const int kh = blockIdx.y, c = blockIdx.z;
    const int G = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = kh * G + warp;
    const int p = pos[r];
    const int lo = first_key(p, win);
    const int start = (lo / CK + c) * CK;
    const int end = min(start + CK, p + 1);
    const int64_t part = (r * NCH + c) * H + h;
    State st;
    st.clear();
    if (start < end) {
        for (int i = threadIdx.x; i < G * QW; i += blockDim.x)
            qs[i] = ((const __half2*)(qo + (r * H + kh * G) * QK))[i];
        const int64_t base = (int64_t)slot[r] * LEN;
        for (int t0 = start; t0 < end; t0 += TILE) {
            __syncthreads();
            load_tile(ks, vs, kc, vc, base, LEN, HK, kh, t0, end);
            __syncthreads();
            tile_step(st, ks, vs, qs + warp * QW, t0, lo, p, scale);
        }
    }
    if (lane == 0) {
        pm[part] = st.m;
        pl[part] = st.l;
    }
    float* o = po + part * VD;
    *(float2*)(o + 2 * lane) = make_float2(st.o[0], st.o[1]);
    *(float2*)(o + 64 + 2 * lane) = make_float2(st.o[2], st.o[3]);
}

// Grid (rows, H / 4), 128 threads (a warp a head): the row's chunks folded in order, the sink last; fp16 out.
extern "C" __global__ void mimo_attn_merge(const float* __restrict__ po, const float* __restrict__ pm,
                                           const float* __restrict__ pl, const float* __restrict__ sinks,
                                           __half* __restrict__ out, int NCH, int H) {
    const int64_t r = blockIdx.x;
    const int h = blockIdx.y * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    State run;
    run.clear();
    for (int c = 0; c < NCH; ++c) {
        const int64_t part = (r * NCH + c) * H + h;
        const float* o = po + part * VD;
        const float2 a = *(const float2*)(o + 2 * lane), b = *(const float2*)(o + 64 + 2 * lane);
        const float ov[4] = {a.x, a.y, b.x, b.y};
        fold(run, pm[part], pl[part], ov);
    }
    finish(run, sinks, h, out + (r * H + h) * VD);
}

// Prompt rows on tensor cores: 64 rows of one slot and one head a block, a warp 16 rows; keys in tiles of 64 from 0.
namespace tc {
constexpr int BM = 64, BN = 64, THREADS = 128;
constexpr int KS = QK + 8, VS = VD + 8;   // smem row strides in halves (16-byte rows, no ldmatrix bank conflicts)
constexpr int STAGE = BN * KS + BN * VS;  // a stage's halves: K then V
constexpr int SMEM = 2 * STAGE * 2;       // two stages, bytes

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const __half* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t (&r)[4], const __half* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// d += a b: m16n8k16, fp16 in, fp32 accumulators.
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack(float x, float y) {
    const __half2 h = __floats2half2_rn(x, y);
    return *(const uint32_t*)&h;
}
// Keys [t0, t0 + 64) of key head kh into a stage (rows past `end` zero): 24 + 16 pieces of 16 bytes a key.
__device__ __forceinline__ void load_kv(__half* st, const __half* kc, const __half* vc, int64_t base, int LEN, int HK,
                                        int kh, int t0, int end) {
    for (int i = threadIdx.x; i < BN * (QK / 8 + VD / 8); i += THREADS) {
        const int key = i / (QK / 8 + VD / 8), w = i % (QK / 8 + VD / 8);
        const bool ok = t0 + key < end;
        const int64_t at = (base + (ok ? t0 + key : 0) % LEN) * HK + kh;
        if (w < QK / 8)
            cp16(st + key * KS + w * 8, kc + at * QK + w * 8, ok);
        else
            cp16(st + BN * KS + key * VS + (w - QK / 8) * 8, vc + at * VD + (w - QK / 8) * 8, ok);
    }
}
}  // namespace tc

// Grid (ceil(n / 64), H), 128 threads, tc::SMEM bytes: full-layer prompt rows; 64-key tiles from 0, chunk-free bits.
extern "C" __global__ void __launch_bounds__(128) mimo_attn_prompt(const __half* __restrict__ qo,
                                                                   const __half* __restrict__ kc,
                                                                   const __half* __restrict__ vc,
                                                                   const int32_t* __restrict__ pos,
                                                                   const int32_t* __restrict__ slot,
                                                                   __half* __restrict__ out, int n, int LEN, int H,
                                                                   int HK, float scale) {
    using namespace tc;
    extern __shared__ __align__(16) __half sm[];
    const int r0 = blockIdx.x * BM, h = blockIdx.y, kh = h / (H / HK);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int rows = min(BM, n - r0);
    // the queries staged in stage 0's key rows, then a warp's 16 rows as A fragments (12 steps of 16 dims)
    for (int i = threadIdx.x; i < BM * (QK / 8); i += THREADS) {
        const int r = i / (QK / 8), w = i % (QK / 8);
        cp16(sm + r * KS + w * 8, qo + ((int64_t)(r0 + min(r, rows - 1)) * H + h) * QK + w * 8, r < rows);
    }
    commit();
    wait<0>();
    __syncthreads();
    uint32_t qa[QK / 16][4];
#pragma unroll
    for (int k = 0; k < QK / 16; ++k) ldsm4(qa[k], sm + (warp * 16 + (lane & 15)) * KS + k * 16 + (lane >> 4) * 8);
    const int ra = warp * 16 + g, rb = ra + 8;  // this lane's two rows of the warp's 16
    const int pa = ra < rows ? pos[r0 + ra] : -1, pb = rb < rows ? pos[r0 + rb] : -1;
    int last = -1;
    for (int r = 0; r < rows; ++r) last = max(last, pos[r0 + r]);
    const int end = last + 1;
    const int64_t base = (int64_t)slot[r0] * LEN;
    float o[VD / 8][4];
#pragma unroll
    for (int d = 0; d < VD / 8; ++d) o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.f;
    float ma = NEG_INF, mb = NEG_INF, la = 0.f, lb = 0.f;
    __syncthreads();  // the queries are read: stage 0 takes keys
    const int tiles = (end + BN - 1) / BN;
    if (tiles > 0) load_kv(sm, kc, vc, base, LEN, HK, kh, 0, end);
    commit();
    for (int j = 0; j < tiles; ++j) {
        const int t0 = j * BN;
        if (j + 1 < tiles) load_kv(sm + ((j + 1) & 1) * STAGE, kc, vc, base, LEN, HK, kh, t0 + BN, end);
        commit();
        wait<1>();
        __syncthreads();
        const __half* ks = sm + (j & 1) * STAGE;
        const __half* vs = ks + BN * KS;
        float sc[BN / 8][4];  // q . k over 192 dims: 8 blocks of 8 keys, two a load
#pragma unroll
        for (int nb = 0; nb < BN / 8; ++nb) sc[nb][0] = sc[nb][1] = sc[nb][2] = sc[nb][3] = 0.f;
#pragma unroll
        for (int k = 0; k < QK / 16; ++k)
#pragma unroll
            for (int nb = 0; nb < BN / 8; nb += 2) {
                uint32_t b[4];
                ldsm4(b, ks + (nb * 8 + (lane & 7) + ((lane >> 4) << 3)) * KS + k * 16 + ((lane >> 3) & 1) * 8);
                mma(sc[nb], qa[k], b[0], b[1]);
                mma(sc[nb + 1], qa[k], b[2], b[3]);
            }
        float xa = NEG_INF, xb = NEG_INF;  // scaled, keys past a row's position masked; the rows' maxima
#pragma unroll
        for (int nb = 0; nb < BN / 8; ++nb)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int key = t0 + nb * 8 + 2 * t + e;
                sc[nb][e] = key <= pa ? sc[nb][e] * scale : NEG_INF;
                sc[nb][2 + e] = key <= pb ? sc[nb][2 + e] * scale : NEG_INF;
                xa = fmaxf(xa, sc[nb][e]);
                xb = fmaxf(xb, sc[nb][2 + e]);
            }
#pragma unroll
        for (int x = 1; x < 4; x <<= 1) {
            xa = fmaxf(xa, __shfl_xor_sync(0xffffffffu, xa, x));
            xb = fmaxf(xb, __shfl_xor_sync(0xffffffffu, xb, x));
        }
        const float na = fmaxf(ma, xa), nbm = fmaxf(mb, xb);  // a row with no key here keeps its state (corr 1, p 0)
        const float ca = na == NEG_INF ? 1.f : expf(ma - na), cb = nbm == NEG_INF ? 1.f : expf(mb - nbm);
        float sa = 0.f, sb = 0.f;
#pragma unroll
        for (int nb = 0; nb < BN / 8; ++nb)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                sc[nb][e] = sc[nb][e] == NEG_INF ? 0.f : expf(sc[nb][e] - na);
                sc[nb][2 + e] = sc[nb][2 + e] == NEG_INF ? 0.f : expf(sc[nb][2 + e] - nbm);
                sa += sc[nb][e];
                sb += sc[nb][2 + e];
            }
#pragma unroll
        for (int x = 1; x < 4; x <<= 1) {
            sa += __shfl_xor_sync(0xffffffffu, sa, x);
            sb += __shfl_xor_sync(0xffffffffu, sb, x);
        }
        la = la * ca + sa;
        lb = lb * cb + sb;
        ma = na;
        mb = nbm;
#pragma unroll
        for (int d = 0; d < VD / 8; ++d) {
            o[d][0] *= ca;
            o[d][1] *= ca;
            o[d][2] *= cb;
            o[d][3] *= cb;
        }
#pragma unroll
        for (int k = 0; k < BN / 16; ++k) {  // o += p v: p in fp16 as A fragments of 16 keys, v transposed
            const uint32_t pk[4] = {pack(sc[2 * k][0], sc[2 * k][1]), pack(sc[2 * k][2], sc[2 * k][3]),
                                    pack(sc[2 * k + 1][0], sc[2 * k + 1][1]), pack(sc[2 * k + 1][2], sc[2 * k + 1][3])};
#pragma unroll
            for (int d = 0; d < VD / 8; d += 2) {
                uint32_t b[4];
                ldsm4t(b, vs + (k * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * VS + d * 8 + (lane >> 4) * 8);
                mma(o[d], pk, b[0], b[1]);
                mma(o[d + 1], pk, b[2], b[3]);
            }
        }
        __syncthreads();  // the stage is read: the next load may take it
    }
#pragma unroll
    for (int d = 0; d < VD / 8; ++d) {  // o / l to fp16: row ra's dims 8 d + 2 t, + 1 (c0, c1), row rb's (c2, c3)
        if (pa >= 0)
            *(__half2*)(out + ((int64_t)(r0 + ra) * H + h) * VD + d * 8 + 2 * t) = __floats2half2_rn(o[d][0] / la, o[d][1] / la);
        if (pb >= 0)
            *(__half2*)(out + ((int64_t)(r0 + rb) * H + h) * VD + d * 8 + 2 * t) = __floats2half2_rn(o[d][2] / lb, o[d][3] / lb);
    }
}
