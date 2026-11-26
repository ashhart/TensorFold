// Cooperative (`R = 3..16`), vectorized/tiled prefill, and 1-pass fused prefill bodies for `hy4_ihc.cuh`.
#pragma once

namespace hy4ihc {

// Arm B: 48-CTA cooperative launch with grid barriers between rank-sum/ssy, mix partials, and collapse/sso.
template <bool POST, int M>
__device__ __forceinline__ void coop_body(const Args& a) {
    namespace cg = cooperative_groups;
    static_assert(MAXR * 32 <= 2 * NT && MAXR * (H / 8) <= NU * NT, "the P2 / P4 register prefetch covers R <= MAXR");
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) float fs[M * 256];
    __shared__ __align__(16) float xs[MAXR * 4 * 64];
    __shared__ float csw[8][KV * NCH];
    __shared__ float rinv_s[MAXR];
    __shared__ __align__(16) bf16 xpre[MAXR * 4 * 64];
    __shared__ __align__(16) bf16 pnw_s[64];
    __shared__ __align__(16) float pp_s[MAXR * 4];
    __shared__ __align__(16) bf16 nnw_s[H];
    __shared__ __align__(16) float bs_s[16];
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    const int b = blockIdx.x, cj = b >> 3, w = b & 7;
    const int R = a.R;
    const int gw = wid * NU + b;
    IHC_TR(0);
    {
        for (int i = t; i < M * 4 * 2 * 8; i += NT) {
            const int q = i & 7, h = (i >> 3) & 1, s = (i >> 4) & 3, m = i >> 6;
            cp16(fs + ((m * 4 + s) * 2 + h) * 32 + q * 4,
                 a.fn + static_cast<size_t>(m) * NSH + s * H + cj * 512 + h * 256 + w * 32 + q * 4);
        }
        for (int i = t; i < R * 32; i += NT) {
            const int quad = i & 3, h = (i >> 2) & 1, s = (i >> 3) & 3, r = i >> 5;
            cp16(xpre + ((r * 4 + s) * 2 + h) * 32 + quad * 8,
                 a.X + (static_cast<size_t>(r) * NS + s) * H + cj * 512 + h * 256 + w * 32 + quad * 8);
        }
        for (int i = t; i < H / 8; i += NT) cp16(nnw_s + i * 8, a.next_nw + i * 8);
        if (t < M) cp4(bs_s + t, a.base + t);
        else if (t >= 32 && t < 32 + (M == 8 ? 2 : 1)) cp4(bs_s + 8 + (t - 32), a.scale + (t - 32));
        if (POST) {
            if (t >= 64 && t < 72) {
                const int i = t - 64, quad = i & 3, h = i >> 2;
                cp16(pnw_s + h * 32 + quad * 8, a.post_nw + cj * 512 + h * 256 + w * 32 + quad * 8);
            }
            if (t >= 128 && t < 128 + R) cp16(pp_s + (t - 128) * 4, a.post_prev + static_cast<size_t>(t - 128) * 4);
        }
        cp_commit();
    }
    if constexpr (POST) {
        for (int q = gw; q < 8 * R; q += NU * 8) {
            const int r = q >> 3, wq = q & 7;
            const float* g0 = a.G + static_cast<size_t>(r) * H + wq * 32 + lane;
            float y[12];
#pragma unroll
            for (int j = 0; j < 12; ++j) y[j] = __ldg(g0 + 256 * j);
            for (int k = 1; k < a.world; ++k) {
                const float* gk = g0 + static_cast<size_t>(k) * a.gs;
#pragma unroll
                for (int j = 0; j < 12; ++j) y[j] = __fadd_rn(y[j], __ldg(gk + 256 * j));
            }
            float ss = 0.f;
            float* yo = a.yb + static_cast<size_t>(r) * H + wq * 32 + lane;
#pragma unroll
            for (int j = 0; j < 12; ++j) {
                const float yv = a.round_in ? rbf(y[j]) : y[j];
                ss = __fmaf_rn(yv, yv, ss);
                yo[256 * j] = yv;
            }
            ss = warp_sum(ss);
            if (lane == 0) a.ssy[r * 8 + wq] = ss;
        }
        IHC_TR(1);
        grid.sync();
        IHC_TR(2);
    }
    float4 yr0[2], yr1[2];
    if constexpr (POST) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int q = t + NT * i;
            if (q < 32 * R) {
                const int quad = q & 3, h = (q >> 2) & 1, r = q >> 5;
                const float4* yp = reinterpret_cast<const float4*>(a.yb + static_cast<size_t>(r) * H + cj * 512 + h * 256 + w * 32 + quad * 8);
                yr0[i] = __ldcg(yp);
                yr1[i] = __ldcg(yp + 1);
            }
        }
        if (t < R) {
            float tot = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) tot = __fadd_rn(tot, __ldcg(a.ssy + t * 8 + k));
            rinv_s[t] = rinv_of(tot, H, a.eps);
        }
    }
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int q = t + NT * i;
        if (q >= 32 * R) break;
        const int quad = q & 3, h = (q >> 2) & 1, s = (q >> 3) & 3, r = q >> 5;
        const int off = ((r * 4 + s) * 2 + h) * 32 + quad * 8;
        const int c = cj * 512 + h * 256 + w * 32 + quad * 8;
        float x[8];
        unpack8(*reinterpret_cast<const uint4*>(xpre + off), x);
        if constexpr (POST) {
            const float rinv = rinv_s[r];
            const float p = pp_s[r * 4 + s];
            float wn[8];
            unpack8(*reinterpret_cast<const uint4*>(pnw_s + h * 32 + quad * 8), wn);
            const float yv[8] = {yr0[i].x, yr0[i].y, yr0[i].z, yr0[i].w, yr1[i].x, yr1[i].y, yr1[i].z, yr1[i].w};
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const float z = rbf(__fmul_rn(__fmul_rn(yv[k], rinv), wn[k]));
                x[k] = rbf(__fadd_rn(__fmul_rn(p, z), x[k]));
            }
            *reinterpret_cast<uint4*>(a.X + (static_cast<size_t>(r) * NS + s) * H + c) = pack8(x);
        }
        reinterpret_cast<float4*>(xs + off)[0] = make_float4(x[0], x[1], x[2], x[3]);
        reinterpret_cast<float4*>(xs + off)[1] = make_float4(x[4], x[5], x[6], x[7]);
    }
    __syncthreads();
    for (int p = wid; p < 4 * R; p += 16) {
        const int p1 = p + 8;
        const bool has1 = p1 < 4 * R;
        const int s0 = p & 3, r0 = p >> 2, s1 = p1 & 3, r1 = has1 ? p1 >> 2 : r0;
        mix_pair<M>(fs + s0 * 64, xs + ((r0 * 4 + s0) * 2) * 32, a.wp + wp_idx(r0, 0, s0 * 6 + cj, w), has1, fs + s1 * 64,
                    xs + ((r1 * 4 + s1) * 2) * 32, a.wp + wp_idx(r1, 0, s1 * 6 + cj, w));
    }
    IHC_TR(3);
    grid.sync();
    IHC_TR(4);
    constexpr int KVN = M + 1;
    for (int q = gw; q < 8 * R; q += NU * 8) {
        const int r = q >> 3, wq = q & 7;
        float4 wa[7], wb[7];
#pragma unroll
        for (int n = 0; n < 7; ++n) {
            const int i = lane + 32 * n;
            if (i < KVN * NCH) {
                const int kk = i / NCH, j = i % NCH;
                const float4* src = reinterpret_cast<const float4*>(a.wp + wp_idx(r, kk < M ? kk : 8, j, 0));
                wa[n] = __ldcg(src);
                wb[n] = __ldcg(src + 1);
            }
        }
        const unsigned short* xr = reinterpret_cast<const unsigned short*>(a.X) + static_cast<size_t>(r) * NSH;
        unsigned short xv[12][4];
#pragma unroll
        for (int j = 0; j < 12; ++j)
#pragma unroll
            for (int s = 0; s < 4; ++s) xv[j][s] = __ldcg(xr + s * H + wq * 32 + lane + 256 * j);
        float* cs = csw[wid];
#pragma unroll
        for (int n = 0; n < 7; ++n) {
            const int i = lane + 32 * n;
            if (i < KVN * NCH) cs[i] = chunk_sum(wa[n], wb[n]);
        }
        __syncwarp();
        float tot = 0.f;
        if (lane < KVN) {
#pragma unroll
            for (int j = 0; j < NCH; ++j) tot = __fadd_rn(tot, cs[lane * NCH + j]);
        }
        __syncwarp();
        const float rinv = rinv_of(__shfl_sync(0xffffffffu, tot, M), NSH, a.eps);
        const int e = lane & 7;
        const float te = __shfl_sync(0xffffffffu, tot, e < M ? e : 0);
        const float cf = hc_coef(te, rinv, bs_s[(M == 8 && e >= 4) ? 9 : 8], bs_s[e < M ? e : 0], e >= 4, a.mag, a.hc_eps);
        float pre[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) pre[s] = __shfl_sync(0xffffffffu, cf, s);
        if constexpr (M == 8) {
            if (wq == 0 && lane >= 4 && lane < 8) a.post_next[r * 4 + (lane - 4)] = cf;
        }
        if (a.dbg_tot != nullptr && wq == 0) {
            const float tk = __shfl_sync(0xffffffffu, tot, lane < M ? lane : M);
            if (lane < M || lane == 8) a.dbg_tot[r * 9 + lane] = tk;
            else if (lane < 9) a.dbg_tot[r * 9 + lane] = 0.f;
        }
        bf16* oo = a.ob + static_cast<size_t>(r) * H;
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 12; ++j) {
            float v = 0.f;
#pragma unroll
            for (int s = 0; s < 4; ++s)
                v = __fadd_rn(v, __fmul_rn(pre[s], __uint_as_float(static_cast<uint32_t>(xv[j][s]) << 16)));
            const float o = rbf(v);
            oo[wq * 32 + lane + 256 * j] = tb(o);
            ss = __fmaf_rn(o, o, ss);
        }
        ss = warp_sum(ss);
        if (lane == 0) a.sso[r * 8 + wq] = ss;
    }
    IHC_TR(5);
    grid.sync();
    IHC_TR(6);
    uint4 orr[1];
    {
        const int q = b * NT + t;
        if (q < R * (H / 8)) orr[0] = ldcg4(a.ob + static_cast<size_t>(q / (H / 8)) * H + (q % (H / 8)) * 8);
    }
    if (t < R) {
        float tot = 0.f;
#pragma unroll
        for (int k = 0; k < 8; ++k) tot = __fadd_rn(tot, __ldcg(a.sso + t * 8 + k));
        rinv_s[t] = rinv_of(tot, H, a.eps);
    }
    __syncthreads();
    for (int q = b * NT + t; q < R * (H / 8); q += NU * NT) {
        const int r = q / (H / 8), c = (q % (H / 8)) * 8;
        float o[8], wn[8];
        unpack8(orr[0], o);
        unpack8(*reinterpret_cast<const uint4*>(nnw_s + c), wn);
        const float rinv = rinv_s[r];
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(__fmul_rn(o[k], rinv), wn[k]);
        *reinterpret_cast<uint4*>(a.A + static_cast<size_t>(r) * H + c) = pack8(o);
    }
    IHC_TR(7);
}


// Prefill iHC pre (mixes + collapse + next norm) over `RB` rows per CTA via 3-stage `cp.async` ring.
template <int M, int RB>
__host__ __device__ constexpr int pre_tiled_smem_bytes() {
    return 3 * RB * 512 * 2 + 8 * RB * KV * 4 + RB * 9 * 4 + RB * 4 * 4 + 64 * 4;
}

template <int M, int RB>
__device__ __forceinline__ void pre_tiled_body(const Args& a) {
    static_assert(M == 8 || M == 4, "M");
    constexpr int STG = 3;
    constexpr int RPV = 16 / M;
    static_assert(RB % RPV == 0 && (RB == 2 || RB == 4 || RB == 8 || RB == 16), "RB");
    constexpr int LOGRB = RB == 2 ? 1 : RB == 4 ? 2 : RB == 8 ? 3 : 4;
    constexpr int KVN = M + 1;
    constexpr int OSR = (STG * RB * 512 * 2) / (H * 2);
    static_assert(OSR >= 1 && OSR <= 4, "OSR");
    extern __shared__ __align__(16) unsigned char dsm[];
    bf16* ring = reinterpret_cast<bf16*>(dsm);
    float* wps = reinterpret_cast<float*>(dsm + STG * RB * 512 * 2);
    float* tots = wps + 8 * RB * KV;
    float* coef = tots + RB * 9;
    float* red = coef + RB * 4;
    float* rinvo = red + 32;
    bf16* os = ring;
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    const int r0 = blockIdx.x * RB;
    const int nr = min(RB, a.R - r0);
    if (nr <= 0) return;
    const bf16* xbase = a.X + static_cast<size_t>(r0) * NSH;
    auto issue = [&](int j) {
        if (j < NCH) {
            bf16* dst = ring + (j % STG) * RB * 512;
            for (int i = t; i < RB * 64; i += NT) {
                const int r = i >> 6, q = i & 63;
                const bool live = r < nr;
                cp16z(dst + r * 512 + q * 8, xbase + static_cast<size_t>(live ? r : 0) * NSH + j * 512 + q * 8, live);
            }
        }
        cp_commit();
    };
    issue(0);
    issue(1);
    float tot = 0.f;
    for (int j = 0; j < NCH; ++j) {
        float f0[M], f1[M];
        const float* fj = a.fn + j * 512 + t;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            f0[m] = __ldg(fj + static_cast<size_t>(m) * NSH);
            f1[m] = __ldg(fj + static_cast<size_t>(m) * NSH + 256);
        }
        cp_wait<1>();
        __syncthreads();
        issue(j + 2);
        const bf16* xt = ring + (j % STG) * RB * 512;
        float ssv[RB];
#pragma unroll
        for (int rg = 0; rg < RB / RPV; ++rg) {
            float v[16];
#pragma unroll
            for (int i = 0; i < RPV; ++i) {
                const int r = rg * RPV + i;
                const float x0 = bfv(xt[r * 512 + t]), x1 = bfv(xt[r * 512 + 256 + t]);
#pragma unroll
                for (int m = 0; m < M; ++m) v[i * M + m] = __fmaf_rn(f1[m], x1, __fmaf_rn(f0[m], x0, 0.f));
                ssv[r] = __fmaf_rn(x1, x1, __fmaf_rn(x0, x0, 0.f));
            }
            const float x = rs_butterfly<4>(v);
            const int k = lane >> 1;
            if (!(lane & 1)) wps[(wid * RB + rg * RPV + k / M) * KV + (k % M)] = x;
        }
        {
            const float x = rs_butterfly<LOGRB>(ssv);
            if ((lane & ((32 >> LOGRB) - 1)) == 0) wps[(wid * RB + (lane >> (5 - LOGRB))) * KV + 8] = x;
        }
        __syncthreads();
        if (t < RB * KVN) {
            const int r = t / KVN, kk = t % KVN, k = kk < M ? kk : 8;
            float c = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) c = __fadd_rn(c, wps[(w * RB + r) * KV + k]);
            tot = __fadd_rn(tot, c);
        }
    }
    cp_wait<0>();
    if (t < RB * KVN) tots[(t / KVN) * 9 + t % KVN] = tot;
    __syncthreads();
    if (t < nr) {
        const float* tt = tots + t * 9;
        const float rinv = rinv_of(tt[M], NSH, a.eps);
        const size_t row = r0 + t;
#pragma unroll
        for (int s = 0; s < 4; ++s) coef[t * 4 + s] = pre_coef(tt[s], rinv, a.scale[0], a.base[s], a.hc_eps);
        if constexpr (M == 8) {
#pragma unroll
            for (int s = 0; s < 4; ++s)
                a.post_next[row * 4 + s] = post_coef(tt[4 + s], rinv, a.scale[1], a.base[4 + s], a.mag, a.hc_eps);
        }
        if (a.dbg_tot != nullptr)
            for (int k = 0; k < 9; ++k) a.dbg_tot[row * 9 + k] = k < M ? tt[k] : (k == 8 ? tt[M] : 0.f);
    }
    __syncthreads();
    for (int rb = 0; rb < nr; rb += OSR) {
        const int nb = min(OSR, nr - rb);
        for (int q = t; q < nb * (H / 8); q += NT) {
            const int r = q / (H / 8), c = (q % (H / 8)) * 8;
            const bf16* xp = xbase + static_cast<size_t>(rb + r) * NSH + c;
            float x[4][8];
#pragma unroll
            for (int s = 0; s < 4; ++s) unpack8(__ldg(reinterpret_cast<const uint4*>(xp + s * H)), x[s]);
            const float* cf = coef + (rb + r) * 4;
            float o[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                float v = 0.f;
#pragma unroll
            for (int s = 0; s < 4; ++s) v = __fadd_rn(v, __fmul_rn(cf[s], x[s][k]));
                o[k] = v;
            }
            *reinterpret_cast<uint4*>(os + r * H + c) = pack8(o);
        }
        __syncthreads();
        float sso[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            sso[r] = 0.f;
            if (r < nb) {
#pragma unroll
                for (int jj = 0; jj < 12; ++jj) {
                    const float o = bfv(os[r * H + t + 256 * jj]);
                    sso[r] = __fmaf_rn(o, o, sso[r]);
                }
            }
        }
        const float x = rs_butterfly<2>(sso);
        if ((lane & 7) == 0) red[(lane >> 3) * 8 + wid] = x;
        __syncthreads();
        if (t < nb) {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) s = __fadd_rn(s, red[t * 8 + w]);
            rinvo[t] = rinv_of(s, H, a.eps);
        }
        __syncthreads();
        for (int q = t; q < nb * (H / 8); q += NT) {
            const int r = q / (H / 8), c = (q % (H / 8)) * 8;
            float o[8], wn[8];
            unpack8(*reinterpret_cast<const uint4*>(os + r * H + c), o);
            unpack8(__ldg(reinterpret_cast<const uint4*>(a.next_nw + c)), wn);
            const float rinv = rinvo[r];
#pragma unroll
            for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(__fmul_rn(o[k], rinv), wn[k]);
            *reinterpret_cast<uint4*>(a.A + static_cast<size_t>(r0 + rb + r) * H + c) = pack8(o);
        }
        __syncthreads();
    }
}

// Prefill (`R > 64`): 1-pass fused post + pre/head (1 row per CTA, 0-bit equal to `post_vec` + `pre_tiled`).
template <int M>
__device__ __forceinline__ void fused_prefill_body(const Args& a) {
    static_assert(M == 8 || M == 4, "M");
    constexpr int KVN = M + 1;
    __shared__ __align__(16) float ysm[H];
    __shared__ __align__(16) bf16 os[H];
    __shared__ float wps[8 * KV];
    __shared__ float red[8];
    __shared__ float tots[9];
    __shared__ float coef[4];
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5, r = blockIdx.x;
    if (r >= a.R) return;

    const float* g0 = a.G + static_cast<size_t>(r) * H;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        const int q = t + NT * i;
        float4 v = __ldg(reinterpret_cast<const float4*>(g0) + q);
        for (int k = 1; k < a.world; ++k) {
            const float4 u = __ldg(reinterpret_cast<const float4*>(g0 + static_cast<size_t>(k) * a.gs) + q);
            v.x = __fadd_rn(v.x, u.x);
            v.y = __fadd_rn(v.y, u.y);
            v.z = __fadd_rn(v.z, u.z);
            v.w = __fadd_rn(v.w, u.w);
        }
        reinterpret_cast<float4*>(ysm)[q] =
            a.round_in ? make_float4(rbf(v.x), rbf(v.y), rbf(v.z), rbf(v.w)) : v;
    }
    __syncthreads();
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 12; ++j) {
        const float y = ysm[t + 256 * j];
        ss = __fmaf_rn(y, y, ss);
    }
    ss = warp_sum(ss);
    if (lane == 0) red[wid] = ss;
    __syncthreads();
    float tot_y = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) tot_y = __fadd_rn(tot_y, red[w]);
    const float rinv_y = rinv_of(tot_y, H, a.eps);

    for (int c = t; c < H; c += NT) {
        const float wn = bfv(__ldg(a.post_nw + c));
        ysm[c] = rbf(__fmul_rn(__fmul_rn(ysm[c], rinv_y), wn));
    }
    __syncthreads();

    float p[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) p[s] = a.post_prev[static_cast<size_t>(r) * 4 + s];
    bf16* xr = a.X + static_cast<size_t>(r) * NSH;

    float tot = 0.f;
    for (int j = 0; j < NCH; ++j) {
        const int s = j / 6;
        const int c0 = (j % 6) * 512 + t;
        const int c1 = c0 + 256;
        const float ps = p[s];
        const float x0_new = rbf(__fadd_rn(__fmul_rn(ps, ysm[c0]), bfv(xr[s * H + c0])));
        const float x1_new = rbf(__fadd_rn(__fmul_rn(ps, ysm[c1]), bfv(xr[s * H + c1])));
        xr[s * H + c0] = tb(x0_new);
        xr[s * H + c1] = tb(x1_new);

        float v[M];
        const float* fj = a.fn + j * 512 + t;
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float f0 = __ldg(fj + static_cast<size_t>(m) * NSH);
            const float f1 = __ldg(fj + static_cast<size_t>(m) * NSH + 256);
            v[m] = __fmaf_rn(f1, x1_new, __fmaf_rn(f0, x0_new, 0.f));
        }
        const float ssv = __fmaf_rn(x1_new, x1_new, __fmaf_rn(x0_new, x0_new, 0.f));
        if constexpr (M == 8) {
            const float x_m = rs_butterfly<3>(v);
            if ((lane & 3) == 0) wps[wid * KV + (lane >> 2)] = x_m;
        } else {
            const float x_m = rs_butterfly<2>(v);
            if ((lane & 7) == 0) wps[wid * KV + (lane >> 3)] = x_m;
        }
        const float x_s = warp_sum(ssv);
        if (lane == 0) wps[wid * KV + 8] = x_s;
        __syncthreads();
        if (t < KVN) {
            const int k = t < M ? t : 8;
            float c = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) c = __fadd_rn(c, wps[w * KV + k]);
            tot = __fadd_rn(tot, c);
        }
        __syncthreads();
    }
    if (t < KVN) tots[t] = tot;
    __syncthreads();
    if (t == 0) {
        const float rinv = rinv_of(tots[M], NSH, a.eps);
#pragma unroll
        for (int s = 0; s < 4; ++s) coef[s] = pre_coef(tots[s], rinv, a.scale[0], a.base[s], a.hc_eps);
        if constexpr (M == 8) {
#pragma unroll
            for (int s = 0; s < 4; ++s)
                a.post_next[static_cast<size_t>(r) * 4 + s] =
                    post_coef(tots[4 + s], rinv, a.scale[1], a.base[4 + s], a.mag, a.hc_eps);
        }
        if (a.dbg_tot != nullptr)
            for (int k = 0; k < 9; ++k) a.dbg_tot[static_cast<size_t>(r) * 9 + k] = k < M ? tots[k] : (k == 8 ? tots[M] : 0.f);
    }
    __syncthreads();

    float cf[4];
#pragma unroll
    for (int s = 0; s < 4; ++s) cf[s] = coef[s];
    for (int q = t; q < H / 8; q += NT) {
        const int c = q * 8;
        float x[4][8];
#pragma unroll
        for (int s = 0; s < 4; ++s) unpack8(__ldg(reinterpret_cast<const uint4*>(xr + s * H + c)), x[s]);
        float o[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = 0.f;
#pragma unroll
            for (int s = 0; s < 4; ++s) v = __fadd_rn(v, __fmul_rn(cf[s], x[s][k]));
            o[k] = v;
        }
        *reinterpret_cast<uint4*>(os + c) = pack8(o);
    }
    __syncthreads();
    float sso = 0.f;
#pragma unroll
    for (int jj = 0; jj < 12; ++jj) {
        const float o = bfv(os[t + 256 * jj]);
        sso = __fmaf_rn(o, o, sso);
    }
    sso = warp_sum(sso);
    if (lane == 0) red[wid] = sso;
    __syncthreads();
    float so = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) so = __fadd_rn(so, red[w]);
    const float rinvo = rinv_of(so, H, a.eps);
    for (int q = t; q < H / 8; q += NT) {
        const int c = q * 8;
        float o[8], wn[8];
        unpack8(*reinterpret_cast<const uint4*>(os + c), o);
        unpack8(__ldg(reinterpret_cast<const uint4*>(a.next_nw + c)), wn);
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(__fmul_rn(o[k], rinvo), wn[k]);
        *reinterpret_cast<uint4*>(a.A + static_cast<size_t>(r) * H + c) = pack8(o);
    }
}

}  // namespace hy4ihc
