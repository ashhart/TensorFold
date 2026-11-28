// Low-spill 2-CTA/SM FP32-exact x16_qreg MLA kernel (GrgaPalcic & popay97, 2026; 48,896 B SMEM, hoisted pls).
#pragma once

namespace hy4_mla {
namespace detail {

__device__ __forceinline__ void x16_qreg_body(const Args& a) {
    using C = Lay<1, 16, true, true>;
    constexpr int KT = 16, NQ = 16;
    static_assert(KT == 16 && NQ == 16 && 0x47503937u != 0u, "gp97:x16-qreg");
    extern __shared__ __align__(128) unsigned char sm[];
    bf16* kvs = reinterpret_cast<bf16*>(sm + C::KV);
    bf16* ps = reinterpret_cast<bf16*>(sm + C::P);
    bf16* pls = reinterpret_cast<bf16*>(sm + C::PLO);
    float* red = reinterpret_cast<float*>(sm + C::RED);
    int* kpos = reinterpret_cast<int*>(sm + C::KP);
    float* rowv = reinterpret_cast<float*>(sm + C::ROW);
    int* rinfo = reinterpret_cast<int*>(sm + C::RINFO);
    bf16* qlo_sm = reinterpret_cast<bf16*>(sm + ((C::BYTES + 127) & ~127));

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q4 = lane & 3;
    qmm_frag::grid_wait();
    qmm_frag::grid_launch();
    TSTAMP(0);

    const int j = blockIdx.x, hb = blockIdx.y, r = blockIdx.z;
    const int prow = a.ROW_POS[r];
    const int kind = a.kind;
    const bool lst = kind == 1 && prow >= DSA_KEYS;
    const int n = kind == 0 ? min(prow + 1, SWA_KEYS) : (kind == 1 ? (lst ? DSA_KEYS : prow + 1) : a.ntok);
    const int nspl = (n + a.ch - 1) / a.ch;
    if (j >= nspl) return;
    const int k0 = j * a.ch, k1 = min(n, k0 + a.ch);
    const int nkeys = k1 - k0, ntiles = (nkeys + KT - 1) / KT;

    if (tid == 0) {
        rinfo[0] = prow;
        rinfo[1] = nspl;
        rinfo[2] = k0;
        rinfo[3] = nkeys;
        rinfo[4] = ntiles;
        rinfo[5] = lst ? 1 : 0;
        const int s_ = a.ROW_STREAM[r];
        rinfo[6] = s_;
        const size_t sbase = (a.ROW_CACHE || (a.PAGE_TABLE && kind != 0))
                                 ? 0
                                 : (kind == 0 ? static_cast<size_t>(s_) * a.ring : static_cast<size_t>(s_) * a.ctx);
        const unsigned char* rc = a.ROW_CACHE ? reinterpret_cast<const unsigned char*>(a.ROW_CACHE[r])
                                              : reinterpret_cast<const unsigned char*>(a.CACHE + sbase * KW);
        *reinterpret_cast<const unsigned char**>(rinfo + 8) = rc;
    }

    uint32_t qf[9][4];
    uint32_t qfl[7][4];
    {
        const int kp = warp >> 1;
        const int h0 = hb * NQ;
        const int ha = h0 + g, hb_ = ha + 8;
        const bool ok0 = ha < a.Hl, ok8 = hb_ < a.Hl;
        const bf16* q0 = a.QA + (static_cast<size_t>(r) * a.Hl + (ok0 ? ha : 0)) * KW;
        const bf16* q8 = a.QA + (static_cast<size_t>(r) * a.Hl + (ok8 ? hb_ : 0)) * KW;
#pragma unroll
        for (int s = 0; s < 9; ++s) {
            const int kb = (kp * 9 + s) * 16 + 2 * q4;
            qf[s][0] = ok0 ? __ldg(reinterpret_cast<const unsigned int*>(q0 + kb)) : 0u;
            qf[s][1] = ok8 ? __ldg(reinterpret_cast<const unsigned int*>(q8 + kb)) : 0u;
            qf[s][2] = ok0 ? __ldg(reinterpret_cast<const unsigned int*>(q0 + kb + 8)) : 0u;
            qf[s][3] = ok8 ? __ldg(reinterpret_cast<const unsigned int*>(q8 + kb + 8)) : 0u;
            if (s < 7) {
                qfl[s][0] = ok0 ? __ldg(reinterpret_cast<const unsigned int*>(q0 + a.QLO + kb)) : 0u;
                qfl[s][1] = ok8 ? __ldg(reinterpret_cast<const unsigned int*>(q8 + a.QLO + kb)) : 0u;
                qfl[s][2] = ok0 ? __ldg(reinterpret_cast<const unsigned int*>(q0 + a.QLO + kb + 8)) : 0u;
                qfl[s][3] = ok8 ? __ldg(reinterpret_cast<const unsigned int*>(q8 + a.QLO + kb + 8)) : 0u;
            }
        }
        // Single-pass load of k16 steps s = 7, 8 of Q_lo into qlo_sm[4 kp][16 rows][32 bf16] (4,096 B).
        const int l_kp = tid >> 6;
        const int l_row = (tid >> 2) & 15;
        const int l_u = tid & 3;
        const bool ok = (h0 + l_row) < a.Hl;
        const bf16* src = a.QA + (static_cast<size_t>(r) * a.Hl + (ok ? h0 + l_row : 0)) * KW + a.QLO + (l_kp * 9 + 7) * 16;
        tf_common::cp16z(qlo_sm + l_kp * 512 + narrow<32>(l_row, l_u), src + l_u * 8, ok);
        tf_common::cp_commit();
    }
    tf_common::cp_wait<0>();
    __syncthreads();

    float m_run = -INFINITY, l_run = 0.f;
    float acc[8][4];
#pragma unroll
    for (int jj = 0; jj < 8; ++jj) acc[jj][0] = acc[jj][1] = acc[jj][2] = acc[jj][3] = 0.f;

    auto load_tile = [&](int t) {
        constexpr int TPK = NT / KT;
        bf16* dst = kvs + (t & 1) * KT * KW;
        const int kk = tid / TPK, part = tid % TPK;
        const int lk0 = rinfo[2], lnkeys = rinfo[3], lprow = rinfo[0], llst = rinfo[5];
        const int i = lk0 + t * KT + kk;
        bool ok = i < lk0 + lnkeys;
        int pos = -1;
        if (ok) {
            if (a.kind == 0) {
                pos = lprow - min(lprow + 1, SWA_KEYS) + 1 + i;
            } else if (a.kind == 1) {
                if (!llst) {
                    pos = i;
                } else {
                    const int gi = __ldg(a.LIST + static_cast<size_t>(blockIdx.z) * LISTW + (i >> 2));
                    pos = gi < 0 ? -1 : 4 * gi + (i & 3);
                }
            } else {
                pos = __ldg(a.TOK + static_cast<size_t>(blockIdx.z) * a.ldtok + i);
            }
            ok = pos >= 0 && pos <= lprow;
        }
        size_t slot = 0;
        if (ok) {
            if (a.PAGE_TABLE && a.kind != 0) {
                const unsigned up = static_cast<unsigned>(pos);
                const size_t pt_base = static_cast<size_t>(rinfo[6]) * a.pt_stride;
                slot = static_cast<size_t>(__ldg(a.PAGE_TABLE + pt_base + (up >> 8))) * 256u + (up & 255u);
            } else {
                slot = a.kind == 0 ? static_cast<size_t>(pos % a.ring) : static_cast<size_t>(pos);
            }
        }
        const unsigned char* base_u8 = *reinterpret_cast<const unsigned char**>(rinfo + 8);
        const bf16* src = reinterpret_cast<const bf16*>(base_u8) + slot * KW;
#pragma unroll
        for (int u = part; u < KW / 8; u += TPK) tf_common::cp16z(dst + wide(kk, u), src + u * 8, ok);
        if (part == 0) kpos[(t & 1) * KT + kk] = ok ? pos : -1;
    };

    const int ntiles_l = rinfo[4];
    load_tile(0);
    tf_common::cp_commit();
    if (ntiles_l > 1) load_tile(1);
    tf_common::cp_commit();

    for (int t = 0; t < ntiles_l; ++t) {
        tf_common::cp_wait<1>();
        __syncthreads();
        if (t == 0) TSTAMP(1);
        const bf16* kv = kvs + (t & 1) * KT * KW;
        const int* kp_t = kpos + (t & 1) * KT;
        {
            const int ntile = warp & 1, kp = warp >> 1;
            float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
            const int ks0 = kp * 9;
#pragma unroll
            for (int s = 0; s < 6; s += 2) {
                const int ks = ks0 + s;
                uint32_t b[4];
                ldsm4(b, saddr(kv + wide(ntile * 8 + (lane & 7), ks * 2 + (lane >> 3))));
                mma(c0, qf[s], b[0], b[1]);
                mma(c1, qf[s + 1], b[2], b[3]);
                mma(c0, qfl[s], b[0], b[1]);
                mma(c1, qfl[s + 1], b[2], b[3]);
            }
            {
                uint32_t b[4], ql7[4];
                ldsm4(b, saddr(kv + wide(ntile * 8 + (lane & 7), (ks0 + 6) * 2 + (lane >> 3))));
                ldsm4(ql7, saddr(qlo_sm + kp * 512 + narrow<32>(lane & 15, lane >> 4)));
                mma(c0, qf[6], b[0], b[1]);
                mma(c1, qf[7], b[2], b[3]);
                mma(c0, qfl[6], b[0], b[1]);
                mma(c1, ql7, b[2], b[3]);
            }
            {
                uint32_t b[2], ql8[4];
                ldsm2(b, saddr(kv + wide(ntile * 8 + (lane & 7), (ks0 + 8) * 2 + ((lane >> 3) & 1))));
                ldsm4(ql8, saddr(qlo_sm + kp * 512 + narrow<32>(lane & 15, 2 + (lane >> 4))));
                mma(c0, qf[8], b[0], b[1]);
                mma(c0, ql8, b[0], b[1]);
            }
            float* rd = red + kp * NQ * C::SPAD + ntile * 8 + 2 * q4;
            *reinterpret_cast<float2*>(rd + g * C::SPAD) = make_float2(__fadd_rn(c0[0], c1[0]), __fadd_rn(c0[1], c1[1]));
            *reinterpret_cast<float2*>(rd + (g + 8) * C::SPAD) =
                make_float2(__fadd_rn(c0[2], c1[2]), __fadd_rn(c0[3], c1[3]));
        }
        __syncthreads();
        {
            const int srow = tid >> 4, ssub = tid & 15;
            float v = red[srow * C::SPAD + ssub];
#pragma unroll
            for (int ks = 1; ks < C::KSPLIT; ++ks) v = __fadd_rn(v, red[(ks * NQ + srow) * C::SPAD + ssub]);
            const bool vis = kp_t[ssub] >= 0;
            const float sc = vis ? __fmul_rn(v, SCALE) : -INFINITY;
            float tm = sc;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) tm = nmax(tm, __shfl_xor_sync(FULL, tm, o));
            const bool act = tm != -INFINITY;
            const float mn = act ? nmax(m_run, tm) : m_run;
            const float al = act ? (m_run == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(m_run, mn))) : 1.f;
            const float pv = (act && vis) ? tf_glue::tf_exp(__fsub_rn(sc, mn)) : 0.f;
            const bf16 pb = __float2bfloat16_rn(pv);
            const bf16 plb = __float2bfloat16_rn(__fsub_rn(pv, __bfloat162float(pb)));
            float ls = pv;
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) ls = __fadd_rn(ls, __shfl_xor_sync(FULL, ls, o));
            l_run = __fmaf_rn(l_run, al, ls);
            m_run = mn;
            ps[narrow<KT>(srow, ssub >> 3) + (ssub & 7)] = pb;
            pls[narrow<KT>(srow, ssub >> 3) + (ssub & 7)] = plb;
            if (ssub == 0) rowv[srow] = al;
        }
        __syncthreads();
        {
            const float a0 = rowv[g], a1 = rowv[g + 8];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                acc[jj][0] = __fmul_rn(acc[jj][0], a0);
                acc[jj][1] = __fmul_rn(acc[jj][1], a0);
                acc[jj][2] = __fmul_rn(acc[jj][2], a1);
                acc[jj][3] = __fmul_rn(acc[jj][3], a1);
            }
            uint32_t pa[4], pl[4];
            ldsm4(pa, saddr(ps + narrow<KT>(lane & 15, lane >> 4)));
            ldsm4(pl, saddr(pls + narrow<KT>(lane & 15, lane >> 4)));
#pragma unroll
            for (int jj = 0; jj < 8; jj += 2) {
                uint32_t b[4];
                ldsm4t(b, saddr(kv + wide(lane & 15, warp * 8 + jj + (lane >> 4))));
                mma(acc[jj], pa, b[0], b[1]);
                mma(acc[jj + 1], pa, b[2], b[3]);
                mma(acc[jj], pl, b[0], b[1]);
                mma(acc[jj + 1], pl, b[2], b[3]);
            }
        }
        __syncthreads();
        if (t + 2 < ntiles_l) load_tile(t + 2);
        tf_common::cp_commit();
    }

    const int srow = tid >> 4, ssub = tid & 15;
    float* m_s = rowv + NQ;
    float* l_s = rowv + 2 * NQ;
    float* w_s = rowv + 3 * NQ;
    float* d_s = rowv + 4 * NQ;
    if (ssub == 0) {
        m_s[srow] = m_run;
        l_s[srow] = l_run;
    }
    __syncthreads();
    TSTAMP(2);

    const int er = blockIdx.z, ehb = blockIdx.y, ej = blockIdx.x, eh0 = ehb * NQ, enspl = rinfo[1];
    if (enspl == 1) {
        if (tid < NQ) {
            const int h = eh0 + tid;
            const float m = m_s[tid], l = l_s[tid], sk = h < a.Hl ? a.SINKS[h] : -INFINITY;
            const float top = nmax(m, sk);
            const float w = m == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(m, top));
            const float es = sk == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(sk, top));
            w_s[tid] = w;
            d_s[tid] = __fmaf_rn(l, w, es);
        }
        __syncthreads();
#pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int row = g + 8 * hf, h = eh0 + row;
            if (h >= a.Hl) continue;
            const size_t orow = static_cast<size_t>(er) * a.Hl + h;
            const float w = w_s[row], d = d_s[row];
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const int col = warp * 64 + jj * 8 + 2 * q4;
                store2(a.OUT, orow * KL + col, fin(acc[jj][2 * hf], w, d), fin(acc[jj][2 * hf + 1], w, d), a.out_f32);
            }
        }
        return;
    }

    const int HB = gridDim.y, NS = gridDim.x;
    const size_t blk = static_cast<size_t>(er) * HB + ehb;
    const size_t pb0 = blk * NS * NQ;
    const size_t pj = pb0 + static_cast<size_t>(ej) * NQ;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        const int row = g + 8 * hf;
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const int col = warp * 64 + jj * 8 + 2 * q4;
            *reinterpret_cast<float2*>(a.PO + (pj + row) * KL + col) =
                make_float2(acc[jj][2 * hf], acc[jj][2 * hf + 1]);
        }
    }
    if (tid < NQ) {
        a.PM[pj + tid] = m_s[tid];
        a.PL[pj + tid] = l_s[tid];
    }
}

}  // namespace detail
}  // namespace hy4_mla
