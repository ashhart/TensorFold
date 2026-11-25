// Templated tensor-core tile body (`attn_body`) for `hy4_mla_attn.cu`.
#pragma once

namespace hy4_mla {
namespace detail {

// `ROWS == false`: heads-in-M (`16 * MT` heads); `ROWS == true`: rows-in-M (`16` rows, `MT` heads).
template <int MT, int KT, bool QREG, bool ROWS, bool EXACT = false>
__device__ __forceinline__ void attn_body(const Args& a) {
    using C = Lay<MT, KT, QREG, EXACT>;
    constexpr int NQ = C::NQ;
    extern __shared__ __align__(128) unsigned char sm[];
    bf16* qs = reinterpret_cast<bf16*>(sm + C::Q);
    bf16* kvs = reinterpret_cast<bf16*>(sm + C::KV);
    bf16* ps = reinterpret_cast<bf16*>(sm + C::P);
    bf16* pls = reinterpret_cast<bf16*>(sm + C::PLO);
    bf16* qls = reinterpret_cast<bf16*>(sm + C::QLO);
    float* red = reinterpret_cast<float*>(sm + C::RED);
    int* kpos = reinterpret_cast<int*>(sm + C::KP);
    float* rowv = reinterpret_cast<float*>(sm + C::ROW);
    int* rinfo = reinterpret_cast<int*>(sm + C::RINFO);
    int* misc = reinterpret_cast<int*>(sm + C::MISC);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q4 = lane & 3;
    const int kind = a.kind;
    qmm_frag::grid_wait();
    if constexpr (!ROWS) qmm_frag::grid_launch();
    TSTAMP(0);

    int r = 0, hb = 0, j = 0, nspl = 1, h0 = 0, prow = 0;
    int k0 = 0, k1 = 0;
    bool lst = false;
    size_t sbase = 0;
    const bf16* row_cache = a.CACHE;
    unsigned live = 0;
    if constexpr (!ROWS) {
        j = blockIdx.x;
        hb = blockIdx.y;
        r = blockIdx.z;
        const int s_ = a.ROW_STREAM[r];
        prow = a.ROW_POS[r];
        lst = kind == 1 && prow >= DSA_KEYS;
        const int n = kind == 0 ? min(prow + 1, SWA_KEYS) : (kind == 1 ? (lst ? DSA_KEYS : prow + 1) : a.ntok);
        nspl = (n + a.ch - 1) / a.ch;
        if (j >= nspl) return;
        k0 = j * a.ch;
        k1 = min(n, k0 + a.ch);
        if (a.PAGE_TABLE) {
            sbase = static_cast<size_t>(s_) * a.pt_stride;
        } else if (a.ROW_CACHE) {
            row_cache = reinterpret_cast<const bf16*>(a.ROW_CACHE[r]);
            sbase = 0;
        } else {
            sbase = kind == 0 ? static_cast<size_t>(s_) * a.ring : static_cast<size_t>(s_) * a.ctx;
        }
        h0 = hb * NQ;
    } else {
        hb = blockIdx.x;
        r = blockIdx.y * 16;
        h0 = hb * MT;
        if (tid < 16) {
            const int rr = r + tid;
            int pp = 0, st = -1;
            if (rr < a.R) {
                pp = a.ROW_POS[rr];
                st = a.ROW_STREAM[rr];
                if (kind == 1 && pp >= DSA_KEYS) st = -1;
            }
            rinfo[tid] = pp;
            rinfo[16 + tid] = st;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) live |= (rinfo[16 + i] >= 0 ? 1u : 0u) << i;
        if (live == 0) return;
    }

    // QREG keeps each warp's `(m tile, k slice)` A fragments in registers across all key tiles.
    uint32_t qf[QREG ? C::KSTEPS : 1][4];
    uint32_t qfl[(QREG && EXACT) ? C::KSTEPS : 1][4];
    if constexpr (QREG) {
        const int ntile = warp % C::NTILE, kp = warp / C::NTILE, mt = ntile / (KT / 8);
        const bf16* q0;
        const bf16* q8;
        bool ok0 = true, ok8 = true;
        if constexpr (!ROWS) {
            const int ha = h0 + mt * 16 + g;
            const int hb_ = ha + 8;
            ok0 = ha < a.Hl;
            ok8 = hb_ < a.Hl;
            q0 = a.QA + (static_cast<size_t>(r) * a.Hl + (ok0 ? ha : 0)) * KW;
            q8 = a.QA + (static_cast<size_t>(r) * a.Hl + (ok8 ? hb_ : 0)) * KW;
        } else {
            const bool hm_ok = (h0 + mt) < a.Hl;
            const int hm = h0 + (hm_ok ? mt : 0);
            ok0 = ((live >> g) & 1) && hm_ok;
            ok8 = ((live >> (g + 8)) & 1) && hm_ok;
            q0 = a.QA + (static_cast<size_t>(ok0 ? r + g : 0) * a.Hl + hm) * KW;
            q8 = a.QA + (static_cast<size_t>(ok8 ? r + g + 8 : 0) * a.Hl + hm) * KW;
        }
#pragma unroll
        for (int s = 0; s < C::KSTEPS; ++s) {
            const int kb = (kp * C::KSTEPS + s) * 16 + 2 * q4;
            qf[s][0] = ok0 ? __ldg(reinterpret_cast<const unsigned int*>(q0 + kb)) : 0u;
            qf[s][1] = ok8 ? __ldg(reinterpret_cast<const unsigned int*>(q8 + kb)) : 0u;
            qf[s][2] = ok0 ? __ldg(reinterpret_cast<const unsigned int*>(q0 + kb + 8)) : 0u;
            qf[s][3] = ok8 ? __ldg(reinterpret_cast<const unsigned int*>(q8 + kb + 8)) : 0u;
            if constexpr (EXACT) {
                qfl[s][0] = ok0 ? __ldg(reinterpret_cast<const unsigned int*>(q0 + a.QLO + kb)) : 0u;
                qfl[s][1] = ok8 ? __ldg(reinterpret_cast<const unsigned int*>(q8 + a.QLO + kb)) : 0u;
                qfl[s][2] = ok0 ? __ldg(reinterpret_cast<const unsigned int*>(q0 + a.QLO + kb + 8)) : 0u;
                qfl[s][3] = ok8 ? __ldg(reinterpret_cast<const unsigned int*>(q8 + a.QLO + kb + 8)) : 0u;
            }
        }
    } else {
        for (int idx = tid; idx < NQ * (KW / 8); idx += NT) {
            const int row = idx / (KW / 8), u = idx - row * (KW / 8);
            const bf16* src;
            bool ok = true;
            if constexpr (!ROWS) {
                ok = (h0 + row) < a.Hl;
                src = a.QA + (static_cast<size_t>(r) * a.Hl + (ok ? h0 + row : 0)) * KW;
            } else {
                const int r16 = row & 15, mt = row >> 4;
                const bool hm_ok = (h0 + mt) < a.Hl;
                ok = ((live >> r16) & 1) && hm_ok;
                src = a.QA + (static_cast<size_t>(ok ? r + r16 : 0) * a.Hl + h0 + (hm_ok ? mt : 0)) * KW;
            }
            tf_common::cp16z(qs + wide(row, u), src + u * 8, ok);
            if constexpr (EXACT) tf_common::cp16z(qls + wide(row, u), src + a.QLO + u * 8, ok);
        }
        tf_common::cp_commit();
    }

    const int srow = tid / C::TPR, ssub = tid % C::TPR;
    float m_run = -INFINITY, l_run = 0.f;
    float acc[MT][8][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int jj = 0; jj < 8; ++jj)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[mt][jj][e] = 0.f;
    int my_p = 0;
    if constexpr (ROWS) my_p = rinfo[srow & 15];

    auto run_pass = [&](int first, int nkeys, int sig, unsigned pass, int vis_lo, int vis_hi) {
        const int ntiles = (nkeys + KT - 1) / KT;
        auto load_tile = [&](int t) {
            constexpr int TPK = NT / KT;
            bf16* dst = kvs + (t & 1) * KT * KW;
            const int kk = tid / TPK, part = tid % TPK;
            const int i = first + t * KT + kk;
            bool ok = i < first + nkeys;
            int pos = -1;
            size_t slot = 0;
            if constexpr (!ROWS) {
                if (ok) {
                    if (kind == 0) {
                        pos = prow - min(prow + 1, SWA_KEYS) + 1 + i;
                    } else if (kind == 1) {
                        if (!lst) {
                            pos = i;
                        } else {
                            const int gi = __ldg(a.LIST + static_cast<size_t>(r) * LISTW + (i >> 2));
                            pos = gi < 0 ? -1 : 4 * gi + (i & 3);
                        }
                    } else {
                        pos = __ldg(a.TOK + static_cast<size_t>(r) * a.ldtok + i);
                    }
                    ok = pos >= 0 && pos <= prow;
                }
                if (ok) {
                    if (a.PAGE_TABLE && kind != 0) {
                        const unsigned up = static_cast<unsigned>(pos);
                        slot = static_cast<size_t>(__ldg(a.PAGE_TABLE + sbase + (up >> 8))) * 256u + (up & 255u);
                    } else {
                        slot = sbase + (kind == 0 ? static_cast<size_t>(pos % a.ring) : static_cast<size_t>(pos));
                    }
                }
            } else {
                pos = i;
                ok = ok && pos >= 0;
                if (ok)
                    slot = kind == 0 ? static_cast<size_t>(sig) * a.ring + pos % a.ring
                                     : static_cast<size_t>(sig) * a.ctx + pos;
            }
            const bf16* src = (ROWS ? a.CACHE : row_cache) + slot * KW;
#pragma unroll
            for (int u = part; u < KW / 8; u += TPK) tf_common::cp16z(dst + wide(kk, u), src + u * 8, ok);
            if (part == 0) kpos[(t & 1) * KT + kk] = ok ? pos : -1;
        };
        load_tile(0);
        tf_common::cp_commit();
        if (ntiles > 1) load_tile(1);
        tf_common::cp_commit();

        const bool my_in = ROWS ? ((pass >> (srow & 15)) & 1) != 0 : true;
        auto sees = [&](int pr, bool in, int kp) -> bool {
            if constexpr (!ROWS) return kp >= 0;
            else return in && kp >= 0 && kp <= pr && (kind != 0 || kp > pr - SWA_KEYS);
        };

        for (int t = 0; t < ntiles; ++t) {
            tf_common::cp_wait<1>();
            __syncthreads();
            if (t == 0) TSTAMP(1);
            const bf16* kv = kvs + (t & 1) * KT * KW;
            const int* kp_t = kpos + (t & 1) * KT;

            bool poisoned = false;
            if constexpr (ROWS) {
                const int tlo = first + t * KT, thi = min(first + nkeys, tlo + KT) - 1;
                const bool need = !(pass == live && tlo >= vis_lo && thi <= vis_hi);
                if (need) {
                    int bad = 0;
                    for (int idx = tid; idx < KT * (KL / 8); idx += NT) {
                        const int kk = idx / (KL / 8), u = idx % (KL / 8);
                        const int kp = kp_t[kk];
                        if (kp >= 0 && (pass != live || kp < vis_lo || kp > vis_hi))
                            bad |= nonfinite8(*reinterpret_cast<const uint4*>(kv + wide(kk, u))) ? 1 : 0;
                    }
                    poisoned = __syncthreads_or(bad) != 0;
                }
            }

            {
                const int ntile = warp % C::NTILE, kp = warp / C::NTILE;
                const int mt = ntile / (KT / 8), nt = ntile % (KT / 8);
                float c0[4] = {0.f, 0.f, 0.f, 0.f}, c1[4] = {0.f, 0.f, 0.f, 0.f};
                const int ks0 = kp * C::KSTEPS;
#pragma unroll
                for (int s = 0; s < C::KSTEPS; s += 2) {
                    const int ks = ks0 + s;
                    if (s + 1 < C::KSTEPS) {
                        uint32_t b[4], a0[4], a1[4];
                        ldsm4(b, saddr(kv + wide(nt * 8 + (lane & 7), ks * 2 + (lane >> 3))));
                        if constexpr (QREG) {
#pragma unroll
                            for (int e = 0; e < 4; ++e) a0[e] = qf[s][e], a1[e] = qf[s + 1][e];
                        } else {
                            ldsm4(a0, saddr(qs + wide(mt * 16 + (lane & 15), ks * 2 + (lane >> 4))));
                            ldsm4(a1, saddr(qs + wide(mt * 16 + (lane & 15), ks * 2 + 2 + (lane >> 4))));
                        }
                        mma(c0, a0, b[0], b[1]);
                        mma(c1, a1, b[2], b[3]);
                        if constexpr (EXACT) {
                            if constexpr (QREG) {
                                mma(c0, qfl[s], b[0], b[1]);
                                mma(c1, qfl[s + 1], b[2], b[3]);
                            } else {
                                ldsm4(a0, saddr(qls + wide(mt * 16 + (lane & 15), ks * 2 + (lane >> 4))));
                                ldsm4(a1, saddr(qls + wide(mt * 16 + (lane & 15), ks * 2 + 2 + (lane >> 4))));
                                mma(c0, a0, b[0], b[1]);
                                mma(c1, a1, b[2], b[3]);
                            }
                        }
                    } else {
                        uint32_t b[2], a0[4];
                        ldsm2(b, saddr(kv + wide(nt * 8 + (lane & 7), ks * 2 + ((lane >> 3) & 1))));
                        if constexpr (QREG) {
#pragma unroll
                            for (int e = 0; e < 4; ++e) a0[e] = qf[s][e];
                        } else {
                            ldsm4(a0, saddr(qs + wide(mt * 16 + (lane & 15), ks * 2 + (lane >> 4))));
                        }
                        mma(c0, a0, b[0], b[1]);
                        if constexpr (EXACT) {
                            if constexpr (QREG) {
                                mma(c0, qfl[s], b[0], b[1]);
                            } else {
                                ldsm4(a0, saddr(qls + wide(mt * 16 + (lane & 15), ks * 2 + (lane >> 4))));
                                mma(c0, a0, b[0], b[1]);
                            }
                        }
                    }
                }
                float* rd = red + (kp * NQ + mt * 16) * C::SPAD + nt * 8 + 2 * q4;
                *reinterpret_cast<float2*>(rd + g * C::SPAD) = make_float2(__fadd_rn(c0[0], c1[0]), __fadd_rn(c0[1], c1[1]));
                *reinterpret_cast<float2*>(rd + (g + 8) * C::SPAD) =
                    make_float2(__fadd_rn(c0[2], c1[2]), __fadd_rn(c0[3], c1[3]));
            }
            __syncthreads();

            {
                float sc[C::CPT];
                bool vis[C::CPT];
#pragma unroll
                for (int e = 0; e < C::CPT; ++e) {
                    const int c = ssub * C::CPT + e;
                    float v = red[srow * C::SPAD + c];
#pragma unroll
                    for (int ks = 1; ks < C::KSPLIT; ++ks) v = __fadd_rn(v, red[(ks * NQ + srow) * C::SPAD + c]);
                    vis[e] = sees(my_p, my_in, kp_t[c]);
                    sc[e] = vis[e] ? __fmul_rn(v, SCALE) : -INFINITY;
                }
                float tm = sc[0];
#pragma unroll
                for (int e = 1; e < C::CPT; ++e) tm = nmax(tm, sc[e]);
#pragma unroll
                for (int o = C::TPR / 2; o > 0; o >>= 1) tm = nmax(tm, __shfl_xor_sync(FULL, tm, o));
                const bool act = tm != -INFINITY;
                const float mn = act ? nmax(m_run, tm) : m_run;
                const float al = act ? (m_run == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(m_run, mn))) : 1.f;
                float ls = 0.f;
                bf16 pb[C::CPT], plb[C::CPT];
#pragma unroll
                for (int e = 0; e < C::CPT; ++e) {
                    const float pv = (act && vis[e]) ? tf_glue::tf_exp(__fsub_rn(sc[e], mn)) : 0.f;
                    pb[e] = __float2bfloat16_rn(pv);
                    plb[e] = __float2bfloat16_rn(__fsub_rn(pv, __bfloat162float(pb[e])));
                    ls = __fadd_rn(ls, EXACT ? pv : __bfloat162float(pb[e]));
                }
#pragma unroll
                for (int o = C::TPR / 2; o > 0; o >>= 1) ls = __fadd_rn(ls, __shfl_xor_sync(FULL, ls, o));
                l_run = __fmaf_rn(l_run, al, ls);
                m_run = mn;
                const int c0 = ssub * C::CPT;
                if constexpr (C::CPT == 1) {
                    ps[narrow<KT>(srow, c0 >> 3) + (c0 & 7)] = pb[0];
                    if constexpr (EXACT) pls[narrow<KT>(srow, c0 >> 3) + (c0 & 7)] = plb[0];
                } else {
#pragma unroll
                    for (int e = 0; e < C::CPT; e += 2) {
                        const int c = c0 + e;
                        __nv_bfloat162 two;
                        two.x = pb[e];
                        two.y = pb[e + 1];
                        *reinterpret_cast<__nv_bfloat162*>(ps + narrow<KT>(srow, c >> 3) + (c & 7)) = two;
                        if constexpr (EXACT) {
                            two.x = plb[e];
                            two.y = plb[e + 1];
                            *reinterpret_cast<__nv_bfloat162*>(pls + narrow<KT>(srow, c >> 3) + (c & 7)) = two;
                        }
                    }
                }
                if (ssub == 0) rowv[srow] = al;
            }
            __syncthreads();

            {
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const float a0 = rowv[mt * 16 + g], a1 = rowv[mt * 16 + g + 8];
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        acc[mt][jj][0] = __fmul_rn(acc[mt][jj][0], a0);
                        acc[mt][jj][1] = __fmul_rn(acc[mt][jj][1], a0);
                        acc[mt][jj][2] = __fmul_rn(acc[mt][jj][2], a1);
                        acc[mt][jj][3] = __fmul_rn(acc[mt][jj][3], a1);
                    }
                }
                if (!poisoned) {
#pragma unroll
                    for (int k16 = 0; k16 < KT / 16; ++k16) {
                        uint32_t pa[MT][4];
#pragma unroll
                        for (int mt = 0; mt < MT; ++mt)
                            ldsm4(pa[mt], saddr(ps + narrow<KT>(mt * 16 + (lane & 15), k16 * 2 + (lane >> 4))));
#pragma unroll
                        for (int jj = 0; jj < 8; jj += 2) {
                            uint32_t b[4];
                            ldsm4t(b, saddr(kv + wide(k16 * 16 + (lane & 15), warp * 8 + jj + (lane >> 4))));
#pragma unroll
                            for (int mt = 0; mt < MT; ++mt) {
                                mma(acc[mt][jj], pa[mt], b[0], b[1]);
                                mma(acc[mt][jj + 1], pa[mt], b[2], b[3]);
                            }
                            if constexpr (EXACT) {
#pragma unroll
                                for (int mt = 0; mt < MT; ++mt) {
                                    uint32_t pl[4];
                                    ldsm4(pl, saddr(pls + narrow<KT>(mt * 16 + (lane & 15), k16 * 2 + (lane >> 4))));
                                    mma(acc[mt][jj], pl, b[0], b[1]);
                                    mma(acc[mt][jj + 1], pl, b[2], b[3]);
                                }
                            }
                        }
                    }
                } else if constexpr (ROWS) {
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            const int row = mt * 16 + g + 8 * (e >> 1), r16 = row & 15;
                            const int pr = rinfo[r16];
                            const bool in = (pass >> r16) & 1;
#pragma unroll
                            for (int jj = 0; jj < 8; ++jj) {
                                const int col = warp * 64 + jj * 8 + 2 * q4 + (e & 1);
                                float x = acc[mt][jj][e];
#pragma unroll 1
                                for (int kk = 0; kk < KT; ++kk) {
                                    if (!sees(pr, in, kp_t[kk])) continue;
                                    float pv = __bfloat162float(ps[narrow<KT>(row, kk >> 3) + (kk & 7)]);
                                    if constexpr (EXACT) pv = __fadd_rn(pv, __bfloat162float(pls[narrow<KT>(row, kk >> 3) + (kk & 7)]));
                                    const float vv = __bfloat162float(kv[wide(kk, col >> 3) + (col & 7)]);
                                    x = __fmaf_rn(pv, vv, x);
                                }
                                acc[mt][jj][e] = x;
                            }
                        }
                }
            }
            __syncthreads();
            if (t + 2 < ntiles) load_tile(t + 2);
            tf_common::cp_commit();
        }
    };

    if constexpr (!ROWS) {
        run_pass(k0, k1 - k0, 0, 0u, 0, 0);
    } else {
        unsigned todo = live;
        while (todo) {
            const int i0 = __ffs(todo) - 1;
            const int sig = rinfo[16 + i0];
            unsigned pass = 0;
            int lo = 0x7fffffff, hi = -1, vlo = -1, vhi = 0x7fffffff;
            for (int i = 0; i < 16; ++i) {
                if (((todo >> i) & 1) && rinfo[16 + i] == sig) {
                    pass |= 1u << i;
                    const int pr = rinfo[i];
                    const int wl = kind == 0 ? max(0, pr - (SWA_KEYS - 1)) : 0;
                    lo = min(lo, wl);
                    hi = max(hi, pr);
                    vlo = max(vlo, wl);
                    vhi = min(vhi, pr);
                }
            }
            todo &= ~pass;
            run_pass(lo, hi - lo + 1, sig, pass, vlo, vhi);
        }
    }

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

    auto head_of = [&](int row) { return ROWS ? h0 + (row >> 4) : h0 + row; };
    if (nspl == 1) {
        if (tid < NQ) {
            const int h = head_of(tid);
            const float m = m_s[tid], l = l_s[tid], sk = h < a.Hl ? a.SINKS[h] : -INFINITY;
            const float top = nmax(m, sk);
            const float w = m == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(m, top));
            const float es = sk == -INFINITY ? 0.f : tf_glue::tf_exp(__fsub_rn(sk, top));
            w_s[tid] = w;
            d_s[tid] = __fmaf_rn(l, w, es);
        }
        __syncthreads();
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const int row = mt * 16 + g + 8 * hf;
                size_t orow;
                if constexpr (!ROWS) {
                    const int h = h0 + row;
                    if (h >= a.Hl) continue;
                    orow = static_cast<size_t>(r) * a.Hl + h;
                } else {
                    const int r16 = row & 15;
                    const int h = h0 + mt;
                    if (!((live >> r16) & 1) || h >= a.Hl) continue;
                    orow = static_cast<size_t>(r + r16) * a.Hl + h;
                }
                const float w = w_s[row], d = d_s[row];
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    const int col = warp * 64 + jj * 8 + 2 * q4;
                    store2(a.OUT, orow * KL + col, fin(acc[mt][jj][2 * hf], w, d), fin(acc[mt][jj][2 * hf + 1], w, d),
                           a.out_f32);
                }
            }
        return;
    }

    if constexpr (!ROWS) {
        const int HB = gridDim.y, NS = gridDim.x;
        const size_t blk = static_cast<size_t>(r) * HB + hb;
        const size_t pb0 = blk * NS * NQ;
        const size_t pj = pb0 + static_cast<size_t>(j) * NQ;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                const int row = mt * 16 + g + 8 * hf;
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    const int col = warp * 64 + jj * 8 + 2 * q4;
                    *reinterpret_cast<float2*>(a.PO + (pj + row) * KL + col) =
                        make_float2(acc[mt][jj][2 * hf], acc[mt][jj][2 * hf + 1]);
                }
            }
        if (tid < NQ) {
            a.PM[pj + tid] = m_s[tid];
            a.PL[pj + tid] = l_s[tid];
        }
        if (a.TICKET == nullptr) return;
        __threadfence();
        __syncthreads();
        if (tid == 0) misc[0] = atomicAdd(a.TICKET + blk, 1) == nspl - 1;
        __syncthreads();
        TSTAMP(3);
        if (!misc[0]) return;
        __threadfence();
        constexpr int WJ_CAP = 256;
        static_assert(WJ_CAP * NQ * 4 <= C::ROW, "smem before ROW holds 256 split weights");
        fold_rows<NQ>(a, pb0, nspl, h0, r, 0, KL, reinterpret_cast<float*>(sm), WJ_CAP, d_s, rowv + 5 * NQ);
        if (tid == 0) atomicExch(a.TICKET + blk, 0);
        __syncthreads();
        TSTAMP(4);
    }
}

}  // namespace detail
}  // namespace hy4_mla
