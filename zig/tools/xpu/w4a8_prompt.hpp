// Prompt int4 GEMM with int8 or int4 activations (w4a8, w4a8_swiglu, w4a4) on integer DPAS, GPTQ g128.
#pragma once
#include "w4_prompt.hpp"

namespace w4 {

// W4A8 prompt GEMM C = A8 · Wᵀ on int8 DPAS (A8 scaled per row and 128 K group); swiglu epilogue as w4g_dpb_swiglu.
template <HW hw>
class W4A8Prompt : public ELFCodeGenerator<hw> {
    NGEN_FORWARD_ELF(hw)

    void mul32(const Subregister &dst, const Subregister &a, const Subregister &b, const Subregister &tmp) {
        mul(1, tmp, a, b.uw(1));
        shl(1, tmp, tmp, 16);
        mul(1, dst, a, b.uw(0));
        add(1, dst, dst, tmp);
    }
    void header(const GRF &h, const Subregister &base, const Subregister &wbytes_m1, const Subregister &height_m1,
                int bw, int bh, int count) {
        mov(1, h.uq(0), base);
        mov(1, h.ud(2), wbytes_m1);
        mov(1, h.ud(3), height_m1);
        mov(1, h.ud(4), wbytes_m1);
        mov(1, h.ud(7), (bw - 1) | ((bh - 1) << 8) | ((count - 1) << 16));
    }

public:
    // a4: W4A4 with rotated int4 activations [M, K/2] bytes; per group one 128 K A tile, two u4 x s4 DPAS on raw words.
    W4A8Prompt(Product product, bool swiglu, bool a4 = false) : ELFCodeGenerator<hw>(product) {
        newArgument("A", ExternalArgumentType::GlobalPtr);
        newArgument("W", ExternalArgumentType::GlobalPtr);
        newArgument("S", ExternalArgumentType::GlobalPtr);
        newArgument("B", ExternalArgumentType::GlobalPtr);
        newArgument("C", ExternalArgumentType::GlobalPtr);
        newArgument("M", DataType::d);
        newArgument("N", DataType::d);
        newArgument("K", DataType::d);
        newArgument("Ahi", ExternalArgumentType::GlobalPtr);
        newArgument("Alo", ExternalArgumentType::GlobalPtr);
        newArgument("Mp", DataType::d);
        newArgument("G", DataType::d);
        newArgument("SA", ExternalArgumentType::GlobalPtr);
        externalName(a4 ? "w4a4" : swiglu ? "w4a8_swiglu" : "w4a8");
        requireSIMD(16);
        requireGRF(256);
        requireLocalID(1);
        requireDPAS();
        finalizeInterface();

        setDefaultNoMask();
        setDefaultAutoSWSB(true);
        prologue();
        or_(1, cr0, cr0, uint16_t(0x14C0));

        const auto A = getArgument("A"), W = getArgument("W"), S = getArgument("S"), B = getArgument("B"),
                   C = getArgument("C"), M = getArgument("M"), N = getArgument("N"), K = getArgument("K"),
                   Mp = getArgument("Mp"), SA = getArgument("SA");

        // Registers: r32-r47 headers, W 48-79, A 80-111, B 112-127, part (int32) 128-191, acc (fp32) 192-255.
        const GRF hA = r32, hW = r33, hC = r34, s = r35, t = r36, addr = r37, swraw = r38, hPW = r39;
        const GRF sw0 = r40, sw1 = r41, sa0 = r42, ftmp = r44, hPA = r46;   // ftmp : r44-r45
        const int W0 = 48, A0 = 80, BB[2] = {112, 120}, PART = 128, ACC = 192;
        Label lEnd, lGroup;

        const Subregister m0 = s.d(0), n0 = s.d(1), g = s.d(2), groups = s.d(3), mt = s.d(4), sg = s.d(5);
        const Subregister kx = s.d(6), tmp = s.d(7);
        mul(1, mt, r0.ud(7), GM);
        add(1, mt, mt, r0.ud(1));
        shl(1, tmp, mt, 6);
        cmp(1 | ge | f0[0], tmp, M);
        jmpi(1 | f0[0], lEnd);
        shr(1, sg, getLocalID(0).uw(0), 4);
        and_(1, t.d(0), sg, 1);
        shl(1, t.d(0), t.d(0), 5);
        add(1, m0, tmp, t.d(0));                          // m0 = mt·64 + (sg % 2)·32
        shr(1, t.d(1), sg, 1);
        shl(1, t.d(1), t.d(1), 5);
        shl(1, n0, r0.ud(6), 8);
        add(1, n0, n0, t.d(1));                           // n0 = nt·256 + (sg / 2)·32

        if (a4) {
            shr(1, t.d(2), K, 1);
            add(1, t.d(2), t.d(2), -1);                   // K/2 − 1 (bytes of an A4 row)
        } else
            add(1, t.d(2), K, -1);                        // K − 1 (bytes of an A8 row)
        add(1, t.d(3), M, -1);
        header(hA, A, t.d(2), t.d(3), 32, 32, 2);
        mov(1, hA.ud(6), m0);
        header(hPA, A, t.d(2), t.d(3), 32, 32, 2);
        mov(1, hPA.ud(6), m0);
        shl(1, t.d(4), N, 2);
        add(1, t.d(4), t.d(4), -1);                       // N·4 − 1
        shr(1, t.d(5), K, 3);
        add(1, t.d(5), t.d(5), -1);                       // K/8 − 1
        header(hW, W, t.d(4), t.d(5), 16, 16, 1);
        header(hPW, W, t.d(4), t.d(5), 16, 16, 1);
        if (swiglu)
            add(1, t.d(6), N, -1);
        else {
            shl(1, t.d(6), N, 1);
            add(1, t.d(6), t.d(6), -1);
        }
        header(hC, C, t.d(6), t.d(3), 16, 8, 1);
        mov(1, hC.ud(5), n0);

        for (int r = 0; r < 64; r++) mov(16, GRF(ACC + r).f(), 0.0f);
        mov(1, g, 0);
        shr(1, groups, K, 7);

        mark(lGroup);
        // Words of the group (16 rows of words per tile), prefetch of the next group (words and A).
        shl(1, hW.ud(6), g, 4);
        mov(1, hW.ud(5), n0);
        load(1, GRF(W0), block_2d(DataSizeLSC::D32, 16, 16), A64, hW);
        add(1, hW.ud(5), n0, 16);
        load(1, GRF(W0 + 16), block_2d(DataSizeLSC::D32, 16, 16), A64, hW);
        add(1, hPW.ud(6), hW.ud(6), 16);
        mov(1, hPW.ud(5), n0);
        load(1, null, block_2d(DataSizeLSC::D32, 16, 16), A64, hPW);
        add(1, hPW.ud(5), n0, 16);
        load(1, null, block_2d(DataSizeLSC::D32, 16, 16), A64, hPW);
        if (a4) {
            shl(1, kx, g, 6);                             // bytes of A4 before this group
            add(1, hPA.ud(5), kx, 64);
            load(1, null, block_2d(DataSizeLSC::D8, 32, 32, 2), A64, hPA);
        } else {
            shl(1, kx, g, 7);
            add(1, hPA.ud(5), kx, 128);
            load(1, null, block_2d(DataSizeLSC::D8, 32, 32, 2), A64, hPA);
            add(1, hPA.ud(5), kx, 192);
            load(1, null, block_2d(DataSizeLSC::D8, 32, 32, 2), A64, hPA);
        }
        // Scales of the weights (32 columns, fp16 → fp32) and of the activations (32 rows, fp32).
        mul32(t.d(0), g, N, t.d(5));
        add(1, t.d(0), t.d(0), n0);
        shl(1, t.d(0), t.d(0), 1);
        mov(1, addr.uq(0), t.ud(0));
        add(1, addr.uq(0), addr.uq(0), S);
        load.ugm(1, swraw, block(DataSizeLSC::D32, 16), A64, addr);
        mul32(t.d(1), g, Mp, t.d(5));
        add(1, t.d(1), t.d(1), m0);
        shl(1, t.d(1), t.d(1), 2);
        mov(1, addr.uq(0), t.ud(1));
        add(1, addr.uq(0), addr.uq(0), SA);
        load.ugm(1, sa0, block(DataSizeLSC::D32, 32), A64, addr);
        if (a4) {                                         // 128 K = 64 bytes: one read, two k64 steps
            mov(1, hA.ud(5), kx);
            load(1, GRF(A0), block_2d(DataSizeLSC::D8, 32, 32, 2), A64, hA);
            for (int st = 0; st < 2; st++)
                for (int j = 0; j < 2; j++)
                    for (int i = 0; i < 4; i++) {
                        const GRF part = GRF(PART + 8 * (i * 2 + j)), a = GRF(A0 + 16 * st + 4 * i);
                        const GRF bw = GRF(W0 + 16 * j + 8 * st);
                        if (st == 0)
                            dpas(16, 8, 8, part.d(), null.d(), bw.u4(), a.s4());
                        else
                            dpas(16, 8, 8, part.d(), part.d(), bw.u4(), a.s4());
                    }
        }
        for (int h = 0; h < 2 && !a4; h++) {
            add(1, hA.ud(5), kx, 64 * h);
            load(1, GRF(A0), block_2d(DataSizeLSC::D8, 32, 32, 2), A64, hA);
            for (int st = 0; st < 2; st++) {
                const int step = 2 * h + st;              // k32 step of the group: rows of words 4·step … + 3
                for (int j = 0; j < 2; j++) {
                    const int bb = BB[j];
                    for (int p = 0; p < 2; p++) {          // B: [w0 & m, w1 & m, (w0 >> 4) & m, (w1 >> 4) & m]
                        const GRF wd = GRF(W0 + 16 * j + 4 * step + 2 * p);
                        and_(32, GRF(bb + 4 * p).ud(), wd.ud(), 0x0F0F0F0Fu);
                        shr(32, GRF(bb + 4 * p + 2).ud(), wd.ud(), 4);
                        and_(32, GRF(bb + 4 * p + 2).ud(), GRF(bb + 4 * p + 2).ud(), 0x0F0F0F0Fu);
                    }
                    for (int i = 0; i < 4; i++) {
                        const GRF part = GRF(PART + 8 * (i * 2 + j)), a = GRF(A0 + 16 * st + 4 * i);
                        if (step == 0)
                            dpas(16, 8, 8, part.d(), null.d(), GRF(bb).ub(), a.b());
                        else
                            dpas(16, 8, 8, part.d(), part.d(), GRF(bb).ub(), a.b());
                    }
                }
            }
        }
        // acc += float(part)·s_w·s_a
        mov(16, sw0.f(), swraw.hf(0)(1));
        mov(16, sw1.f(), swraw.hf(16)(1));
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 2; j++)
                for (int r = 0; r < 8; r += 2) {
                    const int row = 8 * i + r;
                    const GRF part = GRF(PART + 8 * (i * 2 + j) + r), acc = GRF(ACC + 8 * (i * 2 + j) + r);
                    mov(32, ftmp.f(), part.d());
                    mul(32, ftmp.f(), ftmp.f(), (j ? sw1 : sw0).f(0)(0, 16, 1));
                    for (int q = 0; q < 2; q++)
                        mad(16, GRF(acc.getBase() + q).f(), GRF(acc.getBase() + q).f(),
                            GRF(sa0.getBase() + (row + q) / 16).f((row + q) % 16), GRF(ftmp.getBase() + q).f());
                }
        add(1, g, g, 1);
        cmp(1 | lt | f1[0], g, groups);
        jmpi(1 | f1[0], lGroup);

        // Bias by fp16 DPAS: acc += Ahi·(b·128) + Alo·(b·128), in steps of 16 groups (as w4g_dpb).
        {
            const auto Ahi = getArgument("Ahi"), Alo = getArgument("Alo"), Gs = getArgument("G");
            const GRF hAH = r46, hAL = r47, hBB = r43;
            const Subregister st = r45.d(0), nst = r45.d(1);
            Label lBias;
            shl(1, t.d(0), Gs, 1);
            add(1, t.d(0), t.d(0), -1);
            add(1, t.d(1), M, -1);
            header(hAH, Ahi, t.d(0), t.d(1), 16, 32, 1);
            header(hAL, Alo, t.d(0), t.d(1), 16, 32, 1);
            mov(1, hAH.ud(6), m0);
            mov(1, hAL.ud(6), m0);
            shl(1, t.d(2), N, 1);
            add(1, t.d(2), t.d(2), -1);
            add(1, t.d(3), Gs, -1);
            header(hBB, B, t.d(2), t.d(3), 16, 16, 1);
            add(1, nst, Gs, 15);
            shr(1, nst, nst, 4);
            mov(1, st, 0);
            mark(lBias);
            shl(1, hAH.ud(5), st, 4);
            shl(1, hAL.ud(5), st, 4);
            shl(1, hBB.ud(6), st, 4);
            load(1, GRF(A0), block_2d(DataSizeLSC::D16, 16, 32), A64, hAH);
            load(1, GRF(A0 + 16), block_2d(DataSizeLSC::D16, 16, 32), A64, hAL);
            for (int j = 0; j < 2; j++) {
                add(1, hBB.ud(5), n0, 16 * j);
                load(1, GRF(BB[j]), block_2d(DataSizeLSC::D16, 16, 16) | vnni, A64, hBB);
            }
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 4; i++) {
                    const GRF acc = GRF(ACC + 8 * (i * 2 + j));
                    dpas(16, 8, 8, acc.f(), acc.f(), GRF(BB[j]).hf(), GRF(A0 + 4 * i).hf());
                    dpas(16, 8, 8, acc.f(), acc.f(), GRF(BB[j]).hf(), GRF(A0 + 16 + 4 * i).hf());
                }
            add(1, st, st, 1);
            cmp(1 | lt | f1[0], st, nst);
            jmpi(1 | f1[0], lBias);
        }

        const int H = A0;                                  // fp16 output: 4 registers
        if (swiglu) {
            const GRF X = GRF(PART), E = GRF(PART + 1), Q = GRF(PART + 2), U = GRF(PART + 3), O = GRF(PART + 4);
            shr(1, hC.ud(5), n0, 1);
            for (int i = 0; i < 4; i++) {
                for (int r = 0; r < 8; r++) {
                    const GRF gg = GRF(ACC + 8 * (2 * i) + r), u = GRF(ACC + 8 * (2 * i + 1) + r);
                    mov(16, Q.hf(0)(1), gg.f());
                    mov(16, X.f(), Q.hf(0)(1));
                    mov(16, Q.hf(16)(1), u.f());
                    mov(16, U.f(), Q.hf(16)(1));
                    mul(16, E.f(), X.f(), -1.4426950408889634f);
                    math(16, MathFunction::exp, E.f(), E.f());
                    add(16, E.f(), E.f(), 1.0f);
                    math(16, MathFunction::inv, E.f(), E.f());
                    mul(16, X.f(), X.f(), E.f());
                    mul(16, O.f(), X.f(), U.f());
                    shr(16, E.ud(), O.ud(), 16);
                    and_(16, E.ud(), E.ud(), 1);
                    add(16, E.ud(), E.ud(), 0x7FFF);
                    add(16, O.ud(), O.ud(), E.ud());
                    and_(16, O.ud(), O.ud(), 0xFFFF0000u);
                    mov(16, GRF(H + r / 2).hf((r % 2) * 16)(1), O.f());
                }
                add(1, hC.ud(6), m0, 8 * i);
                store(1, block_2d(DataSizeLSC::D16, 16, 8), A64, hC, GRF(H));
            }
        } else
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++) {
                    for (int r = 0; r < 8; r++)
                        mov(16, GRF(H + r / 2).hf((r % 2) * 16)(1), GRF(ACC + 8 * (i * 2 + j) + r).f());
                    add(1, hC.ud(6), m0, 8 * i);
                    add(1, hC.ud(5), n0, 16 * j);
                    store(1, block_2d(DataSizeLSC::D16, 16, 8), A64, hC, GRF(H));
                }
        mark(lEnd);
        epilogue();
    }
};

}  // namespace w4
