// Prompt int4 GEMM with fp16 activations (w4g_dpb, w4g_s4dpb, w4g_dpb_swiglu): C[M, N] = A[M, K] · Wᵀ on fp16 DPAS.
#pragma once
#include <functional>
#include <vector>

#include "ngen_elf.hpp"

namespace w4 {
using namespace ngen;

// Row tiles per packet of the 3D grid (GM, N/256, packets): the hardware walks the packet's columns before the next.
static constexpr int GM = 8;

template <HW hw>
class W4Prompt : public ELFCodeGenerator<hw> {
    NGEN_FORWARD_ELF(hw)

    // dst = a·b in 32 bits: the integer multiplier is 32 × 16 bits, so b is split in halves (a < 2¹⁶).
    void mul32(const Subregister &dst, const Subregister &a, const Subregister &b, const Subregister &tmp) {
        mul(1, tmp, a, b.uw(1));
        shl(1, tmp, tmp, 16);
        mul(1, dst, a, b.uw(0));
        add(1, dst, dst, tmp);
    }

    // 2D header: address, width − 1, height − 1, pitch − 1 (bytes), x, y, block shape.
    void header(const GRF &h, const Subregister &base, const Subregister &wbytes_m1, const Subregister &height_m1,
                int bw, int bh, int count) {
        mov(1, h.uq(0), base);
        mov(1, h.ud(2), wbytes_m1);
        mov(1, h.ud(3), height_m1);
        mov(1, h.ud(4), wbytes_m1);
        mov(1, h.ud(7), (bw - 1) | ((bh - 1) << 8) | ((count - 1) << 16));
    }

public:
    // Subgroup 32 rows × 32 columns (two 16-column tiles, 8 accumulators), work-group 64 × 256. Per group of gsz:
    // a DPAS partial sum from 0, folded into acc by the previous group's scale while this group's DPAS run (lag);
    // the bias by DPAS at the end (Ahi, Alo, b × gsz). b3: B over three buffers. s4: K in 4 slices summed from 0,
    // ((t0 + t1) + t2) + t3 kept in P. asplit: A read in two halves of 16 K, overlapping the other half's DPAS.
    // swiglu: gate|up interleaved by 16 columns, output fp16(bf16(SiLU(g)·u)) [M, N/2].
    W4Prompt(Product product, int gsz, bool s4, bool asplit, bool b3, bool swiglu) : ELFCodeGenerator<hw>(product) {
        const int subs = gsz / 64, tn = 2;
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
        if (s4) newArgument("P", ExternalArgumentType::GlobalPtr);
        externalName(swiglu ? "w4g_dpb_swiglu" : s4 ? "w4g_s4dpb" : "w4g_dpb");
        requireSIMD(16);
        requireGRF(256);
        requireLocalID(1);
        requireDPAS();
        finalizeInterface();

        setDefaultNoMask();
        setDefaultAutoSWSB(true);
        prologue();
        or_(1, cr0, cr0, uint16_t(0x14C0));            // IEEE rounding, f16/f32/f64 denormals kept (like oneDNN)

        const auto A = getArgument("A"), W = getArgument("W"), S = getArgument("S"), B = getArgument("B"),
                   C = getArgument("C"), M = getArgument("M"), N = getArgument("N"), K = getArgument("K");

        // Registers: W 48-63, B 64-71 / 72-79 (/ 248-255 with b3), A 80-111, part 113-176, acc 177-240, output H.
        const GRF hA = r32, hW = r33, hC = r34, s = r35, t = r36, hPA = r37, hPW = r38, addr = r39;
        const GRF sh = r40;
        const int W0 = 48, B0 = 64, B1 = 72, A0 = 80, PART = 113, ACC = 177, H = b3 ? 80 : 250;
        const int TILES = 4 * tn;                         // 8 × 16 accumulators per subgroup
        const int SF2 = 244;                               // scale duplicated for SIMD32: r244-r247
        const int PBF = b3 ? 242 : 248;                    // bias of the previous group (2 registers)
        const int B2 = 248;
        Label lEnd, lGroup;

        // Tile: mt = z·GM + x, nt = y; subgroup = local id of channel 0 / 16.
        const Subregister m0 = s.d(0), n0 = s.d(1), g = s.d(2), groups = s.d(3), mt = s.d(4), sg = s.d(5);
        const Subregister kx = s.d(6), tmp = s.d(7);
        mul(1, mt, r0.ud(7), GM);
        add(1, mt, mt, r0.ud(1));
        shl(1, tmp, mt, 6);
        cmp(1 | ge | f0[0], tmp, M);
        jmpi(1 | f0[0], lEnd);                           // incomplete packet: tile outside M
        shr(1, sg, getLocalID(0).uw(0), 4);
        and_(1, t.d(0), sg, 1);
        shl(1, t.d(0), t.d(0), 5);
        add(1, m0, tmp, t.d(0));                          // m0 = mt·64 + (sg % 2)·32
        shr(1, t.d(1), sg, 1);
        shl(1, t.d(1), t.d(1), 5);
        shl(1, n0, r0.ud(6), 8);
        add(1, n0, n0, t.d(1));                           // n0 = nt·256 + (sg / 2)·32

        // 2D headers (widths and heights − 1).
        shl(1, t.d(2), K, 1);
        add(1, t.d(2), t.d(2), -1);                       // K·2 − 1
        add(1, t.d(3), M, -1);
        header(hA, A, t.d(2), t.d(3), 16, 32, 2);
        mov(1, hA.ud(6), m0);
        header(hPA, A, t.d(2), t.d(3), 16, 32, 2);
        mov(1, hPA.ud(6), m0);
        shl(1, t.d(4), N, 2);
        add(1, t.d(4), t.d(4), -1);                       // N·4 − 1
        shr(1, t.d(5), K, 3);
        add(1, t.d(5), t.d(5), -1);                       // K/8 − 1
        header(hW, W, t.d(4), t.d(5), 16, 8, 1);
        mov(1, hW.ud(5), n0);
        mov(1, hW.ud(6), 0);                               // row of words of the current block
        header(hPW, W, t.d(4), t.d(5), 16, 8, 1);
        mov(1, hPW.ud(5), n0);
        if (swiglu)
            add(1, t.d(6), N, -1);                         // (N/2)·2 − 1
        else {
            shl(1, t.d(6), N, 1);
            add(1, t.d(6), t.d(6), -1);                   // N·2 − 1
        }
        header(hC, C, t.d(6), t.d(3), 16, 8, 1);
        mov(1, hC.ud(5), n0);

        for (int r = 0; r < 8 * TILES; r++) mov(16, GRF(ACC + r).f(), 0.0f);
        mov(1, g, 0);

        // Scale of group gi: 16 fp16 per tile at (gi·N + n0)·2, converted to fp32 (scale × 2²⁴) in r42 + 2j.
        auto scales = [&](const Subregister &gi) {
            mul32(t.d(0), gi, N, t.d(5));
            add(1, t.d(0), t.d(0), n0);
            shl(1, t.d(0), t.d(0), 1);
            mov(1, addr.uq(0), t.ud(0));
            add(1, addr.uq(0), addr.uq(0), S);
            load.ugm(1, sh, block(DataSizeLSC::D32, 8 * tn), A64, addr);
            for (int j = 0; j < tn; j++) {
                mov(16, GRF(42 + 2 * j).f(), sh.hf(16 * j)(1));
                mul(16, GRF(42 + 2 * j).f(), GRF(42 + 2 * j).f(), 16777216.0f);
            }
        };
        // acc += part · scale for one tile, two rows per instruction with the scale duplicated in SF2 (SIMD32).
        auto foldTile = [&](int i, int j) {
            for (int r = 0; r < 8; r += 2) {
                const int tile = i * tn + j;
                const GRF acc = GRF(ACC + 8 * tile + r), part = GRF(PART + 8 * tile + r);
                mad(32, acc.f(), acc.f(), part.f(), GRF(SF2 + 2 * j).f());
            }
        };
        // Unpacking of a unit (k16 step `step`, column tile j) into B buffer bb.
        auto unpackOps = [&](int wbuf, int step, int j, int bb) {
            std::vector<std::function<void()>> ops;
            const GRF w0 = GRF(wbuf + 8 * j + 2 * step), w1 = GRF(wbuf + 8 * j + 2 * step + 1);
            for (int q = 0; q < 4; q++) {
                const GRF d0 = GRF(bb + q), d1 = GRF(bb + 4 + q);
                if (q == 0) {
                    ops.push_back([=] { and_(16, d0.ud(), w0.ud(), 0x000F000Fu); });
                    ops.push_back([=] { and_(16, d1.ud(), w1.ud(), 0x000F000Fu); });
                } else {
                    ops.push_back([=] { shr(16, d0.ud(), w0.ud(), 4 * q); });
                    ops.push_back([=] { and_(16, d0.ud(), d0.ud(), 0x000F000Fu); });
                    ops.push_back([=] { shr(16, d1.ud(), w1.ud(), 4 * q); });
                    ops.push_back([=] { and_(16, d1.ud(), d1.ud(), 0x000F000Fu); });
                }
            }
            return ops;
        };
        // B buffer of unit n of half-block hb: the tile's parity, or a rotation over three (b3).
        auto bufB = [&](int hb, int n, int j) {
            if (!b3) return (j % 2) ? B1 : B0;
            const int idx = (hb * 2 * tn + n) % 3;
            return idx == 0 ? B0 : idx == 1 ? B1 : B2;
        };
        // Half-block (32 K): two k16 steps over both column tiles, A in abuf; afterH(h) once half h's DPAS are out.
        auto half = [&](int abuf, int wbuf, int t2, bool first, std::function<void(int)> afterH = {}, int hb = 0) {
            std::vector<std::pair<int, int>> units;
            for (int h = 0; h < 2; h++)
                for (int j = 0; j < tn; j++) units.push_back({2 * t2 + h, j});
            for (size_t n = 0; n < units.size(); n++) {
                const int step = units[n].first, j = units[n].second, h = step % 2;
                const int bb = bufB(hb, (int)n, j);
                for (auto &op : unpackOps(wbuf, step, j, bb)) op();
                for (int i = 0; i < 4; i++) {
                    const GRF part = GRF(PART + 8 * (i * tn + j));
                    const GRF a = GRF(abuf + h * 16 + i * 4);
                    if (first && h == 0) foldTile(i, j);
                    if (first && h == 0)
                        dpas(16, 8, 8, part.f(), null.f(), GRF(bb).hf(), a.hf());
                    else
                        dpas(16, 8, 8, part.f(), part.f(), GRF(bb).hf(), a.hf());
                }
                if (afterH && (n + 1 == units.size() || units[n + 1].first % 2 != h)) afterH(h);
            }
        };
        // asplit: half h of the next half-block's A (header r241, block 16 × 32) loaded once half h's DPAS are out.
        const GRF hA1 = r241;
        auto loadAHalf = [&](int h, int dx) {
            add(1, hA1.ud(5), kx, dx + 16 * h);
            load(1, GRF(A0 + 16 * h), block_2d(DataSizeLSC::D16, 16, 32, 1), A64, hA1);
        };
        if (asplit) {
            shl(1, t.d(2), K, 1);
            add(1, t.d(2), t.d(2), -1);
            add(1, t.d(3), M, -1);
            header(hA1, A, t.d(2), t.d(3), 16, 32, 1);
            mov(1, hA1.ud(6), m0);
        }

        // s4: current slice, end of the slice, groups per slice.
        const Subregister sl = b3 ? r46.d(0) : r254.d(0), slEnd = b3 ? r46.d(1) : r254.d(1), per = b3 ? r46.d(2) : r254.d(2);
        Label lSlice, lNoLoad, lNoStore;
        if (s4) {
            shr(1, per, K, gsz == 64 ? 6 : 7);
            shr(1, per, per, 2);
            mov(1, sl, 0);
            mov(1, slEnd, 0);
            mov(1, g, 0);
            mov(1, kx, 0);
            if (asplit) for (int h = 0; h < 2; h++) loadAHalf(h, 0);
            mark(lSlice);
            add(1, slEnd, slEnd, per);
        }
        for (int j = 0; j < tn; j++) {                    // group 0 folds the previous scale 0 (acc + 0 exact)
            mov(16, GRF(42 + 2 * j).f(), 0.0f);
            mov(16, GRF(43 + 2 * j).f(), 0.0f);
        }
        for (int r = 0; r < 8 * TILES; r++) mov(16, GRF(PART + r).f(), 0.0f);
        if (s4)
            mov(1, groups, slEnd);
        else {
            shr(1, groups, K, gsz == 64 ? 6 : 7);         // groups
            mov(1, kx, 0);
            if (asplit) for (int h = 0; h < 2; h++) loadAHalf(h, 0);
        }
        mark(lGroup);
        for (int u = 0; u < subs; u++) {
            add(1, hPW.ud(6), hW.ud(6), 8);
            load(1, null, block_2d(DataSizeLSC::D32, 16, 8), A64, hPW);
            add(1, hPA.ud(5), kx, 64);
            load(1, null, block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hPA);
            add(1, hPA.ud(5), kx, 96);
            load(1, null, block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hPA);
            for (int j = 0; j < tn; j++) {
                if (j) add(1, hW.ud(5), hW.ud(5), 16);
                load(1, GRF(W0 + 8 * j), block_2d(DataSizeLSC::D32, 16, 8), A64, hW);
            }
            add(1, hW.ud(5), hW.ud(5), -16 * (tn - 1));
            if (u == 0) {
                // The current group's scale becomes the previous one (duplicated for SIMD32).
                for (int j = 0; j < tn; j++) {
                    mov(16, GRF(SF2 + 2 * j).f(), GRF(42 + 2 * j).f());
                    mov(16, GRF(SF2 + 2 * j + 1).f(), GRF(42 + 2 * j).f());
                    mov(16, GRF(PBF + j).f(), GRF(43 + 2 * j).f());
                }
                add(1, t.d(7), g, -1);
                sel(1 | ge, t.d(7), t.d(7), 0);
                scales(g);
            }
            for (int t2 = 0; t2 < 2; t2++) {
                if (asplit) {
                    half(A0, W0, t2, u == 0 && t2 == 0, [&](int h) { loadAHalf(h, 32 * (t2 + 1)); }, u * 2 + t2);
                } else {
                    add(1, hA.ud(5), kx, 32 * t2);
                    load(1, GRF(A0), block_2d(DataSizeLSC::D16, 16, 32, 2), A64, hA);
                    half(A0, W0, t2, u == 0 && t2 == 0, {}, u * 2 + t2);
                }
            }
            add(1, kx, kx, 64);
            add(1, hW.ud(6), hW.ud(6), 8);
        }
        add(1, g, g, 1);
        cmp(1 | lt | f1[0], g, groups);
        jmpi(1 | f1[0], lGroup);
        // Last group: its scale folds its partial sums.
        for (int j = 0; j < tn; j++) {
            mov(16, GRF(SF2 + 2 * j).f(), GRF(42 + 2 * j).f());
            mov(16, GRF(SF2 + 2 * j + 1).f(), GRF(42 + 2 * j).f());
            mov(16, GRF(PBF + j).f(), GRF(43 + 2 * j).f());
        }
        add(1, t.d(7), groups, -1);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < tn; j++) foldTile(i, j);
        if (s4) {
            // total += slice, tile by tile (read into W0 or A0, free here); slice 0 stored as is, the last kept in acc.
            const GRF hP4 = b3 ? r47 : r255;
            const auto P = getArgument("P");
            shl(1, t.d(0), N, 2);
            add(1, t.d(0), t.d(0), -1);               // N·4 − 1
            add(1, t.d(1), M, -1);
            header(hP4, P, t.d(0), t.d(1), 16, 8, 1);
            cmp(1 | eq | f1[0], sl, 0);
            jmpi(1 | f1[0], lNoLoad);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < tn; j++) {
                    const int tile = i * tn + j, tmp = asplit ? W0 + 8 * (tile % 4) : A0 + 8 * (tile % 4);
                    add(1, hP4.ud(5), n0, 16 * j);
                    add(1, hP4.ud(6), m0, 8 * i);
                    load(1, GRF(tmp), block_2d(DataSizeLSC::D32, 16, 8), A64, hP4);
                    for (int r = 0; r < 8; r++)
                        add(16, GRF(ACC + 8 * tile + r).f(), GRF(tmp + r).f(), GRF(ACC + 8 * tile + r).f());
                }
            mark(lNoLoad);
            cmp(1 | eq | f1[0], sl, 3);
            jmpi(1 | f1[0], lNoStore);
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < tn; j++) {
                    add(1, hP4.ud(5), n0, 16 * j);
                    add(1, hP4.ud(6), m0, 8 * i);
                    store(1, block_2d(DataSizeLSC::D32, 16, 8), A64, hP4, GRF(ACC + 8 * (i * tn + j)));
                }
            memfence(b3 ? r45 : r253);
            fencewait();
            for (int r = 0; r < 8 * TILES; r++) mov(16, GRF(ACC + r).f(), 0.0f);
            mark(lNoStore);
            add(1, sl, sl, 1);
            cmp(1 | lt | f1[0], sl, 4);
            jmpi(1 | f1[0], lSlice);
        }
        {
            // acc += Ahi·B + Alo·B in steps of 16 groups, B [16 groups × 16 columns] in VNNI; out of bounds reads 0.
            const auto Ahi = getArgument("Ahi"), Alo = getArgument("Alo"), Gs = getArgument("G");
            const GRF hAH = b3 ? r40 : r250, hAL = b3 ? r41 : r251, hBB = b3 ? r39 : r252;
            const Subregister st = b3 ? r46.d(4) : r253.d(0), nst = b3 ? r46.d(5) : r253.d(1);
            Label lBias;
            shl(1, t.d(0), Gs, 1);
            add(1, t.d(0), t.d(0), -1);               // G·2 − 1
            add(1, t.d(1), M, -1);
            header(hAH, Ahi, t.d(0), t.d(1), 16, 32, 1);
            header(hAL, Alo, t.d(0), t.d(1), 16, 32, 1);
            mov(1, hAH.ud(6), m0);
            mov(1, hAL.ud(6), m0);
            shl(1, t.d(2), N, 1);
            add(1, t.d(2), t.d(2), -1);               // N·2 − 1
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
            for (int j = 0; j < tn; j++) {
                add(1, hBB.ud(5), n0, 16 * j);
                load(1, GRF(j ? B1 : B0), block_2d(DataSizeLSC::D16, 16, 16) | vnni, A64, hBB);
            }
            for (int j = 0; j < tn; j++)
                for (int i = 0; i < 4; i++) {
                    const GRF acc = GRF(ACC + 8 * (i * tn + j));
                    dpas(16, 8, 8, acc.f(), acc.f(), GRF(j ? B1 : B0).hf(), GRF(A0 + 4 * i).hf());
                    dpas(16, 8, 8, acc.f(), acc.f(), GRF(j ? B1 : B0).hf(), GRF(A0 + 16 + 4 * i).hf());
                }
            add(1, st, st, 1);
            cmp(1 | lt | f1[0], st, nst);
            jmpi(1 | f1[0], lBias);
        }

        if (swiglu) {
            // Activation of 8 rows × 16 columns per row tile i: activation columns n0/2 … + 15.
            const GRF X = GRF(PART), E = GRF(PART + 1), Q = GRF(PART + 2), U = GRF(PART + 3), O = GRF(PART + 4);
            shr(1, hC.ud(5), n0, 1);
            for (int i = 0; i < 4; i++) {
                for (int r = 0; r < 8; r++) {
                    const GRF g = GRF(ACC + 8 * (2 * i) + r), u = GRF(ACC + 8 * (2 * i + 1) + r);
                    mov(16, Q.hf(0)(1), g.f());              // g rounded to fp16
                    mov(16, X.f(), Q.hf(0)(1));
                    mov(16, Q.hf(16)(1), u.f());             // u rounded to fp16
                    mov(16, U.f(), Q.hf(16)(1));
                    mul(16, E.f(), X.f(), -1.4426950408889634f);
                    math(16, MathFunction::exp, E.f(), E.f());
                    add(16, E.f(), E.f(), 1.0f);
                    math(16, MathFunction::inv, E.f(), E.f());
                    mul(16, X.f(), X.f(), E.f());            // SiLU(g)
                    mul(16, O.f(), X.f(), U.f());
                    shr(16, E.ud(), O.ud(), 16);             // bf16 rounding to nearest even
                    and_(16, E.ud(), E.ud(), 1);
                    add(16, E.ud(), E.ud(), 0x7FFF);
                    add(16, O.ud(), O.ud(), E.ud());
                    and_(16, O.ud(), O.ud(), 0xFFFF0000u);
                    mov(16, GRF(H + r / 2).hf((r % 2) * 16)(1), O.f());
                }
                add(1, hC.ud(6), m0, 8 * i);
                store(1, block_2d(DataSizeLSC::D16, 16, 8), A64, hC, GRF(H));
            }
        } else {
            // Output: fp16 rounded to nearest even, written by 2D blocks of 8 rows × 16 columns.
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < tn; j++) {
                    for (int r = 0; r < 8; r++)
                        mov(16, GRF(H + r / 2).hf((r % 2) * 16)(1), GRF(ACC + 8 * (i * tn + j) + r).f());
                    add(1, hC.ud(6), m0, 8 * i);
                    add(1, hC.ud(5), n0, 16 * j);
                    store(1, block_2d(DataSizeLSC::D16, 16, 8), A64, hC, GRF(H));
                }
        }
        mark(lEnd);
        epilogue();
    }
};

}  // namespace w4
