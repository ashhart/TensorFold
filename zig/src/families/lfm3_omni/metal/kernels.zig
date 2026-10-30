//! LFM3-Omni's Metal launches, each binding its buffers and argument struct as zig/kernels/metal/lfm3_omni declares.
const std = @import("std");
const gpu = @import("gpu.zig");

const Gpu = gpu.Gpu;

pub const Act = enum(i32) { none = 0, silu = 1, gelu = 2 };
pub const Norm = enum(i32) { lfm = 0, head = 1 };
pub const DataType = enum { bf16, f32 };

/// Rows the few-row product takes; more go through the tiled product.
pub const gemv_rows = 16;

fn i(x: usize) i32 {
    return @intCast(x);
}

fn blocks(n: usize, per: usize) usize {
    return (n + per - 1) / per;
}

pub const Kernels = struct {
    g: *Gpu,

    fn go(k: *const Kernels, p: gpu.P, n: usize, per: usize, bufs: []const u64, args: anytype) void {
        k.g.go(p, .{ blocks(n, per), 1, 1 }, per, bufs, args);
    }

    // ---- matrix products

    /// Whether `t` rows of `kdim` (rows `ldx` apart) take the few-row product.
    pub fn few(t: usize, kdim: usize, ldx: usize, x: u64) bool {
        return t <= gemv_rows and kdim % 8 == 0 and ldx % 8 == 0 and x % 16 == 0;
    }

    /// The few-row product's result: bf16 / fp32 rows, or SwiGLU of interleaved (a, c) column pairs.
    pub const Epi = enum(i32) { plain = 0, swiglu = 2 };

    pub const RowsArgs = extern struct { K: i32, N: i32, T: i32, ldx: i32, ldy: i32, has_y: i32, has_yf: i32, epi: i32, sw: i64, sx: i64, sy: i64 };

    /// The few-row product into bf16 y and / or fp32 yf; with `.swiglu` columns are (a, c) pairs, y n / 2 wide.
    pub fn product(k: *const Kernels, w: u64, x: u64, ldx: usize, kdim: usize, n: usize, t: usize, y: u64, ldy: usize, yf: u64, batch: usize, sw: usize, sx: usize, sy: usize, epi: Epi) void {
        const p: gpu.P = if (t <= 1) .om_rows_1 else if (t <= 4) .om_rows_4 else if (t <= 8) .om_rows_8 else if (t <= 12) .om_rows_12 else .om_rows_16;
        k.g.go(p, .{ blocks(n, 16), 1, batch }, 128, &.{ w, x, y, yf }, RowsArgs{ .K = i(kdim), .N = i(n), .T = i(t), .ldx = i(ldx), .ldy = i(ldy), .has_y = @intFromBool(y != 0), .has_yf = @intFromBool(yf != 0), .epi = @backingInt(epi), .sw = @intCast(sw), .sx = @intCast(sx), .sy = @intCast(sy) });
    }

    pub fn gemv(k: *const Kernels, w: u64, x: u64, ldx: usize, kdim: usize, n: usize, t: usize, y: u64, ldy: usize, yf: u64, batch: usize, sw: usize, sx: usize, sy: usize) void {
        k.product(w, x, ldx, kdim, n, t, y, ldy, yf, batch, sw, sx, sy, .plain);
    }

    /// C [m, n] = A [m, kd] B [kd, n], A stored [kd, m] when `ta`, B [n, kd] when `tb`; `batch` products.
    pub fn gemm(k: *const Kernels, ta: bool, tb: bool, m: usize, n: usize, kd: usize, a: u64, lda: usize, b: u64, ldb: usize, c: u64, ldc: usize, in_t: DataType, out_t: DataType, batch: usize, sa: usize, sb: usize, sc: usize) void {
        const A = extern struct { M: i32, N: i32, K: i32, lda: i32, ldb: i32, ldc: i32, ta: i32, tb: i32, sa: i64, sb: i64, sc: i64 };
        const p: gpu.P = switch (in_t) {
            .bf16 => if (out_t == .bf16) .om_gemm_bb else .om_gemm_bf,
            .f32 => .om_gemm_ff,
        };
        std.debug.assert(!(in_t == .f32 and out_t == .bf16));
        if (in_t == .bf16 and k.g.hasNax()) {
            const nax: ?gpu.Nax = if (!ta and tb) (if (out_t == .bf16) .om_nax_bb_nt else .om_nax_bf_nt) else if (out_t == .f32 and !ta and !tb) .om_nax_bf_nn else if (out_t == .f32 and ta and tb) .om_nax_bf_tt else null;
            if (nax) |np| {
                const N = extern struct { M: i32, N: i32, K: i32, lda: i32, ldb: i32, ldd: i32, kblocks: i32, pad: i32 = 0, sa: i64, sb: i64, sd: i64 };
                return k.g.goNax(np, .{ blocks(n, 128), blocks(m, 64), batch }, 256, &.{ a, b, c }, N{ .M = i(m), .N = i(n), .K = i(kd), .lda = i(lda), .ldb = i(ldb), .ldd = i(ldc), .kblocks = i(kd / 256), .sa = @intCast(sa), .sb = @intCast(sb), .sd = @intCast(sc) });
            }
        }
        k.g.go(p, .{ blocks(n, 64), blocks(m, 64), batch }, 128, &.{ a, b, c }, A{ .M = i(m), .N = i(n), .K = i(kd), .lda = i(lda), .ldb = i(ldb), .ldc = i(ldc), .ta = @intFromBool(ta), .tb = @intFromBool(tb), .sa = @intCast(sa), .sb = @intCast(sb), .sc = @intCast(sc) });
    }

    /// y [t, n] = x [t, k] w^T with w [n, k]; x rows `ldx` apart, y rows `ldy` apart; bf16 out, fp32 accumulate.
    pub fn linear(k: *const Kernels, w: u64, x: u64, ldx: usize, kdim: usize, n: usize, y: u64, ldy: usize, t: usize) void {
        if (few(t, kdim, ldx, x)) return k.gemv(w, x, ldx, kdim, n, t, y, ldy, 0, 1, 0, 0, 0);
        k.gemm(false, true, t, n, kdim, x, ldx, w, kdim, y, ldy, .bf16, .bf16, 1, 0, 0, 0);
    }

    /// As `linear`, the result in fp32 (for a bias added before the one rounding).
    pub fn linearF32(k: *const Kernels, w: u64, x: u64, ldx: usize, kdim: usize, n: usize, y: u64, t: usize) void {
        if (few(t, kdim, ldx, x)) return k.gemv(w, x, ldx, kdim, n, t, 0, n, y, 1, 0, 0, 0);
        k.gemm(false, true, t, n, kdim, x, ldx, w, kdim, y, n, .bf16, .f32, 1, 0, 0, 0);
    }

    /// Row-major C [m, n] = A [m, kd] B [kd, n] with either transposed in memory (`ta`: A stored [kd, m]).
    pub fn matmul(k: *const Kernels, ta: bool, tb: bool, m: usize, n: usize, kd: usize, a: u64, b: u64, c: u64, in_t: DataType, out_t: DataType) void {
        k.gemm(ta, tb, m, n, kd, a, if (ta) m else kd, b, if (tb) kd else n, c, n, in_t, out_t, 1, 0, 0, 0);
    }

    /// `batch` products y_j [t, n] = x_j [t, kd] w_j^T, the operands `sw`, `sx`, `sy` elements apart; bf16 out.
    pub fn linearBatched(k: *const Kernels, w: u64, sw: usize, x: u64, sx: usize, kd: usize, n: usize, y: u64, sy: usize, t: usize, batch: usize) void {
        if (few(t, kd, kd, x) and sx % 8 == 0) return k.gemv(w, x, kd, kd, n, t, y, n, 0, batch, sw, sx, sy);
        k.gemm(false, true, t, n, kd, x, kd, w, kd, y, n, .bf16, .bf16, batch, sx, sw, sy);
    }

    // ---- elementwise and row kernels

    pub fn rmsnorm(k: *const Kernels, x: u64, r: u64, w: u64, y: u64, d: usize, ld: usize, ldy: usize, rows: usize, eps: f32, mode: Norm) void {
        k.rmsnormPost(x, r, 0, w, 0, y, d, ld, ldy, rows, eps, mode);
    }

    /// As `rmsnorm` with a weight row a row (`ldw` elements apart).
    pub fn rmsnormRows(k: *const Kernels, x: u64, r: u64, w: u64, ldw: usize, y: u64, d: usize, ld: usize, ldy: usize, rows: usize, eps: f32, mode: Norm) void {
        k.rmsnormPost(x, r, 0, w, ldw, y, d, ld, ldy, rows, eps, mode);
    }

    /// As `rmsnormRows`, r first normed by `pw` (a sandwich layer's post norm) when given.
    pub fn rmsnormPost(k: *const Kernels, x: u64, r: u64, pw: u64, w: u64, ldw: usize, y: u64, d: usize, ld: usize, ldy: usize, rows: usize, eps: f32, mode: Norm) void {
        const A = extern struct { D: i32, ld: i32, ldy: i32, eps: f32, mode: i32, ldw: i32, has_r: i32, has_pw: i32 };
        // a few rows: a wide group each (the row's latency is all there is)
        const wide = rows <= gemv_rows and d >= 1024;
        k.g.go(if (wide) .om_rmsnorm_1k else .om_rmsnorm, .{ rows, 1, 1 }, if (wide) 1024 else 256, &.{ x, r, w, y, pw }, A{ .D = i(d), .ld = i(ld), .ldy = i(ldy), .eps = eps, .mode = @backingInt(mode), .ldw = i(ldw), .has_r = @intFromBool(r != 0), .has_pw = @intFromBool(pw != 0) });
    }

    /// q and k heads of `proj` normed and rotated, k and v written to cache row pos0 + t.
    pub fn qkPrep(k: *const Kernels, proj: u64, qn: u64, kn: u64, inv: u64, kc: u64, vc: u64, t: usize, h: usize, hkv: usize, hd: usize, ld: usize, pos0: usize, mode: u8, nk: usize, eps: f32) void {
        const A = extern struct { T: i32, H: i32, Hkv: i32, hd: i32, ld: i32, pos0: i32, mode: i32, nk: i32, eps: f32 };
        std.debug.assert(hd <= 128);
        k.g.go(.om_qk_prep, .{ t * (h + hkv), 1, 1 }, 32, &.{ proj, qn, kn, inv, kc, vc }, A{ .T = i(t), .H = i(h), .Hkv = i(hkv), .hd = i(hd), .ld = i(ld), .pos0 = i(pos0), .mode = mode, .nk = i(nk), .eps = eps });
    }

    pub fn headNorm(k: *const Kernels, x: u64, w: u64, t: usize, h: usize, hd: usize, ld: usize, eps: f32) void {
        const A = extern struct { T: i32, H: i32, hd: i32, ld: i32, eps: f32 };
        k.go(.om_head_norm, t * h * 32, 256, &.{ x, w }, A{ .T = i(t), .H = i(h), .hd = i(hd), .ld = i(ld), .eps = eps });
    }

    pub fn rope(k: *const Kernels, x: u64, inv: u64, t: usize, h: usize, hd: usize, ld: usize, pos0: usize, mode: u8) void {
        const A = extern struct { T: i32, H: i32, hd: i32, ld: i32, pos0: i32, mode: i32 };
        k.go(.om_rope, t * h * hd / 2, 256, &.{ x, inv }, A{ .T = i(t), .H = i(h), .hd = i(hd), .ld = i(ld), .pos0 = i(pos0), .mode = mode });
    }

    pub fn copyRows(k: *const Kernels, src: u64, dst: u64, t: usize, n: usize, lds: usize, ldd: usize) void {
        const A = extern struct { T: i32, n: i32, lds: i32, ldd: i32 };
        k.go(.om_copy_rows, t * n, 256, &.{ src, dst }, A{ .T = i(t), .n = i(n), .lds = i(lds), .ldd = i(ldd) });
    }

    pub fn shortConv(k: *const Kernels, bcx: u64, ld: usize, w: u64, state: u64, y: u64, ldy: usize, t: usize, d: usize, kw: usize) void {
        const A = extern struct { ld: i32, ldy: i32, T: i32, D: i32, K: i32 };
        const a: A = .{ .ld = i(ld), .ldy = i(ldy), .T = i(t), .D = i(d), .K = i(kw) };
        if (t > 4) {
            k.go(.om_short_conv_rows, t * d, 256, &.{ bcx, w, state, y }, a);
            return k.go(.om_conv_state, d, 128, &.{ bcx, state }, a);
        }
        k.go(.om_short_conv, d, 128, &.{ bcx, w, state, y }, a);
    }

    pub const Attn = struct {
        q: u64,
        ldq: usize,
        kc: u64,
        vc: u64,
        ldkv: usize,
        o: u64,
        ldo: usize,
        t: usize,
        heads: usize,
        kv_heads: usize,
        hd: usize,
        pos0: usize,
        kv_len: usize,
        window: usize,
        causal: bool,
        /// LFM3's per-head gate logits (rows `ldg` apart), applied to the output; 0: none.
        gate: u64 = 0,
        ldg: usize = 0,
    };

    pub fn attention(k: *const Kernels, p: Attn) !void {
        if (p.hd % 32 != 0 or p.hd > 128) return error.UnsupportedHeadDim;
        if (p.t == 1 and p.kv_len > 4096) return error.AttentionSpanTooLong;
        if (p.t == 1) {
            const A = extern struct { ldkv: i32, H: i32, Hkv: i32, hd: i32, pos: i32, kv_len: i32, window: i32, causal: i32, has_gate: i32, gate_off: i32, scale: f32 };
            const span = if (p.causal and p.window > 0) @min(p.window, p.kv_len) else p.kv_len;
            if (span > 4096) return error.AttentionSpanTooLong;
            // the gate's logits sit in the query's row (q | k | v | gate)
            const off: usize = if (p.gate != 0) (p.gate - p.q) / 2 else 0;
            return k.g.go(.om_attend, .{ p.heads, 1, 1 }, 256, &.{ p.q, p.kc, p.vc, p.o }, A{ .ldkv = i(p.ldkv), .H = i(p.heads), .Hkv = i(p.kv_heads), .hd = i(p.hd), .pos = i(p.pos0), .kv_len = i(p.kv_len), .window = i(p.window), .causal = @intFromBool(p.causal), .has_gate = @intFromBool(p.gate != 0), .gate_off = i(off), .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(p.hd))) });
        }
        const A = extern struct { ldq: i32, ldkv: i32, ldo: i32, T: i32, H: i32, Hkv: i32, hd: i32, pos0: i32, kv_len: i32, window: i32, causal: i32, scale: f32, has_gate: i32, ldg: i32 };
        const a: A = .{ .ldq = i(p.ldq), .ldkv = i(p.ldkv), .ldo = i(p.ldo), .T = i(p.t), .H = i(p.heads), .Hkv = i(p.kv_heads), .hd = i(p.hd), .pos0 = i(p.pos0), .kv_len = i(p.kv_len), .window = i(p.window), .causal = @intFromBool(p.causal), .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(p.hd))), .has_gate = @intFromBool(p.gate != 0), .ldg = i(p.ldg) };
        k.g.go(.om_attention_tiles, .{ blocks(p.t, 16), p.heads, 1 }, 128, &.{ p.q, p.kc, p.vc, p.o, p.gate }, a);
    }

    /// SwiGLU over rows of 2f: halves, or interleaved (a, c) pairs.
    pub fn swiglu(k: *const Kernels, ac: u64, y: u64, t: usize, f: usize, inter: bool) void {
        const A = extern struct { T: i32, F: i32, inter: i32 };
        k.go(.om_swiglu, t * f, 256, &.{ ac, y }, A{ .T = i(t), .F = i(f), .inter = @intFromBool(inter) });
    }

    pub fn biasAct(k: *const Kernels, acc: u64, bias: u64, y: u64, rows: usize, cols: usize, per_row: bool, act: Act) void {
        const A = extern struct { rows: i32, cols: i32, per_row: i32, act: i32, has_bias: i32 };
        k.go(.om_bias_act, rows * cols, 256, &.{ acc, bias, y }, A{ .rows = i(rows), .cols = i(cols), .per_row = @intFromBool(per_row), .act = @backingInt(act), .has_bias = @intFromBool(bias != 0) });
    }

    pub fn layernorm(k: *const Kernels, x: u64, w: u64, bias: u64, y: u64, d: usize, rows: usize, eps: f32) void {
        const A = extern struct { D: i32, eps: f32 };
        k.g.go(.om_layernorm, .{ rows, 1, 1 }, 256, &.{ x, w, bias, y }, A{ .D = i(d), .eps = eps });
    }

    const EmbedArgs = extern struct { T: i32, D: i32, NQ: i32, size: i32 };

    pub fn embed(k: *const Kernels, table: u64, ids: u64, y: u64, t: usize, d: usize) void {
        k.go(.om_embed, t * d, 256, &.{ table, ids, y }, EmbedArgs{ .T = i(t), .D = i(d), .NQ = 0, .size = 0 });
    }

    pub fn embedCodes(k: *const Kernels, table: u64, codes: u64, y: u64, t: usize, d: usize, nq: usize, size: usize) void {
        k.go(.om_embed_codes, t * d, 256, &.{ table, codes, y }, EmbedArgs{ .T = i(t), .D = i(d), .NQ = i(nq), .size = i(size) });
    }

    pub fn depthIn(k: *const Kernels, din: u64, mask: u64, tables: u64, codes: u64, masked: u64, x: u64, nq: usize, d: usize, size: usize) void {
        k.go(.om_depth_in, nq * d, 256, &.{ din, mask, tables, codes, masked, x }, EmbedArgs{ .T = 0, .D = i(d), .NQ = i(nq), .size = i(size) });
    }

    pub fn conv2dS2(k: *const Kernels, x: u64, cin: usize, h: usize, w: usize, wt: u64, bias: u64, y: u64, c: usize, h2: usize, w2: usize, silu: bool) void {
        const A = extern struct { CIN: i32, H: i32, W: i32, C: i32, H2: i32, W2: i32, silu: i32 };
        k.go(.om_conv2d_s2, c * h2 * w2, 256, &.{ x, wt, bias, y }, A{ .CIN = i(cin), .H = i(h), .W = i(w), .C = i(c), .H2 = i(h2), .W2 = i(w2), .silu = @intFromBool(silu) });
    }

    pub fn f32ToBf16(k: *const Kernels, x: u64, y: u64, n: usize) void {
        k.go(.om_f32_bf16, n, 256, &.{ x, y }, @as(u32, @intCast(n)));
    }

    pub fn bf16ToF32(k: *const Kernels, x: u64, y: u64, n: usize) void {
        k.go(.om_bf16_f32, n, 256, &.{ x, y }, @as(u32, @intCast(n)));
    }

    pub fn fsqDecode(k: *const Kernels, codes: u64, l: u64, n: usize, nq: usize) void {
        const A = extern struct { n: i32, NQ: i32 };
        k.go(.om_fsq_decode, n * nq, 128, &.{ codes, l }, A{ .n = i(n), .NQ = i(nq) });
    }

    pub fn repeatRows(k: *const Kernels, x: u64, y: u64, n: usize, d: usize, rep: usize) void {
        const A = extern struct { n: i32, D: i32, rep: i32 };
        k.go(.om_repeat_rows, n * rep * d, 256, &.{ x, y }, A{ .n = i(n), .D = i(d), .rep = i(rep) });
    }

    pub fn polar(k: *const Kernels, x: u64, spec: u64, t: usize, nb: usize) void {
        const A = extern struct { T: i32, NB: i32 };
        k.go(.om_polar, t * nb, 256, &.{ x, spec }, A{ .T = i(t), .NB = i(nb) });
    }

    /// The first index of the largest of n bf16 logits, `sup` excluded, in two stages through `ws` (8 bytes a part).
    pub fn argmax(k: *const Kernels, x: u64, n: usize, sup: u64, ns: usize, out: u64, ws: u64) void {
        const A = extern struct { N: i32, ns: i32, per: i32 };
        const per = 1024;
        const parts = blocks(n, per);
        k.g.go(.om_argmax_part, .{ parts, 1, 1 }, 256, &.{ x, sup, ws, ws + parts * 4 }, A{ .N = i(n), .ns = i(ns), .per = per });
        k.g.go(.om_argmax_final, .{ 1, 1, 1 }, 32, &.{ ws, ws + parts * 4, out }, @as(i32, i(parts)));
    }

    pub const Pick = struct {
        logits: u64,
        order: u64,
        p0: usize,
        n: usize,
        hist: u64,
        hn: usize,
        penalty: f32,
        temperature: f32,
        seed: u64,
        frame: usize,
        codes: u64,
        masked: u64,
        nq: usize,
        size: usize,
    };

    pub fn depthPick(k: *const Kernels, p: Pick) void {
        const A = extern struct { p0: i32, hn: i32, penalty: f32, temp: f32, seed: u64, frame: i32, NQ: i32, size: i32, pad: i32 = 0 };
        std.debug.assert(p.size <= 2048);
        k.g.go(.om_depth_pick, .{ p.n, 1, 1 }, 256, &.{ p.logits, p.order, p.hist, p.codes, p.masked }, A{ .p0 = i(p.p0), .hn = i(p.hn), .penalty = p.penalty, .temp = p.temperature, .seed = p.seed, .frame = i(p.frame), .NQ = i(p.nq), .size = i(p.size) });
    }

    /// The frame's end: all end-of-speech if any code is; else, with `push`, the frame joins the window at `slot`.
    pub fn frameEnd(k: *const Kernels, codes: u64, hist: u64, nq: usize, eos: usize, slot: usize, push: bool) void {
        const A = extern struct { NQ: i32, eos: i32, slot: i32, push: i32 };
        k.g.go(.om_frame_end, .{ 1, 1, 1 }, 32, &.{ codes, hist }, A{ .NQ = i(nq), .eos = i(eos), .slot = i(slot), .push = @intFromBool(push) });
    }

    pub fn melFrames(k: *const Kernels, x: u64, len: usize, win: u64, fr: u64, nf: usize, nfft: usize, hop: usize) void {
        const A = extern struct { len: i32, nf: i32, nfft: i32, hop: i32 };
        k.go(.om_mel_frames, nf * nfft, 256, &.{ x, win, fr }, A{ .len = i(len), .nf = i(nf), .nfft = i(nfft), .hop = i(hop) });
    }

    pub fn power(k: *const Kernels, spec: u64, p: u64, nf: usize, nb: usize) void {
        const A = extern struct { nf: i32, bins: i32 };
        k.go(.om_power, nf * nb, 256, &.{ spec, p }, A{ .nf = i(nf), .bins = i(nb) });
    }

    pub fn logmel(k: *const Kernels, m: u64, out: u64, nf: usize, mels: usize, floor: f32) void {
        const A = extern struct { nf: i32, mels: i32, floor_: f32 };
        k.go(.om_logmel, nf * mels, 256, &.{ m, out }, A{ .nf = i(nf), .mels = i(mels), .floor_ = floor });
    }
};

test "argument structs match the kernels' layouts" {
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Kernels.RowsArgs));
    const P = extern struct { p0: i32, hn: i32, penalty: f32, temp: f32, seed: u64, frame: i32, NQ: i32, size: i32, pad: i32 = 0 };
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(P));
}
