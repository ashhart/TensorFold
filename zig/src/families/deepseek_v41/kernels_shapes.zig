//! DeepSeek-V4.1 kernel shapes: attention, top-k, mHC, router and prefill GEMM args, symbols and checks.

const std = @import("std");

pub const Error = error{ Shape, Unsupported };

// -- attn_cuda.cu: dsv41_attn -----------------------------------------------------------------------------------

pub const attn = struct {
    pub const NCH = 5;
    pub const BMQ = 16;
    pub const WINDOW = 128;
    pub const NT = 256;
    pub const SMEM: u32 = 98816;

    pub const Args = extern struct {
        q: u64 = 0,
        cv: u64 = 0,
        csc: u64 = 0,
        cvs: i64 = 0,
        css: i64 = 0,
        tok: u64 = 0,
        ts: i64 = 0,
        cnt: u64 = 0,
        sv: u64 = 0,
        ssc: u64 = 0,
        svs: i64 = 0,
        sss: i64 = 0,
        lo: u64 = 0,
        hi: u64 = 0,
        pos: u64 = 0,
        sl: u64 = 0,
        pt: u64 = 0,
        pts: i64 = 0,
        psh: c_int = 0,
        sink: u64 = 0,
        cs: u64 = 0,
        csst: i64 = 0,
        po: u64 = 0,
        pm: u64 = 0,
        pl: u64 = 0,
        out: u64 = 0,
        ticket: u64 = 0,
        R: c_int = 0,
        H: c_int = 0,
        ring: c_int = 0,
        has_comp: c_int = 0,
        qrope: c_int = 0, // R1c (split_kernel only): q arrives un-rotated and is RoPE'd as it loads
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "psh") == 144 and @offsetOf(Args, "sink") == 152 and @offsetOf(Args, "qrope") == 232 and @sizeOf(Args) == 240);
    }

    pub fn symbol(comptime kw: u32, comptime rows: bool, comptime hi: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN10dsv41_attn11attn_kernelILi{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ kw, @intFromBool(rows), @intFromBool(hi) });
    }

    /// R1 split_kernel<KW, ROWS, HAS_HI, P>: P CTAs a chunk, a cluster of P (P 4); the same bits as attn_kernel.
    pub fn splitSymbol(comptime kw: u32, comptime rows: bool, comptime hi: bool, comptime p: u32) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN10dsv41_attn12split_kernelILi{d}ELb{d}ELb{d}ELi{d}EEEvNS_4ArgsE", .{ kw, @intFromBool(rows), @intFromBool(hi), p });
    }

    /// Split<P>::SMEM: P 1 the same as attn_kernel's, P 4 the S rows' B pairs beside the PV slice.
    pub fn splitSmem(p: u32) u32 {
        return if (p == 4) 100864 else SMEM;
    }

    /// The ticket's ranks a (row, head tile) once any launch splits: the ticket buffer must hold R H / 16 * MAXP.
    pub const MAXP = 4;

    /// attn_cuda.split(R) (TF_DSV41_ATTN_SPLIT on): P 4 up to `split_rows` rows (2), P 1 above.
    pub fn splitFor(rows: usize, split_rows: usize) u32 {
        return if (rows <= split_rows) 4 else 1;
    }

    /// dispatch(): the instance index (kw 4 | rows | has_hi), after its checks.
    pub fn instance(a: Args, kw: u32, rows: bool, has_hi: bool) Error!usize {
        if (a.R < 1 or @rem(a.H, BMQ) != 0 or a.ring < WINDOW or (a.ring & (a.ring - 1)) != 0) return error.Shape;
        return @as(usize, if (kw == 4) 4 else 0) + @as(usize, if (rows) 2 else 0) + @intFromBool(has_hi);
    }
};

// topk_cuda.cu: dsv41_topk
pub const topk = struct {
    pub const NT = 512;
    pub const MAXCL = 8;
    pub const MAXEPT = 31;

    pub const Job = extern struct { s: u64 = 0, ss: i64 = 0, cand: u64 = 0, cs: i64 = 0, out: u64 = 0, os: i64 = 0, cnt: u64 = 0, nk: c_int = 0, k: c_int = 0, mode: c_int = 0, ept: c_int = 0 };
    pub const Args = extern struct { job: [2]Job, pos: u64 = 0, pos64: c_int = 0, ratio: c_int = 0, bs: c_int = 0 };

    comptime {
        std.debug.assert(@sizeOf(Job) == 72 and @offsetOf(Args, "pos") == 144 and @sizeOf(Args) == 168);
    }

    pub const symbol = "_ZN10dsv41_topk11topk_kernelENS_4ArgsE";
    /// topk_b.cu (ours, TF_DSV41_INDEX_BOUND): topk_kernel with each row stopped at its own visible end, the same Args
    pub const symbol_b = "_ZN10dsv41_topk13topk_b_kernelENS_4ArgsE";

    pub fn smem(ept: usize) u32 {
        return @intCast(NT * ept * 4);
    }
};

// -- mhc_cuda.cu: dsv41_mhc (decode boundary, <= 16 rows) -------------------------------------------------------

pub const mhc = struct {
    pub const NB = 40;
    pub const NT = 128;
    pub const MAXR = 16;
    pub const ROWB = 128;
    /// coef_kernel is a CTA a row: TF_DSV41_MHC_PFDEC launches it past MAXR, up to a decode window's 64 rows
    pub const COEF_ROWS = 64;

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        xout: u64 = 0,
        g: u64 = 0,
        gr: i64 = 0,
        world: c_int = 1,
        gbf16: c_int = 0,
        post: u64 = 0,
        comb: u64 = 0,
        pre: u64 = 0,
        @"fn": u64 = 0,
        base: u64 = 0,
        scale: u64 = 0,
        part: u64 = 0,
        c: u64 = 0,
        tap: u64 = 0,
        ts: i64 = 0,
        nw: u64 = 0,
        nwbf16: c_int = 0,
        out: u64 = 0,
        opre: u64 = 0,
        opost: u64 = 0,
        ocomb: u64 = 0,
        cnt: u64 = 0,
        R: c_int = 0,
        eps: f32 = 0,
        hc_eps: f32 = 0,
        post_alpha: f32 = 0,
        iters: c_int = 0,
        spin: i64 = -1, // R1 tail: SM cycles a CTA waits for the last arrival (the binding's -1: today's finish)
        @"defer": c_int = 0, // R1 tail: the coefficient items run in coef_kernel on a side stream
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "post") == 48 and @offsetOf(Args, "nwbf16") == 136 and @offsetOf(Args, "R") == 184 and @offsetOf(Args, "spin") == 208 and @sizeOf(Args) == 224);
    }

    pub const coef_symbol = "_ZN9dsv41_mhc11coef_kernelENS_4ArgsE";

    /// (POST, COLL, MIX, ALL4, FN32) of dispatch's cases, by mode * 2 + fn32 (mode 3 ignores fn32).
    pub const instances = [_]struct { post: bool, coll: u32, mix: bool, all4: bool, fn32: bool }{
        .{ .post = true, .coll = 2, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = true, .coll = 2, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = false, .coll = 1, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = false, .coll = 1, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = false, .coll = 2, .mix = true, .all4 = false, .fn32 = false },
        .{ .post = false, .coll = 2, .mix = true, .all4 = false, .fn32 = true },
        .{ .post = true, .coll = 2, .mix = false, .all4 = true, .fn32 = false },
    };

    /// boundary_kernel<POST, COLL, MIX, ALL4, FN32, TAIL> (TAIL: R1's parallel tail, default false: still mangled)
    pub fn symbol(comptime i: usize, comptime tail: bool) [:0]const u8 {
        const s = instances[i];
        return std.fmt.comptimePrint("_ZN9dsv41_mhc15boundary_kernelILb{d}ELi{d}ELb{d}ELb{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ @intFromBool(s.post), s.coll, @intFromBool(s.mix), @intFromBool(s.all4), @intFromBool(s.fn32), @intFromBool(tail) });
    }

    /// dispatch(): mode 0 boundary, 1 site from stream 0, 2 site with a carried pre-mix, 3 final.
    pub fn instance(a: Args, mode: u32, fn32: bool) Error!usize {
        if (a.R < 1 or a.R > MAXR or a.world < 1 or a.world > 8) return error.Shape;
        if (mode == 3) return 6;
        if (mode > 2) return error.Unsupported;
        return mode * 2 + @intFromBool(fn32);
    }
};

// mhc_pf.cu: dsv41_mhc_pf (prefill site, 16 rows a CTA)
pub const mhc_pf = struct {
    pub const NB = 40;
    pub const NT = 128;
    pub const BM = 16;
    pub const MAXW = 8;

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        xout: u64 = 0,
        g: u64 = 0,
        gr: i64 = 0,
        world: c_int = 1,
        gbf16: c_int = 0,
        post: u64 = 0,
        comb: u64 = 0,
        pre: u64 = 0,
        @"fn": u64 = 0,
        part: u64 = 0,
        c: u64 = 0,
        tap: u64 = 0,
        ts: i64 = 0,
        R: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "R") == 112 and @sizeOf(Args) == 120);
    }

    /// Lay<FN32>::BYTES: staged streams, the collapsed rows and fn.
    pub fn smem(fn32: bool) u32 {
        const pb = 128 + 8;
        const pf = 128 + 4;
        const xs = 4 * BM * pb * 2;
        const cs = BM * pb * 2;
        const fnb: u32 = if (fn32) 4 * 24 * pf * 4 else 4 * 24 * pb * 2;
        return xs + cs + fnb;
    }

    /// (POST, COLL, FN32, TAP) by dispatch's case (mode * 4 + fn32 * 2 + tap).
    pub const instances = [_]struct { case: u32, post: bool, coll: u32, fn32: bool, tap: bool }{
        .{ .case = 0, .post = true, .coll = 2, .fn32 = false, .tap = false },
        .{ .case = 1, .post = true, .coll = 2, .fn32 = false, .tap = true },
        .{ .case = 2, .post = true, .coll = 2, .fn32 = true, .tap = false },
        .{ .case = 3, .post = true, .coll = 2, .fn32 = true, .tap = true },
        .{ .case = 4, .post = false, .coll = 1, .fn32 = false, .tap = false },
        .{ .case = 6, .post = false, .coll = 1, .fn32 = true, .tap = false },
        .{ .case = 8, .post = false, .coll = 2, .fn32 = false, .tap = false },
        .{ .case = 10, .post = false, .coll = 2, .fn32 = true, .tap = false },
    };

    pub fn symbol(comptime i: usize) [:0]const u8 {
        const s = instances[i];
        return std.fmt.comptimePrint("_ZN12dsv41_mhc_pf11site_kernelILb{d}ELi{d}ELb{d}ELb{d}EEEvNS_4ArgsE", .{ @intFromBool(s.post), s.coll, @intFromBool(s.fn32), @intFromBool(s.tap) });
    }

    pub fn instance(a: Args, mode: u32, fn32: bool, tap: bool) Error!usize {
        if (a.R < 1 or a.world < 1 or a.world > MAXW or (tap and mode != 0)) return error.Shape;
        const case = mode * 4 + @as(u32, if (fn32) 2 else 0) + @intFromBool(tap);
        for (instances, 0..) |s, i| if (s.case == case) return i;
        return error.Unsupported;
    }
};

// -- router_gemv.cu: dsv41_rg -----------------------------------------------------------------------------------

pub const rg = struct {
    pub const D = 5120;
    pub const EPL = 16;
    pub const KMAX = 8;
    pub const GE_MAX = 512;
    pub const rts = [_]u32{ 1, 2, 4, 8, 16 };
    pub const wide = [_][2]u32{ .{ 8, 1 }, .{ 4, 1 }, .{ 8, 2 }, .{ 4, 2 } };
    pub const narrow = [_]u32{ 1, 2, 4, 8 };

    pub const Args = extern struct {
        x: u64 = 0,
        xs: i64 = 0,
        w: u64 = 0,
        bias: u64 = 0,
        pick: u64 = 0,
        wts: u64 = 0,
        lg: u64 = 0,
        cnt: u64 = 0,
        R: c_int = 0,
        E: c_int = 0,
        K: c_int = 0,
        slots: c_int = 0,
        select: c_int = 0,
        scale: f32 = 0,
        kit: c_int = 0,
        gid: u64 = 0,
        gcnt: u64 = 0,
        gmem: u64 = 0,
        GE: c_int = 0,
        maxm: c_int = 0,
        group: c_int = 0,
        prune: Prune = .{}, // R1c: the prune rule folded into the selection (on 0: off)
    };

    /// router_gemv.cu Prune: [min_w, topp, scale, min_k, orig, drop, dup] (prune.fold_args) in fp32 / int, on = 1.
    pub const Prune = extern struct { min_w: f32 = 0, topp: f32 = 0, scale: f32 = 0, min_k: c_int = 0, orig: c_int = 0, drop: c_int = 0, dup: c_int = 0, on: c_int = 0 };

    /// The binding's prune vector (7 doubles, empty: off) as the kernel's struct (fp32 thresholds, as it rounds them).
    pub fn pruneOf(v: []const f64) error{Shape}!Prune {
        if (v.len == 0) return .{};
        if (v.len != 7) return error.Shape;
        return .{ .min_w = @floatCast(v[0]), .topp = @floatCast(v[1]), .scale = @floatCast(v[2]), .min_k = @intFromFloat(v[3]), .orig = @intFromFloat(v[4]), .drop = @intFromFloat(v[5]), .dup = @intFromFloat(v[6]), .on = 1 };
    }

    comptime {
        std.debug.assert(@sizeOf(Prune) == 32);
        std.debug.assert(@offsetOf(Args, "gid") == 96 and @offsetOf(Args, "group") == 128 and @offsetOf(Args, "prune") == 132 and @sizeOf(Args) == 168);
    }

    pub fn gemvSymbol(comptime rt: u32, comptime nw: u32, comptime ew: u32) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN8dsv41_rg11gemv_kernelILi{d}ELi{d}ELi{d}EEEvNS_4ArgsE", .{ rt, nw, ew });
    }

    pub fn narrowSymbol(comptime rt: u32, comptime nw: u32) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN8dsv41_rg13narrow_kernelILi{d}ELi{d}EEEvNS_4ArgsE", .{ rt, nw });
    }

    /// dsv41_rg_route_cuda's checks (row stride / alignment of x and the buffer sizes are the caller's).
    pub fn check(a: Args, rt: u32, nw: u32, ew: u32, lg_len: usize, cnt_len: usize) Error!void {
        const E: usize = @intCast(a.E);
        const R: usize = @intCast(a.R);
        if (E < 1 or E > 32 * EPL or a.K < 1 or a.K > KMAX or a.K > a.E or a.slots < a.K or a.slots > a.K + 1) return error.Shape;
        if (lg_len < R * E) return error.Shape;
        if (a.select != 0 and cnt_len < (R + rt - 1) / rt) return error.Shape;
        const is_wide = (nw == 4 or nw == 8) and (ew == 1 or ew == 2);
        const is_narrow = ew == 0 and (nw == 1 or nw == 2 or nw == 4 or nw == 8);
        if (std.mem.indexOfScalar(u32, &rts, rt) == null or !(is_wide or is_narrow)) return error.Unsupported;
        if (a.group != 0 and (a.select == 0 or !is_narrow or R > rt or a.GE < 1 or a.GE > GE_MAX)) return error.Shape;
    }
};

// -- pfdense.cuh: dsv41_pfd (fused prefill dense GEMM, one fatbin a width) --------------------------------------

pub const pfd = struct {
    pub const k2s = [_]u32{ 8, 10, 12, 16 };
    /// DSV41_PFD_CFGS: (BM, WM, WN, KS, NST) by id.
    pub const cfgs = [_][5]u32{ .{ 128, 2, 4, 2, 3 }, .{ 128, 2, 4, 2, 4 }, .{ 128, 4, 2, 2, 3 }, .{ 64, 2, 2, 2, 4 }, .{ 64, 1, 4, 2, 4 }, .{ 128, 2, 4, 2, 2 }, .{ 32, 1, 4, 2, 4 }, .{ 256, 4, 2, 2, 3 } };
    pub const OutType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

    pub const Args = extern struct {
        xh: u64 = 0,
        T: u64 = 0,
        stride_k: i64 = 0,
        stride_nb: i64 = 0,
        svh: u64 = 0,
        bias: u64 = 0,
        out: u64 = 0,
        o_stride: i64 = 0,
        M: c_int = 0,
        K: c_int = 0,
        N: c_int = 0,
        group: c_int = 0,
        out_type: c_int = 0,
    };

    comptime {
        std.debug.assert(@offsetOf(Args, "M") == 64 and @sizeOf(Args) == 88);
    }

    pub fn threads(cfg: usize) u32 {
        return cfgs[cfg][1] * cfgs[cfg][2] * 32;
    }

    /// Lay<K2, LANES, Cfg>::SMEM (the layout does not depend on LANES).
    pub fn smem(cfg: usize, k2: u32) u32 {
        const bm, const wm, _, const ks, const nst = cfgs[cfg];
        const a_bytes = bm * (16 * ks) * 2;
        const w_bytes = ks * 8 * (4 * k2) * 4;
        const ring = nst * (a_bytes + w_bytes);
        const main = ring + 2 * (ks * 8 * 32 * 16);
        const epi = (bm / wm) * 128 * 2 * 2;
        return @max(main, epi);
    }

    pub fn symbol(comptime k2: u32, comptime lanes: bool, comptime cfg: usize) [:0]const u8 {
        const c = cfgs[cfg];
        return std.fmt.comptimePrint("_ZN9dsv41_pfd10pfd_kernelILi{d}ELb{d}ENS_3CfgILi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEEEvNS_4ArgsE", .{ k2, @intFromBool(lanes), c[0], c[1], c[2], c[3], c[4] });
    }

    pub fn dequantSymbol(comptime k2: u32, comptime lanes: bool) [:0]const u8 {
        return std.fmt.comptimePrint("_ZN9dsv41_pfd18pfd_dequant_kernelILi{d}ELb{d}EEEvPKjxxP6__halfi", .{ k2, @intFromBool(lanes) });
    }

    /// pfdense.cpp gemm's checks on the shapes and strides (dtype / alignment are the caller's).
    pub fn check(a: Args, k2: u32, lanes: bool, cfg: usize, words: usize) Error!void {
        if (std.mem.indexOfScalar(u32, &k2s, k2) == null or (lanes and k2 == 16)) return error.Unsupported;
        if (cfg >= cfgs.len) return error.Unsupported;
        const ks: i64 = cfgs[cfg][3];
        const K: i64 = a.K;
        const N: i64 = a.N;
        if (a.M < 1 or @rem(K, 128) != 0 or K < 128 or @rem(K, 16 * ks) != 0) return error.Shape;
        if (@rem(N, 128) != 0 or N < 128 or @rem(a.o_stride, 2) != 0 or a.o_stride < N) return error.Shape;
        const tw: i64 = 4 * @as(i64, k2);
        if (@rem(a.stride_k, 4) != 0 or @rem(a.stride_nb, 4) != 0 or a.stride_k < 8 * tw or a.stride_nb < 8 * tw) return error.Shape;
        const last = (@divTrunc(K, 16) - 1) * a.stride_k + (@divTrunc(N, 128) - 1) * a.stride_nb + 8 * tw;
        if (last > @as(i64, @intCast(words))) return error.Shape;
        if (a.group < 1 or a.group > 1024) return error.Shape;
    }

    /// pfdense.heuristic (no sweep table): the largest row tile whose last wave is not mostly empty.
    pub fn heuristic(k: usize, n: usize, k2: u32, m: usize, sms: usize) [2]usize {
        const nb = n / 128;
        var best: ?usize = null;
        var best_eff: f64 = -1.0;
        for ([_]usize{ 0, 3, 6 }) |cfg| {
            if (k % (16 * cfgs[cfg][3]) != 0) continue;
            const bm = cfgs[cfg][0];
            const blocks: usize = if (2 * (smem(cfg, k2) + 1024) <= 102400) 2 else 1;
            const ctas = ((m + bm - 1) / bm) * nb;
            const slots = sms * blocks;
            const eff = @as(f64, @floatFromInt(ctas)) / @as(f64, @floatFromInt(((ctas + slots - 1) / slots) * slots));
            if (eff > best_eff + 0.10) {
                best = cfg;
                best_eff = eff;
            }
        }
        return .{ best orelse 0, 8 };
    }
};
