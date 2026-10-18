//! DeepSeek-V4.1's other CUDA kernels (attention, top-k, mHC, router, prefill GEMM, glue): args and launches.

const std = @import("std");
const cuda = @import("cuda");
const shapes = @import("kernels_shapes.zig");
const glue_ops = @import("kernels_glue_ops.zig");
const fitShared = @import("cuda").qlinear.block.experts.fitShared;

fn dim(x: usize) u32 {
    return @intCast(x);
}

pub const Error = shapes.Error;
pub const attn = shapes.attn;
pub const topk = shapes.topk;
pub const mhc = shapes.mhc;
pub const mhc_pf = shapes.mhc_pf;
pub const rg = shapes.rg;
pub const pfd = shapes.pfd;

// -- Loaded functions and launches ------------------------------------------------------------------------------

pub const Functions = struct {
    attn: [8]cuda.Function,
    attn_split: [2][8]cuda.Function, // [P 1, P 4][instance]
    topk: cuda.Function,
    /// topk_b.cu's row-bounded twin (tf_dsv41_topk_b_v1.topk)
    topk_b: cuda.Function,
    mhc: [2][mhc.instances.len]cuda.Function, // [tail][instance]
    mhc_coef: cuda.Function,
    mhc_pf: [mhc_pf.instances.len]cuda.Function,
    gemv: [rg.rts.len][rg.wide.len]cuda.Function,
    narrow: [rg.rts.len][rg.narrow.len]cuda.Function,
    pfd: [pfd.k2s.len][2][pfd.cfgs.len]?cuda.Function, // [k2][lanes][cfg]; K2 16 has no lanes kernels
    pfd_dequant: [pfd.k2s.len][2]?cuda.Function,
    gate_copy: cuda.Function,
    paced: cuda.Function,
    segments: [2]cuda.Function, // bulk false / true
    topk_keys: cuda.Function,
    swiglu: cuda.Function,
    add: cuda.Function,
    add_bf16: cuda.Function,
    kv_dense: cuda.Function,
    /// kvsplit's compact pack (TF_DSV41_KV_SPLIT_COMPACT); null in a fatbin built before it
    kv_pack: ?cuda.Function = null,
    /// kvsplit's dedup steps (TF_DSV41_KV_SPLIT_DEDUP: mark, flags, pack, tok); null in a fatbin built before them
    kv_dedup: ?[4]cuda.Function = null,
    kv_hist: ?[2]cuda.Function = null,
    kv_gather: cuda.Function,
    glue: [10]cuda.Function, // glue.syms order
    /// glue.cu's DSpark candidates (TF_DSV41_DRAFT_GRAPHS); null in a fatbin built before it (the host's candidates)
    ds_cands: ?cuda.Function = null,
    /// glue.cu's device pick (TF_DSV41_GREEDY_GPU); null in a fatbin built before it (the pairs merged on the host)
    pick_pack: ?cuda.Function = null,
    pick_merge: ?cuda.Function = null,
    /// glue.cu's DSpark statics from the device pick (the speculative pass's device start); optional, as ds_cands
    ds_stage: ?cuda.Function = null,
    /// glue.cu's row-window accept and several-slot statics (the slot passes' device start); optional
    ds_accept_rows: ?cuda.Function = null,
    ds_stage_rows: ?cuda.Function = null,
    /// glue.cu's device keyed choice (vsample.choose; the sampled rounds' device start); optional
    vs_choose: ?cuda.Function = null,
    pfglue: [pfglue.syms.len]cuda.Function, // pfglue.syms order

    /// `pf`: the four pfdense modules (K2 8, 10, 12, 16).
    pub fn resolve(m_attn: cuda.Module, m_topk: cuda.Module, m_mhc: cuda.Module, m_mhc_pf: cuda.Module, m_rg: cuda.Module, pf: [4]cuda.Module, m_gate: cuda.Module, m_pace: cuda.Module, m_l2pf: cuda.Module, m_keys: cuda.Module, m_pw: cuda.Module, m_kv: cuda.Module, m_glue: cuda.Module, m_pfglue: cuda.Module, m_topk_b: cuda.Module, optin: u32) !Functions {
        var f: Functions = undefined;
        inline for (0..8) |i| f.attn[i] = try m_attn.function(attn.symbol(if (i >= 4) 4 else 2, (i & 2) != 0, (i & 1) != 0));
        inline for (f.attn, 0..) |fun, i| {
            if (!try fitShared(fun, attn.symbol(if (i >= 4) 4 else 2, (i & 2) != 0, (i & 1) != 0), attn.SMEM, optin)) return error.SharedMemoryTooSmall;
        }
        inline for (.{ 1, 4 }, 0..) |p, pi| inline for (0..8) |i| {
            const name = attn.splitSymbol(if (i >= 4) 4 else 2, (i & 2) != 0, (i & 1) != 0, p);
            f.attn_split[pi][i] = try m_attn.function(name);
            if (!try fitShared(f.attn_split[pi][i], name, attn.splitSmem(p), optin)) return error.SharedMemoryTooSmall;
        };
        f.topk = try m_topk.function(topk.symbol);
        if (!try fitShared(f.topk, topk.symbol, topk.smem(topk.MAXEPT), optin)) return error.SharedMemoryTooSmall;
        f.topk_b = try m_topk_b.function(topk.symbol_b);
        if (!try fitShared(f.topk_b, topk.symbol_b, topk.smem(topk.MAXEPT), optin)) return error.SharedMemoryTooSmall;
        inline for (.{ false, true }, 0..) |tail, ti| inline for (0..mhc.instances.len) |i| {
            f.mhc[ti][i] = try m_mhc.function(mhc.symbol(i, tail));
        };
        f.mhc_coef = try m_mhc.function(mhc.coef_symbol);
        inline for (0..mhc_pf.instances.len) |i| {
            f.mhc_pf[i] = try m_mhc_pf.function(mhc_pf.symbol(i));
            if (!try fitShared(f.mhc_pf[i], mhc_pf.symbol(i), mhc_pf.smem(mhc_pf.instances[i].fn32), optin)) return error.SharedMemoryTooSmall;
        }
        inline for (rg.rts, 0..) |rt, i| {
            inline for (rg.wide, 0..) |c, j| f.gemv[i][j] = try m_rg.function(rg.gemvSymbol(rt, c[0], c[1]));
            inline for (rg.narrow, 0..) |nw, j| f.narrow[i][j] = try m_rg.function(rg.narrowSymbol(rt, nw));
        }
        inline for (pfd.k2s, 0..) |k2, ki| inline for (.{ false, true }, 0..) |lanes, li| {
            inline for (0..pfd.cfgs.len) |c| {
                if (lanes and k2 == 16) {
                    f.pfd[ki][li][c] = null;
                } else {
                    const fun = try pf[ki].function(pfd.symbol(k2, lanes, c));
                    f.pfd[ki][li][c] = if (try fitShared(fun, pfd.symbol(k2, lanes, c), pfd.smem(c, k2), optin)) fun else null;
                }
            }
            f.pfd_dequant[ki][li] = if (lanes and k2 == 16) null else try pf[ki].function(pfd.dequantSymbol(k2, lanes));
        };
        f.gate_copy = try m_gate.function("_ZN17dsv41_engram_gate16gate_copy_kernelEPxiiPKhxPhxxi");
        f.paced = try m_pace.function("_ZN12dsv41_l2pace12paced_kernelEPKlijmm");
        f.segments = .{ try m_l2pf.function("_ZN6tfl2pf15segments_kernelILb0EEEvPKlij"), try m_l2pf.function("_ZN6tfl2pf15segments_kernelILb1EEEvPKlij") };
        f.topk_keys = try m_keys.function("_ZN15dsv41_topk_keys11topk_kernelEPKfxiiPfPx");
        f.swiglu = try m_pw.function("_ZN15dsv41_pointwise13swiglu_kernelEPKfxS1_xPfxif");
        f.add = try m_pw.function("_ZN15dsv41_pointwise10add_kernelEPfPKfx");
        f.add_bf16 = try m_pw.function("_ZN15dsv41_pointwise15add_bf16_kernelEPKfS1_P13__nv_bfloat16x");
        f.kv_dense = try m_kv.function("_ZN13dsv41_kvsplit12dense_kernelENS_9DenseArgsE");
        f.kv_gather = try m_kv.function("_ZN13dsv41_kvsplit13gather_kernelEPKhjPKjjPh");
        f.kv_pack = m_kv.function("_ZN13dsv41_kvsplit11pack_kernelENS_8PackArgsE") catch null;
        f.kv_dedup = kvsplit.dedupFns(m_kv) catch null;
        f.kv_hist = kvsplit.histFns(m_kv) catch null;
        inline for (glue.syms, 0..) |name, i| f.glue[i] = try m_glue.function(name);
        f.ds_cands = m_glue.function(glue.ds_cands_sym) catch null;
        f.pick_pack = m_glue.function(glue.pick_pack_sym) catch null;
        f.pick_merge = m_glue.function(glue.pick_merge_sym) catch null;
        f.ds_stage = m_glue.function(glue.ds_stage_sym) catch null;
        f.ds_accept_rows = m_glue.function(glue.ds_accept_rows_sym) catch null;
        f.ds_stage_rows = m_glue.function(glue.ds_stage_rows_sym) catch null;
        f.vs_choose = m_glue.function(glue.vs_choose_sym) catch null;
        inline for (pfglue.syms, 0..) |name, i| f.pfglue[i] = try m_pfglue.function(name);
        return f;
    }
};

/// kvsplit.cu: split KV's exchange copies (kv/split.zig's DenseArgs, the same layout; HostKernels the reference).
pub const kvsplit = struct {
    pub const DenseArgs = extern struct {
        sel: u64,
        rows: u32,
        k: u32,
        table: u64,
        pts: u32,
        rslot: u64,
        psh: u32,
        base: u64,
        row_bytes: u32,
        world: u32,
        send: u64,
        tok: u64,
    };

    /// kv/split.zig's PackArgs (DenseArgs' fields at the same offsets, rank in psh's padding, lens after tok).
    pub const PackArgs = extern struct {
        sel: u64,
        rows: u32,
        k: u32,
        table: u64,
        pts: u32,
        rslot: u64,
        psh: u32,
        rank: u32,
        base: u64,
        row_bytes: u32,
        world: u32,
        send: u64,
        tok: u64,
        lens: u64,
    };

    /// kv/split.zig's DedupArgs: PackArgs, then the map and the scratch.
    pub const DedupArgs = extern struct {
        p: PackArgs,
        map: u64,
        first: u64,
        place: u64,
        bcnt: u64,
    };

    const dedup_syms = [4][:0]const u8{
        "_ZN13dsv41_kvsplit17dedup_mark_kernelENS_9DedupArgsE",
        "_ZN13dsv41_kvsplit18dedup_flags_kernelENS_9DedupArgsE",
        "_ZN13dsv41_kvsplit17dedup_pack_kernelENS_9DedupArgsE",
        "_ZN13dsv41_kvsplit16dedup_tok_kernelENS_9DedupArgsE",
    };

    fn dedupFns(m: cuda.Module) ![4]cuda.Function {
        var out: [4]cuda.Function = undefined;
        for (dedup_syms, &out) |name, *f| f.* = try m.function(name);
        return out;
    }

    /// kv/split.zig's HistArgs (the history union's gather and tokens).
    pub const HistArgs = extern struct {
        sel: u64,
        rows: u32,
        k: u32,
        table: u64,
        pages: u32,
        psh: u32,
        rank: u32,
        world: u32,
        m: u32,
        row_bytes: u32,
        base: u64,
        send: u64,
        tok: u64,
    };

    const hist_syms = [2][:0]const u8{ "_ZN13dsv41_kvsplit18hist_gather_kernelENS_8HistArgsE", "_ZN13dsv41_kvsplit15hist_tok_kernelENS_8HistArgsE" };

    fn histFns(m: cuda.Module) ![2]cuda.Function {
        return .{ try m.function(hist_syms[0]), try m.function(hist_syms[1]) };
    }

    comptime {
        std.debug.assert(@sizeOf(DedupArgs) == 120 and @offsetOf(DedupArgs, "map") == 88 and @offsetOf(DedupArgs, "bcnt") == 112);
        std.debug.assert(@sizeOf(HistArgs) == 72 and @offsetOf(HistArgs, "m") == 40 and @offsetOf(HistArgs, "base") == 48);
        std.debug.assert(@sizeOf(DenseArgs) == 80 and @offsetOf(DenseArgs, "rslot") == 32 and @offsetOf(DenseArgs, "send") == 64);
        std.debug.assert(@sizeOf(PackArgs) == 88 and @offsetOf(PackArgs, "rank") == 44 and @offsetOf(PackArgs, "lens") == 80);
    }

    fn grid(words: u64) usize {
        return @intCast(@max(1, @min((words + 255) / 256, 4096)));
    }
};

/// glue.cu: the window's torch glue, torch's arithmetic bit for bit.
pub const glue = struct {
    pub const syms = [_][:0]const u8{
        "_ZN10dsv41_glue17embed_send_kernelEPKxiPKtxiiPt",
        "_ZN10dsv41_glue16embed_sum_kernelEPKtiiiPt",
        "_ZN10dsv41_glue16cast_bf16_kernelEPKfPtx",
        "_ZN10dsv41_glue18kit_weights_kernelEPKfPfx",
        "_ZN10dsv41_glue17kit_logits_kernelEPfx",
        "_ZN10dsv41_glue16positions_kernelEiiPiPxS0_",
        "_ZN10dsv41_glue18gather_cols_kernelEPKtiiiPt",
        "_ZN10dsv41_glue12carry_kernelEPKtxPKiiiPf",
        "_ZN10dsv41_glue20positions_dev_kernelEPKiiPiPxS2_",
        "_ZN10dsv41_glue16widen_f32_kernelEPKtPfx",
    };
    pub const Kind = enum(u4) { embed_send, embed_sum, cast_bf16, kit_weights, kit_logits, positions, gather_cols, carry, positions_dev, widen_f32 };
    /// DSpark's candidates (optional: resolved apart, so an older fatbin still loads)
    pub const ds_cands_sym: [:0]const u8 = "_ZN10dsv41_glue15ds_cands_kernelEPKfxiiiPf";
    /// the most candidates a rank (the kernel's shared sort)
    pub const ds_cands_max = 128;
    /// DSpark's statics from the device pick (optional, as ds_cands)
    pub const ds_stage_sym: [:0]const u8 = "_ZN10dsv41_glue15ds_stage_kernelEPKxiPxPiiiixiiiiiii";
    pub const ds_accept_rows_sym: [:0]const u8 = "_ZN10dsv41_glue21ds_accept_rows_kernelEPKxS1_S1_iPx";
    pub const ds_stage_rows_sym: [:0]const u8 = "_ZN10dsv41_glue20ds_stage_rows_kernelEPKxS1_iPxPiiiiiiiiiii";
    pub const vs_choose_sym: [:0]const u8 = "_ZN10dsv41_glue16vs_choose_kernelEPKfiiiiyxidddPx";
    /// the candidates vs_choose merges at most (W k)
    pub const vs_max = 2048;
    /// the device pick (optional, as ds_cands)
    pub const pick_pack_sym: [:0]const u8 = "_ZN10dsv41_glue16pick_pack_kernelEPKfPKxixPd";
    pub const pick_merge_sym: [:0]const u8 = "_ZN10dsv41_glue17pick_merge_kernelEPKdiiPKxxPx";
    /// the most rows pick_merge takes (one block)
    pub const pick_max_rows = 1024;

    pub fn grid(n: u64) usize {
        return @intCast(@max(1, @min((n + 255) / 256, 4096)));
    }
};

/// prefill_glue.cu: the prefill segment's torch glue (block_prefill.zig's glue steps), torch's integers and casts.
pub const pfglue = struct {
    pub const syms = [_][:0]const u8{
        "_ZN12dsv41_pfglue20top_positions_kernelEPKxxiiPi",
        "_ZN12dsv41_pfglue16cand_keys_kernelEPKixiiiPi",
        "_ZN12dsv41_pfglue13counts_kernelEPKiiPKxiiPi",
        "_ZN12dsv41_pfglue15gm_picks_kernelEPKiPKfiiiPiPf",
        "_ZN12dsv41_pfglue17width_mask_kernelEPKixS1_iiPi",
        "_ZN12dsv41_pfglue19f64_bf16_f32_kernelEPKdPfx",
        "_ZN12dsv41_pfglue16widen_cat_kernelEPKtS1_iiPf",
        "_ZN12dsv41_pfglue16ring_copy_kernelEPKhiPhixxx",
    };
    pub const Kind = enum(u3) { top_positions, cand_keys, counts, gm_picks, width_mask, f64_bf16_f32, widen_cat, ring_copy };
    /// index.top_positions' largest `count` (TOPK 512, TOPK_BLOCKS 2,048)
    pub const max_count = 2048;
};

/// topk_keys.cu: pick.top's bounds (C columns a row within pick.COLS, k within the shared sort).
pub const topk_keys = struct {
    pub const max_cols = 1 << 16;
    pub const max_k = 1024;
};

pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,

    pub fn go(o: Ops, f: cuda.Function, grid: [3]usize, block: usize, shared: u32, cluster: ?cuda.Dim3, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = dim(grid[0]), .y = dim(grid[1]), .z = dim(grid[2]) }, .block = .{ .x = dim(block) }, .shared = shared, .cluster = cluster }, o.s, a);
    }

    /// dsv41_attn_run_cuda: `rows` slots table given, `has_hi` window ends given, `kw` Triton k width, `split` R1 CTAs.
    pub fn attention(o: Ops, a: attn.Args, kw: u32, rows: bool, has_hi: bool, split: u32) !void {
        const i = try attn.instance(a, kw, rows, has_hi);
        if (a.qrope != 0 and split == 0) return error.Unsupported; // "q's RoPE folds into the split kernel only"
        var args: cuda.Args = .{};
        args.add(a);
        const hb: usize = @intCast(@divTrunc(a.H, attn.BMQ));
        const R: usize = @intCast(a.R);
        switch (split) {
            0 => try o.go(o.f.attn[i], .{ attn.NCH, hb, R }, attn.NT, attn.SMEM, null, &args),
            1 => try o.go(o.f.attn_split[0][i], .{ attn.NCH, hb, R }, attn.NT, attn.splitSmem(1), null, &args),
            4 => try o.go(o.f.attn_split[1][i], .{ attn.NCH * 4, hb, R }, attn.NT, attn.splitSmem(4), .{ .x = 4 }, &args),
            else => return error.Unsupported,
        }
    }

    /// dsv41_topk_run_cuda: `jobs` 1-2 selections, R rows, CL CTAs a cluster, ept entries a thread.
    pub fn topK(o: Ops, a: topk.Args, jobs: usize, R: usize, CL: usize, ept: usize) !void {
        return o.topKOf(o.f.topk, a, jobs, R, CL, ept);
    }

    /// topk_b.cu (TF_DSV41_INDEX_BOUND): topK's launch of the row-bounded twin (the same args and bytes).
    pub fn topKB(o: Ops, a: topk.Args, jobs: usize, R: usize, CL: usize, ept: usize) !void {
        return o.topKOf(o.f.topk_b, a, jobs, R, CL, ept);
    }

    fn topKOf(o: Ops, f: cuda.Function, a: topk.Args, jobs: usize, R: usize, CL: usize, ept: usize) !void {
        if (R < 1 or jobs < 1 or jobs > 2 or CL < 1 or CL > topk.MAXCL or ept < 1 or ept > topk.MAXEPT) return error.Shape;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(f, .{ CL, R, jobs }, topk.NT, topk.smem(ept), .{ .x = dim(CL) }, &args);
    }

    /// dsv41_mhc_run_cuda (`fn32`: fn is fp32); `a.spin` >= 0 R1's parallel tail, `a.defer` 1 leaves coefs to mhcCoef.
    pub fn mhcBoundary(o: Ops, a: mhc.Args, mode: u32, fn32: bool) !void {
        const i = try mhc.instance(a, mode, fn32);
        var b = a;
        if (b.spin < 0) b.@"defer" = 0; // as the binding: defer only with the tail
        var args: cuda.Args = .{};
        args.add(b);
        try o.go(o.f.mhc[@intFromBool(b.spin >= 0)][i], .{ mhc.NB, if (mhc.instances[i].all4) 1 else 4, 1 }, mhc.NT, @intCast(@as(u32, @intCast(a.R)) * mhc.ROWB), null, &args);
    }

    /// dsv41_mhc_coef_cuda (R1): the next site's coefficients of a deferred boundary, a CTA a row.
    pub fn mhcCoef(o: Ops, a: mhc.Args) !void {
        if (a.R < 1 or a.R > mhc.COEF_ROWS) return error.Shape;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.mhc_coef, .{ @intCast(a.R), 1, 1 }, mhc.NT, 0, null, &args);
    }

    /// dsv41_mhc_pf_run_cuda: `tap` = a tap buffer is given.
    pub fn mhcSite(o: Ops, a: mhc_pf.Args, mode: u32, fn32: bool, tap: bool) !void {
        const i = try mhc_pf.instance(a, mode, fn32, tap);
        var args: cuda.Args = .{};
        args.add(a);
        const R: usize = @intCast(a.R);
        try o.go(o.f.mhc_pf[i], .{ mhc_pf.NB, (R + mhc_pf.BM - 1) / mhc_pf.BM, 1 }, mhc_pf.NT, mhc_pf.smem(mhc_pf.instances[i].fn32), null, &args);
    }

    /// dsv41_rg_route_cuda: (rt, nw, ew) a wide instance, or ew 0 the narrow kernel (default carveout).
    pub fn route(o: Ops, a: rg.Args, rt: u32, nw: u32, ew: u32, lg_len: usize, cnt_len: usize) !void {
        try rg.check(a, rt, nw, ew, lg_len, cnt_len);
        if (a.R == 0) return;
        const ri = std.mem.indexOfScalar(u32, &rg.rts, rt).?;
        const E: usize = @intCast(a.E);
        const R: usize = @intCast(a.R);
        var args: cuda.Args = .{};
        args.add(a);
        if (ew == 0) {
            const ni = std.mem.indexOfScalar(u32, &rg.narrow, nw).?;
            return o.go(o.f.narrow[ri][ni], .{ (E + nw - 1) / nw, (R + rt - 1) / rt, 1 }, nw * 32, 0, null, &args);
        }
        var wi: usize = 0;
        for (rg.wide, 0..) |c, j| if (c[0] == nw and c[1] == ew) {
            wi = j;
        };
        try o.go(o.f.gemv[ri][wi], .{ (E + nw * ew - 1) / (nw * ew), (R + rt - 1) / rt, 1 }, nw * 32, 0, null, &args);
    }

    /// pfdense.cpp gemm: out [M, N] = xh [M, K] (rotated, fp16) through the trellis decoded in the kernel.
    pub fn pfdGemm(o: Ops, a: pfd.Args, k2: u32, lanes: bool, cfg: usize, words: usize) !void {
        try pfd.check(a, k2, lanes, cfg, words);
        const ki = std.mem.indexOfScalar(u32, &pfd.k2s, k2).?;
        const f = o.f.pfd[ki][@intFromBool(lanes)][cfg] orelse return error.Unsupported;
        const bm = pfd.cfgs[cfg][0];
        const M: usize = @intCast(a.M);
        const N: usize = @intCast(a.N);
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(f, .{ ((M + bm - 1) / bm) * (N >> 7), 1, 1 }, pfd.threads(cfg), pfd.smem(cfg, k2), null, &args);
    }

    /// pfdense.cpp dequant: W_q [K, N] fp16 through the kernel's decode (tests).
    pub fn pfdDequant(o: Ops, T: u64, stride_k: i64, stride_nb: i64, out: u64, K: usize, N: usize, k2: u32, lanes: bool) !void {
        const ki = std.mem.indexOfScalar(u32, &pfd.k2s, k2) orelse return error.Unsupported;
        const f = o.f.pfd_dequant[ki][@intFromBool(lanes)] orelse return error.Unsupported;
        var args: cuda.Args = .{};
        args.add(T);
        args.add(stride_k);
        args.add(stride_nb);
        args.add(out);
        args.add(@as(c_int, @intCast(N)));
        try o.go(f, .{ K / 32, N / 128, 1 }, 128, 0, null, &args);
    }

    /// dsv41_engram_gate_copy: wait (bounded) for the host's ready word, then copy the selected buffer.
    pub fn engramGate(o: Ops, ctl: u64, ready: c_int, err_slot: c_int, base: u64, stride: i64, dst: u64, nbytes: i64, timeout_ns: i64, vec: bool) !void {
        const units: i64 = if (vec) (nbytes + 15) >> 4 else nbytes;
        const blocks: usize = @intCast(std.math.clamp(@divTrunc(units + 255, 256), 1, 64));
        var args: cuda.Args = .{};
        args.add(ctl);
        args.add(ready);
        args.add(err_slot);
        args.add(base);
        args.add(stride);
        args.add(dst);
        args.add(nbytes);
        args.add(timeout_ns);
        args.add(@as(c_int, @intFromBool(vec)));
        try o.go(o.f.gate_copy, .{ blocks, 1, 1 }, 256, 0, null, &args);
    }

    /// dsv41_l2pace launch_paced: (address, bytes) table prefetched to L2 at a paced rate on `ctas` warps.
    pub fn l2Paced(o: Ops, table: u64, n: usize, ctas: usize, chunk: u32, ns_per_piece: u64, delay_ns: u64) !void {
        var args: cuda.Args = .{};
        args.add(table);
        args.add(@as(c_int, @intCast(n)));
        args.add(chunk);
        args.add(ns_per_piece);
        args.add(delay_ns);
        try o.go(o.f.paced, .{ ctas, 1, 1 }, 32, 0, null, &args);
    }

    /// pick.top (topk_keys.cu): each row's best k of bf16-valued fp32 logits, value desc then column asc.
    pub fn topKeys(o: Ops, lg: u64, ld: usize, rows: usize, C: usize, k: usize, vals: u64, cols: u64) !void {
        if (rows < 1 or C < 1 or C > topk_keys.max_cols or k < 1 or k > @min(C, topk_keys.max_k)) return error.Shape;
        var args: cuda.Args = .{};
        args.add(lg);
        args.add(@as(i64, @intCast(ld)));
        args.add(@as(c_int, @intCast(C)));
        args.add(@as(c_int, @intCast(k)));
        args.add(vals);
        args.add(cols);
        try o.go(o.f.topk_keys, .{ rows, 1, 1 }, 1024, 0, null, &args);
    }

    /// The shared expert's pointwise part: act = silu(min(g, limit)) * clamp(u, +-limit) (fp32, strides in elements).
    pub fn swiglu(o: Ops, g: u64, ldg: usize, u: u64, ldu: usize, act: u64, lda: usize, rows: usize, n: usize, limit: f32) !void {
        var args: cuda.Args = .{};
        args.add(g);
        args.add(@as(i64, @intCast(ldg)));
        args.add(u);
        args.add(@as(i64, @intCast(ldu)));
        args.add(act);
        args.add(@as(i64, @intCast(lda)));
        args.add(@as(c_int, @intCast(n)));
        args.add(limit);
        try o.go(o.f.swiglu, .{ @min((n + 255) / 256, 64), rows, 1 }, 256, 0, null, &args);
    }

    /// out += b over n fp32 elements (prefill_moe.forward: the routed partial plus the shared expert's).
    pub fn addInto(o: Ops, out: u64, b: u64, n: usize) !void {
        var args: cuda.Args = .{};
        args.add(out);
        args.add(b);
        args.add(@as(i64, @intCast(n)));
        try o.go(o.f.add, .{ @min((n + 255) / 256, 4096), 1, 1 }, 256, 0, null, &args);
    }

    /// y = bf16_rn(a + b) over n elements (torch.add(out, shared, out=bf16): TF_DSV41_PF_COPIES).
    pub fn addBf16(o: Ops, a: u64, b: u64, y: u64, n: usize) !void {
        var args: cuda.Args = .{};
        args.add(a);
        args.add(b);
        args.add(y);
        args.add(@as(i64, @intCast(n)));
        try o.go(o.f.add_bf16, .{ @min((n + 255) / 256, 4096), 1, 1 }, 256, 0, null, &args);
    }

    /// kvsplit dense: a split family's selected rows packed for the all-gather (`a` @bitCast from kv/split.zig).
    pub fn kvDense(o: Ops, a: kvsplit.DenseArgs) !void {
        if (a.row_bytes == 0 or a.row_bytes % 8 != 0 or a.k == 0 or a.world == 0 or a.psh > 30) return error.Shape;
        const words = @as(u64, a.rows) * a.k * (a.row_bytes / 8);
        if (words == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(o.f.kv_dense, .{ kvsplit.grid(words), 1, 1 }, 256, 0, null, &args);
    }

    /// kvsplit pack (compact): this rank's owned selected rows first, lens[W] each owner's bytes; block 256.
    pub fn kvPack(o: Ops, a: kvsplit.PackArgs) !void {
        const f = o.f.kv_pack orelse return error.Unsupported;
        if (a.row_bytes == 0 or a.row_bytes % 8 != 0 or a.k == 0 or a.world == 0 or a.world > 8 or a.psh > 30) return error.Shape;
        const n = @as(u64, a.rows) * a.k;
        if (n == 0) return;
        var args: cuda.Args = .{};
        args.add(a);
        try o.go(f, .{ @intCast((n + 255) / 256), 1, 1 }, 256, 0, null, &args);
    }

    /// kvsplit dedup: the pack over each slot's unique rows (mark, flags, pack, tok: four launches of 256 threads).
    pub fn kvDedup(o: Ops, d: kvsplit.DedupArgs) !void {
        const fs = o.f.kv_dedup orelse return error.Unsupported;
        const a = d.p;
        if (a.row_bytes == 0 or a.row_bytes % 8 != 0 or a.k == 0 or a.world == 0 or a.world > 8 or a.psh > 30) return error.Shape;
        const n = @as(u64, a.rows) * a.k;
        if (n == 0) return;
        const blocks: usize = @intCast((n + 255) / 256);
        for (fs, [4]usize{ kvsplit.grid(n), blocks, blocks, kvsplit.grid(n) }) |f, g| {
            var args: cuda.Args = .{};
            args.add(d);
            try o.go(f, .{ g, 1, 1 }, 256, 0, null, &args);
        }
    }

    /// kvsplit hist_gather + hist_tok: the history union's M rows of this rank and the selection's tokens.
    pub fn kvHistory(o: Ops, a: kvsplit.HistArgs) !void {
        const fs = o.f.kv_hist orelse return error.Unsupported;
        if (a.row_bytes == 0 or a.row_bytes % 8 != 0 or a.k == 0 or a.world == 0 or a.world > 8 or a.psh > 30) return error.Shape;
        const words = @as(u64, a.m) * (a.row_bytes / 8);
        const n = @as(u64, a.rows) * a.k;
        for (fs, [2]u64{ words, n }) |f, w| {
            if (w == 0) continue;
            var args: cuda.Args = .{};
            args.add(a);
            try o.go(f, .{ kvsplit.grid(w), 1, 1 }, 256, 0, null, &args);
        }
    }

    /// kvsplit gather: send[i] = base row phys[i] (uint32 [n]).
    pub fn kvGather(o: Ops, base: u64, row_bytes: u32, phys: u64, n: u32, send: u64) !void {
        if (row_bytes == 0 or row_bytes % 8 != 0) return error.Shape;
        const words = @as(u64, n) * (row_bytes / 8);
        if (words == 0) return;
        var args: cuda.Args = .{};
        args.add(base);
        args.add(row_bytes);
        args.add(phys);
        args.add(n);
        args.add(send);
        try o.go(o.f.kv_gather, .{ kvsplit.grid(words), 1, 1 }, 256, 0, null, &args);
    }

    pub const glueGo = glue_ops.glueGo;
    pub const embedSend = glue_ops.embedSend;
    pub const embedSum = glue_ops.embedSum;
    pub const castBf16 = glue_ops.castBf16;
    pub const kitWeights = glue_ops.kitWeights;
    pub const kitLogits = glue_ops.kitLogits;
    pub const positions = glue_ops.positions;
    pub const dsCandidates = glue_ops.dsCandidates;
    pub const dsStage = glue_ops.dsStage;
    pub const dsAcceptRows = glue_ops.dsAcceptRows;
    pub const dsStageRows = glue_ops.dsStageRows;
    pub const vsChoose = glue_ops.vsChoose;
    pub const widenF32 = glue_ops.widenF32;
    pub const pfGo = glue_ops.pfGo;
    pub const topPositions = glue_ops.topPositions;
    pub const candKeys = glue_ops.candKeys;
    pub const visibleCounts = glue_ops.visibleCounts;
    pub const gmPicks = glue_ops.gmPicks;
    pub const widthMask = glue_ops.widthMask;
    pub const f64Bf16F32 = glue_ops.f64Bf16F32;
    pub const widenCat = glue_ops.widenCat;
    pub const ringCopy = glue_ops.ringCopy;
    pub const positionsDev = glue_ops.positionsDev;
    pub const gatherCols = glue_ops.gatherCols;
    pub const pickPack = glue_ops.pickPack;
    pub const pickMerge = glue_ops.pickMerge;
    pub const carry = glue_ops.carry;

    /// tfl2pf launch_segments: (address, bytes) table prefetched to L2 (bulk: cp.async.bulk.prefetch).
    pub fn l2Segments(o: Ops, table: u64, n: usize, grid: usize, threads: usize, chunk: u32, bulk: bool) !void {
        var args: cuda.Args = .{};
        args.add(table);
        args.add(@as(c_int, @intCast(n)));
        args.add(chunk);
        try o.go(o.f.segments[@intFromBool(bulk)], .{ grid, 1, 1 }, threads, 0, null, &args);
    }
};

test "pfdense shared memory and heuristic as the reference" {
    // cfg 0 (128, 2, 4, 2, 3) at K2 10: 3 x (128 x 32 x 2 + 2 x 8 x 40 x 4) + 2 x 8192 = 3 x 10752 + 16384
    try std.testing.expectEqual(@as(u32, 3 * 10752 + 16384), pfd.smem(0, 10));
    try std.testing.expectEqual(@as(u32, 256), pfd.threads(0));
    // a 2,048-row [5120 -> 1024] call on 48 SMs: 16 x 8 = 128 CTAs of BM 128
    const h = pfd.heuristic(5120, 1024, 10, 2048, 48);
    try std.testing.expectEqual(@as(usize, 8), h[1]);
}

test "mhc instances follow dispatch" {
    var a: mhc.Args = .{ .R = 3 };
    try std.testing.expectEqual(@as(usize, 6), try mhc.instance(a, 3, true));
    try std.testing.expectEqual(@as(usize, 3), try mhc.instance(a, 1, true));
    a.R = 17;
    try std.testing.expectError(error.Shape, mhc.instance(a, 0, false));
    const p: mhc_pf.Args = .{ .R = 40 };
    try std.testing.expectEqual(@as(usize, 5), try mhc_pf.instance(p, 1, true, false));
    try std.testing.expectError(error.Shape, mhc_pf.instance(p, 1, false, true));
    try std.testing.expectEqual(@as(u32, 47872), mhc_pf.smem(false));
}
