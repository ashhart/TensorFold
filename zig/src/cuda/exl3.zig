//! The Python engine's EXL3 kernels (copies in kernels/cuda/mimo_v2) launched as its bindings do, after Jay Leaton.

const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Args = launch_.Args;

/// linear.cu's DType: what `y` (and rot_in's `x`) hold.
pub const DType = enum(c_int) { f16 = 0, bf16 = 1, f32 = 2 };

/// mul1 (CODEBOOK_IDS["mul1"]): the only codebook these instances are resolved at.
pub const cb_mul1: u32 = 2;
/// The K2 (half-bits a value) widths linear.cu builds at mul1: TF_EXL3_WIDTHS(X, 2) plus 3, 5, 7.
pub const linear_k2s = [_]u32{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 };
pub const linear_wks = [_]u32{ 2, 4, 8 };

// Symbols (cuobjdump -elf of the mimo_exl3_* fatbins)

pub fn linearSymbol(comptime k2: u32, comptime wk: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN14tf_exl3_linear13linear_kernelILi{d}ELi2ELi{d}EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii", .{ k2, wk });
}
pub fn unpackSymbol(comptime k2: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN14tf_exl3_linear13unpack_kernelILi{d}ELi2EEEvPKjP6__halfill", .{k2});
}
pub fn groupedSymbol(comptime nt: u32, comptime w: u32, comptime pf: u32, comptime lo: u32, comptime hi: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN8tf_exl3x14grouped_kernelILi2ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiii", .{ nt, w, pf, lo, hi });
}
pub fn dequantSymbol(comptime k2: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN8tf_exl3x14dequant_kernelILi2ELi{d}EEEvPKjP6__halfii", .{k2});
}
pub const sym = struct {
    pub const linear_rot_in = "_ZN14tf_exl3_linear13rot_in_kernelEPKviPK6__halfPS2_i";
    pub const group = "_ZN15tf_exl3_experts12group_kernelEPKiPiS2_S2_iiii";
    pub const rot_in_bf16 = "_ZN15tf_exl3_experts13rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
    pub const rot_in_f16 = "_ZN15tf_exl3_experts13rot_in_kernelI6__halfEEvPKT_iPKiPKS1_S8_PS1_S9_iii";
    pub const gateup_epilogue = "_ZN15tf_exl3_experts22gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiifi";
    pub const down_epilogue = "_ZN15tf_exl3_experts20down_epilogue_kernelEPKfPKiPK6__halfPfiiii";
    pub const combine = "_ZN15tf_exl3_experts14combine_kernelEPKfS1_Pfii";
    pub const down_combine = "_ZN15tf_exl3_experts19down_combine_kernelEPKfPKiPK6__halfPfS1_S7_iiiii";
};

/// grouped_launch's (nt, warps, pf) settings and TF_RANGES (lo, hi), in instance order.
pub const grouped_cfgs = [_][3]u32{ .{ 8, 4, 1 }, .{ 8, 4, 2 }, .{ 4, 4, 2 } };
pub const grouped_ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 16 } };

/// TF_RANGES: (8, 8) when every width is 4 bits, (2, 10) up to 5 bits, else (2, 16).
pub fn rangeIndex(lo: u32, hi: u32) usize {
    if (lo == 8 and hi == 8) return 0;
    if (lo >= 2 and hi <= 10) return 1;
    return 2;
}

fn cfgIndex(nt: u32, w: u32, pf: u32) ?usize {
    for (grouped_cfgs, 0..) |c, i| if (c[0] == nt and c[1] == w and c[2] == pf) return i;
    return null;
}

fn k2Index(k2: u32) ?usize {
    return std.mem.indexOfScalar(u32, &linear_k2s, k2);
}

fn wkIndex(wk: u32) ?usize {
    return std.mem.indexOfScalar(u32, &linear_wks, wk);
}

// Plans (host): the Python engine's choices, which fix each layer's reduction order

/// linear.plan: (K splits, warps a program) for a K x N layer, the shape's alone.
pub fn plan(k: usize, n: usize) [2]usize {
    const blocks: usize = 192;
    const min_tiles: usize = 8;
    const kt = k / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ sk, wk };
}

/// experts.default_config: (n tiles a block, warps, K splits, tiles in flight) for a K -> N projection.
pub fn defaultConfig(k: usize, n: usize, gateup: bool) ![4]u32 {
    const cands = [_][4]u32{ if (gateup) .{ 8, 4, 4, 1 } else .{ 8, 4, 1, 1 }, .{ 8, 4, 2, 1 }, .{ 8, 4, 1, 1 }, .{ 4, 4, 2, 2 }, .{ 4, 4, 1, 2 } };
    for (cands) |c| {
        if (k % (16 * c[2] * c[1]) == 0 and n % (16 * c[0]) == 0) return c;
    }
    return error.NoTileSetting;
}

// Weights

/// One EXL3 linear layer in the "strips" layout ([N/128, K/16, 8, 4 * K2] words): Exl3Linear.
pub const Linear = struct {
    words: u64,
    suh: u64, // fp16 [K]
    svh: u64, // fp16 [N]
    bias: u64 = 0, // fp16 [N] or 0
    counters: u64, // int32 [8 * N / 128], zeroed once (the split tickets)
    k: usize,
    n: usize,
    k2: u32,
    split: [2]usize, // plan(k, n)

    /// (words between k tiles, words between 128-column blocks): Exl3Linear.strides for "strips".
    pub fn strides(l: Linear) [2]i64 {
        const tw: i64 = 4 * @as(i64, l.k2);
        return .{ 8 * tw, @as(i64, @intCast(l.k / 16)) * 8 * tw };
    }

    pub fn wordsBytes(k: usize, n: usize, k2: u32) usize {
        return (k / 16) * (n / 16) * 16 * k2;
    }
};

/// One layer's routed experts (Exl3RoutedExperts): device tables of trellis pointers and widths, stacked scales.
pub const Experts = struct {
    gate_ptr: u64, // int64 [E]
    up_ptr: u64,
    down_ptr: u64,
    gate_k2: u64, // int32 [E]
    up_k2: u64,
    down_k2: u64,
    suh_g: u64, // fp16 [E, D]
    suh_u: u64,
    svh_g: u64, // fp16 [E, I]
    svh_u: u64,
    suh_d: u64, // fp16 [E, I]
    svh_d: u64, // fp16 [E, D]
    count: usize, // E
    dims: usize, // D
    width: usize, // I (this rank's)
    k2_gu: [2]u32, // (min, max) over gate and up
    k2_d: [2]u32,
    /// Host copies of the pointer and width tables, [gate E, up E, down E] (the prompt path's per-expert GEMMs).
    host_ptr: []const u64 = &.{},
    host_k2: []const i32 = &.{},

    /// Expert e's gate (j 0), up (1) or down (2) as an Exl3Linear view of its stored trellis (moe._linears).
    pub fn linear(x: Experts, e: usize, j: usize) Linear {
        const E = x.count;
        const D = x.dims;
        const I = x.width;
        return switch (j) {
            0 => .{ .words = x.host_ptr[e], .suh = x.suh_g + e * D * 2, .svh = x.svh_g + e * I * 2, .counters = 0, .k = D, .n = I, .k2 = @intCast(x.host_k2[e]), .split = .{ 1, 1 } },
            1 => .{ .words = x.host_ptr[E + e], .suh = x.suh_u + e * D * 2, .svh = x.svh_u + e * I * 2, .counters = 0, .k = D, .n = I, .k2 = @intCast(x.host_k2[E + e]), .split = .{ 1, 1 } },
            else => .{ .words = x.host_ptr[2 * E + e], .suh = x.suh_d + e * I * 2, .svh = x.svh_d + e * D * 2, .counters = 0, .k = I, .n = D, .k2 = @intCast(x.host_k2[2 * E + e]), .split = .{ 1, 1 } },
        };
    }
};

/// Scratch for up to `rows` rows of `slots` slots (experts.Scratch); device addresses, sized by `bytes`.
pub const Scratch = struct {
    xg: u64, // fp16 [P, D]
    xu: u64,
    xd: u64, // fp16 [P, I]
    z: u64, // fp32 [max(2 * sk_gu * I, sk_d * D) * P]
    y: u64, // fp32 [P, D]
    ids: u64, // int32 [min(P, E)]
    count: u64, // int32 [1]
    members: u64, // int32 [min(P, E) * rows], -1 filled
    rows: usize,
    slots: usize,
    cfg_gu: [4]u32,
    cfg_d: [4]u32,

    pub const Sizes = struct { xg: usize, xd: usize, z: usize, y: usize, ids: usize, members: usize };

    pub fn bytes(e: Experts, rows: usize, slots: usize, cfg_gu: [4]u32, cfg_d: [4]u32) Sizes {
        const p = rows * slots;
        const maxu = @min(p, e.count);
        return .{
            .xg = p * e.dims * 2,
            .xd = p * e.width * 2,
            .z = @max(2 * @as(usize, cfg_gu[2]) * e.width, @as(usize, cfg_d[2]) * e.dims) * p * 4,
            .y = p * e.dims * 4,
            .ids = maxu * 4,
            .members = maxu * rows * 4,
        };
    }

    /// Scratch.window: (distinct experts at most, members row length) for R rows.
    pub fn window(s: Scratch, experts: usize, r: usize) [2]usize {
        return .{ @min(r * s.slots, experts), r };
    }
};

// Kernels

pub const Kernels = struct {
    mods: [3]Module,
    f_rot_in: Function,
    f_linear: [linear_k2s.len][linear_wks.len]Function,
    f_unpack: [linear_k2s.len]Function,
    f_grouped: [grouped_cfgs.len][grouped_ranges.len]Function,
    f_dequant: [15]Function, // K2 2..16
    f_group: Function,
    f_x_rot_bf16: Function,
    f_x_rot_f16: Function,
    f_gateup_epilogue: Function,
    f_down_epilogue: Function,
    f_combine: Function,
    f_down_combine: Function,

    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.KernelsNotBuilt;
        var k: Kernels = undefined;
        k.mods[0] = try Module.load(d, kernels.mimo_exl3_linear);
        errdefer k.mods[0].unload();
        k.mods[1] = try Module.load(d, kernels.mimo_exl3_experts);
        errdefer k.mods[1].unload();
        k.mods[2] = try Module.load(d, kernels.mimo_exl3_experts_cb2);
        errdefer k.mods[2].unload();
        const lin, const ex, const cb2 = .{ k.mods[0], k.mods[1], k.mods[2] };
        k.f_rot_in = try lin.function(sym.linear_rot_in);
        inline for (linear_k2s, 0..) |k2, i| {
            inline for (linear_wks, 0..) |wk, j| k.f_linear[i][j] = try lin.function(linearSymbol(k2, wk));
            k.f_unpack[i] = try lin.function(unpackSymbol(k2));
        }
        inline for (grouped_cfgs, 0..) |c, i| {
            inline for (grouped_ranges, 0..) |r, j| k.f_grouped[i][j] = try cb2.function(groupedSymbol(c[0], c[1], c[2], r[0], r[1]));
        }
        inline for (2..17) |k2| k.f_dequant[k2 - 2] = try cb2.function(dequantSymbol(@as(u32, k2)));
        k.f_group = try ex.function(sym.group);
        k.f_x_rot_bf16 = try ex.function(sym.rot_in_bf16);
        k.f_x_rot_f16 = try ex.function(sym.rot_in_f16);
        k.f_gateup_epilogue = try ex.function(sym.gateup_epilogue);
        k.f_down_epilogue = try ex.function(sym.down_epilogue);
        k.f_combine = try ex.function(sym.combine);
        k.f_down_combine = try ex.function(sym.down_combine);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
    }

    // -- linear.cu ----------------------------------------------------------------------------------------------

    /// exl3_rot_in_cuda: xh (M, K) fp16 = Hadamard-rotated x * suh.
    pub fn linearRotIn(k: *const Kernels, s: Stream, x: u64, x_dtype: DType, suh: u64, xh: u64, m: usize, kk: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(@backingInt(x_dtype));
        a.add(suh);
        a.add(xh);
        a.add(int(kk));
        try launch_.launch(k.f_rot_in, .{ .grid = .{ .x = u(((kk / 128) + 3) / 4), .y = u(m), .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, s, &a);
    }

    /// exl3_linear_cuda: y (M, N) of dtype `y_dtype` = xh @ W * svh (+ bias); z: fp32 [SK * M * N] when SK > 1.
    pub fn linearRows(k: *const Kernels, s: Stream, l: Linear, xh: u64, y: u64, y_dtype: DType, z: u64, m: usize) !void {
        const sk, const wk = l.split;
        if (m < 1 or m > 128) return error.RowCount;
        if ((l.k / 16) % (sk * wk) != 0) return error.KSplit; // "K / 16 must split evenly over SK * WK warps"
        if (sk > 1 and z == 0) return error.NeedZ; // "Z is needed with more than one split"
        const ki = k2Index(l.k2) orelse return error.UnsupportedWidth;
        const wi = wkIndex(u(wk)) orelse return error.UnsupportedWarps;
        const f = k.f_linear[ki][wi];
        const smem: u32 = u(wk * @min(m, 8) * 128 * 4);
        if (smem > 48 * 1024) try f.allowDynamicShared(smem);
        const st = l.strides();
        var a: Args = .{};
        a.add(xh);
        a.add(l.words);
        a.add(st[0]);
        a.add(st[1]);
        a.add(l.svh);
        a.add(l.bias);
        a.add(y);
        a.add(@backingInt(y_dtype));
        a.add(z);
        a.add(l.counters);
        a.add(int(m));
        a.add(int(l.k));
        a.add(int(l.n));
        a.add(int(sk));
        try launch_.launch(f, .{ .grid = .{ .x = u(l.n / 128), .y = u(sk), .z = 1 }, .block = .{ .x = u(wk * 32), .y = 1, .z = 1 }, .shared = smem }, s, &a);
    }

    /// Exl3Linear.__call__: rot_in then linear.
    pub fn linear(k: *const Kernels, s: Stream, l: Linear, x: u64, x_dtype: DType, xh: u64, y: u64, y_dtype: DType, z: u64, m: usize) !void {
        try k.linearRotIn(s, x, x_dtype, l.suh, xh, m, l.k);
        try k.linearRows(s, l, xh, y, y_dtype, z, m);
    }

    /// exl3_unpack_cuda: W (K, N) fp16 from the strips layout.
    pub fn unpackWeight(k: *const Kernels, s: Stream, l: Linear, w: u64) !void {
        return k.unpackStrided(s, l, w, l.strides());
    }

    /// Exl3Linear.strides for the "stored" layout [K/16, N/16, W] (experts.prepare's trellises).
    pub fn storedStrides(l: Linear) [2]i64 {
        const tw: i64 = 4 * @as(i64, l.k2);
        return .{ @as(i64, @intCast(l.n / 16)) * tw, 8 * tw };
    }

    /// exl3_unpack_cuda with explicit (k tile, 128-column block) strides in words.
    pub fn unpackStrided(k: *const Kernels, s: Stream, l: Linear, w: u64, st: [2]i64) !void {
        const ki = k2Index(l.k2) orelse return error.UnsupportedWidth;
        var a: Args = .{};
        a.add(l.words);
        a.add(w);
        a.add(int(l.n));
        a.add(st[0]);
        a.add(st[1]);
        try launch_.launch(k.f_unpack[ki], .{ .grid = .{ .x = u(l.n / 16), .y = u(l.k / 16), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    // -- experts.cu / experts_grouped.cuh ------------------------------------------------------------------------

    /// exl3x_group_cuda: distinct experts of R rows' picks (ids, count) and their members.
    pub fn group(k: *const Kernels, s: Stream, pick: u64, ids: u64, count: u64, members: u64, r: usize, slots: usize, e: usize, maxm: usize) !void {
        if (e > 1024 * 4) return error.TooManyExperts;
        if (slots > 32) return error.TooManySlots;
        const smem: u32 = u(r * slots * 4);
        if (smem + 1024 / 32 * 4 > 48 * 1024) try k.f_group.allowDynamicShared(smem);
        var a: Args = .{};
        a.add(pick);
        a.add(ids);
        a.add(count);
        a.add(members);
        a.add(int(r));
        a.add(int(slots));
        a.add(int(e));
        a.add(int(maxm));
        try launch_.launch(k.f_group, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024, .y = 1, .z = 1 }, .shared = smem }, s, &a);
    }

    /// exl3x_rot_in_cuda: each (row, slot)'s input rotated by its expert's gate and up suh.
    pub fn expertsRotIn(k: *const Kernels, s: Stream, x: u64, x_bf16: bool, x_stride: usize, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, rows: usize, kk: usize, slots: usize, e: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(pick);
        a.add(suh0);
        a.add(suh1);
        a.add(out0);
        a.add(out1);
        a.add(int(kk));
        a.add(int(slots));
        a.add(int(e));
        try launch_.launch(if (x_bf16) k.f_x_rot_bf16 else k.f_x_rot_f16, .{ .grid = .{ .x = u(rows * slots), .y = u(kk / 128), .z = 2 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    pub const GroupedArgs = struct {
        x0: u64,
        x1: u64,
        tp0: u64,
        tp1: u64,
        k2_0: u64,
        k2_1: u64,
        uids: u64,
        ucount: u64,
        members: u64,
        z: u64,
        mats: usize,
        k: usize,
        n: usize,
        p: usize,
        sk: usize,
        slots: usize,
        maxm: usize, // members.size(1)
        nexp_max: usize, // uids.size(0)
        nt: u32,
        warps: u32,
        pf: u32,
        lo: u32,
        hi: u32,
    };

    /// exl3x_grouped_cuda at mul1 (grouped_launch<2>).
    pub fn grouped(k: *const Kernels, s: Stream, g: GroupedArgs) !void {
        if (g.k % (16 * g.sk * g.warps) != 0 or g.n % (16 * g.nt) != 0) return error.UnevenSplit;
        const ci = cfgIndex(g.nt, g.warps, g.pf) orelse return error.UnsupportedTile;
        const f = k.f_grouped[ci][rangeIndex(g.lo, g.hi)];
        const mt = (g.maxm + 15) / 16;
        var a: Args = .{};
        inline for (.{ g.x0, g.x1, g.tp0, g.tp1, g.k2_0, g.k2_1, g.uids, g.ucount, g.members, g.z }) |p| a.add(p);
        inline for (.{ g.k, g.n, g.p, g.sk, g.maxm, g.slots }) |v| a.add(int(v));
        try launch_.launch(f, .{ .grid = .{ .x = u(g.nexp_max), .y = u(g.n / (16 * g.nt)), .z = u(g.mats * g.sk * mt) }, .block = .{ .x = g.warps * 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// exl3x_gateup_epilogue_cuda.
    pub fn gateupEpilogue(k: *const Kernels, s: Stream, z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, rows: usize, p: usize, n: usize, sk: usize, slots: usize, e: usize, limit: f32, act_mode: c_int) !void {
        var a: Args = .{};
        inline for (.{ z, pick, svh_g, svh_u, suh_d, xd }) |v| a.add(v);
        inline for (.{ p, n, sk, e }) |v| a.add(int(v));
        a.add(limit);
        a.add(act_mode);
        try launch_.launch(k.f_gateup_epilogue, .{ .grid = .{ .x = u(rows * slots), .y = u(n / 128), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// exl3x_down_epilogue_cuda.
    pub fn downEpilogue(k: *const Kernels, s: Stream, z: u64, pick: u64, svh_d: u64, y: u64, rows: usize, p: usize, d: usize, sk: usize, slots: usize, e: usize) !void {
        var a: Args = .{};
        inline for (.{ z, pick, svh_d, y }) |v| a.add(v);
        inline for (.{ p, d, sk, e }) |v| a.add(int(v));
        try launch_.launch(k.f_down_epilogue, .{ .grid = .{ .x = u(rows * slots), .y = u(d / 128), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// exl3x_down_combine_cuda: down epilogue and the weighted slot sum in one launch.
    pub fn downCombine(k: *const Kernels, s: Stream, z: u64, pick: u64, svh_d: u64, y: u64, wts: u64, out: u64, rows: usize, p: usize, d: usize, sk: usize, slots: usize, e: usize) !void {
        if (slots > 32) return error.TooManySlots;
        var a: Args = .{};
        inline for (.{ z, pick, svh_d, y, wts, out }) |v| a.add(v);
        inline for (.{ p, d, sk, e, slots }) |v| a.add(int(v));
        try launch_.launch(k.f_down_combine, .{ .grid = .{ .x = u(rows), .y = u(d / 128), .z = 1 }, .block = .{ .x = u(32 * slots), .y = 1, .z = 1 } }, s, &a);
    }

    /// exl3x_combine_cuda.
    pub fn combine(k: *const Kernels, s: Stream, y: u64, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        var a: Args = .{};
        inline for (.{ y, wts, out }) |v| a.add(v);
        a.add(int(d));
        a.add(int(slots));
        try launch_.launch(k.f_combine, .{ .grid = .{ .x = u(rows), .y = u((d + 255) / 256), .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }

    /// exl3x_dequant_cuda at mul1: W (K, N) fp16 from trellis words in the stored layout (tests).
    pub fn dequant(k: *const Kernels, s: Stream, t: u64, out: u64, kk: usize, n: usize, k2: u32) !void {
        if (k2 < 2 or k2 > 16) return error.UnsupportedWidth;
        var a: Args = .{};
        a.add(t);
        a.add(out);
        a.add(int(kk));
        a.add(int(n));
        try launch_.launch(k.f_dequant[k2 - 2], .{ .grid = .{ .x = u(kk / 16), .y = u(n / 16), .z = 1 }, .block = .{ .x = 32, .y = 1, .z = 1 } }, s, &a);
    }

    /// experts.routed with `wts` (the served path): group, rotate, gate/up, epilogue, down, down+combine.
    pub fn routed(k: *const Kernels, s: Stream, x: u64, x_bf16: bool, x_stride: usize, pick: u64, wts: u64, e: Experts, sc: Scratch, out: u64, r: usize, limit: f32, act_mode: c_int, do_group: bool) !void {
        if (r > sc.rows) return error.WindowTooWide;
        const d, const i_, const n_exp = .{ e.dims, e.width, e.count };
        const slots = sc.slots;
        const p = r * slots;
        const w = sc.window(n_exp, r);
        if (do_group) try k.group(s, pick, sc.ids, sc.count, sc.members, r, slots, n_exp, w[1]);
        try k.expertsRotIn(s, x, x_bf16, x_stride, pick, e.suh_g, e.suh_u, sc.xg, sc.xu, r, d, slots, n_exp);
        const gu = sc.cfg_gu;
        try k.grouped(s, .{ .x0 = sc.xg, .x1 = sc.xu, .tp0 = e.gate_ptr, .tp1 = e.up_ptr, .k2_0 = e.gate_k2, .k2_1 = e.up_k2, .uids = sc.ids, .ucount = sc.count, .members = sc.members, .z = sc.z, .mats = 2, .k = d, .n = i_, .p = p, .sk = gu[2], .slots = slots, .maxm = w[1], .nexp_max = w[0], .nt = gu[0], .warps = gu[1], .pf = gu[3], .lo = e.k2_gu[0], .hi = e.k2_gu[1] });
        try k.gateupEpilogue(s, sc.z, pick, e.svh_g, e.svh_u, e.suh_d, sc.xd, r, p, i_, gu[2], slots, n_exp, limit, act_mode);
        const dn = sc.cfg_d;
        try k.grouped(s, .{ .x0 = sc.xd, .x1 = sc.xd, .tp0 = e.down_ptr, .tp1 = e.down_ptr, .k2_0 = e.down_k2, .k2_1 = e.down_k2, .uids = sc.ids, .ucount = sc.count, .members = sc.members, .z = sc.z, .mats = 1, .k = i_, .n = d, .p = p, .sk = dn[2], .slots = slots, .maxm = w[1], .nexp_max = w[0], .nt = dn[0], .warps = dn[1], .pf = dn[3], .lo = e.k2_d[0], .hi = e.k2_d[1] });
        try k.downCombine(s, sc.z, pick, e.svh_d, sc.y, wts, out, r, p, d, dn[2], slots, n_exp);
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

// Host tests: the plans equal the Python engine's linear.plan and default_config, the symbols the fatbins hold.

test "linear.plan matches the Python engine at MiMo's shapes" {
    const vectors = [_][4]usize{
        .{ 4096, 6144, 4, 4 },  .{ 4096, 384, 8, 4 },   .{ 4096, 256, 8, 4 },   .{ 4096, 768, 8, 4 },
        .{ 4096, 512, 8, 4 },   .{ 4096, 4096, 8, 4 },  .{ 4096, 8192, 1, 8 },  .{ 8192, 4096, 8, 4 },
        .{ 4096, 76288, 1, 8 }, .{ 4096, 128, 8, 4 },   .{ 1024, 4096, 2, 4 },  .{ 2048, 4096, 4, 4 },
        .{ 4096, 12288, 1, 8 }, .{ 16384, 4096, 8, 4 }, .{ 4096, 16384, 1, 8 }, .{ 20480, 4096, 8, 4 },
    };
    for (vectors) |v| try std.testing.expectEqual([2]usize{ v[2], v[3] }, plan(v[0], v[1]));
}

test "experts.default_config matches the Python engine" {
    const vectors = [_]struct { usize, usize, bool, [4]u32 }{
        .{ 4096, 1024, true, .{ 8, 4, 4, 1 } }, .{ 1024, 4096, false, .{ 8, 4, 1, 1 } },
        .{ 4096, 2048, true, .{ 8, 4, 4, 1 } }, .{ 2048, 4096, false, .{ 8, 4, 1, 1 } },
        .{ 4096, 512, true, .{ 8, 4, 4, 1 } },  .{ 512, 4096, false, .{ 8, 4, 1, 1 } },
    };
    for (vectors) |v| try std.testing.expectEqual(v[3], try defaultConfig(v[0], v[1], v[2]));
}

test "symbols are the fatbins' (cuobjdump -elf of mimo_exl3_*)" {
    try std.testing.expectEqualStrings("_ZN14tf_exl3_linear13linear_kernelILi4ELi2ELi4EEEvPK6__halfPKjxxS3_S3_PviPfPiiiii", linearSymbol(4, 4));
    try std.testing.expectEqualStrings("_ZN14tf_exl3_linear13unpack_kernelILi4ELi2EEEvPKjP6__halfill", unpackSymbol(4));
    try std.testing.expectEqualStrings("_ZN8tf_exl3x14grouped_kernelILi2ELi8ELi4ELi1ELi8ELi8EEEvPK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiii", groupedSymbol(8, 4, 1, 8, 8));
    try std.testing.expectEqualStrings("_ZN8tf_exl3x14dequant_kernelILi2ELi4EEEvPKjP6__halfii", dequantSymbol(4));
}

test "TF_RANGES" {
    try std.testing.expectEqual(@as(usize, 0), rangeIndex(8, 8));
    try std.testing.expectEqual(@as(usize, 1), rangeIndex(4, 8));
    try std.testing.expectEqual(@as(usize, 1), rangeIndex(2, 10));
    try std.testing.expectEqual(@as(usize, 2), rangeIndex(4, 12));
}
