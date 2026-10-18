//! DeepSeek-V4.1's routed EXL3 expert kernels: x3ld, x3pf, x3gm and the grouped kernel with its helpers.

const std = @import("std");
const cuda = @import("root.zig");
const gmk = @import("exl3_block_x3gm.zig");

// -- Instances (the copies' footers hold the matching explicit instantiations) ----------------------------------

/// The (lo, hi) K2 ranges x3ld / x3pf are built for, tried in order (expert_loads.RANGES / expert_prefill.RANGES).
pub const ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 12 } };
/// Upstream's TF_RANGES: (8, 8), (2, 10), else (2, 16).
pub const upstream_ranges = [_][2]u32{ .{ 8, 8 }, .{ 2, 10 }, .{ 2, 16 } };

/// dispatch_ranges: the first instance whose range holds [lo, hi], or null (the binding's "unsupported").
pub fn rangeIndex(lo: u32, hi: u32) ?usize {
    if (lo == 8 and hi == 8) return 0;
    if (lo >= 2 and hi <= 10) return 1;
    if (lo >= 2 and hi <= 12) return 2;
    return null;
}

/// grouped_launch's TF_RANGES (always an instance: 2..16 catches the rest).
pub fn upstreamRangeIndex(lo: u32, hi: u32) usize {
    if (lo == 8 and hi == 8) return 0;
    if (lo >= 2 and hi <= 10) return 1;
    return 2;
}

/// x3ld's (nt, pd) settings (expert_loads.CFGS); probe 3 (timing only) has the first two.
pub const ld_cfgs = [_][2]u32{ .{ 8, 1 }, .{ 8, 2 }, .{ 4, 2 } };
/// x3pf's (nt, mtl) settings.
pub const pf_cfgs = [_][2]u32{ .{ 4, 4 }, .{ 8, 2 } };
/// upstream grouped_kernel's (nt, warps, pf) settings.
pub const grouped_cfgs = [_][3]u32{ .{ 8, 4, 1 }, .{ 8, 4, 2 }, .{ 4, 4, 2 } };

fn cfgIndex(comptime n: usize, table: []const [n]u32, want: [n]u32) ?usize {
    for (table, 0..) |c, i| if (std.mem.eql(u32, &c, &want)) return i;
    return null;
}

const tail16 = "PK6__halfS3_PKlS5_PKiS7_S7_S7_S7_Pfiiiiii"; // (X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, 6 ints)

pub fn ldSymbol(comptime nt: u32, comptime pd: u32, comptime lo: u32, comptime hi: u32, comptime probe: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10dsv41_x3ld9ld_kernelILi2ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEv" ++ tail16, .{ nt, pd, lo, hi, probe });
}

/// TF_DSV41_X3LD_EPI (x3ld_epi.cu): x3ld (nt 8) with the gate/up (`gu`) or down_combine (`dn` fp32, `dnb` bf16) tail.
pub const EpiKind = enum { gu, dn, dnb };
pub const epi_pds = [_]u32{ 1, 2 };

pub fn epiSymbol(comptime kind: EpiKind, comptime pd: u32, comptime lo: u32, comptime hi: u32) [:0]const u8 {
    return std.fmt.comptimePrint("dsv41_x3ld_epi_{s}_{d}_{d}_{d}", .{ @tagName(kind), pd, lo, hi });
}

pub fn pfSymbol(comptime nt: u32, comptime mtl: u32, comptime lo: u32, comptime hi: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10dsv41_x3pf9pf_kernelILi2ELi{d}ELi{d}ELi{d}ELi{d}EEEv" ++ tail16, .{ nt, mtl, lo, hi });
}

pub fn groupedSymbol(comptime nt: u32, comptime w: u32, comptime pf: u32, comptime lo: u32, comptime hi: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN8tf_exl3x14grouped_kernelILi2ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEv" ++ tail16, .{ nt, w, pf, lo, hi });
}

pub fn dequantSymbol(comptime k2: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN8tf_exl3x14dequant_kernelILi2ELi{d}EEEvPKjP6__halfii", .{k2});
}

const up_sym = struct {
    const group = "_ZN15tf_exl3_experts12group_kernelEPKiPiS2_S2_iiii";
    const rot_in_bf16 = "_ZN15tf_exl3_experts13rot_in_kernelI13__nv_bfloat16EEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
    const rot_in_f16 = "_ZN15tf_exl3_experts13rot_in_kernelI6__halfEEvPKT_iPKiPKS1_S8_PS1_S9_iii";
    const gateup_epilogue = "_ZN15tf_exl3_experts22gateup_epilogue_kernelEPKfPKiPK6__halfS6_S6_PS4_iiiifi";
    const down_epilogue = "_ZN15tf_exl3_experts20down_epilogue_kernelEPKfPKiPK6__halfPfiiii";
    const combine = "_ZN15tf_exl3_experts14combine_kernelEPKfS1_Pfii";
    // templated on the output (R1: fp32, or bf16 rounded as stored = fp32 then .to(bf16))
    const down_combine_f32 = "_ZN15tf_exl3_experts19down_combine_kernelIfEEvPKfPKiPK6__halfPfS2_PT_iiiii";
    const down_combine_bf16 = "_ZN15tf_exl3_experts19down_combine_kernelI13__nv_bfloat16EEvPKfPKiPK6__halfPfS3_PT_iiiii";
};

pub const gm_k2s = gmk.gm_k2s;
pub const gu_tiles = gmk.gu_tiles;
pub const dn_tiles = gmk.dn_tiles;
pub const gm_smem_max = gmk.gm_smem_max;
pub const gmK2Index = gmk.gmK2Index;
pub const GmShape = gmk.GmShape;
pub const v2_gu = gmk.v2_gu;
pub const gu2_cfg = gmk.gu2_cfg;
pub const v2_dn = gmk.v2_dn;
pub const gm2Shape = gmk.gm2Shape;
pub const V3 = gmk.V3;
pub const v3_variants = gmk.v3_variants;
pub const gm3Symbol = gmk.gm3Symbol;
pub const v3Of = gmk.v3Of;
pub const gmV3 = gmk.gmV3;
pub const v3_default = gmk.v3_default;
pub const gmTuned2 = gmk.gmTuned2;
pub const gm2WidthsOk = gmk.gm2WidthsOk;
pub const gm2GuSymbol = gmk.gm2GuSymbol;
pub const gm2DnSymbol = gmk.gm2DnSymbol;
pub const gmSmemBytes = gmk.gmSmemBytes;
pub const GmKind = gmk.GmKind;
pub const plan_max_experts = gmk.plan_max_experts;
pub const planPasses = gmk.planPasses;
pub const gmFits = gmk.gmFits;
pub const gmMembers = gmk.gmMembers;
pub const TuneRow = gmk.TuneRow;
pub const tune_default = gmk.tune_default;
pub const gmTuned = gmk.gmTuned;
pub const gmGuSymbol = gmk.gmGuSymbol;
pub const gmDnSymbol = gmk.gmDnSymbol;
pub const gmDequantSymbol = gmk.gmDequantSymbol;
pub const GmArgs = gmk.GmArgs;
pub const gmCheck = gmk.gmCheck;

/// TF_DSV41_PF_TBO_GM_SMS (under TF_DSV41_PF_TBO): the SMs x3gm's persistent grid covers at most; 0: every SM
pub var gm_sms_cap: usize = 0;

const gm_sym = struct {
    const rot_bf16_1 = "_ZN10dsv41_x3gm10rot_kernelI13__nv_bfloat16Li1EEEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
    const rot_bf16_2 = "_ZN10dsv41_x3gm10rot_kernelI13__nv_bfloat16Li2EEEvPKT_iPKiPK6__halfS9_PS7_SA_iii";
    const rot_f16_1 = "_ZN10dsv41_x3gm10rot_kernelI6__halfLi1EEEvPKT_iPKiPKS1_S8_PS1_S9_iii";
    const rot_f16_2 = "_ZN10dsv41_x3gm10rot_kernelI6__halfLi2EEEvPKT_iPKiPKS1_S8_PS1_S9_iii";
};

// -- Loaded functions -------------------------------------------------------------------------------------------

pub const Gm = struct { f: cuda.Function, threads: u32, smem: u32, per_sm: u32 };

pub const Functions = struct {
    ld: [2][ld_cfgs.len][ranges.len]?cuda.Function, // [probe 0 / 3][cfg][range]
    pf: [pf_cfgs.len][ranges.len]cuda.Function,
    grouped: [grouped_cfgs.len][upstream_ranges.len]cuda.Function,
    dequant: [15]cuda.Function, // K2 2..16
    group: cuda.Function,
    group_rot: cuda.Function,
    rot_in: [2]cuda.Function, // bf16, fp16 input
    gateup_epilogue: cuda.Function,
    down_epilogue: cuda.Function,
    combine: cuda.Function,
    down_combine: [2]cuda.Function, // fp32, bf16 out
    gm_gu: [gm_k2s.len][2][gu_tiles.len]Gm, // [k2][xm - 1][cfg]
    gm_dn: [gm_k2s.len][dn_tiles.len]Gm,
    gm_rot: [2][2]cuda.Function, // [bf16, fp16][mats - 1]
    gm_dequant: [gm_k2s.len]cuda.Function,
    gm_plan: cuda.Function,
    gm2_gu: [2][v2_gu.len]Gm, // [xm - 1][cfg]
    gm2_dn: Gm,
    /// gm2_kernel at GU2 (128-member gate / up, two inputs): x3gm3.cu's instance (TF_DSV41_GM_GU2)
    gm2_gu2: Gm,
    gm3: [v3_variants.len]Gm,
    epi: [3][epi_pds.len][ranges.len]cuda.Function, // [EpiKind][pd - 1][range]

    /// Resolves every instance (a wrong name fails here, not on a rare path) and opts x3gm into its shared memory.
    pub fn resolve(experts: cuda.Module, x3ld: cuda.Module, x3pf: cuda.Module, x3gm: cuda.Module, x3gm_plan: cuda.Module, x3gm3: cuda.Module, router_glue: cuda.Module, x3ld_epi: cuda.Module, optin: u32) !Functions {
        var f: Functions = undefined;
        inline for (ld_cfgs, 0..) |c, ci| inline for (ranges, 0..) |r, ri| {
            f.ld[0][ci][ri] = try x3ld.function(ldSymbol(c[0], c[1], r[0], r[1], 0));
            f.ld[1][ci][ri] = if (ci < 2) try x3ld.function(ldSymbol(c[0], c[1], r[0], r[1], 3)) else null;
        };
        inline for (pf_cfgs, 0..) |c, ci| inline for (ranges, 0..) |r, ri| {
            f.pf[ci][ri] = try x3pf.function(pfSymbol(c[0], c[1], r[0], r[1]));
        };
        inline for (.{ EpiKind.gu, EpiKind.dn, EpiKind.dnb }, 0..) |kind, ki| inline for (epi_pds, 0..) |pd, pi| inline for (ranges, 0..) |r, ri| {
            f.epi[ki][pi][ri] = try x3ld_epi.function(epiSymbol(kind, pd, r[0], r[1]));
        };
        inline for (grouped_cfgs, 0..) |c, ci| inline for (upstream_ranges, 0..) |r, ri| {
            f.grouped[ci][ri] = try experts.function(groupedSymbol(c[0], c[1], c[2], r[0], r[1]));
        };
        inline for (0..15) |i| f.dequant[i] = try experts.function(dequantSymbol(i + 2));
        f.group = try experts.function(up_sym.group);
        f.group_rot = try router_glue.function("router_group_rot");
        f.rot_in = .{ try experts.function(up_sym.rot_in_bf16), try experts.function(up_sym.rot_in_f16) };
        f.gateup_epilogue = try experts.function(up_sym.gateup_epilogue);
        f.down_epilogue = try experts.function(up_sym.down_epilogue);
        f.combine = try experts.function(up_sym.combine);
        f.down_combine = .{ try experts.function(up_sym.down_combine_f32), try experts.function(up_sym.down_combine_bf16) };
        inline for (gm_k2s, 0..) |k2, ki| {
            inline for (.{ 1, 2 }) |xm| inline for (0..gu_tiles.len) |c| {
                f.gm_gu[ki][xm - 1][c] = try gmResolve(x3gm, gmGuSymbol(xm, k2, c), GmShape.of(2, xm, k2, gu_tiles[c]), optin);
            };
            inline for (0..dn_tiles.len) |c| f.gm_dn[ki][c] = try gmResolve(x3gm, gmDnSymbol(k2, c), GmShape.of(1, 1, k2, dn_tiles[c]), optin);
            f.gm_dequant[ki] = try x3gm.function(gmDequantSymbol(k2));
        }
        inline for (.{ 1, 2 }) |xm| inline for (v2_gu, 0..) |c, ci| {
            f.gm2_gu[xm - 1][ci] = try gmResolve(x3gm, gm2GuSymbol(xm, c), gm2Shape(2, xm, gu_tiles[c]), optin);
        };
        f.gm2_dn = try gmResolve(x3gm, gm2DnSymbol(0), gm2Shape(1, 1, dn_tiles[0]), optin);
        f.gm2_gu2 = try gmResolve(x3gm3, gm2GuSymbol(2, gu2_cfg), gm2Shape(2, 2, gu_tiles[gu2_cfg]), optin);
        inline for (v3_variants, 0..) |v, i| f.gm3[i] = try gmResolve(x3gm3, gm3Symbol(v), gm2Shape(v.mats, v.xm, v.t), optin);
        f.gm_plan = try x3gm_plan.function("_ZN15dsv41_x3gm_plan11plan_kernelEPKiiiiPiS2_S2_S2_S2_");
        f.gm_rot = .{ .{ try x3gm.function(gm_sym.rot_bf16_1), try x3gm.function(gm_sym.rot_bf16_2) }, .{ try x3gm.function(gm_sym.rot_f16_1), try x3gm.function(gm_sym.rot_f16_2) } };
        return f;
    }
};

/// Opts `f` into `bytes` of dynamic shared memory if it fits the opt-in limit; false refuses the instance.
pub fn fitShared(f: cuda.Function, name: []const u8, bytes: u32, optin: u32) !bool {
    const static: u32 = @intCast(try f.attribute(.shared_size_bytes));
    if (!@import("gpu_memory").fits(static, bytes, optin)) {
        std.log.info("dsv41 kernel refused on this GPU ({d} + {d} B shared > {d}): {s}", .{ static, bytes, optin, name });
        return false;
    }
    if (bytes > 0) try f.allowDynamicShared(bytes); // set whatever the size
    return true;
}

/// x3gm.cu launch<>'s first call: the shared memory attribute, then the CTAs an SM it allows (0: does not fit).
fn gmResolve(m: cuda.Module, name: [:0]const u8, shape: GmShape, optin: u32) !Gm {
    const f = try m.function(name);
    const ok = try fitShared(f, name, shape.smem, optin);
    return .{ .f = f, .threads = shape.threads, .smem = shape.smem, .per_sm = if (ok) try f.occupancy(shape.threads, shape.smem) else 0 };
}

// -- Launches ---------------------------------------------------------------------------------------------------

pub const Input = enum { bf16, f16 };

pub fn int(x: usize) c_int {
    return @intCast(x);
}

fn dim(x: usize) u32 {
    return @intCast(x);
}

pub fn go(f: cuda.Function, s: cuda.Stream, grid: [3]usize, block: usize, shared: usize, pdl: bool, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = dim(grid[0]), .y = dim(grid[1]), .z = dim(grid[2]) }, .block = .{ .x = dim(block) }, .shared = dim(shared), .pdl = pdl }, s, a);
}

/// The arguments every grouped decode kernel (upstream's grouped_kernel, x3ld, x3pf) takes, plus the grid's sizes.
pub const Grouped = struct {
    x0: u64, // fp16 [P, K] rotated inputs of matrix 0 (and of matrix 1 in x1)
    x1: u64,
    tp0: u64, // int64 [E]: each expert's trellis words
    tp1: u64,
    k2_0: u64, // int32 [E]: each expert's K2
    k2_1: u64,
    uids: u64, // int32 [nexp]: distinct experts (group_kernel)
    ucount: u64, // int32 [1]
    members: u64, // int32 [nexp, maxm]: row * 32 + slot codes, -1 after the last
    z: u64, // fp32 [mats, SK, P, N]
    mats: usize,
    K: usize,
    N: usize,
    P: usize,
    SK: usize,
    slots: usize,
    maxm: usize, // members.size(1)
    nexp: usize, // uids.size(0)
    lo: u32, // the widths' range over the table
    hi: u32,

    fn add(g: Grouped, a: *cuda.Args) void {
        for ([_]u64{ g.x0, g.x1, g.tp0, g.tp1, g.k2_0, g.k2_1, g.uids, g.ucount, g.members, g.z }) |v| a.add(v);
        for ([_]usize{ g.K, g.N, g.P, g.SK, g.maxm, g.slots }) |v| a.add(int(v));
    }
};

pub const Error = error{ Unsupported, Shape };

/// dsv41_x3ld_grouped_cuda's checks: mul1, even splits, the ring depth a divisor of a warp's k steps.
pub fn x3ldCheck(g: Grouped, nt: u32, pd: u32, probe: u32) Error!struct { cfg: usize, range: usize } {
    const cfg = cfgIndex(2, &ld_cfgs, .{ nt, pd }) orelse return error.Unsupported;
    if (probe != 0 and (probe != 3 or cfg >= 2)) return error.Unsupported;
    const range = rangeIndex(g.lo, g.hi) orelse return error.Unsupported;
    if (g.SK == 0 or g.K % (16 * g.SK * 4) != 0 or g.N % (16 * nt) != 0) return error.Shape;
    const per_warp = g.K / (16 * g.SK * 4);
    if (per_warp < pd or per_warp % pd != 0) return error.Shape;
    return .{ .cfg = cfg, .range = range };
}

/// dsv41_x3pf_grouped_cuda's checks.
pub fn x3pfCheck(g: Grouped, nt: u32, mtl: u32) Error!struct { cfg: usize, range: usize } {
    const cfg = cfgIndex(2, &pf_cfgs, .{ nt, mtl }) orelse return error.Unsupported;
    const range = rangeIndex(g.lo, g.hi) orelse return error.Unsupported;
    if (g.SK == 0 or g.K % (16 * g.SK * 4) != 0 or g.N % (16 * nt) != 0) return error.Shape;
    return .{ .cfg = cfg, .range = range };
}

/// x3ld_epi.cu's Epi (by value): the epilogue's operands, 0 where a kind does not use one.
pub const Epi = extern struct {
    pick: u64 = 0,
    sv0: u64 = 0, // gate/up: svh_g; down: svh_d
    sv1: u64 = 0, // gate/up: svh_u
    sd: u64 = 0, // gate/up: suh_d
    xd: u64 = 0, // gate/up: Xd
    y: u64 = 0, // down: y
    wts: u64 = 0, // down: the combine's weights
    out: u64 = 0, // down: L.moe (fp32 or bf16)
    ticket: u64 = 0, // int32, zero between launches
    E: c_int = 0,
    limit: f32 = 0,
    act_mode: c_int = 0,
};

comptime {
    std.debug.assert(@sizeOf(Epi) == 88 and @offsetOf(Epi, "E") == 72);
}

/// The ticket words a fused launch counts on (gate/up: expert x member tile x 128 columns; down: row x 128 columns).
pub fn epiTicketWords(g: Grouped, kind: EpiKind) usize {
    return if (kind == .gu) g.nexp * ((g.maxm + 15) / 16) * (g.N / 128) else (g.P / g.slots) * (g.N / 128);
}

/// TF_DSV41_X3LD_EPI's checks: x3ld's at nt 8 and pd 1 / 2, plus the fused tails' shapes (down: one split).
pub fn x3ldEpiCheck(g: Grouped, kind: EpiKind, pd: u32) Error!struct { pd: usize, range: usize } {
    const c = try x3ldCheck(g, 8, pd, 0);
    const pi = std.mem.indexOfScalar(u32, &epi_pds, pd) orelse return error.Unsupported;
    if (g.N % 128 != 0 or g.slots == 0 or g.slots > 32 or g.P % g.slots != 0) return error.Shape;
    switch (kind) {
        .gu => if (g.mats != 2) return error.Shape,
        .dn, .dnb => if (g.mats != 1 or g.SK != 1) return error.Shape,
    }
    return .{ .pd = pi, .range = c.range };
}

pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,
    sms: usize,

    /// x3ld (expert_loads.grouped): Z of every (mat, split, live member row, column), upstream's bits.
    pub fn x3ld(o: Ops, g: Grouped, nt: u32, pd: u32, probe: u32, pdl: bool) !void {
        const c = try x3ldCheck(g, nt, pd, probe);
        const mt = (g.maxm + 15) / 16;
        var a: cuda.Args = .{};
        g.add(&a);
        try go(o.f.ld[if (probe == 3) 1 else 0][c.cfg][c.range].?, o.s, .{ g.nexp, g.N / (16 * nt), g.mats * g.SK * mt }, 128, 0, pdl, &a);
    }

    /// TF_DSV41_X3LD_EPI: x3ld (nt 8) with the gateup_epilogue (`gu`) or down_combine (`dn` / `dnb`) in its tail.
    pub fn x3ldEpi(o: Ops, g: Grouped, kind: EpiKind, pd: u32, epi: Epi) !void {
        const c = try x3ldEpiCheck(g, kind, pd);
        if (epi.ticket == 0 or epi.pick == 0) return error.Shape;
        const mt = (g.maxm + 15) / 16;
        var a: cuda.Args = .{};
        g.add(&a);
        a.add(epi);
        try go(o.f.epi[@backingInt(kind)][c.pd][c.range], o.s, .{ g.nexp, g.N / 128, g.mats * g.SK * mt }, 128, 0, false, &a);
    }

    /// x3pf (expert_prefill.grouped): the same Z, MTL member tiles a program.
    pub fn x3pf(o: Ops, g: Grouped, nt: u32, mtl: u32) !void {
        const c = try x3pfCheck(g, nt, mtl);
        const mg = (g.maxm + 16 * mtl - 1) / (16 * mtl);
        var a: cuda.Args = .{};
        g.add(&a);
        try go(o.f.pf[c.cfg][c.range], o.s, .{ g.nexp, g.N / (16 * nt), g.mats * g.SK * mg }, 128, 0, false, &a);
    }

    /// Upstream's grouped kernel (exl3x_grouped_cuda at cb 2): the reference x3ld and x3pf keep the bits of.
    pub fn grouped(o: Ops, g: Grouped, nt: u32, warps: u32, pf: u32) !void {
        const cfg = cfgIndex(3, &grouped_cfgs, .{ nt, warps, pf }) orelse return error.Unsupported;
        if (g.SK == 0 or g.K % (16 * g.SK * warps) != 0 or g.N % (16 * nt) != 0) return error.Shape;
        const mt = (g.maxm + 15) / 16;
        var a: cuda.Args = .{};
        g.add(&a);
        try go(o.f.grouped[cfg][upstreamRangeIndex(g.lo, g.hi)], o.s, .{ g.nexp, g.N / (16 * nt), g.mats * g.SK * mt }, warps * 32, 0, false, &a);
    }

    /// exl3x_group_cuda: distinct experts of `pick` [R, slots] in id order and their members, one 1,024-thread block.
    pub fn group(o: Ops, pick: u64, uids: u64, ucount: u64, members: u64, R: usize, slots: usize, E: usize, maxm: usize) !void {
        if (E > 1024 * 4 or slots > 32) return error.Shape;
        const smem = R * slots * 4;
        if (smem > 48 * 1024) return error.Shape; // the binding never opts in to more
        var a: cuda.Args = .{};
        for ([_]u64{ pick, uids, ucount, members }) |v| a.add(v);
        for ([_]usize{ R, slots, E, maxm }) |v| a.add(int(v));
        try go(o.f.group, o.s, .{ 1, 1, 1 }, 1024, smem, false, &a);
    }

    /// exl3x_rot_in_cuda: Xh = fp16((x * suh) H / sqrt 128) for gate and up of every routed slot.
    pub fn rotIn(o: Ops, input: Input, x: u64, x_stride: usize, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, rows: usize, K: usize, slots: usize, E: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        for ([_]u64{ pick, suh0, suh1, out0, out1 }) |v| a.add(v);
        for ([_]usize{ K, slots, E }) |v| a.add(int(v));
        try go(o.f.rot_in[@backingInt(input)], o.s, .{ rows * slots, K / 128, 2 }, 32, 0, false, &a);
    }

    /// Independent grouping/rotation CTAs share a launch, with no inter-CTA handoff. Invalid picks leave Xh untouched.
    pub fn groupRotIn(o: Ops, x: u64, x_stride: usize, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, uids: u64, ucount: u64, members: u64, rows: usize, K: usize, slots: usize, E: usize, maxm: usize) !void {
        if (rows < 1 or rows > 64 or K != 5120 or slots < 1 or slots > 9 or E < 1 or E > 512 or x_stride < K or maxm < 1) return error.Shape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        for ([_]u64{ pick, suh0, suh1, out0, out1, uids, ucount, members }) |v| a.add(v);
        for ([_]usize{ rows, K, slots, E, maxm }) |v| a.add(int(v));
        const tiles = rows * slots * (K / 128) * 2;
        try go(o.f.group_rot, o.s, .{ 1 + (tiles + 31) / 32, 1, 1 }, 1024, 0, false, &a);
    }

    /// exl3x_gateup_epilogue_cuda: splits summed, rotated, * svh, SwiGLU (act_mode 0 bf16 roundings, 1 fp32), Xd.
    pub fn gateupEpilogue(o: Ops, z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, rows: usize, P: usize, N: usize, SK: usize, slots: usize, E: usize, limit: f32, act_mode: c_int) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ z, pick, svh_g, svh_u, suh_d, xd }) |v| a.add(v);
        for ([_]usize{ P, N, SK, E }) |v| a.add(int(v));
        a.add(limit);
        a.add(act_mode);
        try go(o.f.gateup_epilogue, o.s, .{ rows * slots, N / 128, 1 }, 32, 0, false, &a);
    }

    /// exl3x_down_epilogue_cuda: Y = (splits summed) H * svh_d, fp32.
    pub fn downEpilogue(o: Ops, z: u64, pick: u64, svh_d: u64, y: u64, rows: usize, P: usize, D: usize, SK: usize, slots: usize, E: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ z, pick, svh_d, y }) |v| a.add(v);
        for ([_]usize{ P, D, SK, E }) |v| a.add(int(v));
        try go(o.f.down_epilogue, o.s, .{ rows * slots, D / 128, 1 }, 32, 0, false, &a);
    }

    /// exl3x_combine_cuda: out[r] = fma chain over slots in order of wts[r][k] * y[r * slots + k].
    pub fn combine(o: Ops, y: u64, wts: u64, out: u64, rows: usize, D: usize, slots: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ y, wts, out }) |v| a.add(v);
        for ([_]usize{ D, slots }) |v| a.add(int(v));
        try go(o.f.combine, o.s, .{ rows, (D + 255) / 256, 1 }, 256, 0, false, &a);
    }

    /// exl3x_down_combine_cuda: down_epilogue then combine in one launch, same bits; `out_bf16`: bf16 out (RNE).
    pub fn downCombine(o: Ops, z: u64, pick: u64, svh_d: u64, y: u64, wts: u64, out: u64, rows: usize, P: usize, D: usize, SK: usize, slots: usize, E: usize, out_bf16: bool) !void {
        if (slots > 32) return error.Shape;
        var a: cuda.Args = .{};
        for ([_]u64{ z, pick, svh_d, y, wts, out }) |v| a.add(v);
        for ([_]usize{ P, D, SK, E, slots }) |v| a.add(int(v));
        try go(o.f.down_combine[@intFromBool(out_bf16)], o.s, .{ rows, D / 128, 1 }, 32 * slots, 0, false, &a);
    }

    /// exl3x_dequant_cuda at cb 2: W_q [K, N] fp16 through the grouped kernels' lane decode (tests).
    pub fn dequant(o: Ops, t: u64, out: u64, K: usize, N: usize, k2: u32) !void {
        if (k2 < 2 or k2 > 16) return error.Unsupported;
        var a: cuda.Args = .{};
        a.add(t);
        a.add(out);
        a.add(int(K));
        a.add(int(N));
        try go(o.f.dequant[k2 - 2], o.s, .{ K / 16, N / 16, 1 }, 32, 0, false, &a);
    }

    pub const gmRot = gmk.gmRot;
    pub const gmGateup = gmk.gmGateup;
    pub const gmDown = gmk.gmDown;
    pub const gm2Gateup = gmk.gm2Gateup;
    pub const gm2Down = gmk.gm2Down;
    pub const gm3Gateup = gmk.gm3Gateup;
    pub const gm3Down = gmk.gm3Down;
    pub const gmPlan = gmk.gmPlan;
    pub const gmDequant = gmk.gmDequant;

    pub fn gmLaunch(o: Ops, g: Gm, args: GmArgs, ticket: bool) !void {
        if (g.per_sm == 0) return error.Unsupported; // "the configuration does not fit an SM"
        var b = args;
        if (!ticket) b.ticket = 0;
        var a: cuda.Args = .{};
        a.add(b);
        // TF_DSV41_PF_TBO_GM_SMS: persistent CTAs on at most that many SMs; which CTA runs an item changes no bit
        const sms = if (gm_sms_cap > 0) @min(o.sms, gm_sms_cap) else o.sms;
        try go(g.f, o.s, .{ sms * g.per_sm, 1, 1 }, g.threads, g.smem, false, &a);
    }
};

test "ranges as dispatch_ranges and TF_RANGES pick them" {
    try std.testing.expectEqual(@as(?usize, 0), rangeIndex(8, 8));
    try std.testing.expectEqual(@as(?usize, 1), rangeIndex(4, 10));
    try std.testing.expectEqual(@as(?usize, 1), rangeIndex(8, 10));
    try std.testing.expectEqual(@as(?usize, 2), rangeIndex(4, 12));
    try std.testing.expectEqual(@as(?usize, null), rangeIndex(4, 16));
    try std.testing.expectEqual(@as(?usize, null), rangeIndex(1, 4));
    try std.testing.expectEqual(@as(usize, 2), upstreamRangeIndex(4, 16));
}

test "x3ld checks mirror the binding's" {
    const g: Grouped = .{ .x0 = 0, .x1 = 0, .tp0 = 0, .tp1 = 0, .k2_0 = 0, .k2_1 = 0, .uids = 0, .ucount = 0, .members = 0, .z = 0, .mats = 2, .K = 4096, .N = 1024, .P = 64, .SK = 1, .slots = 8, .maxm = 64, .nexp = 64, .lo = 4, .hi = 12 };
    const c = try x3ldCheck(g, 8, 2, 0);
    try std.testing.expectEqual(@as(usize, 1), c.cfg);
    try std.testing.expectEqual(@as(usize, 2), c.range);
    try std.testing.expectError(error.Unsupported, x3ldCheck(g, 4, 2, 3));
    try std.testing.expectError(error.Unsupported, x3ldCheck(g, 4, 1, 0));
    var bad = g;
    bad.K = 4096 + 16; // not 16 x SK x 4 warps
    try std.testing.expectError(error.Shape, x3ldCheck(bad, 8, 1, 0));
    bad = g;
    bad.K = 16 * 4 * 3; // three k steps a warp: not a multiple of pd 2
    try std.testing.expectError(error.Shape, x3ldCheck(bad, 8, 2, 0));
}

test "x3gm v2 shapes and tuning" {
    // gate/up cfg 0, one input: K2 10's ring cut from 3 to 2 stages keeps two CTAs an SM; the widest width sets SMEM
    const s = gm2Shape(2, 1, gu_tiles[0]);
    try std.testing.expectEqual(@as(u32, 256), s.threads);
    try std.testing.expect(2 * (@as(usize, s.smem) + 1536) <= 102400);
    try std.testing.expectEqual([2]usize{ 0, 0 }, gmTuned2(null, null, true));
    try std.testing.expectEqual([2]usize{ 1, 0 }, gmTuned2(3, 2, false));
    try std.testing.expect(gm2WidthsOk(&.{ 3, 10, 6 }));
    try std.testing.expect(!gm2WidthsOk(&.{ 2, 6 }));
}

test "x3gm v2 at GU2 (TF_DSV41_GM_GU2, x3gm3.cu's instance): 512 threads, every width's ring in sm_121's opt-in memory" {
    const s = gm2Shape(2, 2, gu_tiles[gu2_cfg]);
    try std.testing.expectEqual(@as(u32, 512), s.threads);
    try std.testing.expect(s.smem > 0 and s.smem + 4 * 8 * gu_tiles[gu2_cfg][1] + 4 <= gm_smem_max);
    try std.testing.expectEqualStrings("_ZN10dsv41_x3gm10gm2_kernelILi2ELi2ELi1ELi16ELi2ELi4ELi4EEEvNS_4ArgsE", gm2GuSymbol(2, gu2_cfg));
}

test "x3gm shapes, fits and tuning as the reference tuner" {
    // G6 gate/up cfg 0 at K2 6, one input: 3 x (64 x 64 x 2 + 2 x 4 x 8 x 24 x 4) = 49,152 + 4,608 ... (smem_bytes)
    const s = GmShape.of(2, 1, 6, gu_tiles[0]);
    try std.testing.expectEqual(@as(u32, 256), s.threads);
    try std.testing.expectEqual(@as(u32, 3 * (64 * 64 * 2 + 2 * 4 * 8 * 24 * 4)), s.smem);
    try std.testing.expect(gmFits(.gu, 0, 10, 1));
    try std.testing.expect(!gmFits(.gu, 3, 10, 2));
    try std.testing.expectEqual(@as(u32, 128), gmMembers(.dn, 2));
    // 2,048 rows x 6 slots over 384 experts (32 members): G6's (0 / 1, 0)
    try std.testing.expectEqual([2]usize{ 0, 0 }, gmTuned(&tune_default, null, null, 2048, 6, 384, true, 6, 6));
    try std.testing.expectEqual([2]usize{ 1, 0 }, gmTuned(&tune_default, null, null, 2048, 6, 384, false, 6, 6));
    // 4,096 rows (64 members): the 4K entry, cfg 3 / 2 when it fits the width
    try std.testing.expectEqual([2]usize{ 3, 2 }, gmTuned(&tune_default, null, null, 4096, 6, 384, true, 6, 6));
    try std.testing.expectEqual([2]usize{ 2, 2 }, gmTuned(&tune_default, null, null, 4096, 6, 384, false, 6, 6));
    try std.testing.expectEqual([2]usize{ 5, 4 }, gmTuned(&tune_default, 5, 4, 4096, 6, 384, false, 6, 6));
}

test "x3ld_epi checks" {
    const g: Grouped = .{ .x0 = 0, .x1 = 0, .tp0 = 0, .tp1 = 0, .k2_0 = 0, .k2_1 = 0, .uids = 0, .ucount = 0, .members = 0, .z = 0, .mats = 2, .K = 5120, .N = 1152, .P = 28, .SK = 4, .slots = 7, .maxm = 4, .nexp = 28, .lo = 4, .hi = 10 };
    _ = try x3ldEpiCheck(g, .gu, 1);
    try std.testing.expectError(error.Shape, x3ldEpiCheck(g, .dn, 1));
    var d = g;
    d.mats = 1;
    d.K = 1152;
    d.N = 5120;
    d.SK = 1;
    _ = try x3ldEpiCheck(d, .dnb, 1);
    try std.testing.expectEqual(@as(usize, 4 * 40), epiTicketWords(d, .dn));
    try std.testing.expectEqual(@as(usize, 28 * 1 * 9), epiTicketWords(g, .gu));
    try std.testing.expectEqualStrings("dsv41_x3ld_epi_dnb_2_2_10", epiSymbol(.dnb, 2, 2, 10));
}
