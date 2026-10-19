//! DeepSeek-V4.1's x3gm routed-expert GEMM: width and tile tables, v2/v3 shapes and tuning, args and launches.

const std = @import("std");
const cuda = @import("root.zig");
const exl3 = @import("exl3_block_exl3.zig");

const Ops = exl3.Ops;
const Input = exl3.Input;
const Error = exl3.Error;
const int = exl3.int;
const go = exl3.go;

/// x3gm's widths (x3gm.K2S) and tile tables (MTL, NG, KS, NSA, NGR) by configuration id (x3gm.cu GM_GU* / GM_DN*).
pub const gm_k2s = [_]u32{ 3, 4, 5, 6, 7, 8, 10 };
pub const gu_tiles = [_][5]u32{ .{ 2, 8, 4, 3, 4 }, .{ 2, 8, 2, 4, 4 }, .{ 1, 16, 2, 4, 4 }, .{ 1, 16, 4, 3, 4 }, .{ 2, 16, 2, 3, 4 }, .{ 2, 8, 2, 6, 4 } };
pub const dn_tiles = [_][5]u32{ .{ 2, 8, 4, 4, 8 }, .{ 1, 8, 4, 4, 8 }, .{ 1, 16, 4, 4, 8 }, .{ 1, 16, 2, 6, 8 }, .{ 2, 16, 4, 3, 16 } };
/// sm_121's opt-in dynamic shared memory a CTA (x3gm.SMEM_MAX).
pub const gm_smem_max: usize = 101376;

pub fn gmK2Index(k2: u32) ?usize {
    return std.mem.indexOfScalar(u32, &gm_k2s, k2);
}

/// x3gm.cu Cfg<MATS, XM, K2, MTL, NG, KS, NSA, NGR>: threads and dynamic shared memory of one instance.
pub const GmShape = struct {
    threads: u32,
    smem: u32,

    pub fn of(mats: u32, xm: u32, k2: u32, t: [5]u32) GmShape {
        const mtl, const ng, const ks, const nsa, const ngr = t;
        const bm = ng * 8;
        const w = mats * (8 / mtl);
        const a_bytes: usize = @as(usize, xm) * bm * (ks * 16) * 2;
        const w_bytes: usize = @as(usize, mats) * ks * 8 * (4 * k2) * 4;
        const ring = nsa * (a_bytes + w_bytes);
        const epi: usize = @as(usize, mats) * (ngr * 8) * 132 * 4;
        return .{ .threads = w * 32, .smem = @intCast(@max(ring, epi)) };
    }
};

/// x3gm v2 (gm2_kernel): the tiles it is built at (V2_GU / V2_DN).
pub const v2_gu = [_]usize{ 0, 1 };
/// gate / up's 128-member tile (gu_tiles[2], NG 16) that x3gm3.cu builds gm2_kernel at, two inputs only
pub const gu2_cfg: usize = 2;
pub const v2_dn = [_]usize{0};

/// x3gm.cu Smem2<MATS, XM, tile>: the widest width's dynamic shared memory; rings cut (Ring2) to keep 2 CTAs an SM.
pub fn gm2Shape(mats: u32, xm: u32, t: [5]u32) GmShape {
    const mtl, const ng, const ks, const nsa, const ngr = t;
    const four = GmShape.of(mats, xm, 4, t);
    const two_ctas = 2 * (@as(usize, four.smem) + 1536) <= 102400 and four.threads <= 256 and mtl * ng <= 16; // Cfg<K2 4>::MINB == 2
    var smem: u32 = 0;
    for (gm_k2s) |k2| {
        const stage: usize = @as(usize, xm) * (ng * 8) * (ks * 16) * 2 + @as(usize, mats) * ks * 8 * (4 * k2) * 4;
        var ring: u32 = nsa;
        if (two_ctas) {
            while (ring > 2 and 2 * (stage * ring + 1536) > 102400) ring -= 1;
        }
        smem = @max(smem, GmShape.of(mats, xm, k2, .{ mtl, ng, ks, ring, ngr }).smem);
    }
    return .{ .threads = four.threads, .smem = smem };
}

/// x3gm v3 (x3gm3.cu's gm3_kernel): v2 plus next-item read-ahead and LEAN reads; variants in its instance order.
pub const V3 = struct { mats: u32, xm: u32, t: [5]u32, lean: u32 };
pub const v3_variants = [_]V3{
    .{ .mats = 2, .xm = 2, .t = .{ 2, 8, 2, 4, 4 }, .lean = 0 }, // v2's GU1 + claim-ahead
    .{ .mats = 2, .xm = 2, .t = .{ 2, 8, 2, 4, 4 }, .lean = 1 },
    .{ .mats = 2, .xm = 2, .t = .{ 1, 8, 2, 4, 4 }, .lean = 1 }, // 16-warp CTAs
    .{ .mats = 2, .xm = 2, .t = .{ 1, 8, 4, 3, 4 }, .lean = 1 },
    .{ .mats = 2, .xm = 1, .t = .{ 2, 8, 4, 3, 4 }, .lean = 1 }, // v2's GU0 (one input) + claim-ahead + LEAN
    .{ .mats = 2, .xm = 1, .t = .{ 1, 8, 4, 3, 4 }, .lean = 1 },
    .{ .mats = 1, .xm = 1, .t = .{ 2, 8, 4, 4, 8 }, .lean = 0 }, // v2's DN0 + claim-ahead
    .{ .mats = 1, .xm = 1, .t = .{ 2, 8, 4, 4, 8 }, .lean = 1 },
    .{ .mats = 1, .xm = 1, .t = .{ 1, 8, 4, 4, 8 }, .lean = 0 }, // 8-warp CTAs: 16 warps an SM
    .{ .mats = 1, .xm = 1, .t = .{ 1, 8, 4, 4, 8 }, .lean = 1 },
    .{ .mats = 1, .xm = 1, .t = .{ 1, 8, 2, 6, 8 }, .lean = 1 },
};

pub fn gm3Symbol(comptime v: V3) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10dsv41_x3gm10gm3_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvNS_4ArgsE", .{ v.mats, v.xm, v.t[0], v.t[1], v.t[2], v.t[3], v.t[4], v.lean });
}

/// v3's variants for one projection: gate / up at one or two inputs (xm), or down.
pub fn v3Of(mats: u32, xm: u32, out: []usize) []usize {
    var n: usize = 0;
    for (v3_variants, 0..) |v, i| if (v.mats == mats and v.xm == xm) {
        out[n] = i;
        n += 1;
    };
    return out[0..n];
}

/// TF_DSV41_GM_V3=1: prefill's x3gm through v3 (v3_default's variants); unset or 0: v2 (default; v3 is bit-identical).
pub fn gmV3() bool {
    const v = std.c.getenv("TF_DSV41_GM_V3") orelse return false;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// v3's default variant of a projection (from per-shape timing).
pub const v3_default: struct { gu: [2]usize, dn: usize } = .{ .gu = .{ 4, 1 }, .dn = 9 };

/// x3gm.tuned2: (gate/up cfg, down cfg) of a v2 block: forced ids where built, else 0 / 1 by rotated inputs; down 0.
pub fn gmTuned2(force_gu: ?usize, force_dn: ?usize, shx: bool) [2]usize {
    const gu = if (force_gu) |f| (if (std.mem.indexOfScalar(usize, &v2_gu, f) != null) f else null) else null;
    const dn = if (force_dn) |f| (if (std.mem.indexOfScalar(usize, &v2_dn, f) != null) f else null) else null;
    return .{ gu orelse (if (shx) 0 else 1), dn orelse 0 };
}

/// The host check before a v2 launch: every expert's width is one gm2_kernel dispatches.
pub fn gm2WidthsOk(widths: []const u32) bool {
    for (widths) |w| if (gmK2Index(w) == null) return false;
    return true;
}

pub fn gm2GuSymbol(comptime xm: u32, comptime cfg: usize) [:0]const u8 {
    const t = gu_tiles[cfg];
    return std.fmt.comptimePrint("_ZN10dsv41_x3gm10gm2_kernelILi2ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvNS_4ArgsE", .{ xm, t[0], t[1], t[2], t[3], t[4] });
}

pub fn gm2DnSymbol(comptime cfg: usize) [:0]const u8 {
    const t = dn_tiles[cfg];
    return std.fmt.comptimePrint("_ZN10dsv41_x3gm10gm2_kernelILi1ELi1ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvNS_4ArgsE", .{ t[0], t[1], t[2], t[3], t[4] });
}

/// x3gm.smem_bytes: Cfg::SMEM plus the static rows / item words.
pub fn gmSmemBytes(t: [5]u32, mats: u32, xm: u32, k2: u32) usize {
    return @as(usize, GmShape.of(mats, xm, k2, t).smem) + 4 * 8 * t[1] + 4;
}

pub const GmKind = enum { gu, dn };

/// x3gm_plan.cu takes E < 512 experts.
pub const plan_max_experts = 512;

/// x3gm.plan's table length: P / bm + E + 1 passes at most.
pub fn planPasses(pairs: usize, experts: usize, bm: usize) usize {
    return pairs / bm + experts + 1;
}

/// x3gm.fits: whether a configuration builds a CTA at this width (gate/up: `xm` rotated inputs) on sm_121.
pub fn gmFits(kind: GmKind, cfg: usize, k2: u32, xm: u32) bool {
    return switch (kind) {
        .gu => cfg < gu_tiles.len and gmSmemBytes(gu_tiles[cfg], 2, xm, k2) <= gm_smem_max,
        .dn => cfg < dn_tiles.len and gmSmemBytes(dn_tiles[cfg], 1, 1, k2) <= gm_smem_max,
    };
}

/// x3gm.bm_of: members a pass of a configuration.
pub fn gmMembers(kind: GmKind, cfg: usize) u32 {
    return 8 * (if (kind == .gu) gu_tiles[cfg] else dn_tiles[cfg])[1];
}

/// x3gm.TUNE row: from `members` an expert on average, gate/up `gu` (null: by shx; else one / two inputs), down `dn`.
pub const TuneRow = struct { members: f64, gu: ?[2]u32, dn: u32 };
pub const tune_default = [_]TuneRow{ .{ .members = 0, .gu = null, .dn = 0 }, .{ .members = 48, .gu = .{ 3, 2 }, .dn = 2 } };

/// x3gm.tuned: (gate/up cfg, down cfg) of a block of `rows` rows; `force_gu` / `force_dn` are TF_DSV41_GM_CFG's ids.
pub fn gmTuned(table: []const TuneRow, force_gu: ?usize, force_dn: ?usize, rows: usize, slots: usize, experts: usize, shx: bool, k2gu: u32, k2d: u32) [2]usize {
    const m = @as(f64, @floatFromInt(rows * slots)) / @as(f64, @floatFromInt(@max(1, experts)));
    var ent = table[0];
    for (table) |r| {
        if (m > r.members or r.members == 0) ent = r;
    }
    var gu: usize = if (ent.gu) |p| (if (shx) p[0] else p[1]) else (if (shx) 0 else 1);
    if (force_gu) |f| {
        gu = f;
    } else if (!gmFits(.gu, gu, k2gu, if (shx) 1 else 2)) {
        gu = if (shx) 0 else 1;
    }
    var dn: usize = ent.dn;
    if (force_dn) |f| {
        dn = f;
    } else if (!gmFits(.dn, dn, k2d, 1)) {
        dn = 0;
    }
    return .{ gu, dn };
}

pub fn gmGuSymbol(comptime xm: u32, comptime k2: u32, comptime cfg: usize) [:0]const u8 {
    const t = gu_tiles[cfg];
    return std.fmt.comptimePrint("_ZN10dsv41_x3gm9gm_kernelILi2ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvNS_4ArgsE", .{ xm, k2, t[0], t[1], t[2], t[3], t[4] });
}

pub fn gmDnSymbol(comptime k2: u32, comptime cfg: usize) [:0]const u8 {
    const t = dn_tiles[cfg];
    return std.fmt.comptimePrint("_ZN10dsv41_x3gm9gm_kernelILi1ELi1ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi{d}EEEvNS_4ArgsE", .{ k2, t[0], t[1], t[2], t[3], t[4] });
}

pub fn gmDequantSymbol(comptime k2: u32) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10dsv41_x3gm14dequant_kernelILi{d}EEEvPKjP6__halfi", .{k2});
}

/// x3gm.cu's Args, passed by value.
pub const GmArgs = extern struct {
    x0: u64 = 0,
    x1: u64 = 0,
    t0: u64 = 0,
    t1: u64 = 0,
    tp0: u64 = 0,
    tp1: u64 = 0,
    k2e: u64 = 0, // int32 [E] each expert's K2 (gm2_kernel; gm_kernel ignores it)
    order: u64 = 0,
    pe: u64 = 0,
    poff: u64 = 0,
    pcnt: u64 = 0,
    npass: u64 = 0,
    ticket: u64 = 0,
    sv0: u64 = 0,
    sv1: u64 = 0,
    sd: u64 = 0,
    xd: u64 = 0,
    y: u64 = 0,
    K: c_int = 0,
    N: c_int = 0,
    limit: f32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(GmArgs) == 160 and @offsetOf(GmArgs, "k2e") == 48 and @offsetOf(GmArgs, "K") == 144 and @offsetOf(GmArgs, "limit") == 152);
}

/// x3gm's per-launch checks (launch<>): K in whole stages, N in 128-column Hadamard blocks.
pub fn gmCheck(kind: GmKind, cfg: usize, K: usize, N: usize) Error!void {
    const ks = (if (kind == .gu) gu_tiles[cfg] else dn_tiles[cfg])[2];
    if ((K / 16) % ks != 0 or N % 128 != 0) return error.Shape;
}

/// dsv41_x3gm_rot_cuda: out_m[p] = fp16((x[p / slots] * suh_m[e]) H / sqrt 128) for `mats` matrices.
pub fn gmRot(o: Ops, input: Input, x: u64, x_stride: usize, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, K: usize, slots: usize, pairs: usize, mats: usize) !void {
    if (mats != 1 and mats != 2) return error.Unsupported;
    const items = pairs * (K / 128);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(int(x_stride));
    for ([_]u64{ pick, suh0, suh1, out0, out1 }) |v| a.add(v);
    for ([_]usize{ K, slots, items }) |v| a.add(int(v));
    try go(o.f.gm_rot[@backingInt(input)][mats - 1], o.s, .{ (items + 7) / 8, 1, 1 }, 256, 0, false, &a);
}

/// dsv41_x3gm_gateup_cuda: Xd of every pass's members; `tables`: tg / tu are int64 pointer tables (ragged).
pub fn gmGateup(o: Ops, xg: u64, xu: u64, tg: u64, tu: u64, order: u64, pe: u64, poff: u64, pcnt: u64, npass: u64, ticket: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, K: usize, N: usize, k2: u32, shx: bool, cfg: usize, use_ticket: bool, limit: f32, tables: bool) !void {
    const ki = gmK2Index(k2) orelse return error.Unsupported;
    if (cfg >= gu_tiles.len) return error.Unsupported;
    try gmCheck(.gu, cfg, K, N);
    var a: GmArgs = .{ .x0 = xg, .x1 = xu, .order = order, .pe = pe, .poff = poff, .pcnt = pcnt, .npass = npass, .ticket = ticket, .sv0 = svh_g, .sv1 = svh_u, .sd = suh_d, .xd = xd, .K = int(K), .N = int(N), .limit = limit };
    if (tables) {
        a.tp0 = tg;
        a.tp1 = tu;
    } else {
        a.t0 = tg;
        a.t1 = tu;
    }
    try o.gmLaunch(o.f.gm_gu[ki][if (shx) 0 else 1][cfg], a, use_ticket);
}

/// dsv41_x3gm_down_cuda: Y (fp32) of every pass's members.
pub fn gmDown(o: Ops, xd: u64, td: u64, order: u64, pe: u64, poff: u64, pcnt: u64, npass: u64, ticket: u64, svh_d: u64, y: u64, K: usize, N: usize, k2: u32, cfg: usize, use_ticket: bool, tables: bool) !void {
    const ki = gmK2Index(k2) orelse return error.Unsupported;
    if (cfg >= dn_tiles.len) return error.Unsupported;
    try gmCheck(.dn, cfg, K, N);
    var a: GmArgs = .{ .x0 = xd, .x1 = xd, .order = order, .pe = pe, .poff = poff, .pcnt = pcnt, .npass = npass, .ticket = ticket, .sv0 = svh_d, .y = y, .K = int(K), .N = int(N) };
    if (tables) {
        a.tp0 = td;
        a.tp1 = td;
    } else {
        a.t0 = td;
        a.t1 = td;
    }
    try o.gmLaunch(o.f.gm_dn[ki][cfg], a, use_ticket);
}

/// dsv41_x3gm_gateup2_cuda (x3gm v2): Xd of every pass of every width in one launch; `tables`: ragged pointers.
pub fn gm2Gateup(o: Ops, xg: u64, xu: u64, tg: u64, tu: u64, k2e: u64, order: u64, pe: u64, poff: u64, pcnt: u64, npass: u64, ticket: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, K: usize, N: usize, shx: bool, cfg: usize, use_ticket: bool, limit: f32, tables: bool) !void {
    const gu2 = cfg == gu2_cfg and !shx;
    const ci = std.mem.indexOfScalar(usize, &v2_gu, cfg) orelse if (gu2) 0 else return error.Unsupported;
    try gmCheck(.gu, cfg, K, N);
    var a: GmArgs = .{ .x0 = xg, .x1 = xu, .k2e = k2e, .order = order, .pe = pe, .poff = poff, .pcnt = pcnt, .npass = npass, .ticket = ticket, .sv0 = svh_g, .sv1 = svh_u, .sd = suh_d, .xd = xd, .K = int(K), .N = int(N), .limit = limit };
    if (tables) {
        a.tp0 = tg;
        a.tp1 = tu;
    } else {
        a.t0 = tg;
        a.t1 = tu;
    }
    try o.gmLaunch(if (gu2) o.f.gm2_gu2 else o.f.gm2_gu[if (shx) 0 else 1][ci], a, use_ticket);
}

/// dsv41_x3gm_down2_cuda (x3gm v2): Y of every pass of every width in one launch; `k2e` the experts' down widths.
pub fn gm2Down(o: Ops, xd: u64, td: u64, k2e: u64, order: u64, pe: u64, poff: u64, pcnt: u64, npass: u64, ticket: u64, svh_d: u64, y: u64, K: usize, N: usize, cfg: usize, use_ticket: bool, tables: bool) !void {
    if (std.mem.indexOfScalar(usize, &v2_dn, cfg) == null) return error.Unsupported;
    try gmCheck(.dn, cfg, K, N);
    var a: GmArgs = .{ .x0 = xd, .x1 = xd, .k2e = k2e, .order = order, .pe = pe, .poff = poff, .pcnt = pcnt, .npass = npass, .ticket = ticket, .sv0 = svh_d, .y = y, .K = int(K), .N = int(N) };
    if (tables) {
        a.tp0 = td;
        a.tp1 = td;
    } else {
        a.t0 = td;
        a.t1 = td;
    }
    try o.gmLaunch(o.f.gm2_dn, a, use_ticket);
}

/// x3gm v3 gate/up (gm3_kernel), variant `v` of v3_variants (mats 2, its xm): gm2Gateup's arguments and bits.
pub fn gm3Gateup(o: Ops, v: usize, xg: u64, xu: u64, tg: u64, tu: u64, k2e: u64, order: u64, pe: u64, poff: u64, pcnt: u64, npass: u64, ticket: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, K: usize, N: usize, shx: bool, use_ticket: bool, limit: f32, tables: bool) !void {
    const vv = v3_variants[v];
    if (vv.mats != 2 or vv.xm != @as(u32, if (shx) 1 else 2)) return error.Unsupported;
    if ((K / 16) % vv.t[2] != 0 or N % 128 != 0) return error.Shape;
    var a: GmArgs = .{ .x0 = xg, .x1 = xu, .k2e = k2e, .order = order, .pe = pe, .poff = poff, .pcnt = pcnt, .npass = npass, .ticket = ticket, .sv0 = svh_g, .sv1 = svh_u, .sd = suh_d, .xd = xd, .K = int(K), .N = int(N), .limit = limit };
    if (tables) {
        a.tp0 = tg;
        a.tp1 = tu;
    } else {
        a.t0 = tg;
        a.t1 = tu;
    }
    try o.gmLaunch(o.f.gm3[v], a, use_ticket);
}

/// x3gm v3 down (gm3_kernel), variant `v` (mats 1): gm2Down's arguments and bits.
pub fn gm3Down(o: Ops, v: usize, xd: u64, td: u64, k2e: u64, order: u64, pe: u64, poff: u64, pcnt: u64, npass: u64, ticket: u64, svh_d: u64, y: u64, K: usize, N: usize, use_ticket: bool, tables: bool) !void {
    const vv = v3_variants[v];
    if (vv.mats != 1) return error.Unsupported;
    if ((K / 16) % vv.t[2] != 0 or N % 128 != 0) return error.Shape;
    var a: GmArgs = .{ .x0 = xd, .x1 = xd, .k2e = k2e, .order = order, .pe = pe, .poff = poff, .pcnt = pcnt, .npass = npass, .ticket = ticket, .sv0 = svh_d, .y = y, .K = int(K), .N = int(N) };
    if (tables) {
        a.tp0 = td;
        a.tp1 = td;
    } else {
        a.t0 = td;
        a.t1 = td;
    }
    try o.gmLaunch(o.f.gm3[v], a, use_ticket);
}

/// x3gm.plan on the device (x3gm_plan.cu): order [pairs], pe / poff / pcnt [planPasses], npass [1] in `bm` members.
pub fn gmPlan(o: Ops, pick: u64, pairs: usize, E: usize, bm: usize, order: u64, pe: u64, poff: u64, pcnt: u64, npass: u64) !void {
    if (E < 1 or E >= plan_max_experts or bm < 1 or pairs < 1) return error.Shape;
    var a: cuda.Args = .{};
    a.add(pick);
    for ([_]usize{ pairs, E, bm }) |v| a.add(int(v));
    for ([_]u64{ order, pe, poff, pcnt, npass }) |v| a.add(v);
    try go(o.f.gm_plan, o.s, .{ 1, 1, 1 }, 512, 0, false, &a);
}

/// dsv41_x3gm_dequant_cuda: W_q [K, N] fp16 through the gm kernels' decode_a (tests).
pub fn gmDequant(o: Ops, t: u64, out: u64, K: usize, N: usize, k2: u32) !void {
    const ki = gmK2Index(k2) orelse return error.Unsupported;
    var a: cuda.Args = .{};
    a.add(t);
    a.add(out);
    a.add(int(N));
    try go(o.f.gm_dequant[ki], o.s, .{ K / 16, N / 16, 1 }, 32, 0, false, &a);
}
