//! Launches of the release experts (release_experts.cu, image dsv41_rel_ex), sm_121a only.
const std = @import("std");
const cuda = @import("cuda");

/// Mirrors RelGroup: ids [m, topk], slot_of [nids] (-1: unheld) into offs, pair_tok, pair_of, active, nactive.
pub const Group = extern struct {
    ids: u64,
    slot_of: u64,
    offs: u64,
    pair_tok: u64,
    pair_of: u64,
    active: u64,
    nactive: u64,
    m: c_int,
    topk: c_int,
    nslots: c_int,
    nids: c_int,
};

/// Mirrors RelGateUp: row codes and scales (views), w1/w3, route weights, Group's lists; codes and scales out.
pub const GateUp = extern struct {
    x: u64,
    x_off: u64 = 0,
    x_ld: i64,
    xs: u64,
    xs_off: u64 = 0,
    xs_ld: i64,
    w1: u64,
    s1: u64,
    w3: u64,
    s3: u64,
    wts: u64,
    pair_tok: u64,
    offs: u64,
    active: u64,
    nactive: u64,
    codes: u64,
    scales: u64,
    gsum: u64 = 0,
    usum: u64 = 0,
    v: u64 = 0,
    n: c_int,
    k: c_int,
    topk: c_int,
    pad: c_int = 0,
};

/// Mirrors RelDown: the grouped pairs' codes [pairs, k] and scales, w2 and its scales; bf16 partials [pairs, n].
pub const Down = extern struct {
    x: u64,
    xs: u64,
    w2: u64,
    s2: u64,
    offs: u64,
    active: u64,
    nactive: u64,
    part: u64,
    n: c_int,
    k: c_int,
};

/// Mirrors RelCombine: grouped partials, pair_of, the shared expert's bf16 rows (optional) into y.
pub const Combine = extern struct {
    part: u64,
    pair_of: u64,
    shared: u64 = 0,
    shared_ld: i64 = 0,
    y: u64,
    y_ld: i64,
    y_col: i64 = 0,
    m: c_int,
    n: c_int,
    topk: c_int,
    f32: c_int = 0,
};

/// Mirrors RelSwiglu: the shared expert's bf16 gate and up rows to codes [m, n] and scales [m, n/32].
pub const Swiglu = extern struct {
    gate: u64,
    gate_ld: i64,
    up: u64,
    up_ld: i64,
    codes: u64,
    scales: u64,
    v: u64 = 0,
    m: c_int,
    n: c_int,
};

comptime {
    std.debug.assert(@sizeOf(Group) == 72 and @sizeOf(GateUp) == 176 and @sizeOf(Down) == 72);
    std.debug.assert(@sizeOf(Combine) == 72 and @sizeOf(Swiglu) == 64);
}

pub const group_threads = 1024;
/// The most slots a grouping takes (its shared counters: 4 bytes a slot).
pub const max_slots = 1024;
pub const group_smem = 4 * max_slots;
pub const tile_threads = 128;
pub const glue_threads = 256;
const most_ctas = 4096;

/// The scratch a grouping of m rows at top-k over nslots slots needs, in i32 words, by buffer.
pub const Scratch = struct {
    offs: usize,
    pairs: usize,
    active: usize,

    pub fn of(m: usize, topk: usize, nslots: usize) Scratch {
        return .{ .offs = nslots + 1, .pairs = m * topk, .active = nslots + 1 };
    }
};

/// A staged pair tile (release_moe_tile.cuh's ExCfg): pairs and threads a CTA, dynamic shared bytes.
pub const Tile = struct { pairs: u32, cols: u32, threads: u32, smem: u32 };
pub const Size = enum { decode, prompt };

/// nb matrices (2: gate/up, 1: down): 16 pairs x 32 columns (decode) or 64 x 64 (prompt), 4 stages of 4 K groups.
pub fn tile(nb: u32, size: Size) Tile {
    const bm: u32 = if (size == .decode) 16 else 64;
    const bn: u32 = if (size == .decode) 32 else 64;
    const kg: u32 = 4;
    const stage = (bm * 32 * kg + nb * bn * 16 * kg + bm * kg + nb * bn * kg + 127) & ~@as(u32, 127);
    return .{ .pairs = bm, .cols = bn, .threads = if (size == .decode) 32 else 128, .smem = 4 * stage + bm * 16 };
}

/// The staged tile a routed launch takes: 16 pairs while a held slot averages at most 16 pairs, else 64.
pub fn sizeOf(m: usize, topk: usize, nslots: usize) Size {
    return if (m * topk <= 16 * nslots) .decode else .prompt;
}

pub const Functions = struct {
    group: cuda.Function,
    /// [decode, prompt, warp]: the staged tiles and the stage-free warp form (bit-equal)
    gateup: [3]cuda.Function,
    down: [3]cuda.Function,
    combine: cuda.Function,
    swiglu: cuda.Function,

    pub fn resolve(m: cuda.Module) !Functions {
        const f: Functions = .{
            .group = try m.function("tf_dsv41_rel_group"),
            .gateup = .{ try m.function("tf_dsv41_rel_gateup"), try m.function("tf_dsv41_rel_gateup_prompt"), try m.function("tf_dsv41_rel_gateup_warp") },
            .down = .{ try m.function("tf_dsv41_rel_down"), try m.function("tf_dsv41_rel_down_prompt"), try m.function("tf_dsv41_rel_down_warp") },
            .combine = try m.function("tf_dsv41_rel_combine"),
            .swiglu = try m.function("tf_dsv41_rel_swiglu_actq"),
        };
        for ([_]Size{ .decode, .prompt }) |size| for ([_]u32{ 2, 1 }, [_]cuda.Function{ f.gateup[@backingInt(size)], f.down[@backingInt(size)] }) |nb, fun| {
            const t = tile(nb, size);
            if (t.smem > 48 * 1024) try fun.allowDynamicShared(t.smem);
        };
        return f;
    }
};

fn ctas(n: u64, per: u64) u32 {
    return @intCast(@max(1, @min((n + per - 1) / per, most_ctas)));
}

/// Warp tiles of a routed launch: every (active slot, 32-column block); active slots are at most min(nslots, pairs).
pub fn tileWarps(m: usize, topk: usize, nslots: usize, n: usize) u64 {
    return @min(nslots, m * topk) * (n / 32);
}

/// CTA tiles of a staged routed launch: every (active slot, column block of `cols`).
pub fn tileCtas(m: usize, topk: usize, nslots: usize, n: usize, cols: usize) u64 {
    return @min(nslots, m * topk) * ((n + cols - 1) / cols);
}

/// How a routed launch runs: the staged tiles at the served size, one CTA looping them, or the warp form.
pub const Launch = union(enum) { served, ctas: u32, warps: u32 };

pub const Ops = struct {
    f: *const Functions,
    s: cuda.Stream,

    pub fn group(o: Ops, p: Group) !void {
        if (p.m <= 0 or p.topk <= 0 or p.nslots <= 0 or p.nids <= 0 or p.nslots > max_slots) return error.Shape;
        var args: cuda.Args = .{};
        args.add(p);
        try cuda.launch.launch(o.f.group, .{ .grid = .{ .x = 1 }, .block = .{ .x = group_threads }, .shared = group_smem }, o.s, &args);
    }

    fn checkTiles(n: c_int, k: c_int) !void {
        if (n <= 0 or k <= 0 or @rem(n, 32) != 0 or @rem(k, 32) != 0) return error.Shape;
    }

    /// A routed launch of `funs` (decode, prompt, warp) over the tiles of m rows; `how` (tests) picks other grids.
    fn routed(o: Ops, funs: [3]cuda.Function, args: *cuda.Args, nb: u32, m: usize, topk: usize, nslots: usize, n: usize, how: Launch) !void {
        switch (how) {
            .warps => |grid| return cuda.launch.launch(funs[2], .{ .grid = .{ .x = grid }, .block = .{ .x = tile_threads } }, o.s, args),
            else => {},
        }
        const size = sizeOf(m, topk, nslots);
        const t = tile(nb, size);
        const grid: u32 = switch (how) {
            .ctas => |g| g,
            else => @intCast(@max(1, tileCtas(m, topk, nslots, n, t.cols))),
        };
        try cuda.launch.launch(funs[@backingInt(size)], .{ .grid = .{ .x = grid }, .block = .{ .x = t.threads }, .shared = t.smem }, o.s, args);
    }

    /// Gate/up of every grouped pair: the rows' codes 16-byte aligned (offset and stride), K and N whole groups.
    pub fn gateUp(o: Ops, p: GateUp, m: usize, nslots: usize, how: Launch) !void {
        try checkGateUp(p);
        var args: cuda.Args = .{};
        args.add(p);
        try o.routed(o.f.gateup, &args, 2, m, @intCast(p.topk), nslots, @intCast(p.n), how);
    }

    pub fn down(o: Ops, p: Down, m: usize, topk: usize, nslots: usize, how: Launch) !void {
        try checkDown(p);
        var args: cuda.Args = .{};
        args.add(p);
        try o.routed(o.f.down, &args, 1, m, topk, nslots, @intCast(p.n), how);
    }

    pub fn combine(o: Ops, p: Combine) !void {
        try checkCombine(p);
        var args: cuda.Args = .{};
        args.add(p);
        const total = @as(u64, @intCast(p.m)) * @as(u64, @intCast(p.n));
        try cuda.launch.launch(o.f.combine, .{ .grid = .{ .x = ctas(total, glue_threads) }, .block = .{ .x = glue_threads } }, o.s, &args);
    }

    pub fn swigluActq(o: Ops, p: Swiglu) !void {
        if (p.m <= 0 or p.n <= 0 or @rem(p.n, 32) != 0 or p.gate_ld < p.n or p.up_ld < p.n) return error.Shape;
        var args: cuda.Args = .{};
        args.add(p);
        const blocks = @as(u64, @intCast(p.m)) * @as(u64, @intCast(@divExact(p.n, 32)));
        try cuda.launch.launch(o.f.swiglu, .{ .grid = .{ .x = ctas(blocks, 8) }, .block = .{ .x = glue_threads } }, o.s, &args);
    }
};

/// Gate/up's shape checks: whole groups, 16-byte code rows and weight rows (the stages read 16-byte chunks).
pub fn checkGateUp(p: GateUp) !void {
    try Ops.checkTiles(p.n, p.k);
    if (p.x_ld < p.k or @rem(p.x_ld, 16) != 0 or (p.x + p.x_off) % 16 != 0 or p.xs_ld < @divExact(p.k, 32)) return error.Shape;
    if (p.w1 % 16 != 0 or p.w3 % 16 != 0 or p.w1 == 0 or p.w3 == 0 or p.s1 == 0 or p.s3 == 0) return error.Shape;
}

/// Down's: whole groups, 16-byte code rows and weight rows.
pub fn checkDown(p: Down) !void {
    try Ops.checkTiles(p.n, p.k);
    if (p.x % 16 != 0 or p.w2 % 16 != 0 or p.w2 == 0 or p.s2 == 0) return error.Shape;
}

/// Combine's: rows and columns, an output column offset at or past 0 inside its row stride.
pub fn checkCombine(p: Combine) !void {
    if (p.m <= 0 or p.n <= 0 or p.topk <= 0 or p.y == 0 or p.y_col < 0 or p.y_ld < p.y_col + p.n) return error.Shape;
    if (p.shared != 0 and p.shared_ld < p.n) return error.Shape;
}

test "release experts: refusals of misaligned weights and code rows, and of a combine column offset outside y" {
    const gu: GateUp = .{ .x = 256, .x_ld = 5120, .xs = 8, .xs_ld = 160, .w1 = 512, .s1 = 1, .w3 = 1024, .s3 = 1, .wts = 4, .pair_tok = 4, .offs = 4, .active = 4, .nactive = 4, .codes = 4, .scales = 4, .n = 576, .k = 5120, .topk = 6 };
    try checkGateUp(gu);
    var g = gu;
    g.w1 = 520;
    try std.testing.expectError(error.Shape, checkGateUp(g));
    g = gu;
    g.w3 = 1032;
    try std.testing.expectError(error.Shape, checkGateUp(g));
    g = gu;
    g.x_off = 8;
    try std.testing.expectError(error.Shape, checkGateUp(g));
    const dn: Down = .{ .x = 256, .xs = 1, .w2 = 512, .s2 = 1, .offs = 4, .active = 4, .nactive = 4, .part = 4, .n = 5120, .k = 576 };
    try checkDown(dn);
    var d = dn;
    d.w2 = 520;
    try std.testing.expectError(error.Shape, checkDown(d));
    d = dn;
    d.x = 264;
    try std.testing.expectError(error.Shape, checkDown(d));
    const cb: Combine = .{ .part = 2, .pair_of = 4, .y = 64, .y_ld = 5120, .m = 3, .n = 5120, .topk = 6 };
    try checkCombine(cb);
    var c = cb;
    c.y_col = -1;
    c.y_ld = 5200;
    try std.testing.expectError(error.Shape, checkCombine(c));
    c = cb;
    c.y_col = 8;
    try std.testing.expectError(error.Shape, checkCombine(c));
    c.y_ld = 5128;
    try checkCombine(c);
}

test "release experts: the warp tiles cover every active slot's column blocks; shapes refused" {
    try std.testing.expectEqual(@as(u64, 6 * 18), tileWarps(1, 6, 384, 576));
    try std.testing.expectEqual(@as(u64, 384 * 160), tileWarps(2048, 6, 384, 5120));
    try std.testing.expectEqual(@as(u64, 3 * 18), tileWarps(1, 3, 128, 576));
    try std.testing.expectError(error.Shape, Ops.checkTiles(570, 5120));
    try std.testing.expectEqual(@as(usize, 2048 * 6), Scratch.of(2048, 6, 384).pairs);
    try std.testing.expectEqual(@as(u32, 26368), tile(2, .decode).smem);
    try std.testing.expectEqual(@as(u32, 69632), tile(2, .prompt).smem);
    try std.testing.expectEqual(@as(u32, 17664), tile(1, .decode).smem);
    try std.testing.expectEqual(@as(u32, 52224), tile(1, .prompt).smem);
    try std.testing.expectEqual(Size.decode, sizeOf(16, 6, 384));
    try std.testing.expectEqual(Size.prompt, sizeOf(2048, 6, 384));
}
