//! Same-input lane projections in one qmmf launch, each keeping its own tile, K slices and cluster sum (qmmf.cu).
const std = @import("std");
const q = @import("qmmf.zig");
const launch_ = @import("launch.zig");
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;

/// Projections a launch takes, and the decode rows (one 16-row tile) it serves.
pub const most = 4;
pub const max_rows = 16;

/// qmmf.cu GroupSeg: one projection's weights, output, scale, shape, K slices and first block.
pub const Seg = extern struct { w: u64 = 0, bs: u64 = 0, out: u64 = 0, scale: f32 = 0, n: i32 = 0, sk: i32 = 1, npad: i32 = 0, first: i32 = 0, tiles: i32 = 0 };
pub const Segs = extern struct { s: [most]Seg = @splat(.{}), count: i32 = 0, pad: i32 = 0 };

comptime {
    std.debug.assert(@sizeOf(Seg) == 48 and @sizeOf(Segs) == 200);
}

/// The instance a group takes: its mode and column tile (64, or the 32-column half tile).
pub const Kind = enum { fp4, fp8, bf16, bf16_half };
const kinds = [4]Kind{ .fp4, .fp8, .bf16, .bf16_half };

const symbols = [4][2][:0]const u8{
    .{ "tf_qmmf_group_fp4", "tf_qmmf_group_fp4_c" },
    .{ "tf_qmmf_group_fp8", "tf_qmmf_group_fp8_c" },
    .{ "tf_qmmf_group_bf16", "tf_qmmf_group_bf16_c" },
    .{ "tf_qmmf_group_bf16_half", "tf_qmmf_group_bf16_half_c" },
};

fn shared(kind: Kind) u32 {
    return switch (kind) {
        .fp4 => q.sharedBytes(.fp4, 16),
        .fp8 => q.sharedBytes(.fp8, 16),
        .bf16 => q.sharedBytes(.bf16, 16),
        .bf16_half => q.halfShared(.bf16),
    };
}

/// One grouped launch: its instance, cluster width (the widest K split), blocks and projections.
pub const Shape = struct { kind: Kind, cluster: u32, blocks: u32, segs: Segs };

/// The grouped launch of `ws` into `outs` over `m` rows, or null where any projection's own launch differs in form.
pub fn plan(l: q.Lane, ws: []const q.Weight, outs: []const u64, m: usize) ?Shape {
    if (ws.len < 2 or ws.len > most or ws.len != outs.len or m == 0 or m > max_rows) return null;
    if (!l.clusters or l.fuse_split_n != 0) return null; // pre-sm_90 slices meet in reduce_kernel; fused calls differ
    const mode = ws[0].mode;
    if (mode != .fp4 and mode != .fp8 and mode != .bf16) return null;
    var half: ?bool = null;
    var widest: u32 = 1;
    for (ws, 0..) |w, i| {
        if (w.mode != mode or w.k != ws[0].k or w.k % q.group != 0 or w.n == 0) return null;
        for (outs[i + 1 ..]) |o| if (o == outs[i]) return null;
        const p = q.planFor(m, w, l.clusters);
        if (p.bm != 16 or p.fused or p.reduce or p.cluster != (p.sk > 1) or l.usesPrompt(m, w)) return null;
        // matmulOut's own choice: half tiles for small BF16 grids
        const h = q.halfModeIndex(w.mode) != null and ((w.n + 63) / 64) * p.sk < l.half_below;
        if (half) |prev| if (prev != h) return null;
        half = h;
        widest = @max(widest, p.sk);
    }
    const cols: u32 = if (half.?) 32 else 64;
    var segs: Segs = .{ .count = @intCast(ws.len) };
    var first: u32 = 0;
    for (ws, outs, 0..) |w, o, i| {
        const sk = q.slices(w);
        const tiles = (w.n + cols - 1) / cols;
        segs.s[i] = .{ .w = w.codes, .bs = w.scales, .out = o, .scale = w.scale, .n = @intCast(w.n), .sk = @intCast(sk), .npad = @intCast(w.npad), .first = @intCast(first), .tiles = @intCast(tiles) };
        first += (tiles * sk + widest - 1) / widest * widest; // a cluster never spans two projections
    }
    const kind: Kind = switch (mode) {
        .fp4 => .fp4,
        .fp8 => .fp8,
        else => if (half.?) .bf16_half else .bf16,
    };
    return .{ .kind = kind, .cluster = widest, .blocks = first, .segs = segs };
}

/// The grouped instances, from the lane's module.
pub const Group = struct {
    fns: [4][2]Function,

    pub fn load(l: *const q.Lane) !Group {
        var g: Group = undefined;
        for (symbols, &g.fns, kinds) |pair, *fs, kind| for (pair, fs) |name, *f| {
            f.* = try l.mod.function(name);
            try f.allowDynamicShared(shared(kind));
        };
        return g;
    }

    /// The planned launch over x [m, k] bf16 (rows k apart); `pdl` as the lane's.
    pub fn launch(g: Group, s: Stream, x: u64, m: usize, k: usize, sh: Shape, pdl: bool) !void {
        var args: launch_.Args = .{};
        args.add(x);
        args.add(sh.segs);
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(k)));
        try launch_.launch(g.fns[std.mem.indexOfScalar(Kind, &kinds, sh.kind).?][@intFromBool(sh.cluster > 1)], .{
            .grid = .{ .x = sh.blocks },
            .block = .{ .x = 128 },
            .shared = shared(sh.kind),
            .cluster = if (sh.cluster > 1) .{ .x = sh.cluster, .y = 1, .z = 1 } else null,
            .pdl = pdl,
        }, s, &args);
    }
};

fn testLane(clusters: bool) q.Lane {
    return .{ .mod = undefined, .fns = undefined, .f32_fns = undefined, .reduce = undefined, .half = undefined, .clusters = clusters };
}

fn weight(mode: q.Mode, n: u32, k: u32) q.Weight {
    return .{ .mode = mode, .codes = 0x1000 * @as(u64, n), .scales = 0, .scale = 0.5, .n = n, .k = k, .npad = (n + 127) / 128 * 128 };
}

test "grouped projections: the served Qwen3.8-27B groups keep each projection's slices, tiles and blocks" {
    const t = std.testing;
    const l = testLane(true);
    const outs = [_]u64{ 0x10, 0x20, 0x30, 0x40 };
    for (1..max_rows + 1) |m| {
        // MLP gate + up, NVFP4: one slice each, 272 + 272 blocks, no cluster
        const mlp = plan(l, &.{ weight(.fp4, 17408, 5120), weight(.fp4, 17408, 5120) }, outs[0..2], m).?;
        try t.expect(mlp.kind == .fp4 and mlp.cluster == 1 and mlp.blocks == 544 and mlp.segs.s[1].first == 272);
        // GDN qkv + z, FP8: two slices each, 320 + 192 blocks in clusters of 2
        const gdn = plan(l, &.{ weight(.fp8, 10240, 5120), weight(.fp8, 6144, 5120) }, outs[0..2], m).?;
        try t.expect(gdn.kind == .fp8 and gdn.cluster == 2 and gdn.blocks == 512 and gdn.segs.s[1].sk == 2);
        // GDN a + b, BF16 heads: half tiles as their own launches, 8 slices, 16 + 16 blocks
        const ab = plan(l, &.{ weight(.bf16, 48, 5120), weight(.bf16, 48, 5120) }, outs[0..2], m).?;
        try t.expect(ab.kind == .bf16_half and ab.cluster == 8 and ab.blocks == 32 and ab.segs.s[0].tiles == 2);
        // attention q + k + v, FP8: q one slice (192 blocks), k and v 8 slices (128 each); clusters of 8 by projection
        const qkv = plan(l, &.{ weight(.fp8, 12288, 5120), weight(.fp8, 1024, 5120), weight(.fp8, 1024, 5120) }, outs[0..3], m).?;
        try t.expect(qkv.cluster == 8 and qkv.blocks == 448 and qkv.segs.count == 3);
        try t.expectEqual(@as(i32, 1), qkv.segs.s[0].sk);
        try t.expectEqual(@as(i32, 8), qkv.segs.s[2].sk);
        try t.expectEqual(@as(i32, 320), qkv.segs.s[2].first);
    }
    // 193 one-slice tiles round up to 200 blocks so the 8-slice projection's clusters start on a cluster
    const pad = plan(l, &.{ weight(.fp4, 12352, 4096), weight(.fp4, 200, 4096) }, outs[0..2], 3).?;
    try t.expect(pad.cluster == 8 and pad.segs.s[0].tiles == 193 and pad.segs.s[1].first == 200 and pad.blocks == 232);
}

test "grouped projections refuse what their own launches would run another way" {
    const t = std.testing;
    const l = testLane(true);
    const outs = [_]u64{ 0x10, 0x20, 0x30, 0x40, 0x50 };
    const gate = weight(.fp4, 17408, 5120);
    try t.expect(plan(l, &.{ gate, gate }, outs[0..2], 17) == null); // a 32-row tile
    try t.expect(plan(l, &.{ gate, gate }, outs[0..2], 0) == null);
    try t.expect(plan(l, &.{gate}, outs[0..1], 1) == null);
    try t.expect(plan(l, &.{ gate, gate, gate, gate, gate }, outs[0..5], 1) == null);
    try t.expect(plan(l, &.{ gate, gate }, &.{ 0x10, 0x10 }, 1) == null); // one output twice
    try t.expect(plan(l, &.{ gate, weight(.fp8, 17408, 5120) }, outs[0..2], 1) == null); // two modes
    try t.expect(plan(l, &.{ gate, weight(.fp4, 17408, 4096) }, outs[0..2], 1) == null); // two inputs' widths
    try t.expect(plan(l, &.{ weight(.fp8g, 1024, 5120), weight(.fp8g, 1024, 5120) }, outs[0..2], 1) == null);
    try t.expect(plan(testLane(false), &.{ gate, gate }, outs[0..2], 1) == null); // slices meet in reduce_kernel
    var fused = testLane(true);
    fused.fuse_split_n = 1;
    try t.expect(plan(fused, &.{ gate, gate }, outs[0..2], 1) == null);
    // BF16: a half-tile projection and a 64-column one are two instances
    try t.expect(plan(l, &.{ weight(.bf16, 48, 5120), weight(.bf16, 12288, 5120) }, outs[0..2], 1) == null);
    var no_half = testLane(true);
    no_half.half_below = 0;
    try t.expect(plan(no_half, &.{ weight(.bf16, 48, 5120), weight(.bf16, 48, 5120) }, outs[0..2], 1).?.kind == .bf16);
}
