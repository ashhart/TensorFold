//! The MLX affine tiles of tiles/dot2_tiles.hip and tiles/dot2.hip launched from Zig; the registry picks one.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const Stream = @import("../stream.zig").Stream;
const util = @import("util.zig");

const Caps = @import("../caps.zig").Caps;
const registry = @import("core").registry;
const quant = @import("core").quant;
const tuning = @import("../tuning/tuning.zig");
const mlx_entries = @import("mlx_entries.zig");
const affine_stream = @import("affine_stream.zig");
const affine_load = @import("affine_load.zig");

const Error = runtime.Error;

/// Matrix cores for products: `off` keeps every product on dot2 tiles; `auto` and `on` use them where a tile exists.
pub const Matrix = enum { auto, on, off };

/// Which product tiles a run takes where two give the same bits; every one defaults on.
pub const Choices = struct { matrix: Matrix = .auto, stream: bool = true, gemm: bool = true };

/// quant/mlx.hpp's GroupTable, Routing and Affine as the kernels take them by value.
pub const GroupTable = extern struct { p: u64, kind: c_int };
pub const Routing = extern struct { items: u64 = 0, members: u64 = 0, x_div: c_int = 1, first: c_int = 0 };
pub const Arg = extern struct {
    x: u64,
    words: u64,
    scale: GroupTable,
    bias: GroupTable,
    out: u64,
    m: c_int,
    n: c_int,
    k: c_int,
    bits: c_int,
    group: c_int,
    fp16: c_int,
    route: Routing = .{},
    out16: u64 = 0,
};

/// quant/mlx_tiles.hpp's StreamSides: up to four products that share x, K, width and group in one launch.
pub const StreamSides = extern struct {
    words: [4]u64 = @splat(0),
    scale: [4]u64 = @splat(0),
    bias: [4]u64 = @splat(0),
    out: [4]u64 = @splat(0),
    n: [4]c_int = @splat(0),
    first: [5]c_int = @splat(0),
    count: c_int = 0,
    out_half: c_int = 0,
    pair_cols: c_int = 0,
    limit: f32 = 0,
    /// > 1: the launch's row blocks are the grid's x and its column blocks its y (stream.hpp stream_tile)
    row_blocks: c_int = 0,
};

/// One side of a group: its matrix and where its (rows, n) output goes.
pub const Side = struct { words: u64, scale: u64, bias: u64, n: c_int, out: u64 };

comptime {
    std.debug.assert(@sizeOf(GroupTable) == 16 and @sizeOf(Routing) == 24 and @sizeOf(Arg) == 112 and @sizeOf(StreamSides) == 184);
}

/// Everything a launch needs beside the kernels.
pub const Call = struct {
    r: *const runtime.Runtime,
    s: abi.Stream,
    arg: Arg,
    items: c_int = 1,
    sides: []const Side = &.{},
    limit: f32 = 0,
    out_half: bool = false,
    partial: u64 = 0,
    parts: c_int = 1,
};

/// A kernel of the registry: its launch on this backend's kernels.
pub const Launch = *const fn (*const Kernels, Call) Error!void;
pub const Registry = registry.Registry(Launch);
pub const Entry = Registry.E;

pub const bit_widths = [_]c_int{ 2, 3, 4, 5, 6, 8 };
pub const row_counts = [_]u8{ 1, 2, 4, 8 };
pub const piece_counts = [_]u8{ 1, 2, 4 };

pub const lane_rows = 8; // kLaneRows
pub const lane_group_max = 128;
const fast_n = 256;
pub const stream_cbs = [_]c_int{ 1, 2, 4, 8 }; // columns a lane carries in the stream tile (1: plain products only)
pub const stream_rows = [_]u8{ 1, 2, 4 }; // its row counts: columns * rows <= 16
pub const stream_waves = 4; // kStreamWaves
pub const streams_wanted = 3000; // kStreamWavesWanted
pub const stream_code_words = 32; // kStreamCodeWords
pub const stream_pair_code_words = affine_stream.stream_pair_code_words;
pub const stream_routed_cb_top = affine_stream.stream_routed_cb_top;
pub const stream_one_col_n = affine_stream.stream_one_col_n;

/// The K-parallel tiles of a short prompt: cb columns a lane set, r rows a pass over up to rb row blocks.
pub const KpTile = struct { cb: c_int, r: c_int, waves: c_int, rb: c_int, loop: bool = true };
pub const kp_tiles = [_]KpTile{
    .{ .cb = 8, .r = 1, .waves = 2, .rb = 2, .loop = false },
    .{ .cb = 4, .r = 4, .waves = 2, .rb = 4 },
    .{ .cb = 4, .r = 8, .waves = 2, .rb = 4, .loop = false },
    .{ .cb = 4, .r = 2, .waves = 2, .rb = 4 },
};

/// The GEMM tiles of the m >= 64 products: `gemm` (the default) or the reference `block`.
pub const Tile = enum { gemm, block };

/// Which GEMM tile of the 128-row family a launch runs: the matrix cores, the dot2 tile, or the reference tile.
pub const BlockKind = enum { matrix, gemm, block };

pub const Kernels = struct {
    /// Whether block shape `tier` can take `m` rows: a tile that does not loop over its rows holds rb * r of them.
    pub fn tierTakes(tier: usize, m: c_int) bool {
        return tier >= kp_tiles.len or kp_tiles[tier].loop or m <= kp_tiles[tier].rb * kp_tiles[tier].r;
    }

    caps: Caps,
    wmma: bool,
    /// gfx11: the matrix-core GEMM tile runs (gfx12's WMMA has other layouts).
    matrix: bool,
    /// `off` runs products on dot2 tiles, `on` and `auto` on matrix cores where a tile exists.
    mode: Matrix,
    /// Which GEMM tile the products take; the policy's reference switch picks the reference one.
    tile: Tile,
    lanes: [bit_widths.len][row_counts.len][piece_counts.len]Function,
    row: [bit_widths.len][piece_counts.len]Function,
    /// The stream tile (tiles/stream.hpp) by width, rows and columns a lane; null where columns * rows > 16.
    stream: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// The same tile with the (gate | up) activation as its epilogue.
    pairs: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// The same tile with Nemotron's relu^2 as its epilogue, and with two such products in one launch.
    relu2: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    relu2_pair: [bit_widths.len][stream_rows.len][stream_cbs.len]?Function,
    /// Off keeps the decode tiles before the stream tile (the policy's reference switch).
    stream_on: bool,
    /// A test's forced columns a lane (a stream_cbs value; 0: the rule), where the shape takes it.
    stream_cb_force: c_int,
    block: [bit_widths.len]Function,
    gemm: [bit_widths.len]Function,
    wmma_gemm: [bit_widths.len]Function,
    kp: [bit_widths.len][kp_tiles.len]Function,
    span: [2]Function,
    fold: Function,
    wide: Function,
    tiled: Function,
    reference: Function,
    fill: Function,
    reg: Registry,

    pub const load = affine_load.load;
    pub const fillByteLut = affine_load.fillByteLut;

    /// Launches on `s`; the counting stream runs nothing.
    pub fn go(r: *const runtime.Runtime, f: Function, grid: hl.Dim3, block: hl.Dim3, s: abi.Stream, args: *hl.Args) Error!void {
        if (s == util.counting) return;
        try hl.launch(f, .{ .grid = grid, .block = block }, .{ .r = r, .handle = s }, args);
    }

    pub fn refuse(what: []const u8) Error {
        std.log.err("affine: {s}", .{what});
        return error.Invalid;
    }

    pub fn bitIndex(bits: c_int) ?usize {
        return std.mem.indexOfScalar(c_int, &bit_widths, bits);
    }

    fn pieceIndex(group: c_int) usize {
        return switch (group) {
            32 => 0,
            64 => 1,
            else => 2,
        };
    }

    pub fn cdiv(n: c_int, by: c_int) c_uint {
        return @intCast(@divTrunc(n + by - 1, by));
    }

    /// Groups a lane covers: the groups rounded up to a power of two, at most 32; none under four groups.
    pub fn rowLanes(groups: c_int) c_int {
        if (groups < 4) return 0;
        var lpc: c_int = 4;
        while (lpc < groups and lpc < 32) lpc <<= 1;
        return lpc;
    }

    /// The lanes a column takes in the K-parallel tile: one a group, 32 at most (kp_lanes).
    fn kpLanes(groups: c_int) c_int {
        return if (groups > 16) 32 else if (groups > 8) 16 else 8;
    }

    /// The bytes the GEMM tile's piece loads need the words aligned to, by width.
    pub fn gemmAlign(bits: c_int) u64 {
        return switch (bits) {
            4, 8 => 16,
            2, 6 => 8,
            else => 4,
        };
    }

    // ---- what the choice sees of this build and of a call ----

    /// The GPU and the run's switches as the registry reads them.
    pub fn env(k: *const Kernels) registry.Env {
        return .{ .bf16 = k.wmma, .matrix = k.matrix, .matrix_on = k.matrix and k.mode != .off, .stream_on = k.stream_on, .gemm_on = k.tile == .gemm };
    }

    /// The MLX entry that takes a product, or null.
    pub fn choose(k: *const Kernels, op: registry.Op, path: registry.Path, shape: registry.Shape) ?*const Entry {
        return k.reg.select(k.env(), .mlx, op, path, shape);
    }

    /// The costs of the GPU's generation: RDNA2's, the gfx11 ones for RDNA3 and RDNA3.5, RDNA4's own.
    pub fn tableFor(caps: Caps) *const tuning.Table {
        return switch (caps.generation) {
            .rdna2 => &tuning.gfx1030,
            .rdna3, .rdna3_5 => &tuning.gfx1100,
            .rdna4 => &tuning.gfx1201,
        };
    }

    /// The product of `a` as a shape: `items` is a routed plan's item count.
    pub fn shapeOf(a: Arg, items: c_int) registry.Shape {
        const kind: quant.Tables = switch (a.scale.kind) {
            1 => .bf16,
            2 => .f16,
            else => .f32,
        };
        return .{
            .m = @intCast(@max(a.m, 0)),
            .n = @intCast(@max(a.n, 0)),
            .k = @intCast(@max(a.k, 0)),
            .bits = std.math.cast(u8, a.bits) orelse 0,
            .group = std.math.cast(u16, a.group) orelse 0,
            .fp16 = a.fp16 != 0,
            .tables = kind,
            .tables_alike = a.scale.kind == a.bias.kind,
            .x_aligned = a.x % 16 == 0,
            .words_aligned = a.words % gemmAlign(a.bits) == 0,
            .items = if (a.route.items != 0) @intCast(@max(items, 0)) else 0,
        };
    }

    // ---- the tiles ----

    /// The stream tile on `total_n` columns (a group's sum); `sides` counts them off block by block when it has any.
    pub fn streamGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, sides_in: StreamSides, total_n: c_int, items: c_int, s: abi.Stream) Error!void {
        return affine_stream.streamWith(k, r, a, sides_in, total_n, items, s, false);
    }

    /// `streamGo` with Nemotron's relu^2 as the epilogue: `a.out16` takes relu(round(sum))^2 in the activation type.
    pub fn relu2Go(k: *const Kernels, r: *const runtime.Runtime, a: Arg, items: c_int, s: abi.Stream) Error!void {
        if (a.out16 == 0 or a.out != 0) return refuse("relu2 output");
        return affine_stream.streamWith(k, r, a, .{}, a.n, items, s, true);
    }

    /// One row on the row tile: a lane owns a group, a wave reads a column's row in one run.
    pub fn rowGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(rowLanes(@divExact(a.k, a.group)));
        try go(r, k.row[b][pieceIndex(a.group)], .{ .x = cdiv(a.n, 128), .z = @intCast(items) }, .{ .x = 128 }, s, &args);
    }

    /// Up to 8 rows on the column tile.
    pub fn lanesGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        try args.add(a);
        const ri: usize = if (a.m == 1) 0 else if (a.m == 2) 1 else if (a.m <= 4) 2 else 3;
        try go(r, k.lanes[b][ri][pieceIndex(a.group)], .{ .x = cdiv(a.n, 32), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    /// The wide tile of the RDNA2 decode products of 9 rows and more.
    pub fn wideGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, k.wide, .{ .x = cdiv(a.n, 32), .y = cdiv(a.m, 128) }, .{ .x = 512 }, s, &args);
    }

    /// Any group width up to 128 that the others do not take, 16 columns by 16 rows a block.
    pub fn tiledGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, k.tiled, .{ .x = cdiv(a.n, 16), .y = cdiv(a.m, 16) }, .{ .x = 256 }, s, &args);
    }

    /// The one-thread tile every product has: the reference of the others.
    pub fn referenceGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream) Error!void {
        var args: hl.Args = .{};
        try args.add(a);
        try go(r, k.reference, .{ .x = cdiv(a.n, 32), .y = @intCast(a.m) }, .{ .x = 32 }, s, &args);
    }

    /// The decode tile over groups [z * per, (z + 1) * per) into `partial`, then the fold in group order.
    pub fn splitGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, partial: u64, parts: c_int, s: abi.Stream) Error!void {
        const groups = @divTrunc(a.k, a.group);
        var args: hl.Args = .{};
        try args.add(a);
        try args.add(partial);
        try args.add(@divExact(groups, parts));
        try go(r, k.span[if (a.m == 1) 0 else 1], .{ .x = cdiv(a.n, fast_n), .y = cdiv(a.m, 8), .z = @intCast(parts) }, .{ .x = fast_n }, s, &args);
        var fold: hl.Args = .{};
        try fold.add(partial);
        try fold.add(a.scale);
        try fold.add(a.bias);
        try fold.add(a.out);
        try fold.add(a.m);
        try fold.add(a.n);
        try fold.add(groups);
        try go(r, k.fold, .{ .x = cdiv(a.n, fast_n), .y = @intCast(a.m) }, .{ .x = fast_n }, s, &fold);
    }

    /// The 128 x 128 tile of a kind over every item of a plan (or the plain product).
    pub fn blockGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int, kind: BlockKind) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        try args.add(a);
        const f = switch (kind) {
            .matrix => k.wmma_gemm[b],
            .gemm => k.gemm[b],
            .block => k.block[b],
        };
        try go(r, f, .{ .x = cdiv(a.n, 128), .y = cdiv(a.m, 128), .z = @intCast(items) }, .{ .x = 256 }, s, &args);
    }

    /// K-parallel tile `tier` (an index of kp_tiles) of a short prompt.
    pub fn kpGo(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int, tier: usize) Error!void {
        const b = bitIndex(a.bits) orelse return refuse("bits");
        var args: hl.Args = .{};
        try args.add(a);
        const t = kp_tiles[tier];
        const sets = @divExact(32, kpLanes(@divExact(a.k, a.group)));
        const blocks = @min(cdiv(a.m, t.r), @as(c_uint, @intCast(t.rb)));
        try go(r, k.kp[b][tier], .{ .x = cdiv(a.n, t.waves * t.cb * sets) * blocks, .z = @intCast(items) }, .{ .x = @intCast(32 * t.waves) }, s, &args);
    }

    // ---- the entry points ----

    /// Prefill's tile, same bits however a prompt is cut: matrix on gfx11 unless off, else dot2 GEMM; few rows K-split.
    pub fn prefillLaunch(k: *const Kernels, r: *const runtime.Runtime, a: Arg, s: abi.Stream, items: c_int, pairs: c_int) Error!void {
        const op: registry.Op = if (a.route.items != 0) .routed else .project;
        var shape = shapeOf(a, items);
        shape.plan_pairs = @intCast(@max(pairs, 0));
        const e = k.choose(op, .prefill, shape) orelse return refuse("prefill shape");
        try e.launch(k, .{ .r = r, .s = s, .arg = a, .items = items });
    }

    /// launch_affine on the auto schedule: `schedule` 3 is prefill's, 4 a lane round's; `partial`, `parts` the tests'.
    pub fn run(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, schedule: c_int, s: abi.Stream, partial: u64, parts: c_int, out_half: bool) Error!void {
        var a = arg;
        if (schedule == 1) {
            if (a.fp16 == 0 or k.wmma) return refuse("the reference schedule takes fp16 on RDNA2");
            return k.referenceGo(r, a, s);
        }
        if (schedule == 2 or schedule > 4) return refuse("no such schedule");
        var shape = shapeOf(a, 1);
        shape.round = schedule == 4;
        shape.parts = if (partial != 0 and parts > 1) @intCast(parts) else 1;
        const e = k.choose(.project, if (schedule == 3) .prefill else .decode, shape) orelse return refuse("no tile takes this product");
        if (out_half) {
            // the activation type's output is the decode tile's: fp16 on RDNA2, and either type from the stream tile
            if (schedule == 3 or parts > 1 or !e.roundsAct(shape)) return refuse("fp16 output is the decode tile");
            a.out16 = a.out;
            a.out = 0;
        }
        try e.launch(k, .{ .r = r, .s = s, .arg = a, .partial = partial, .parts = parts, .out_half = out_half });
    }

    /// affine_routed_launch: every item of a plan in one launch; `arg.m` is the most rows an item holds.
    pub fn routed(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, items: c_int, s: abi.Stream) Error!void {
        return k.routedWith(r, arg, items, s, k.tile);
    }

    pub fn routedWith(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, items: c_int, s: abi.Stream, tile: Tile) Error!void {
        // the family's own activation type: bf16 on the WMMA build, fp16 on RDNA2
        if ((arg.fp16 != 0) == k.wmma or arg.route.items == 0 or arg.route.members == 0 or items < 1 or arg.route.x_div < 1) return refuse("routed plan");
        var environment = k.env();
        environment.gemm_on = tile == .gemm;
        const e = k.reg.select(environment, .mlx, .routed, .decode, shapeOf(arg, items)) orelse return refuse("no tile takes this plan");
        try e.launch(k, .{ .r = r, .s = s, .arg = arg, .items = items });
    }

    /// `routed` with relu(round(y))^2 into `arg.out16` as the stream tile's epilogue; false where another tile runs.
    pub fn routedRelu2(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, items: c_int, s: abi.Stream) Error!bool {
        if ((arg.fp16 != 0) == k.wmma or items < 1) return false;
        if (arg.route.items == 0 or arg.route.members == 0 or arg.route.x_div < 1) return false;
        var plain = arg;
        plain.out16 = 0;
        const e = k.reg.select(k.env(), .mlx, .routed, .decode, shapeOf(plain, items)) orelse return false;
        if (e.family != .stream) return false;
        try k.relu2Go(r, arg, items, s);
        return true;
    }

    /// The stacked (gate | up) product as silu(gate) * up into `out16`, clamped by `limit` above 0; false if not taken.
    pub fn pairRun(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, limit: f32, items: c_int, s: abi.Stream) Error!bool {
        var shape = shapeOf(arg, items);
        shape.pairs = arg.out16 != 0;
        const e = k.choose(.routed_act, .decode, shape) orelse return false;
        try e.launch(k, .{ .r = r, .s = s, .arg = arg, .items = items, .limit = limit });
        return true;
    }

    /// Up to four products of `m` rows over one x in one launch; outputs (m, n) fp32, or activation type if `out_half`.
    pub fn groupRun(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, group: []const Side, out_half: bool, s: abi.Stream) Error!void {
        if (group.len == 0 or group.len > 4) return refuse("group shape");
        var a = arg;
        a.n = 0;
        for (group) |side| a.n = @max(a.n, side.n);
        var shape = shapeOf(a, 1);
        shape.round = true;
        const e = k.choose(.group, .decode, shape) orelse return refuse("group shape");
        try e.launch(k, .{ .r = r, .s = s, .arg = a, .sides = group, .out_half = out_half });
    }

    /// The stream tile over a group: the sides' addresses and widths in one launch.
    pub fn groupGo(k: *const Kernels, r: *const runtime.Runtime, arg: Arg, group: []const Side, out_half: bool, s: abi.Stream) Error!void {
        var a = arg;
        var sides: StreamSides = .{ .count = @intCast(group.len), .out_half = @intFromBool(out_half) };
        var total: c_int = 0;
        a.n = 0;
        for (group, 0..) |side, i| {
            sides.words[i] = side.words;
            sides.scale[i] = side.scale;
            sides.bias[i] = side.bias;
            sides.out[i] = side.out;
            sides.n[i] = side.n;
            total += side.n;
            a.n = @max(a.n, side.n);
        }
        a.words = group[0].words;
        a.scale.p = group[0].scale;
        a.bias.p = group[0].bias;
        try k.streamGo(r, a, sides, total, 1, s);
    }

    /// The split count of a decode launch: 1 unless `mode` is 2 (the tests' forced comparison) and the groups divide.
    pub fn splitCount(m: c_int, n: c_int, k: c_int, group: c_int, mode: c_int) c_int {
        if (mode != 2 or m < 1 or n < 512) return 1;
        if (!(group == 32 or group == 64 or group == 128) or k < group or @rem(k, group) != 0) return 1;
        const groups = @divTrunc(k, group);
        if (groups < 2) return 1;
        // A full row tile or a wide column grid already fills the card.
        const blocks = @divTrunc(n + fast_n - 1, fast_n) * @divTrunc(m + 8 - 1, 8);
        if (blocks < 1) return 1;
        var want = @divTrunc(256, blocks);
        if (want < 2) want = 2;
        if (want > groups) want = groups;
        var count = want;
        while (count >= 2) : (count -= 1) {
            if (@rem(groups, count) == 0) return count;
        }
        return 1;
    }
};

test "the split count follows the C rule" {
    try std.testing.expectEqual(@as(c_int, 1), Kernels.splitCount(1, 4096, 4096, 64, 0));
    try std.testing.expectEqual(@as(c_int, 1), Kernels.splitCount(1, 256, 4096, 64, 2));
    try std.testing.expectEqual(@as(c_int, 8), Kernels.splitCount(1, 4096, 512, 64, 2));
}

test "the lane groups of a row tile" {
    try std.testing.expectEqual(@as(c_int, 0), Kernels.rowLanes(3));
    try std.testing.expectEqual(@as(c_int, 4), Kernels.rowLanes(4));
    try std.testing.expectEqual(@as(c_int, 32), Kernels.rowLanes(64));
}

test {
    _ = mlx_entries;
    _ = affine_stream;
    _ = affine_load;
}
