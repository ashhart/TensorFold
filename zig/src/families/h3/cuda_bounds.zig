//! The H3 CUDA library's checks on a host's inputs, before any cast or allocation; std only, tested on any host.
const std = @import("std");

/// A sequence-parallel world's ranks, as h3_xchg.c allows them.
pub const max_world = 8;

pub const World = struct { world: u32, rank: u32 };

/// The C ABI's signed world and rank, checked before either is cast: 2 to 8 ranks and 0 <= rank < world.
pub fn world(ranks: c_int, rank: c_int) error{BadWorld}!World {
    if (ranks < 2 or ranks > max_world or rank < 0 or rank >= ranks) return error.BadWorld;
    return .{ .world = @intCast(ranks), .rank = @intCast(rank) };
}

/// The C ABI's signed peer, checked before it is cast: another rank of the world.
pub fn peer(p: c_int, w: World) error{BadWorld}!u32 {
    if (p < 0 or p >= w.world or p == w.rank) return error.BadWorld;
    return @intCast(p);
}

/// What a packed sequence's buffers are sized from.
pub const Dims = struct {
    rows: usize,
    tiles: usize,
    prefix: usize,
    keep: usize,
    hidden: usize,
    inner: usize,
    heads: usize,
    mlp: usize,
    head_dim: usize,
    slot_tile: usize,
    exchange: bool,
};

/// Every sequence buffer's bytes; `region` is what the exchange needs of the registered region (0 without one).
pub const Bytes = struct {
    stream: usize,
    qkvc: usize,
    tiled: usize,
    pooled: usize,
    scores: usize,
    chosen: usize,
    ya: usize,
    wide: usize,
    act: usize,
    row_of: usize,
    all: usize,
    unit: usize = 0,
    half: usize = 0,
    region: usize = 0,
};

fn product(factors: []const usize) error{GeometryTooLarge}!usize {
    var p: usize = 1;
    for (factors) |f| p = std.math.mul(usize, p, f) catch return error.GeometryTooLarge;
    return p;
}

/// Every buffer's bytes for `d`, each product checked (error.GeometryTooLarge on overflow or past the kernels' int).
pub fn bytes(d: Dims) error{ GeometryTooLarge, BadGeometry }!Bytes {
    if (d.prefix > d.tiles) return error.BadGeometry;
    const int_max: usize = std.math.maxInt(c_int);
    const slots = try product(&.{ d.tiles, d.slot_tile });
    if (d.rows > int_max or slots > int_max) return error.GeometryTooLarge;
    const routed = @max(d.tiles - d.prefix, 1);
    const kept = std.math.add(usize, d.prefix, d.keep) catch return error.GeometryTooLarge;
    var b: Bytes = .{
        .stream = try product(&.{ d.rows, d.hidden, 2 }),
        .qkvc = try product(&.{ d.rows, 4, d.inner, 2 }),
        .tiled = try product(&.{ d.heads, slots, d.head_dim, 2 }),
        .pooled = try product(&.{ d.heads, d.tiles, d.head_dim, 4 }),
        .scores = try product(&.{ d.heads, d.tiles, d.tiles, 4 }),
        .chosen = try product(&.{ d.heads, routed, kept, 4 }),
        .ya = try product(&.{ d.rows, d.inner, 2 }),
        .wide = try product(&.{ d.rows, 2, d.mlp, 2 }),
        .act = try product(&.{ d.rows, d.mlp, 2 }),
        .row_of = try product(&.{ slots, 4 }),
        .all = try product(&.{ d.tiles, 4 }),
    };
    if (d.exchange) {
        // a tile's keys and values in bf16 and its two pooled rows in float, for every head
        const tile = try product(&.{ 2, d.slot_tile, d.head_dim, 2 });
        const means = try product(&.{ 2, d.head_dim, 4 });
        b.unit = try product(&.{ d.heads, std.math.add(usize, tile, means) catch return error.GeometryTooLarge });
        const used = try product(&.{ d.tiles, b.unit });
        if (used > std.math.maxInt(usize) - 4095) return error.GeometryTooLarge;
        b.half = std.mem.alignForward(usize, used, 4096);
        b.region = try product(&.{ 2, b.half });
    }
    return b;
}

const sample: Dims = .{ .rows = 4096, .tiles = 64, .prefix = 4, .keep = 16, .hidden = 3072, .inner = 3072, .heads = 24, .mlp = 8192, .head_dim = 128, .slot_tile = 64, .exchange = true };

test "world and peer: signed values are refused before they are cast" {
    try std.testing.expectEqual(World{ .world = 2, .rank = 1 }, try world(2, 1));
    try std.testing.expectEqual(World{ .world = 8, .rank = 0 }, try world(8, 0));
    for ([_][2]c_int{ .{ 2, -1 }, .{ -2, 0 }, .{ 0, 0 }, .{ 1, 0 }, .{ 9, 0 }, .{ 2, 2 }, .{ 3, 7 }, .{ std.math.minInt(c_int), 0 }, .{ std.math.maxInt(c_int), 1 } }) |c|
        try std.testing.expectError(error.BadWorld, world(c[0], c[1]));
    const w = try world(3, 1);
    try std.testing.expectEqual(@as(u32, 2), try peer(2, w));
    for ([_]c_int{ -1, 1, 3, std.math.minInt(c_int) }) |p| try std.testing.expectError(error.BadWorld, peer(p, w));
}

test "bytes: a real sequence's sizes, the region's half on a page" {
    const b = try bytes(sample);
    try std.testing.expectEqual(@as(usize, 4096 * 3072 * 2), b.stream);
    try std.testing.expectEqual(@as(usize, 24 * 64 * 64 * 4), b.scores);
    try std.testing.expectEqual(@as(usize, 24 * 60 * 20 * 4), b.chosen);
    try std.testing.expectEqual(@as(usize, 24 * (2 * 64 * 128 * 2 + 2 * 128 * 4)), b.unit);
    try std.testing.expectEqual(@as(usize, 0), b.half % 4096);
    try std.testing.expect(b.half >= 64 * b.unit and b.half - 64 * b.unit < 4096);
    try std.testing.expectEqual(2 * b.half, b.region);
    var single = sample;
    single.exchange = false;
    try std.testing.expectEqual(@as(usize, 0), (try bytes(single)).region);
}

test "bytes: products that overflow are refused by name" {
    var d = sample;
    // a host's u32 tile count: its slots no longer fit the kernels' int
    d.tiles = std.math.maxInt(u32);
    try std.testing.expectError(error.GeometryTooLarge, bytes(d));
    d = sample;
    d.keep = std.math.maxInt(usize);
    try std.testing.expectError(error.GeometryTooLarge, bytes(d));
    d = sample;
    d.rows = @as(usize, std.math.maxInt(c_int)) + 1;
    try std.testing.expectError(error.GeometryTooLarge, bytes(d));
    d = sample;
    d.rows = std.math.maxInt(c_int);
    d.mlp = std.math.maxInt(u32) << 1;
    try std.testing.expectError(error.GeometryTooLarge, bytes(d));
    d = sample;
    // every product with heads wraps around 64 bits
    d.heads = std.math.maxInt(usize) / 8;
    try std.testing.expectError(error.GeometryTooLarge, bytes(d));
    d = sample;
    d.prefix = d.tiles + 1;
    try std.testing.expectError(error.BadGeometry, bytes(d));
}
