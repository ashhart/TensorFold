//! Neutral table identities for resident separate arrays and packed per-rank rows.
const std = @import("std");
const Pack = @import("pack.zig").Pack;
const Info = @import("pack.zig").Info;
const Config = @import("config.zig").Config;
const eh = @import("engram_host.zig");

/// A rank's packed rows of one Engram layer: rank `rank` of `world`, global rows [lo, hi) of `total`.
pub const Span = struct {
    rank: u32 = 0,
    world: u32 = 1,
    lo: u64,
    hi: u64,
    total: u64,

    /// The whole table as rank 0 of 1.
    pub fn whole(rows: u64) Span {
        return .{ .lo = 0, .hi = rows, .total = rows };
    }
};

/// Rank `rank` of `world`'s global rows of Engram layer `layer`: its whole hash columns (engram_host.headShard).
pub fn rankRows(c: *const Config, layer: u32, rank: u32, world: u32) !Span {
    const ids = c.engram_layers.items();
    const li = std.mem.indexOfScalar(u32, ids, layer) orelse return error.NotEngram;
    if (world == 0 or rank >= world or ids.len > eh.max_layers or c.engram_max_ngram < 2 or c.engram_max_ngram > eh.max_ngram) return error.BadEngramHost;
    var h: eh.Host = .{ .layer_ids = ids, .max_ngram = c.engram_max_ngram, .heads = c.engram_heads, .token_map = &.{}, .pad_id = 0 };
    const n = h.cols();
    if (n == 0 or n > eh.max_cols) return error.BadEngramHost;
    h.layout(c.engram_vocab);
    const total: u64 = @intCast(h.offsets[li][n - 1] + h.primes[li][n - 1]);
    const sh = h.headShard(rank, world);
    const at = struct {
        fn row(hh: *const eh.Host, l: usize, col: u32, cols: u32, end: u64) u64 {
            return if (col < cols) @intCast(hh.offsets[l][col]) else end;
        }
    }.row;
    return .{ .rank = rank, .world = world, .lo = at(&h, li, sh[0], n, total), .hi = at(&h, li, sh[0] + sh[1], n, total), .total = total };
}
pub const Plan = struct {
    layer: u32,
    rows: u64,
    weight: Info,
    scale: Info,
    pub fn fromPack(pack: *const Pack, layer: u32, rows: u64) !Plan {
        if (rows == 0 or rows > std.math.maxInt(u32)) return error.InvalidEngramTable;
        var buf: [128]u8 = undefined;
        const weight = try pack.need(try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.weight", .{layer}));
        const scale = try pack.need(try std.fmt.bufPrint(&buf, "layers.{d}.engram.embed.scale", .{layer}));
        if (!weight.is(.f8_e4m3, &.{ @intCast(rows), 256 }) or !scale.is(.f8_e8m0, &.{ @intCast(rows), 8 }))
            return error.InvalidEngramTable;
        return .{ .layer = layer, .rows = rows, .weight = weight, .scale = scale };
    }
};
test "separate rowwise scales use eight entries per row, not a matrix scale grid" {
    const a = std.testing.allocator;
    var p = Pack.init(a);
    defer p.deinit();
    try p.addHeader("table", "{\"layers.1.engram.embed.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[4,256],\"data_offsets\":[0,1024]},\"layers.1.engram.embed.scale\":{\"dtype\":\"F8_E8M0\",\"shape\":[4,8],\"data_offsets\":[1024,1056]}}", 1056);
    _ = try Plan.fromPack(&p, 1, 4);
    try std.testing.expectError(error.InvalidEngramTable, Plan.fromPack(&p, 1, 5));
}

test "a rank's Engram rows are its whole hash columns, the ranks tile the release's tables" {
    const c: Config = .{};
    for (c.engram_layers.items(), c.engram_rows.items()) |layer, rows| {
        for ([_]u32{ 1, 2, 3, 4, 8 }) |world| {
            var cursor: u64 = 0;
            for (0..world) |r| {
                const s = try rankRows(&c, layer, @intCast(r), world);
                try std.testing.expectEqual(rows, s.total);
                try std.testing.expectEqual(cursor, s.lo);
                try std.testing.expect(s.hi >= s.lo);
                cursor = s.hi;
            }
            try std.testing.expectEqual(rows, cursor);
        }
    }
    const w4 = try rankRows(&c, 14, 3, 4);
    try std.testing.expect(w4.lo > 0 and w4.hi == w4.total and w4.world == 4 and w4.rank == 3);
    try std.testing.expectEqualDeep(Span.whole(c.engram_rows.items()[0]), try rankRows(&c, 1, 0, 1));
    try std.testing.expectError(error.NotEngram, rankRows(&c, 2, 0, 4));
    try std.testing.expectError(error.BadEngramHost, rankRows(&c, 1, 4, 4));
}
