//! A release checkpoint's Engram shards for this rank, checked at boot: each layer's file, header and row span.
const std = @import("std");
const er = @import("engram_rows.zig");
const ep = @import("engram_pack.zig");
const et = @import("engram_tables.zig");
const Config = @import("config.zig").Config;

const Io = std.Io;

/// Every Engram layer's shard for rank `rank` of `world` under `dir`: present, its layer, rows and size as planned.
pub fn checkShards(io: Io, dir: ?[]const u8, c: *const Config, rank: u32, world: u32) !void {
    const root = dir orelse return error.EngramTablesMissing;
    for (c.engram_layers.items()) |layer| {
        const want = try et.rankRows(c, layer, rank, world);
        var nb: [64]u8 = undefined;
        var pb: [4096]u8 = undefined;
        const path = try std.fmt.bufPrint(&pb, "{s}/{s}", .{ root, try ep.shardName(&nb, layer, want) });
        const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.EngramShardMissing;
        defer f.close(io);
        var hb: [48]u8 = undefined;
        if (try f.readPositionalAll(io, &hb, 0) != hb.len) return error.EngramShardWrongSpan;
        const h = er.Header.parse(&hb, try f.length(io)) catch return error.EngramShardWrongSpan;
        if (h.layer != layer or h.lo != want.lo or h.hi != want.hi or h.total != want.total) return error.EngramShardWrongSpan;
    }
}

fn writeShard(io: Io, dir: std.Io.Dir, layer: u32, span: et.Span) !void {
    var nb: [64]u8 = undefined;
    const head = ep.header(layer, span);
    const f = try dir.createFile(io, try ep.shardName(&nb, layer, span), .{});
    defer f.close(io);
    try f.writePositionalAll(io, &head, 0);
    if (span.hi > span.lo) try f.writePositionalAll(io, &[_]u8{0}, er.header_bytes + (span.hi - span.lo) * er.row_bytes - 1);
}

test "boot: a rank's shards must exist with its own rows; a valid file with another range is refused" {
    const io = std.testing.io;
    const c: Config = .{};
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root);
    try std.testing.expectError(error.EngramTablesMissing, checkShards(io, null, &c, 1, 4));
    try std.testing.expectError(error.EngramShardMissing, checkShards(io, root, &c, 1, 4));
    // a well-formed shard of the right size, but rank 2's rows under rank 1's name
    for (c.engram_layers.items()) |layer| {
        var wrong = try et.rankRows(&c, layer, 2, 4);
        wrong.rank = 1;
        try writeShard(io, tmp.dir, layer, wrong);
    }
    try std.testing.expectError(error.EngramShardWrongSpan, checkShards(io, root, &c, 1, 4));
    for (c.engram_layers.items()) |layer| try writeShard(io, tmp.dir, layer, try et.rankRows(&c, layer, 1, 4));
    try checkShards(io, root, &c, 1, 4);
}
