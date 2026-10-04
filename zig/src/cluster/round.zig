//! What the leader broadcasts each round: last round's rollbacks and releases, then every row (decode windows and prompt rows).
const std = @import("std");

/// One row: feed `token` into slot `slot`'s cache at `index`; `sample` rows come back as logits for the leader to draw.
pub const Row = struct {
    slot: u32,
    token: u32,
    index: u32,
    sample: bool,
};

/// Trim slot `slot`'s caches to `len` tokens (the kept rows of its last window).
pub const Keep = struct { slot: u32, len: u32 };

pub const Kind = enum(u8) { round = 1, release = 2, stop = 3 };

pub const Command = struct {
    kind: Kind,
    number: u64 = 0,
    keeps: []const Keep = &.{},
    releases: []const u32 = &.{},
    rows: []const Row = &.{},

    pub fn sampled(c: Command) usize {
        var n: usize = 0;
        for (c.rows) |r| n += @intFromBool(r.sample);
        return n;
    }
};

pub const Error = error{ Short, Corrupt, OutOfMemory };

pub fn encode(a: std.mem.Allocator, c: Command) Error![]u8 {
    const n = 1 + 8 + 12 + c.keeps.len * 8 + c.releases.len * 4 + c.rows.len * 13;
    const out = try a.alloc(u8, n);
    out[0] = @intFromEnum(c.kind);
    std.mem.writeInt(u64, out[1..9], c.number, .little);
    std.mem.writeInt(u32, out[9..13], @intCast(c.keeps.len), .little);
    std.mem.writeInt(u32, out[13..17], @intCast(c.releases.len), .little);
    std.mem.writeInt(u32, out[17..21], @intCast(c.rows.len), .little);
    var at: usize = 21;
    for (c.keeps) |k| {
        std.mem.writeInt(u32, out[at..][0..4], k.slot, .little);
        std.mem.writeInt(u32, out[at + 4 ..][0..4], k.len, .little);
        at += 8;
    }
    for (c.releases) |r| {
        std.mem.writeInt(u32, out[at..][0..4], r, .little);
        at += 4;
    }
    for (c.rows) |r| {
        std.mem.writeInt(u32, out[at..][0..4], r.slot, .little);
        std.mem.writeInt(u32, out[at + 4 ..][0..4], r.token, .little);
        std.mem.writeInt(u32, out[at + 8 ..][0..4], r.index, .little);
        out[at + 12] = @intFromBool(r.sample);
        at += 13;
    }
    return out;
}

/// Decode into `a` (slices owned by it).
pub fn decode(a: std.mem.Allocator, b: []const u8) Error!Command {
    if (b.len < 21) return error.Short;
    var c: Command = .{ .kind = std.enums.fromInt(Kind, b[0]) orelse return error.Corrupt, .number = std.mem.readInt(u64, b[1..9], .little) };
    const nk = std.mem.readInt(u32, b[9..13], .little);
    const nr = std.mem.readInt(u32, b[13..17], .little);
    const nw = std.mem.readInt(u32, b[17..21], .little);
    if (b.len != 21 + @as(usize, nk) * 8 + @as(usize, nr) * 4 + @as(usize, nw) * 13) return error.Corrupt;
    var at: usize = 21;
    const keeps = try a.alloc(Keep, nk);
    for (keeps) |*k| {
        k.* = .{ .slot = std.mem.readInt(u32, b[at..][0..4], .little), .len = std.mem.readInt(u32, b[at + 4 ..][0..4], .little) };
        at += 8;
    }
    const releases = try a.alloc(u32, nr);
    for (releases) |*r| {
        r.* = std.mem.readInt(u32, b[at..][0..4], .little);
        at += 4;
    }
    const rows = try a.alloc(Row, nw);
    for (rows) |*r| {
        r.* = .{ .slot = std.mem.readInt(u32, b[at..][0..4], .little), .token = std.mem.readInt(u32, b[at + 4 ..][0..4], .little), .index = std.mem.readInt(u32, b[at + 8 ..][0..4], .little), .sample = b[at + 12] != 0 };
        at += 13;
    }
    c.keeps = keeps;
    c.releases = releases;
    c.rows = rows;
    return c;
}

test "commands round-trip and refuse bad lengths" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c: Command = .{ .kind = .round, .number = 9, .keeps = &.{.{ .slot = 1, .len = 40 }}, .releases = &.{3}, .rows = &.{ .{ .slot = 1, .token = 5, .index = 40, .sample = true }, .{ .slot = 2, .token = 7, .index = 0, .sample = false } } };
    const b = try encode(a, c);
    const d = try decode(a, b);
    try std.testing.expectEqual(c.number, d.number);
    try std.testing.expectEqualSlices(Keep, c.keeps, d.keeps);
    try std.testing.expectEqualSlices(u32, c.releases, d.releases);
    try std.testing.expectEqualSlices(Row, c.rows, d.rows);
    try std.testing.expectEqual(@as(usize, 1), d.sampled());
    try std.testing.expectError(error.Corrupt, decode(a, b[0 .. b.len - 1]));
    var bad = try a.dupe(u8, b);
    bad[0] = 9;
    try std.testing.expectError(error.Corrupt, decode(a, bad));
}
