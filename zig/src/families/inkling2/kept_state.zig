//! A kept prompt state's attention rows for any backend: which rows of which key head of which visit, in packed order.
const std = @import("std");

/// One visit's K and V caches, each head-major ([key head][ring][row]); `k` and `v` are where they start.
pub const Visit = struct { k: u64, v: u64, heads: u32, ring: u64, row: u64, global: bool };

/// Rows a visit keeps at `at`: a global layer all of them, a sliding one the `window - 1` its next token reads.
pub fn kept(global: bool, window: u32, at: u64) u64 {
    return if (global) at else @min(at, window - 1);
}

/// The packed bytes of every visit's kept rows at `at`; a backend's other state (its convolutions) follows them.
pub fn bytes(visits: []const Visit, window: u32, at: u64) u64 {
    var n: u64 = 0;
    for (visits) |v| n += 2 * @as(u64, v.heads) * kept(v.global, window, at) * v.row;
    return n;
}

/// Whether every visit's ring holds the rows it keeps at `at` (a cache too short for the state cannot take it).
pub fn fits(visits: []const Visit, window: u32, at: u64) bool {
    for (visits) |v| if (kept(v.global, window, at) > v.ring) return false;
    return true;
}

/// One contiguous copy: `len` bytes at `cache` (an address in a visit's caches) and at `offset` in the packed state.
pub const Piece = struct { cache: u64, offset: u64, len: u64 };

/// Every piece of the kept rows, in packed order: visit, K then V, key head, position; a ring wrap splits a head.
pub const Walk = struct {
    visits: []const Visit,
    window: u32,
    at: u64,
    visit: usize = 0,
    side: u1 = 0,
    head: u32 = 0,
    pos: ?u64 = null,
    offset: u64 = 0,

    pub fn init(visits: []const Visit, window: u32, at: u64) Walk {
        return .{ .visits = visits, .window = window, .at = at };
    }

    pub fn next(w: *Walk) ?Piece {
        while (w.visit < w.visits.len) {
            const v = w.visits[w.visit];
            if (w.head == v.heads) {
                w.head = 0;
                if (w.side == 1) w.visit += 1;
                w.side +%= 1;
                continue;
            }
            const pos = w.pos orelse w.at - kept(v.global, w.window, w.at);
            if (pos == w.at) {
                w.pos = null;
                w.head += 1;
                continue;
            }
            const slot = pos % v.ring;
            const n = @min(w.at - pos, v.ring - slot);
            const base = (if (w.side == 0) v.k else v.v) + w.head * v.ring * v.row;
            const piece: Piece = .{ .cache = base + slot * v.row, .offset = w.offset, .len = n * v.row };
            w.offset += piece.len;
            w.pos = pos + n;
            return piece;
        }
        return null;
    }
};

const testing = std.testing;

/// A host cache: `visits` laid out one after another, rings of `global` or `sliding` rows, each row `row` bytes.
fn layout(out: []Visit, global: u64, sliding: u64) u64 {
    const heads = 3;
    const row = 4;
    var at: u64 = 0;
    for (out, 0..) |*v, i| {
        const g = i % 3 == 1;
        const ring = if (g) global else sliding;
        v.* = .{ .k = at, .v = at + heads * ring * row, .heads = heads, .ring = ring, .row = row, .global = g };
        at += 2 * heads * ring * row;
    }
    return at;
}

/// The row a fresh prompt leaves at `pos` of a visit's key head: what any later reader of that position must see.
fn rowOf(visit: usize, side: u1, head: u32, pos: u64) [4]u8 {
    const x: u32 = @truncate((visit * 2 + side) * 7919 + head * 104729 + pos * 31);
    return @bitCast(x);
}

/// Writes positions 0..at in order, as a prompt does: a sliding ring's later rows overwrite its earlier ones.
fn fill(cache: []u8, visits: []const Visit, at: u64) void {
    for (visits, 0..) |v, i| for (0..2) |side| for (0..v.heads) |h| for (0..at) |pos| {
        const base = (if (side == 0) v.k else v.v) + h * v.ring * v.row;
        @memcpy(cache[base + (pos % v.ring) * v.row ..][0..4], &rowOf(i, @intCast(side), @intCast(h), pos));
    };
}

fn copyAll(cache: []u8, state: []u8, visits: []const Visit, window: u32, at: u64, out: bool) u64 {
    var w = Walk.init(visits, window, at);
    while (w.next()) |p| {
        const c = cache[p.cache..][0..p.len];
        const s = state[p.offset..][0..p.len];
        if (out) @memcpy(s, c) else @memcpy(c, s);
    }
    return w.offset;
}

test "a state restores each head's kept rows into a cache of another capacity" {
    const window = 6;
    var from_v: [6]Visit = undefined;
    var to_v: [6]Visit = undefined;
    // Sliding rings 8 and 12, global rings 16 and 40: every kept row lands at another slot after the restore.
    const from_len = layout(&from_v, 16, 8);
    const to_len = layout(&to_v, 40, 12);
    const from = try testing.allocator.alloc(u8, from_len);
    defer testing.allocator.free(from);
    const to = try testing.allocator.alloc(u8, to_len);
    defer testing.allocator.free(to);
    const state = try testing.allocator.alloc(u8, bytes(&from_v, window, 16));
    defer testing.allocator.free(state);
    for ([_]u64{ 1, 4, 5, 6, 9, 13, 16 }) |at| {
        try testing.expect(fits(&from_v, window, at) and fits(&to_v, window, at));
        @memset(from, 0);
        fill(from, &from_v, at);
        try testing.expectEqual(bytes(&from_v, window, at), copyAll(from, state, &from_v, window, at, true));
        @memset(to, 0xa5);
        try testing.expectEqual(bytes(&to_v, window, at), copyAll(to, state, &to_v, window, at, false));
        for (to_v, 0..) |v, i| for (0..2) |side| for (0..v.heads) |h| {
            for (at - kept(v.global, window, at)..at) |pos| {
                const base = (if (side == 0) v.k else v.v) + h * v.ring * v.row;
                const want = rowOf(i, @intCast(side), @intCast(h), pos);
                try testing.expectEqualSlices(u8, &want, to[base + (pos % v.ring) * v.row ..][0..4]);
            }
        };
    }
}

test "the walk keeps a sliding layer's window less one and splits a head where its ring wraps" {
    const v = [_]Visit{.{ .k = 0, .v = 1000, .heads = 2, .ring = 8, .row = 10, .global = false }};
    var w = Walk.init(&v, 6, 11);
    // Positions 6..10 sit in slots 6, 7, then 0..2: two pieces a head, K's two heads before V's.
    const want = [_]Piece{
        .{ .cache = 60, .offset = 0, .len = 20 },     .{ .cache = 0, .offset = 20, .len = 30 },
        .{ .cache = 140, .offset = 50, .len = 20 },   .{ .cache = 80, .offset = 70, .len = 30 },
        .{ .cache = 1060, .offset = 100, .len = 20 }, .{ .cache = 1000, .offset = 120, .len = 30 },
        .{ .cache = 1140, .offset = 150, .len = 20 }, .{ .cache = 1080, .offset = 170, .len = 30 },
    };
    for (want) |p| try testing.expectEqual(p, w.next().?);
    try testing.expectEqual(@as(?Piece, null), w.next());
    try testing.expectEqual(bytes(&v, 6, 11), w.offset);
    try testing.expect(!fits(&.{.{ .k = 0, .v = 0, .heads = 1, .ring = 8, .row = 1, .global = true }}, 6, 9));
}
