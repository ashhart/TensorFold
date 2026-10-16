//! A lane round's plan on the device; launches depend only on its `Shape`, so one graph serves any streams and caches.

const std = @import("std");
const hip = @import("hip");
const state = @import("state.zig");

const round_shape = @import("round_shape.zig");

pub const buckets = round_shape.buckets;
pub const min_span = round_shape.min_span;
pub const bucketOf = round_shape.bucketOf;
pub const spanOf = round_shape.spanOf;
pub const Shape = round_shape.Shape;

/// Where each array sits in the plan buffer, in 4-byte words (the 8-byte arrays on even words).
pub const Layout = struct {
    layers: usize,
    tokens: usize,
    pos: usize,
    slot: usize,
    first: usize,
    count: usize,
    keep: usize,
    desc: usize,
    snaps: usize,
    words: usize,

    pub fn of(rows: usize, slots: usize, layers: usize) Layout {
        var l: Layout = undefined;
        l.layers = layers;
        l.tokens = 0;
        l.pos = rows;
        l.slot = 2 * rows;
        l.first = 3 * rows;
        l.count = l.first + slots;
        l.keep = l.count + slots;
        l.desc = std.mem.alignForward(usize, l.keep + slots, 2);
        l.snaps = l.desc + 2 * slots;
        l.words = l.snaps + 4 * layers;
        return l;
    }
};

fn put64(w: []u32, v: u64) void {
    w[0] = @truncate(v);
    w[1] = @truncate(v >> 32);
}

/// One stream's window of a round: its descriptor, the slot its first row takes, its tokens.
pub const Window = struct { desc: u64, pos: usize, tokens: []const u32 };

/// The plan buffer of one engine, as large as its biggest round, and the pinned words a round is written in.
pub const Buffer = struct {
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    /// Pages a head of a layer's pool holds: the kernels find a page's keys and values with it.
    pool_pages: u32,

    /// Device bytes a plan of `rows` rows over `layers` layers holds.
    pub fn deviceBytes(rows: usize, layers: usize) usize {
        return 4 * Layout.of(rows, rows + 1, layers).words;
    }

    pub fn init(d: *const hip.Runtime, rows: usize, layers: usize, pool_pages: usize) !Buffer {
        const bytes = deviceBytes(rows, layers);
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        return .{ .host = host, .dev = try hip.DeviceBuffer.alloc(d, bytes), .pool_pages = @intCast(pool_pages) };
    }

    pub fn deinit(b: *Buffer) void {
        b.dev.free();
        b.host.free();
    }

    /// The device words of `shape` as launches read them.
    pub fn args(b: *const Buffer, l: Layout) hip.plan_ops.Args {
        const at = b.dev.base();
        return .{
            .pos = at + 4 * l.pos,
            .slot = at + 4 * l.slot,
            .first = at + 4 * l.first,
            .count = at + 4 * l.count,
            .desc = at + 4 * l.desc,
            .snaps = at + 4 * l.snaps,
            .pages = @intCast(state.header_words + 2 * l.layers),
            .pool = b.pool_pages,
        };
    }

    /// Writes the round into the pinned words: windows, the scratch slot's `pad` rows, each linear layer's snapshots.
    pub fn fill(b: *Buffer, l: Layout, shape: Shape, windows: []const Window, scratch: u64, snaps: []const [2]u64) void {
        const w = b.host.slice(u32);
        const slots: usize = shape.slots;
        var at: usize = 0;
        for (windows, 0..) |win, s| {
            w[l.first + s] = @intCast(at);
            w[l.count + s] = @intCast(win.tokens.len);
            w[l.keep + s] = std.math.maxInt(u32);
            put64(w[l.desc + 2 * s ..], win.desc);
            for (win.tokens, 0..) |t, i| {
                w[l.tokens + at + i] = t;
                w[l.pos + at + i] = @intCast(win.pos + i);
                w[l.slot + at + i] = @intCast(s);
            }
            at += win.tokens.len;
        }
        // slots past the streams' are empty and the last is the scratch slot, where padding rows run from its start
        for (windows.len..slots - 1) |s| {
            w[l.first + s] = 0;
            w[l.count + s] = 0;
            w[l.keep + s] = std.math.maxInt(u32);
            put64(w[l.desc + 2 * s ..], 0);
        }
        const pad = shape.rows - at;
        w[l.first + slots - 1] = @intCast(at);
        w[l.count + slots - 1] = @intCast(pad);
        w[l.keep + slots - 1] = std.math.maxInt(u32);
        put64(w[l.desc + 2 * (slots - 1) ..], scratch);
        for (0..pad) |i| {
            w[l.tokens + at + i] = 0;
            w[l.pos + at + i] = @intCast(i);
            w[l.slot + at + i] = @intCast(slots - 1);
        }
        for (snaps, 0..) |pair, i| {
            put64(w[l.snaps + 4 * i ..], pair[0]);
            put64(w[l.snaps + 4 * i + 2 ..], pair[1]);
        }
    }

    /// The filled words up to the device in one copy (a graph holds the copy: it runs from the same pinned words).
    pub fn send(b: *const Buffer, stream: hip.abi.Stream, l: Layout) !void {
        try hip.raw.upload(b.dev, 0, b.host.bytes[0 .. 4 * l.words], stream);
    }
};

test "layout keeps the eight-byte arrays on even words" {
    const l = Layout.of(5, 4, 3);
    try std.testing.expect(l.desc % 2 == 0 and l.snaps % 2 == 0);
    try std.testing.expect(l.words >= l.snaps + 12);
}
