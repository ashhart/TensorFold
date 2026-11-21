//! A stream's caches for the prompt cache: each sliding layer's last window - 1 ring rows, each global's first `at`.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const fwd = @import("cuda_model.zig");

/// What a state was made under: its layers, which of them are global, the window and the KV width.
pub const Key = struct { layers: u32, global: [cfg.max_layers]bool, window: u32, kv_width: u32 };

/// The key every state of a model at `c` carries.
pub fn keyOf(c: cfg.Config) Key {
    return .{ .layers = @intCast(c.layers), .global = c.global, .window = @intCast(c.window), .kv_width = @intCast(c.kvWidth()) };
}

pub fn sameKey(a: Key, b: Key) bool {
    return std.meta.eql(a, b);
}

/// A kept state: a layer at a time, K then V, global layers' positions [0, at), sliding rings' last positions in order.
pub const Saved = struct { at: u32, key: Key, buf: cuda.DeviceBuffer };

/// Positions a sliding ring keeps at `at`: a later row at `at` or past reads keys from at - window + 1 on.
fn tail(window: usize, at: usize) usize {
    return @min(at, window - 1);
}

/// Bytes a state at `at` takes.
pub fn bytes(c: cfg.Config, at: usize) u64 {
    const row = c.kvWidth() * 2;
    var n: usize = 0;
    for (0..c.layers) |i| n += 2 * (if (c.global[i]) at else tail(c.window, at)) * row;
    return n;
}

/// Each layer's K then V rows of `slot` for positions [first, at): one call a run of consecutive rows, in order.
fn each(a: *const fwd.Arena, window: usize, slot: usize, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    var off: usize = 0;
    for (0..a.caches.len) |i| {
        const sp = a.span(i, slot);
        for ([_]u64{ sp.k, sp.v }) |base| {
            var p = if (sp.ring) at - tail(window, at) else 0;
            while (p < at) {
                const row = if (sp.ring) p % sp.rows else p; // a global layer's table is the identity
                const run = @min(at - p, sp.rows - row);
                try f(ctx, base + row * sp.row_bytes, off, run * sp.row_bytes);
                off += run * sp.row_bytes;
                p += run;
            }
        }
    }
}

const Copy = struct { d: *const cuda.Driver, s: cuda.Stream, buf: u64, out: bool };

fn copy(c: Copy, dev: u64, off: usize, len: usize) anyerror!void {
    const src = if (c.out) dev else c.buf + off;
    const dst = if (c.out) c.buf + off else dev;
    try c.d.check(c.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, c.s.handle), "cuMemcpyDtoDAsync");
}

/// Copy `slot`'s state at `at` into new storage, queued on the model's stream behind the chunk that ends there.
pub fn save(gpa: std.mem.Allocator, m: *const fwd.Model, slot: usize, at: u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .key = keyOf(m.c), .buf = try cuda.DeviceBuffer.alloc(m.d, @max(bytes(m.c, at), 256)) };
    errdefer k.buf.free();
    try each(&m.arena, m.c.window, slot, at, Copy{ .d = m.d, .s = m.s, .buf = k.buf.ptr, .out = true }, copy);
    return k;
}

/// Make `slot`'s caches hold `k`'s state, queued before its next chunk; another shape's state is refused.
pub fn restore(m: *const fwd.Model, slot: usize, k: *const Saved) !void {
    if (!sameKey(k.key, keyOf(m.c))) return error.StateOfAnotherShape;
    if (k.at == 0 or k.at > m.context) return error.PastContext;
    try each(&m.arena, m.c.window, slot, k.at, Copy{ .d = m.d, .s = m.s, .buf = k.buf.ptr, .out = false }, copy);
}

/// Free a state once the queued copies of it are done.
pub fn drop(gpa: std.mem.Allocator, s: cuda.Stream, k: *Saved) void {
    s.synchronize() catch {};
    k.buf.free();
    gpa.destroy(k);
}

const testing = std.testing;

const Count = struct {
    runs: *std.ArrayList([3]usize),
    fn add(c: Count, dev: u64, off: usize, len: usize) anyerror!void {
        try c.runs.append(testing.allocator, .{ dev, off, len });
    }
};

fn release() !cfg.Config {
    var c = try cfg.parse(testing.allocator, @embedFile("fixtures/config.json"));
    c.fp4_mlp = true;
    return c;
}

/// An arena of `slots` slots at `context` positions with fake device addresses (no device memory): layer i at i << 40.
fn fakeArena(c: cfg.Config, caches: []fwd.Cache, slots: usize, context: usize) fwd.Arena {
    const per = fwd.pagesOf(context);
    const kvw = c.kvWidth();
    for (caches, 0..) |*x, i| {
        const half: usize = slots * if (c.global[i]) per * fwd.page_tokens * kvw * 2 else fwd.ring_rows * kvw * 2;
        x.* = .{ .kc = @as(u64, i + 1) << 40, .vc = (@as(u64, i + 1) << 40) + half };
        if (c.global[i]) x.pg = .{ .table = 0, .page_elems = fwd.page_tokens * kvw, .tokens = fwd.page_tokens, .stride = per };
    }
    return .{ .gpa = testing.allocator, .kv = undefined, .caches = caches, .ids = &.{}, .slots = slots, .per_slot = per, .kvw = kvw };
}

test "ricursive cuda state: each ring's last 1,023 rows at their ring rows, wrapped in two runs; globals' first at rows" {
    const c = try release();
    var caches: [cfg.max_layers]fwd.Cache = undefined;
    const a = fakeArena(c, caches[0..c.layers], 3, 8192);
    const rb = c.kvWidth() * 2;
    const at = 3500; // a ring of 3,072: positions 2,477 to 3,071 then 3,072 to 3,499 at ring rows 0 to 427
    try testing.expectEqual(@as(usize, 2048), rb);
    try testing.expectEqual(@as(u64, 2 * (16 * at + 64 * 1023) * rb), bytes(c, at));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&a, c.window, 2, at, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2 * (16 + 64 * 2)), runs.items.len);
    // layer 0 is sliding: slot 2's ring starts 2 * 3,072 rows in; its K rows 2,477.. then 0..
    const k0 = caches[0].kc + 2 * fwd.ring_rows * rb;
    try testing.expectEqual([3]usize{ k0 + (at - 1023) * rb, 0, (fwd.ring_rows - (at - 1023)) * rb }, runs.items[0]);
    try testing.expectEqual([3]usize{ k0, runs.items[0][2], (at % fwd.ring_rows) * rb }, runs.items[1]);
    try testing.expectEqual(caches[0].vc + 2 * fwd.ring_rows * rb + (at - 1023) * rb, runs.items[2][0]);
    // layer 4 is global: one run of K and one of V from the slot's first page (the identity table)
    try testing.expect(c.global[4]);
    const g = runs.items[4 * 4 ..][0..2];
    try testing.expectEqual([3]usize{ caches[4].kc + 2 * fwd.pagesOf(8192) * fwd.page_tokens * rb, g[0][1], at * rb }, g[0]);
    try testing.expectEqual(caches[4].vc + 2 * fwd.pagesOf(8192) * fwd.page_tokens * rb, g[1][0]);
    var total: usize = 0;
    for (runs.items, 0..) |r, i| {
        if (i > 0) try testing.expectEqual(runs.items[i - 1][1] + runs.items[i - 1][2], r[1]); // packed in order
        total += r[2];
    }
    try testing.expectEqual(bytes(c, at), total);
}

test "ricursive cuda state: before the window fills, a ring keeps every position from 0 in one run" {
    const c = try release();
    var caches: [cfg.max_layers]fwd.Cache = undefined;
    const a = fakeArena(c, caches[0..c.layers], 1, 4096);
    const at = 510; // a page edge two rows on
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&a, c.window, 0, at, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2 * c.layers), runs.items.len);
    for (runs.items) |r| try testing.expectEqual(@as(usize, at * 2048), r[2]);
    try testing.expectEqual(@as(u64, 2 * c.layers * at * 2048), bytes(c, at));
}

test "ricursive cuda state: the key names the shape; another shape's state is refused" {
    const c = try release();
    var other = c;
    other.global[0] = true;
    try testing.expect(sameKey(keyOf(c), keyOf(c)));
    try testing.expect(!sameKey(keyOf(c), keyOf(other)));
    var m: fwd.Model = undefined;
    m.c = c;
    m.context = 8192;
    var buf: cuda.DeviceBuffer = undefined;
    buf.ptr = 0;
    const k: Saved = .{ .at = 10, .key = keyOf(other), .buf = buf };
    try testing.expectError(error.StateOfAnotherShape, restore(&m, 0, &k));
}
