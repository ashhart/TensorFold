//! A stream's caches at a prompt position: full layers' [0, at), each sliding layer's last window - 1 positions.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const fwd = @import("forward.zig");

/// A kept state: per layer K then V, full layers' positions [0, at), sliding layers' last positions in order.
pub const Saved = struct { at: u32, buf: cuda.DeviceBuffer };

/// Positions a sliding layer keeps at `at`: the keys the next row's window reaches back to.
fn tail(at: usize) usize {
    return @min(at, cfg.window - 1);
}

fn rowBytes(m: *const fwd.Model) usize {
    return m.geo.kvWidth() * 2;
}

/// Bytes a state at `at` takes.
pub fn bytes(m: *const fwd.Model, at: usize) u64 {
    var n: usize = 0;
    for (0..m.layers) |i| n += 2 * (if (cfg.full(i)) at else tail(at)) * rowBytes(m);
    return n;
}

/// Each layer's cache, K then V, positions [first, at) of `slot`: one copy a run of ring rows, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    var off: usize = 0;
    const rb = rowBytes(m);
    for (0..m.layers) |i| for ([_][]u64{ m.kc, m.vc }) |cache| {
        if (cfg.full(i) and m.paged) { // pages: p in page p / page_tokens of the slot's row, a copy a run within one
            var p: usize = 0;
            while (p < at) {
                const into = p % fwd.page_tokens;
                const run = @min(at - p, fwd.page_tokens - into);
                const row = @as(usize, m.ids[slot * m.perSlot() + p / fwd.page_tokens]) * fwd.page_tokens + into;
                try f(ctx, cache[i] + row * rb, off, run * rb);
                off += run * rb;
                p += run;
            }
            continue;
        }
        const len = m.cacheLen(i);
        const base = cache[i] + slot * len * rb;
        var p = if (cfg.full(i)) 0 else at - tail(at);
        while (p < at) {
            const row = p % len;
            const run = @min(at - p, len - row);
            try f(ctx, base + row * rb, off, run * rb);
            off += run * rb;
            p += run;
        }
    };
}

const Copy = struct { d: *const cuda.Driver, s: cuda.Stream, buf: u64, out: bool };

fn copy(c: Copy, dev: u64, off: usize, len: usize) anyerror!void {
    const src = if (c.out) dev else c.buf + off;
    const dst = if (c.out) c.buf + off else dev;
    try c.d.check(c.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, c.s.handle), "cuMemcpyDtoDAsync");
}

/// Copy `slot`'s state at `at` into new storage, queued on `s` behind the chunk that ends there.
pub fn save(gpa: std.mem.Allocator, m: *const fwd.Model, s: cuda.Stream, slot: usize, at: u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .buf = try cuda.DeviceBuffer.alloc(m.d, bytes(m, at)) };
    errdefer k.buf.free();
    try each(m, slot, at, Copy{ .d = m.d, .s = s, .buf = k.buf.ptr, .out = true }, copy);
    return k;
}

/// Make `slot`'s caches hold `k`'s state, queued on `s` before the next chunk.
pub fn restore(m: *const fwd.Model, s: cuda.Stream, slot: usize, k: *const Saved) !void {
    if (k.at > m.context) return error.PastContext;
    try each(m, slot, k.at, Copy{ .d = m.d, .s = s, .buf = k.buf.ptr, .out = false }, copy);
}

/// Free a state once the stream's queued copies of it are done.
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

test "a state holds full layers' positions and each sliding ring's last window - 1, wrapped runs in order" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30 };
    var m: fwd.Model = undefined;
    m.geo = cfg.spark4b;
    m.layers = 4; // layer 3 is full
    m.context = 4096;
    m.paged = false;
    m.kc = &k;
    m.vc = &v;
    const at = 1500; // sliding: positions 989..1499, ring rows 989..1023 then 0..475
    const rb = 4 * 256 * 2;
    try testing.expectEqual(@as(u64, 3 * 2 * 511 * rb + 2 * at * rb), bytes(&m, at));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 3 * 2 * 2 + 2), runs.items.len);
    try testing.expectEqual([3]usize{ (1 << 30) + (fwd.ring + 989) * rb, 0, (fwd.ring - 989) * rb }, runs.items[0]);
    try testing.expectEqual([3]usize{ (1 << 30) + fwd.ring * rb, (fwd.ring - 989) * rb, 476 * rb }, runs.items[1]);
    const last = runs.items[runs.items.len - 1];
    try testing.expectEqual([3]usize{ (10 << 30) + 4096 * rb, 2 * 3 * 511 * rb + at * rb, at * rb }, last);
}

test "a paged state takes a full layer's positions through the slot's pages, a run a page, in position order" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30 };
    var ids = [_]u32{ 0, 0, 0, 0, 7, 2, 9, 0 }; // slot 1's row: pages 7, 2, 9, then unmapped
    var m: fwd.Model = undefined;
    m.geo = cfg.spark4b;
    m.layers = 4; // layer 3 is full
    m.context = 4 * fwd.page_tokens;
    m.paged = true;
    m.ids = &ids;
    m.kc = &k;
    m.vc = &v;
    const at = 2 * fwd.page_tokens + 10; // pages 7 and 2 whole, then 10 rows of page 9
    const rb = 4 * 256 * 2;
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 3 * 2 + 2 * 3), runs.items.len); // a sliding cache's 511 rows: one run each
    const pt = fwd.page_tokens;
    const kr = runs.items[runs.items.len - 6 ..][0..3];
    try testing.expectEqual([3]usize{ (4 << 30) + 7 * pt * rb, kr[0][1], pt * rb }, kr[0]);
    try testing.expectEqual([3]usize{ (4 << 30) + 2 * pt * rb, kr[0][1] + pt * rb, pt * rb }, kr[1]);
    try testing.expectEqual([3]usize{ (4 << 30) + 9 * pt * rb, kr[0][1] + 2 * pt * rb, 10 * rb }, kr[2]);
    try testing.expectEqual(bytes(&m, at), runs.items[runs.items.len - 1][1] + runs.items[runs.items.len - 1][2]);
}
