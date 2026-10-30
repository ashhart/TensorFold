//! A stream's caches at a prompt position: global layers' [0, at), each sliding layer's last window - 1 positions.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("cuda_config.zig");
const fwd = @import("cuda_forward.zig");

/// A kept state: per layer K then V, global layers' positions [0, at), sliding layers' last positions in order.
pub const Saved = struct { at: u32, buf: cuda.DeviceBuffer, draft: ?cuda.DeviceBuffer = null };

/// Positions a sliding layer keeps at `at`.
fn tail(at: usize) usize {
    return @min(at, cfg.window - 1);
}

fn rowBytes(m: *const fwd.Model, i: usize) usize {
    return m.geo.kvHeads(i) * cfg.headDim(i) * 2;
}

/// Bytes a state at `at` takes.
pub fn bytes(m: *const fwd.Model, at: usize) u64 {
    var n: usize = 0;
    for (0..m.layers) |i| n += 2 * (if (cfg.global(i)) at else tail(at)) * rowBytes(m, i);
    return n;
}

/// Each layer's cache, K then V, positions [first, at) of `slot`: one copy a run of ring rows, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    var off: usize = 0;
    for (0..m.layers) |i| for ([_][]u64{ m.kc, m.vc }) |cache| {
        const rb = rowBytes(m, i);
        const global = cfg.global(i);
        const len = m.cacheLen(i);
        const base = cache[i] + slot * len * rb;
        var p = if (global) 0 else at - tail(at);
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
    if (k.draft) |*b| b.free();
    gpa.destroy(k);
}

const testing = std.testing;

const Count = struct {
    runs: *std.ArrayList([3]usize),
    fn add(c: Count, dev: u64, off: usize, len: usize) anyerror!void {
        try c.runs.append(testing.allocator, .{ dev, off, len });
    }
};

test "a state holds global layers' positions and each sliding ring's last window - 1, wrapped runs in order" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30, 5 << 30, 6 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30, 11 << 30, 12 << 30 };
    var m: fwd.Model = undefined;
    m.geo = cfg.unified12b;
    m.layers = 6; // layer 5 is global
    m.context = 4096;
    m.kc = &k;
    m.vc = &v;
    const at = 2000; // sliding: positions 977..1999, ring rows 977..1535 then 0..463
    const sb = 8 * 256 * 2;
    const gb = 1 * 512 * 2;
    try testing.expectEqual(@as(u64, 5 * 2 * 1023 * sb + 2 * at * gb), bytes(&m, at));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 5 * 2 * 2 + 2), runs.items.len);
    const r0 = runs.items[0];
    try testing.expectEqual([3]usize{ (1 << 30) + (fwd.ring + 977) * sb, 0, (fwd.ring - 977) * sb }, r0);
    try testing.expectEqual([3]usize{ (1 << 30) + fwd.ring * sb, (fwd.ring - 977) * sb, 464 * sb }, runs.items[1]);
    const last = runs.items[runs.items.len - 1];
    try testing.expectEqual([3]usize{ (12 << 30) + 4096 * gb, 2 * 5 * 1023 * sb + at * gb, at * gb }, last);
}
