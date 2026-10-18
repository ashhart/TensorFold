//! A stream's cache state at a prompt position, for the core prompt cache: full layers' keys, sliding layers' window.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");

/// A kept state: the keys of positions [0, at) of full layers, then the window's last keys of each sliding layer.
pub const Saved = struct { at: u32, buf: cuda.DeviceBuffer };

/// Positions a sliding layer keeps at `at`: the next row reads (at - window, at], so window - 1 before it.
fn tail(m: *const fwd.Model, at: usize) usize {
    return @min(at, m.c.window - 1);
}

fn rowBytes(m: *const fwd.Model) usize {
    return m.c.kv_heads * m.c.head_dim * 2;
}

/// Bytes a state at `at` takes.
pub fn bytes(m: *const fwd.Model, at: usize) u64 {
    var n: usize = 0;
    for (m.c.full[0..m.c.layers]) |full| n += 2 * (if (full) at else tail(m, at));
    return n * rowBytes(m);
}

/// Each layer's cache, K then V, and positions [first, at) of `slot`: one copy a run of ring rows, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    const rb = rowBytes(m);
    var off: usize = 0;
    for (0..m.c.layers) |i| for ([_][]u64{ m.k, m.v }) |cache| {
        const full = m.c.full[i];
        const len = if (full) m.context else m.ring;
        const base = cache[i] + slot * len * rb;
        var p = if (full) 0 else at - tail(m, at);
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

test "a state holds full layers' positions and each sliding ring's window, wrapped runs in position order" {
    const Config = @import("config.zig").Config;
    var c: Config = .{ .layers = 2, .hidden = 128, .heads = 4, .kv_heads = 1, .head_dim = 128, .vocab = 8, .experts = 2, .top_k = 1, .moe_width = 128, .shared_width = 128, .window = 5, .rope_theta = 1e4, .eps = 1e-6 };
    c.full[1] = true;
    var k = [_]u64{ 1 << 20, 2 << 20 };
    var v = [_]u64{ 3 << 20, 4 << 20 };
    var m: fwd.Model = undefined;
    m.c = c;
    m.context = 64;
    m.ring = 8;
    m.k = &k;
    m.v = &v;
    const rb = 256;
    try testing.expectEqual(@as(u64, 2 * (4 + 10) * rb), bytes(&m, 10));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, 10, Count{ .runs = &runs }, Count.add);
    // sliding K: positions 6..9 at ring rows 6, 7 then 0, 1 of slot 1; then V; then full K and V, positions 0..9
    const want = [_][3]usize{
        .{ (1 << 20) + (8 + 6) * rb, 0, 2 * rb },      .{ (1 << 20) + 8 * rb, 2 * rb, 2 * rb },
        .{ (3 << 20) + (8 + 6) * rb, 4 * rb, 2 * rb }, .{ (3 << 20) + 8 * rb, 6 * rb, 2 * rb },
        .{ (2 << 20) + 64 * rb, 8 * rb, 10 * rb },     .{ (4 << 20) + 64 * rb, 18 * rb, 10 * rb },
    };
    try testing.expectEqualSlices([3]usize, &want, runs.items);
}
