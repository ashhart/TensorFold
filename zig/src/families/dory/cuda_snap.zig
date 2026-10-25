//! A stream's attention caches at a prompt position: 8 + 20 L + 8 per_loop caches, the sliding ones copied as rings.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const fwd = @import("cuda_model.zig");
const pages = @import("cuda_pages.zig");

/// What a state was made under: its loop count and KV mode (a state never crosses either).
pub const Key = struct { loops: u8, mode: Mode };
/// per_loop is the only mode served (last_loop is refused at config parse).
pub const Mode = enum(u8) { per_loop };

/// The key every state of a model at `c` carries.
pub fn keyOf(c: config.Config) Key {
    return .{ .loops = @intCast(c.loops), .mode = .per_loop };
}

pub fn sameKey(a: Key, b: Key) bool {
    return a.loops == b.loops and a.mode == b.mode;
}

/// A kept state: per cache K then V, full caches' positions [0, at), sliding rings' last positions in order.
pub const Saved = struct { at: u32, key: Key, buf: cuda.DeviceBuffer };

/// Positions a sliding ring keeps at `at`: a later row at `at` or past reads keys from at - window + 1 on.
fn tail(window: usize, at: usize) usize {
    return @min(at, window - 1);
}

/// Bytes a state at `at` takes.
pub fn bytes(m: *const fwd.Model, at: usize) u64 {
    const row = m.c.kvDim() * 2;
    var n: usize = 0;
    for (m.sched.cache_kind[0..m.sched.caches]) |kind| n += 2 * (if (kind == .full) at else tail(m.c.window, at)) * row;
    return n;
}

/// Each cache, K then V, positions [first, at) of `slot`: one copy a run of ring rows, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    const rb = m.c.kvDim() * 2;
    var off: usize = 0;
    for (0..m.sched.caches) |i| for ([_][]u64{ m.k, m.v }) |cache| {
        const full = m.sched.cache_kind[i] == .full;
        const len = m.cacheLen(i);
        if (full) if (m.paging) |pg| { // a full cache's positions, a run in each page they lie in
            var q: usize = 0;
            while (q < at) {
                const row = q % pages.tokens;
                const run = @min(at - q, pages.tokens - row);
                const id = pg.ids[slot * pg.per_slot + q / pages.tokens];
                try f(ctx, cache[i] + @as(usize, id) * pg.pageBytes() + row * rb, off, run * rb);
                off += run * rb;
                q += run;
            }
            continue;
        };
        const base = cache[i] + slot * len * rb;
        var p = if (full) 0 else at - tail(m.c.window, at);
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
    k.* = .{ .at = at, .key = keyOf(m.c), .buf = try cuda.DeviceBuffer.alloc(m.d, @max(bytes(m, at), 256)) };
    errdefer k.buf.free();
    try each(m, slot, at, Copy{ .d = m.d, .s = s, .buf = k.buf.ptr, .out = true }, copy);
    return k;
}

/// Make `slot`'s caches hold `k`'s state, queued on `s` before the next chunk; another key's state is refused.
pub fn restore(m: *const fwd.Model, s: cuda.Stream, slot: usize, k: *const Saved) !void {
    if (!sameKey(k.key, keyOf(m.c))) return error.StateOfAnotherLoopCount;
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

fn pinned(loops: usize) !config.Config {
    const text = try config.pinnedText(testing.allocator);
    defer testing.allocator.free(text);
    var why: config.Why = .{};
    return (try config.parse(testing.allocator, text, &why)).withLoops(loops);
}

test "a state holds every full cache's positions and each ring's last window - 1, wrapped runs in order" {
    const c = try pinned(2);
    var m: fwd.Model = undefined;
    m.c = c;
    m.sched = @import("schedule.zig").build(c);
    m.context = 8192;
    m.paging = null;
    m.ring = fwd.ringSize(fwd.prompt_rows, c.window);
    var ks: [64]u64 = undefined;
    var vs: [64]u64 = undefined;
    for (&ks, &vs, 0..) |*k, *v, i| {
        k.* = (i + 1) << 32;
        v.* = (i + 101) << 32;
    }
    m.k = ks[0..m.sched.caches];
    m.v = vs[0..m.sched.caches];
    const at = 4000; // a ring of 3072: positions 2977..3071 then 0..927 (ring rows), 1023 in all
    const rb = c.kvDim() * 2;
    try testing.expectEqual(@as(usize, 8 + 20 * 2 + 8), m.sched.caches);
    try testing.expectEqual(@as(u64, 2 * (14 * at + 42 * 1023) * rb), bytes(&m, at));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2 * (14 + 42 * 2)), runs.items.len);
    // cache 0 is sliding: slot 1's ring starts 3072 rows in, and position 2977 is ring row 2977 + 3072 * 1 - 3072
    const r0 = runs.items[0];
    try testing.expectEqual([3]usize{ (1 << 32) + (3072 + (at - 1023) % 3072) * rb, 0, (3072 - (at - 1023) % 3072) * rb }, r0);
    try testing.expectEqual([3]usize{ (1 << 32) + 3072 * rb, r0[2], (at % 3072) * rb }, runs.items[1]);
    var total: usize = 0;
    for (runs.items) |r| total += r[2];
    try testing.expectEqual(bytes(&m, at), total);
}

test "a state's key carries the loop count and KV mode: another loop count's state is refused" {
    const two = keyOf(try pinned(2));
    const four = keyOf(try pinned(4));
    try testing.expect(sameKey(two, two));
    try testing.expect(!sameKey(two, four));
    var m: fwd.Model = undefined;
    m.c = try pinned(4);
    m.context = 8192;
    var buf: cuda.DeviceBuffer = undefined;
    buf.ptr = 0;
    const k: Saved = .{ .at = 10, .key = two, .buf = buf };
    try testing.expectError(error.StateOfAnotherLoopCount, restore(&m, undefined, 0, &k));
}
