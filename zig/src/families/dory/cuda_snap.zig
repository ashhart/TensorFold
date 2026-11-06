//! A stream's attention caches at a prompt position: 8 + 20 L + 8 per_loop caches, the sliding ones copied as rings.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const fwd = @import("cuda_model.zig");
const pages = @import("cuda_pages.zig");
const Blob = @import("core").kv_prompts.Blob;

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

/// A kept state: per cache K then V, full caches' [from, at), sliding rings' last positions; [0, from) the store's.
pub const Saved = struct {
    at: u32,
    key: Key,
    buf: cuda.DeviceBuffer,
    from: u32 = 0,
    blob: []u32 = &.{}, // the copied rows in pages the store holds (TF_KV_SHARE), else in `buf`

    fn target(k: *const Saved, m: *const fwd.Model, s: cuda.Stream, out: bool) Copy {
        if (k.blob.len > 0) return .{ .m = m, .s = s, .blob = .{ .ids = k.blob, .chunk = chunk(m), .regions = regions(m) }, .out = out };
        return .{ .m = m, .s = s, .buf = k.buf.ptr, .out = out };
    }
};

/// Positions a sliding ring keeps at `at`: a later row at `at` or past reads keys from at - window + 1 on.
fn tail(window: usize, at: usize) usize {
    return @min(at, window - 1);
}

/// Bytes a state at `at` takes.
pub fn bytes(m: *const fwd.Model, at: usize) u64 {
    return copied(m, at, 0);
}

/// Bytes a state at `at` copies with full caches from `from`.
fn copied(m: *const fwd.Model, at: usize, from: usize) u64 {
    const row = m.c.kvDim() * 2;
    var n: usize = 0;
    for (m.sched.cache_kind[0..m.sched.caches]) |kind| n += 2 * (if (kind == .full) at - from else tail(m.c.window, at)) * row;
    return n;
}

/// The bytes a page holds in one full cache's K or V.
fn chunk(m: *const fwd.Model) u64 {
    return pages.tokens * m.c.kvDim() * 2;
}

fn regions(m: *const fwd.Model) usize {
    return 2 * m.sched.cacheCount(.full);
}

/// Pages a state at `at` takes when the store holds its full rows and its rings are kept in pool pages.
pub fn blobPages(m: *const fwd.Model, at: usize) usize {
    return Blob.pagesFor(copied(m, at, at), chunk(m), regions(m));
}

/// Cache index of the `j`th full cache.
fn fullCache(m: *const fwd.Model, j: usize) usize {
    var n: usize = 0;
    for (m.sched.cache_kind[0..m.sched.caches], 0..) |kind, i| if (kind == .full) {
        if (n == j) return i;
        n += 1;
    };
    unreachable;
}

/// Each cache, K then V, of `slot`: rings' last rows, full caches' [from, at), one copy a run, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, from: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    const rb = m.c.kvDim() * 2;
    var off: usize = 0;
    for (0..m.sched.caches) |i| for ([_][]u64{ m.k, m.v }) |cache| {
        const full = m.sched.cache_kind[i] == .full;
        const len = m.cacheLen(i);
        if (full) if (m.paging) |pg| { // a full cache's positions: one copy a run of consecutive page ids
            var q = from;
            while (q < at) {
                const id = pg.ids[slot * pg.per_slot + q / pages.tokens];
                var end = (q / pages.tokens + 1) * pages.tokens;
                while (end < at and pg.ids[slot * pg.per_slot + end / pages.tokens] == id + (end - q) / pages.tokens) end += pages.tokens;
                const run = @min(at, end) - q;
                try f(ctx, cache[i] + (@as(usize, id) * pages.tokens + q % pages.tokens) * rb, off, run * rb);
                off += run * rb;
                q += run;
            }
            continue;
        };
        const base = cache[i] + slot * len * rb;
        var p = if (full) from else at - tail(m.c.window, at);
        while (p < at) {
            const row = p % len;
            const run = @min(at - p, len - row);
            try f(ctx, base + row * rb, off, run * rb);
            off += run * rb;
            p += run;
        }
    };
}

const Copy = struct { m: *const fwd.Model, s: cuda.Stream, buf: u64 = 0, blob: ?Blob = null, out: bool };

fn copy(c: Copy, dev: u64, off: usize, len: usize) anyerror!void {
    if (c.blob) |b| return b.each(off, len, Piece{ .c = c, .dev = dev }, Piece.copy);
    try move(c, dev, c.buf + off, len);
}

fn move(c: Copy, dev: u64, kept: u64, len: u64) !void {
    const src = if (c.out) dev else kept;
    const dst = if (c.out) kept else dev;
    try c.m.d.check(c.m.d.api.cuMemcpyDtoDAsync_v2(dst, src, len, c.s.handle), "cuMemcpyDtoDAsync");
}

/// A run's piece in one blob chunk: region r is full cache r / 2's K (even) or V (odd), page id's rows there.
const Piece = struct {
    c: Copy,
    dev: u64,
    fn copy(p: Piece, region: usize, id: u32, within: u64, from: u64, n: u64) anyerror!void {
        try move(p.c, p.dev + from, pieceAt(p.c.m, region, id) + within, n);
    }
};

fn pieceAt(m: *const fwd.Model, region: usize, id: u32) u64 {
    const i = fullCache(m, region / 2);
    return (if (region % 2 == 0) m.k else m.v)[i] + id * chunk(m);
}

/// Copy `slot`'s state at `at` into new storage, queued on `s` behind the chunk that ends there.
pub fn save(gpa: std.mem.Allocator, m: *const fwd.Model, s: cuda.Stream, slot: usize, at: u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .key = keyOf(m.c), .buf = try cuda.DeviceBuffer.alloc(m.d, @max(bytes(m, at), 256)) };
    errdefer k.buf.free();
    try each(m, slot, at, 0, k.target(m, s, true), copy);
    return k;
}

/// TF_KV_SHARE: `slot`'s rings at `at` into `blob`'s pages, queued like `save`; the store holds [0, at) as pages.
pub fn saveRings(gpa: std.mem.Allocator, m: *const fwd.Model, s: cuda.Stream, slot: usize, at: u32, blob: []u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .key = keyOf(m.c), .from = at, .blob = blob, .buf = .{ .d = m.d, .ptr = 0, .len = 0, .borrowed = true } };
    try each(m, slot, at, at, k.target(m, s, true), copy);
    return k;
}

/// Make `slot`'s caches hold `k`'s state, queued on `s` before the next chunk; another key's state is refused.
pub fn restore(m: *const fwd.Model, s: cuda.Stream, slot: usize, k: *const Saved) !void {
    if (!sameKey(k.key, keyOf(m.c))) return error.StateOfAnotherLoopCount;
    if (k.at > m.context) return error.PastContext;
    try each(m, slot, k.at, k.from, k.target(m, s, false), copy);
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
    try each(&m, 1, at, 0, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2 * (14 + 42 * 2)), runs.items.len);
    // cache 0 is sliding: slot 1's ring starts 3072 rows in, and position 2977 is ring row 2977 + 3072 * 1 - 3072
    const r0 = runs.items[0];
    try testing.expectEqual([3]usize{ (1 << 32) + (3072 + (at - 1023) % 3072) * rb, 0, (3072 - (at - 1023) % 3072) * rb }, r0);
    try testing.expectEqual([3]usize{ (1 << 32) + 3072 * rb, r0[2], (at % 3072) * rb }, runs.items[1]);
    var total: usize = 0;
    for (runs.items) |r| total += r[2];
    try testing.expectEqual(bytes(&m, at), total);
}

test "TF_KV_SHARE: a state's rings fill whole pages, no full row in it, each chunk one full cache's K or V of a page" {
    const c = try pinned(2);
    var m: fwd.Model = undefined;
    m.c = c;
    m.sched = @import("schedule.zig").build(c);
    m.context = 8192;
    m.ring = fwd.ringSize(fwd.prompt_rows, c.window);
    var ks: [64]u64 = undefined;
    var vs: [64]u64 = undefined;
    for (&ks, &vs, 0..) |*k, *v, i| {
        k.* = (i + 1) << 40;
        v.* = (i + 101) << 40;
    }
    m.k = ks[0..m.sched.caches];
    m.v = vs[0..m.sched.caches];
    const page = pages.tokens * 2 * c.kvDim() * 2 * m.sched.cacheCount(.full);
    try testing.expectEqual(@as(usize, 14 << 20), page); // 14 full caches at 2 loops, 1 MiB a page each
    const rings = copied(&m, 4096, 4096);
    try testing.expectEqual(@as(u64, 2 * 42 * 1023 * c.kvDim() * 2), rings);
    try testing.expectEqual((rings + page - 1) / page, blobPages(&m, 4096));
    try testing.expectEqual(@as(usize, 12), blobPages(&m, 4096));
    try testing.expectEqual(bytes(&m, 4096), rings + 14 * 4096 * 2 * c.kvDim() * 2);
    const full0 = fullCache(&m, 0);
    try testing.expectEqual(m.k[full0] + 7 * chunk(&m), pieceAt(&m, 0, 7));
    try testing.expectEqual(m.v[full0] + 7 * chunk(&m), pieceAt(&m, 1, 7));
    const last = fullCache(&m, 13);
    try testing.expectEqual(m.v[last] + 2 * chunk(&m), pieceAt(&m, regions(&m) - 1, 2));
    m.paging = null;
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 0, 4096, 4096, Count{ .runs = &runs }, Count.add);
    var total: usize = 0;
    for (runs.items) |r| total += r[2];
    try testing.expectEqual(rings, total);
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
