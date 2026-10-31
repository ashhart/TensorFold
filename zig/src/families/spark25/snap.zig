//! A stream's caches at a prompt position: full layers' [0, at), each sliding layer's last window - 1 positions.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const fwd = @import("forward.zig");
const Blob = @import("core").kv_prompts.Blob;

/// A kept state: per layer K then V, full [from, at), sliding last positions; [0, from) is the store's pages.
pub const Saved = struct {
    at: u32,
    from: u32 = 0,
    blob: []u32 = &.{}, // the copied rows in pages the store holds (TF_KV_SHARE), else in `buf`
    buf: ?cuda.DeviceBuffer = null,

    fn copyTo(k: *const Saved, m: *const fwd.Model, s: cuda.Stream, out: bool) Copy {
        if (k.blob.len > 0) return .{ .m = m, .s = s, .blob = .{ .ids = k.blob, .chunk = chunk(m), .regions = regions(m) }, .out = out };
        return .{ .m = m, .s = s, .buf = k.buf.?.ptr, .out = out };
    }
};

/// Positions a sliding layer keeps at `at`: the keys the next row's window reaches back to.
fn tail(at: usize) usize {
    return @min(at, cfg.window - 1);
}

fn rowBytes(m: *const fwd.Model) usize {
    return m.geo.kvWidth() * 2;
}

/// Bytes a state at `at` copies, its full layers' positions below `from` held as pages.
pub fn bytes(m: *const fwd.Model, at: usize, from: usize) u64 {
    var n: usize = 0;
    for (0..m.layers) |i| n += 2 * (if (cfg.full(i)) at - from else tail(at)) * rowBytes(m);
    return n;
}

/// Each layer's cache, K then V, positions [first, at) of `slot` (full: from `from`), one copy a run, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, from: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    if (from > 0 and (!m.paged or from % fwd.page_tokens != 0 or from > at)) return error.KeptPagesUnpaged;
    var off: usize = 0;
    const rb = rowBytes(m);
    for (0..m.layers) |i| for ([_][]u64{ m.kc, m.vc }) |cache| {
        if (cfg.full(i) and m.paged) { // pages: p in page p / page_tokens of the slot's row, a copy a run within one
            var p: usize = from;
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

/// A run's piece in one blob chunk: region r is full layer r / 2's K (even) or V (odd) region, page id's rows there.
const Piece = struct {
    c: Copy,
    dev: u64,
    fn copy(p: Piece, region: usize, id: u32, within: u64, from: u64, n: u64) anyerror!void {
        const i = fullLayer(p.c.m, region / 2);
        const base = (if (region % 2 == 0) p.c.m.kc else p.c.m.vc)[i];
        try move(p.c, p.dev + from, base + id * chunk(p.c.m) + within, n);
    }
};

/// The bytes a page holds in one full layer's K or V region.
fn chunk(m: *const fwd.Model) u64 {
    return m.pageElems() * 2;
}

fn regions(m: *const fwd.Model) usize {
    return @intCast(m.pageBytes() / chunk(m));
}

fn fullLayer(m: *const fwd.Model, j: usize) usize {
    var n: usize = 0;
    for (0..m.layers) |i| if (cfg.full(i)) {
        if (n == j) return i;
        n += 1;
    };
    unreachable;
}

/// Pages a state at `at` (full layers from `from`) takes when its copied rows are kept in pool pages.
pub fn blobPages(m: *const fwd.Model, at: usize, from: usize) usize {
    return Blob.pagesFor(bytes(m, at, from), chunk(m), regions(m));
}

/// Copy `slot`'s state at `at` (full layers from `from`) to new storage or `blob`'s pages, queued after its chunk.
pub fn save(gpa: std.mem.Allocator, m: *const fwd.Model, s: cuda.Stream, slot: usize, at: u32, from: u32, blob: []u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .from = from, .blob = blob };
    if (blob.len == 0) k.buf = try cuda.DeviceBuffer.alloc(m.d, @max(1, bytes(m, at, from)));
    errdefer if (k.buf) |*b| b.free();
    try each(m, slot, at, from, k.copyTo(m, s, true), copy);
    return k;
}

/// Make `slot`'s caches hold `k`'s state, queued on `s` before the next chunk.
pub fn restore(m: *const fwd.Model, s: cuda.Stream, slot: usize, k: *const Saved) !void {
    if (k.at > m.context) return error.PastContext;
    try each(m, slot, k.at, k.from, k.copyTo(m, s, false), copy);
}

/// Free a state once the stream's queued copies of it are done.
pub fn drop(gpa: std.mem.Allocator, s: cuda.Stream, k: *Saved) void {
    s.synchronize() catch {};
    if (k.buf) |*b| b.free();
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
    try testing.expectEqual(@as(u64, 3 * 2 * 511 * rb + 2 * at * rb), bytes(&m, at, 0));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, 0, Count{ .runs = &runs }, Count.add);
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
    try each(&m, 1, at, 0, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 3 * 2 + 2 * 3), runs.items.len); // a sliding cache's 511 rows: one run each
    const pt = fwd.page_tokens;
    const kr = runs.items[runs.items.len - 6 ..][0..3];
    try testing.expectEqual([3]usize{ (4 << 30) + 7 * pt * rb, kr[0][1], pt * rb }, kr[0]);
    try testing.expectEqual([3]usize{ (4 << 30) + 2 * pt * rb, kr[0][1] + pt * rb, pt * rb }, kr[1]);
    try testing.expectEqual([3]usize{ (4 << 30) + 9 * pt * rb, kr[0][1] + 2 * pt * rb, 10 * rb }, kr[2]);
    try testing.expectEqual(bytes(&m, at, 0), runs.items[runs.items.len - 1][1] + runs.items[runs.items.len - 1][2]);
    runs.clearRetainingCapacity(); // pages 7 and 2 held by reference: only page 9's 10 rows are copied
    try each(&m, 1, at, 2 * pt, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 3 * 2 + 2), runs.items.len);
    const tk = runs.items[runs.items.len - 2];
    try testing.expectEqual([3]usize{ (4 << 30) + 9 * pt * rb, tk[1], 10 * rb }, tk);
    try testing.expectEqual(bytes(&m, at, 2 * pt), runs.items[runs.items.len - 1][1] + runs.items[runs.items.len - 1][2]);
    try testing.expectError(error.KeptPagesUnpaged, each(&m, 1, at, pt + 1, Count{ .runs = &runs }, Count.add));
}
