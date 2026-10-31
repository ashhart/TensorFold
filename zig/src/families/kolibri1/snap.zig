//! A stream's cache state at a prompt position, for the core prompt cache: full layers' keys, sliding layers' window.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const Blob = @import("core").kv_prompts.Blob;

/// A kept state: full layers' keys [from, at), then each sliding layer's window; [0, from) is the store's pages.
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

/// Positions a sliding layer keeps at `at`: the next row reads (at - window, at], so window - 1 before it.
fn tail(m: *const fwd.Model, at: usize) usize {
    return @min(at, m.c.window - 1);
}

fn rowBytes(m: *const fwd.Model) usize {
    return m.c.kv_heads * m.c.head_dim * 2;
}

/// Bytes a state at `at` copies, its full layers' positions below `from` held as pages.
pub fn bytes(m: *const fwd.Model, at: usize, from: usize) u64 {
    var n: usize = 0;
    for (m.c.full[0..m.c.layers]) |full| n += 2 * (if (full) at - from else tail(m, at));
    return n * rowBytes(m);
}

/// Each layer's cache, K then V, and positions [first, at) of `slot` (full: from `from`), one copy a run, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, from: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    if (from > 0 and (!m.paged or from % fwd.page_tokens != 0 or from > at)) return error.KeptPagesUnpaged;
    const rb = rowBytes(m);
    var off: usize = 0;
    for (0..m.c.layers) |i| for ([_][]u64{ m.k, m.v }) |cache| {
        const full = m.c.full[i];
        if (full and m.paged) { // pages: p in page p / page_tokens of the slot's row, a copy a run within one
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
        const base = (if (region % 2 == 0) p.c.m.k else p.c.m.v)[i];
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
    for (0..m.c.layers) |i| if (m.c.full[i]) {
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
    m.paged = false;
    m.k = &k;
    m.v = &v;
    const rb = 256;
    try testing.expectEqual(@as(u64, 2 * (4 + 10) * rb), bytes(&m, 10, 0));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, 10, 0, Count{ .runs = &runs }, Count.add);
    // sliding K: positions 6..9 at ring rows 6, 7 then 0, 1 of slot 1; then V; then full K and V, positions 0..9
    const want = [_][3]usize{
        .{ (1 << 20) + (8 + 6) * rb, 0, 2 * rb },      .{ (1 << 20) + 8 * rb, 2 * rb, 2 * rb },
        .{ (3 << 20) + (8 + 6) * rb, 4 * rb, 2 * rb }, .{ (3 << 20) + 8 * rb, 6 * rb, 2 * rb },
        .{ (2 << 20) + 64 * rb, 8 * rb, 10 * rb },     .{ (4 << 20) + 64 * rb, 18 * rb, 10 * rb },
    };
    try testing.expectEqualSlices([3]usize, &want, runs.items);
}

test "a paged state takes a full layer's positions through the slot's pages, a run a page, in position order" {
    const Config = @import("config.zig").Config;
    var c: Config = .{ .layers = 2, .hidden = 128, .heads = 4, .kv_heads = 1, .head_dim = 128, .vocab = 8, .experts = 2, .top_k = 1, .moe_width = 128, .shared_width = 128, .window = 5, .rope_theta = 1e4, .eps = 1e-6 };
    c.full[1] = true;
    var k = [_]u64{ 1 << 30, 2 << 30 };
    var v = [_]u64{ 3 << 30, 4 << 30 };
    var ids = [_]u32{ 0, 0, 0, 5, 1, 0 }; // slot 1's row: pages 5, 1, then unmapped
    var m: fwd.Model = undefined;
    m.c = c;
    m.context = 3 * fwd.page_tokens;
    m.ring = 8;
    m.paged = true;
    m.ids = &ids;
    m.k = &k;
    m.v = &v;
    const rb = 256;
    const pt = fwd.page_tokens;
    const at = pt + 7; // page 5 whole, then 7 rows of page 1
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, 0, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2 * 1 + 2 * 2), runs.items.len); // sliding K and V: one run each
    const kr = runs.items[2..4];
    try testing.expectEqual([3]usize{ (2 << 30) + 5 * pt * rb, 8 * rb, pt * rb }, kr[0]);
    try testing.expectEqual([3]usize{ (2 << 30) + 1 * pt * rb, 8 * rb + pt * rb, 7 * rb }, kr[1]);
    try testing.expectEqual([3]usize{ (4 << 30) + 1 * pt * rb, bytes(&m, at, 0) - 7 * rb, 7 * rb }, runs.items[5]);
    runs.clearRetainingCapacity(); // page 5 held by reference: only page 1's 7 rows are copied
    try each(&m, 1, at, pt, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 2 * 1 + 2), runs.items.len);
    try testing.expectEqual([3]usize{ (2 << 30) + 1 * pt * rb, runs.items[2][1], 7 * rb }, runs.items[2]);
    try testing.expectEqual(bytes(&m, at, pt), runs.items[3][1] + runs.items[3][2]);
    try testing.expectError(error.KeptPagesUnpaged, each(&m, 1, at, pt + 1, Count{ .runs = &runs }, Count.add));
}
