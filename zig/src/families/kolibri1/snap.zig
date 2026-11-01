//! A stream's caches at a prompt position: full layers' [0, at), each sliding layer's last window - 1 positions.
const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const Blob = @import("core").kv_prompts.Blob;

/// A kept state: per layer K then V, full [from, at), sliding last positions; [0, from) is the store's pages.
pub const Saved = struct {
    at: u32,
    from: u32 = 0,
    blob: []u32 = &.{}, // the copied rows in pages the store holds (TF_KV_SHARE), else in `buf`
    buf: cuda.DeviceBuffer,

    fn copyTo(k: *const Saved, m: *const fwd.Model, s: cuda.Stream, out: bool) Copy {
        if (k.blob.len > 0) return .{ .m = m, .s = s, .blob = .{ .ids = k.blob, .chunk = chunk(m), .regions = regions(m) }, .out = out };
        return .{ .m = m, .s = s, .buf = k.buf.ptr, .out = out };
    }
};

/// Positions a sliding layer keeps at `at`: the keys the next row's window reaches back to.
fn tail(m: *const fwd.Model, at: usize) usize {
    return @min(at, m.c.window - 1);
}

fn rowBytes(m: *const fwd.Model, _: usize) usize {
    return m.c.kv_heads * m.c.head_dim * 2;
}

/// Bytes a state at `at` copies, its full layers' positions below `from` held as pages.
pub fn copyBytes(m: *const fwd.Model, at: usize, from: usize) u64 {
    var n: usize = 0;
    for (0..m.c.layers) |i| n += 2 * (if (m.c.full[i]) at - from else tail(m, at)) * rowBytes(m, i);
    return n;
}

/// Each layer's cache, K then V, positions [first, at) of `slot` (full: from `from`), one copy a run, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, from: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    if (from > 0 and (!m.paged or from % fwd.pages.tokens != 0 or from > at)) return error.KeptPagesUnpaged;
    var off: usize = 0;
    for (0..m.c.layers) |i| for ([_][]u64{ m.k, m.v }) |cache| {
        const rb = rowBytes(m, i);
        if (m.c.full[i] and m.paged) { // pages: p in page p / page_tokens of the slot's row, a copy a run within one
            var p: usize = from;
            while (p < at) {
                const into = p % fwd.pages.tokens;
                const run = @min(at - p, fwd.pages.tokens - into);
                const row = @as(usize, m.ids[slot * fwd.pages.perSlot(m) + p / fwd.pages.tokens]) * fwd.pages.tokens + into;
                try f(ctx, cache[i] + row * rb, off, run * rb);
                off += run * rb;
                p += run;
            }
            continue;
        }
        const len = if (m.c.full[i]) m.context else m.ring;
        const base = cache[i] + slot * len * rb;
        var p = if (m.c.full[i]) 0 else at - tail(m, at);
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
    return fwd.pages.pageElems(m) * 2;
}

fn regions(m: *const fwd.Model) usize {
    return @intCast(fwd.pages.pageBytes(m) / chunk(m));
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
    return Blob.pagesFor(copyBytes(m, at, from), chunk(m), regions(m));
}

/// Copy `slot`'s state at `at` (full layers from `from`) to new storage or `blob`'s pages, queued after its chunk.
pub fn saveShared(gpa: std.mem.Allocator, m: *const fwd.Model, s: cuda.Stream, slot: usize, at: u32, from: u32, blob: []u32) !*Saved {
    if (at == 0 or at > m.context) return error.NotAtMark;
    if (from > at or (from > 0 and (!m.paged or from % fwd.pages.tokens != 0))) return error.KeptPagesUnpaged;
    const k = try gpa.create(Saved);
    errdefer gpa.destroy(k);
    k.* = .{ .at = at, .from = from, .blob = blob, .buf = .{ .d = m.d, .ptr = 0, .len = 0, .borrowed = true } };
    if (blob.len == 0) k.buf = try cuda.DeviceBuffer.alloc(m.d, copyBytes(m, at, from));
    errdefer k.buf.free();
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
    k.buf.free();
    gpa.destroy(k);
}

pub fn bytes(m: *const fwd.Model, at: usize) u64 {
    return copyBytes(m, at, 0);
}

pub fn save(gpa: std.mem.Allocator, m: *const fwd.Model, s: cuda.Stream, slot: usize, at: u32) !*Saved {
    return saveShared(gpa, m, s, slot, at, 0, &.{});
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
    m.paged = false;
    m.context = 64;
    m.ring = 8;
    m.k = &k;
    m.v = &v;
    const rb = 256;
    try testing.expectEqual(@as(u64, 2 * (4 + 10) * rb), bytes(&m, 10));
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
