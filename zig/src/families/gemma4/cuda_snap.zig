//! A stream's caches at a prompt position: full layers' [0, at), sliding layers' last window - 1; owning layers only.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("cuda_config.zig");
const fwd = @import("cuda_forward.zig");
const Blob = @import("core").kv_prompts.Blob;

/// A kept state: per layer K then V, full [from, at), sliding last positions; [0, from) is the store's pages.
pub const Saved = struct {
    at: u32,
    from: u32 = 0,
    blob: []u32 = &.{}, // the copied rows in pages the store holds (TF_KV_SHARE), else in `buf`
    buf: cuda.DeviceBuffer,
    draft: ?cuda.DeviceBuffer = null,

    fn copyTo(k: *const Saved, m: *const fwd.Model, s: cuda.Stream, out: bool) Copy {
        if (k.blob.len > 0) return .{ .m = m, .s = s, .blob = .{ .ids = k.blob, .chunk = chunk(m), .regions = regions(m) }, .out = out };
        return .{ .m = m, .s = s, .buf = k.buf.ptr, .out = out };
    }
};

/// Positions a sliding layer keeps at `at`: the keys the next row's window reaches back to.
fn tail(at: usize) usize {
    return @min(at, cfg.window - 1);
}

fn rowBytes(m: *const fwd.Model, i: usize) usize {
    return m.geo.kvHeads(i) * cfg.headDim(i) * 2;
}

/// Bytes a state at `at` copies, its full layers' positions below `from` held as pages.
pub fn copyBytes(m: *const fwd.Model, at: usize, from: usize) u64 {
    var n: usize = 0;
    for (0..fwd.pages.ownLayers(m)) |i| {
        const rows = if (cfg.global(i)) fwd.pages.copies(m) * (at - from) else 2 * tail(at); // V-only: V alone
        n += rows * rowBytes(m, i);
    }
    return n;
}

/// Each layer's cache, K then V, positions [first, at) of `slot` (full: from `from`), one copy a run, in order.
fn each(m: *const fwd.Model, slot: usize, at: usize, from: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx), dev: u64, off: usize, len: usize) anyerror!void) !void {
    if (from > 0 and (!m.paged or from % fwd.pages.tokens != 0 or from > at)) return error.KeptPagesUnpaged;
    var off: usize = 0;
    for (0..fwd.pages.ownLayers(m)) |i| for ([_][]u64{ m.kc, m.vc }, 0..) |cache, kv| {
        if (kv == 0 and m.v_only and cfg.global(i)) continue; // V-only pages: kc aliases vc (TF_GEMMA4_V_ONLY)
        const rb = rowBytes(m, i);
        if (cfg.global(i) and m.paged) { // pages: p in page p / page_tokens of the slot's row, a copy a run within one
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
        const len = m.cacheLen(i);
        const base = cache[i] + slot * len * rb;
        var p = if (cfg.global(i)) 0 else at - tail(at);
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
    return fwd.pages.pageElems(m) * 2;
}

fn regions(m: *const fwd.Model) usize {
    return @intCast(fwd.pages.pageBytes(m) / chunk(m));
}

fn fullLayer(m: *const fwd.Model, j: usize) usize {
    var n: usize = 0;
    for (0..fwd.pages.ownLayers(m)) |i| if (cfg.global(i)) {
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
    if (k.draft) |*b| b.free();
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

test "a state holds global layers' positions and each sliding ring's last window - 1, wrapped runs in order" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30, 5 << 30, 6 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30, 11 << 30, 12 << 30 };
    var m: fwd.Model = undefined;
    m.geo = cfg.unified12b;
    m.layers = 6; // layer 5 is global
    m.context = 4096;
    m.paged = false;
    m.v_only = false;
    m.prompt_most = fwd.prompt_rows;
    m.kc = &k;
    m.vc = &v;
    const at = 2000; // sliding: positions 977..1999, ring rows 977..1535 then 0..463
    const sb = 8 * 256 * 2;
    const gb = 1 * 512 * 2;
    try testing.expectEqual(@as(u64, 5 * 2 * 1023 * sb + 2 * at * gb), bytes(&m, at));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, 0, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 5 * 2 * 2 + 2), runs.items.len);
    const r0 = runs.items[0];
    try testing.expectEqual([3]usize{ (1 << 30) + (fwd.ring + 977) * sb, 0, (fwd.ring - 977) * sb }, r0);
    try testing.expectEqual([3]usize{ (1 << 30) + fwd.ring * sb, (fwd.ring - 977) * sb, 464 * sb }, runs.items[1]);
    const last = runs.items[runs.items.len - 1];
    try testing.expectEqual([3]usize{ (12 << 30) + 4096 * gb, 2 * 5 * 1023 * sb + at * gb, at * gb }, last);
}

test "a solo model's wider ring holds the same window - 1 positions, unwrapped where the grid ring wraps" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30, 5 << 30, 6 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30, 11 << 30, 12 << 30 };
    var m: fwd.Model = undefined;
    m.geo = cfg.unified12b;
    m.layers = 6;
    m.context = 4096;
    m.paged = false;
    m.v_only = false;
    m.prompt_most = 2048; // ring rows 3072: positions 977..1999 sit in rows 977..1999
    m.kc = &k;
    m.vc = &v;
    const at = 2000;
    const sb = 8 * 256 * 2;
    const gb = 1 * 512 * 2;
    try testing.expectEqual(@as(u64, 5 * 2 * 1023 * sb + 2 * at * gb), bytes(&m, at));
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, 0, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 5 * 2 + 2), runs.items.len);
    try testing.expectEqual([3]usize{ (1 << 30) + (3072 + 977) * sb, 0, 1023 * sb }, runs.items[0]);
}

test "a paged state takes a global layer's positions through the slot's pages, a run a page, in position order" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30, 5 << 30, 6 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30, 11 << 30, 12 << 30 };
    var ids = [_]u32{ 0, 0, 0, 0, 7, 2, 9, 0 }; // slot 1's row: pages 7, 2, 9, then unmapped
    var m: fwd.Model = undefined;
    m.geo = cfg.unified12b;
    m.layers = 6; // layer 5 is global
    m.context = 4 * fwd.pages.tokens;
    m.paged = true;
    m.v_only = false;
    m.prompt_most = fwd.prompt_rows;
    m.ids = &ids;
    m.kc = &k;
    m.vc = &v;
    const at = 2 * fwd.pages.tokens + 10; // pages 7 and 2 whole, then 10 rows of page 9
    const gb = 1 * 512 * 2;
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, 0, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 5 * 2 + 2 * 3), runs.items.len); // a sliding cache's 522 rows: one run each
    const pt = fwd.pages.tokens;
    const kr = runs.items[runs.items.len - 6 ..][0..3];
    try testing.expectEqual([3]usize{ (6 << 30) + 7 * pt * gb, kr[0][1], pt * gb }, kr[0]);
    try testing.expectEqual([3]usize{ (6 << 30) + 2 * pt * gb, kr[0][1] + pt * gb, pt * gb }, kr[1]);
    try testing.expectEqual([3]usize{ (6 << 30) + 9 * pt * gb, kr[0][1] + 2 * pt * gb, 10 * gb }, kr[2]);
    try testing.expectEqual(bytes(&m, at), runs.items[runs.items.len - 1][1] + runs.items[runs.items.len - 1][2]);
}

test "a V-only state copies a global layer's V pages alone, and as many bytes as it says" {
    var k = [_]u64{ 1 << 30, 2 << 30, 3 << 30, 4 << 30, 5 << 30, 6 << 30 };
    var v = [_]u64{ 7 << 30, 8 << 30, 9 << 30, 10 << 30, 11 << 30, 12 << 30 };
    var ids = [_]u32{ 0, 0, 0, 0, 7, 2, 9, 0 };
    var m: fwd.Model = undefined;
    m.geo = cfg.unified12b;
    m.layers = 6; // layer 5 is global
    m.context = 4 * fwd.pages.tokens;
    m.paged = true;
    m.v_only = true;
    m.prompt_most = fwd.prompt_rows;
    m.ids = &ids;
    m.kc = &k;
    m.vc = &v;
    const at = 2 * fwd.pages.tokens + 10;
    var runs: std.ArrayList([3]usize) = .empty;
    defer runs.deinit(testing.allocator);
    try each(&m, 1, at, 0, Count{ .runs = &runs }, Count.add);
    try testing.expectEqual(@as(usize, 5 * 2 + 3), runs.items.len); // the global layer's V runs, no K run
    for (runs.items[runs.items.len - 3 ..]) |r| try testing.expect(r[0] >= 12 << 30);
    const last = runs.items[runs.items.len - 1];
    try testing.expectEqual(bytes(&m, at), last[1] + last[2]);
    const vo = bytes(&m, at);
    m.v_only = false;
    try testing.expectEqual(bytes(&m, at) - at * 512 * 2, vo); // the global layer's keys
}
