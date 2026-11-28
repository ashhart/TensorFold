//! A kept state's byte image as its family lays it out: rows by position, which blocks share, and its own parts.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// One part of a state's image at its position.
pub const Part = struct {
    bytes: u64, // its bytes in the image
    row: u32 = 0, // a row's bytes; 0: the state's own (recurrent and conv states, a family's words)
    per: u32 = 1, // tokens a row covers (pooled rows: the last, partial one stays the state's own)
    window: u32 = 0, // rows of the last `window` positions, position p at row p % window; 0: rows from position 0
};

/// `len` bytes at `off` of an image.
pub const Range = struct { off: u64, len: u64 };

/// The largest payload copy staged on the engine thread.
pub const copyChunk: u64 = 16 << 20;

/// The next range and byte offset within it of a chunked image copy.
pub const Cursor = struct { index: usize = 0, off: u64 = 0 };

/// Consecutive slices from `cursor`, at most `limit` bytes together; the cursor moves only after success.
pub fn nextChunk(ranges: []const Range, cursor: *Cursor, limit: u64, out: *std.ArrayList(Range), a: Allocator) !u64 {
    std.debug.assert(limit > 0);
    out.clearRetainingCapacity();
    var next = cursor.*;
    var n: u64 = 0;
    while (next.index < ranges.len and n < limit) {
        const r = ranges[next.index];
        if (next.off == r.len) {
            next.index += 1;
            next.off = 0;
            continue;
        }
        const take = @min(r.len - next.off, limit - n);
        try out.append(a, .{ .off = r.off + next.off, .len = take });
        next.off += take;
        n += take;
    }
    while (next.index < ranges.len and next.off == ranges[next.index].len) {
        next.index += 1;
        next.off = 0;
    }
    cursor.* = next;
    return n;
}

/// Where a group's whole blocks of `block` tokens lie for a state at `at`: blocks [first, end).
pub const Span = struct { first: u32, end: u32 };

/// The whole blocks a state at `at` holds of the rows kept for `window` positions (0: from position 0).
pub fn wholeBlocks(at: u32, block: u32, window: u32) Span {
    const lo = if (window == 0) 0 else at -| window;
    const first = (lo + block - 1) / block;
    const end = at / block;
    return if (first < end) .{ .first = first, .end = end } else .{ .first = 0, .end = 0 };
}

/// The distinct windows of an image's row parts, 0 (rows from position 0) first: each is a group of block files.
pub fn groups(parts: []const Part, out: *[8]u32) ![]const u32 {
    var n: usize = 0;
    for (parts) |p| {
        if (p.row == 0) continue;
        if (p.window > 0 and p.per != 1) return error.PooledRing;
        if (std.mem.indexOfScalar(u32, out[0..n], p.window) != null) continue;
        if (n == out.len) return error.TooManyGroups;
        out[n] = p.window;
        n += 1;
    }
    std.mem.sort(u32, out[0..n], {}, std.sort.asc(u32));
    return out[0..n];
}

/// Bytes of one block of `block` tokens in group `window`: every row part of that group, back to back.
pub fn blockBytes(parts: []const Part, block: u32, window: u32) u64 {
    var n: u64 = 0;
    for (parts) |p| if (p.row > 0 and p.window == window) {
        n += @as(u64, block / p.per) * p.row;
    };
    return n;
}

/// Positions [lo, hi) of a ring part at `off` as ranges of its rows (one, or two where they wrap).
fn ring(out: *std.ArrayList(Range), a: Allocator, off: u64, p: Part, lo: u32, hi: u32) !void {
    if (hi <= lo) return;
    const w = p.window;
    const s = lo % w;
    const n = hi - lo;
    const first = @min(n, w - s);
    try out.append(a, .{ .off = off + @as(u64, s) * p.row, .len = @as(u64, first) * p.row });
    if (n > first) try out.append(a, .{ .off = off, .len = @as(u64, n - first) * p.row });
}

/// The image ranges of block `k` (tokens [k*block, (k+1)*block)) of group `window`, in part order.
pub fn blockRanges(parts: []const Part, block: u32, window: u32, k: u32, out: *std.ArrayList(Range), a: Allocator) !void {
    var off: u64 = 0;
    for (parts) |p| {
        defer off += p.bytes;
        if (p.row == 0 or p.window != window) continue;
        if (window > 0) {
            try ring(out, a, off, p, k * block, (k + 1) * block);
            continue;
        }
        const rows = block / p.per;
        try out.append(a, .{ .off = off + @as(u64, k) * rows * p.row, .len = @as(u64, rows) * p.row });
    }
}

/// The image ranges a state at `at` keeps in its own file: whole parts, then each row part's rows outside its blocks.
pub fn ownRanges(parts: []const Part, at: u32, block: u32, out: *std.ArrayList(Range), a: Allocator) !void {
    var off: u64 = 0;
    for (parts) |p| {
        defer off += p.bytes;
        if (p.row == 0) {
            if (p.bytes > 0) try out.append(a, .{ .off = off, .len = p.bytes });
            continue;
        }
        const w = wholeBlocks(at, block, p.window);
        if (p.window > 0) {
            const lo = at -| p.window;
            if (w.end == 0) {
                try ring(out, a, off, p, lo, at);
            } else {
                try ring(out, a, off, p, lo, w.first * block);
                try ring(out, a, off, p, w.end * block, at);
            }
            continue;
        }
        const from = @as(u64, w.end) * (block / p.per) * p.row;
        if (from < p.bytes) try out.append(a, .{ .off = off + from, .len = p.bytes - from });
    }
}

/// The bytes of `ranges`.
pub fn total(ranges: []const Range) u64 {
    var n: u64 = 0;
    for (ranges) |r| n += r.len;
    return n;
}

/// The image's bytes: every part's.
pub fn imageBytes(parts: []const Part) u64 {
    var n: u64 = 0;
    for (parts) |p| n += p.bytes;
    return n;
}

/// Whether `parts` describe a state at `at` that blocks of `block` tokens can split (rows whole, pools inside a block).
pub fn valid(parts: []const Part, at: u32, block: u32) bool {
    for (parts) |p| {
        if (p.row == 0) continue;
        if (p.per == 0 or block % p.per != 0 or p.bytes % p.row != 0) return false;
        const rows = p.bytes / p.row;
        if (p.window > 0 and (p.per != 1 or rows != @min(at, p.window))) return false;
        if (p.window == 0 and (rows < at / p.per or rows > at / p.per + 1)) return false;
    }
    return true;
}

test "whole blocks: rows from 0 hold every finished block; a ring holds those inside its window" {
    try std.testing.expectEqual(Span{ .first = 0, .end = 3 }, wholeBlocks(800, 256, 0));
    try std.testing.expectEqual(Span{ .first = 0, .end = 0 }, wholeBlocks(200, 256, 0));
    try std.testing.expectEqual(Span{ .first = 1, .end = 3 }, wholeBlocks(800, 256, 600)); // positions 200..800
    try std.testing.expectEqual(Span{ .first = 0, .end = 0 }, wholeBlocks(700, 256, 300)); // 400..700: no block inside
}

test "a block's ranges and a state's own ranges cover every byte of the image once" {
    const a = std.testing.allocator;
    const at: u32 = 1000;
    const parts = [_]Part{
        .{ .bytes = 40 }, // a recurrent state
        .{ .bytes = at * 8, .row = 8 }, // keys
        .{ .bytes = (at / 4 + 1) * 6, .row = 6, .per = 4 }, // a pooled index, its partial last row
        .{ .bytes = 300 * 4, .row = 4, .window = 300 }, // a ring of the last 300 positions
        .{ .bytes = 12 }, // a family's words
    };
    try std.testing.expect(valid(&parts, at, 64));
    var seen = try a.alloc(u8, @intCast(imageBytes(&parts)));
    defer a.free(seen);
    @memset(seen, 0);
    var list: std.ArrayList(Range) = .empty;
    defer list.deinit(a);
    var gbuf: [8]u32 = undefined;
    for (try groups(&parts, &gbuf)) |g| {
        const w = wholeBlocks(at, 64, g);
        var k = w.first;
        while (k < w.end) : (k += 1) {
            list.clearRetainingCapacity();
            try blockRanges(&parts, 64, g, k, &list, a);
            try std.testing.expectEqual(blockBytes(&parts, 64, g), total(list.items));
            for (list.items) |r| for (seen[@intCast(r.off)..][0..@intCast(r.len)]) |*b| {
                b.* += 1;
            };
        }
    }
    list.clearRetainingCapacity();
    try ownRanges(&parts, at, 64, &list, a);
    for (list.items) |r| for (seen[@intCast(r.off)..][0..@intCast(r.len)]) |*b| {
        b.* += 1;
    };
    for (seen) |b| try std.testing.expectEqual(@as(u8, 1), b);
}

test "a ring's block that wraps its window comes back as two ranges in position order" {
    const a = std.testing.allocator;
    const parts = [_]Part{.{ .bytes = 100 * 2, .row = 2, .window = 100 }};
    var list: std.ArrayList(Range) = .empty;
    defer list.deinit(a);
    try blockRanges(&parts, 64, 100, 1, &list, a); // positions 64..128: rows 64..99, then 0..27
    try std.testing.expectEqualSlices(Range, &.{ .{ .off = 128, .len = 72 }, .{ .off = 0, .len = 56 } }, list.items);
}

test "nextChunk cuts only at its byte limit and resumes at the exact range seam" {
    const ranges = [_]Range{ .{ .off = 10, .len = 5 }, .{ .off = 40, .len = 9 }, .{ .off = 80, .len = 4 } };
    var cursor: Cursor = .{};
    var out: std.ArrayList(Range) = .empty;
    defer out.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u64, 6), try nextChunk(&ranges, &cursor, 6, &out, std.testing.allocator));
    try std.testing.expectEqualSlices(Range, &.{ .{ .off = 10, .len = 5 }, .{ .off = 40, .len = 1 } }, out.items);
    try std.testing.expectEqual(Cursor{ .index = 1, .off = 1 }, cursor);
    try std.testing.expectEqual(@as(u64, 6), try nextChunk(&ranges, &cursor, 6, &out, std.testing.allocator));
    try std.testing.expectEqualSlices(Range, &.{.{ .off = 41, .len = 6 }}, out.items);
    try std.testing.expectEqual(@as(u64, 6), try nextChunk(&ranges, &cursor, 6, &out, std.testing.allocator));
    try std.testing.expectEqualSlices(Range, &.{ .{ .off = 47, .len = 2 }, .{ .off = 80, .len = 4 } }, out.items);
    try std.testing.expectEqual(ranges.len, cursor.index);
    try std.testing.expectEqual(@as(u64, 0), try nextChunk(&ranges, &cursor, 6, &out, std.testing.allocator));
    try std.testing.expectEqual(@as(usize, 0), out.items.len);
}

test "bounded chunks gather exact destinations and copy failures leave the source intact" {
    const a = std.testing.allocator;
    const n: usize = @intCast(copyChunk + 11);
    const source = try a.alloc(u8, n);
    defer a.free(source);
    for (source, 0..) |*b, i| b.* = @truncate(i *% 73 +% 19);
    const before = try a.dupe(u8, source);
    defer a.free(before);
    const dest = try a.alloc(u8, n);
    defer a.free(dest);
    @memset(dest, 0xaa);
    const ranges = [_]Range{ .{ .off = 0, .len = copyChunk - 3 }, .{ .off = copyChunk - 3, .len = 8 }, .{ .off = copyChunk + 5, .len = 6 } };
    const Gather = struct {
        fn apply(src: []const u8, rs: []const Range, out: []u8, fail: bool) !void {
            if (fail) return error.GatherFailed;
            var done: usize = 0;
            for (rs) |r| {
                const len: usize = @intCast(r.len);
                @memcpy(out[done..][0..len], src[@intCast(r.off)..][0..len]);
                done += len;
            }
        }
    };
    var out: std.ArrayList(Range) = .empty;
    defer out.deinit(a);
    var cursor: Cursor = .{};
    const first = try nextChunk(&ranges, &cursor, copyChunk, &out, a);
    try std.testing.expect(first <= copyChunk);
    const stage = try a.alloc(u8, @intCast(first));
    defer a.free(stage);
    try std.testing.expectError(error.GatherFailed, Gather.apply(source, out.items, stage, true));
    try std.testing.expectEqualSlices(u8, before, source);
    try std.testing.expectEqualSlices(u8, &.{0xaa}, dest[0..1]);

    cursor = .{};
    var done: usize = 0;
    while (cursor.index < ranges.len) {
        const bytes = try nextChunk(&ranges, &cursor, copyChunk, &out, a);
        try std.testing.expect(bytes > 0 and bytes <= copyChunk);
        const bounded = try a.alloc(u8, @intCast(bytes));
        defer a.free(bounded);
        try Gather.apply(source, out.items, bounded, false);
        @memcpy(dest[done..][0..bounded.len], bounded);
        done += bounded.len;
    }
    try std.testing.expectEqual(n, done);
    try std.testing.expectEqualSlices(u8, source, dest);
    try std.testing.expectEqualSlices(u8, before, source);

    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    var fresh: std.ArrayList(Range) = .empty;
    defer fresh.deinit(a);
    cursor = .{};
    try std.testing.expectError(error.OutOfMemory, nextChunk(&ranges, &cursor, copyChunk, &fresh, failing.allocator()));
    try std.testing.expectEqual(Cursor{}, cursor);
    try std.testing.expectEqualSlices(u8, before, source);
}

test "a later chunk allocation fails with a bounded peak and an unchanged source" {
    const a = std.testing.allocator;
    const n: usize = @intCast(copyChunk + 11);
    const source = try a.alloc(u8, n);
    defer a.free(source);
    for (source, 0..) |*b, i| b.* = @truncate(i *% 73 +% 19);
    const before = try a.dupe(u8, source);
    defer a.free(before);
    const dest = try a.alloc(u8, n);
    defer a.free(dest);
    @memset(dest, 0xaa);
    const ranges = [_]Range{ .{ .off = 0, .len = copyChunk - 3 }, .{ .off = copyChunk - 3, .len = 8 }, .{ .off = copyChunk + 5, .len = 6 } };
    var out: std.ArrayList(Range) = .empty;
    defer out.deinit(a);
    var cursor: Cursor = .{};
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 1 });
    const fa = failing.allocator();
    const first = try nextChunk(&ranges, &cursor, copyChunk, &out, a);
    const one = try fa.alloc(u8, @intCast(first));
    var copied: usize = 0;
    for (out.items) |r| {
        const len: usize = @intCast(r.len);
        @memcpy(one[copied..][0..len], source[@intCast(r.off)..][0..len]);
        @memcpy(dest[copied..][0..len], one[copied..][0..len]);
        copied += len;
    }
    fa.free(one);
    _ = try nextChunk(&ranges, &cursor, copyChunk, &out, a);
    try std.testing.expectError(error.OutOfMemory, fa.alloc(u8, @intCast(total(out.items))));
    try std.testing.expectEqualSlices(u8, before, source);

    var peak: @import("rows_peak_allocator.zig").Peak = .{ .child = a };
    const pa = peak.allocator();
    cursor = .{};
    var done: usize = 0;
    while (cursor.index < ranges.len) {
        const bytes = try nextChunk(&ranges, &cursor, copyChunk, &out, a);
        const stage = try pa.alloc(u8, @intCast(bytes));
        var off: usize = 0;
        for (out.items) |r| {
            const len: usize = @intCast(r.len);
            @memcpy(stage[off..][0..len], source[@intCast(r.off)..][0..@intCast(r.len)]);
            off += len;
        }
        @memcpy(dest[done..][0..@intCast(bytes)], stage);
        pa.free(stage);
        done += @intCast(bytes);
    }
    try std.testing.expectEqualSlices(u8, source, dest);
    try std.testing.expectEqual(@as(usize, copyChunk), peak.peak);
    try std.testing.expectEqual(@as(usize, 0), peak.live);
}
