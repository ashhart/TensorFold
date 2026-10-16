//! Image rows for the prompt: which images a pass needs, one device buffer of their rows, the splice into embeddings.
const std = @import("std");
const cuda = @import("cuda");
const Image = @import("lanes").stream.Image;
const Allocator = std.mem.Allocator;

/// An image's placeholder rows [at, at + n) of the prompt, its rows from `row` of the buffer, `image` its index.
pub const Span = struct { at: u32, n: u32, row: u32, image: u32 };

/// The images a prompt pass from `from` reads, in prompt order, their rows back to back from row 0.
pub fn plan(a: Allocator, images: []const Image, from: u32) ![]Span {
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(a);
    var row: u32 = 0;
    var end: u32 = 0;
    for (images, 0..) |img, i| {
        if (img.at < end or img.tokens == 0) return error.ImageSpans; // in order, apart, each with rows
        end = img.at + img.tokens;
        if (end <= from) continue; // a restored prefix holds these rows' states already
        try out.append(a, .{ .at = img.at, .n = img.tokens, .row = row, .image = @intCast(i) });
        row += img.tokens;
    }
    return out.toOwnedSlice(a);
}

/// Rows the spans hold in all.
pub fn total(spans: []const Span) u32 {
    var n: u32 = 0;
    for (spans) |s| n += s.n;
    return n;
}

/// bf16 [limit, hidden] row-major: a pass's image rows, filled on rank 0 and broadcast whole to the others.
pub const Rows = struct {
    d: *const cuda.Driver,
    buf: cuda.DeviceBuffer,
    hidden: usize,
    limit: u32,

    pub fn init(d: *const cuda.Driver, limit: u32, hidden: usize) !Rows {
        return .{ .d = d, .buf = try cuda.DeviceBuffer.alloc(d, @as(usize, limit) * hidden * 2), .hidden = hidden, .limit = limit };
    }

    pub fn deinit(r: *Rows) void {
        r.buf.free();
    }

    /// The bytes `n` rows take from the buffer's start (one broadcast of the planned spans).
    pub fn bytes(r: *const Rows, n: u32) usize {
        return @as(usize, n) * r.hidden * 2;
    }

    /// Where span `s`'s rows start.
    pub fn at(r: *const Rows, s: Span) u64 {
        return r.buf.ptr + @as(u64, s.row) * r.hidden * 2;
    }

    /// Prompt rows [lo, hi) embedded at `emb` (row lo first) take the image rows of the spans they cross.
    pub fn splice(r: *const Rows, s: cuda.Stream, spans: []const Span, lo: u32, hi: u32, emb: u64) !void {
        const w = r.hidden * 2;
        for (spans) |sp| {
            if (sp.row + sp.n > r.limit) return error.ImageRows;
            const from, const row, const n = cut(sp, lo, hi) orelse continue;
            const src = r.buf.ptr + @as(u64, row) * w;
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(emb + @as(u64, from - lo) * w, src, @as(usize, n) * w, s.handle), "cuMemcpyDtoDAsync");
        }
    }
};

/// The part of `sp` prompt rows [lo, hi) cross: its first prompt row, its first buffer row, its rows; null for none.
pub fn cut(sp: Span, lo: u32, hi: u32) ?[3]u32 {
    const from = @max(lo, sp.at);
    const to = @min(hi, sp.at + sp.n);
    return if (from < to) .{ from, sp.row + from - sp.at, to - from } else null;
}

test "a pass from a restored point plans only the images that end past it, rows back to back" {
    const a = std.testing.allocator;
    const imgs = [_]Image{
        .{ .pixels = &.{}, .gh = 32, .gw = 32, .at = 5, .tokens = 256 },
        .{ .pixels = &.{}, .gh = 40, .gw = 30, .at = 300, .tokens = 300 },
        .{ .pixels = &.{}, .gh = 32, .gw = 32, .at = 700, .tokens = 256 },
    };
    const all = try plan(a, &imgs, 0);
    defer a.free(all);
    try std.testing.expectEqualSlices(Span, &.{ .{ .at = 5, .n = 256, .row = 0, .image = 0 }, .{ .at = 300, .n = 300, .row = 256, .image = 1 }, .{ .at = 700, .n = 256, .row = 556, .image = 2 } }, all);
    const later = try plan(a, &imgs, 400);
    defer a.free(later);
    try std.testing.expectEqualSlices(Span, &.{ .{ .at = 300, .n = 300, .row = 0, .image = 1 }, .{ .at = 700, .n = 256, .row = 300, .image = 2 } }, later);
    try std.testing.expectEqual(@as(u32, 556), total(later));
    const overlapping = [_]Image{ imgs[1], imgs[0] };
    try std.testing.expectError(error.ImageSpans, plan(a, &overlapping, 0));
}

test "a chunk takes the rows it crosses: a span cut at both ends, and none past it" {
    const first: Span = .{ .at = 5, .n = 256, .row = 0, .image = 0 };
    const second: Span = .{ .at = 300, .n = 300, .row = 256, .image = 1 };
    try std.testing.expectEqual([3]u32{ 200, 195, 61 }, cut(first, 200, 400).?);
    try std.testing.expectEqual([3]u32{ 300, 256, 100 }, cut(second, 200, 400).?);
    try std.testing.expectEqual([3]u32{ 5, 0, 256 }, cut(first, 0, 1000).?);
    try std.testing.expect(cut(first, 261, 300) == null and cut(second, 600, 700) == null);
}
