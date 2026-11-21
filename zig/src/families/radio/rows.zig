//! Image rows for the prompt: which images a pass needs, one device buffer of their rows, the splice into embeddings.
const std = @import("std");
const cuda = @import("cuda");
const Image = @import("lanes").stream.Image;
const image_rows = @import("core").image_rows;
const Allocator = std.mem.Allocator;

pub const Span = image_rows.Span;
pub const total = image_rows.total;

/// The images a prompt pass from `from` reads, in prompt order, their rows back to back from row 0.
pub fn plan(a: Allocator, images: []const Image, from: u32) ![]Span {
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(a);
    var row: u32 = 0;
    var end: u32 = 0;
    for (images, 0..) |img, i| {
        if (img.at < end or img.tokens == 0) return error.ImageSpans; // in order, apart, each with rows
        end = std.math.add(u32, img.at, img.tokens) catch return error.ImageSpans;
        if (end <= from) continue; // a restored prefix holds these rows' states already
        try out.append(a, .{ .at = img.at, .n = img.tokens, .row = row, .image = @intCast(i) });
        row = std.math.add(u32, row, img.tokens) catch return error.ImageSpans;
    }
    return out.toOwnedSlice(a);
}

/// bf16 [limit, hidden] row-major: a pass's image rows, filled on rank 0 and broadcast whole to the others.
pub const Rows = struct {
    d: *const cuda.Driver,
    buf: cuda.DeviceBuffer,
    hidden: usize,
    limit: u32,

    pub fn init(d: *const cuda.Driver, limit: u32, hidden: usize) !Rows {
        if (limit == 0 or hidden == 0) return error.ImageRows;
        const elements = std.math.mul(usize, limit, hidden) catch return error.ImageRows;
        const row_bytes = std.math.mul(usize, elements, 2) catch return error.ImageRows;
        return .{ .d = d, .buf = try cuda.DeviceBuffer.alloc(d, row_bytes), .hidden = hidden, .limit = limit };
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
            const c = image_rows.cut(sp, lo, hi) orelse continue;
            const src = r.buf.ptr + @as(u64, c.row) * w;
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(emb + @as(u64, c.at - lo) * w, src, c.n * w, s.handle), "cuMemcpyDtoDAsync");
        }
    }
};

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
