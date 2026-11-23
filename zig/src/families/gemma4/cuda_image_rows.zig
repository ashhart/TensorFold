//! Gemma 4 on CUDA: the images a prompt pass encodes, and the rows each chunk splices, from where the pass resumes.
const std = @import("std");
const Image = @import("lanes").stream.Image;

/// Whether a pass resumed at `from` reads image `im`'s rows: they end past it (a resume never cuts an image).
pub fn needed(im: Image, from: usize) bool {
    return @as(usize, im.at) + im.tokens > from;
}

/// `enc.encode(im, row)` for each image the pass reads; `row` is its first row in a buffer laid out for every image.
pub fn encode(images: []const Image, from: usize, enc: anytype) !void {
    var row: usize = 0;
    for (images) |im| {
        if (needed(im, from)) try enc.encode(im, row);
        row += im.tokens;
    }
}

/// An image a chunk splices whole: its first row in the chunk, its rows, and its first row in the buffer.
pub const Splice = struct { row: usize, rows: usize, off: usize };

/// The images chunk [at, end) splices into `out`, in prompt order; how many.
pub fn splices(images: []const Image, at: usize, end: usize, out: []Splice) !usize {
    var n: usize = 0;
    var off: usize = 0;
    for (images) |im| {
        if (im.at >= at and im.at + im.tokens <= end) {
            if (n == out.len) return error.GemmaImagesInChunk;
            out[n] = .{ .row = im.at - at, .rows = im.tokens, .off = off };
            n += 1;
        }
        off += im.tokens;
    }
    return n;
}

/// A stand-in encoder: each image's rows hold its index and row, and every call is counted.
const Fake = struct {
    rows: []u32,
    calls: usize = 0,
    seen: [8]u32 = undefined,

    fn encode(f: *Fake, im: Image, row: usize) !void {
        f.seen[f.calls] = im.at;
        f.calls += 1;
        for (0..im.tokens) |r| f.rows[row + r] = im.at * 1000 + @as(u32, @intCast(r));
    }
};

fn pass(images: []const Image, from: usize, rows: []u32) !Fake {
    @memset(rows, 0xdead);
    var f: Fake = .{ .rows = rows };
    try encode(images, from, &f);
    return f;
}

test "Gemma 4 CUDA images: a resumed pass encodes only the images past its resume, splicing a cold pass's rows" {
    const t = std.testing;
    const px: []const f32 = &.{};
    const images = [_]Image{
        .{ .pixels = px, .gh = 6, .gw = 6, .at = 10, .tokens = 4 },
        .{ .pixels = px, .gh = 6, .gw = 6, .at = 20, .tokens = 4 },
        .{ .pixels = px, .gh = 6, .gw = 6, .at = 40, .tokens = 4 },
    };
    var cold_rows: [12]u32 = undefined;
    var warm_rows: [12]u32 = undefined;
    const cold = try pass(&images, 0, &cold_rows);
    try t.expectEqual(@as(usize, 3), cold.calls); // main encodes every image, resumed or not
    // a prompt resumed at 30 holds the first two images in its kept state: only the third is encoded
    const warm = try pass(&images, 30, &warm_rows);
    try t.expectEqual(@as(usize, 1), warm.calls);
    try t.expectEqual(@as(u32, 40), warm.seen[0]);
    // every chunk from the resume splices only encoded rows, and they are the cold pass's
    var out: [8]Splice = undefined;
    for ([_][2]usize{ .{ 30, 48 }, .{ 48, 64 } }) |c| {
        const n = try splices(&images, c[0], c[1], &out);
        for (out[0..n]) |sp| {
            try t.expectEqualSlices(u32, cold_rows[sp.off..][0..sp.rows], warm_rows[sp.off..][0..sp.rows]);
            try t.expect(warm_rows[sp.off] != 0xdead);
        }
    }
    // the resume right at an image's end keeps it out; one row earlier would cut it, which resumeAt refuses
    try t.expect(!needed(images[1], 24) and needed(images[1], 23));
    try t.expectEqual(@as(usize, 3), (try pass(&images, 0, &warm_rows)).calls);
}

test "Gemma 4 CUDA images: a chunk splices the images it holds whole, at their buffer rows" {
    const px: []const f32 = &.{};
    const images = [_]Image{
        .{ .pixels = px, .gh = 6, .gw = 6, .at = 10, .tokens = 4 },
        .{ .pixels = px, .gh = 6, .gw = 6, .at = 20, .tokens = 4 },
    };
    var out: [1]Splice = undefined;
    try std.testing.expectEqual(@as(usize, 1), try splices(&images, 16, 32, &out));
    try std.testing.expectEqual(Splice{ .row = 4, .rows = 4, .off = 4 }, out[0]);
    try std.testing.expectError(error.GemmaImagesInChunk, splices(&images, 0, 32, &out));
}
