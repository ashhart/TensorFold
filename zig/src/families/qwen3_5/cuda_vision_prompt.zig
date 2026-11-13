//! An image prompt's rows for the CUDA engine: its images through the tower, its M-RoPE positions (core/qwen_vl).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const vision = @import("cuda_vision.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Image = lanes.stream.Image;
const Grid = vision.Grid;
const merge = vision.merge;
const patch_width = vision.patch_width;
const max_patches = vision.max_patches;
const Layout = core.qwen_vl.rope.Layout;

/// An image prompt's encoded rows: the tower's features on the device, each prompt row's feature row and positions.
pub const Media = struct {
    gpa: std.mem.Allocator,
    features: DeviceBuffer, // bf16 [image rows, out_hidden]
    src: []i32, // [prompt]: feature row or -1
    pos: []i32, // [3][prompt]: t, h, w
    delta: i32,

    /// `images` through the tower but runs a kept state at `from` holds; positions from all images (get_rope_index).
    pub fn init(gpa: std.mem.Allocator, v: *vision.Vision, prompt: []const u32, images: []const Image, from: usize) !Media {
        const layout = try Layout.init(prompt, images, v.ids(), merge);
        for (images) |img| {
            if (@as(usize, img.gh) * img.gw > max_patches) return error.BadImageGrid;
            if (img.pixels.len != @as(usize, img.gh) * img.gw * patch_width) return error.BadImagePatches;
        }
        const c = v.tower.c;
        const n = prompt.len;
        var rows: usize = 0;
        for (images) |img| rows += img.tokens;
        const features = try DeviceBuffer.alloc(v.tower.d, @max(rows, 1) * c.out_hidden * 2);
        var m: Media = .{ .gpa = gpa, .features = features, .src = &.{}, .pos = &.{}, .delta = layout.shift };
        errdefer m.features.free();
        m.src = try gpa.alloc(i32, n);
        errdefer gpa.free(m.src);
        m.pos = try gpa.alloc(i32, 3 * n);
        errdefer gpa.free(m.pos);
        const Tower = struct {
            gpa: std.mem.Allocator,
            v: *vision.Vision,
            base: u64,
            width: usize,
            fn run(t: @This(), run_images: []const Image, patches: usize, row: usize) !void {
                const grids = try t.gpa.alloc(Grid, run_images.len);
                defer t.gpa.free(grids);
                for (run_images, grids) |img, *g| g.* = .{ .h = img.gh, .w = img.gw };
                const out = t.base + row * t.width;
                if (run_images.len == 1) return t.v.tower.encode(run_images[0].pixels, grids, out);
                const px = try t.gpa.alloc(f32, patches * patch_width);
                defer t.gpa.free(px);
                var at: usize = 0;
                for (run_images) |img| {
                    @memcpy(px[at..][0..img.pixels.len], img.pixels);
                    at += img.pixels.len;
                }
                try t.v.tower.encode(px, grids, out);
            }
        };
        try runs(images, from, Tower{ .gpa = gpa, .v = v, .base = m.features.ptr, .width = c.out_hidden * 2 });
        fill(&layout, images, m.src, m.pos);
        return m;
    }

    pub fn deinit(m: *Media) void {
        m.features.free();
        m.gpa.free(m.src);
        m.gpa.free(m.pos);
    }
};

/// The tower's runs of whole images (one call's patches each), `enc.run` at their first row, but runs before `from`.
pub fn runs(images: []const Image, from: usize, enc: anytype) !void {
    var done: usize = 0; // feature rows laid out
    var i: usize = 0;
    while (i < images.len) {
        var j = i;
        var patches: usize = 0;
        while (j < images.len) : (j += 1) {
            const more = @as(usize, images[j].gh) * images[j].gw;
            if (j > i and patches + more > max_patches) break;
            patches += more;
        }
        // a run ending by `from` is in the kept state; others keep a cold pass's members (cuBLASLt picks by shape)
        const last = images[j - 1];
        if (@as(usize, last.at) + last.tokens > from) try enc.run(images[i..j], patches, done);
        for (images[i..j]) |img| done += img.tokens;
        i = j;
    }
}

/// Each prompt row's feature row (-1 for text) and its t, h and w positions ([3][n]) from `layout`.
pub fn fill(layout: *const Layout, images: []const Image, src: []i32, pos: []i32) void {
    const n = src.len;
    @memset(src, -1);
    var row: i32 = 0;
    for (images) |img| for (0..img.tokens) |k| {
        src[img.at + k] = row;
        row += 1;
    };
    for (0..n) |p| {
        const x = layout.position(@intCast(p));
        for (0..3) |a| pos[a * n + p] = x[a];
    }
}

const ids: core.qwen_vl.rope.Ids = .{ .start = 52, .image = 50, .end = 53 };

test "media positions: text counts up, an image sits at one position with its grid, text resumes past its side" {
    const prompt = [_]u32{ 7, 52, 50, 50, 50, 50, 50, 50, 53, 9 };
    const px: [24 * patch_width]f32 = @splat(0);
    const images = [_]Image{.{ .pixels = &px, .gh = 4, .gw = 6, .at = 2, .tokens = 6 }};
    const layout = try Layout.init(&prompt, &images, ids, merge);
    var src: [10]i32 = undefined;
    var pos: [30]i32 = undefined;
    fill(&layout, &images, &src, &pos);
    try std.testing.expectEqual(@as(i32, -3), layout.shift);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 2, 2, 2, 5, 6 }, pos[0..10]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 3, 3, 3, 5, 6 }, pos[10..20]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 2, 3, 4, 5, 6 }, pos[20..30]);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, 1, 2, 3, 4, 5, -1, -1 }, &src);
}

test "two images: feature rows follow prompt order, and the text after each shifts" {
    const prompt = [_]u32{ 52, 50, 50, 50, 53, 8, 52, 50, 50, 50, 50, 53, 9 };
    const a: [12 * patch_width]f32 = @splat(0);
    const b: [16 * patch_width]f32 = @splat(0);
    const images = [_]Image{
        .{ .pixels = &a, .gh = 6, .gw = 2, .at = 1, .tokens = 3 },
        .{ .pixels = &b, .gh = 4, .gw = 4, .at = 7, .tokens = 4 },
    };
    const layout = try Layout.init(&prompt, &images, ids, merge);
    var src: [13]i32 = undefined;
    var pos: [39]i32 = undefined;
    fill(&layout, &images, &src, &pos);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2, -1, -1, -1, 3, 4, 5, 6, -1, -1 }, &src);
    try std.testing.expectEqual(@as(i32, -2), layout.shift);
    try std.testing.expectEqual(@as(i32, 10), pos[12]); // the last text row's t: index 12 shifted by -2
}

test "placeholders that do not match their images are refused before the tower" {
    const px: [24 * patch_width]f32 = @splat(0);
    const img: Image = .{ .pixels = &px, .gh = 4, .gw = 6, .at = 1, .tokens = 6 };
    try std.testing.expectError(error.BadImagePlaceholders, Layout.init(&.{ 52, 50, 50, 53 }, &[_]Image{img}, ids, merge));
    const short: Image = .{ .pixels = &px, .gh = 4, .gw = 6, .at = 1, .tokens = 5 };
    const prompt = [_]u32{ 52, 50, 50, 50, 50, 50, 50, 53 };
    try std.testing.expectError(error.BadImagePlaceholders, Layout.init(&prompt, &[_]Image{short}, ids, merge));
}

/// A stand-in tower: each run's members and first row are recorded, and each image's rows hold its position and row.
const Runs = struct {
    rows: []u32,
    calls: usize = 0,
    firsts: [4]u32 = undefined,
    sizes: [4]usize = undefined,

    fn run(r: *Runs, run_images: []const Image, _: usize, row: usize) !void {
        r.firsts[r.calls] = run_images[0].at;
        r.sizes[r.calls] = run_images.len;
        r.calls += 1;
        var at = row;
        for (run_images) |img| for (0..img.tokens) |k| {
            r.rows[at] = img.at * 1000 + @as(u32, @intCast(k));
            at += 1;
        };
    }
};

test "a resumed image prompt skips only the tower runs its kept state holds, and splices a cold pass's rows" {
    const t = std.testing;
    const px: []const f32 = &.{};
    // two 64x128 grids fill one call's patches; the third starts a run of its own
    const images = [_]Image{
        .{ .pixels = px, .gh = 64, .gw = 128, .at = 10, .tokens = 4 },
        .{ .pixels = px, .gh = 64, .gw = 128, .at = 20, .tokens = 4 },
        .{ .pixels = px, .gh = 64, .gw = 128, .at = 40, .tokens = 4 },
    };
    var cold_rows: [12]u32 = @splat(0xdead);
    var cold: Runs = .{ .rows = &cold_rows };
    try runs(&images, 0, &cold);
    try t.expectEqual(@as(usize, 2), cold.calls);
    for ([_]usize{ 30, 22 }) |from| { // past both images of the first run, then inside its second image
        var warm_rows: [12]u32 = @splat(0xdead);
        var warm: Runs = .{ .rows = &warm_rows };
        try runs(&images, from, &warm);
        // a run is skipped whole or kept whole, with a cold pass's members
        try t.expectEqual(@as(usize, if (from == 30) 1 else 2), warm.calls);
        try t.expectEqual(@as(u32, if (from == 30) 40 else 10), warm.firsts[0]);
        for (0..warm.calls) |c| try t.expectEqual(cold.sizes[cold.calls - warm.calls + c], warm.sizes[c]);
        // every row a chunk from `from` splices is a cold pass's
        var row: usize = 0;
        for (images) |img| {
            for (0..img.tokens) |k| if (img.at + k >= from) try t.expectEqual(cold_rows[row + k], warm_rows[row + k]);
            row += img.tokens;
        }
    }
}
