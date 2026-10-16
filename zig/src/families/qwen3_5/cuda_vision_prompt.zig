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

    /// `images` (the request's, in prompt order) through the tower; the prompt's positions as get_rope_index lays them.
    pub fn init(gpa: std.mem.Allocator, v: *vision.Vision, prompt: []const u32, images: []const Image) !Media {
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
        // the tower: runs of whole images within one call's patches, in prompt order
        var done: usize = 0; // feature rows written
        var i: usize = 0;
        while (i < images.len) {
            var j = i;
            var patches: usize = 0;
            while (j < images.len) : (j += 1) {
                const more = @as(usize, images[j].gh) * images[j].gw;
                if (j > i and patches + more > max_patches) break;
                patches += more;
            }
            const grids = try gpa.alloc(Grid, j - i);
            defer gpa.free(grids);
            for (images[i..j], grids) |img, *g| g.* = .{ .h = img.gh, .w = img.gw };
            const out = m.features.ptr + done * c.out_hidden * 2;
            if (j - i == 1) {
                try v.tower.encode(images[i].pixels, grids, out);
            } else {
                const px = try gpa.alloc(f32, patches * patch_width);
                defer gpa.free(px);
                var at: usize = 0;
                for (images[i..j]) |img| {
                    @memcpy(px[at..][0..img.pixels.len], img.pixels);
                    at += img.pixels.len;
                }
                try v.tower.encode(px, grids, out);
            }
            for (images[i..j]) |img| done += img.tokens;
            i = j;
        }
        fill(&layout, images, m.src, m.pos);
        return m;
    }

    pub fn deinit(m: *Media) void {
        m.features.free();
        m.gpa.free(m.src);
        m.gpa.free(m.pos);
    }
};

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
