//! An image prompt's rows for the CUDA engine: its images through the tower, its M-RoPE positions (get_rope_index).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const vision = @import("cuda_vision.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Config = vision.Config;
const Tower = vision.Tower;
const Grid = vision.Grid;
const merge = vision.merge;
const patch_width = vision.patch_width;
const max_patches = vision.max_patches;

/// An image prompt's encoded rows: the tower's features on the device, each prompt row's feature row and positions.
pub const Media = struct {
    gpa: std.mem.Allocator,
    features: DeviceBuffer, // bf16 [image rows, out_hidden]
    src: []i32, // [prompt]: feature row or -1
    pos: []i32, // [3][prompt]: t, h, w
    delta: i32,

    /// `images` (the request's, in prompt order) through the tower; the prompt's positions as get_rope_index lays them.
    pub fn init(gpa: std.mem.Allocator, tower: *Tower, prompt: []const u32, images: []const lanes.Image) !Media {
        const c = tower.c;
        const n = prompt.len;
        const runs = try gpa.alloc(Run, images.len);
        defer gpa.free(runs);
        try placeholders(c, prompt, images, runs);
        var rows: usize = 0;
        for (runs) |r| rows += r.tokens();
        var m: Media = .{ .gpa = gpa, .features = try DeviceBuffer.alloc(tower.d, @max(rows, 1) * c.out_hidden * 2), .src = &.{}, .pos = &.{}, .delta = 0 };
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
            while (j < images.len and (j == i or patches + runs[j].patches() <= max_patches)) : (j += 1) patches += runs[j].patches();
            const grids = try gpa.alloc(Grid, j - i);
            defer gpa.free(grids);
            for (runs[i..j], grids) |r, *g| g.* = .{ .h = r.gh, .w = r.gw };
            if (j - i == 1) {
                try tower.encode(images[i].patches, grids, m.features.ptr + done * c.out_hidden * 2);
            } else {
                const px = try gpa.alloc(f32, patches * patch_width);
                defer gpa.free(px);
                var at: usize = 0;
                for (images[i..j]) |img| {
                    @memcpy(px[at..][0..img.patches.len], img.patches);
                    at += img.patches.len;
                }
                try tower.encode(px, grids, m.features.ptr + done * c.out_hidden * 2);
            }
            for (runs[i..j]) |r| done += r.tokens();
            i = j;
        }
        m.delta = layout(n, runs, m.src, m.pos);
        return m;
    }

    pub fn deinit(m: *Media) void {
        m.features.free();
        m.gpa.free(m.src);
        m.gpa.free(m.pos);
    }
};

/// An image's placeholder run in the prompt: its first row and its patch grid.
pub const Run = struct {
    at: usize,
    gh: u32,
    gw: u32,

    pub fn tokens(r: Run) usize {
        return @as(usize, r.gh / merge) * (r.gw / merge);
    }

    pub fn patches(r: Run) usize {
        return @as(usize, r.gh) * r.gw;
    }
};

/// Each image's run of image_token rows between vision_start and vision_end, in order; videos are refused by name.
pub fn placeholders(c: Config, prompt: []const u32, images: []const lanes.Image, runs: []Run) !void {
    var k: usize = 0;
    var r: usize = 0;
    while (r < prompt.len) : (r += 1) {
        const t = prompt[r];
        if (t == c.video_token) return error.QwenVideoUnsupported;
        if (t != c.image_token) continue;
        if (r == 0 or prompt[r - 1] != c.vision_start) return error.BadImagePlaceholders;
        if (k == images.len) return error.BadImagePlaceholders;
        const img = images[k];
        if (img.grid[0] != 1) return error.QwenVideoUnsupported;
        if (img.grid[1] <= 0 or img.grid[2] <= 0 or img.grid[1] > 4096 or img.grid[2] > 4096) return error.BadImageGrid;
        const run: Run = .{ .at = r, .gh = @intCast(img.grid[1]), .gw = @intCast(img.grid[2]) };
        if (run.gh % merge != 0 or run.gw % merge != 0 or run.patches() > max_patches) return error.BadImageGrid;
        if (img.patches.len != run.patches() * patch_width) return error.BadImagePatches;
        const end = r + run.tokens();
        if (end >= prompt.len or prompt[end] != c.vision_end) return error.BadImagePlaceholders;
        for (prompt[r..end]) |x| if (x != c.image_token) return error.BadImagePlaceholders;
        runs[k] = run;
        k += 1;
        r = end;
    }
    if (k != images.len) return error.BadImagePlaceholders;
}

/// The prompt's rotary positions ([3][n]) and feature rows (src, -1 for text) per get_rope_index; returns rope_delta.
pub fn layout(n: usize, runs: []const Run, src: []i32, pos: []i32) i32 {
    @memset(src, -1);
    var next: i32 = 0;
    var cursor: usize = 0;
    var row: i32 = 0;
    for (runs) |img| {
        const begin = img.at;
        const h: usize = img.gh / merge;
        const w: usize = img.gw / merge;
        for (cursor..begin) |r| {
            const v = next + @as(i32, @intCast(r - cursor));
            for (0..3) |a| pos[a * n + r] = v;
        }
        const base = next + @as(i32, @intCast(begin - cursor));
        for (0..h * w) |k| {
            const r = begin + k;
            pos[r] = base;
            pos[n + r] = base + @as(i32, @intCast(k / w));
            pos[2 * n + r] = base + @as(i32, @intCast(k % w));
            src[r] = row;
            row += 1;
        }
        next = base + @as(i32, @intCast(@max(h, w)));
        cursor = begin + h * w;
    }
    for (cursor..n) |r| {
        const v = next + @as(i32, @intCast(r - cursor));
        for (0..3) |a| pos[a * n + r] = v;
    }
    return next - @as(i32, @intCast(cursor));
}

const test_config: Config = .{ .hidden = 1152, .heads = 16, .depth = 27, .intermediate = 4304, .out_hidden = 8, .image_token = 50, .video_token = 51, .vision_start = 52, .vision_end = 53 };

test "media positions: text counts up, an image sits at one position with its grid, text resumes past its side" {
    const prompt = [_]u32{ 7, 52, 50, 50, 50, 50, 50, 50, 53, 9 };
    const px: [24 * patch_width]f32 = @splat(0);
    var runs: [1]Run = undefined;
    try placeholders(test_config, &prompt, &.{.{ .patches = &px, .grid = .{ 1, 4, 6 } }}, &runs);
    try std.testing.expectEqual(@as(usize, 2), runs[0].at);
    var src: [10]i32 = undefined;
    var pos: [30]i32 = undefined;
    const delta = layout(prompt.len, &runs, &src, &pos);
    try std.testing.expectEqual(@as(i32, -3), delta);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 2, 2, 2, 5, 6 }, pos[0..10]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 2, 2, 3, 3, 3, 5, 6 }, pos[10..20]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 2, 3, 4, 5, 6 }, pos[20..30]);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, 0, 1, 2, 3, 4, 5, -1, -1 }, &src);
}

test "placeholders refuse a run of the wrong length, a missing image, an extra image and a video" {
    const px: [24 * patch_width]f32 = @splat(0);
    const img: lanes.Image = .{ .patches = &px, .grid = .{ 1, 4, 6 } };
    var runs: [2]Run = undefined;
    try std.testing.expectError(error.BadImagePlaceholders, placeholders(test_config, &.{ 52, 50, 50, 53 }, &.{img}, runs[0..1]));
    try std.testing.expectError(error.BadImagePlaceholders, placeholders(test_config, &.{ 52, 50, 50, 50, 50, 50, 50, 53 }, &.{ img, img }, &runs));
    try std.testing.expectError(error.BadImagePlaceholders, placeholders(test_config, &.{ 7, 8 }, &.{img}, runs[0..1]));
    try std.testing.expectError(error.QwenVideoUnsupported, placeholders(test_config, &.{ 52, 51, 53 }, &.{}, runs[0..0]));
    try std.testing.expectError(error.BadImagePlaceholders, placeholders(test_config, &.{ 7, 50 }, &.{img}, runs[0..1]));
}
