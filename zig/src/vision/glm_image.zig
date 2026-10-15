//! GLM-5.3-Flash's image and video preparation as mlx-vlm's glm5_next/processing.py: canvas, resize, norm, patches.

const std = @import("std");
const qi = @import("qwen_image.zig");

pub const PATCH = 14;
pub const MERGE = 2;
pub const TEMPORAL = 2;
pub const FACTOR = PATCH * MERGE;
const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
const PER = 3 * TEMPORAL * PATCH * PATCH;

pub const Error = qi.Error;

/// Python's round(): halves to even.
fn pyRound(x: f64) f64 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

fn alignUp(v: f64, f: f64) f64 {
    return @ceil(v / f) * f;
}

/// smart_resize: the canvas (height, width) of ``frames`` frames within [min_tokens, max_tokens] merged tokens.
pub fn smartResize(frames: u32, height: u32, width: u32, min_tokens: u64, max_tokens: u64) Error![2]u32 {
    if (frames == 0 or height == 0 or width == 0 or min_tokens == 0 or min_tokens > max_tokens) return error.UnsupportedImage;
    const f: f64 = FACTOR;
    const ppt: f64 = TEMPORAL * f * f;
    const min_px = @as(f64, @floatFromInt(min_tokens)) * ppt;
    const max_px = @as(f64, @floatFromInt(max_tokens)) * ppt;
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const n: f64 = @floatFromInt(frames);
    const af = @max(@as(f64, TEMPORAL), pyRound(n / TEMPORAL) * TEMPORAL);
    var ah = alignUp(h, f);
    var aw = alignUp(w, f);
    var px = af * ah * aw;
    if (px < min_px) {
        const scale = @sqrt(min_px / (n * h * w));
        ah = alignUp(@max(1, @ceil(h * scale)), f);
        aw = alignUp(@max(1, @ceil(w * scale)), f);
        px = af * ah * aw;
    }
    if (px > max_px) { // the tallest content within the budget, by bisection on its height
        if (max_px < af * f * f) return error.UnsupportedImage;
        var low: f64 = 1;
        var high: f64 = h;
        var best: [2]f64 = .{ f, f };
        while (low <= high) {
            const ch = @floor((low + high) / 2);
            const cw = @max(1, @floor(w * ch / h));
            const cand = [2]f64{ alignUp(ch, f), alignUp(cw, f) };
            if (af * cand[0] * cand[1] <= max_px) {
                best = cand;
                low = ch + 1;
            } else high = ch - 1;
        }
        ah = best[0];
        aw = best[1];
    }
    return .{ @intFromFloat(ah), @intFromFloat(aw) };
}

/// _resize_geometry: the canvas (th, tw) and the content (ch, cw) resized into its top-left corner.
pub fn geometry(frames: u32, height: u32, width: u32, min_tokens: u64, max_tokens: u64) Error![4]u32 {
    const t = try smartResize(frames, height, width, min_tokens, max_tokens);
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    var scale = @min(@as(f64, @floatFromInt(t[0])) / h, @as(f64, @floatFromInt(t[1])) / w);
    const ppt: f64 = TEMPORAL * FACTOR * FACTOR;
    if (@as(f64, @floatFromInt(frames)) * h * w >= @as(f64, @floatFromInt(min_tokens)) * ppt) scale = @min(1.0, scale);
    const ch: u32 = @intFromFloat(@max(1, @min(@as(f64, @floatFromInt(t[0])), @floor(h * scale))));
    const cw: u32 = @intFromFloat(@max(1, @min(@as(f64, @floatFromInt(t[1])), @floor(w * scale))));
    return .{ t[0], t[1], ch, cw };
}

pub const Prepared = struct {
    patches: []f32, // [gh * gw, 1176]
    grid: [3]i64,

    pub fn tokens(p: Prepared) usize {
        return @intCast(@divExact(p.grid[1] * p.grid[2], MERGE * MERGE));
    }

    pub fn deinit(p: *Prepared, gpa: std.mem.Allocator) void {
        gpa.free(p.patches);
    }
};

fn norm(v: u8, c: usize) f32 {
    const x = @as(f32, @floatFromInt(v)) * @as(f32, @floatCast(@as(f64, 1.0) / 255.0));
    return (x - MEAN[c]) / STD[c];
}

/// A frame's normalized value at canvas (y, x) (zero padding past the content's ch x cw).
fn at(frame: []const u8, cw: usize, ch: usize, y: usize, x: usize, c: usize) f32 {
    return norm(if (y < ch and x < cw) frame[(y * cw + x) * 3 + c] else 0, c);
}

/// ``im`` on its canvas within [min_tokens, max_tokens] tokens, normalized and cut into patches.
pub fn prepare(gpa: std.mem.Allocator, im: qi.Image, min_tokens: u64, max_tokens: u64) Error!Prepared {
    const g = try geometry(TEMPORAL, im.h, im.w, min_tokens, max_tokens);
    var r = if (g[2] == im.h and g[3] == im.w) null else try qi.resize(gpa, im, g[3], g[2]);
    defer if (r) |*x| x.deinit(gpa);
    const src = if (r) |x| x.rgb else im.rgb;
    const gh: usize = g[0] / PATCH;
    const gw: usize = g[1] / PATCH;
    const out = try gpa.alloc(f32, gh * gw * PER);
    var n: usize = 0;
    for (0..gh / MERGE) |bh| for (0..gw / MERGE) |bw| for (0..MERGE) |mh| for (0..MERGE) |mw| {
        const py = (bh * MERGE + mh) * PATCH;
        const px = (bw * MERGE + mw) * PATCH;
        const dst = out[n * PER ..][0..PER];
        for (0..3) |c| for (0..PATCH) |y| for (0..PATCH) |x| {
            const v = at(src, g[3], g[2], py + y, px + x, c);
            for (0..TEMPORAL) |t| dst[((c * TEMPORAL + t) * PATCH + y) * PATCH + x] = v;
        };
        n += 1;
    };
    return .{ .patches = out, .grid = .{ 1, @intCast(gh), @intCast(gw) } };
}

/// A video's frame groups: patches of two consecutive frames (an odd last repeats) and the first frame's seconds.
pub const Video = struct {
    patches: []f32, // groups x gh x gw x 1176
    groups: usize,
    gh: usize,
    gw: usize,
    times: []f64,

    pub fn groupPatches(v: Video, g: usize) []f32 {
        const per = v.gh * v.gw * PER;
        return v.patches[g * per ..][0..per];
    }

    pub fn tokens(v: Video) usize {
        return v.gh * v.gw / (MERGE * MERGE);
    }

    pub fn deinit(v: *Video, gpa: std.mem.Allocator) void {
        gpa.free(v.patches);
        gpa.free(v.times);
    }
};

/// ``rgb``: ``frames`` RGB frames at the content size (ch x cw: geometry's), placed on the th x tw canvas.
pub fn prepareVideo(gpa: std.mem.Allocator, rgb: []const u8, frames: usize, ch: usize, cw: usize, th: usize, tw: usize, indices: []const i32, fps: f64) Error!Video {
    if (th % FACTOR != 0 or tw % FACTOR != 0 or ch > th or cw > tw or frames < 1) return error.UnsupportedImage;
    const groups = (frames + TEMPORAL - 1) / TEMPORAL;
    const gh = th / PATCH;
    const gw = tw / PATCH;
    const out = try gpa.alloc(f32, groups * gh * gw * PER);
    errdefer gpa.free(out);
    var n: usize = 0;
    for (0..groups) |g| {
        for (0..gh / MERGE) |bh| for (0..gw / MERGE) |bw| for (0..MERGE) |mh| for (0..MERGE) |mw| {
            const py = (bh * MERGE + mh) * PATCH;
            const px = (bw * MERGE + mw) * PATCH;
            const dst = out[n * PER ..][0..PER];
            for (0..3) |c| for (0..TEMPORAL) |t| {
                const f = @min(g * TEMPORAL + t, frames - 1);
                const frame = rgb[f * ch * cw * 3 ..][0 .. ch * cw * 3];
                for (0..PATCH) |y| for (0..PATCH) |x| {
                    dst[((c * TEMPORAL + t) * PATCH + y) * PATCH + x] = at(frame, cw, ch, py + y, px + x, c);
                };
            };
            n += 1;
        };
    }
    const times = try gpa.alloc(f64, groups);
    for (0..groups) |g| times[g] = @as(f64, @floatFromInt(indices[g * TEMPORAL])) / fps;
    return .{ .patches = out, .groups = groups, .gh = gh, .gw = gw, .times = times };
}

test "smart_resize and geometry match the processor's" {
    // 640 x 480 within 8000 tokens: aligned to 504 x 644, content unscaled
    try std.testing.expectEqual([4]u32{ 504, 644, 480, 640 }, try geometry(TEMPORAL, 480, 640, 16, 8000));
    // 40 x 30 below 16 tokens: scaled up, content scaled
    const g = try geometry(TEMPORAL, 30, 40, 16, 8000);
    try std.testing.expectEqual(@as(u32, 112), g[0]);
    try std.testing.expectEqual(@as(u32, 140), g[1]);
    // 300 x 900 (tall)
    const t = try geometry(TEMPORAL, 900, 300, 16, 8000);
    try std.testing.expectEqual([2]u32{ 924, 308 }, .{ t[0], t[1] });
}
