//! Image geometry and layout as DeepSeek's image_processor.py has them: resize plan, patches, span layout.
const std = @import("std");
const vision = @import("vision.zig");

/// The first virtual id of an image position: text ids sit below it, and the Engram hash reads it as dead.
pub const vbase: u32 = 1 << 24;

pub fn isVid(id: u32) bool {
    return id >= vbase;
}

/// What the aligner makes of a patch grid: ceil(patches / ratio) a side.
pub fn llmGrid(gh: u32, gw: u32, ratio: u32) [2]u32 {
    return .{ (gh + ratio - 1) / ratio, (gw + ratio - 1) / ratio };
}

/// Tokens of an image's span: start, each row of aligner columns then a newline, end.
pub fn spanTokens(lh: u32, lw: u32) u32 {
    return lh * (lw + 1) + 2;
}

/// What a span position holds: the start delimiter, an aligner row, a row's newline, or the end delimiter.
pub const Kind = enum { start, row, newline, end };

/// Span position `p` of an image whose aligner grid is lh x lw: its kind and, for a row, its index in reading order.
pub fn kindAt(p: u32, lh: u32, lw: u32) struct { Kind, u32 } {
    if (p == 0) return .{ .start, 0 };
    if (p == spanTokens(lh, lw) - 1) return .{ .end, 0 };
    const q = p - 1;
    const c = q % (lw + 1);
    if (c == lw) return .{ .newline, 0 };
    return .{ .row, (q / (lw + 1)) * lw + c };
}

/// A virtual id for position `p` of an image: 31 bits of the image's hash, mixed with the position, above `vbase`.
pub fn vid(hash: u64, p: u32) u32 {
    var h = std.hash.Wyhash.init(0x6d6167);
    h.update(std.mem.asBytes(&hash));
    h.update(std.mem.asBytes(&p));
    const x: u32 = @truncate(h.final());
    return vbase + x % (std.math.maxInt(u32) - vbase);
}

/// The pixel size an image is padded to and its grids.
pub const Plan = struct { height: u32, width: u32, vit_h: u32, vit_w: u32, llm_h: u32, llm_w: u32 };

fn ceilDiv(x: u32, y: u32) u32 {
    return (x + y - 1) / y;
}

/// The largest aspect-preserving pixel size (height, width) whose token grid fits in `max_n`.
fn solve(height: f64, width: f64, patch: u32, ratio: u32, max_n: u32) [2]u32 {
    const r = height / width;
    const max_w = @sqrt(@as(f64, @floatFromInt(max_n - 2)) / r + 0.25) - 0.5;
    const max_h = max_w * r;
    const cell: u32 = patch * ratio;
    if (max_w < 1.0) return .{ (max_n - 2) / 2 * cell, cell };
    if (max_h < 1.0) return .{ cell, (max_n - 3) * cell };
    const c: f64 = @floatFromInt(cell);
    const beta = @min(@floor(max_w) * c / width, @floor(max_h) * c / height);
    const p: f64 = @floatFromInt(patch);
    return .{ @as(u32, @intFromFloat(@floor(height * beta / p))) * patch, @as(u32, @intFromFloat(@floor(width * beta / p))) * patch };
}

/// plan_image_grid: the pixel size for an image of `width` x `height`, its ViT patch grid and its aligner grid.
pub fn plan(width_in: u32, height_in: u32, cfg: vision.Config) Plan {
    var w: f64 = @floatFromInt(width_in);
    var h: f64 = @floatFromInt(height_in);
    if (cfg.max_wh_ratio) |r| if (w > h * @as(f64, @floatFromInt(r))) {
        w = h * @as(f64, @floatFromInt(r));
    };
    if (w * h > 0 and w * h < @as(f64, @floatFromInt(cfg.min_pixels))) {
        const ratio = @sqrt(@as(f64, @floatFromInt(cfg.min_pixels)) / (w * h));
        w = @floatFromInt(@as(u64, @intFromFloat(w * ratio)));
        h = @floatFromInt(@as(u64, @intFromFloat(h * ratio)));
    }
    const p = cfg.patch;
    var bw: u32 = @as(u32, @intFromFloat(@ceil(w / @as(f64, @floatFromInt(p))))) * p;
    var bh: u32 = @as(u32, @intFromFloat(@ceil(h / @as(f64, @floatFromInt(p))))) * p;
    var g = llmGrid(bh / p, bw / p, cfg.ratio);
    if (spanTokens(g[0], g[1]) > cfg.max_tokens) {
        const s = solve(h, w, p, cfg.ratio, cfg.max_tokens);
        bh = s[0];
        bw = s[1];
        g = llmGrid(bh / p, bw / p, cfg.ratio);
    }
    return .{ .height = bh, .width = bw, .vit_h = bh / p, .vit_w = bw / p, .llm_h = g[0], .llm_w = g[1] };
}

/// Python's round(): halves to the even integer.
pub fn roundEven(x: f64) u32 {
    const f = @floor(x);
    const d = x - f;
    const i: u32 = @intFromFloat(f);
    if (d > 0.5) return i + 1;
    if (d < 0.5) return i;
    return if (i % 2 == 0) i else i + 1;
}

/// ImageOps.contain's size: the image fitted inside `bw` x `bh`, aspect kept.
pub fn contain(w: u32, h: u32, bw: u32, bh: u32) [2]u32 {
    const im: f64 = @as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h));
    const dest: f64 = @as(f64, @floatFromInt(bw)) / @as(f64, @floatFromInt(bh));
    if (im == dest) return .{ bw, bh };
    if (im > dest) {
        const nh = roundEven(@as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(w)) * @as(f64, @floatFromInt(bw)));
        return .{ bw, if (nh != bh) nh else bh };
    }
    const nw = roundEven(@as(f64, @floatFromInt(w)) / @as(f64, @floatFromInt(h)) * @as(f64, @floatFromInt(bh)));
    return .{ if (nw != bw) nw else bw, bh };
}

/// The offset ImageOps.pad pastes at along the one axis that is short: round(gap * 0.5).
pub fn padOffset(gap: u32) u32 {
    return roundEven(@as(f64, @floatFromInt(gap)) * 0.5);
}

fn bf16Value(v: f32) f32 {
    return @as(f32, @bitCast(@as(u32, @import("weights.zig").bf16Round(v)) << 16));
}

/// The ViT's patches [n_h * n_w][3 * p * p] of RGB bytes: (v / 255 - 0.5) / 0.5 in fp32, rounded to bf16.
pub fn patches(rgb: []const u8, height: u32, width: u32, p: u32, out: []f32) void {
    const nh = height / p;
    const nw = width / p;
    std.debug.assert(rgb.len == @as(usize, height) * width * 3 and out.len == @as(usize, nh) * nw * 3 * p * p);
    for (0..nh) |by| for (0..nw) |bx| for (0..3) |c| for (0..p) |y| for (0..p) |x| {
        const v: f32 = @floatFromInt(rgb[((by * p + y) * width + bx * p + x) * 3 + c]);
        const t = ((v / 255.0) - 0.5) / 0.5;
        out[((by * nw + bx) * 3 + c) * p * p + y * p + x] = bf16Value(t);
    };
}

test "the span's layout: start, rows with a newline each, end" {
    try std.testing.expectEqual(@as(u32, 2 * 4 + 2), spanTokens(2, 3));
    const k = [_]Kind{ .start, .row, .row, .row, .newline, .row, .row, .row, .newline, .end };
    for (k, 0..) |want, p| try std.testing.expectEqual(want, kindAt(@intCast(p), 2, 3)[0]);
    try std.testing.expectEqual(@as(u32, 4), kindAt(6, 2, 3)[1]);
}

test "the plan: a small image grows to the minimum pixels, a large one shrinks to the token budget" {
    const cfg: vision.Config = .{};
    const small = plan(100, 100, cfg);
    try std.testing.expect(@as(u64, small.height) * small.width >= cfg.min_pixels);
    try std.testing.expectEqual(small.height % 14, 0);
    const big = plan(4000, 3000, cfg);
    try std.testing.expect(spanTokens(big.llm_h, big.llm_w) <= cfg.max_tokens);
    try std.testing.expectEqual(llmGrid(big.vit_h, big.vit_w, 3), [2]u32{ big.llm_h, big.llm_w });
    const tall = plan(10, 20000, cfg);
    try std.testing.expect(spanTokens(tall.llm_h, tall.llm_w) <= cfg.max_tokens);
}

test "Python's round: halves go to even, contain and pad follow" {
    try std.testing.expectEqual(@as(u32, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(u32, 4), roundEven(3.5));
    try std.testing.expectEqual(@as(u32, 3), roundEven(3.2));
    try std.testing.expectEqual([2]u32{ 100, 50 }, contain(400, 200, 100, 100));
    try std.testing.expectEqual([2]u32{ 50, 100 }, contain(200, 400, 100, 100));
    try std.testing.expectEqual(@as(u32, 2), padOffset(5));
}

test "virtual ids sit above the base and differ by image and position" {
    try std.testing.expect(vid(1, 0) >= vbase and vid(1, 5) >= vbase);
    try std.testing.expect(vid(1, 0) != vid(2, 0) and vid(1, 0) != vid(1, 1));
    try std.testing.expectEqual(vid(7, 3), vid(7, 3));
}
