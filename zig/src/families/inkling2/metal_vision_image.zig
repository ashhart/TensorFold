//! Inkling 2's image processor (HF Inkling2ImageProcessor), host side: alpha, pad, normalize, patchify.
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const identity = picture.identity;

/// The tower's native patch side, the merge side, and so the pixels one image token covers (40 x 40).
pub const patch = 20;
pub const merge = 2;
pub const cell = patch * merge;
/// One patch's values: rows, columns, channels (HWC), the order the patch projection's weight reads.
pub const patch_width = 3 * patch * patch;
/// One token's values: its 4 patches in merge-block order (row, column), each `patch_width`.
pub const token_width = merge * merge * patch_width;

/// The processor's rescale factor and mean / std (all channels).
const rescale: f32 = 0.00392156862745098;
const mean: f32 = 0.5;
const std_dev: f32 = 0.5;
const pad_value: f32 = -1.0;

/// An image as tower input: `tokens` = gh * gw rows of the language model's, `pixels` = tokens * token_width f32.
pub const Prepared = struct {
    pixels: []f32,
    gh: u32,
    gw: u32,
    tokens: u32,
    hash: u64,
};

pub const Error = picture.Error || error{NoImageBudget};

/// The words for a refusal the server cannot word itself.
pub fn problem(err: anyerror) ?[]const u8 {
    if (err == error.NoImageBudget) return "the request has no image token budget left";
    return picture.decoderProblem(err);
}

/// Image tokens an image of this pixel size takes: its 40-pixel cells.
pub fn tokensOf(width: usize, height: usize) usize {
    return cells(height) * cells(width);
}

fn cells(n: usize) usize {
    return (n + cell - 1) / cell;
}

/// Decoded samples as oriented RGB, alpha composited over the processor's 8-pixel checkerboard of 204 and 153.
fn composite(a: std.mem.Allocator, s: picture.Samples) !picture.Rgb {
    const o = if (s.orientation >= 2 and s.orientation <= 8) s.orientation else 1;
    const turned = o >= 5;
    const w, const h = if (turned) .{ s.height, s.width } else .{ s.width, s.height };
    const out = try a.alloc(u8, w * h * 3);
    for (0..h) |y| for (0..w) |x| {
        const sx, const sy = switch (o) {
            2 => .{ s.width - 1 - x, y },
            3 => .{ s.width - 1 - x, s.height - 1 - y },
            4 => .{ x, s.height - 1 - y },
            5 => .{ y, x },
            6 => .{ y, s.height - 1 - x },
            7 => .{ s.width - 1 - y, s.height - 1 - x },
            8 => .{ s.width - 1 - y, x },
            else => .{ x, y },
        };
        const px = s.data[(sy * s.width + sx) * s.channels ..][0..s.channels];
        const alpha: u32 = if (s.channels == 4) px[3] else 255;
        const bg: u32 = if ((y / 8 + x / 8) % 2 == 0) 204 else 153;
        for (out[(y * w + x) * 3 ..][0..3], px[0..3]) |*c, v| c.* = @intCast((@as(u32, v) * alpha + bg * (255 - alpha) + 127) / 255);
    };
    return .{ .width = w, .height = h, .pixels = out };
}

/// The patches of `rgb` (w x h, three bytes a pixel) padded to whole cells, in token raster and merge-block order.
pub fn patchify(gpa: std.mem.Allocator, rgb: []const u8, w: usize, h: usize) ![]f32 {
    const rows = cells(h);
    const cols = cells(w);
    const out = try gpa.alloc(f32, rows * cols * token_width);
    errdefer gpa.free(out);
    for (0..rows) |tr| for (0..cols) |tc| {
        const token = out[(tr * cols + tc) * token_width ..][0..token_width];
        for (0..merge * merge) |q| {
            const y0 = tr * cell + (q / merge) * patch;
            const x0 = tc * cell + (q % merge) * patch;
            const dst = token[q * patch_width ..][0..patch_width];
            for (0..patch) |py| for (0..patch) |px| for (0..3) |c| {
                const y = y0 + py;
                const x = x0 + px;
                const v: f32 = if (y < h and x < w) @floatFromInt(rgb[(y * w + x) * 3 + c]) else pad_value;
                dst[(py * patch + px) * 3 + c] = (v * rescale - mean) / std_dev;
            };
        }
    };
    return out;
}

/// The largest size within `budget` tokens that keeps the aspect: only an over-budget image is resized.
fn fit(w: usize, h: usize, budget: usize) [2]usize {
    var nw = w;
    var nh = h;
    if (cells(w) * cells(h) <= budget) return .{ w, h };
    const scale = @sqrt(@as(f64, @floatFromInt(budget * cell * cell)) / @as(f64, @floatFromInt(w * h)));
    nw = @max(1, @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(w)) * scale))));
    nh = @max(1, @as(usize, @intFromFloat(@floor(@as(f64, @floatFromInt(h)) * scale))));
    while (cells(nw) * cells(nh) > budget) {
        if (nw >= nh and nw > 1) nw -= 1 else if (nh > 1) nh -= 1 else break;
    }
    return .{ nw, nh };
}

/// Oriented RGB to tower input within `max_tokens`; the hash covers `bytes_hash` and the grid.
pub fn fromRgb(gpa: std.mem.Allocator, rgb: picture.Rgb, bytes_hash: u64, max_tokens: u32) !Prepared {
    if (max_tokens == 0) return error.NoImageBudget;
    const size = fit(rgb.width, rgb.height, max_tokens);
    var resized: ?[]u8 = null;
    defer if (resized) |r| gpa.free(r);
    var pixels = rgb.pixels;
    if (size[0] != rgb.width or size[1] != rgb.height) {
        resized = try picture.resample.resize(gpa, .torch, .of(rgb.pixels, rgb.width, rgb.height), size[0], size[1]);
        pixels = resized.?;
    }
    const gh: u32 = @intCast(cells(size[1]));
    const gw: u32 = @intCast(cells(size[0]));
    const out = try patchify(gpa, pixels, size[0], size[1]);
    return .{ .pixels = out, .gh = gh, .gw = gw, .tokens = gh * gw, .hash = identity.withGrid(bytes_hash, .{ gh, gw }) };
}

/// Image bytes (PNG, JPEG, WebP) to tower input on the HTTP thread, at most `max_tokens` tokens.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !Prepared {
    if (max_tokens == 0) return error.NoImageBudget;
    const samples = switch (picture.sniff(bytes)) {
        .png => try picture.png.decodeWithLimits(gpa, bytes, .{}),
        .jpeg => try picture.jpeg.decodeWithLimits(io, gpa, bytes, .{}),
        .webp => try picture.webp.decodeWithLimits(io, gpa, bytes, .{}),
        .unknown => return error.Invalid,
    };
    defer gpa.free(samples.data);
    const hash = identity.ofBytes(0x1c2d, bytes);
    if (samples.channels == 3 and (samples.orientation <= 1 or samples.orientation > 8)) {
        return fromRgb(gpa, .{ .width = samples.width, .height = samples.height, .pixels = samples.data }, hash, max_tokens);
    }
    var rgb = try composite(gpa, samples);
    defer rgb.deinit(gpa);
    return fromRgb(gpa, rgb, hash, max_tokens);
}
