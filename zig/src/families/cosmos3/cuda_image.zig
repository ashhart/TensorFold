//! image.zig and vision.zig's host half for the CUDA build (its module reads the core through `core`): Cosmos3-Edge's
//! processor (sides multiples of 32, torch's antialiased bicubic, (x - 127.5) / 127.5, 16-pixel patches in 2 x 2 block
//! order, HWC in a patch) and the learned 16 x 16 position grid resized as torch's antialiased bilinear.
const std = @import("std");
const core = @import("core");
const c = @import("config.zig");
const picture = core.picture;
const geometry = core.qwen_vl.geometry;

pub const factor = c.patch * c.merge;
pub const min_pixels: u64 = 65536;
pub const max_pixels: u64 = 16777216;

pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

pub fn decoderProblem(err: anyerror) ?[]const u8 {
    return geometry.problem(err);
}

/// Bytes (PNG, JPEG or WebP) to patches under a `max_tokens` budget.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !Prepared {
    var img = try picture.decode(io, gpa, bytes);
    defer img.deinit(gpa);
    return prepareRgb(gpa, img, picture.identity.ofBytes(0xc053, bytes), max_tokens);
}

pub fn prepareRgb(gpa: std.mem.Allocator, img: picture.Rgb, bytes_hash: u64, max_tokens: u32) !Prepared {
    if (max_tokens == 0) return error.ImageTooSmall;
    const most = @min(max_pixels, @as(u64, max_tokens) * factor * factor);
    const size = try geometry.smartResize(@intCast(img.height), @intCast(img.width), factor, @min(min_pixels, most), most);
    const resized = try picture.resample.resize(gpa, .torch, .of(img.pixels, img.width, img.height), size.w, size.h);
    defer gpa.free(resized);
    const gh = size.h / c.patch;
    const gw = size.w / c.patch;
    return .{ .pixels = try patchify(gpa, resized, size.w, gh, gw), .gh = gh, .gw = gw, .tokens = gh * gw / (c.merge * c.merge), .hash = picture.identity.withGrid(bytes_hash, .{ gh, gw }) };
}

/// Patches [gh * gw, 768]: block row, block column, row in block, column in block; each patch (y, x, channel).
pub fn patchify(gpa: std.mem.Allocator, rgb: []const u8, width: usize, gh: usize, gw: usize) ![]f32 {
    const P = c.patch;
    const M = c.merge;
    const out = try gpa.alloc(f32, gh * gw * c.patch_width);
    // transformers' torchvision backend folds the rescale into the mean and std: (x - 127.5) / 127.5 in float32
    var lut: [256]f32 = undefined;
    for (&lut, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) - 127.5) / 127.5;
    var n: usize = 0;
    for (0..gh / M) |by| for (0..gw / M) |bx| for (0..M) |my| for (0..M) |mx| {
        const y0 = (by * M + my) * P;
        const x0 = (bx * M + mx) * P;
        const dst = out[n * c.patch_width ..][0..c.patch_width];
        for (0..P) |dy| for (0..P) |dx| for (0..3) |ch| {
            dst[(dy * P + dx) * 3 + ch] = lut[rgb[((y0 + dy) * width + x0 + dx) * 3 + ch]];
        };
        n += 1;
    };
    return out;
}

/// f32 to bf16, round to nearest even (torch's .to(bfloat16)).
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7fc0;
    const rounded = b +% 0x7fff +% ((b >> 16) & 1);
    return @truncate(rounded >> 16);
}

/// torch's antialiased bilinear weights for one axis from `in` to `out` samples: each output's first input and its
/// weights, normalized, in f32 as on the CPU.
const Taps = struct { first: []usize, count: []usize, weights: []f32, width: usize };

fn taps(gpa: std.mem.Allocator, in: usize, out: usize) !Taps {
    const scale: f32 = @as(f32, @floatFromInt(in)) / @as(f32, @floatFromInt(out));
    const support: f32 = if (scale >= 1.0) scale else 1.0;
    const invscale: f32 = if (scale >= 1.0) 1.0 / scale else 1.0;
    const width: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    var t: Taps = .{ .first = try gpa.alloc(usize, out), .count = try gpa.alloc(usize, out), .weights = try gpa.alloc(f32, out * width), .width = width };
    @memset(t.weights, 0);
    for (0..out) |i| {
        const center = scale * (@as(f32, @floatFromInt(i)) + 0.5);
        const lo: i64 = @intFromFloat(@trunc(center - support + 0.5));
        const hi: i64 = @intFromFloat(@trunc(center + support + 0.5));
        const xmin: usize = @intCast(@max(lo, 0));
        const xsize: usize = @intCast(@min(hi, @as(i64, @intCast(in))) - @as(i64, @intCast(xmin)));
        var total: f32 = 0;
        const w = t.weights[i * width ..][0..width];
        for (0..xsize) |j| {
            var d = (@as(f32, @floatFromInt(j + xmin)) - center + 0.5) * invscale;
            if (d < 0) d = -d;
            w[j] = if (d < 1.0) 1.0 - d else 0.0;
            total += w[j];
        }
        if (total != 0) for (w[0..xsize]) |*x| {
            x.* /= total;
        };
        t.first[i] = xmin;
        t.count[i] = xsize;
    }
    return t;
}

fn freeTaps(gpa: std.mem.Allocator, t: Taps) void {
    gpa.free(t.first);
    gpa.free(t.count);
    gpa.free(t.weights);
}

/// The 16 x 16 grid resized to gh x gw (width pass, then height pass, f32), rounded to bf16, in block-major order.
pub fn positions(gpa: std.mem.Allocator, grid: []const f32, gh: u32, gw: u32, out: []u16) !void {
    const W = c.v_hidden;
    const G = c.grid;
    const tx = try taps(gpa, G, gw);
    defer freeTaps(gpa, tx);
    const ty = try taps(gpa, G, gh);
    defer freeTaps(gpa, ty);
    const wide = try gpa.alloc(f32, G * gw * W);
    defer gpa.free(wide);
    @memset(wide, 0);
    for (0..G) |y| for (0..gw) |x| {
        const dst = wide[(y * gw + x) * W ..][0..W];
        for (0..tx.count[x]) |j| {
            const wt = tx.weights[x * tx.width + j];
            const src = grid[(y * G + tx.first[x] + j) * W ..][0..W];
            for (dst, src) |*d, s| d.* += wt * s;
        }
    };
    const acc = try gpa.alloc(f32, W);
    defer gpa.free(acc);
    const M = c.merge;
    for (0..gh) |y| for (0..gw) |x| {
        @memset(acc, 0);
        for (0..ty.count[y]) |j| {
            const wt = ty.weights[y * ty.width + j];
            const src = wide[((ty.first[y] + j) * gw + x) * W ..][0..W];
            for (acc, src) |*d, s| d.* += wt * s;
        }
        // block-major: (block row, block column, row in block, column in block)
        const n = ((y / M) * (gw / M) + x / M) * M * M + (y % M) * M + x % M;
        for (out[n * W ..][0..W], acc) |*d, s| d.* = bf16(s);
    };
}
