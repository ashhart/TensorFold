//! torchvision's processor backend, bit for bit: transformers 5.19's Lfm3VlImageProcessor on synthetic pictures.
const std = @import("std");
const patches = @import("patches.zig");

/// LiquidAI/LFM3-VL-2B (c823fade)'s processor: Qwen-VL geometry at patch 16, torchvision's resize, fused mean and std.
const lfm3_vl: patches.Patches = .{
    .patch = 16,
    .temporal = 2,
    .merge = 2,
    .min_pixels = 65536,
    .max_pixels = 16777216,
    .rescale = 0.00392156862745098,
    .mean = @splat(0.5),
    .std = @splat(0.5),
    .resize = .torch,
    .fused = true,
};

/// (x * 7 + y * 13 + 50 c + (x y) mod 31 + seed) mod 256 at (y, x, channel c).
fn synth(a: std.mem.Allocator, h: usize, w: usize, seed: usize) ![]u8 {
    const out = try a.alloc(u8, h * w * 3);
    for (0..h) |y| for (0..w) |x| for (0..3) |c| {
        out[(y * w + x) * 3 + c] = @intCast((x * 7 + y * 13 + c * 50 + (x * y) % 31 + seed) % 256);
    };
    return out;
}

fn hash(pixels: []const f32) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(pixels), &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

// sha256 of pixel_values as float32 little-endian bytes, and the grid (transformers 5.19.0, torchvision 0.29.0)
const cases = [_]struct { h: usize, w: usize, seed: usize, gh: u32, gw: u32, sha: []const u8 }{
    .{ .h = 217, .w = 300, .seed = 0, .gh = 14, .gw = 20, .sha = "53153c22eb4e065031bbdf4de0f038db37ef736a880e3438ac8e0d73f82dcea8" },
    .{ .h = 37, .w = 1000, .seed = 1, .gh = 4, .gw = 84, .sha = "ffeed836281b997e9f12e222714e9081d9baf8f24475e7be30038b3ca891dd9f" },
    .{ .h = 900, .w = 1280, .seed = 2, .gh = 56, .gw = 80, .sha = "7d2383e43dcd3442701d51525b550d484b56031cd8402013e36f68b440f88157" },
    .{ .h = 513, .w = 511, .seed = 3, .gh = 32, .gw = 32, .sha = "4b38bb33b85d99c5b8ba13bd9501488a10d1a53c7b6a6ff706b5ed6ae45173c5" },
    .{ .h = 2000, .w = 3000, .seed = 4, .gh = 124, .gw = 188, .sha = "10552216c2e4048734407caa729748e22666f297132e2dcd80ad4b0b62da093c" },
};

test "torchvision's resize and fused normalization give the processor's pixel_values, bit for bit" {
    const a = std.testing.allocator;
    for (cases) |k| {
        const px = try synth(a, k.h, k.w, k.seed);
        defer a.free(px);
        const p = try patches.prepareRgb(a, .{ .width = k.w, .height = k.h, .pixels = px }, 0, lfm3_vl, 1 << 20);
        defer a.free(p.pixels);
        try std.testing.expectEqual([2]u32{ k.gh, k.gw }, [2]u32{ p.gh, p.gw });
        try std.testing.expectEqualStrings(k.sha, &hash(p.pixels));
    }
}

test "Pillow's resize or the unfused normalization each miss the processor's bits" {
    const a = std.testing.allocator;
    const px = try synth(a, 217, 300, 0);
    defer a.free(px);
    for ([_]patches.Patches{ blk: {
        var p = lfm3_vl;
        p.resize = .pillow;
        break :blk p;
    }, blk: {
        var p = lfm3_vl;
        p.fused = false;
        break :blk p;
    } }) |settings| {
        const p = try patches.prepareRgb(a, .{ .width = 300, .height = 217, .pixels = px }, 0, settings, 1 << 20);
        defer a.free(p.pixels);
        try std.testing.expect(!std.mem.eql(u8, cases[0].sha, &hash(p.pixels)));
    }
}
