//! An image's bytes to Gemma 4's tower inputs, as transformers' Gemma4ImageProcessorPil makes them.
const std = @import("std");
const img = @import("../glm/image.zig");

pub const Error = img.Error || error{TooFewTokens};

/// The processor's constants: patch side, the pooling kernel, and the soft-token budgets it accepts.
pub const patch = 16;
pub const pool = 3;
pub const patch_width = 3 * patch * patch; // a patch's values: row, column, channel (input_proj's weight order)
const side_mult = pool * patch;
pub const budgets = [_]u32{ 70, 140, 280, 560, 1120 };
/// The 31B's processor_config.json max_soft_tokens: the most an image takes.
pub const max_budget = 280;
pub const Ids = struct { boi: u32 = 255999, image: u32 = 258880, eoi: u32 = 258882 };

/// An image as tower inputs (engine_api.PreparedImage's fields): its [gh * gw, patch_width] patches in [0, 1].
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };

/// get_aspect_ratio_preserving_size: the largest sides, multiples of 48, whose patches fit `max_patches`.
pub fn targetSize(height: u32, width: u32, max_patches: u32) Error![2]u32 {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const factor = @sqrt(@as(f64, @floatFromInt(max_patches * patch * patch)) / (h * w));
    var th: u32 = @intFromFloat(@floor(factor * h / side_mult) * side_mult);
    var tw: u32 = @intFromFloat(@floor(factor * w / side_mult) * side_mult);
    if (th == 0 and tw == 0) return error.ImageTooSmall;
    const longest = (max_patches / (pool * pool)) * side_mult;
    if (th == 0) {
        th = side_mult;
        tw = @min(@as(u32, @intFromFloat(@floor(w / h))) * side_mult, longest);
    } else if (tw == 0) {
        tw = side_mult;
        th = @min(@as(u32, @intFromFloat(@floor(h / w))) * side_mult, longest);
    }
    if (@as(u64, th) * tw > @as(u64, max_patches) * patch * patch) return error.ImageTooLarge;
    return .{ th, tw };
}

/// The largest budget the processor takes within `max_tokens` (the server's share an image), up to the checkpoint's.
pub fn budget(max_tokens: u32) Error!u32 {
    var best: ?u32 = null;
    for (budgets) |b| if (b <= @min(max_tokens, max_budget)) {
        best = b;
    };
    return best orelse error.TooFewTokens;
}

/// The image's bytes to its patches: resized with PIL's bicubic filter, scaled to [0, 1], patches in row order.
pub fn prepare(gpa: std.mem.Allocator, bytes: []const u8, max_tokens: u32) Error!Prepared {
    const max_patches = try budget(max_tokens) * pool * pool;
    var src = try img.decode(gpa, bytes);
    defer src.deinit(gpa);
    const size = try targetSize(src.h, src.w, max_patches);
    var scaled = if (size[0] == src.h and size[1] == src.w) src else try img.resize(gpa, src, size[1], size[0]);
    defer if (scaled.px.ptr != src.px.ptr) scaled.deinit(gpa);
    const gh = size[0] / patch;
    const gw = size[1] / patch;
    return .{ .pixels = try patches(gpa, scaled), .gh = gh, .gw = gw, .tokens = gh * gw / (pool * pool), .hash = std.hash.Wyhash.hash(0x6e34, bytes) };
}

/// convert_image_to_patches after np_rescale (float64 times 1/255, then float32): a patch's pixels in row order, RGB.
fn patches(gpa: std.mem.Allocator, src: img.Rgb) Error![]f32 {
    const gh = src.h / patch;
    const gw = src.w / patch;
    const out = try gpa.alloc(f32, @as(usize, gh) * gw * patch_width);
    for (0..gh) |py| for (0..gw) |px| for (0..patch) |y| for (0..patch) |x| for (0..3) |c| {
        const v = src.px[((py * patch + y) * src.w + px * patch + x) * 3 + c];
        out[(py * gw + px) * patch_width + (y * patch + x) * 3 + c] = @floatCast(@as(f64, @floatFromInt(v)) * (1.0 / 255.0));
    };
    return out;
}

test "target sizes as the processor sizes the fixtures" {
    const cases = [_][4]u32{ // height, width, then the processor's patch grid (rows, columns) at 280 tokens
        .{ 389, 517, 42, 57 },
        .{ 800, 1200, 39, 60 },
        .{ 1400, 180, 138, 18 },
        .{ 15, 20, 42, 57 },
        .{ 672, 672, 48, 48 },
    };
    for (cases) |c| {
        const s = try targetSize(c[0], c[1], 280 * 9);
        try std.testing.expectEqual(c[2], s[0] / patch);
        try std.testing.expectEqual(c[3], s[1] / patch);
    }
    // one side under 48 once scaled: that side is 48 and the other its whole ratio, capped
    try std.testing.expectEqual([2]u32{ 48, 48 * 280 }, try targetSize(10, 30000, 280 * 9));
    try std.testing.expectEqual([2]u32{ 48 * 280, 48 }, try targetSize(60000, 2, 280 * 9));
}

test "budgets within the server's share" {
    try std.testing.expectEqual(@as(u32, 280), try budget(4096));
    try std.testing.expectEqual(@as(u32, 140), try budget(256));
    try std.testing.expectEqual(@as(u32, 70), try budget(70));
    try std.testing.expectError(error.TooFewTokens, budget(69));
}
