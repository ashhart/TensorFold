//! image.zig for CUDA builds, line for line: an image to the 12B's merged patches (core.picture imported by name).
const std = @import("std");
const picture = @import("core").picture;

/// The shared decoder's refusals, and sizes or budgets the processor cannot take.
pub const Error = picture.Error || error{ ImageTooSmall, ImageTooLarge, TooFewTokens };
pub const decoderProblem = picture.decoderProblem;

/// The processor's constants: patch side, the pooling kernel, and the soft-token budgets it accepts.
pub const patch = 16;
pub const pool = 3;
pub const patch_width = 3 * patch * patch; // a patch's values: row, column, channel (input_proj's weight order)
const side_mult = pool * patch;
pub const budgets = [_]u32{ 70, 140, 280, 560, 1120 };
/// The 31B's processor_config.json max_soft_tokens: the most an image takes.
pub const max_budget = 280;
pub const Ids = struct { boi: u32 = 255999, image: u32 = 258880, eoi: u32 = 258882 };

/// The 12B's patches: Gemma4UnifiedImageProcessor's patches_merge of each 3 x 3 group of 16-px patches, a 48-px patch.
pub const merged_width = pool * pool * patch_width;

/// The tower's layout: 16-px patches in grid order (the 31B), or 48-px ones, each a 3 x 3 group's (the 12B).
pub const Layout = enum { patches, merged };

/// An image as tower inputs (engine_api.PreparedImage's fields): patches in [0, 1]; gh and gw count 16-px patches.
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

/// Bytes to patches (core's picture decoder, Pillow's bicubic resize, [0, 1]); the hash covers the bytes and the grid.
pub fn prepare(io: std.Io, gpa: std.mem.Allocator, bytes: []const u8, max_tokens: u32, layout: Layout) Error!Prepared {
    const max_patches = try budget(max_tokens) * pool * pool;
    var src = try picture.decode(io, gpa, bytes);
    defer src.deinit(gpa);
    const size = try targetSize(@intCast(src.height), @intCast(src.width), max_patches);
    const resized = try picture.resample.resize(gpa, .pillow, .of(src.pixels, src.width, src.height), size[1], size[0]);
    defer gpa.free(resized);
    const gh = size[0] / patch;
    const gw = size[1] / patch;
    const grid = [2]u32{ gh, gw };
    const hash = picture.identity.withGrid(picture.identity.ofBytes(if (layout == .patches) 0x6e34 else 0x6e35, bytes), grid);
    const side: usize = if (layout == .patches) patch else side_mult;
    const rgb: picture.Rgb = .{ .width = size[1], .height = size[0], .pixels = resized };
    return .{ .pixels = try patches(gpa, rgb, side), .gh = gh, .gw = gw, .tokens = gh * gw / (pool * pool), .hash = hash };
}

/// convert_image_to_patches after np_rescale at `side` px (48: patches_merge's order): pixels in row order, RGB.
fn patches(gpa: std.mem.Allocator, src: picture.Rgb, side: usize) Error![]f32 {
    const gh = src.height / side;
    const gw = src.width / side;
    const width = 3 * side * side;
    const out = try gpa.alloc(f32, gh * gw * width);
    for (0..gh) |py| for (0..gw) |px| for (0..side) |y| for (0..side) |x| for (0..3) |c| {
        const v = src.pixels[((py * side + y) * src.width + px * side + x) * 3 + c];
        out[(py * gw + px) * width + (y * side + x) * 3 + c] = @floatCast(@as(f64, @floatFromInt(v)) * (1.0 / 255.0));
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
