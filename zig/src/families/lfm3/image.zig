//! LFM3 image inputs: Torch bicubic RGB, merge-block patches and repeated temporal samples.
const std = @import("std");
const picture = @import("../../core/picture/picture.zig");
const geometry = @import("../../core/qwen_vl/geometry.zig");

pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };
pub const patch = 16;
pub const merge = 2;
pub const temporal = 2;
pub const patch_width = 1536;
pub const min_pixels = 65536;
pub const max_pixels = 16777216;

/// The image token budget a served tower admits: 64 to 4096 tokens, 256 when --vision sets none.
pub const min_budget = 64;
pub const max_budget = 4096;
pub const default_budget = 256;

/// The tower's token budget for an open; `BadLfm3ImageBudget` outside 64..4096 tokens or with no image allowed.
pub fn admitBudget(tokens_set: bool, tokens: u32, max_images: u32) !u32 {
    const budget = if (tokens_set) tokens else default_budget;
    if (budget < min_budget or budget > max_budget or max_images == 0) return error.BadLfm3ImageBudget;
    return budget;
}

/// BF16 round-to-nearest-even of a finite value, as the tower's patch embedding takes its pixels.
pub fn roundBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

/// Pixels to the tower's BF16 patch words; a NaN or infinity refuses the image, and `out` is then not to be used.
pub fn packPixels(pixels: []const f32, out: []u16) !void {
    if (pixels.len != out.len) return error.BadLfm3ImageShape;
    for (pixels, out) |value, *word| {
        if (!std.math.isFinite(value)) return error.NonfiniteLfm3Image;
        word.* = roundBf16(value);
    }
}

pub fn validate(inputs: anytype, prompt_len: usize, budget: usize, max_images: usize) !void {
    if (inputs.len > max_images) return error.TooManyImages;
    var previous: usize = 0;
    var total: usize = 0;
    for (inputs) |img| {
        if (img.gh == 0 or img.gw == 0 or img.gh % 2 != 0 or img.gw % 2 != 0)
            return error.BadLfm3ImageShape;
        const patches = std.math.mul(usize, img.gh, img.gw) catch return error.BadLfm3ImageShape;
        if (patches > max_pixels / (patch * patch) or img.tokens != patches / 4 or
            img.pixels.len != patches * patch_width) return error.BadLfm3ImageShape;
        if (patches < min_pixels / (patch * patch)) return error.ImageTooSmall;
        if (img.at < previous or img.at > prompt_len or img.tokens > prompt_len - img.at)
            return error.BadLfm3ImageSpans;
        previous = @as(usize, img.at) + img.tokens;
        if (img.tokens > budget -| total) return error.ImageTooLarge;
        total += img.tokens;
    }
}

pub fn processor(a: std.mem.Allocator, bytes: []const u8) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadLfm3Processor;
    const value = parsed.value.object.get("image_processor") orelse return error.BadLfm3Processor;
    if (value != .object) return error.BadLfm3Processor;
    const o = value.object;
    for ([_][]const u8{ "do_convert_rgb", "do_resize", "do_rescale", "do_normalize" }) |name| {
        const flag = o.get(name) orelse return error.BadLfm3Processor;
        if (flag != .bool or !flag.bool) return error.BadLfm3Processor;
    }
    const kind = o.get("image_processor_type") orelse return error.BadLfm3Processor;
    if (kind != .string or !std.mem.eql(u8, kind.string, "Lfm3VlImageProcessor")) return error.BadLfm3Processor;
    inline for (.{ .{ "patch_size", 16 }, .{ "merge_size", 2 }, .{ "temporal_patch_size", 2 }, .{ "resample", 3 } }) |pair| {
        const v = o.get(pair[0]) orelse return error.BadLfm3Processor;
        if (v != .integer or v.integer != pair[1]) return error.BadLfm3Processor;
    }
    const factor = o.get("rescale_factor") orelse return error.BadLfm3Processor;
    if (factor != .float or factor.float != 1.0 / 255.0) return error.BadLfm3Processor;
    for ([_][]const u8{ "image_mean", "image_std" }) |name| {
        const values = o.get(name) orelse return error.BadLfm3Processor;
        if (values != .array or values.array.items.len != 3) return error.BadLfm3Processor;
        for (values.array.items) |v| if (v != .float or v.float != 0.5) return error.BadLfm3Processor;
    }
    const bounds = o.get("size") orelse return error.BadLfm3Processor;
    if (bounds != .object) return error.BadLfm3Processor;
    const lower = bounds.object.get("shortest_edge") orelse return error.BadLfm3Processor;
    const upper = bounds.object.get("longest_edge") orelse return error.BadLfm3Processor;
    if (lower != .integer or upper != .integer or lower.integer != min_pixels or upper.integer != max_pixels)
        return error.BadLfm3Processor;
}

/// The checkpoint folder's `processor_config.json`, checked against the one image processor the tower implements.
pub fn loadProcessor(io: std.Io, a: std.mem.Allocator, dir: []const u8) !void {
    const path = try std.fs.path.join(a, &.{ dir, "processor_config.json" });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20));
    defer a.free(bytes);
    try processor(a, bytes);
}

pub fn prepare(io: std.Io, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !Prepared {
    var rgb = try picture.decode(io, a, bytes);
    defer rgb.deinit(a);
    return prepareRgb(a, rgb, picture.identity.ofBytes(0x1f33, bytes), max_tokens);
}

pub fn prepareRgb(a: std.mem.Allocator, rgb: picture.Rgb, identity: u64, max_tokens: u32) !Prepared {
    if (rgb.width == 0 or rgb.height == 0 or max_tokens < min_budget) return error.ImageTooSmall;
    const budget = @min(@as(u64, max_pixels), @as(u64, max_tokens) * 1024);
    const size = try geometry.smartResize(@intCast(rgb.height), @intCast(rgb.width), 32, min_pixels, budget);
    const gh = size.h / patch;
    const gw = size.w / patch;
    const tokens = gh * gw / 4;
    if (tokens > max_tokens) return error.ImageTooLarge;
    const resized = try picture.resample.resize(a, .torch, .of(rgb.pixels, rgb.width, rgb.height), size.w, size.h);
    defer a.free(resized);
    const pixels = try patchify(a, .{ .pixels = resized, .width = size.w, .height = size.h });
    return .{ .pixels = pixels, .gh = gh, .gw = gw, .tokens = tokens, .hash = picture.identity.withGrid(identity, .{ gh, gw }) };
}

pub fn patchify(a: std.mem.Allocator, rgb: picture.Rgb) ![]f32 {
    if (rgb.width % 32 != 0 or rgb.height % 32 != 0 or rgb.width == 0 or rgb.height == 0)
        return error.BadLfm3PatchGrid;
    const gh = rgb.height / patch;
    const gw = rgb.width / patch;
    const pixels = try a.alloc(f32, gh * gw * patch_width);
    var lut: [256]f32 = undefined;
    for (&lut, 0..) |*value, i| {
        value.* = (@as(f32, @floatFromInt(i)) - 127.5) / 127.5;
    }
    var row: usize = 0;
    for (0..gh / merge) |by| for (0..gw / merge) |bx| for (0..merge) |dy| for (0..merge) |dx| {
        const dst = pixels[row * patch_width ..][0..patch_width];
        for (0..3) |channel| for (0..temporal) |frame| for (0..patch) |y| for (0..patch) |x| {
            const iy = (by * merge + dy) * patch + y;
            const ix = (bx * merge + dx) * patch + x;
            dst[((channel * temporal + frame) * patch + y) * patch + x] = lut[rgb.pixels[(iy * rgb.width + ix) * 3 + channel]];
        };
        row += 1;
    };
    return pixels;
}

pub fn explain(err: anyerror) ?[]const u8 {
    if (err == error.ImageTooSmall) return "LFM3 images require at least 64 tokens after resizing and a budget of at least 64 tokens";
    if (err == error.ImageTooLarge) return "the resized LFM3 image exceeds the image token budget";
    return geometry.problem(err);
}

test "LFM3 patches preserve merge order, channel order and duplicated temporal frames" {
    var pixels: [32 * 64 * 3]u8 = undefined;
    for (0..32) |y| for (0..64) |x| {
        pixels[(y * 64 + x) * 3] = @intCast(y / 16 * 4 + x / 16);
        pixels[(y * 64 + x) * 3 + 1] = 255;
        pixels[(y * 64 + x) * 3 + 2] = 0;
    };
    const out = try patchify(std.testing.allocator, .{ .pixels = &pixels, .width = 64, .height = 32 });
    defer std.testing.allocator.free(out);
    const order = [_]u8{ 0, 1, 4, 5, 2, 3, 6, 7 };
    for (order, 0..) |red, row| {
        const expected = (@as(f32, @floatFromInt(red)) - 127.5) / 127.5;
        try std.testing.expectEqual(expected, out[row * patch_width]);
        try std.testing.expectEqual(expected, out[row * patch_width + 256]);
        try std.testing.expectEqual(@as(f32, 1), out[row * patch_width + 512]);
        try std.testing.expectEqual(@as(f32, -1), out[row * patch_width + 1024]);
    }
}

test "LFM3 rejects image metadata that disagrees with its grid, budget or placeholder spans" {
    const Input = struct { pixels: []const f32, gh: u32 = 16, gw: u32 = 16, tokens: u32 = 64, at: u32 = 0 };
    const pixels = try std.testing.allocator.alloc(f32, 256 * patch_width);
    defer std.testing.allocator.free(pixels);
    var inputs = [_]Input{ .{ .pixels = pixels }, .{ .pixels = pixels, .at = 64 } };
    try validate(&inputs, 128, 128, 2);
    inputs[0].tokens = 65;
    try std.testing.expectError(error.BadLfm3ImageShape, validate(&inputs, 129, 129, 2));
    inputs[0].tokens = 64;
    inputs[1].at = 0;
    try std.testing.expectError(error.BadLfm3ImageSpans, validate(&inputs, 128, 128, 2));
    inputs[1].at = 64;
    try std.testing.expectError(error.BadLfm3ImageSpans, validate(&inputs, 127, 128, 2));
    try std.testing.expectError(error.ImageTooLarge, validate(&inputs, 128, 127, 2));
    try std.testing.expectError(error.TooManyImages, validate(&inputs, 128, 128, 1));
}

test "LFM3 image patches match the processor float32 pixel fixture" {
    const a = std.testing.allocator;
    const rgb = try a.alloc(u8, 217 * 300 * 3);
    defer a.free(rgb);
    for (0..217) |y| for (0..300) |x| for (0..3) |channel| {
        rgb[(y * 300 + x) * 3 + channel] = @intCast((x * 7 + y * 13 + channel * 50 + (x * y) % 31) % 256);
    };
    const prepared = try prepareRgb(a, .{ .pixels = rgb, .width = 300, .height = 217 }, 0, 1 << 20);
    defer a.free(prepared.pixels);
    try std.testing.expectEqual([2]u32{ 14, 20 }, [2]u32{ prepared.gh, prepared.gw });
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(prepared.pixels), &digest, .{});
    const actual = std.fmt.bytesToHex(digest, .lower);
    try std.testing.expectEqualStrings("53153c22eb4e065031bbdf4de0f038db37ef736a880e3438ac8e0d73f82dcea8", &actual);
}
