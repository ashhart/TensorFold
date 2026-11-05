//! Transformers v5.5.4 LightOnOcrProcessor: Pixtral resize and normalization, raster patches and plain image tokens.

// Resize adapted from Hugging Face Transformers v5.5.4, copyright 2024 The HuggingFace Inc. team, Apache-2.0.
const std = @import("std");
const picture = @import("core").picture;
const Config = @import("config.zig").Config;
const Allocator = std.mem.Allocator;

pub const Error = picture.Error || error{ ImageTooSmall, ImageTokensExceeded, BadLightOnProcessor, UnsupportedLightOnProcessor };
pub const Rgb = picture.Rgb;
pub const Size = struct { h: u32, w: u32 };
pub const Prepared = struct { pixels: []f32, gh: u32, gw: u32, tokens: u32, hash: u64 };
pub const patch: u32 = 14;
pub const merge: u32 = 2;
pub const patch_width: u32 = 3 * patch * patch;
pub const mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
pub const sd = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
pub const rescale: f64 = 1.0 / 255.0;

/// floor after shrinking, then round each side up to a 14-pixel boundary; never align the grid to the merger.
pub fn targetSize(height: u32, width: u32, longest_edge: u32) Error!Size {
    if (height == 0 or width == 0 or longest_edge == 0 or longest_edge > 1540) return error.ImageTooSmall;
    var h: f64 = @floatFromInt(height);
    var w: f64 = @floatFromInt(width);
    const edge: f64 = @floatFromInt(longest_edge);
    const ratio = @max(h / edge, w / edge);
    if (ratio > 1) {
        h = @floor(h / ratio);
        w = @floor(w / ratio);
    }
    if (h < 1 or w < 1) return error.ImageTooSmall;
    const ih: u32 = @intFromFloat(h);
    const iw: u32 = @intFromFloat(w);
    return .{ .h = ((ih - 1) / patch + 1) * patch, .w = ((iw - 1) / patch + 1) * patch };
}

/// Read the checkpoint's processor before advertising this fixed, qualified processor implementation.
pub fn validateProcessor(gpa: Allocator, bytes: []const u8, c: Config) Error!void {
    const doc = std.json.parseFromSlice(std.json.Value, gpa, bytes, .{}) catch |err|
        return if (err == error.OutOfMemory) error.OutOfMemory else error.BadLightOnProcessor;
    defer doc.deinit();
    const root = try object(doc.value);
    const proc = try object(root.get("image_processor") orelse return error.BadLightOnProcessor);
    const proc_type = proc.get("image_processor_type") orelse return error.BadLightOnProcessor;
    if (proc_type != .string) return error.BadLightOnProcessor;
    if (!std.mem.eql(u8, proc_type.string, "PixtralImageProcessorFast") and
        !std.mem.eql(u8, proc_type.string, "PixtralImageProcessor")) return error.UnsupportedLightOnProcessor;
    try integer(root, "patch_size", patch);
    try integer(root, "spatial_merge_size", merge);
    try string(root, "image_token", "<|image_pad|>");
    try integer(proc, "patch_size", patch);
    try integer(proc, "resample", 3);
    inline for (.{ "do_convert_rgb", "do_resize", "do_rescale", "do_normalize" }) |key| try boolean(proc, key, true);
    inline for (.{ "do_center_crop", "do_pad" }) |key| if (proc.get(key)) |value| {
        if (value != .null and (value != .bool or value.bool)) return error.UnsupportedLightOnProcessor;
    };
    if (proc.get("data_format")) |value| {
        if (value != .string or !std.mem.eql(u8, value.string, "channels_first")) return error.UnsupportedLightOnProcessor;
    }
    inline for (.{ "input_data_format", "pad_size", "crop_size" }) |key| if (proc.get(key)) |value| {
        if (value != .null) return error.UnsupportedLightOnProcessor;
    };
    const size = try object(proc.get("size") orelse return error.BadLightOnProcessor);
    try integer(size, "longest_edge", c.vision.side);
    if (c.vision.patch != patch or c.merge != merge or c.vision.side != 1540) return error.UnsupportedLightOnProcessor;
    if (try number(proc.get("rescale_factor")) != rescale) return error.UnsupportedLightOnProcessor;
    inline for (.{ "image_mean", "image_std" }, .{ mean, sd }) |key, expected| {
        const values = proc.get(key) orelse return error.BadLightOnProcessor;
        if (values != .array or values.array.items.len != 3) return error.BadLightOnProcessor;
        for (values.array.items, expected) |value, v| {
            if (@as(f32, @floatCast(try number(value))) != v) return error.UnsupportedLightOnProcessor;
        }
    }
}

pub fn readProcessor(gpa: Allocator, io: std.Io, dir: []const u8, c: Config) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "processor_config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    try validateProcessor(gpa, bytes, c);
}

/// The encoder reads every patch; the projector's 2x2 unfold drops an unpaired last row or column.
pub fn tokens(gh: u32, gw: u32) u32 {
    return (gh / merge) * (gw / merge);
}

pub fn prepare(io: std.Io, gpa: Allocator, bytes: []const u8, c: Config, max_tokens: u32) Error!Prepared {
    var rgb = try picture.decode(io, gpa, bytes);
    defer rgb.deinit(gpa);
    return prepareRgb(gpa, rgb, picture.identity.ofBytes(0x10c31, bytes), c, max_tokens);
}

/// Keep the processor's default resolution and refuse a budget below it, so a budget cannot silently alter pixels.
pub fn prepareRgb(gpa: Allocator, rgb: picture.Rgb, bytes_hash: u64, c: Config, max_tokens: u32) Error!Prepared {
    if (c.vision.patch != patch or c.merge != merge or c.vision.side != 1540) return error.UnsupportedLightOnProcessor;
    if (rgb.width > picture.max_side or rgb.height > picture.max_side) return error.Oversized;
    try picture.fits(rgb.width, rgb.height);
    if (rgb.pixels.len != rgb.width * rgb.height * 3) return error.Invalid;
    const size = try targetSize(@intCast(rgb.height), @intCast(rgb.width), c.vision.side);
    const gh = size.h / patch;
    const gw = size.w / patch;
    const count = tokens(gh, gw);
    if (count == 0) return error.ImageTooSmall;
    if (count > max_tokens) return error.ImageTokensExceeded;
    const resized = try picture.resample.resize(gpa, .torch, .of(rgb.pixels, rgb.width, rgb.height), size.w, size.h);
    defer gpa.free(resized);
    const lut = picture.normalize.table(.fused, rescale, mean, sd);
    const out = try gpa.alloc(f32, @as(usize, gh) * gw * patch_width);
    for (0..gh) |py| for (0..gw) |px| for (0..3) |ch| for (0..patch) |y| for (0..patch) |x| {
        const v = resized[((py * patch + y) * size.w + px * patch + x) * 3 + ch];
        out[(py * gw + px) * patch_width + (ch * patch + y) * patch + x] = lut[ch][v];
    };
    return .{ .pixels = out, .gh = gh, .gw = gw, .tokens = count, .hash = picture.identity.withGrid(bytes_hash, .{ gh, gw }) };
}

pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.ImageTooSmall => "LightOnOCR needs at least two 14-pixel patches on each image side after resizing",
        error.ImageTokensExceeded => "the image exceeds this request's image-token budget at the checkpoint's resolution",
        error.BadLightOnProcessor, error.UnsupportedLightOnProcessor => "this LightOnOCR image processor is unsupported",
        else => picture.decoderProblem(err),
    };
}

fn object(value: std.json.Value) Error!std.json.ObjectMap {
    return if (value == .object) value.object else error.BadLightOnProcessor;
}

fn integer(o: std.json.ObjectMap, key: []const u8, expected: u32) Error!void {
    const value = o.get(key) orelse return error.BadLightOnProcessor;
    if (value != .integer) return error.BadLightOnProcessor;
    if (value.integer != expected) return error.UnsupportedLightOnProcessor;
}

fn string(o: std.json.ObjectMap, key: []const u8, expected: []const u8) Error!void {
    const value = o.get(key) orelse return error.BadLightOnProcessor;
    if (value != .string) return error.BadLightOnProcessor;
    if (!std.mem.eql(u8, value.string, expected)) return error.UnsupportedLightOnProcessor;
}

fn boolean(o: std.json.ObjectMap, key: []const u8, expected: bool) Error!void {
    const value = o.get(key) orelse return error.BadLightOnProcessor;
    if (value != .bool) return error.BadLightOnProcessor;
    if (value.bool != expected) return error.UnsupportedLightOnProcessor;
}

fn number(value: ?std.json.Value) Error!f64 {
    const v = value orelse return error.BadLightOnProcessor;
    const n: f64 = switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => return error.BadLightOnProcessor,
    };
    return if (std.math.isFinite(n)) n else error.BadLightOnProcessor;
}

test {
    _ = @import("image_test.zig");
}
