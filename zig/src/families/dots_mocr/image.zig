//! Slow Qwen2VL image preparation and merge-ordered axial positions for dots.mocr.
const std = @import("std");
const picture = @import("core").picture;
const patches = picture.patches;
const cfg = @import("config.zig");
pub const Prepared = patches.Prepared;
pub const Error = cfg.Error || patches.Error || error{ DotsImageSize, DotsImageBudget, DotsGrid };

pub const Processor = struct {
    patches: patches.Patches,

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: cfg.Config) !Processor {
        const text = try cfg.readFile(a, io, dir, "preprocessor_config.json");
        defer a.free(text);
        return parse(a, text, c);
    }

    pub fn parse(a: std.mem.Allocator, text: []const u8, c: cfg.Config) cfg.Error!Processor {
        const doc = std.json.parseFromSlice(std.json.Value, a, text, .{}) catch |e| return cfg.jsonError(e);
        defer doc.deinit();
        const o = try cfg.object(doc.value);
        try cfg.expectText(o, "image_processor_type", "Qwen2VLImageProcessor");
        try cfg.expectText(o, "processor_class", "DotsVLProcessor");
        if (try cfg.positive(o, "patch_size") != c.vision.patch or try cfg.positive(o, "merge_size") != c.vision.merge or
            try cfg.positive(o, "temporal_patch_size") != 1) return error.UnsupportedDotsConfig;
        inline for (.{ "do_resize", "do_rescale", "do_normalize", "do_convert_rgb" }) |key|
            if (o.get(key)) |_| try cfg.expectBool(o, key, true);
        if (o.get("resample")) |_| if (try cfg.positive(o, "resample") != 3) return error.UnsupportedDotsConfig;
        const rescale = if (o.get("rescale_factor") != null) try cfg.number(o, "rescale_factor") else 1.0 / 255.0;
        if (rescale != 1.0 / 255.0) return error.UnsupportedDotsConfig;
        const p: patches.Patches = .{
            .patch = c.vision.patch,
            .temporal = 1,
            .merge = c.vision.merge,
            .min_pixels = try cfg.positive(o, "min_pixels"),
            .max_pixels = try cfg.positive(o, "max_pixels"),
            .rescale = rescale,
            .mean = try vector(o, "image_mean", false),
            .std = try vector(o, "image_std", true),
            .resize = .pillow,
            .fused = false,
        };
        const area = @as(u64, p.factor()) * p.factor();
        if (p.min_pixels < area or p.min_pixels > p.max_pixels or p.max_pixels > picture.max_pixels)
            return error.UnsupportedDotsConfig;
        return .{ .patches = p };
    }

    pub fn prepare(p: Processor, io: std.Io, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) Error!Prepared {
        var rgb = try picture.decode(io, a, bytes);
        defer rgb.deinit(a);
        return p.prepareRgb(a, rgb, picture.identity.ofBytes(0xd075, bytes), max_tokens);
    }

    /// The slow processor refuses a side below one factor; a too-small token budget is an explicit refusal.
    pub fn prepareRgb(p: Processor, a: std.mem.Allocator, rgb: picture.Rgb, bytes_hash: u64, max_tokens: u32) Error!Prepared {
        const factor = p.patches.factor();
        if (rgb.height < factor or rgb.width < factor) return error.DotsImageSize;
        try picture.fits(rgb.width, rgb.height);
        if (rgb.pixels.len != rgb.width * rgb.height * 3) return error.Invalid;
        if (max_tokens == 0) return error.DotsImageBudget;
        const result = try patches.prepareRgb(a, rgb, bytes_hash, p.patches, max_tokens);
        errdefer a.free(result.pixels);
        if (result.gh == 0 or result.gw == 0 or result.tokens > max_tokens) return error.DotsImageBudget;
        return result;
    }
};

fn vector(o: std.json.ObjectMap, key: []const u8, positive: bool) cfg.Error![3]f32 {
    const v = o.get(key) orelse return error.BadDotsConfig;
    if (v != .array or v.array.items.len != 3) return error.BadDotsConfig;
    var out: [3]f32 = undefined;
    for (v.array.items, &out) |x, *dst| {
        const n: f64 = switch (x) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => return error.BadDotsConfig,
        };
        if (!std.math.isFinite(n) or @abs(n) > std.math.floatMax(f32) or (positive and n <= 0)) return error.BadDotsConfig;
        dst.* = @floatCast(n);
        if (positive and dst.* == 0) return error.BadDotsConfig;
    }
    return out;
}

/// Each returned pair is (y, x); groups of four patches feed one merger row.
pub fn positions(a: std.mem.Allocator, gh: u32, gw: u32, merge: u32) ![][2]u32 {
    if (merge == 0 or gh == 0 or gw == 0 or gh % merge != 0 or gw % merge != 0) return error.DotsGrid;
    const count = std.math.mul(usize, gh, gw) catch return error.DotsGrid;
    const out = try a.alloc([2]u32, count);
    errdefer a.free(out);
    try fillPositions(out, gh, gw, merge);
    return out;
}

pub fn fillPositions(out: [][2]u32, gh: u32, gw: u32, merge: u32) !void {
    if (merge == 0 or gh == 0 or gw == 0 or gh % merge != 0 or gw % merge != 0 or
        out.len != @as(u64, gh) * gw) return error.DotsGrid;
    var i: usize = 0;
    for (0..gh / merge) |by| for (0..gw / merge) |bx| for (0..merge) |dy| for (0..merge) |dx| {
        out[i] = .{ @intCast(by * merge + dy), @intCast(bx * merge + dx) };
        i += 1;
    };
}
