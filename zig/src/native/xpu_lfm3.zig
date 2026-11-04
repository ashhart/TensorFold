//! The LFM3 entry of the Intel GPU lane host: its family table row, loader and prompt window.
const std = @import("std");
const xpu = @import("xpu");
const api = @import("engine_api");
const lfm3 = @import("lfm3_xpu");
const image = @import("lfm3_image").image;
const Allocator = std.mem.Allocator;

pub const model_type = "lfm3";
/// LFM3-VL-2B: the same text decoder (groups of 32, an 8-bit embedding) with the vision tower.
pub const vl_model_type = "lfm3_vl";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const name = "xpu-lfm3";
pub const Engine = lfm3.engine.Engine;

/// Windows cut at any row give the same bits: states are kept and resumed anywhere, short prompts too.
pub const cache_rules: api.prompt_cache.Rules = .{ .min_prompt = 0, .min_gap = 64 };

/// Copies drafted after an 8-token match (the core's 4 lost 6% on a short chat on Iris Xe; XPU_COPY_MATCH overrides).
pub const copy_match: i64 = 8;

/// Rows of one prompt window (pdense above one row; the scratch is sized for it): 128 beat 32 and 64 on Iris Xe.
const prompt_rows: u32 = 128;

pub fn promptRows(_: *const Engine) u32 {
    return prompt_rows;
}

/// What the startup line adds about the engine's settings.
pub fn note(a: Allocator, e: *const Engine) ![]const u8 {
    return std.fmt.allocPrint(a, ", prompt windows of {d} rows, copy drafts of the lane core (no head)", .{e.window_rows});
}

/// The engine for `o.dir`, or null with `problem` set (the caller releases the device).
pub fn init(a: Allocator, io: std.Io, ctx: *const xpu.Context, o: api.Open, window: usize, problem: *[]const u8) !?*Engine {
    return Engine.init(std.heap.page_allocator, io, ctx, o.dir, .{ .context = window, .window_rows = prompt_rows }) catch |e| {
        problem.* = if (e == error.UnsupportedFormat)
            try std.fmt.allocPrint(a, "{s}: the Intel GPU engine reads LFM3 in MLX affine 4-bit: group 64, or group 32 with an 8-bit embedding (LFM3-VL)", .{o.dir})
        else
            try std.fmt.allocPrint(a, "the Intel GPU engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
}

/// LFM3-VL's image input as #1502's Metal engine offers it (256 image tokens unless --vision-image-tokens); else null.
pub fn vision(a: Allocator, e: *Engine, o: api.Open, problem: *[]const u8) !?api.Vision {
    if (e.c.kind != .vision_language) {
        if (o.vision == null) return null;
        problem.* = "LFM3-1.7B is text-only; --vision requires an LFM3-VL checkpoint";
        return error.NoVisionTower;
    }
    const v = o.vision orelse api.VisionOpen{};
    const bytes = xpu.loader.readFile(a, o.dir, "processor_config.json") catch |err| {
        problem.* = "LFM3-VL: processor_config.json is missing or unreadable";
        return err;
    };
    image.processor(a, bytes) catch |err| {
        problem.* = "LFM3-VL: processor_config.json names another image processor than Lfm3VlImageProcessor's";
        return err;
    };
    const capacity = if (v.image_tokens_set) v.image_tokens else 256;
    if (capacity < 64 or capacity > 4096 or v.max_images == 0) {
        problem.* = "LFM3-VL reads 64 to 4096 image tokens a request (--vision-image-tokens) and at least one image";
        return error.BadLfm3ImageBudget;
    }
    try e.openVision(capacity);
    std.log.info("LFM3-VL: image tower loaded ({d} MiB with scratch for {d} image tokens a request)", .{ e.tower.?.bytes >> 20, capacity });
    return .{
        .ctx = e,
        .prepare = prepare,
        .image_token = 125030,
        .markers = &.{ 125009, 125030, 125010 },
        .image_tokens = capacity,
        .max_images = @min(v.max_images, Engine.max_pictures),
        .part_type = "image",
        .explain = image.explain,
    };
}

/// Info.vision's prepare: #1502's image.zig (torch bicubic, merge-order patches), on the HTTP thread.
fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const e: *Engine = @ptrCast(@alignCast(ctx));
    const p = try image.prepare(e.io, a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}
