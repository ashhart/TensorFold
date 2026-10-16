//! GLM-5.3-Flash's image tower: its shape and config reading, Metal-free (every backend's image.zig reads it).
const std = @import("std");

// The tower this file reads; `Config.parse` refuses a checkpoint whose vision_config differs.
pub const hidden = 1024;
pub const heads = 16;
pub const head_dim = 64;
pub const depth = 24;
pub const mlp_inter = 4096;
pub const out_hidden = 4096; // the language model's width
pub const merger_inter = 10240;
pub const patch = 14;
pub const temporal = 2;
pub const merge = 2;
pub const patch_width = 3 * temporal * patch * patch; // 1176: a patch's values, channel by frame by row by column
pub const patches_per_token = merge * merge;

/// --vision's limits: image tokens a prompt may hold in all (scratch and rows are sized for them), and images.
pub const Limits = struct { image_tokens: u32 = 4096, max_images: u32 = 4 };

pub const Config = struct {
    eps: f32 = 1e-5,
    limit: f32 = 10, // swiglu_limit
    image_token: u32,
    image_start: u32,
    image_end: u32,
    video: ?[3]u32 = null, // video_start_token_id, video_token_id, video_end_token_id: the checkpoint reads videos
    mean: [3]f32,
    std: [3]f32,
    min_tokens: u32,
    max_tokens: u32,

    /// config.json's vision_config and image token ids, processor_config.json's image processor; null: no tower.
    pub fn parse(arena: std.mem.Allocator, config_json: []const u8, processor_json: ?[]const u8) !?Config {
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, config_json, .{});
        const vc = doc.object.get("vision_config") orelse return null;
        if (vc != .object) return null;
        const want = .{ .{ "depth", depth }, .{ "hidden_size", hidden }, .{ "num_heads", heads }, .{ "intermediate_size", mlp_inter }, .{ "out_hidden_size", out_hidden }, .{ "projection_intermediate_size", merger_inter }, .{ "patch_size", patch }, .{ "temporal_patch_size", temporal }, .{ "spatial_merge_size", merge }, .{ "in_channels", 3 } };
        inline for (want) |kv| {
            const v = vc.object.get(kv[0]) orelse return error.VisionConfig;
            if (v != .integer or v.integer != kv[1]) {
                std.log.err("glm vision: vision_config.{s} is {f}; this tower reads {d}", .{ kv[0], std.json.fmt(v, .{}), kv[1] });
                return error.VisionConfig;
            }
        }
        const act = vc.object.get("hidden_act") orelse return error.VisionConfig;
        if (act != .string or !std.mem.eql(u8, act.string, "silu")) return error.VisionConfig;
        var c: Config = .{ .image_token = 0, .image_start = 0, .image_end = 0, .mean = .{ 0.48145466, 0.4578275, 0.40821073 }, .std = .{ 0.26862954, 0.26130258, 0.27577711 }, .min_tokens = 16, .max_tokens = 8000 };
        if (vc.object.get("rms_norm_eps")) |v| c.eps = @floatCast(number(v) orelse return error.VisionConfig);
        if (vc.object.get("swiglu_limit")) |v| c.limit = @floatCast(number(v) orelse return error.VisionConfig);
        inline for (.{ .{ "image_token_id", "image_token" }, .{ "image_start_token_id", "image_start" }, .{ "image_end_token_id", "image_end" } }) |kv| {
            const v = doc.object.get(kv[0]) orelse return error.VisionConfig;
            if (v != .integer) return error.VisionConfig;
            @field(c, kv[1]) = @intCast(v.integer);
        }
        if (doc.object.get("video_start_token_id")) |s| if (doc.object.get("video_token_id")) |t| if (doc.object.get("video_end_token_id")) |e| {
            if (s == .integer and t == .integer and e == .integer) c.video = .{ @intCast(s.integer), @intCast(t.integer), @intCast(e.integer) };
        };
        const pj = processor_json orelse return c;
        const pdoc = try std.json.parseFromSliceLeaky(std.json.Value, arena, pj, .{});
        const ip = pdoc.object.get("image_processor") orelse return c;
        if (ip != .object) return error.VisionConfig;
        inline for (.{ .{ "patch_size", patch }, .{ "temporal_patch_size", temporal }, .{ "merge_size", merge } }) |kv| {
            if (ip.object.get(kv[0])) |v| if (v != .integer or v.integer != kv[1]) return error.VisionConfig;
        }
        if (ip.object.get("patch_expand_factor")) |v| if (v != .integer or v.integer != 1) return error.VisionConfig;
        inline for (.{ "image_mean", "image_std" }, .{ &c.mean, &c.std }) |key, dst| if (ip.object.get(key)) |v| {
            if (v != .array or v.array.items.len != 3) return error.VisionConfig;
            for (v.array.items, 0..) |x, i| dst[i] = @floatCast(number(x) orelse return error.VisionConfig);
        };
        if (ip.object.get("min_image_tokens")) |v| c.min_tokens = @intCast(v.integer);
        if (ip.object.get("max_image_tokens")) |v| c.max_tokens = @intCast(v.integer);
        return c;
    }

    fn number(v: std.json.Value) ?f64 {
        return switch (v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => null,
        };
    }
};
