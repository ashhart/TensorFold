//! A Qwen-VL checkpoint's image settings: its tower's shape, the image marker ids and the image processor's bounds.
const std = @import("std");

pub const Error = error{ OutOfMemory, VisionConfig };

pub const Config = struct {
    depth: u32,
    hidden: u32,
    heads: u32,
    intermediate: u32,
    out_hidden: u32, // the language model's width: the merger's output
    patch: u32,
    temporal: u32,
    merge: u32,
    positions: u32, // learned positions, a square grid
    image_token: u32,
    vision_start: u32,
    vision_end: u32,
    min_pixels: u64,
    max_pixels: u64,
    rescale: f64,
    mean: [3]f32,
    std: [3]f32,

    pub fn headDim(c: Config) u32 {
        return c.hidden / c.heads;
    }

    /// A patch's values: channel by frame by row by column.
    pub fn patchWidth(c: Config) u32 {
        return 3 * c.temporal * c.patch * c.patch;
    }

    /// Image sides are multiples of this many pixels.
    pub fn factor(c: Config) u32 {
        return c.patch * c.merge;
    }

    /// The learned grid's side.
    pub fn side(c: Config) u32 {
        return std.math.sqrt(c.positions);
    }

    /// A merged row's width: `merge` x `merge` patch rows side by side.
    pub fn merged(c: Config) u32 {
        return c.hidden * c.merge * c.merge;
    }

    /// config.json's vision_config and marker ids, then the image processor (preprocessor, then processor_config.json).
    pub fn parse(arena: std.mem.Allocator, config_json: []const u8, preprocessor_json: ?[]const u8, processor_json: ?[]const u8) Error!?Config {
        const doc = std.json.parseFromSliceLeaky(std.json.Value, arena, config_json, .{}) catch return error.VisionConfig;
        if (doc != .object) return error.VisionConfig;
        const vc = doc.object.get("vision_config") orelse return null;
        if (vc != .object) return null;
        var c: Config = .{ .depth = 0, .hidden = 0, .heads = 0, .intermediate = 0, .out_hidden = 0, .patch = 0, .temporal = 0, .merge = 0, .positions = 0, .image_token = 0, .vision_start = 0, .vision_end = 0, .min_pixels = 56 * 56, .max_pixels = 28 * 28 * 1280, .rescale = 1.0 / 255.0, .mean = .{ 0.48145466, 0.4578275, 0.40821073 }, .std = .{ 0.26862954, 0.26130258, 0.27577711 } };
        inline for (.{ .{ "depth", "depth" }, .{ "hidden_size", "hidden" }, .{ "num_heads", "heads" }, .{ "intermediate_size", "intermediate" }, .{ "out_hidden_size", "out_hidden" }, .{ "patch_size", "patch" }, .{ "temporal_patch_size", "temporal" }, .{ "spatial_merge_size", "merge" }, .{ "num_position_embeddings", "positions" } }) |kv| {
            @field(c, kv[1]) = try positive(vc.object.get(kv[0]));
        }
        inline for (.{ .{ "image_token_id", "image_token" }, .{ "vision_start_token_id", "vision_start" }, .{ "vision_end_token_id", "vision_end" } }) |kv| {
            @field(c, kv[1]) = try positive(doc.object.get(kv[0]));
        }
        if (vc.object.get("in_channels")) |v| if (try positive(v) != 3) return error.VisionConfig;
        const act = vc.object.get("hidden_act") orelse return error.VisionConfig;
        if (act != .string or !std.mem.eql(u8, act.string, "gelu_pytorch_tanh")) return error.VisionConfig;
        if (vc.object.get("deepstack_visual_indexes")) |d| if (d != .array or d.array.items.len != 0) return error.VisionConfig;
        const s = c.side();
        if (s * s != c.positions or s < 2 or c.hidden % c.heads != 0 or c.headDim() % 4 != 0) return error.VisionConfig;
        for ([_]?[]const u8{ preprocessor_json, processor_json }, 0..) |text, i| if (text) |t| {
            const p = std.json.parseFromSliceLeaky(std.json.Value, arena, t, .{}) catch return error.VisionConfig;
            if (p != .object) return error.VisionConfig;
            const ip = if (i == 0) p else p.object.get("image_processor") orelse continue;
            if (ip != .object) return error.VisionConfig;
            try c.processor(ip.object);
        };
        if (c.min_pixels > c.max_pixels or c.rescale <= 0) return error.VisionConfig;
        return c;
    }

    /// One processor object's fields over the defaults; its patch geometry must be the tower's.
    fn processor(c: *Config, o: std.json.ObjectMap) Error!void {
        inline for (.{ .{ "patch_size", c.patch }, .{ "temporal_patch_size", c.temporal }, .{ "merge_size", c.merge } }) |kv| {
            if (o.get(kv[0])) |v| if (try positive(v) != kv[1]) return error.VisionConfig;
        }
        if (o.get("size")) |size| if (size == .object) {
            if (size.object.get("shortest_edge")) |v| c.min_pixels = try positive(v);
            if (size.object.get("longest_edge")) |v| c.max_pixels = try positive(v);
        };
        if (o.get("min_pixels")) |v| c.min_pixels = try positive(v);
        if (o.get("max_pixels")) |v| c.max_pixels = try positive(v);
        if (o.get("rescale_factor")) |v| c.rescale = number(v) orelse return error.VisionConfig;
        inline for (.{ "image_mean", "image_std" }, .{ &c.mean, &c.std }) |key, dst| if (o.get(key)) |v| {
            if (v != .array or v.array.items.len != 3) return error.VisionConfig;
            for (v.array.items, 0..) |x, i| dst[i] = @floatCast(number(x) orelse return error.VisionConfig);
        };
        for (c.std) |x| if (x == 0) return error.VisionConfig;
    }
};

fn positive(v: ?std.json.Value) Error!u32 {
    const x = v orelse return error.VisionConfig;
    if (x != .integer or x.integer <= 0 or x.integer > std.math.maxInt(u32)) return error.VisionConfig;
    return @intCast(x.integer);
}

fn number(v: std.json.Value) ?f64 {
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

/// The 27B's own files, trimmed: its vision_config, marker ids and image processor.
pub const sample_config =
    \\{"image_token_id": 248056, "video_token_id": 248057, "vision_start_token_id": 248053, "vision_end_token_id": 248054,
    \\ "vision_config": {"deepstack_visual_indexes": [], "depth": 27, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 1152,
    \\  "in_channels": 3, "intermediate_size": 4304, "num_heads": 16, "num_position_embeddings": 2304, "out_hidden_size": 5120,
    \\  "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}}
;
pub const sample_preprocessor =
    \\{"size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16, "temporal_patch_size": 2,
    \\ "merge_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
;
pub const sample_processor =
    \\{"image_processor": {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "max_pixels": 16777216,
    \\ "merge_size": 2, "min_pixels": 65536, "patch_size": 16, "rescale_factor": 0.00392156862745098, "temporal_patch_size": 2}}
;

test "the 27B's files parse to its tower, markers and pixel bounds" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = (try Config.parse(a, sample_config, sample_preprocessor, sample_processor)).?;
    try std.testing.expectEqual(@as(u32, 27), c.depth);
    try std.testing.expectEqual(@as(u32, 72), c.headDim());
    try std.testing.expectEqual(@as(u32, 1536), c.patchWidth());
    try std.testing.expectEqual(@as(u32, 48), c.side());
    try std.testing.expectEqual(@as(u32, 4608), c.merged());
    try std.testing.expectEqual(@as(u32, 5120), c.out_hidden);
    try std.testing.expectEqual(@as(u32, 248056), c.image_token);
    try std.testing.expectEqual(@as(u64, 65536), c.min_pixels);
    try std.testing.expectEqual(@as(u64, 16777216), c.max_pixels);
    try std.testing.expectEqual(@as(f64, 1.0 / 255.0), c.rescale); // the JSON's shortest repr of 1/255
    try std.testing.expectEqual([3]f32{ 0.5, 0.5, 0.5 }, c.mean);
    try std.testing.expect((try Config.parse(a, "{\"model_type\": \"qwen3_5\"}", null, null)) == null);
}

test "a tower these kernels do not read is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const swaps = [_][2][]const u8{
        .{ "\"deepstack_visual_indexes\": []", "\"deepstack_visual_indexes\": [8]" },
        .{ "gelu_pytorch_tanh", "silu" },
        .{ "\"num_position_embeddings\": 2304", "\"num_position_embeddings\": 2300" },
        .{ "\"num_heads\": 16", "\"num_heads\": 0" },
    };
    for (swaps) |s| {
        const text = try std.mem.replaceOwned(u8, a, sample_config, s[0], s[1]);
        try std.testing.expectError(error.VisionConfig, Config.parse(a, text, null, null));
    }
    const other_patch = try std.mem.replaceOwned(u8, a, sample_preprocessor, "\"patch_size\": 16", "\"patch_size\": 14");
    try std.testing.expectError(error.VisionConfig, Config.parse(a, sample_config, other_patch, null));
}
