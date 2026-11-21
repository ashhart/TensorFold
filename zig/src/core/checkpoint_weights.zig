//! Header-based admission bytes for the text side and an optional vision tower, including mixed shards.
const std = @import("std");
const ck = @import("checkpoint.zig");
const st = @import("safetensors.zig");

/// Tower and projector prefixes: RADIO's, then Qwen-VL's, Gemma 4's, LFM2-VL's, LightOn's, DeepSeek's and GLM's.
const vision_prefixes = [_][]const u8{
    "vision_model.",         "vision_projector.",        "mlp1.",            "model.visual.", "visual.",
    "vision_tower.",         "model.vision_tower.",      "vision_embedder.", "embed_vision.", "model.vision_model.",
    "model.vision_encoder.", "model.vision_projection.", "vision.",
};

pub fn visionTensor(name: []const u8) bool {
    for (vision_prefixes) |p|
        if (std.mem.startsWith(u8, name, p)) return true;
    return false;
}

/// Every shard's tensor bytes from its header alone, one header held at a time; a repeated name counts each time.
pub fn count(a: std.mem.Allocator, io: std.Io, dir: []const u8, vision: bool) !u64 {
    try @import("save_guard").check(io, dir);
    const paths = try ck.shardFiles(a, io, dir);
    defer ck.freeShardFiles(a, paths);
    var total: u64 = 0;
    for (paths) |path| {
        var header = try st.Metadata.read(a, io, path);
        defer header.deinit();
        for (header.names.keys(), header.names.values()) |name, entry| {
            if (!vision and visionTensor(name)) continue;
            total = try std.math.add(u64, total, entry.end - entry.begin);
        }
    }
    return total;
}

test {
    _ = @import("checkpoint_weights_test.zig");
}

test "text admission excludes RADIO tensors even in a shared shard" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"language_model.lm_head.weight\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}," ++
        "\"vision_model.patch.weight\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[8,16]}," ++
        "\"mlp1.fc1.weight\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[16,24]}}";
    var image: std.ArrayList(u8) = .empty;
    defer image.deinit(a);
    try image.appendNTimes(a, 0, 8);
    std.mem.writeInt(u64, image.items[0..8], header.len, .little);
    try image.appendSlice(a, header);
    try image.appendNTimes(a, 0, 24);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = image.items });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    try std.testing.expectEqual(@as(u64, 8), try count(a, io, dir, false));
    try std.testing.expectEqual(@as(u64, 24), try count(a, io, dir, true));
}

test "every family's tower prefix is a vision tensor, and no text tensor of theirs is" {
    for ([_][]const u8{ "model.visual.blocks.0.attn.qkv.weight", "vision_tower.blocks.0.norm1.weight", "vision.blocks.3.mlp.w1.weight", "model.vision_model.embeddings.patch_embedding.weight", "model.vision_tower.vision_model.post_layernorm.weight", "vision_embedder.patch_dense.weight", "model.vision_encoder.patch_conv.weight" }) |n|
        try std.testing.expect(visionTensor(n));
    for ([_][]const u8{ "model.language_model.embed_tokens.weight", "language_model.model.layers.0.mlp.gate_proj.weight", "model.layers.0.self_attn.q_proj.weight", "backbone.layers.0.mixer.in_proj.weight", "lm_head.weight", "model.embed_tokens.weight", "mtp.layers.0.eh_proj.weight", "model.multi_modal_projector.linear_1.weight" }) |n|
        try std.testing.expect(!visionTensor(n));
}
