//! A plain BF16 checkpoint's preflight bytes: its text tensors only, not the tower, MTP or joint head beside them.
const std = @import("std");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");

/// The names the CUDA engine loads from a plain BF16 checkpoint (the ModelOpt loader's HF spelling).
const text = [_][]const u8{ "model.language_model.", "lm_head." };

/// The text tensors' stored bytes of a plain BF16 checkpoint; null for every other format (its files are counted).
pub fn textWeightBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !?u64 {
    const g = geometry.Geometry.read(a, io, dir) catch return null; // the open reports a config it cannot read
    if (!g.bf16) return null;
    var ck = try core.Checkpoint.openModelPrefixes(a, io, dir, &text);
    defer ck.close();
    var total: u64 = 0;
    for (ck.files.items) |*f| for (f.names.keys()) |name| {
        total += f.get(name).?.bytes.len;
    };
    // a tied head packs the embedding's rows a second time (cuda_weights.zig)
    if (g.tied) total += @as(u64, g.vocab) * g.hidden * 2;
    return total;
}

/// A BF16 chat tower's weights and one call's scratch on rank 0, which holds it; 0 for other formats and for Clef.
pub fn visionBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, _: u32) !u64 {
    if (rank != 0) return 0;
    const g = geometry.Geometry.read(a, io, dir) catch return 0;
    if (!g.bf16 or @import("cuda_decide.zig").present(a, io, dir)) return 0;
    const v = @import("qwen_vision");
    // a tower the CUDA tower cannot read counts nothing here: cuda_decide refuses it by name in the family's open
    const read = v.Settings.read(a, io, dir) catch |err| return if (err == error.OutOfMemory) err else 0;
    const c = (read orelse return 0).tower;
    return v.Tower.weightBytes(c) + v.Tower.scratchBytes(c, v.max_patches);
}

const config_text =
    \\{"model_type":"qwen3_5","tie_word_embeddings":false,"text_config":{"model_type":"qwen3_5_text",
    \\ "attention_bias":false,"attn_output_gate":true,"hidden_act":"silu","head_dim":256,"linear_key_head_dim":128,
    \\ "linear_value_head_dim":128,"linear_conv_kernel_dim":4,"hidden_size":4096,"intermediate_size":12288,
    \\ "num_hidden_layers":4,"vocab_size":248320,"linear_num_key_heads":16,"linear_num_value_heads":32,
    \\ "num_attention_heads":16,"num_key_value_heads":4,"rms_norm_eps":1e-06,"max_position_embeddings":262144,
    \\ "layer_types":["linear_attention","linear_attention","linear_attention","full_attention"],
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25}}}
;

/// One safetensors file of BF16 tensors, `sizes[i]` elements each, under `names[i]`.
fn writeShard(dir: std.Io.Dir, io: std.Io, path: []const u8, names: []const []const u8, sizes: []const usize) !void {
    const a = std.testing.allocator;
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    var at: usize = 0;
    for (names, sizes, 0..) |n, s, i| {
        if (i > 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ n, s, at, at + 2 * s });
        at += 2 * s;
    }
    try header.append(a, '}');
    const bytes = try a.alloc(u8, 8 + header.items.len + at);
    defer a.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.items.len, .little);
    @memcpy(bytes[8..][0..header.items.len], header.items);
    @memset(bytes[8 + header.items.len ..], 0);
    try dir.writeFile(io, .{ .sub_path = path, .data = bytes });
}

test "plain BF16 preflight: text tensors counted, the tower, MTP and joint head left out; other formats null" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_text });
    // a shard of text alone, and one where the tower and MTP share the file with text (as Clef-Flash's last shard)
    try writeShard(tmp.dir, io, "model-1.safetensors", &.{ "model.language_model.embed_tokens.weight", "lm_head.weight" }, &.{ 64, 32 });
    try writeShard(tmp.dir, io, "model-2.safetensors", &.{ "model.language_model.norm.weight", "model.visual.blocks.0.w", "mtp.fc.weight" }, &.{ 16, 1000, 500 });
    try writeShard(tmp.dir, io, "joint_head.safetensors", &.{"head.w"}, &.{4000}); // beside the model, not in the index
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"model.language_model.embed_tokens.weight":"model-1.safetensors","lm_head.weight":"model-1.safetensors",
        \\ "model.language_model.norm.weight":"model-2.safetensors","model.visual.blocks.0.w":"model-2.safetensors",
        \\ "mtp.fc.weight":"model-2.safetensors"}}
    });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    try std.testing.expectEqual(@as(?u64, 2 * (64 + 32 + 16)), try textWeightBytes(a, io, dir));
    // an MLX config (it parses, as bf16 = false) counts its files as before
    const mlx = try std.mem.replaceOwned(u8, a, config_text, "\"tie_word_embeddings\":false,", "\"tie_word_embeddings\":false,\"quantization\":{\"group_size\":64,\"bits\":4},");
    defer a.free(mlx);
    try std.testing.expect(!(try geometry.parse(a, mlx)).bf16);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = mlx });
    try std.testing.expectEqual(@as(?u64, null), try textWeightBytes(a, io, dir));
}

test "tied BF16 preflight: the head's packed copy of the embedding counts beside the text tensors" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = @embedFile("fixtures/lighton-ocr-0.8b.json") });
    try writeShard(tmp.dir, io, "model.safetensors", &.{ "model.language_model.embed_tokens.weight", "model.visual.blocks.0.w" }, &.{ 64, 1000 });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const g = try geometry.Geometry.read(a, io, dir);
    try std.testing.expect(g.bf16 and g.tied);
    try std.testing.expectEqual(@as(?u64, 2 * 64 + @as(u64, g.vocab) * g.hidden * 2), try textWeightBytes(a, io, dir));
}

test "BF16 chat preflight: the tower's weights and scratch, none for a decision backbone or another format" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const lighton = @embedFile("fixtures/lighton-ocr-0.8b.json");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = lighton });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    const v = @import("qwen_vision");
    const c = (try v.Settings.read(a, io, dir)).?.tower;
    try std.testing.expectEqual(@as(u64, v.Tower.weightBytes(c) + v.Tower.scratchBytes(c, v.max_patches)), try visionBytes(a, io, dir, 0, 1));
    // with a joint head beside it, --vision is the decision backbone's refusal (cuda_decide), not a tower budget
    try tmp.dir.writeFile(io, .{ .sub_path = @import("cuda_decide.zig").head_file, .data = "{}" });
    try std.testing.expectEqual(@as(u64, 0), try visionBytes(a, io, dir, 0, 1));
    try tmp.dir.deleteFile(io, @import("cuda_decide.zig").head_file);
    // an unreadable tower (14-pixel patches) counts nothing, so preflight leaves its refusal to the family's open
    const patch14 = try std.mem.replaceOwned(u8, a, lighton, "\"patch_size\": 16", "\"patch_size\": 14");
    defer a.free(patch14);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = patch14 });
    try std.testing.expectEqual(@as(u64, 0), try visionBytes(a, io, dir, 0, 1));
    // an MLX checkpoint's files count its tower already
    const mlx = try std.mem.replaceOwned(u8, a, config_text, "\"tie_word_embeddings\":false,", "\"tie_word_embeddings\":false,\"quantization\":{\"group_size\":64,\"bits\":4},");
    defer a.free(mlx);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = mlx });
    try std.testing.expectEqual(@as(u64, 0), try visionBytes(a, io, dir, 0, 1));
}
