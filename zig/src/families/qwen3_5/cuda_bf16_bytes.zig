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
    return total;
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
