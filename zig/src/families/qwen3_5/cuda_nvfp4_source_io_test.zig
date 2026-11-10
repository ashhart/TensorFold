//! Real checkpoint mappings and source-preflight allocation failures close before any GPU allocation exists.
const std = @import("std");
const core = @import("core");
const source = @import("cuda_nvfp4_mlp.zig");
const geometry = @import("cuda_geometry.zig");
const t = std.testing;

fn normFile(dir: std.Io.Dir, name: []const u8, bits: u16) !void {
    const header = try std.fmt.allocPrint(t.allocator, "{{\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[5120],\"data_offsets\":[0,10240]}}}}", .{name});
    defer t.allocator.free(header);
    const data = try t.allocator.alloc(u8, 8 + header.len + 10240);
    defer t.allocator.free(data);
    std.mem.writeInt(u64, data[0..8], header.len, .little);
    @memcpy(data[8..][0..header.len], header);
    for (0..5120) |i| std.mem.writeInt(u16, data[8 + header.len + i * 2 ..][0..2], bits, .little);
    try dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = data });
}

fn settings() ![]u8 {
    var layers: [64][]const u8 = undefined;
    for (&layers, 0..) |*layer, i| layer.* = if (i % 4 == 3) "full_attention" else "linear_attention";
    return std.json.Stringify.valueAlloc(t.allocator, .{
        .model_type = "qwen3_5",
        .quantization_config = .{ .quant_method = "compressed-tensors" },
        .text_config = .{
            .model_type = "qwen3_5_text",
            .attention_bias = false,
            .attn_output_gate = true,
            .hidden_act = "silu",
            .head_dim = 256,
            .linear_key_head_dim = 128,
            .linear_value_head_dim = 128,
            .linear_conv_kernel_dim = 4,
            .hidden_size = 5120,
            .intermediate_size = 17408,
            .num_hidden_layers = 64,
            .vocab_size = 248320,
            .linear_num_key_heads = 16,
            .linear_num_value_heads = 48,
            .num_attention_heads = 24,
            .num_key_value_heads = 4,
            .rms_norm_eps = @as(f64, 1e-6),
            .max_position_embeddings = 131072,
            .tie_word_embeddings = false,
            .layer_types = &layers,
            .rope_parameters = .{ .rope_type = "default", .rope_theta = 10000000, .partial_rotary_factor = @as(f64, 0.25) },
        },
    }, .{});
}

fn failingOpen(a: std.mem.Allocator, primary: []const u8, alternative: []const u8) !void {
    var no_resize = t.FailingAllocator.init(a, .{ .resize_fail_index = 0 });
    const allocator = no_resize.allocator();
    var g = try geometry.Geometry.read(allocator, t.io, alternative);
    g.kinds = .{};
    g.bits = .{};
    var original = try core.Checkpoint.openModel(allocator, t.io, primary);
    defer core.checkpoint.cache.close(&original);
    var opened = source.openFrom(allocator, t.io, g, &original, alternative, .layers, primary) catch |err| {
        if (err == error.Nvfp4MlpBaseMismatch) return;
        return err;
    };
    opened.deinit();
    return error.ExpectedSourceMismatch;
}

test "NVFP4 source: real mapped preflight refuses a mismatched norm and frees every failed allocation" {
    var primary = t.tmpDir(.{});
    defer primary.cleanup();
    var other = t.tmpDir(.{});
    defer other.cleanup();
    try normFile(primary.dir, "language_model.model.layers.0.input_layernorm.weight", 0x3f81);
    try normFile(other.dir, "model.language_model.layers.0.input_layernorm.weight", 0);
    const config = try settings();
    defer t.allocator.free(config);
    try other.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = config });
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(t.allocator);
    try index.appendSlice(t.allocator, "{\"weight_map\":{");
    for (0..64) |i| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |slot, j| {
        try index.print(t.allocator, "{s}\"model.language_model.layers.{d}.mlp.{s}.weight_packed\":\"model.safetensors\"", .{ if (i == 0 and j == 0) "" else ",", i, slot });
    };
    try index.appendSlice(t.allocator, "}}");
    try other.dir.writeFile(t.io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    const p = try primary.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(p);
    const q = try other.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(q);
    try failingOpen(t.allocator, p, q);
    try t.checkAllAllocationFailures(t.allocator, failingOpen, .{ p, q });
}
