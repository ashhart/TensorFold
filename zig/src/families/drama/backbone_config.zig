//! Admission for the Drama 3 (fish_omni) text backbone: the pinned Qwen3.5-35B-A3B hybrid geometry, block-FP8 experts.
const std = @import("std");

pub const hidden = 2048;
pub const vocab = 252288;
pub const layers = 40;
pub const full_interval = 4; // layers 3, 7, ..., 39 are full attention, the rest Gated DeltaNet
pub const full_layers = layers / full_interval;
pub const linear_layers = layers - full_layers;

pub const linear_key_heads = 16;
pub const linear_value_heads = 32;
pub const linear_dim = 128; // key and value head width
pub const conv_taps = 4;
pub const conv_dim = 2 * linear_key_heads * linear_dim + linear_value_heads * linear_dim; // q | k | v = 8192
pub const value_dim = linear_value_heads * linear_dim; // 4096 (z and the recurrence output)

pub const query_heads = 16;
pub const kv_heads = 2;
pub const head_dim = 256;
pub const rotary_dim = 64;
pub const theta: f64 = 10000000;

pub const experts = 256;
pub const top_k = 8;
pub const expert_inner = 512;
pub const shared_inner = 512;
pub const block = 128; // FP8 weight block rows and columns; activation group width

pub const eps: f32 = 1e-6;

pub const Config = struct {
    eos: u32,
    semantic_start: u32,
    semantic_end: u32,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};

/// Whether layer `index` is a Gated DeltaNet ("linear_attention") layer.
pub fn linear(index: usize) bool {
    return index % full_interval != full_interval - 1;
}

/// The layer's slot among its kind (DeltaNet state or attention cache index).
pub fn slot(index: usize) usize {
    return if (linear(index)) index - index / full_interval else index / full_interval;
}

fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadDramaConfig;
}

fn field(o: std.json.ObjectMap, key: []const u8) !std.json.Value {
    return o.get(key) orelse error.BadDramaConfig;
}

fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDramaConfig,
    };
}

fn integer(o: std.json.ObjectMap, key: []const u8) !u64 {
    const v = try field(o, key);
    if (v != .integer or v.integer < 0) return error.BadDramaConfig;
    return @intCast(v.integer);
}

fn string(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const v = try field(o, key);
    if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedDramaConfig;
}

fn boolean(o: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const v = try field(o, key);
    if (v != .bool or v.bool != expected) return error.UnsupportedDramaConfig;
}

fn quantization(o: std.json.ObjectMap) !void {
    try string(o, "quant_method", "bsgemm-moe");
    try string(o, "activation_scheme", "dynamic");
    const size = try field(o, "weight_block_size");
    if (size != .array or size.array.items.len != 2) return error.UnsupportedDramaQuantization;
    for (size.array.items) |s| if (s != .integer or s.integer != block) return error.UnsupportedDramaQuantization;
}

pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    try string(root, "model_type", "fish_omni");
    try quantization(try object(try field(root, "quantization_config")));
    const text = try object(try field(root, "text_config"));
    try string(text, "model_type", "fish_omni_text");
    try boolean(text, "tie_word_embeddings", false);
    try boolean(text, "attention_bias", false);
    try boolean(text, "attn_output_gate", true);
    try boolean(text, "use_qk_norm", true);
    try boolean(text, "layernorm_zero_centered_gamma", true);
    try boolean(text, "shared_expert_gate", true);
    // absent in the pinned text_config: Qwen3_5MoeTextConfig's default (true) applies, as SGLang's TopK reads it
    if (text.get("norm_topk_prob")) |v| if (v != .bool or !v.bool) return error.UnsupportedDramaConfig;
    try boolean(text, "rotary_interleaved", false);
    try string(text, "hidden_act", "silu");
    try string(text, "mamba_ssm_dtype", "float32");
    const dims = .{
        .{ "hidden_size", hidden },                    .{ "vocab_size", vocab },                          .{ "num_hidden_layers", layers },
        .{ "linear_num_key_heads", linear_key_heads }, .{ "linear_num_value_heads", linear_value_heads }, .{ "linear_key_head_dim", linear_dim },
        .{ "linear_value_head_dim", linear_dim },      .{ "linear_conv_kernel_dim", conv_taps },          .{ "num_attention_heads", query_heads },
        .{ "num_key_value_heads", kv_heads },          .{ "head_dim", head_dim },                         .{ "num_experts", experts },
        .{ "num_experts_per_tok", top_k },             .{ "moe_intermediate_size", expert_inner },        .{ "shared_expert_intermediate_size", shared_inner },
    };
    inline for (dims) |d| if (try integer(text, d[0]) != d[1]) return error.UnsupportedDramaGeometry;
    if (try number(try field(text, "rms_norm_eps")) != 1e-6) return error.UnsupportedDramaConfig;
    const types = try field(text, "layer_types");
    if (types != .array or types.array.items.len != layers) return error.UnsupportedDramaGeometry;
    for (types.array.items, 0..) |kind, i| {
        if (kind != .string or !std.mem.eql(u8, kind.string, if (linear(i)) "linear_attention" else "full_attention"))
            return error.UnsupportedDramaGeometry;
    }
    const mlp = try field(text, "mlp_layer_types");
    if (mlp != .array or mlp.array.items.len != layers) return error.UnsupportedDramaGeometry;
    for (mlp.array.items) |kind| if (kind != .string or !std.mem.eql(u8, kind.string, "moe")) return error.UnsupportedDramaGeometry;
    const rope = try object(try field(text, "rope_parameters"));
    try string(rope, "rope_type", "default");
    if (try number(try field(rope, "rope_theta")) != theta or
        try number(try field(rope, "partial_rotary_factor")) != 0.25) return error.UnsupportedDramaRotary;
    const audio = try object(try field(root, "audio_decoder_config"));
    if (try integer(audio, "vq_backbone_codebooks") != 1) return error.UnsupportedDramaConfig;
    try string(audio, "vq_feature_placement", "backbone_partial");
    const eos = try integer(root, "eos_token_id");
    const start = try integer(root, "semantic_start_token_id");
    const end = try integer(root, "semantic_end_token_id");
    if (try integer(text, "eos_token_id") != eos) return error.BadDramaConfig;
    // semantic logits are EOS then the 4096-code band; the band sits above EOS inside the vocabulary
    if (end < start or end - start + 1 != 4096 or end >= vocab or eos >= start) return error.UnsupportedDramaConfig;
    return .{ .eos = @intCast(eos), .semantic_start = @intCast(start), .semantic_end = @intCast(end) };
}

test "layer kinds and slots follow the 3 DeltaNet : 1 attention pattern" {
    try std.testing.expect(linear(0) and linear(2) and !linear(3) and !linear(39));
    try std.testing.expectEqual(@as(usize, 2), slot(2));
    try std.testing.expectEqual(@as(usize, 3), slot(4)); // the fourth DeltaNet layer
    try std.testing.expectEqual(@as(usize, 9), slot(39)); // the tenth attention layer
    try std.testing.expectEqual(@as(usize, linear_layers - 1), slot(38));
}
