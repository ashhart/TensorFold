//! Qwen35 GGUF metadata expressed as the existing native checkpoint config, with every required shape checked.
const std = @import("std");
const gg = @import("../../core/gguf.zig");
const cfg = @import("config.zig");
pub const Config = cfg.Config;
pub fn required(d: *const gg.Document, key: []const u8) !usize {
    return std.math.cast(usize, try (d.get(key) orelse return error.MissingGgufMetadata).unsigned()) orelse error.BadGgufConfig;
}
fn string(a: std.mem.Allocator, s: []const u8) !std.json.Value {
    return .{ .string = try a.dupe(u8, s) };
}
fn object() std.json.Value {
    return .{ .object = .empty };
}
fn integer(v: usize) !std.json.Value {
    return .{ .integer = std.math.cast(i64, v) orelse return error.BadGgufConfig };
}
pub fn set(a: std.mem.Allocator, v: *std.json.Value, k: []const u8, x: std.json.Value) !void {
    try v.object.put(a, k, x);
}
pub fn build(a: std.mem.Allocator, d: *const gg.Document) !std.json.Value {
    const v = try d.config();
    if (!std.mem.eql(u8, v.architecture, "qwen35")) return error.NotQwenDenseGguf;
    if (v.rope_scaling) |s| if (!std.mem.eql(u8, s, "none")) return error.UnsupportedRopeScaling;
    if (d.get("qwen35.expert_count")) |x| if (try x.unsigned() != 0) return error.NotQwenDenseGguf;
    var out = object();
    try set(a, &out, "model_type", try string(a, "qwen3_5"));
    try set(a, &out, "tie_word_embeddings", .{ .bool = false });
    const fields = [_][2][]const u8{
        .{ "hidden_size", "qwen35.embedding_length" },                .{ "intermediate_size", "qwen35.feed_forward_length" },
        .{ "num_hidden_layers", "qwen35.block_count" },               .{ "num_attention_heads", "qwen35.attention.head_count" },
        .{ "num_key_value_heads", "qwen35.attention.head_count_kv" }, .{ "head_dim", "qwen35.attention.key_length" },
        .{ "linear_num_key_heads", "qwen35.ssm.group_count" },        .{ "linear_num_value_heads", "qwen35.ssm.time_step_rank" },
        .{ "linear_key_head_dim", "qwen35.ssm.state_size" },          .{ "linear_conv_kernel_dim", "qwen35.ssm.conv_kernel" },
        .{ "max_position_embeddings", "qwen35.context_length" },      .{ "eos_token_id", "tokenizer.ggml.eos_token_id" },
    };
    for (fields) |f| try set(a, &out, f[0], try integer(try required(d, f[1])));
    const heads = try required(d, "qwen35.ssm.time_step_rank");
    const inner = try required(d, "qwen35.ssm.inner_size");
    if (heads == 0 or inner % heads != 0) return error.BadGgufConfig;
    try set(a, &out, "linear_value_head_dim", try integer(inner / heads));
    if (try required(d, "qwen35.attention.value_length") != try required(d, "qwen35.attention.key_length")) return error.BadGgufConfig;
    const tokens = (try d.tokenizer()).tokens orelse return error.MissingGgufTokens;
    const embed = d.getTensor("token_embd.weight") orelse return error.MissingGgufTensor;
    if (embed.rank != 2 or embed.shape[1] != tokens.len or try required(d, "tokenizer.ggml.eos_token_id") >= tokens.len) return error.BadGgufVocabulary;
    try set(a, &out, "vocab_size", try integer(tokens.len));
    const eps = v.rms_epsilon orelse return error.MissingGgufMetadata;
    const theta = v.rope_base orelse return error.MissingGgufMetadata;
    if (!std.math.isFinite(eps) or !std.math.isFinite(theta)) return error.BadGgufConfig;
    try set(a, &out, "rms_norm_eps", .{ .float = eps });
    const hd = try required(d, "qwen35.attention.key_length");
    const rd = try required(d, "qwen35.rope.dimension_count");
    if (hd == 0 or rd == 0 or rd > hd or rd % 2 != 0) return error.BadGgufConfig;
    try set(a, &out, "partial_rotary_factor", .{ .float = @as(f64, @floatFromInt(rd)) / @as(f64, @floatFromInt(hd)) });
    var rope = object();
    try set(a, &rope, "rope_theta", .{ .float = theta });
    try set(a, &rope, "mrope_interleaved", .{ .bool = true });
    const sections = d.get("qwen35.rope.dimension_sections") orelse return error.MissingGgufMetadata;
    if (sections != .array or sections.array.len != 4 or sections.array.kind != .int32) return error.BadGgufConfig;
    var dims: std.json.Value = .{ .array = std.json.Array.init(a) };
    var sum: usize = 0;
    for (0..4) |i| {
        const x = try (try sections.array.at(i)).unsigned();
        if (i == 3) {
            if (x != 0) return error.UnsupportedMrope;
        } else {
            const n = std.math.cast(usize, x) orelse return error.BadGgufConfig;
            sum = try std.math.add(usize, sum, n);
            try dims.array.append(try integer(n));
        }
    }
    if (sum != rd / 2) return error.BadGgufConfig;
    try set(a, &rope, "mrope_section", dims);
    try set(a, &out, "rope_parameters", rope);
    const count = try required(d, "qwen35.block_count");
    if (count == 0 or count > cfg.max_layers) return error.BadGgufConfig;
    var layers: std.json.Value = .{ .array = std.json.Array.init(a) };
    if (d.get("qwen35.attention.recurrent_layers")) |schedule| {
        if (schedule != .array or schedule.array.kind != .boolean or schedule.array.len != count) return error.BadGgufConfig;
        var it = schedule.array.iterator();
        while (try it.next()) |k| try layers.array.append(try string(a, if (k.boolean) "linear_attention" else "full_attention"));
    } else {
        const interval = try required(d, "qwen35.full_attention_interval");
        if (interval == 0) return error.BadGgufConfig;
        for (0..count) |i| try layers.array.append(try string(a, if ((i + 1) % interval == 0) "full_attention" else "linear_attention"));
    }
    try set(a, &out, "layer_types", layers);
    return out;
}
pub fn parse(a: std.mem.Allocator, value: std.json.Value) !Config {
    const text = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(text);
    return cfg.parse(a, text, null);
}
