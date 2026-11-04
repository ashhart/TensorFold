//! A Qwen3 dense checkpoint's config.json: the geometry the HIP forward runs, and refusals of what it does not.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{ InvalidConfig, UnsupportedConfig } || Allocator.Error;

/// The forward's geometry and constants, as config.json names them.
pub const Spec = struct {
    layers: usize,
    hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    inter: usize,
    vocab: usize,
    eps: f64,
    theta: f64,
    max_positions: usize,
    /// `tie_word_embeddings`: the output head is the embedding table.
    tied: bool,
};

/// The spec of config.json's `bytes`; a missing or malformed key is InvalidConfig, an unrun feature UnsupportedConfig.
pub fn parse(gpa: Allocator, bytes: []const u8) Error!Spec {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            std.log.warn("qwen3 config.json is not JSON ({s})", .{@errorName(err)});
            return error.InvalidConfig;
        },
    };
    if (doc != .object) return invalid("the document", "is not an object");
    const o = doc.object;
    const model_type = o.get("model_type") orelse return invalid("model_type", "is missing");
    if (model_type != .string or !std.mem.eql(u8, model_type.string, "qwen3")) return unsupported("model_type", "is not qwen3");
    const s: Spec = .{
        .layers = try count(o, "num_hidden_layers"),
        .hidden = try count(o, "hidden_size"),
        .heads = try count(o, "num_attention_heads"),
        .kv_heads = try count(o, "num_key_value_heads"),
        // Qwen3Config's own default when an older config.json leaves it out
        .head_dim = if (o.get("head_dim")) |v| (if (v == .null) 128 else try count(o, "head_dim")) else 128,
        .inter = try count(o, "intermediate_size"),
        .vocab = try count(o, "vocab_size"),
        .eps = try number(o, "rms_norm_eps"),
        .theta = try number(o, "rope_theta"),
        .max_positions = try count(o, "max_position_embeddings"),
        .tied = try flag(o, "tie_word_embeddings", false),
    };
    if (try flag(o, "attention_bias", false)) return unsupported("attention_bias", "is true: the forward has no projection biases");
    if (try flag(o, "use_sliding_window", false)) return unsupported("use_sliding_window", "is true: the forward attends to the whole input");
    if (o.get("rope_scaling")) |r| if (r != .null) return unsupported("rope_scaling", "is set: the forward runs plain RoPE");
    if (s.head_dim > 256) return unsupported("head_dim", "is above 256, the attention kernel's limit");
    if (s.heads % s.kv_heads != 0) return unsupported("num_attention_heads", "is not a multiple of num_key_value_heads");
    const act = o.get("hidden_act") orelse return invalid("hidden_act", "is missing");
    if (act != .string or !std.mem.eql(u8, act.string, "silu")) return unsupported("hidden_act", "is not silu");
    if (!(s.eps > 0) or !(s.theta > 0)) return invalid(if (s.eps > 0) "rope_theta" else "rms_norm_eps", "is not positive");
    return s;
}

fn invalid(key: []const u8, why: []const u8) error{InvalidConfig} {
    std.log.warn("qwen3 config.json: {s} {s}", .{ key, why });
    return error.InvalidConfig;
}

fn unsupported(key: []const u8, why: []const u8) error{UnsupportedConfig} {
    std.log.warn("qwen3 config.json: {s} {s}", .{ key, why });
    return error.UnsupportedConfig;
}

/// A positive integer key.
fn count(o: std.json.ObjectMap, key: []const u8) Error!usize {
    const v = o.get(key) orelse return invalid(key, "is missing");
    if (v != .integer or v.integer < 1) return invalid(key, "is not a positive integer");
    return @intCast(v.integer);
}

/// A number key, integer or float.
fn number(o: std.json.ObjectMap, key: []const u8) Error!f64 {
    const v = o.get(key) orelse return invalid(key, "is missing");
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => invalid(key, "is not a number"),
    };
}

/// A boolean key, `default` when absent or null.
fn flag(o: std.json.ObjectMap, key: []const u8, default: bool) Error!bool {
    const v = o.get(key) orelse return default;
    return switch (v) {
        .null => default,
        .bool => |b| b,
        else => invalid(key, "is not a boolean"),
    };
}

/// Qwen3-Embedding-8B's config.json (the reranker's differs only in vocab_size 151669).
const embedding =
    \\{"architectures": ["Qwen3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
    \\ "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu",
    \\ "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 12288,
    \\ "max_position_embeddings": 40960, "max_window_layers": 36, "model_type": "qwen3",
    \\ "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
    \\ "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null, "tie_word_embeddings": false,
    \\ "torch_dtype": "bfloat16", "transformers_version": "4.51.2", "use_cache": true,
    \\ "use_sliding_window": false, "vocab_size": 151665}
;

test "the 8B embedding and reranker configs give their geometry" {
    const a = std.testing.allocator;
    const s = try parse(a, embedding);
    try std.testing.expectEqual(Spec{ .layers = 36, .hidden = 4096, .heads = 32, .kv_heads = 8, .head_dim = 128, .inter = 12288, .vocab = 151665, .eps = 1e-6, .theta = 1e6, .max_positions = 40960, .tied = false }, s);
    const reranker = try std.mem.replaceOwned(u8, a, embedding, "\"vocab_size\": 151665", "\"vocab_size\": 151669");
    defer a.free(reranker);
    try std.testing.expectEqual(@as(usize, 151669), (try parse(a, reranker)).vocab);
    const older = try std.mem.replaceOwned(u8, a, embedding, "\"head_dim\": 128, ", "");
    defer a.free(older);
    try std.testing.expect(!std.mem.eql(u8, older, embedding));
    try std.testing.expectEqual(@as(usize, 128), (try parse(a, older)).head_dim);
}

test "each feature the forward does not run is refused by name, and each malformed key is invalid" {
    const a = std.testing.allocator;
    const Case = struct { []const u8, []const u8, Error };
    const cases = [_]Case{
        .{ "\"model_type\": \"qwen3\"", "\"model_type\": \"qwen3_moe\"", error.UnsupportedConfig },
        .{ "\"attention_bias\": false", "\"attention_bias\": true", error.UnsupportedConfig },
        .{ "\"use_sliding_window\": false", "\"use_sliding_window\": true", error.UnsupportedConfig },
        .{ "\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"yarn\", \"factor\": 4}", error.UnsupportedConfig },
        .{ "\"head_dim\": 128", "\"head_dim\": 512", error.UnsupportedConfig },
        .{ "\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\"", error.UnsupportedConfig },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 7", error.UnsupportedConfig },
        .{ "\"model_type\": \"qwen3\",", "", error.InvalidConfig },
        .{ "\"hidden_size\": 4096", "\"hidden_size\": 0", error.InvalidConfig },
        .{ "\"num_hidden_layers\": 36", "\"num_hidden_layers\": \"36\"", error.InvalidConfig },
        .{ "\"intermediate_size\": 12288,", "", error.InvalidConfig },
        .{ "\"rms_norm_eps\": 1e-06", "\"rms_norm_eps\": 0", error.InvalidConfig },
        .{ "\"rope_theta\": 1000000", "\"rope_theta\": \"1e6\"", error.InvalidConfig },
        .{ "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": 0", error.InvalidConfig },
        .{ "\"hidden_act\": \"silu\",", "", error.InvalidConfig },
    };
    for (cases) |case| {
        const bytes = try std.mem.replaceOwned(u8, a, embedding, case[0], case[1]);
        defer a.free(bytes);
        try std.testing.expect(!std.mem.eql(u8, bytes, embedding));
        try std.testing.expectError(case[2], parse(a, bytes));
    }
    try std.testing.expectError(error.InvalidConfig, parse(a, "[1]"));
    try std.testing.expectError(error.InvalidConfig, parse(a, "{not json"));
}
