//! Kolibri 1.1's config fields over Kolibri-1's: dense MLP layers, layers reading another's cache, the context cap.
const std = @import("std");
const config = @import("config.zig");
const Config = config.Config;
const Why = config.Why;

fn ints(o: std.json.ObjectMap, key: []const u8, why: *Why) ![]const std.json.Value {
    const v = o.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    if (v != .array) {
        why.set("config.json's {s} is not a list", .{key});
        return error.BadConfig;
    }
    for (v.array.items) |x| if (x != .integer or x.integer < 0) {
        why.set("config.json's {s} holds something other than a layer number", .{key});
        return error.BadConfig;
    };
    return v.array.items;
}

fn positive(o: std.json.ObjectMap, key: []const u8, why: *Why) !usize {
    const v = o.get(key) orelse {
        why.set("config.json has no {s}", .{key});
        return error.BadConfig;
    };
    if (v != .integer or v.integer <= 0) {
        why.set("config.json's {s} is not a positive integer", .{key});
        return error.BadConfig;
    }
    return @intCast(v.integer);
}

/// The 1.1 fields into `c`, whose layer kinds `parse` has read (`full`: indexed_attention).
pub fn extend(c: *Config, o: std.json.ObjectMap, why: *Why) !void {
    c.v11 = true;
    c.dense_width = try positive(o, "mlp_only_intermediate_size", why);
    if (c.dense_width % 128 != 0) {
        why.set("dense layers {d} wide: this engine takes whole 128 blocks", .{c.dense_width});
        return error.UnsupportedShapes;
    }
    for (try ints(o, "mlp_only_layers", why)) |x| {
        const i: usize = @intCast(x.integer);
        if (i >= c.layers or c.dense[i]) {
            why.set("mlp_only_layers names layer {d} twice or past the {d} layers", .{ i, c.layers });
            return error.BadConfig;
        }
        c.dense[i] = true;
    }
    const src = try ints(o, "kv_source_layers", why);
    if (src.len != c.layers) {
        why.set("kv_source_layers names {d} layers, not {d}", .{ src.len, c.layers });
        return error.BadConfig;
    }
    for (src, 0..) |x, i| {
        const s: usize = @intCast(x.integer);
        if (s == i) continue;
        // a layer reads an earlier indexed layer's own cache; sliding layers keep theirs (rings, RoPE'd keys)
        if (s > i or !c.full[i] or !c.full[s] or c.kv_from[s] != null) {
            why.set("kv_source_layers: layer {d} reads layer {d}'s keys; this engine takes an earlier indexed layer that keeps its own", .{ i, s });
            return error.UnsupportedLayer;
        }
        c.kv_from[i] = @intCast(s);
    }
    // where every key is still attended: the selected, the local and the sink keys together (card, blog)
    c.max_context = try positive(o, "index_topk", why) + try positive(o, "index_num_local_tokens", why) +
        try positive(o, "index_num_sink_tokens", why);
}

/// Kolibri 1.1's config.json at e9515166 (indexed layers every fifth), its module list cut to two entries.
pub const pinned =
    \\{"model_type": "kolibri1_1", "hidden_size": 2560, "num_hidden_layers": 50, "num_attention_heads": 48,
    \\ "num_key_value_heads": 4, "head_dim": 128, "rms_norm_eps": 1e-06, "vocab_size": 128000, "rope_theta": 10000.0,
    \\ "num_experts": 384, "num_experts_per_tok": 6, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
    \\ "mlp_only_layers": [0, 1], "mlp_only_intermediate_size": 3584,
    \\ "norm_topk_prob": false, "sliding_window": 513, "eos_token_id": 127906, "pad_token_id": 127901,
    \\ "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
    \\  "indexed_attention"],
    \\ "index_layers": [4, 14, 19, 29],
    \\ "kv_source_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 4, 10, 11, 12, 13, 4, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    \\  25, 26, 27, 28, 29, 30, 31, 32, 33, 29, 35, 36, 37, 38, 29, 40, 41, 42, 43, 29, 45, 46, 47, 48, 4],
    \\ "index_topk": 2048, "index_pool_size": 16384, "index_compression": 16, "index_num_local_tokens": 128,
    \\ "index_num_sink_tokens": 8, "index_num_heads": 12, "kv_cache_qat": true,
    \\ "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [128, 128],
    \\  "modules_to_not_convert": ["lm_head", "model.layers.0.self_attn.q_proj"]}}
;

fn replaced(from: []const u8, to: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, std.testing.allocator, pinned, from, to);
}

fn refused(text: []const u8, err: anyerror, word: []const u8) !void {
    var why: Why = .{};
    try std.testing.expectError(err, config.parse(std.testing.allocator, text, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

test "kolibri1_1: dense layers, the shared caches and the 2,184-key cap from the pinned config" {
    var why: Why = .{};
    const c = try config.parse(std.testing.allocator, pinned, &why);
    try config.checkShapes(c, &why);
    try std.testing.expect(c.v11 and c.dense[0] and c.dense[1] and !c.dense[2]);
    try std.testing.expectEqual(@as(usize, 3584), c.dense_width);
    try std.testing.expect(c.full[4] and c.full[49] and !c.full[5]);
    const want = [_]struct { usize, ?u8 }{ .{ 4, null }, .{ 9, 4 }, .{ 14, 4 }, .{ 19, null }, .{ 24, null }, .{ 29, null }, .{ 34, 29 }, .{ 39, 29 }, .{ 44, 29 }, .{ 49, 4 }, .{ 0, null } };
    for (want) |x| try std.testing.expectEqual(x[1], c.kv_from[x[0]]);
    try std.testing.expectEqual(@as(usize, 2184), c.max_context);
}

test "kolibri1_1 refusals: a sliding or later source, a source that borrows, bad dense lists, 1.1 fields on kolibri1" {
    const sliding = try replaced("[0, 1, 2, 3, 4,", "[0, 0, 2, 3, 4,");
    defer std.testing.allocator.free(sliding);
    try refused(sliding, error.UnsupportedLayer, "layer 1 reads layer 0");
    const chain = try replaced("29, 45, 46, 47, 48, 4]", "29, 45, 46, 47, 48, 9]");
    defer std.testing.allocator.free(chain);
    try refused(chain, error.UnsupportedLayer, "layer 49 reads layer 9");
    const later = try replaced("23, 24,", "23, 29,");
    defer std.testing.allocator.free(later);
    try refused(later, error.UnsupportedLayer, "layer 24 reads layer 29");
    const short = try replaced(", 29, 45, 46, 47, 48, 4]", ", 29, 45, 46, 47, 48]");
    defer std.testing.allocator.free(short);
    try refused(short, error.BadConfig, "49 layers, not 50");
    const twice = try replaced("\"mlp_only_layers\": [0, 1]", "\"mlp_only_layers\": [0, 0]");
    defer std.testing.allocator.free(twice);
    try refused(twice, error.BadConfig, "layer 0 twice");
    const odd = try replaced("\"mlp_only_intermediate_size\": 3584", "\"mlp_only_intermediate_size\": 3500");
    defer std.testing.allocator.free(odd);
    try refused(odd, error.UnsupportedShapes, "3500 wide");
    const nocap = try replaced("\"index_topk\": 2048, ", "");
    defer std.testing.allocator.free(nocap);
    try refused(nocap, error.BadConfig, "index_topk");
    const old = try std.mem.replaceOwned(u8, std.testing.allocator, config.pinned, "\"norm_topk_prob\"", "\"mlp_only_layers\": [0], \"norm_topk_prob\"");
    defer std.testing.allocator.free(old);
    try refused(old, error.BadConfig, "belong to kolibri1_1");
    const mixed = try replaced("\"indexed_attention\"]", "\"full_attention\"]");
    defer std.testing.allocator.free(mixed);
    try refused(mixed, error.UnsupportedLayer, "full_attention");
}
