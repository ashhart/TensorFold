//! Synthetic configs are explicit; none claims to describe an unreleased checkpoint.
const std = @import("std");
const cfg = @import("config.zig");
const quant = @import("quant.zig");
const tiny = @embedFile("fixtures/tiny-dense.json");
const a = std.testing.allocator;
test "Qwen4 admission: explicit dense dimensions and both mixers" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, tiny, &p);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 64), c.wide());
    try std.testing.expectEqual(@as(usize, 16), c.convWidth());
    try std.testing.expectEqual(@as(usize, 32), c.intermediate);
    try std.testing.expectEqual(cfg.Kind.indexed_attention, c.kinds[1]);
    try std.testing.expectEqual(@as(usize, 1), c.ple_count);
    try std.testing.expectEqual(quant.Format.bf16, c.quant.default.format);
}
fn changed(field: []const u8, value: std.json.Value, wanted: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), tiny, .{});
    var o = doc.object;
    try o.put(arena.allocator(), field, value);
    const bytes = try std.json.Stringify.valueAlloc(arena.allocator(), std.json.Value{ .object = o }, .{});
    var p: cfg.Problem = .{};
    try std.testing.expectError(error.UnsupportedQwen4, cfg.parse(a, bytes, &p));
    try std.testing.expectEqualStrings(wanted, p.name());
}
test "Qwen4 admission: every unsupported semantic is refused by name" {
    try changed("new_rotary_method", .{ .string = "unknown" }, "new_rotary_method");
    try changed("model_type", .{ .string = "qwen4_exp" }, "model_type");
    try changed("intermediate_size", .{ .integer = 0 }, "intermediate_size");
    try changed("hidden_act", .{ .string = "gelu" }, "hidden_act");
    try changed("attention_bias", .{ .bool = true }, "attention_bias");
    try changed("attn_output_gate", .{ .bool = false }, "attn_output_gate");
    try changed("num_experts", .{ .integer = 512 }, "num_experts");
    try changed("mtp_num_hidden_layers", .{ .integer = 1 }, "mtp_num_hidden_layers");
    try changed("num_key_value_heads", .{ .integer = 3 }, "num_key_value_heads");
    try changed("indexer_kv_heads", .{ .integer = 2 }, "indexer_kv_heads");
    try changed("indexer_budget", .{ .integer = 7 }, "indexer_budget");
    try changed("hc_count", .{ .integer = 1 }, "hc_count");
    try changed("eos_token_id", .{ .integer = 32 }, "eos_token_id");
    try changed("linear_conv_kernel_dim", .{ .integer = 1 }, "linear_conv_kernel_dim");
    try changed("rms_norm_eps", .{ .float = 1e-100 }, "rms_norm_eps");
    try changed("hidden_size", .{ .integer = std.math.maxInt(i64) }, "dimensions");
}
test "Qwen4 admission: named layer roles and PLE placement cannot be discarded" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var types: std.json.Array = .init(arena.allocator());
    try types.append(.{ .string = "linear_attention" });
    try types.append(.{ .string = "sliding_attention" });
    try changed("layer_types", .{ .array = types }, "layer_types");
    var ple: std.json.Array = .init(arena.allocator());
    try ple.append(.{ .integer = 2 });
    try changed("ple_layer_ids", .{ .array = ple }, "ple_layer_ids");
}
fn readQuant(bytes: []const u8, p: *cfg.Problem) !struct { doc: std.json.Parsed(std.json.Value), q: quant.Config } {
    var doc = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    errdefer doc.deinit();
    return .{ .doc = doc, .q = try quant.parse(doc.value.object, doc.value.object, p) };
}
test "Qwen4 quant: MLX overrides and weight-only ModelOpt recipes" {
    var p: cfg.Problem = .{};
    var mlx = try readQuant(
        \\{"quantization":{"bits":8,"group_size":128,"model.layers.0.mlp.gate_proj":{"bits":4},"lm_head":false}}
    , &p);
    defer mlx.doc.deinit();
    try std.testing.expectEqual(quant.Recipe{ .format = .mlx8, .group = 128 }, try mlx.q.of("model.layers.0.mlp.up_proj", &p));
    try std.testing.expectEqual(quant.Recipe{ .format = .mlx4, .group = 64 }, try mlx.q.of("model.layers.0.mlp.gate_proj", &p));
    try std.testing.expectEqual(quant.Format.bf16, (try mlx.q.of("lm_head", &p)).format);
    var mo = try readQuant(
        \\{"quantization_config":{"quant_method":"modelopt","quant_algo":"MIXED_PRECISION","quantized_layers":{"model.layers.0.mlp.gate_proj":{"quant_algo":"W4A16_NVFP4","group_size":16},"model.layers.0.self_attn.q_proj":{"quant_algo":"FP8"}}}}
    , &p);
    defer mo.doc.deinit();
    try std.testing.expectEqual(quant.Format.nvfp4, (try mo.q.of("model.layers.0.mlp.gate_proj", &p)).format);
    try std.testing.expectEqual(quant.Format.fp8, (try mo.q.of("model.layers.0.self_attn.q_proj", &p)).format);
    try std.testing.expectEqual(quant.Format.bf16, (try mo.q.of("model.layers.0.mlp.up_proj", &p)).format);
}

test "Qwen4 quant: conflicting aliases refuse and matching aliases preserve the recipe" {
    var p: cfg.Problem = .{};
    var q = try readQuant(
        \\{"quantization":{"bits":4,"model.layers.0.mlp.gate_proj":{"bits":4},"language_model.model.layers.0.mlp.gate_proj":{"bits":8}}}
    , &p);
    defer q.doc.deinit();
    try std.testing.expectError(error.UnsupportedQwen4, q.q.of("model.layers.0.mlp.gate_proj", &p));
    try std.testing.expectEqualStrings("language_model.model.layers.0.mlp.gate_proj", p.name());
}

test "Qwen4 admission: wrapped embedding ties are inherited and conflicting declarations refuse" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var text = try std.json.parseFromSliceLeaky(std.json.Value, mem, tiny, .{});
    try text.object.put(mem, "model_type", .{ .string = "qwen4_text" });
    _ = text.object.swapRemove("tie_word_embeddings");
    var root: std.json.ObjectMap = .empty;
    try root.put(mem, "model_type", .{ .string = "qwen4" });
    try root.put(mem, "text_config", text);
    try root.put(mem, "tie_word_embeddings", .{ .bool = true });
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, try std.json.Stringify.valueAlloc(mem, std.json.Value{ .object = root }, .{}), &p);
    defer c.deinit();
    try std.testing.expect(c.tie_embeddings);
    try text.object.put(mem, "tie_word_embeddings", .{ .bool = false });
    try root.put(mem, "text_config", text);
    try std.testing.expectError(error.UnsupportedQwen4, cfg.parse(a, try std.json.Stringify.valueAlloc(mem, std.json.Value{ .object = root }, .{}), &p));
    try std.testing.expectEqualStrings("tie_word_embeddings", p.name());
}

test "Qwen4 admission: unknown rotary semantics and nested vision refuse by name" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var text = try std.json.parseFromSliceLeaky(std.json.Value, mem, tiny, .{});
    var rope = text.object.get("rope_parameters").?.object;
    try rope.put(mem, "new_frequency_rule", .{ .bool = true });
    try changed("rope_parameters", .{ .object = rope }, "new_frequency_rule");
    try text.object.put(mem, "rope_parameters", .{ .object = rope });
    var vision: std.json.ObjectMap = .empty;
    try vision.put(mem, "hidden_size", .{ .integer = 32 });
    try text.object.put(mem, "vision_config", .{ .object = vision });
    try text.object.put(mem, "model_type", .{ .string = "qwen4_text" });
    var root: std.json.ObjectMap = .empty;
    try root.put(mem, "model_type", .{ .string = "qwen4" });
    try root.put(mem, "text_config", text);
    var problem: cfg.Problem = .{};
    try std.testing.expectError(error.UnsupportedQwen4, cfg.parse(a, try std.json.Stringify.valueAlloc(mem, std.json.Value{ .object = root }, .{}), &problem));
    try std.testing.expectEqualStrings("vision_config", problem.name());
}

test "Qwen4 admission: generation stops never replace the text model's lexical EOS" {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    var text = try std.json.parseFromSliceLeaky(std.json.Value, mem, tiny, .{});
    try text.object.put(mem, "model_type", .{ .string = "qwen4_text" });
    var root: std.json.ObjectMap = .empty;
    try root.put(mem, "model_type", .{ .string = "qwen4" });
    try root.put(mem, "eos_token_id", .{ .integer = 9 });
    try root.put(mem, "text_config", text);
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, try std.json.Stringify.valueAlloc(mem, std.json.Value{ .object = root }, .{}), &p);
    defer c.deinit();
    try std.testing.expectEqual(@as(u32, 9), c.eos[0]);
    try std.testing.expectEqual(@as(u32, 2), c.ple_eos);
    _ = text.object.swapRemove("eos_token_id");
    try root.put(mem, "text_config", text);
    try std.testing.expectError(error.UnsupportedQwen4, cfg.parse(a, try std.json.Stringify.valueAlloc(mem, std.json.Value{ .object = root }, .{}), &p));
    try std.testing.expectEqualStrings("text_config.eos_token_id", p.name());
}
