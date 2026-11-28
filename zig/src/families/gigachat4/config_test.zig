//! config.json: both releases parse to their shapes, the BF16 copy reads as BF16, and every refusal names its key.
const std = @import("std");
const config = @import("config.zig");

const t = std.testing;
/// The 708B release's config.json, byte for byte (MIT); the 37B shape is that file with `small` applied.
pub const json_708b = @embedFile("fixtures/config_708b.json");
/// A key set to JSON text, or removed when the text is null.
pub const Edit = struct { []const u8, ?[]const u8 };

/// The keys in which the 37B release's config.json differs from the 708B's, with the 37B's values.
pub const small = [_]Edit{
    .{ "hidden_size", "2560" },
    .{ "intermediate_size", "8192" },
    .{ "moe_intermediate_size", "768" },
    .{ "num_hidden_layers", "32" },
    .{ "num_attention_heads", "32" },
    .{ "num_key_value_heads", "32" },
    .{ "n_routed_experts", "192" },
    .{ "first_k_dense_replace", "1" },
    .{ "initializer_range", "0.006" },
    .{ "full_attention_layers", "[3, 7, 11, 15, 19, 23, 27, 31]" },
    .{ "linear_num_key_heads", "16" },
    .{ "linear_num_value_heads", "32" },
    .{ "swiglu_limit", "10.0" },
};

/// `base` with `edits` applied, as JSON text in `a`.
pub fn edited(a: std.mem.Allocator, base: []const u8, edits: []const Edit) ![]const u8 {
    var doc = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    for (edits) |e| {
        if (e[1]) |v| {
            try doc.object.put(a, e[0], try std.json.parseFromSliceLeaky(std.json.Value, a, v, .{}));
        } else _ = doc.object.orderedRemove(e[0]);
    }
    return std.json.Stringify.valueAlloc(a, doc, .{});
}

/// The 37B release's config.json as the edits rebuild it.
pub fn json37b(a: std.mem.Allocator) ![]const u8 {
    return edited(a, json_708b, &small);
}

/// The 37B BF16 copy's config.json: the 37B's without quantization_config.
pub fn json37bBf16(a: std.mem.Allocator) ![]const u8 {
    return edited(a, try json37b(a), &.{.{ "quantization_config", null }});
}

fn layersOf(c: *const config.Config, out: []u32) []u32 {
    var n: usize = 0;
    for (0..c.layers) |i| if (c.isMla(@intCast(i))) {
        out[n] = @intCast(i);
        n += 1;
    };
    return out[0..n];
}

test "the 708B release parses to its published shape" {
    var p: config.Problem = .{};
    const c = try config.parse(t.allocator, json_708b, &p);
    try t.expectEqual(@as(u32, 7168), c.hidden);
    try t.expectEqual(@as(u32, 64), c.layers);
    try t.expectEqual(@as(u32, 128256), c.vocab);
    try t.expectEqual(@as(u32, 1048576), c.max_context);
    try t.expectEqual(config.Storage.fp8, c.storage);
    var buf: [config.max_layers]u32 = undefined;
    try t.expectEqualSlices(u32, &.{ 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63 }, layersOf(&c, &buf));
    try t.expectEqual(config.Mla{ .heads = 64, .q_lora = 1536, .kv_lora = 512, .nope = 128, .rope = 64, .value = 128 }, c.mla);
    try t.expectEqual(config.Indexer{ .heads = 64, .dim = 128, .topk = 2048, .eps = 1e-6 }, c.index);
    try t.expectEqual(config.DeltaNet{ .key_heads = 32, .value_heads = 64, .key_dim = 128, .value_dim = 128, .conv = 4, .gate_scale = 2, .eps = 1e-6 }, c.delta);
    try t.expectEqual(config.Moe{ .experts = 256, .topk = 8, .inter = 2048, .shared = 1, .dense_layers = 3, .dense_inter = 18432, .scale = 2.5 }, c.moe);
    try t.expectEqual(config.Norm{ .eps = 1e-6, .gating_weight = 2 }, c.norm);
    try t.expectEqual(config.Rope{ .theta = 100000, .factor = 32, .original = 32768, .beta_fast = 32, .beta_slow = 1, .mscale = 1, .mscale_all_dim = 1 }, c.rope);
    try t.expectEqual(@as(f64, 10), c.swiglu_limit);
    try t.expectEqual(@as(f64, 30), c.softcap);
    try t.expectEqual(@as(u32, 1), c.mtp_layers);
}

test "the 37B release and its BF16 copy parse to the 37B shape" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var p: config.Problem = .{};
    const c = try config.parse(t.allocator, try json37b(a), &p);
    try t.expectEqual(@as(u32, 2560), c.hidden);
    try t.expectEqual(@as(u32, 32), c.layers);
    var buf: [config.max_layers]u32 = undefined;
    try t.expectEqualSlices(u32, &.{ 3, 7, 11, 15, 19, 23, 27, 31 }, layersOf(&c, &buf));
    try t.expectEqual(config.Mla{ .heads = 32, .q_lora = 1536, .kv_lora = 512, .nope = 128, .rope = 64, .value = 128 }, c.mla);
    try t.expectEqual(config.DeltaNet{ .key_heads = 16, .value_heads = 32, .key_dim = 128, .value_dim = 128, .conv = 4, .gate_scale = 2, .eps = 1e-6 }, c.delta);
    try t.expectEqual(config.Moe{ .experts = 192, .topk = 8, .inter = 768, .shared = 1, .dense_layers = 1, .dense_inter = 8192, .scale = 2.5 }, c.moe);
    try t.expect(c.isDense(0) and !c.isDense(1) and c.isDsa(3) and !c.isMla(0));
    try t.expectEqual(config.Storage.fp8, c.storage);
    const b = try config.parse(t.allocator, try json37bBf16(a), &p);
    try t.expectEqual(config.Storage.bf16, b.storage);
    try t.expectEqual(c.mla, b.mla);
    try t.expectEqual(c.moe, b.moe);
}

test "a pinned key with another value is refused by name, and so is a missing one" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, ?[]const u8, anyerror, []const u8 }{
        .{ "norm_type", "\"LlamaRMSNorm\"", error.Unsupported, "norm_type must be \"GatedNorm\": the weight map follows it" },
        .{ "layernorm_type", "\"pre\"", error.Unsupported, "layernorm_type must be \"pre_post\"" },
        .{ "linear_gating_type", "\"sigmoid_gate\"", error.Unsupported, "the reference code runs no other value here" },
        .{ "use_shared_expert_sigmoid", "true", error.Unsupported, "use_shared_expert_sigmoid must be false" },
        .{ "num_nextn_predict_layers", "2", error.Unsupported, "num_nextn_predict_layers must be 1" },
        .{ "nextn_is_sparse", "true", error.Unsupported, "its meaning is not read yet" },
        .{ "indexer_rotate_activation_enabled", "true", error.Unsupported, "indexer_rotate_activation_enabled must be null" },
        .{ "indexer_norm_type", "\"GatedNorm\"", error.Unsupported, "indexer_norm_type must be \"LlamaRMSNorm\" or absent" },
        .{ "gated_attention", null, error.BadConfig, "config.json: gated_attention is missing" },
        .{ "model_type", "\"deepseek_v3\"", error.NotGigaChat4, "not a GigaChat 4 checkpoint" },
        .{ "rope_scaling", "{\"type\": \"linear\", \"factor\": 4}", error.Unsupported, "type must be \"yarn\"" },
    };
    for (cases) |c| {
        var p: config.Problem = .{};
        try t.expectError(c[2], config.parse(t.allocator, try edited(a, json_708b, &.{.{ c[0], c[1] }}), &p));
        if (std.mem.indexOf(u8, p.text(), c[3]) == null) {
            std.debug.print("{s}: got \"{s}\"\n", .{ c[0], p.text() });
            return error.TestUnexpectedRefusal;
        }
    }
}

test "layouts the map cannot follow are refused by name" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, []const u8, anyerror, []const u8 }{
        .{ "full_attention_layers", "[3, 64]", error.Layout, "names a layer past the 64 layers" },
        .{ "full_attention_layers", "[3, 3]", error.Layout, "names layer 3 twice" },
        .{ "full_attention_layers", "[]", error.Layout, "full_attention_layers must list layers" },
        .{ "dense_attention_layers", "[4]", error.Layout, "dense_attention_layers must be a subset" },
        .{ "num_key_value_heads", "8", error.Layout, "as many key/value heads as query heads" },
        .{ "qk_head_dim", "128", error.Layout, "qk_head_dim must be qk_nope + qk_rope" },
        .{ "linear_num_key_heads", "48", error.Layout, "value heads must split over key heads" },
        .{ "n_routed_experts", "4096", error.Layout, "n_routed_experts must not exceed 1024" },
        .{ "num_experts_per_tok", "300", error.Layout, "num_experts_per_tok exceeds n_routed_experts" },
        .{ "index_head_dim", "32", error.Layout, "index_head_dim must hold the qk_rope dims" },
        .{ "first_k_dense_replace", "65", error.BadConfig, "first_k_dense_replace is 65, outside 0..64" },
        .{ "num_hidden_layers", "129", error.Layout, "129 layers, the plan tables hold 128" },
    };
    for (cases) |c| {
        var p: config.Problem = .{};
        try t.expectError(c[2], config.parse(t.allocator, try edited(a, json_708b, &.{.{ c[0], c[1] }}), &p));
        if (std.mem.indexOf(u8, p.text(), c[3]) == null) {
            std.debug.print("{s}: got \"{s}\"\n", .{ c[0], p.text() });
            return error.TestUnexpectedRefusal;
        }
    }
}

test "only the release's FP8 layout or no quantization is admitted" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const q = "{\"activation_scheme\": \"dynamic\", \"fmt\": \"e4m3\", \"quant_method\": \"fp8\", \"weight_block_size\": [";
    for ([_][]const u8{
        q ++ "128, 64]}",
        "{\"quant_method\": \"fp8\", \"fmt\": \"e5m2\", \"activation_scheme\": \"dynamic\", \"weight_block_size\": [128, 128]}",
        "{\"quant_method\": \"fp8\", \"fmt\": \"e4m3\", \"activation_scheme\": \"static\", \"weight_block_size\": [128, 128]}",
        q ++ "128, 128], \"ignored_layers\": [\"lm_head\"]}",
        q ++ "128, 128]}",
        "\"fp8\"",
    }) |bad| {
        var p: config.Problem = .{};
        try t.expectError(error.Quant, config.parse(t.allocator, try edited(a, json_708b, &.{.{ "quantization_config", bad }}), &p));
        try t.expect(std.mem.indexOf(u8, p.text(), "quantization_config") != null);
    }
}

test "the 37B release's own config.json parses as the edits rebuild it (TF_GIGACHAT4_CONFIG_37B)" {
    const path = std.mem.span(std.c.getenv("TF_GIGACHAT4_CONFIG_37B") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const real = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 20));
    var p: config.Problem = .{};
    const want = try config.parse(t.allocator, try json37b(a), &p);
    const got = try config.parse(t.allocator, real, &p);
    try t.expect(std.meta.eql(want, got));
    const r = try std.json.parseFromSliceLeaky(std.json.Value, a, real, .{});
    const e = try std.json.parseFromSliceLeaky(std.json.Value, a, try json37b(a), .{});
    try t.expectEqual(r.object.count(), e.object.count());
}
