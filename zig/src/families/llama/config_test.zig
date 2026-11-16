//! Llama admission on the mainstream checkpoints' published configs: Llama 3's rope, tied heads, Mistral, MLX.
const std = @import("std");
const cfg = @import("config.zig");
const rope_llama3 = @import("../../core/rope_llama3.zig");
const t = std.testing;

const llama31: cfg.Llama3 = .{ .factor = 8, .low = 1, .high = 4, .original = 8192 };
const llama32: cfg.Llama3 = .{ .factor = 32, .low = 1, .high = 4, .original = 8192 };

/// The rows a device qualification would add for the mainstream checkpoints, their geometry as published.
pub const expected = [_]cfg.Qualified{
    .{ .name = "Llama-3.1-8B-Instruct", .role = .served, .weights = .bf16, .drafted = false, .hidden = 4096, .intermediate = 14336, .layers = 32, .heads = 32, .kv_heads = 8, .vocab = 128256, .theta = 500000, .context = 131072, .llama3 = llama31 },
    .{ .name = "Llama-3.3-70B-Instruct", .role = .served, .weights = .bf16, .drafted = false, .hidden = 8192, .intermediate = 28672, .layers = 80, .heads = 64, .kv_heads = 8, .vocab = 128256, .theta = 500000, .context = 131072, .llama3 = llama31 },
    .{ .name = "Llama-3.2-3B-Instruct", .role = .served, .weights = .bf16, .drafted = false, .hidden = 3072, .intermediate = 8192, .layers = 28, .heads = 24, .kv_heads = 8, .vocab = 128256, .theta = 500000, .context = 131072, .llama3 = llama32, .tied = true },
    .{ .name = "Mistral-7B-Instruct-v0.3", .role = .served, .weights = .bf16, .drafted = false, .hidden = 4096, .intermediate = 14336, .layers = 32, .heads = 32, .kv_heads = 8, .vocab = 32768, .theta = 1000000, .context = 32768 },
    .{ .name = "Mistral-Nemo-Instruct-2407", .role = .served, .weights = .bf16, .drafted = false, .hidden = 5120, .intermediate = 14336, .layers = 40, .heads = 32, .kv_heads = 8, .vocab = 131072, .theta = 1000000, .context = 131072 },
    .{ .name = "Mistral-Small-24B-Instruct-2501", .role = .served, .weights = .bf16, .drafted = false, .hidden = 5120, .intermediate = 32768, .layers = 40, .heads = 32, .kv_heads = 8, .vocab = 131072, .theta = 100000000, .context = 32768 },
    .{ .name = "Llama-3.2-3B-Instruct MLX 4-bit", .role = .served, .weights = .mlx_q4_g64, .drafted = false, .hidden = 3072, .intermediate = 8192, .layers = 28, .heads = 24, .kv_heads = 8, .vocab = 128256, .theta = 500000, .context = 131072, .llama3 = llama32, .tied = true },
};

const published = [_]struct { []const u8, []const u8 }{
    .{ "Llama-3.1-8B-Instruct", @embedFile("fixtures/llama-3.1-8b.json") },
    .{ "Llama-3.3-70B-Instruct", @embedFile("fixtures/llama-3.3-70b.json") },
    .{ "Llama-3.2-3B-Instruct", @embedFile("fixtures/llama-3.2-3b.json") },
    .{ "Mistral-7B-Instruct-v0.3", @embedFile("fixtures/mistral-7b-v0.3.json") },
    .{ "Mistral-Nemo-Instruct-2407", @embedFile("fixtures/mistral-nemo-12b.json") },
    .{ "Mistral-Small-24B-Instruct-2501", @embedFile("fixtures/mistral-small-24b.json") },
    .{ "Llama-3.2-3B-Instruct MLX 4-bit", @embedFile("fixtures/llama-3.2-3b-mlx-4bit.json") },
};
const nanbeige = @embedFile("fixtures/config.json");
const mistral7b = published[3][1];
const nemo = published[4][1];

test "llama config: each published config parses, names its expected row and no qualified one" {
    for (published) |p| {
        const c = try cfg.parse(t.allocator, p[1]);
        try t.expectEqualStrings(p[0], c.qualificationIn(&expected).?.name);
        try t.expectError(error.UnqualifiedLlamaCheckpoint, c.qualification());
    }
    // the qualified table keeps its rows: Nanbeige's own configs still name them
    try t.expectEqualStrings("Nanbeige4.3-33B", (try (try cfg.parse(t.allocator, nanbeige)).qualification()).name);
    try t.expectEqual(@as(?*const cfg.Qualified, null), (try cfg.parse(t.allocator, nanbeige)).qualificationIn(&expected));
}

test "llama config: Llama 3.1's and 3.2's rope tables are transformers' words, the default rope's unchanged" {
    var f: [cfg.head_dim / 2]f32 = undefined;
    const c31 = try cfg.parse(t.allocator, published[0][1]);
    try t.expectEqual(llama31, c31.llama3.?);
    c31.inverseFrequencies(&f);
    for (rope_llama3.llama31_words, f) |w, g| try t.expectEqual(w, @as(u32, @bitCast(g)));
    const c32 = try cfg.parse(t.allocator, published[2][1]);
    c32.inverseFrequencies(&f);
    for (rope_llama3.llama32_words, f) |w, g| try t.expectEqual(w, @as(u32, @bitCast(g)));
    // a default rope's table is the family's 1 / theta^(2i / 128), rounded once
    const m = try cfg.parse(t.allocator, mistral7b);
    try t.expectEqual(@as(?cfg.Llama3, null), m.llama3);
    m.inverseFrequencies(&f);
    for (f, 0..) |g, i| try t.expectEqual(@as(f32, @floatCast(1.0 / std.math.pow(f64, 1e6, @as(f64, @floatFromInt(2 * i)) / 128.0))), g);
}

/// `base` with each edit in turn: every one must parse to `want`'s error.
fn refuses(base: []const u8, edits: []const [2][]const u8, want: cfg.Error) !void {
    for (edits) |e| {
        const bytes = try std.mem.replaceOwned(u8, t.allocator, base, e[0], e[1]);
        defer t.allocator.free(bytes);
        try t.expect(!std.mem.eql(u8, bytes, base));
        try t.expectError(want, cfg.parse(t.allocator, bytes));
    }
}

test "llama config: another rope type, a llama3 table short of a field or with another key is refused by name" {
    const rope = "\"rope_type\": \"llama3\"";
    try refuses(published[0][1], &.{
        .{ rope, "\"rope_type\": \"linear\"" },
        .{ rope, "\"rope_type\": \"dynamic\"" },
        .{ rope, "\"rope_type\": \"yarn\"" },
        .{ rope, "\"rope_type\": \"longrope\"" },
        .{ rope, "\"rope_type\": \"llama3\", \"type\": \"default\"" },
        .{ rope, "\"rope_type\": \"llama3\", \"attention_factor\": 1.0" },
        .{ "\"low_freq_factor\": 1.0,", "" },
        .{ "\"high_freq_factor\": 4.0,", "\"high_freq_factor\": 0.5," },
        .{ "\"factor\": 8.0,", "\"factor\": 0.5," },
        .{ "\"original_max_position_embeddings\": 8192", "\"original_max_position_embeddings\": 0" },
    }, error.UnsupportedLlamaRotary);
    // the other spelling alone, and both naming llama3, are the same table
    const typed = try std.mem.replaceOwned(u8, t.allocator, published[0][1], rope, "\"type\": \"llama3\"");
    defer t.allocator.free(typed);
    try t.expectEqual(llama31, (try cfg.parse(t.allocator, typed)).llama3.?);
    const both = try std.mem.replaceOwned(u8, t.allocator, published[0][1], rope, "\"rope_type\": \"llama3\", \"type\": \"llama3\"");
    defer t.allocator.free(both);
    try t.expectEqual(llama31, (try cfg.parse(t.allocator, both)).llama3.?);
}

test "llama config: mistral is llama's layout; a sliding window is refused by name, a null one admitted" {
    const m = try cfg.parse(t.allocator, mistral7b);
    try t.expectEqual(@as(usize, 4096), m.queryWidth());
    try refuses(mistral7b, &.{.{ "\"sliding_window\": null", "\"sliding_window\": 4096" }}, error.UnsupportedLlamaSlidingWindow);
    const absent = try std.mem.replaceOwned(u8, t.allocator, mistral7b, "  \"sliding_window\": null,\n", "");
    defer t.allocator.free(absent);
    try t.expectEqualStrings("Mistral-7B-Instruct-v0.3", (try cfg.parse(t.allocator, absent)).qualificationIn(&expected).?.name);
    // Mistral Small 3.1 and 3.2 publish mistral3, a vision tower around this text model: not this family
    try t.expectError(error.BadLlamaConfig, cfg.parse(t.allocator, "{\"model_type\": \"mistral3\"}"));
}

test "llama config: an explicit 128-wide head under a wider hidden (NeMo, Small) keeps q, o and the caches a query row" {
    const c = try cfg.parse(t.allocator, nemo);
    try t.expectEqual(@as(usize, 5120), c.hidden);
    try t.expectEqual(@as(usize, 4096), c.queryWidth());
    try t.expectEqual(@as(usize, 1024), c.kvWidth());
    try t.expectEqual(@as(usize, 6144), c.qkvWidth());
    try t.expectEqual(@as(usize, 4), c.group());
    // the same hidden without head_dim is 160-wide heads: refused, as is an odd head count's query row (31 x 128)
    try refuses(nemo, &.{
        .{ "\"head_dim\": 128,", "" },
        .{ "\"num_attention_heads\": 32,\n  \"num_hidden_layers\": 40,\n  \"num_key_value_heads\": 8", "\"num_attention_heads\": 31,\n  \"num_hidden_layers\": 40,\n  \"num_key_value_heads\": 1" },
    }, error.UnsupportedLlamaGeometry);
}

test "llama config: tie_word_embeddings sets the tied head, and a tied checkpoint is another row" {
    const c = try cfg.parse(t.allocator, published[2][1]);
    try t.expect(c.tied);
    try t.expect(!(try cfg.parse(t.allocator, published[0][1])).tied);
    var untied = c;
    untied.tied = false;
    try t.expectEqual(@as(?*const cfg.Qualified, null), untied.qualificationIn(&expected));
    const tied = try std.mem.replaceOwned(u8, t.allocator, nanbeige, "\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true");
    defer t.allocator.free(tied);
    try t.expectError(error.UnqualifiedLlamaCheckpoint, (try cfg.parse(t.allocator, tied)).qualification());
}

test "llama config: every field a row shares with the config decides admission" {
    const row = expected[0];
    const base = try cfg.parse(t.allocator, published[0][1]);
    try t.expect(base.qualificationIn(&.{row}) != null);
    var shared: usize = 0;
    inline for (@typeInfo(cfg.Qualified).@"struct".field_names) |f| if (@hasField(cfg.Config, f)) {
        shared += 1;
        var c = base;
        const T = @FieldType(cfg.Config, f);
        @field(c, f) = switch (@typeInfo(T)) {
            .int, .float => @field(c, f) + 1,
            .bool => !@field(c, f),
            .optional => null,
            .@"enum" => if (@field(c, f) == .bf16) .mlx_q8_g64 else .bf16,
            else => @compileError("a row field the test cannot move: " ++ f),
        };
        try t.expectEqual(@as(?*const cfg.Qualified, null), c.qualificationIn(&.{row}));
    };
    // weights, the seven dimensions, eps, theta, llama3 and tied
    try t.expectEqual(@as(usize, 12), shared);
}

const mlx = published[6][1];

test "llama config: an MLX table of one width, mirrored or not by quantization_config, names it" {
    const c = try cfg.parse(t.allocator, mlx);
    try t.expectEqual(cfg.Weights.mlx_q4_g64, c.weights);
    try t.expectEqual(@as(u8, 4), c.weights.mlx().?.bits);
    try t.expectEqual(@as(u16, 64), c.weights.mlx().?.group);
    const widths = [_]struct { []const u8, cfg.Weights }{
        .{ "\"group_size\": 32,\n    \"bits\": 4", .mlx_q4_g32 },
        .{ "\"group_size\": 64,\n    \"bits\": 8", .mlx_q8_g64 },
        .{ "\"group_size\": 32,\n    \"bits\": 8", .mlx_q8_g32 },
    };
    for (widths) |w| {
        const both = try std.mem.replaceOwned(u8, t.allocator, mlx, "\"group_size\": 64,\n    \"bits\": 4", w[0]);
        defer t.allocator.free(both);
        try t.expectEqual(w[1], (try cfg.parse(t.allocator, both)).weights);
    }
    const lone = try std.mem.replaceOwned(u8, t.allocator, mlx, "  \"quantization_config\": {\n    \"group_size\": 64,\n    \"bits\": 4\n  },\n", "");
    defer t.allocator.free(lone);
    try t.expectEqual(cfg.Weights.mlx_q4_g64, (try cfg.parse(t.allocator, lone)).weights);
    const affine = try std.mem.replaceOwned(u8, t.allocator, mlx, "\"bits\": 4\n  },\n  \"quantization_config\"", "\"bits\": 4,\n    \"mode\": \"affine\"\n  },\n  \"quantization_config\"");
    defer t.allocator.free(affine);
    try t.expectEqual(cfg.Weights.mlx_q4_g64, (try cfg.parse(t.allocator, affine)).weights);
}

test "llama config: other MLX tables are refused by name: widths, modes, per-module entries, a copy that differs" {
    const table = "\"quantization\": {\n    \"group_size\": 64,\n    \"bits\": 4\n  }";
    try refuses(mlx, &.{
        .{ table, "\"quantization\": {\n    \"group_size\": 64,\n    \"bits\": 3\n  }" },
        .{ table, "\"quantization\": {\n    \"group_size\": 64,\n    \"bits\": 6\n  }" },
        .{ table, "\"quantization\": {\n    \"group_size\": 128,\n    \"bits\": 4\n  }" },
        .{ table, "\"quantization\": {\n    \"group_size\": 64,\n    \"bits\": 4,\n    \"mode\": \"mxfp4\"\n  }" },
        .{ table, "\"quantization\": {\n    \"group_size\": 64,\n    \"bits\": 4,\n    \"model.layers.0.mlp.down_proj\": {\"group_size\": 64, \"bits\": 8}\n  }" },
        .{ "\"quantization_config\": {\n    \"group_size\": 64,\n    \"bits\": 4\n  }", "\"quantization_config\": {\n    \"group_size\": 64,\n    \"bits\": 8\n  }" },
        .{ "\"quantization_config\": {\n    \"group_size\": 64,\n    \"bits\": 4\n  }", "\"quantization_config\": {\"quant_method\": \"compressed-tensors\"}" },
    }, error.UnsupportedLlamaPrecision);
}
