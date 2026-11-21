//! Admission must match the all-sliding DFlash2 kernels before any weights or device state are loaded.
const std = @import("std");
const Config = @import("cuda_draft_config.zig").Config;
const Geometry = @import("cuda_engine.zig").Geometry;
const a = std.testing.allocator;
const t = std.testing;

// Synthetic metadata matching z-lab's 27B DFlash2 schema; the MoE case below matches incoai's 35B DFlash2.
const fixture =
    \\{"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":248070,
    \\"selector_rank":256,"selector_top_k":16,"target_layer_ids":[5,19,33,47,61]},
    \\"head_dim":128,"hidden_size":5120,"intermediate_size":17408,"num_attention_heads":32,
    \\"num_hidden_layers":5,"num_key_value_heads":8,"rms_norm_eps":0.000001,
    \\"rope_parameters":{"rope_type":"default","rope_theta":10000000},"sliding_window":2048,"vocab_size":248320,
    \\"use_sliding_window":true,
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"]}
;

fn target(hidden: usize, layers: usize) Geometry {
    return .{ .hidden = hidden, .intermediate = 17408, .layers = layers, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 262144, .linear = @splat(false) };
}

fn changed(key: []const u8, value: ?[]const u8) ![]u8 {
    var doc = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer doc.deinit();
    if (value) |raw| {
        const replacement = try std.json.parseFromSlice(std.json.Value, a, raw, .{});
        defer replacement.deinit();
        try doc.value.object.put(doc.arena.allocator(), key, replacement.value);
        return std.json.Stringify.valueAlloc(a, doc.value, .{});
    }
    try t.expect(doc.value.object.swapRemove(key));
    return std.json.Stringify.valueAlloc(a, doc.value, .{});
}

test "Qwen DFlash2 config: dense and MoE all-sliding schemas preserve every parsed option" {
    const expected: Config = .{ .hidden = 5120, .intermediate = 17408, .layers = 5, .vocab = 248320, .heads = 32, .kv_heads = 8, .head_dim = 128, .block = 16, .mask = 248070, .window = 2048, .group = 16, .rank = 256, .topk = 16, .eps = 1e-6, .theta = 10000000, .taps = .{ 5, 19, 33, 47, 61, 0, 0, 0 }, .n_taps = 5 };
    try t.expectEqualDeep(expected, try Config.parse(a, fixture, target(5120, 64)));
    var moe: []const u8 = fixture;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    for ([_][2][]const u8{
        .{ "5120", "2048" },                                       .{ "17408", "6144" },                              .{ "248070", "248077" },
        .{ "\"num_hidden_layers\":5", "\"num_hidden_layers\":6" }, .{ "[5,19,33,47,61]", "[1,6,11,16,22,27,32,37]" }, .{ "\"layer_types\":[", "\"layer_types\":[\"sliding_attention\"," },
    }) |r| moe = try std.mem.replaceOwned(u8, arena.allocator(), moe, r[0], r[1]);
    var moe_expected = expected;
    moe_expected.hidden = 2048;
    moe_expected.intermediate = 6144;
    moe_expected.layers = 6;
    moe_expected.mask = 248077;
    moe_expected.taps = .{ 1, 6, 11, 16, 22, 27, 32, 37 };
    moe_expected.n_taps = 8;
    try t.expectEqualDeep(moe_expected, try Config.parse(a, moe, target(2048, 40)));
}

test "Qwen DFlash2 config: a tap layer named twice is refused, as the ring would hold one residual in two taps" {
    for ([_][]const u8{ "[5,19,33,19,61]", "[5,5]" }) |taps| {
        const bytes = try std.mem.replaceOwned(u8, a, fixture, "[5,19,33,47,61]", taps);
        defer a.free(bytes);
        try t.expectError(error.BadDraftConfig, Config.parse(a, bytes, target(5120, 64)));
    }
    const one = try std.mem.replaceOwned(u8, a, fixture, "[5,19,33,47,61]", "[61]");
    defer a.free(one);
    const c = try Config.parse(a, one, target(5120, 64));
    try t.expectEqualSlices(u32, &.{61}, c.taps[0..c.n_taps]);
}

test "Qwen DFlash2 config: missing or malformed attention declarations are refused" {
    for ([_]?[]const u8{ null, "null", "0", "1", "\"true\"", "[]", "{}" }) |raw| {
        const bytes = try changed("use_sliding_window", raw);
        defer a.free(bytes);
        try t.expectError(error.BadDraftConfig, Config.parse(a, bytes, target(5120, 64)));
    }
    for ([_]?[]const u8{
        null,                                                                                                                                    "null",                                                                                           "true", "1", "\"sliding_attention\"", "{}", "[]", "[\"sliding_attention\"]",
        "[\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\"]", "[\"sliding_attention\",\"sliding_attention\",null,\"sliding_attention\",\"sliding_attention\"]",
    }) |raw| {
        const bytes = try changed("layer_types", raw);
        defer a.free(bytes);
        try t.expectError(error.BadDraftConfig, Config.parse(a, bytes, target(5120, 64)));
    }
}

test "Qwen DFlash2 config: disabled sliding and unsupported layers at every position cannot be reinterpreted" {
    const disabled = try changed("use_sliding_window", "false");
    defer a.free(disabled);
    try t.expectError(error.UnsupportedDraftConfig, Config.parse(a, disabled, target(5120, 64)));
    for ([_][]const u8{ "full_attention", "linear_attention", "unknown", "", "Sliding_attention" }) |kind| {
        for (0..5) |i| {
            var types: [5][]const u8 = @splat("sliding_attention");
            types[i] = kind;
            const raw = try std.json.Stringify.valueAlloc(a, types, .{});
            defer a.free(raw);
            const bytes = try changed("layer_types", raw);
            defer a.free(bytes);
            try t.expectError(error.UnsupportedDraftConfig, Config.parse(a, bytes, target(5120, 64)));
        }
    }
}

test "Qwen DFlash2 config: the on-disk reader refuses a mixed layer without any weights or device" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    defer a.free(dir);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = fixture });
    try t.expectEqualDeep(try Config.parse(a, fixture, target(5120, 64)), try Config.read(a, t.io, dir, target(5120, 64)));
    const mixed = try std.mem.replaceOwned(u8, a, fixture, "\"layer_types\":[\"sliding_attention\"", "\"layer_types\":[\"full_attention\"");
    defer a.free(mixed);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = mixed });
    try t.expectError(error.UnsupportedDraftConfig, Config.read(a, t.io, dir, target(5120, 64)));
}
