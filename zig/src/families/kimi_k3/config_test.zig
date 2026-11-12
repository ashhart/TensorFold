//! K3 and K3.1 release configs must not select the same attention behavior.
const std = @import("std");
const config = @import("config.zig");
const k31_fixture = @embedFile("fixtures/k31-config.json");

test "K3.1 release config cannot silently select K3 dense MLA" {
    const k3 = try config.parse(std.testing.allocator, @embedFile("fixtures/k3-config.json"));
    const k31 = try config.parse(std.testing.allocator, @embedFile("fixtures/k31-config.json"));
    try std.testing.expect(!std.meta.eql(k3, k31));
}

test "K3 release config retains every previous default" {
    const k3 = try config.parse(std.testing.allocator, @embedFile("fixtures/k3-config.json"));
    try std.testing.expectEqualDeep(config.Config{}, k3);
    const k31 = try config.parse(std.testing.allocator, k31_fixture);
    try std.testing.expectEqual(@as(u32, 32), k31.dsa.?.heads);
    try std.testing.expectEqual(@as(u32, 128), k31.dsa.?.dim);
    try std.testing.expectEqual(@as(u32, 2048), k31.dsa.?.topk);
}

test "K3.1 refuses unsupported index geometry and cadence instead of ignoring them" {
    const cases = .{
        .{ "\"index_topk\": 2048", "\"index_topk\": 2047" },
        .{ "\"index_head_dim\": 128", "\"index_head_dim\": 64" },
        .{ "\"index_num_heads\": 32", "\"index_num_heads\": 16" },
        .{ "\"index_rope_dim\": 0", "\"index_rope_dim\": 64" },
        .{ "\"index_k_norm_type\": \"layernorm\"", "\"index_k_norm_type\": \"rmsnorm\"" },
        .{ "\"index_topk_freq\": 1", "\"index_topk_freq\": 2" },
        .{ "\"index_skip_topk_offset\": 1", "\"index_skip_topk_offset\": 0" },
        .{ "\"hidden_size\": 7168", "\"hidden_size\": -1" },
        .{ "\"q_lora_rank\": 1536", "\"q_lora_rank\": 1536.5" },
        .{ "\"gate_lower_bound\": -5.0", "\"gate_lower_bound\": -4.0" },
        .{ "\"mla_use_output_gate\": true", "\"mla_use_output_gate\": false" },
    };
    inline for (cases) |pair| {
        const changed = try std.mem.replaceOwned(u8, std.testing.allocator, k31_fixture, pair[0], pair[1]);
        defer std.testing.allocator.free(changed);
        try std.testing.expect(!std.mem.eql(u8, changed, k31_fixture));
        try std.testing.expectError(error.UnsupportedK31Shape, config.parse(std.testing.allocator, changed));
    }
}

test "K3.1 dsa layers must equal full attention layers without duplicates or invalid indices" {
    inline for (.{ "0", "94", "8", "5" }) |bad| {
        const changed = try std.mem.replaceOwned(u8, std.testing.allocator, k31_fixture, "\"dsa_layers\": [\n        4,", "\"dsa_layers\": [\n        " ++ bad ++ ",");
        defer std.testing.allocator.free(changed);
        try std.testing.expect(!std.mem.eql(u8, changed, k31_fixture));
        try std.testing.expectError(error.BadK31Layers, config.parse(std.testing.allocator, changed));
    }
}

test "K3.1 requires DSA and cannot be hidden behind K3 identity or a null config" {
    const no_dsa = try std.mem.replaceOwned(u8, std.testing.allocator, k31_fixture, "\"dsa_config\"", "\"unsupported_dsa\"");
    defer std.testing.allocator.free(no_dsa);
    try std.testing.expectError(error.MissingK31Dsa, config.parse(std.testing.allocator, no_dsa));
    const disguised = try std.mem.replaceOwned(u8, std.testing.allocator, k31_fixture, "\"kimi_k31\"", "\"kimi_k3\"");
    defer std.testing.allocator.free(disguised);
    try std.testing.expectError(error.UnsupportedK31Identity, config.parse(std.testing.allocator, disguised));
    try std.testing.expectError(error.BadConfig, config.parse(std.testing.allocator, "null"));
    try std.testing.expectError(error.BadConfig, config.parse(std.testing.allocator, "{\"text_config\":null}"));
}

test "K3.1 cap includes prefix and every speculative or prompt row without overflow" {
    const c = try config.parse(std.testing.allocator, k31_fixture);
    try c.checkEnd(0, 2047);
    try c.checkEnd(2046, 1);
    try std.testing.expectError(error.K31ContextLimit, c.checkEnd(0, 2048));
    try std.testing.expectError(error.K31ContextLimit, c.checkEnd(2047, 1));
    try std.testing.expectError(error.K31ContextLimit, c.checkEnd(2000, 48));
    try std.testing.expectError(error.K31ContextLimit, c.checkEnd(std.math.maxInt(u32), 1));
    const legacy = config.Config{};
    try legacy.checkEnd(65536, 32);
}
