//! DFlash2 is a separate five-layer GQA contract; the target's GDN and draft conventions are not interchangeable.
const std = @import("std");
pub const repository = "z-lab/Qwen3.8-27B-DFlash2";
pub const tap_ids = [5]u32{ 5, 19, 33, 47, 61 };
pub const Config = struct {
    hidden: u32 = 5120,
    intermediate: u32 = 17408,
    layers: u32 = 5,
    vocab: u32 = 248320,
    heads: u32 = 32,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    target_layers: u32 = 64,
    block: u32 = 8,
    mask: u32 = 248070,
    window: u32 = 2048,
    group: u32 = 16,
    conv: u32 = 2,
    rank: u32 = 256,
    topk: u32 = 16,
    eps: f32 = 1e-6,
    theta: f64 = 10000000,
    taps: [5]u32 = tap_ids,
    pub fn qWidth(c: Config) u32 {
        return c.heads * c.head_dim;
    }
    pub fn kvWidth(c: Config) u32 {
        return c.kv_heads * c.head_dim;
    }
    pub fn tapWidth(c: Config) u32 {
        return c.hidden * 5;
    }
    pub fn dynamicWidth(c: Config) u32 {
        return 4 * (c.hidden / c.group);
    }
    pub fn depth(c: Config) u32 {
        return c.block - 1;
    }
    pub fn tensorCount(c: Config) usize {
        return 15 * @as(usize, c.layers) + 6;
    }
    pub fn linearCount(c: Config) usize {
        return 9 * @as(usize, c.layers) + 2;
    }
    pub fn check(c: Config) !void {
        const dimensions = [_]u32{ c.hidden, c.intermediate, c.layers, c.vocab, c.heads, c.kv_heads, c.head_dim, c.target_layers, c.group, c.rank, c.topk };
        for (dimensions) |n| if (n == 0) return error.BadDraftConfig;
        if (c.hidden % 64 != 0 or c.intermediate % 64 != 0 or c.hidden % c.group != 0 or c.heads % c.kv_heads != 0 or c.head_dim % 2 != 0 or c.block != 8 or c.conv != 2 or c.layers > 8 or c.mask >= c.vocab or c.topk > c.vocab or c.window < c.block) return error.BadDraftConfig;
        if (!std.math.isFinite(c.eps) or c.eps <= 0 or !std.math.isFinite(c.theta) or c.theta <= 0) return error.BadDraftConfig;
        if (c.target_layers == 64 and !std.mem.eql(u32, &c.taps, &tap_ids)) return error.UnsupportedDraftTaps;
        for (c.taps, 0..) |tap, i| for (c.taps[0..i]) |previous| if (tap == previous) return error.UnsupportedDraftTaps;
        for (c.taps) |i| if (i >= c.target_layers) return error.BadDraftConfig;
        const q = try std.math.mul(u32, c.heads, c.head_dim);
        _ = try std.math.mul(u32, c.kv_heads, c.head_dim);
        _ = try std.math.mul(u32, c.hidden, 5);
        _ = try std.math.mul(u32, c.hidden / c.group, 4);
        if (q % 64 != 0) return error.BadDraftConfig;
    }
    pub fn requireStock(c: Config) !void {
        try c.check();
        if (c.hidden != 5120 or c.intermediate != 17408 or c.layers != 5 or c.vocab != 248320 or c.heads != 32 or c.kv_heads != 8 or c.head_dim != 128 or c.target_layers != 64 or c.mask != 248070 or c.window != 2048 or c.group != 16 or c.rank != 256 or c.topk != 16 or c.eps != 1e-6 or c.theta != 10000000) return error.NotStockDFlash2;
    }
};
fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadDraftConfig;
}
fn integer(o: std.json.ObjectMap, name: []const u8) !u32 {
    const v = o.get(name) orelse return error.BadDraftConfig;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadDraftConfig;
    return @intCast(v.integer);
}
fn number(o: std.json.ObjectMap, name: []const u8) !f64 {
    const v = o.get(name) orelse return error.BadDraftConfig;
    const value: f64 = switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => return error.BadDraftConfig,
    };
    return if (std.math.isFinite(value)) value else error.BadDraftConfig;
}
fn text(o: std.json.ObjectMap, name: []const u8, wanted: []const u8) !void {
    const v = o.get(name) orelse return error.BadDraftConfig;
    if (v != .string or !std.mem.eql(u8, v.string, wanted)) return error.BadDraftConfig;
}
pub fn parse(gpa: std.mem.Allocator, raw: []const u8) !Config {
    const p = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer p.deinit();
    const o = try object(p.value);
    const d = try object(o.get("dflash_config") orelse return error.BadDraftConfig);
    const architectures = o.get("architectures") orelse return error.BadDraftConfig;
    if (architectures != .array or architectures.array.items.len != 1 or architectures.array.items[0] != .string or !std.mem.eql(u8, architectures.array.items[0].string, "DFlash2DraftModel")) return error.BadDraftConfig;
    try text(o, "model_type", "qwen3");
    try text(o, "dtype", "bfloat16");
    try text(o, "hidden_act", "silu");
    for ([_][]const u8{ "is_causal", "attention_bias", "tie_word_embeddings" }) |key| {
        const v = o.get(key) orelse return error.BadDraftConfig;
        if (v != .bool or v.bool) return error.BadDraftConfig;
    }
    const sliding = o.get("use_sliding_window") orelse return error.BadDraftConfig;
    if (sliding != .bool or !sliding.bool or try number(o, "attention_dropout") != 0) return error.BadDraftConfig;
    const rope = try object(o.get("rope_parameters") orelse return error.BadDraftConfig);
    try text(rope, "rope_type", "default");
    var c = Config{ .hidden = try integer(o, "hidden_size"), .intermediate = try integer(o, "intermediate_size"), .layers = try integer(o, "num_hidden_layers"), .vocab = try integer(o, "vocab_size"), .heads = try integer(o, "num_attention_heads"), .kv_heads = try integer(o, "num_key_value_heads"), .head_dim = try integer(o, "head_dim"), .target_layers = try integer(o, "num_target_layers"), .block = try integer(d, "block_size"), .mask = try integer(d, "mask_token_id"), .window = try integer(o, "sliding_window"), .group = try integer(d, "conv_group_size"), .conv = try integer(d, "conv_kernel_size"), .rank = try integer(d, "selector_rank"), .topk = try integer(d, "selector_top_k"), .eps = @floatCast(try number(o, "rms_norm_eps")), .theta = try number(rope, "rope_theta") };
    const taps = d.get("target_layer_ids") orelse return error.BadDraftConfig;
    if (taps != .array or taps.array.items.len != 5) return error.BadDraftConfig;
    for (taps.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.BadDraftConfig;
        c.taps[i] = @intCast(v.integer);
    }
    const types = o.get("layer_types") orelse return error.BadDraftConfig;
    if (types != .array or types.array.items.len != c.layers) return error.BadDraftConfig;
    for (types.array.items) |v| if (v != .string or !std.mem.eql(u8, v.string, "sliding_attention")) return error.BadDraftConfig;
    try c.requireStock();
    return c;
}

test "stock contracts count81native tensors47eligiblelinears and seven mask predictions" {
    const c = Config{};
    try c.requireStock();
    try std.testing.expectEqual(@as(usize, 81), c.tensorCount());
    try std.testing.expectEqual(@as(usize, 47), c.linearCount());
    try std.testing.expectEqual(@as(u32, 7), c.depth());
}
test "tap convention is fixed after decoder blocks and cannot be shifted" {
    var c = Config{};
    c.taps[0] = 4;
    try std.testing.expectError(error.UnsupportedDraftTaps, c.check());
}
test "block slots and finite dimensions are admitted before operators" {
    var c = Config{};
    c.block = 16;
    try std.testing.expectError(error.BadDraftConfig, c.check());
    c = .{};
    c.hidden = 0;
    try std.testing.expectError(error.BadDraftConfig, c.check());
    c = .{};
    c.eps = std.math.nan(f32);
    try std.testing.expectError(error.BadDraftConfig, c.check());
}

const fixture =
    \\{"architectures":["DFlash2DraftModel"],"model_type":"qwen3","dtype":"bfloat16","hidden_act":"silu",
    \\"hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":5,"vocab_size":248320,
    \\"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"num_target_layers":64,
    \\"is_causal":false,"attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0.0,
    \\"use_sliding_window":true,"sliding_window":2048,"rms_norm_eps":0.000001,
    \\"rope_parameters":{"rope_type":"default","rope_theta":10000000},
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
    \\"dflash_config":{"block_size":8,"mask_token_id":248070,"conv_group_size":16,"conv_kernel_size":2,
    \\"selector_rank":256,"selector_top_k":16,"target_layer_ids":[5,19,33,47,61]}}
;
test "actual nested cached schema resolves the stock noncausal draft and tap order" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqualDeep(Config{}, c);
}
test "a causal reading or changed native storage cannot silently become the default" {
    const a = std.testing.allocator;
    var p = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer p.deinit();
    try p.value.object.put(p.arena.allocator(), "is_causal", .{ .bool = true });
    const text_json = try std.json.Stringify.valueAlloc(a, p.value, .{});
    defer a.free(text_json);
    try std.testing.expectError(error.BadDraftConfig, parse(a, text_json));
}
