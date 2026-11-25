//! GLM-5.3-Flash's DFlash2 drafter config: incoai's five-tap contract read into the shared draft Config.
const std = @import("std");
const draft = @import("../qwen27/dflash/config.zig");

pub const repository = "incoai/GLM-5.3-Flash-DFlash2";
pub const tap_ids = [5]u32{ 5, 14, 24, 33, 42 };
pub const Config = draft.Config;

/// The five taps in the shared config's slots, the rest zero.
const tap_slots: [draft.max_taps]u32 = blk: {
    var t: [draft.max_taps]u32 = @splat(0);
    @memcpy(t[0..tap_ids.len], &tap_ids);
    break :blk t;
};

/// The stock incoai drafter: the shared draft contract's fields at GLM-5.3-Flash's shape.
pub const stock = Config{
    .hidden = 4096,
    .intermediate = 12288,
    .layers = 5,
    .vocab = 154880,
    .heads = 32,
    .kv_heads = 8,
    .head_dim = 128,
    .target_layers = 45,
    .block = 8,
    .mask = 154856,
    .window = 2048,
    .group = 16,
    .conv = 2,
    .rank = 256,
    .topk = 16,
    .eps = 1e-5,
    .theta = 10000,
    .taps = tap_slots,
    .tap_count = tap_ids.len,
};

/// The GLM drafter's own refusals, each naming the field that is not incoai's five-tap contract.
pub const Refusal = error{
    GlmDflash2Schema,
    GlmDflash2Architecture,
    GlmDflash2ModelType,
    GlmDflash2Dtype,
    GlmDflash2Hidden,
    GlmDflash2Intermediate,
    GlmDflash2Layers,
    GlmDflash2Vocab,
    GlmDflash2Heads,
    GlmDflash2KvHeads,
    GlmDflash2HeadDim,
    GlmDflash2TargetLayers,
    GlmDflash2Block,
    GlmDflash2Mask,
    GlmDflash2Window,
    GlmDflash2Group,
    GlmDflash2Conv,
    GlmDflash2Rank,
    GlmDflash2Topk,
    GlmDflash2Eps,
    GlmDflash2Theta,
    GlmDflash2Taps,
    GlmDflash2Attention,
    GlmDflash2Rope,
};

/// `c` is incoai's drafter for a 45-layer GLM-5.3-Flash target, or the field that is not.
pub fn check(c: Config) Refusal!void {
    if (c.hidden != stock.hidden) return error.GlmDflash2Hidden;
    if (c.intermediate != stock.intermediate) return error.GlmDflash2Intermediate;
    if (c.layers != stock.layers) return error.GlmDflash2Layers;
    if (c.vocab != stock.vocab) return error.GlmDflash2Vocab;
    if (c.heads != stock.heads) return error.GlmDflash2Heads;
    if (c.kv_heads != stock.kv_heads) return error.GlmDflash2KvHeads;
    if (c.head_dim != stock.head_dim) return error.GlmDflash2HeadDim;
    if (c.target_layers != stock.target_layers) return error.GlmDflash2TargetLayers;
    if (c.block != stock.block) return error.GlmDflash2Block;
    if (c.mask != stock.mask) return error.GlmDflash2Mask;
    if (c.window != stock.window) return error.GlmDflash2Window;
    if (c.group != stock.group) return error.GlmDflash2Group;
    if (c.conv != stock.conv) return error.GlmDflash2Conv;
    if (c.rank != stock.rank) return error.GlmDflash2Rank;
    if (c.topk != stock.topk) return error.GlmDflash2Topk;
    if (c.eps != stock.eps) return error.GlmDflash2Eps;
    if (c.theta != stock.theta) return error.GlmDflash2Theta;
    if (!std.mem.eql(u32, c.tapIds(), &tap_ids)) return error.GlmDflash2Taps;
    c.check() catch return error.GlmDflash2Schema;
}

/// The target's shape this drafter binds to: hidden, vocabulary and layer count, as the engine reads them.
pub const Target = struct { hidden: u32, vocab: u32, layers: u32 };

/// `t` is the GLM-5.3-Flash target this drafter's taps and vocabulary belong to.
pub fn bind(c: Config, t: Target) Refusal!void {
    try check(c);
    if (t.hidden != c.hidden) return error.GlmDflash2Hidden;
    if (t.vocab != c.vocab) return error.GlmDflash2Vocab;
    if (t.layers != c.target_layers) return error.GlmDflash2TargetLayers;
}

fn object(v: std.json.Value) Refusal!std.json.ObjectMap {
    return if (v == .object) v.object else error.GlmDflash2Schema;
}

fn integer(o: std.json.ObjectMap, name: []const u8) Refusal!u32 {
    const v = o.get(name) orelse return error.GlmDflash2Schema;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.GlmDflash2Schema;
    return @intCast(v.integer);
}

fn number(o: std.json.ObjectMap, name: []const u8) Refusal!f64 {
    const v = o.get(name) orelse return error.GlmDflash2Schema;
    const value: f64 = switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => return error.GlmDflash2Schema,
    };
    return if (std.math.isFinite(value)) value else error.GlmDflash2Schema;
}

fn text(o: std.json.ObjectMap, name: []const u8, wanted: []const u8) Refusal!void {
    const v = o.get(name) orelse return error.GlmDflash2Schema;
    if (v != .string or !std.mem.eql(u8, v.string, wanted)) return error.GlmDflash2Schema;
}

fn flag(o: std.json.ObjectMap, name: []const u8, wanted: bool) Refusal!void {
    const v = o.get(name) orelse return error.GlmDflash2Schema;
    if (v != .bool or v.bool != wanted) return error.GlmDflash2Schema;
}

/// The drafter's `config.json` text as incoai's five-tap contract, or the field that is not.
pub fn parse(gpa: std.mem.Allocator, raw: []const u8) !Config {
    const p = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer p.deinit();
    const o = try object(p.value);
    const d = try object(o.get("dflash_config") orelse return error.GlmDflash2Schema);
    const architectures = o.get("architectures") orelse return error.GlmDflash2Schema;
    if (architectures != .array or architectures.array.items.len != 1 or architectures.array.items[0] != .string) return error.GlmDflash2Schema;
    if (!std.mem.eql(u8, architectures.array.items[0].string, "DFlash2DraftModel")) return error.GlmDflash2Architecture;
    try text(o, "model_type", "qwen3");
    try text(o, "dtype", "bfloat16");
    try text(o, "hidden_act", "silu");
    try flag(o, "is_causal", false);
    try flag(o, "attention_bias", false);
    try flag(o, "tie_word_embeddings", false);
    try flag(o, "use_sliding_window", true);
    if (try number(o, "attention_dropout") != 0) return error.GlmDflash2Attention;
    const rope = try object(o.get("rope_parameters") orelse return error.GlmDflash2Schema);
    try text(rope, "rope_type", "default");
    if (o.get("rope_scaling")) |v| if (v != .null) return error.GlmDflash2Rope;
    var c = Config{
        .hidden = try integer(o, "hidden_size"),
        .intermediate = try integer(o, "intermediate_size"),
        .layers = try integer(o, "num_hidden_layers"),
        .vocab = try integer(o, "vocab_size"),
        .heads = try integer(o, "num_attention_heads"),
        .kv_heads = try integer(o, "num_key_value_heads"),
        .head_dim = try integer(o, "head_dim"),
        .target_layers = try integer(o, "num_target_layers"),
        .block = try integer(d, "block_size"),
        .mask = try integer(d, "mask_token_id"),
        .window = try integer(o, "sliding_window"),
        .group = try integer(d, "conv_group_size"),
        .conv = try integer(d, "conv_kernel_size"),
        .rank = try integer(d, "selector_rank"),
        .topk = try integer(d, "selector_top_k"),
        .eps = @floatCast(try number(o, "rms_norm_eps")),
        .theta = try number(rope, "rope_theta"),
    };
    const taps = d.get("target_layer_ids") orelse return error.GlmDflash2Schema;
    if (taps != .array or taps.array.items.len != tap_ids.len) return error.GlmDflash2Taps;
    c.taps = @splat(0);
    c.tap_count = tap_ids.len;
    for (taps.array.items, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.GlmDflash2Schema;
        c.taps[i] = @intCast(v.integer);
    }
    const types = o.get("layer_types") orelse return error.GlmDflash2Schema;
    if (types != .array or types.array.items.len != c.layers) return error.GlmDflash2Schema;
    for (types.array.items) |v| if (v != .string or !std.mem.eql(u8, v.string, "sliding_attention")) return error.GlmDflash2Attention;
    try check(c);
    return c;
}

const fixture =
    \\{"architectures":["DFlash2DraftModel"],"model_type":"qwen3","dtype":"bfloat16","hidden_act":"silu",
    \\"hidden_size":4096,"intermediate_size":12288,"num_hidden_layers":5,"vocab_size":154880,
    \\"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"num_target_layers":45,
    \\"is_causal":false,"attention_bias":false,"tie_word_embeddings":false,"attention_dropout":0.0,
    \\"use_sliding_window":true,"sliding_window":2048,"rms_norm_eps":0.00001,
    \\"rope_parameters":{"rope_type":"default","rope_theta":10000.0},
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
    \\"dflash_config":{"block_size":8,"mask_token_id":154856,"conv_group_size":16,"conv_kernel_size":2,
    \\"selector_rank":256,"selector_top_k":16,"target_layer_ids":[5,14,24,33,42]}}
;

test "incoai's cached schema resolves the stock five-tap GLM drafter" {
    const c = try parse(std.testing.allocator, fixture);
    try std.testing.expectEqualDeep(stock, c);
    try std.testing.expectEqual(@as(usize, 81), c.tensorCount());
    try std.testing.expectEqual(@as(usize, 47), c.linearCount());
    try std.testing.expectEqual(@as(u32, 20480), c.tapWidth());
}

test "a target of another shape is refused by the field that differs" {
    try std.testing.expectError(error.GlmDflash2Hidden, bind(stock, .{ .hidden = 5120, .vocab = 154880, .layers = 45 }));
    try std.testing.expectError(error.GlmDflash2Vocab, bind(stock, .{ .hidden = 4096, .vocab = 248320, .layers = 45 }));
    try std.testing.expectError(error.GlmDflash2TargetLayers, bind(stock, .{ .hidden = 4096, .vocab = 154880, .layers = 64 }));
    try bind(stock, .{ .hidden = 4096, .vocab = 154880, .layers = 45 });
}

test "qwen27's stock drafter is not a GLM drafter" {
    try std.testing.expectError(error.GlmDflash2Hidden, check(draft.Config{}));
}

test "a nine-tap or shifted-tap drafter is refused before any weight is read" {
    var c = stock;
    c.taps = .{ 5, 9, 14, 19, 24, 0, 0, 0 };
    try std.testing.expectError(error.GlmDflash2Taps, check(c));
    c = stock;
    c.target_layers = 64;
    try std.testing.expectError(error.GlmDflash2TargetLayers, check(c));
}

test "a causal reading, a scaled rope or a full-attention layer cannot become the default" {
    const a = std.testing.allocator;
    const causal = try std.mem.replaceOwned(u8, a, fixture, "\"is_causal\":false", "\"is_causal\":true");
    defer a.free(causal);
    try std.testing.expectError(error.GlmDflash2Schema, parse(a, causal));
    const scaled = try std.mem.replaceOwned(u8, a, fixture, "\"layer_types\"", "\"rope_scaling\":{\"type\":\"yarn\",\"factor\":4.0},\"layer_types\"");
    defer a.free(scaled);
    try std.testing.expectError(error.GlmDflash2Rope, parse(a, scaled));
    const full = try std.mem.replaceOwned(u8, a, fixture, "\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\"", "\"full_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\"");
    defer a.free(full);
    try std.testing.expectError(error.GlmDflash2Attention, parse(a, full));
}

test "a drafter of another vocabulary or norm epsilon is refused by name" {
    const a = std.testing.allocator;
    const vocab = try std.mem.replaceOwned(u8, a, fixture, "\"vocab_size\":154880", "\"vocab_size\":248320");
    defer a.free(vocab);
    try std.testing.expectError(error.GlmDflash2Vocab, parse(a, vocab));
    const eps = try std.mem.replaceOwned(u8, a, fixture, "\"rms_norm_eps\":0.00001", "\"rms_norm_eps\":0.000001");
    defer a.free(eps);
    try std.testing.expectError(error.GlmDflash2Eps, parse(a, eps));
    const theta = try std.mem.replaceOwned(u8, a, fixture, "\"rope_theta\":10000.0", "\"rope_theta\":10000000.0");
    defer a.free(theta);
    try std.testing.expectError(error.GlmDflash2Theta, parse(a, theta));
}
