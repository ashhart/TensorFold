//! What a Qwen3.5 config must declare for the HIP engine: activation, gate, state, layers, rope, end ids, sizes.

const std = @import("std");

pub const Error = error{ UnsupportedActivation, UnsupportedAttentionGate, UnsupportedStateType, BadEos, TooManyEos, MissingEos, InvalidConfig, InvalidLayerTypes, InvalidRotary };

/// The most end ids a checkpoint may declare across config.json and generation_config.json.
pub const max_eos = 8;

/// The most decoder layers a config may declare.
pub const max_layers = 128;

/// Each layer's kind (true: full attention), from `layer_types` in any order, else every `interval`-th layer.
pub fn layers(out: *[max_layers]bool, tower: std.json.ObjectMap, count: usize, interval: usize) Error!void {
    if (count == 0 or count > max_layers or interval == 0) return error.InvalidConfig;
    out.* = @splat(false);
    const list = tower.get("layer_types") orelse .null;
    if (list == .null) {
        for (out[0..count], 0..) |*full, i| full.* = (i + 1) % interval == 0;
        return;
    }
    if (list != .array or list.array.items.len != count) return error.InvalidLayerTypes;
    for (list.array.items, out[0..count]) |item, *full| {
        if (isString(item, "full_attention")) full.* = true else if (!isString(item, "linear_attention")) return error.InvalidLayerTypes;
    }
}

/// The multimodal rope's three sections and their interleaving: the text and image positions a vision tower feeds.
pub const Mrope = struct { section: [3]usize = .{ 11, 11, 10 }, interleaved: bool = true };

/// `rope_parameters`' mrope fields: three sections filling half the rotary width, interleaved (the kernels' layout).
pub fn mrope(rope: std.json.ObjectMap, rotary: usize) Error!Mrope {
    var m: Mrope = .{};
    if (rope.get("mrope_interleaved")) |v| if (!isFlag(v, true)) return error.InvalidRotary;
    const list = rope.get("mrope_section") orelse return m;
    if (list == .null) return m;
    if (list != .array or list.array.items.len != 3) return error.InvalidRotary;
    var sum: usize = 0;
    for (list.array.items, &m.section) |item, *x| {
        if (item != .integer or item.integer < 0) return error.InvalidRotary;
        x.* = @intCast(item.integer);
        sum += x.*;
    }
    if (sum * 2 != rotary) return error.InvalidRotary;
    return m;
}

/// `max_position_embeddings`: a positive token count, else the family's 262,144.
pub fn maxPosition(tower: std.json.ObjectMap) Error!usize {
    const v = tower.get("max_position_embeddings") orelse return 262144;
    if (v != .integer or v.integer <= 0 or v.integer > std.math.maxInt(u32)) return error.InvalidConfig;
    return @intCast(v.integer);
}

/// Refuses a tower whose activation, attention gate, attention bias or recurrent state type the kernels do not compute.
pub fn computes(tower: std.json.ObjectMap) Error!void {
    if (tower.get("hidden_act")) |v| if (!isString(v, "silu")) return error.UnsupportedActivation;
    // "swish" is the gate the kernels apply, att * sigmoid(gate), as the qwen27 family reads it
    if (tower.get("output_gate_type")) |v| if (!isString(v, "swish")) return error.UnsupportedAttentionGate;
    if (tower.get("attn_output_gate")) |v| if (!isFlag(v, true)) return error.UnsupportedAttentionGate;
    if (tower.get("attention_bias")) |v| if (!isFlag(v, false)) return error.UnsupportedAttentionGate;
    inline for (.{ "linear_attn_state_dtype", "mamba_ssm_dtype" }) |key| {
        if (tower.get(key)) |v| if (!isString(v, "float32")) return error.UnsupportedStateType;
    }
}

/// The sizes the kernels index with 32-bit counts and the host multiplies: every product must fit.
pub const Sizes = struct {
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    conv: usize,
    experts: usize,
    moe_width: usize,
    shared_width: usize,
};

pub fn products(s: Sizes) Error!void {
    if (s.heads % s.kv_heads != 0 or s.value_heads % s.key_heads != 0) return error.InvalidConfig;
    const limit: usize = std.math.maxInt(u32);
    const q = try mul(try mul(s.heads, s.head_dim), 2);
    const kv = try mul(s.kv_heads, s.head_dim);
    const k = try mul(s.key_heads, s.key_dim);
    const v = try mul(s.value_heads, s.value_dim);
    const conv = try mul(try add(try mul(k, 2), v), s.conv);
    const state = try mul(try mul(v, s.key_dim), 4);
    const widths = [_]usize{ q, kv, conv, state, try mul(s.hidden, s.intermediate), try mul(s.hidden, s.vocab), try mul(s.hidden, try mul(s.experts, s.moe_width)), try mul(s.hidden, s.shared_width) };
    for (widths) |w| if (w > limit) return error.InvalidConfig;
}

/// The end ids, config.json's (root, then text tower) and generation_config.json's, each once and below `vocab`.
pub fn eos(out: *[max_eos]u32, root: std.json.ObjectMap, tower: std.json.ObjectMap, generation: ?std.json.ObjectMap, vocab: usize) Error![]const u32 {
    var n: usize = 0;
    const holders = [_]?std.json.Value{ root.get("eos_token_id"), tower.get("eos_token_id"), if (generation) |g| g.get("eos_token_id") else null };
    for (holders) |h| if (h) |v| try collect(out, &n, v, vocab);
    if (n == 0) return error.MissingEos;
    return out[0..n];
}

fn collect(out: *[max_eos]u32, n: *usize, v: std.json.Value, vocab: usize) Error!void {
    switch (v) {
        .null => {},
        .array => |list| for (list.items) |item| try collect(out, n, item, vocab),
        .integer => |i| {
            if (i < 0 or i >= vocab) return error.BadEos;
            const id: u32 = @intCast(i);
            if (std.mem.indexOfScalar(u32, out[0..n.*], id) != null) return;
            if (n.* == max_eos) return error.TooManyEos;
            out[n.*] = id;
            n.* += 1;
        },
        else => return error.BadEos,
    }
}

fn mul(a: usize, b: usize) Error!usize {
    return std.math.mul(usize, a, b) catch error.InvalidConfig;
}

fn add(a: usize, b: usize) Error!usize {
    return std.math.add(usize, a, b) catch error.InvalidConfig;
}

fn isString(v: std.json.Value, want: []const u8) bool {
    return v == .string and std.mem.eql(u8, v.string, want);
}

/// A JSON boolean, or the 0 / 1 some conversions write in its place.
fn isFlag(v: std.json.Value, want: bool) bool {
    return switch (v) {
        .bool => |b| b == want,
        .integer => |i| i == @intFromBool(want),
        else => false,
    };
}

fn parsed(a: std.mem.Allocator, text: []const u8) !std.json.ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
}

test "the computed parts: silu, the swish gate, no bias, fp32 state; 0 and 1 stand for false and true" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try computes(try parsed(a, "{}"));
    try computes(try parsed(a, "{\"hidden_act\":\"silu\",\"output_gate_type\":\"swish\",\"attn_output_gate\":1,\"attention_bias\":0,\"mamba_ssm_dtype\":\"float32\"}"));
    try std.testing.expectError(error.UnsupportedActivation, computes(try parsed(a, "{\"hidden_act\":\"gelu\"}")));
    try std.testing.expectError(error.UnsupportedAttentionGate, computes(try parsed(a, "{\"output_gate_type\":\"sigmoid\"}")));
    try std.testing.expectError(error.UnsupportedAttentionGate, computes(try parsed(a, "{\"attn_output_gate\":false}")));
    try std.testing.expectError(error.UnsupportedAttentionGate, computes(try parsed(a, "{\"attention_bias\":true}")));
    try std.testing.expectError(error.UnsupportedStateType, computes(try parsed(a, "{\"mamba_ssm_dtype\":\"bfloat16\"}")));
    try std.testing.expectError(error.UnsupportedStateType, computes(try parsed(a, "{\"linear_attn_state_dtype\":\"float16\"}")));
}

test "end ids from config.json and generation_config.json, once each, refused past the vocabulary" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var out: [max_eos]u32 = undefined;
    // the 27B: config.json names 248044, generation_config.json adds 248046
    const ids = try eos(&out, try parsed(a, "{}"), try parsed(a, "{\"eos_token_id\":248044}"), try parsed(a, "{\"eos_token_id\":[248046,248044]}"), 248320);
    try std.testing.expectEqualSlices(u32, &.{ 248044, 248046 }, ids);
    try std.testing.expectError(error.MissingEos, eos(&out, try parsed(a, "{}"), try parsed(a, "{}"), null, 10));
    try std.testing.expectError(error.BadEos, eos(&out, try parsed(a, "{\"eos_token_id\":10}"), try parsed(a, "{}"), null, 10));
    try std.testing.expectError(error.BadEos, eos(&out, try parsed(a, "{\"eos_token_id\":\"2\"}"), try parsed(a, "{}"), null, 10));
    try std.testing.expectError(error.TooManyEos, eos(&out, try parsed(a, "{\"eos_token_id\":[0,1,2,3,4,5,6,7,8]}"), try parsed(a, "{}"), null, 10));
}

test "sizes: products past 32 bits and uneven head groups are refused" {
    const ok: Sizes = .{ .hidden = 5120, .intermediate = 17408, .vocab = 248320, .heads = 24, .kv_heads = 4, .head_dim = 256, .key_heads = 16, .value_heads = 48, .key_dim = 128, .value_dim = 128, .conv = 4, .experts = 0, .moe_width = 0, .shared_width = 0 };
    try products(ok);
    var moe = ok;
    moe.experts = 256;
    moe.moe_width = 512;
    try products(moe);
    var wide = ok;
    wide.vocab = 1 << 20;
    try std.testing.expectError(error.InvalidConfig, products(wide));
    var uneven = ok;
    uneven.kv_heads = 5;
    try std.testing.expectError(error.InvalidConfig, products(uneven));
    var huge = ok;
    huge.hidden = std.math.maxInt(usize) / 2;
    try std.testing.expectError(error.InvalidConfig, products(huge));
}

test "layer kinds: any order from layer_types, the interval otherwise; wrong length or names refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var kinds: [max_layers]bool = undefined;
    try layers(&kinds, try parsed(a, "{}"), 8, 4);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, true, false, false, false, true }, kinds[0..8]);
    try layers(&kinds, try parsed(a, "{\"layer_types\":[\"full_attention\",\"linear_attention\",\"full_attention\"]}"), 3, 4);
    try std.testing.expectEqualSlices(bool, &.{ true, false, true }, kinds[0..3]);
    try std.testing.expectError(error.InvalidLayerTypes, layers(&kinds, try parsed(a, "{\"layer_types\":[\"full_attention\"]}"), 3, 4));
    try std.testing.expectError(error.InvalidLayerTypes, layers(&kinds, try parsed(a, "{\"layer_types\":[\"sliding_attention\"]}"), 1, 4));
    try std.testing.expectError(error.InvalidConfig, layers(&kinds, try parsed(a, "{}"), max_layers + 1, 4));
}

test "mrope: three sections filling half the rotary width, interleaved; max_position positive" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try mrope(try parsed(a, "{\"mrope_section\":[11,11,10],\"mrope_interleaved\":true}"), 64);
    try std.testing.expectEqual([3]usize{ 11, 11, 10 }, m.section);
    try std.testing.expectEqual(Mrope{}, try mrope(try parsed(a, "{}"), 64));
    try std.testing.expectError(error.InvalidRotary, mrope(try parsed(a, "{\"mrope_section\":[11,11,11]}"), 64));
    try std.testing.expectError(error.InvalidRotary, mrope(try parsed(a, "{\"mrope_section\":[32,32]}"), 64));
    try std.testing.expectError(error.InvalidRotary, mrope(try parsed(a, "{\"mrope_interleaved\":false}"), 64));
    try std.testing.expectEqual(@as(usize, 262144), try maxPosition(try parsed(a, "{}")));
    try std.testing.expectEqual(@as(usize, 32768), try maxPosition(try parsed(a, "{\"max_position_embeddings\":32768}")));
    try std.testing.expectError(error.InvalidConfig, maxPosition(try parsed(a, "{\"max_position_embeddings\":0}")));
}
