//! Dense Qwen text shapes, layer schedule, rotary metadata and ordered checkpoint EOS ids.
const std = @import("std");

pub const Kind = enum { linear, attention };
pub const max_layers = 128;
pub const max_eos = 8;

pub const Config = struct {
    hidden: usize,
    intermediate: usize,
    layers: usize,
    vocab: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    k_heads: usize,
    v_heads: usize,
    dk: usize,
    dv: usize,
    conv_kernel: usize,
    eps: f32,
    rope_dims: usize,
    rope_theta: f64,
    mrope_section: [3]usize = .{ 11, 11, 10 },
    kinds: [max_layers]Kind = @splat(.linear),
    eos: [max_eos]u32 = @splat(0),
    eos_count: usize = 0,
    max_position: usize = 262144,
    mrope_interleaved: bool = true,

    pub fn kind(c: Config, layer: usize) Kind {
        return c.kinds[layer];
    }

    pub fn qDim(c: Config) usize {
        return 2 * c.heads * c.head_dim;
    }

    pub fn kvDim(c: Config) usize {
        return c.kv_heads * c.head_dim;
    }

    pub fn gdnValueDim(c: Config) usize {
        return c.v_heads * c.dv;
    }

    pub fn gdnQkvDim(c: Config) usize {
        return 2 * c.k_heads * c.dk + c.gdnValueDim();
    }

    pub fn isEos(c: Config, token: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos[0..c.eos_count], token) != null;
    }
};

pub fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.BadConfig;
}

fn integer(v: std.json.Value, zero: bool) !usize {
    if (v != .integer or v.integer < 0 or (!zero and v.integer == 0)) return error.BadConfig;
    if (v.integer > std.math.maxInt(u32)) return error.BadConfig;
    return @intCast(v.integer);
}

fn required(o: std.json.ObjectMap, key: []const u8) !usize {
    return integer(o.get(key) orelse return error.BadConfig, false);
}

fn number(v: std.json.Value) !f64 {
    const n: f64 = switch (v) {
        .integer => |x| @floatFromInt(x),
        .float => |x| x,
        else => return error.BadConfig,
    };
    return if (std.math.isFinite(n)) n else error.BadConfig;
}

fn optional(o: std.json.ObjectMap, key: []const u8, fallback: f64) !f64 {
    const v = o.get(key) orelse return fallback;
    return if (v == .null) fallback else number(v);
}

fn flag(o: std.json.ObjectMap, key: []const u8) !bool {
    const v = o.get(key) orelse return false;
    return if (v == .bool) v.bool else error.BadConfig;
}

fn eos(c: *Config, v: std.json.Value) !void {
    if (v == .null) return;
    if (v == .array) {
        for (v.array.items) |id| try eos(c, id);
        return;
    }
    const id: u32 = @intCast(try integer(v, true));
    if (id >= c.vocab) return error.BadEos;
    if (c.isEos(id)) return;
    if (c.eos_count == max_eos) return error.TooManyEos;
    c.eos[c.eos_count] = id;
    c.eos_count += 1;
}

fn shapeProducts(c: Config) !void {
    const q = try std.math.mul(usize, c.heads, c.head_dim);
    _ = try std.math.mul(usize, q, 2);
    _ = try std.math.mul(usize, c.kv_heads, c.head_dim);
    const k = try std.math.mul(usize, c.k_heads, c.dk);
    const v = try std.math.mul(usize, c.v_heads, c.dv);
    const conv = try std.math.add(usize, try std.math.mul(usize, k, 2), v);
    _ = try std.math.mul(usize, conv, c.conv_kernel);
    _ = try std.math.mul(usize, try std.math.mul(usize, v, c.dk), 4);
    _ = try std.math.mul(usize, c.hidden, c.intermediate);
    _ = try std.math.mul(usize, c.hidden, c.vocab);
}

pub fn parse(gpa: std.mem.Allocator, json: []const u8, generation: ?[]const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const root = try object(parsed.value);
    const t = if (root.get("text_config")) |v| try object(v) else root;
    const model = t.get("model_type") orelse root.get("model_type") orelse return error.BadConfig;
    if (model != .string or (!std.mem.eql(u8, model.string, "qwen3_5") and !std.mem.eql(u8, model.string, "qwen3_5_text"))) return error.NotQwenDense;
    if (t.get("num_experts")) |v| if (try integer(v, true) != 0) return error.NotQwenDense;
    if (try flag(root, "tie_word_embeddings") or try flag(t, "tie_word_embeddings")) return error.TiedEmbeddingHead;
    if (t.get("attn_output_gate")) |v| if (v != .bool or !v.bool) return error.UnsupportedAttentionGate;
    inline for (.{ .{ "hidden_act", "silu" }, .{ "output_gate_type", "swish" }, .{ "linear_attn_state_dtype", "float32" } }) |entry| {
        if (t.get(entry[0])) |v| if (v != .string or !std.mem.eql(u8, v.string, entry[1])) return error.UnsupportedActivation;
    }
    const heads = try required(t, "num_attention_heads");
    const hidden = try required(t, "hidden_size");
    const head_dim = if (t.get("head_dim")) |v| blk: {
        if (v == .null or (v == .integer and v.integer == 0)) break :blk hidden / heads;
        break :blk try integer(v, false);
    } else hidden / heads;
    const rope = if (t.get("rope_parameters")) |v| (if (v == .null) null else try object(v)) else null;
    const factor = if (rope) |r| try optional(r, "partial_rotary_factor", try optional(t, "partial_rotary_factor", 0.25)) else try optional(t, "partial_rotary_factor", 0.25);
    if (factor <= 0 or factor > 1) return error.BadConfig;
    var c = Config{
        .hidden = hidden,
        .intermediate = try required(t, "intermediate_size"),
        .layers = try required(t, "num_hidden_layers"),
        .vocab = try required(t, "vocab_size"),
        .heads = heads,
        .kv_heads = try required(t, "num_key_value_heads"),
        .head_dim = head_dim,
        .k_heads = try required(t, "linear_num_key_heads"),
        .v_heads = try required(t, "linear_num_value_heads"),
        .dk = try required(t, "linear_key_head_dim"),
        .dv = try required(t, "linear_value_head_dim"),
        .conv_kernel = try required(t, "linear_conv_kernel_dim"),
        .eps = @floatCast(try optional(t, "rms_norm_eps", 1e-6)),
        .rope_dims = @intFromFloat(@as(f64, @floatFromInt(head_dim)) * factor),
        .rope_theta = if (rope) |r| try optional(r, "rope_theta", try optional(t, "rope_theta", 10000000)) else try optional(t, "rope_theta", 10000000),
    };
    if (t.get("max_position_embeddings")) |v| c.max_position = try integer(v, false);
    if (rope) |r| if (r.get("mrope_interleaved")) |v| {
        if (v != .bool or !v.bool) return error.UnsupportedMrope;
        c.mrope_interleaved = v.bool;
    };
    if (c.layers > max_layers or c.head_dim == 0 or c.heads % c.kv_heads != 0 or c.v_heads % c.k_heads != 0) return error.BadConfig;
    if (!std.math.isFinite(c.eps) or c.eps <= 0 or c.rope_theta <= 0 or c.rope_dims == 0 or c.rope_dims % 2 != 0) return error.BadConfig;
    if (rope) |r| if (r.get("mrope_section")) |v| {
        if (v != .array or v.array.items.len != 3) return error.BadConfig;
        for (v.array.items, 0..) |x, i| c.mrope_section[i] = try integer(x, true);
    };
    if (t.get("layer_types")) |v| {
        if (v != .array or v.array.items.len != c.layers) return error.BadConfig;
        for (v.array.items, 0..) |x, i| {
            if (x != .string) return error.BadConfig;
            c.kinds[i] = if (std.mem.eql(u8, x.string, "linear_attention")) .linear else if (std.mem.eql(u8, x.string, "full_attention")) .attention else return error.UnsupportedLayer;
        }
    } else {
        const interval = if (t.get("full_attention_interval")) |v| try integer(v, false) else 4;
        for (c.kinds[0..c.layers], 0..) |*k, i| k.* = if ((i + 1) % interval == 0) .attention else .linear;
    }
    if (root.get("eos_token_id") orelse t.get("eos_token_id")) |v| try eos(&c, v);
    if (generation) |text| {
        const gen = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer gen.deinit();
        if ((try object(gen.value)).get("eos_token_id")) |v| try eos(&c, v);
    }
    if (c.eos_count == 0) return error.MissingEos;
    try shapeProducts(c);
    return c;
}

const fixture =
    \\{"model_type":"qwen3_5","hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":97,
    \\"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,
    \\"linear_value_head_dim":16,"linear_conv_kernel_dim":4,"eos_token_id":[2,11]}
;

fn edit(a: std.mem.Allocator, key: []const u8, value: std.json.Value) ![]u8 {
    var p = try std.json.parseFromSlice(std.json.Value, a, fixture, .{});
    defer p.deinit();
    try p.value.object.put(p.arena.allocator(), key, value);
    return std.json.Stringify.valueAlloc(a, p.value, .{});
}

test "flat config preserves dimensions and interval schedule" {
    const c = try parse(std.testing.allocator, fixture, null);
    try std.testing.expectEqual(@as(usize, 128), c.qDim());
    try std.testing.expectEqual(@as(usize, 32), c.kvDim());
    try std.testing.expectEqual(@as(usize, 128), c.gdnQkvDim());
    try std.testing.expectEqual(Kind.linear, c.kind(0));
    try std.testing.expectEqual(Kind.attention, c.kind(3));
}

test "nested config merges generation EOS in stable order" {
    const a = std.testing.allocator;
    const text = try std.fmt.allocPrint(a, "{{\"model_type\":\"qwen3_5\",\"eos_token_id\":[11,2],\"text_config\":{s}}}", .{fixture});
    defer a.free(text);
    const c = try parse(a, text, "{\"eos_token_id\":[2,3,11]}");
    try std.testing.expectEqualSlices(u32, &.{ 11, 2, 3 }, c.eos[0..c.eos_count]);
    try std.testing.expect(c.isEos(3) and !c.isEos(4));
}

test "explicit layer types preserve their order" {
    const a = std.testing.allocator;
    var p = try std.json.parseFromSlice(std.json.Value, a, "[\"full_attention\",\"linear_attention\",\"full_attention\",\"linear_attention\"]", .{});
    defer p.deinit();
    const json = try edit(a, "layer_types", p.value);
    defer a.free(json);
    const c = try parse(a, json, null);
    try std.testing.expectEqualSlices(Kind, &.{ .attention, .linear, .attention, .linear }, c.kinds[0..4]);
}

test "rope parameters preserve precision and section metadata" {
    const a = std.testing.allocator;
    const p = try std.json.parseFromSlice(std.json.Value, a, "{\"rope_theta\":1234567.25,\"partial_rotary_factor\":0.5,\"mrope_section\":[2,1,1]}", .{});
    defer p.deinit();
    const json = try edit(a, "rope_parameters", p.value);
    defer a.free(json);
    const c = try parse(a, json, null);
    try std.testing.expectEqual(@as(f64, 1234567.25), c.rope_theta);
    try std.testing.expectEqual(@as(usize, 8), c.rope_dims);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 1 }, &c.mrope_section);
}

test "dense config refuses MoE and tied embedding heads" {
    const a = std.testing.allocator;
    const moe = try edit(a, "num_experts", .{ .integer = 8 });
    defer a.free(moe);
    try std.testing.expectError(error.NotQwenDense, parse(a, moe, null));
    const tied = try edit(a, "tie_word_embeddings", .{ .bool = true });
    defer a.free(tied);
    try std.testing.expectError(error.TiedEmbeddingHead, parse(a, tied, null));
}

test "invalid scalar shapes and types fail without a cast trap" {
    const a = std.testing.allocator;
    for ([_]std.json.Value{ .{ .integer = -1 }, .{ .integer = 0 }, .{ .bool = true }, .{ .integer = 4294967296 } }) |v| {
        const json = try edit(a, "hidden_size", v);
        defer a.free(json);
        try std.testing.expectError(error.BadConfig, parse(a, json, null));
    }
    const json = try edit(a, "num_hidden_layers", .{ .integer = 129 });
    defer a.free(json);
    try std.testing.expectError(error.BadConfig, parse(a, json, null));
}

test "missing and out of vocabulary EOS fail" {
    const a = std.testing.allocator;
    const missing = try edit(a, "eos_token_id", .null);
    defer a.free(missing);
    try std.testing.expectError(error.MissingEos, parse(a, missing, null));
    const c = try parse(a, missing, "{\"eos_token_id\":7}");
    try std.testing.expect(c.isEos(7));
    const wrong = try edit(a, "eos_token_id", .{ .integer = 97 });
    defer a.free(wrong);
    try std.testing.expectError(error.BadEos, parse(a, wrong, null));
}

test "malformed layer and rope metadata fail" {
    const a = std.testing.allocator;
    const layers = try edit(a, "layer_types", .{ .bool = true });
    defer a.free(layers);
    try std.testing.expectError(error.BadConfig, parse(a, layers, null));
    const rope = try edit(a, "partial_rotary_factor", .{ .float = 2 });
    defer a.free(rope);
    try std.testing.expectError(error.BadConfig, parse(a, rope, null));
    try std.testing.expectError(error.BadConfig, parse(a, "[]", null));
}

test "head grouping requires complete key and value groups" {
    const a = std.testing.allocator;
    const attention = try edit(a, "num_key_value_heads", .{ .integer = 3 });
    defer a.free(attention);
    try std.testing.expectError(error.BadConfig, parse(a, attention, null));
    const gdn = try edit(a, "linear_num_value_heads", .{ .integer = 3 });
    defer a.free(gdn);
    try std.testing.expectError(error.BadConfig, parse(a, gdn, null));
}

test "text execution refuses unsupported gates and activation contracts" {
    const a = std.testing.allocator;
    const gate = try edit(a, "attn_output_gate", .{ .bool = false });
    defer a.free(gate);
    try std.testing.expectError(error.UnsupportedAttentionGate, parse(a, gate, null));
    const act = try edit(a, "output_gate_type", .{ .string = "sigmoid" });
    defer a.free(act);
    try std.testing.expectError(error.UnsupportedActivation, parse(a, act, null));
}
