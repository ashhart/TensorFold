//! Qwen text shapes (a dense MLP or routed experts), layer schedule, rotary metadata and ordered checkpoint EOS ids.
const std = @import("std");
const family = @import("core").qwen35_config;

pub const Kind = enum { linear, attention };
pub const max_layers = family.checks.max_layers;
pub const max_eos = family.checks.max_eos;

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
    /// Routed experts (0: every layer's MLP is dense, `intermediate` wide), picks a token, and the expert widths.
    experts: usize = 0,
    topk: usize = 0,
    moe_intermediate: usize = 0,
    shared_intermediate: usize = 0,

    pub fn kind(c: Config, layer: usize) Kind {
        return c.kinds[layer];
    }

    pub fn moe(c: Config) bool {
        return c.experts > 0;
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

fn flag(o: std.json.ObjectMap, key: []const u8) !bool {
    const v = o.get(key) orelse return false;
    return if (v == .bool) v.bool else error.BadConfig;
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

/// config.json through the family's reader, the CUDA and HIP engines' too; the Metal engine refuses a tied head.
pub fn parse(gpa: std.mem.Allocator, json: []const u8, generation: ?[]const u8) !Config {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    return read(arena.allocator(), json, generation) catch |e| switch (e) {
        error.InvalidConfig, error.MissingField, error.InvalidMoe, error.InvalidRotary, error.InvalidLayerTypes => error.BadConfig,
        else => e,
    };
}

fn read(a: std.mem.Allocator, json: []const u8, generation: ?[]const u8) !Config {
    const root = try object(try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}));
    const t = try family.textTower(root);
    const model = t.get("model_type") orelse root.get("model_type") orelse return error.BadConfig;
    if (model != .string) return error.NotQwenDense;
    const moe = std.mem.eql(u8, model.string, "qwen3_5_moe") or std.mem.eql(u8, model.string, "qwen3_5_moe_text");
    if (!moe and !std.mem.eql(u8, model.string, "qwen3_5") and !std.mem.eql(u8, model.string, "qwen3_5_text")) return error.NotQwenDense;
    if (!moe) if (t.get("num_experts")) |v| if (v != .null and !(v == .integer and v.integer == 0)) return error.NotQwenDense;
    // the router's weights are the picks' softmax, renormalized (Qwen's default)
    if (moe) if (t.get("norm_topk_prob")) |v| if (v != .bool or !v.bool) return error.UnsupportedRouter;
    if (try flag(root, "tie_word_embeddings") or try flag(t, "tie_word_embeddings")) return error.TiedEmbeddingHead;
    // the bits and group are the checkpoint's, read per module (widths.zig)
    const s = try family.geometry(t, 4, 64);
    if (!moe and s.experts != 0) return error.NotQwenDense;
    if (moe and (s.experts == 0 or s.shared_width == 0)) return error.BadConfig;
    if (!moe and s.intermediate == 0) return error.BadConfig;
    for ([_]usize{ s.hidden, s.n_layers, s.vocab, s.heads, s.kv_heads, s.head_dim, s.key_heads, s.value_heads, s.key_dim, s.value_dim, s.conv }) |x| {
        if (x == 0 or x > std.math.maxInt(u32)) return error.BadConfig;
    }
    var c = Config{
        .hidden = s.hidden,
        .intermediate = if (moe) 0 else s.intermediate,
        .layers = s.n_layers,
        .vocab = s.vocab,
        .heads = s.heads,
        .kv_heads = s.kv_heads,
        .head_dim = s.head_dim,
        .k_heads = s.key_heads,
        .v_heads = s.value_heads,
        .dk = s.key_dim,
        .dv = s.value_dim,
        .conv_kernel = s.conv,
        .eps = @floatCast(s.eps),
        .rope_dims = s.rotary_dim,
        .rope_theta = s.rope_theta,
        .mrope_section = s.mrope.section,
        .mrope_interleaved = s.mrope.interleaved,
        .max_position = s.max_position,
        .experts = s.experts,
        .topk = s.top_k,
        .moe_intermediate = s.moe_width,
        .shared_intermediate = s.shared_width,
    };
    if (!std.math.isFinite(c.eps) or c.eps <= 0 or !std.math.isFinite(c.rope_theta) or c.rope_theta <= 0) return error.BadConfig;
    for (c.kinds[0..c.layers], s.full_layers[0..c.layers]) |*k, full| k.* = if (full) .attention else .linear;
    const gen = if (generation) |text| try object(try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})) else null;
    var ids: [family.checks.max_eos]u32 = undefined;
    const eos = try family.checks.eos(&ids, root, t, gen, c.vocab);
    @memcpy(c.eos[0..eos.len], eos);
    c.eos_count = eos.len;
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

test "a MoE config reads its experts and has no dense MLP width" {
    const a = std.testing.allocator;
    const text =
        \\{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":64,
        \\"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":97,
        \\"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,"linear_value_head_dim":16,
        \\"linear_conv_kernel_dim":4,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":32,
        \\"shared_expert_intermediate_size":32,"eos_token_id":2}}
    ;
    const c = try parse(a, text, null);
    try std.testing.expect(c.moe());
    try std.testing.expectEqual(@as(usize, 0), c.intermediate);
    try std.testing.expectEqual([4]usize{ 8, 2, 32, 32 }, [4]usize{ c.experts, c.topk, c.moe_intermediate, c.shared_intermediate });
    const plain = try std.mem.replaceOwned(u8, a, text, "\"num_experts_per_tok\":2", "\"num_experts_per_tok\":2,\"norm_topk_prob\":false");
    defer a.free(plain);
    try std.testing.expectError(error.UnsupportedRouter, parse(a, plain, null));
    const missing = try std.mem.replaceOwned(u8, a, text, "\"num_experts\":8,", "");
    defer a.free(missing);
    try std.testing.expectError(error.BadConfig, parse(a, missing, null));
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
    const out_gate = try edit(a, "output_gate_type", .{ .string = "sigmoid" });
    defer a.free(out_gate);
    try std.testing.expectError(error.UnsupportedAttentionGate, parse(a, out_gate, null));
    const act = try edit(a, "hidden_act", .{ .string = "gelu" });
    defer a.free(act);
    try std.testing.expectError(error.UnsupportedActivation, parse(a, act, null));
}
