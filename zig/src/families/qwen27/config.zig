//! Qwen3.5-class text shapes (dense or routed), layer schedule, rotary metadata and ordered checkpoint EOS ids.
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
    /// Static YaRN: factor (0: plain), stretched window, ramp betas, attention factor (0: 0.1 ln(factor) + 1).
    yarn_factor: f64 = 0,
    yarn_original: usize = 0,
    yarn_beta: [2]f64 = .{ 32, 1 },
    yarn_attention: f64 = 0,
    /// Routed experts (qwen3_5_moe): their count and the picks a row makes; 0: a dense MLP of `intermediate`.
    experts: usize = 0,
    top_k: usize = 0,
    /// A routed expert's width; `intermediate` is then the shared expert's (the same, as the CUDA family requires).
    moe_intermediate: usize = 0,
    /// Whether a row's top-k softmax weights are renormalized to sum to one (`norm_topk_prob`).
    renorm: bool = false,

    pub fn moe(c: Config) bool {
        return c.experts != 0;
    }

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

    /// Fit `context` past the trained window with static YaRN at the window multiple rounded up (at most 4: 1M).
    pub fn extend(c: *Config, context: usize) !void {
        if (context <= c.max_position) return;
        if (c.yarn_factor != 0 or c.rope_dims != 64) return error.ContextFull;
        const factor = std.math.divCeil(usize, context, c.max_position) catch unreachable;
        if (factor > 4) return error.ContextFull;
        c.yarn_factor = @floatFromInt(factor);
        c.yarn_original = c.max_position;
        c.max_position *= factor;
    }

    /// The rotary pairs' YaRN frequencies and cos/sin scale (null: plain rope), built as vLLM's YaRN rotary does.
    pub fn yarn(c: Config) ?Yarn {
        if (c.yarn_factor == 0) return null;
        const dim: f64 = 64;
        const original: f64 = @floatFromInt(c.yarn_original);
        const dimOf = struct {
            fn f(rotations: f64, base: f64, positions: f64) f64 {
                return 64 * @log(positions / (rotations * 2 * std.math.pi)) / (2 * @log(base));
            }
        }.f;
        const low = @max(@floor(dimOf(c.yarn_beta[0], c.rope_theta, original)), 0);
        var high = @min(@ceil(dimOf(c.yarn_beta[1], c.rope_theta, original)), dim - 1);
        if (low == high) high += 0.001;
        var y: Yarn = .{ .freq = undefined, .scale = @floatCast(if (c.yarn_attention != 0) c.yarn_attention else 0.1 * @log(c.yarn_factor) + 1) };
        for (&y.freq, 0..) |*f, k| {
            const kf: f64 = @floatFromInt(k);
            const pos_freq: f32 = @floatCast(std.math.pow(f64, c.rope_theta, @as(f64, @as(f32, @floatCast(2 * kf / dim)))));
            const extrapolated: f32 = 1.0 / pos_freq;
            const interpolated: f32 = 1.0 / (@as(f32, @floatCast(c.yarn_factor)) * pos_freq);
            const keep: f32 = 1 - @as(f32, @floatCast(std.math.clamp((kf - low) / (high - low), 0, 1)));
            f.* = interpolated * (1 - keep) + extrapolated * keep;
        }
        return y;
    }

    pub fn isEos(c: Config, token: u32) bool {
        return std.mem.indexOfScalar(u32, c.eos[0..c.eos_count], token) != null;
    }
};

pub const Yarn = struct { freq: [32]f32, scale: f32 };

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

/// `rope_parameters.rope_type` (default or yarn), or a `rope_scaling` block beside it in either config level.
fn scaling(c: *Config, root: std.json.ObjectMap, t: std.json.ObjectMap, rope: ?std.json.ObjectMap) !void {
    const block = for ([_]?std.json.ObjectMap{ if (t.get("rope_scaling")) |v| (if (v == .null) null else try object(v)) else null, if (root.get("rope_scaling")) |v| (if (v == .null) null else try object(v)) else null, rope }) |b| {
        const o = b orelse continue;
        const kind = o.get("rope_type") orelse o.get("type") orelse continue;
        if (kind != .string) return error.UnsupportedRope;
        if (std.mem.eql(u8, kind.string, "default")) continue;
        if (!std.mem.eql(u8, kind.string, "yarn")) return error.UnsupportedRope;
        break o;
    } else return;
    if (block.get("mscale") != null or block.get("mscale_all_dim") != null or c.rope_dims != 64) return error.UnsupportedRope;
    c.yarn_factor = try number(block.get("factor") orelse return error.BadConfig);
    if (c.yarn_factor <= 1) return error.BadConfig;
    c.yarn_original = if (block.get("original_max_position_embeddings")) |v| try integer(v, false) else c.max_position;
    c.yarn_beta = .{ try optional(block, "beta_fast", 32), try optional(block, "beta_slow", 1) };
    c.yarn_attention = try optional(block, "attention_factor", 0);
    if (c.yarn_beta[0] <= c.yarn_beta[1] or c.yarn_beta[1] <= 0 or c.yarn_attention < 0) return error.BadConfig;
    c.max_position = @max(c.max_position, @as(usize, @intFromFloat(@floor(c.yarn_factor * @as(f64, @floatFromInt(c.yarn_original))))));
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
    const kinds_of = [_][]const u8{ "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text" };
    if (model != .string) return error.NotQwenDense;
    const routed = for (kinds_of, 0..) |k, i| {
        if (std.mem.eql(u8, model.string, k)) break i >= 2;
    } else return error.NotQwenDense;
    const experts = if (t.get("num_experts")) |v| try integer(v, true) else 0;
    if (routed != (experts != 0)) return error.NotQwenDense;
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
        .intermediate = if (routed) try required(t, "shared_expert_intermediate_size") else try required(t, "intermediate_size"),
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
    if (routed) {
        c.experts = experts;
        c.top_k = try required(t, "num_experts_per_tok");
        c.moe_intermediate = try required(t, "moe_intermediate_size");
        // absent means true, as Qwen3.5-MoE's reference config and the CUDA family read it
        c.renorm = if (t.get("norm_topk_prob")) |v| (if (v == .bool) v.bool else return error.BadConfig) else true;
        if (c.experts > 256 or c.top_k >= c.experts or c.top_k > 15 or c.moe_intermediate != c.intermediate) return error.UnsupportedRouting;
    }
    if (t.get("max_position_embeddings")) |v| c.max_position = try integer(v, false);
    if (rope) |r| if (r.get("mrope_interleaved")) |v| {
        if (v != .bool or !v.bool) return error.UnsupportedMrope;
        c.mrope_interleaved = v.bool;
    };
    if (c.layers > max_layers or c.head_dim == 0 or c.heads % c.kv_heads != 0 or c.v_heads % c.k_heads != 0) return error.BadConfig;
    if (!std.math.isFinite(c.eps) or c.eps <= 0 or c.rope_theta <= 0 or c.rope_dims == 0 or c.rope_dims % 2 != 0) return error.BadConfig;
    try scaling(&c, root, t, rope);
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

test "a routed config reads its experts, the shared expert's width as the layer's and the routing flag" {
    const a = std.testing.allocator;
    const json =
        \\{"model_type":"qwen3_5_moe","text_config":{"model_type":"qwen3_5_moe_text","hidden_size":3072,"intermediate_size":1024,
        \\"moe_intermediate_size":1024,"shared_expert_intermediate_size":1024,"num_experts":256,"num_experts_per_tok":8,
        \\"num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":2,"head_dim":256,"vocab_size":248320,
        \\"linear_num_key_heads":16,"linear_num_value_heads":64,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\"linear_conv_kernel_dim":4,"partial_rotary_factor":0.25,"eos_token_id":248044}}
    ;
    const c = try parse(a, json, null);
    try std.testing.expect(c.moe());
    try std.testing.expectEqual(@as(usize, 256), c.experts);
    try std.testing.expectEqual(@as(usize, 8), c.top_k);
    try std.testing.expectEqual(@as(usize, 1024), c.moe_intermediate);
    try std.testing.expectEqual(@as(usize, 1024), c.intermediate);
    try std.testing.expect(c.renorm); // absent: renormalized
    try std.testing.expectEqual(Kind.attention, c.kind(3));
    try std.testing.expectEqual(Kind.linear, c.kind(2));
    // a routed model type without experts, and a dense one with them, are refused
    const dense_typed = try std.mem.replaceOwned(u8, a, json, "\"num_experts\":256", "\"num_experts\":0");
    defer a.free(dense_typed);
    try std.testing.expectError(error.NotQwenDense, parse(a, dense_typed, null));
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

test "a yarn rope_scaling block stretches the window, its table vLLM's (theta 1e7, x4 over 262,144: ramp 14 to 22)" {
    const a = std.testing.allocator;
    const json = try std.mem.replaceOwned(u8, a, fixture, "\"head_dim\":16,", "\"head_dim\":256,\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":262144},");
    defer a.free(json);
    const c = try parse(a, json, null);
    try std.testing.expectEqual(@as(usize, 1048576), c.max_position);
    const y = c.yarn().?;
    try std.testing.expectApproxEqRel(@as(f32, 1.1386294), y.scale, 1e-6);
    // pairs below the ramp keep their frequency, above it a quarter of it, between a blend (numpy float32)
    for ([_]usize{ 0, 5, 10, 16, 20, 31 }, [_]f32{ 1.0, 0.08058421313762665, 0.006493816617876291, 0.00025693507632240653, 1.844922371674329e-05, 4.1370427084075345e-08 }) |k, want| {
        try std.testing.expectApproxEqRel(want, y.freq[k], 2e-6);
    }
}

test "a context past the window turns on YaRN by the window multiple, up to four; other rope types refuse" {
    const a = std.testing.allocator;
    var c = try parse(a, fixture, null);
    c.rope_dims = 64;
    try c.extend(262144);
    try std.testing.expect(c.yarn() == null);
    var half = c;
    try half.extend(500000);
    try std.testing.expectEqual(@as(f64, 2), half.yarn_factor);
    try c.extend(1000000);
    try std.testing.expectEqual(@as(f64, 4), c.yarn_factor);
    try std.testing.expectEqual(@as(usize, 1048576), c.max_position);
    var far = try parse(a, fixture, null);
    far.rope_dims = 64;
    try std.testing.expectError(error.ContextFull, far.extend(1048577));
    const p = try std.json.parseFromSlice(std.json.Value, a, "{\"rope_type\":\"linear\",\"factor\":2.0}", .{});
    defer p.deinit();
    const json = try edit(a, "rope_scaling", p.value);
    defer a.free(json);
    try std.testing.expectError(error.UnsupportedRope, parse(a, json, null));
}
