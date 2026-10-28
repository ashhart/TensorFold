//! Gemma 4's MTP assistant (gemma4_unified_assistant): config admitted, bf16 linears and the tied head made 4-bit.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const dw = @import("dflash_weights.zig");
const weights = @import("weights.zig");

pub const Linear = dw.Linear;
pub const Ref = dw.Ref;

/// Layers an admitted assistant has at most: the size of per-layer arrays.
pub const max_layers = 8;
/// The stock assistant's sizes (google/gemma-4-12B-it-assistant); another config is refused with its reason.
const stock = .{ .hidden = 1024, .layers = 4, .intermediate = 8192, .heads = 16 };

pub const Config = struct {
    layers: u32,
    hidden: u32,
    intermediate: u32,
    heads: u32,
    /// Each layer's kind: true for a sliding layer (the target's last sliding cache), false for a full one.
    sliding: [max_layers]bool = @splat(true),
    /// The assistant's sizes as the glue's geometry: its width, its query heads over the target's key heads.
    glue: cfg.Geometry,

    /// `geo`: the target's sizes, whose width the assistant was trained on and whose caches it reads.
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, geo: cfg.Geometry) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes, geo);
    }

    pub fn parse(gpa: std.mem.Allocator, bytes: []const u8, geo: cfg.Geometry) !Config {
        var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer parsed.deinit();
        const o = if (parsed.value == .object) parsed.value.object else return error.BadDraftConfig;
        if (!isString(o.get("model_type"), "gemma4_unified_assistant")) return error.UnsupportedAssistantType;
        if (isTrue(o, "use_ordered_embeddings")) return error.AssistantOrderedEmbeddingsNotServed;
        if (try integer(o, "backbone_hidden_size") != geo.hidden) return error.AssistantForAnotherTarget;
        if (o.get("quantization") != null or o.get("quantization_config") != null) return error.UnsupportedAssistantQuantization;
        const t = if (o.get("text_config")) |v| (if (v == .object) v.object else return error.BadDraftConfig) else return error.BadDraftConfig;
        if (!isString(t.get("model_type"), "gemma4_unified_text")) return error.BadDraftConfig;
        var c: Config = .{ .layers = @intCast(try integer(t, "num_hidden_layers")), .hidden = @intCast(try integer(t, "hidden_size")), .intermediate = @intCast(try integer(t, "intermediate_size")), .heads = @intCast(try integer(t, "num_attention_heads")), .glue = undefined };
        if (c.hidden != stock.hidden or c.layers != stock.layers or c.intermediate != stock.intermediate or c.heads != stock.heads) return error.UnsupportedAssistantGeometry;
        if (try integer(t, "num_key_value_heads") != geo.sliding_kv_heads or try integer(t, "num_global_key_value_heads") != geo.global_kv_heads) return error.AssistantForAnotherTarget;
        if (try integer(t, "num_kv_shared_layers") != c.layers) return error.UnsupportedAssistantGeometry; // every layer reads the target's
        const dims = .{ .{ "head_dim", cfg.sliding_dim }, .{ "global_head_dim", cfg.global_dim }, .{ "sliding_window", cfg.window }, .{ "vocab_size", cfg.vocab } };
        inline for (dims) |d| if (try integer(t, d[0]) != d[1]) return error.UnsupportedAssistantGeometry;
        if (isTrue(t, "enable_moe_block") or isTrue(t, "use_double_wide_mlp") or isTrue(t, "attention_bias")) return error.UnsupportedAssistantGeometry;
        if (t.get("hidden_size_per_layer_input")) |v| if (v != .integer or v.integer != 0) return error.UnsupportedAssistantGeometry;
        if (!isTrue(t, "attention_k_eq_v") or !isTrue(t, "tie_word_embeddings")) return error.UnsupportedAssistantGeometry;
        if (!isString(t.get("hidden_activation"), "gelu_pytorch_tanh")) return error.UnsupportedAssistantGeometry;
        if (t.get("final_logit_softcapping")) |v| if (v != .null) return error.UnsupportedAssistantGeometry;
        if (try number(t, "rms_norm_eps") != 1e-6) return error.UnsupportedAssistantGeometry;
        const ropes = if (t.get("rope_parameters")) |v| (if (v == .object) v.object else return error.BadDraftConfig) else return error.BadDraftConfig;
        try rope(ropes, "sliding_attention", "default", cfg.sliding_theta, null);
        try rope(ropes, "full_attention", "proportional", cfg.global_theta, 0.25);
        const kinds = t.get("layer_types") orelse return error.BadDraftConfig;
        if (kinds != .array or kinds.array.items.len != c.layers) return error.BadDraftConfig;
        for (kinds.array.items, 0..) |k, i| {
            c.sliding[i] = isString(k, "sliding_attention");
            if (!c.sliding[i] and !isString(k, "full_attention")) return error.UnsupportedAssistantGeometry;
        }
        c.glue = .{ .name = "assistant", .hidden = c.hidden, .layers = c.layers, .intermediate = c.intermediate, .query_heads = c.heads, .sliding_kv_heads = geo.sliding_kv_heads, .global_kv_heads = geo.global_kv_heads };
        return c;
    }

    /// Layer `i`'s head dim: the target's sliding or global one.
    pub fn headDim(c: *const Config, i: usize) usize {
        return if (c.sliding[i]) cfg.sliding_dim else cfg.global_dim;
    }

    /// The shape tensor `name` must have, or null for a tensor the assistant does not read.
    pub fn expected(c: *const Config, geo: cfg.Geometry, name: []const u8) ?[2]usize {
        const hid: usize = c.hidden;
        if (std.mem.eql(u8, name, "pre_projection.weight")) return .{ hid, 2 * geo.hidden };
        if (std.mem.eql(u8, name, "post_projection.weight")) return .{ geo.hidden, hid };
        if (std.mem.eql(u8, name, "model.embed_tokens.weight")) return .{ cfg.vocab, hid };
        if (std.mem.eql(u8, name, "model.norm.weight")) return .{ hid, 1 };
        const rest = if (std.mem.startsWith(u8, name, "model.layers.")) name["model.layers.".len..] else return null;
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
        const i = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
        if (i >= c.layers) return null;
        const q = c.heads * c.headDim(i);
        const shapes = .{
            .{ "self_attn.q_proj.weight", [2]usize{ q, hid } },          .{ "self_attn.o_proj.weight", [2]usize{ hid, q } },
            .{ "self_attn.q_norm.weight", [2]usize{ c.headDim(i), 1 } }, .{ "mlp.gate_proj.weight", [2]usize{ c.intermediate, hid } },
            .{ "mlp.up_proj.weight", [2]usize{ c.intermediate, hid } },  .{ "mlp.down_proj.weight", [2]usize{ hid, c.intermediate } },
            .{ "input_layernorm.weight", [2]usize{ hid, 1 } },           .{ "post_attention_layernorm.weight", [2]usize{ hid, 1 } },
            .{ "pre_feedforward_layernorm.weight", [2]usize{ hid, 1 } }, .{ "post_feedforward_layernorm.weight", [2]usize{ hid, 1 } },
            .{ "layer_scalar", [2]usize{ 1, 1 } },
        };
        inline for (shapes) |s| if (std.mem.eql(u8, rest[dot + 1 ..], s[0])) return s[1];
        return null;
    }
};

/// Whether `dir` holds a Gemma 4 assistant (its config's model_type names one), not a DFlash drafter.
pub fn isAssistant(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    return typeIsAssistant(gpa, bytes);
}

fn typeIsAssistant(gpa: std.mem.Allocator, bytes: []const u8) !bool {
    var parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    defer parsed.deinit();
    const t = if (parsed.value == .object) parsed.value.object.get("model_type") else null;
    return t != null and t.? == .string and std.mem.startsWith(u8, t.?.string, "gemma4") and std.mem.endsWith(u8, t.?.string, "assistant");
}

fn isString(v: ?std.json.Value, expected: []const u8) bool {
    const s = v orelse return false;
    return s == .string and std.mem.eql(u8, s.string, expected);
}

fn isTrue(o: std.json.ObjectMap, key: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool;
}

fn integer(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return if (v == .integer) v.integer else error.BadDraftConfig;
}

fn number(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadDraftConfig;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.BadDraftConfig,
    };
}

fn rope(o: std.json.ObjectMap, kind: []const u8, kind_type: []const u8, theta: f64, partial: ?f64) !void {
    const v = o.get(kind) orelse return error.BadDraftConfig;
    const r = if (v == .object) v.object else return error.BadDraftConfig;
    if (!isString(r.get("rope_type"), kind_type) or try number(r, "rope_theta") != theta) return error.UnsupportedAssistantRotary;
    if (partial) |p| if (try number(r, "partial_rotary_factor") != p) return error.UnsupportedAssistantRotary;
}

pub const Layer = struct {
    q: Linear,
    o: Linear,
    gate_up: Linear,
    down: Linear,
    q_norm: Ref,
    input_norm: Ref,
    post_attention: Ref,
    pre_feedforward: Ref,
    post_feedforward: Ref,
    scalar: Ref,
};

pub const Weights = struct {
    /// [embedding | hidden] to the assistant's width, and its output back to the target's.
    pre: Linear,
    post: Linear,
    /// The tied head: embed_tokens as a 4-bit linear over the vocabulary.
    head: Linear,
    norm: Ref,
    layers: [max_layers]Layer,
    owned: std.ArrayList(mtl.Buffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.owned.items) |b| b.deinit();
        w.owned.deinit(gpa);
    }
};

/// Every tensor the directory holds has the shape `c` expects (an unknown one is refused), and each linear is 4-bit.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, io: std.Io, dir: []const u8, c: *const Config, geo: cfg.Geometry) !Weights {
    var check = ckpt.Checkpoint.init(gpa);
    defer check.deinit();
    const files = try shards.shardFiles(gpa, io, dir);
    defer shards.freeShardFiles(gpa, files);
    for (files) |path| try check.addFile(device, path, "");
    var it = check.tensors.iterator();
    while (it.next()) |e| {
        const want = c.expected(geo, e.key_ptr.*) orelse {
            std.log.err("gemma4 assistant: {s} is not a tensor the assistant reads", .{e.key_ptr.*});
            return error.BadDraftWeight;
        };
        const t = e.value_ptr.*;
        const got: [2]usize = if (t.rank == 1) .{ t.shape[0], 1 } else .{ t.shape[0], t.shape[1] };
        if (t.dtype != .bf16 or t.rank == 0 or t.rank > 2 or !std.mem.eql(usize, &got, &want)) return error.BadDraftWeight;
    }
    var w: Weights = undefined;
    w.owned = .empty;
    errdefer w.deinit(gpa);
    var p: [64]u8 = undefined;
    var name: [160]u8 = undefined;
    w.pre = try dw.quantized(gpa, device, &check, &w.owned, geo, "", &.{"pre_projection"});
    w.post = try dw.quantized(gpa, device, &check, &w.owned, geo, "", &.{"post_projection"});
    w.head = try dw.quantized(gpa, device, &check, &w.owned, geo, "model.", &.{"embed_tokens"});
    w.norm = try dw.vector(gpa, device, &check, &w.owned, "model.norm.weight", c.hidden);
    for (w.layers[0..c.layers], 0..) |*l, i| {
        const pre = try std.fmt.bufPrint(&p, "model.layers.{d}.", .{i});
        l.q = try dw.quantized(gpa, device, &check, &w.owned, geo, pre, &.{"self_attn.q_proj"});
        l.o = try dw.quantized(gpa, device, &check, &w.owned, geo, pre, &.{"self_attn.o_proj"});
        l.gate_up = try dw.quantized(gpa, device, &check, &w.owned, geo, pre, &.{ "mlp.gate_proj", "mlp.up_proj" });
        l.down = try dw.quantized(gpa, device, &check, &w.owned, geo, pre, &.{"mlp.down_proj"});
        const fields = .{ "q_norm", "input_norm", "post_attention", "pre_feedforward", "post_feedforward", "scalar" };
        const names = .{ "self_attn.q_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "layer_scalar" };
        inline for (fields, names) |field, t| {
            const full = try std.fmt.bufPrint(&name, "{s}{s}", .{ pre, t });
            @field(l, field) = try dw.vector(gpa, device, &check, &w.owned, full, c.expected(geo, full).?[0]);
        }
    }
    return w;
}

const fixture = @embedFile("fixtures/assistant_config.json");

fn edited(gpa: std.mem.Allocator, from: []const u8, to: []const u8) ![]u8 {
    const out = try std.mem.replaceOwned(u8, gpa, fixture, from, to);
    if (std.mem.eql(u8, out, fixture)) return error.FixtureUnchanged;
    return out;
}

test "the 12B's assistant is admitted against the 12B, with the target's key heads" {
    const c = try Config.parse(std.testing.allocator, fixture, cfg.unified12b);
    try std.testing.expectEqual(@as(u32, 4), c.layers);
    try std.testing.expect(c.sliding[0] and c.sliding[2] and !c.sliding[3]);
    try std.testing.expectEqual(@as(usize, 8), c.glue.sliding_kv_heads);
    try std.testing.expectEqual(@as(usize, 1), c.glue.global_kv_heads);
    try std.testing.expectEqual(@as(usize, 1024), c.glue.hidden);
    try std.testing.expectEqual(@as(usize, 512), c.headDim(3));
}

test "an assistant for another target or layout is refused by name" {
    const gpa = std.testing.allocator;
    try std.testing.expectError(error.AssistantForAnotherTarget, Config.parse(gpa, fixture, cfg.dense31b));
    const cases = .{
        .{ "\"use_ordered_embeddings\": false", "\"use_ordered_embeddings\": true", error.AssistantOrderedEmbeddingsNotServed },
        .{ "\"backbone_hidden_size\": 3840", "\"backbone_hidden_size\": 5376", error.AssistantForAnotherTarget },
        .{ "\"model_type\": \"gemma4_unified_assistant\"", "\"model_type\": \"gemma4_assistant\"", error.UnsupportedAssistantType },
        .{ "\"num_key_value_heads\": 8", "\"num_key_value_heads\": 16", error.AssistantForAnotherTarget },
        .{ "\"final_logit_softcapping\": null", "\"final_logit_softcapping\": 30.0", error.UnsupportedAssistantGeometry },
        .{ "\"hidden_size\": 1024", "\"hidden_size\": 2048", error.UnsupportedAssistantGeometry },
        .{ "\"num_kv_shared_layers\": 4", "\"num_kv_shared_layers\": 2", error.UnsupportedAssistantGeometry },
        .{ "\"rope_theta\": 10000.0", "\"rope_theta\": 1000.0", error.UnsupportedAssistantRotary },
        .{ "\"tie_word_embeddings\": true,\n  \"transformers", "\"quantization\": {\"bits\": 4, \"group_size\": 64},\n  \"transformers", error.UnsupportedAssistantQuantization },
    };
    inline for (cases) |k| {
        const bytes = try edited(gpa, k[0], k[1]);
        defer gpa.free(bytes);
        try std.testing.expectError(k[2], Config.parse(gpa, bytes, cfg.unified12b));
    }
}

test "a drafter directory is the assistant's by its model type" {
    const gpa = std.testing.allocator;
    try std.testing.expect(try typeIsAssistant(gpa, fixture));
    try std.testing.expect(try typeIsAssistant(gpa, "{\"model_type\":\"gemma4_assistant\"}"));
    try std.testing.expect(!try typeIsAssistant(gpa, "{\"architectures\":[\"DFlashDraftModel\"],\"model_type\":\"qwen3\"}"));
    try std.testing.expect(!try typeIsAssistant(gpa, "{\"architectures\":[\"DFlashDraftModel\"]}"));
}

test "the stock checkpoint's tensors are the shapes the assistant reads, and others are unknown" {
    const c = try Config.parse(std.testing.allocator, fixture, cfg.unified12b);
    const g = cfg.unified12b;
    const cases = .{
        .{ "model.embed_tokens.weight", [2]usize{ 262144, 1024 } },                   .{ "pre_projection.weight", [2]usize{ 1024, 7680 } },
        .{ "post_projection.weight", [2]usize{ 3840, 1024 } },                        .{ "model.norm.weight", [2]usize{ 1024, 1 } },
        .{ "model.layers.0.self_attn.q_proj.weight", [2]usize{ 4096, 1024 } },        .{ "model.layers.3.self_attn.q_proj.weight", [2]usize{ 8192, 1024 } },
        .{ "model.layers.0.self_attn.o_proj.weight", [2]usize{ 1024, 4096 } },        .{ "model.layers.3.self_attn.o_proj.weight", [2]usize{ 1024, 8192 } },
        .{ "model.layers.2.self_attn.q_norm.weight", [2]usize{ 256, 1 } },            .{ "model.layers.3.self_attn.q_norm.weight", [2]usize{ 512, 1 } },
        .{ "model.layers.1.mlp.gate_proj.weight", [2]usize{ 8192, 1024 } },           .{ "model.layers.1.mlp.down_proj.weight", [2]usize{ 1024, 8192 } },
        .{ "model.layers.3.post_feedforward_layernorm.weight", [2]usize{ 1024, 1 } }, .{ "model.layers.3.layer_scalar", [2]usize{ 1, 1 } },
    };
    inline for (cases) |k| try std.testing.expectEqual(@as(?[2]usize, k[1]), c.expected(g, k[0]));
    for ([_][]const u8{ "model.layers.0.self_attn.k_proj.weight", "model.layers.4.layer_scalar", "lm_head.weight", "model.token_ordering" }) |name|
        try std.testing.expectEqual(@as(?[2]usize, null), c.expected(g, name));
}

test "every assistant linear fits the reg lane layout at the target's splits" {
    const c = try Config.parse(std.testing.allocator, fixture, cfg.unified12b);
    const g = cfg.unified12b;
    var shapes: [3 + 4 * max_layers][2]usize = undefined;
    var count: usize = 0;
    for ([_][]const u8{ "pre_projection.weight", "post_projection.weight", "model.embed_tokens.weight" }) |name| {
        shapes[count] = c.expected(g, name).?;
        count += 1;
    }
    for (0..c.layers) |i| {
        const q = c.heads * c.headDim(i);
        for ([_][2]usize{ .{ q, c.hidden }, .{ c.hidden, q }, .{ 2 * c.intermediate, c.hidden }, .{ c.hidden, c.intermediate } }) |s| {
            shapes[count] = s;
            count += 1;
        }
    }
    for (shapes[0..count]) |s| try weights.layout(g, .{ .w = undefined, .sb = undefined, .n = s[0], .k = s[1] }).validate();
}
