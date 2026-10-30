//! plan.zig for CUDA builds, line for line (cuda_config.zig in place of config.zig): the tensors a geometry stores.
const std = @import("std");
const cfg = @import("cuda_config.zig");

const Width = cfg.widths.Width;

/// Where the language model's tensors sit in a converted checkpoint.
pub const prefix = "language_model.model.";

pub const Error = error{BadGemmaWeight};

/// A linear as stored: `n` rows of `k` inputs, MLX affine at `width`, or BF16 `.weight` [n, k] when null.
pub const Linear = struct {
    n: usize,
    k: usize,
    width: ?Width,

    /// Its tensors under `name`, each expected at its shape.
    pub fn check(l: Linear, ck: anytype, name: []const u8) !void {
        var buf: [192]u8 = undefined;
        const w = l.width orelse {
            _ = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .bf16, &.{ l.n, l.k });
            return;
        };
        if (l.k % w.group != 0 or l.k * w.bits % 32 != 0) return error.BadGemmaWeight;
        _ = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), .u32, &.{ l.n, l.k * w.bits / 32 });
        _ = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.scales", .{name}), .bf16, &.{ l.n, l.k / w.group });
        _ = try ck.expect(try std.fmt.bufPrint(&buf, "{s}.biases", .{name}), .bf16, &.{ l.n, l.k / w.group });
    }
};

/// A layer's per-layer input pair (E-series): the gate from the hidden state, and the projection back to it.
pub const LayerInput = struct { gate: Linear, project: Linear };

/// What a layer stores. A layer without `k` owns no cache and reads `owner`'s; it stores no k_proj, v_proj or k_norm.
pub const Layer = struct {
    q: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    k: ?Linear,
    /// Null on a layer with a cache when its values are its keys (a global layer of a checkpoint without v_proj there).
    v: ?Linear,
    head_dim: usize,
    layer_input: ?LayerInput,
    /// The layer whose cache this one reads (itself when it owns one).
    owner: usize,
};

/// What the model stores outside its layers; the per-layer input parts are null without them.
pub const Model = struct {
    embed: Linear,
    layer_table: ?Linear,
    model_project: ?Linear,
};

pub fn layer(c: cfg.Config, index: usize) Layer {
    const g = c.geo;
    const ws = &c.widths;
    const dh = cfg.headDim(index);
    const q_out = g.query_heads * dh;
    const kv_out = g.kvHeads(index) * dh;
    const owns = index < g.cacheLayers();
    const has_v = owns and (!cfg.global(index) or !g.keys_are_values);
    return .{
        .q = .{ .n = q_out, .k = g.hidden, .width = ws.of(index, .q) },
        .o = .{ .n = g.hidden, .k = q_out, .width = ws.of(index, .o) },
        .gate = .{ .n = g.intermediate, .k = g.hidden, .width = ws.of(index, .gate) },
        .up = .{ .n = g.intermediate, .k = g.hidden, .width = ws.of(index, .up) },
        .down = .{ .n = g.hidden, .k = g.intermediate, .width = ws.of(index, .down) },
        .k = if (owns) .{ .n = kv_out, .k = g.hidden, .width = ws.of(index, .k) } else null,
        .v = if (has_v) .{ .n = kv_out, .k = g.hidden, .width = ws.of(index, .v) } else null,
        .head_dim = dh,
        .layer_input = if (g.layer_input == 0) null else .{
            .gate = .{ .n = g.layer_input, .k = g.hidden, .width = ws.layer_gate[index] },
            .project = .{ .n = g.hidden, .k = g.layer_input, .width = ws.layer_project[index] },
        },
        .owner = cfg.cacheOwner(index, g.cacheLayers()).?,
    };
}

pub fn model(c: cfg.Config) Model {
    const g = c.geo;
    const stacked = g.layers * g.layer_input;
    return .{
        .embed = .{ .n = cfg.vocab, .k = g.hidden, .width = c.widths.embed },
        .layer_table = if (g.layer_input == 0) null else .{ .n = cfg.vocab, .k = stacked, .width = c.widths.layer_table },
        .model_project = if (g.layer_input == 0) null else .{ .n = stacked, .k = g.hidden, .width = c.widths.model_project },
    };
}

fn vector(ck: anytype, comptime fmt: []const u8, args: anytype, len: usize) !void {
    var buf: [192]u8 = undefined;
    _ = try ck.expect(try std.fmt.bufPrint(&buf, fmt, args), .bf16, &.{len});
}

/// Expects every tensor the geometry stores; skips the k and v a checkpoint carries for cache-reading layers.
pub fn check(c: cfg.Config, ck: anytype) !void {
    const g = c.geo;
    const m = model(c);
    var name: [192]u8 = undefined;
    try m.embed.check(ck, prefix ++ "embed_tokens");
    try vector(ck, prefix ++ "norm.weight", .{}, g.hidden);
    if (m.layer_table) |t| try t.check(ck, prefix ++ "embed_tokens_per_layer");
    if (m.model_project) |p| try p.check(ck, prefix ++ "per_layer_model_projection");
    if (g.layer_input != 0) try vector(ck, prefix ++ "per_layer_projection_norm.weight", .{}, g.layer_input);
    for (0..g.layers) |i| {
        const l = layer(c, i);
        const p = try std.fmt.bufPrint(&name, prefix ++ "layers.{d}.", .{i});
        try checkLayer(l, ck, p, g.hidden);
        if (l.k == null) for ([_][]const u8{ "self_attn.k_proj", "self_attn.v_proj", "self_attn.k_norm" }) |t| {
            var skipped: [192]u8 = undefined;
            try ck.skip(try std.fmt.bufPrint(&skipped, "{s}{s}", .{ p, t }));
        };
    }
}

fn checkLayer(l: Layer, ck: anytype, p: []const u8, hidden: usize) !void {
    var buf: [192]u8 = undefined;
    inline for (.{ .{ "self_attn.q_proj", "q" }, .{ "self_attn.o_proj", "o" }, .{ "mlp.gate_proj", "gate" }, .{ "mlp.up_proj", "up" }, .{ "mlp.down_proj", "down" } }) |t|
        try @field(l, t[1]).check(ck, try std.fmt.bufPrint(&buf, "{s}{s}", .{ p, t[0] }));
    if (l.k) |k| try k.check(ck, try std.fmt.bufPrint(&buf, "{s}self_attn.k_proj", .{p}));
    if (l.v) |v| try v.check(ck, try std.fmt.bufPrint(&buf, "{s}self_attn.v_proj", .{p}));
    try vector(ck, "{s}self_attn.q_norm.weight", .{p}, l.head_dim);
    if (l.k != null) try vector(ck, "{s}self_attn.k_norm.weight", .{p}, l.head_dim);
    inline for (.{ "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm" }) |t|
        try vector(ck, "{s}" ++ t ++ ".weight", .{p}, hidden);
    try vector(ck, "{s}layer_scalar", .{p}, 1);
    if (l.layer_input) |li| {
        try li.gate.check(ck, try std.fmt.bufPrint(&buf, "{s}per_layer_input_gate", .{p}));
        try li.project.check(ck, try std.fmt.bufPrint(&buf, "{s}per_layer_projection", .{p}));
        try vector(ck, "{s}post_per_layer_input_norm.weight", .{p}, hidden);
    }
}

const DType = enum { u32, bf16 };

/// A checkpoint as `check` sees it: tensors by name, the names asked for and the prefixes skipped.
const Fake = struct {
    arena: std.heap.ArenaAllocator = .init(std.testing.allocator),
    tensors: std.StringHashMapUnmanaged(struct { DType, [2]usize, usize }) = .empty,
    asked: std.ArrayList([]const u8) = .empty,
    skipped: std.ArrayList([]const u8) = .empty,

    fn deinit(f: *Fake) void {
        f.arena.deinit();
    }

    fn put(f: *Fake, comptime fmt: []const u8, args: anytype, dtype: DType, shape: []const usize) !void {
        const a = f.arena.allocator();
        var s: [2]usize = .{ 0, 0 };
        @memcpy(s[0..shape.len], shape);
        try f.tensors.put(a, try std.fmt.allocPrint(a, fmt, args), .{ dtype, s, shape.len });
    }

    /// A linear stored as `bits` MLX affine groups of 64, or BF16 when `bits` is 0.
    fn linear(f: *Fake, comptime fmt: []const u8, args: anytype, bits: usize, n: usize, k: usize) !void {
        if (bits == 0) return f.put(fmt ++ ".weight", args, .bf16, &.{ n, k });
        try f.put(fmt ++ ".weight", args, .u32, &.{ n, k * bits / 32 });
        try f.put(fmt ++ ".scales", args, .bf16, &.{ n, k / 64 });
        try f.put(fmt ++ ".biases", args, .bf16, &.{ n, k / 64 });
    }

    pub fn expect(f: *Fake, name: []const u8, dtype: DType, shape: []const usize) !void {
        try f.asked.append(f.arena.allocator(), try f.arena.allocator().dupe(u8, name));
        const t = f.tensors.get(name) orelse return error.MissingTensor;
        if (t[0] != dtype or !std.mem.eql(usize, t[1][0..t[2]], shape)) return error.UnexpectedTensor;
    }

    pub fn skip(f: *Fake, name: []const u8) !void {
        try f.skipped.append(f.arena.allocator(), try f.arena.allocator().dupe(u8, name));
    }

    fn asks(f: *const Fake, needle: []const u8) bool {
        for (f.asked.items) |n| if (std.mem.indexOf(u8, n, needle) != null) return true;
        return false;
    }
};

/// Twelve layers (globals 5 and 11) of which the last four read a cache, written out by hand from the HF names.
const tiny: cfg.Geometry = .{ .name = "tiny", .hidden = 64, .layers = 12, .intermediate = 128, .query_heads = 2, .sliding_kv_heads = 1, .global_kv_heads = 1, .layer_input = 64, .shared_layers = 4, .keys_are_values = false };
const none = std.math.maxInt(usize);

/// The tiny checkpoint; layer `bf16_gate` stores its per_layer_input_gate in BF16, layer `q8` its q_proj at 8 bits.
fn tinyCheckpoint(f: *Fake, bf16_gate: usize, q8: usize) !void {
    const pre = prefix;
    try f.linear(pre ++ "embed_tokens", .{}, 4, 262144, 64);
    try f.put(pre ++ "norm.weight", .{}, .bf16, &.{64});
    try f.linear(pre ++ "embed_tokens_per_layer", .{}, 4, 262144, 12 * 64);
    try f.linear(pre ++ "per_layer_model_projection", .{}, 0, 12 * 64, 64);
    try f.put(pre ++ "per_layer_projection_norm.weight", .{}, .bf16, &.{64});
    for (0..12) |i| {
        const global = i == 5 or i == 11;
        const dh: usize = if (global) 512 else 256;
        const own = i < 8;
        const l = .{i};
        try f.linear(pre ++ "layers.{d}.self_attn.q_proj", l, if (i == q8) 8 else 4, 2 * dh, 64);
        try f.linear(pre ++ "layers.{d}.self_attn.o_proj", l, 4, 64, 2 * dh);
        try f.linear(pre ++ "layers.{d}.mlp.gate_proj", l, 4, 128, 64);
        try f.linear(pre ++ "layers.{d}.mlp.up_proj", l, 4, 128, 64);
        try f.linear(pre ++ "layers.{d}.mlp.down_proj", l, 4, 64, 128);
        try f.put(pre ++ "layers.{d}.self_attn.q_norm.weight", l, .bf16, &.{dh});
        if (own) {
            try f.linear(pre ++ "layers.{d}.self_attn.k_proj", l, 4, dh, 64);
            try f.linear(pre ++ "layers.{d}.self_attn.v_proj", l, 4, dh, 64);
            try f.put(pre ++ "layers.{d}.self_attn.k_norm.weight", l, .bf16, &.{dh});
        } else { // carried by the checkpoint, of a shape nothing reads
            try f.put(pre ++ "layers.{d}.self_attn.k_proj.weight", l, .u32, &.{ 1, 1 });
            try f.put(pre ++ "layers.{d}.self_attn.v_proj.weight", l, .u32, &.{ 1, 1 });
            try f.put(pre ++ "layers.{d}.self_attn.k_norm.weight", l, .bf16, &.{1});
        }
        inline for (.{ "input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm" }) |n|
            try f.put(pre ++ "layers.{d}." ++ n ++ ".weight", l, .bf16, &.{64});
        try f.put(pre ++ "layers.{d}.layer_scalar", l, .bf16, &.{1});
        try f.linear(pre ++ "layers.{d}.per_layer_input_gate", l, if (i == bf16_gate) 0 else 4, 64, 64);
        try f.linear(pre ++ "layers.{d}.per_layer_projection", l, 4, 64, 64);
    }
}

fn tinyConfig() cfg.Config {
    var c: cfg.Config = .{ .geo = tiny, .context = 1 };
    c.widths.model_project = null; // stored BF16
    return c;
}

test "a checkpoint with per-layer inputs, separate global values and shared layers passes, and shared k and v are not read" {
    var f: Fake = .{};
    defer f.deinit();
    try tinyCheckpoint(&f, none, none);
    try check(tinyConfig(), &f);
    for (8..12) |i| {
        var name: [64]u8 = undefined;
        for ([_][]const u8{ "self_attn.k_proj", "self_attn.v_proj", "self_attn.k_norm" }) |t| {
            const want = try std.fmt.bufPrint(&name, prefix ++ "layers.{d}.{s}", .{ i, t });
            try std.testing.expect(!f.asks(want));
            var skipped = false;
            for (f.skipped.items) |s| skipped = skipped or std.mem.eql(u8, s, want);
            try std.testing.expect(skipped);
        }
    }
    try std.testing.expect(f.asks("layers.7.self_attn.k_proj.weight") and f.asks("layers.5.self_attn.v_proj.weight"));
    try std.testing.expect(f.asks("layers.11.per_layer_projection.weight") and f.asks("layers.0.post_per_layer_input_norm.weight"));
    try std.testing.expect(f.asks("embed_tokens_per_layer.weight") and f.asks("per_layer_model_projection.weight"));
}

test "a BF16 per-layer gate and an 8-bit attention projection change only the widths read" {
    var f: Fake = .{};
    defer f.deinit();
    try tinyCheckpoint(&f, 2, 3);
    var c = tinyConfig();
    c.widths.layer_gate[2] = null;
    c.widths.layers[3][@backingInt(cfg.widths.Module.q)] = .{ .bits = 8, .group = 64 };
    try check(c, &f);
    try std.testing.expect(f.asks("layers.2.per_layer_input_gate.weight") and !f.asks("layers.2.per_layer_input_gate.scales"));
    try std.testing.expect(f.asks("layers.3.self_attn.q_proj.scales") and f.asks("layers.3.per_layer_input_gate.scales"));
    var g: Fake = .{};
    defer g.deinit();
    try tinyCheckpoint(&g, none, none); // the same checkpoint at the default widths: the planned widths are not guessed
    try std.testing.expectError(error.UnexpectedTensor, check(c, &g));
}

test "a missing tensor or a wrong shape is refused by name" {
    const cases = .{
        .{ prefix ++ "layers.5.self_attn.v_proj.weight", error.MissingTensor },
        .{ prefix ++ "layers.11.post_per_layer_input_norm.weight", error.MissingTensor },
        .{ prefix ++ "embed_tokens_per_layer.scales", error.MissingTensor },
        .{ prefix ++ "per_layer_projection_norm.weight", error.MissingTensor },
        .{ prefix ++ "layers.7.self_attn.k_norm.weight", error.MissingTensor },
    };
    inline for (cases) |c| {
        var f: Fake = .{};
        defer f.deinit();
        try tinyCheckpoint(&f, none, none);
        try std.testing.expect(f.tensors.remove(c[0]));
        try std.testing.expectError(c[1], check(tinyConfig(), &f));
    }
    var f: Fake = .{};
    defer f.deinit();
    try tinyCheckpoint(&f, none, none);
    try f.put(prefix ++ "layers.4.per_layer_projection.weight", .{}, .u32, &.{ 64, 16 }); // words are 8 wide
    try std.testing.expectError(error.UnexpectedTensor, check(tinyConfig(), &f));
    try tinyCheckpoint(&f, none, none);
    try f.put(prefix ++ "layers.5.self_attn.v_proj.weight", .{}, .u32, &.{ 256, 8 }); // global values are 512 wide
    try std.testing.expectError(error.UnexpectedTensor, check(tinyConfig(), &f));
    try tinyCheckpoint(&f, none, none);
    try f.put(prefix ++ "embed_tokens_per_layer.weight", .{}, .u32, &.{ 262144, 8 * 64 }); // 11 layers' worth
    try std.testing.expectError(error.UnexpectedTensor, check(tinyConfig(), &f));
}

test "the E4B plan: per-layer shapes, which layers own K and V, and whose cache the others read" {
    const c: cfg.Config = .{ .geo = cfg.e4b, .context = 1 };
    const m = model(c);
    try std.testing.expectEqual(Linear{ .n = 262144, .k = 10752, .width = cfg.widths.fast }, m.layer_table.?);
    try std.testing.expectEqual(Linear{ .n = 10752, .k = 2560, .width = cfg.widths.fast }, m.model_project.?);
    var owners: usize = 0;
    for (0..42) |i| {
        const l = layer(c, i);
        try std.testing.expectEqual(Linear{ .n = 256, .k = 2560, .width = cfg.widths.fast }, l.layer_input.?.gate);
        try std.testing.expectEqual(Linear{ .n = 2560, .k = 256, .width = cfg.widths.fast }, l.layer_input.?.project);
        try std.testing.expectEqual(i < 24, l.k != null);
        try std.testing.expectEqual(i < 24, l.v != null); // global layers have their own v_proj here
        if (l.k) |k| try std.testing.expectEqual(c.geo.kvHeads(i) * l.head_dim, k.n);
        if (l.k != null) owners += 1;
        try std.testing.expectEqual(if (i < 24) i else @as(usize, if (cfg.global(i)) 23 else 22), l.owner);
    }
    try std.testing.expectEqual(c.geo.cacheLayers(), owners);
    try std.testing.expectEqual(@as(usize, 1024), layer(c, 5).v.?.n);
    try std.testing.expectEqual(@as(usize, 4096), layer(c, 5).q.n);
}

test "the dense geometries plan no per-layer input and no v_proj on global layers" {
    for ([_]cfg.Geometry{ cfg.dense31b, cfg.unified12b }) |g| {
        const c: cfg.Config = .{ .geo = g, .context = 1 };
        try std.testing.expect(model(c).layer_table == null and model(c).model_project == null);
        for (0..g.layers) |i| {
            const l = layer(c, i);
            try std.testing.expect(l.layer_input == null and l.k != null and l.owner == i);
            try std.testing.expectEqual(!cfg.global(i), l.v != null);
        }
    }
}
