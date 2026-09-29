const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const ops = @import("gemma_ops.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;

const Cache = struct {
    keys: A = mx.empty,
    values: A = mx.empty,
    fn deinit(c: *Cache) void {
        mx.free(c.keys);
        mx.free(c.values);
        c.* = .{};
    }
};
pub const Pass = struct {
    scope: mx.Scope = .{},
    logits: A = mx.empty,
    hidden: A = mx.empty,
    taps: A = mx.empty,
    records: [30]Cache = @splat(.{}),
    position: i32,
    generation: u64,
    rows: usize,
    pub fn deinit(p: *Pass) void {
        p.scope.deinit();
    }
};
pub const Model = struct {
    weights: cp.Store,
    kernels: mx.Kernels,
    activations: @import("prefill_ops.zig").Ops = .{},
    cache: [30]Cache = @splat(.{}),
    position: i32 = 0,
    generation: u64 = 0,
    draft: ?@import("dflash.zig").Draft = null,
    has_mtp: bool = false,
    pub const vocab = 262144;
    pub fn eos(id: i32) bool {
        return id == 1 or id == 106 or id == 50;
    }
    pub fn init(io: std.Io, dir: []const u8) !Model {
        var m = Model{ .weights = cp.Store.init(64), .kernels = mx.Kernels.init() };
        errdefer m.deinit();
        var buf: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer parsed.deinit();
        try config(parsed.value);
        try @import("schema.zig").checkCheckpoint(.gemma, io, dir);
        try m.weights.load(io, dir, "language_model.");
        try @import("schema.zig").validateConfig(.gemma, &m.weights.arrays, false, parsed.value);
        var entries = m.weights.arrays.iterator();
        while (entries.next()) |entry| {
            if (mx.dtype(entry.value_ptr.*) != mx.c.MLX_UINT32) continue;
            const spec = (try @import("quantization.zig").resolve(parsed.value, entry.key_ptr.*)) orelse return error.UnsupportedQuantization;
            const bits: i32 = if (std.mem.endsWith(u8, entry.key_ptr.*, ".router.proj.weight")) 8 else 4;
            if (spec.bits != bits) return error.UnsupportedQuantization;
        }
        m.weights.group = (try @import("quantization.zig").resolve(parsed.value, "model.embed_tokens")).?.group_size;
        var s = mx.Scope{};
        defer s.deinit();
        var local: [128]f32 = undefined;
        for (&local, 0..) |*v, i| v.* = @floatCast(@exp2(-@as(f64, @floatFromInt(i)) / 128 * @log2(@as(f64, 10000))));
        try m.weights.put("inv_local", try s.data(&local, &.{128}, mx.f32t));
        var exponents: [64]f32 = undefined;
        for (&exponents, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 2)) / 512;
        const powers = try s.binary(mx.c.mlx_power, try s.scalar(1000000), try s.data(&exponents, &.{64}, mx.f32t));
        const inverse = try s.binary(mx.c.mlx_divide, try s.scalar(1), powers);
        try m.weights.put("inv_global", try s.cat(&.{ inverse, try s.zeros(&.{192}, mx.f32t) }, 0));
        try m.weights.put("freq_global", try s.cat(&.{ powers, try s.binary(mx.c.mlx_add, try s.zeros(&.{192}, mx.f32t), try s.scalar(std.math.inf(f32))) }, 0));
        try m.weights.put("eps", try s.scalar(1e-6));
        for (0..30) |i| {
            try m.stack(&s, i, "qkv", if (sliding(i)) &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" } else &.{ "self_attn.q_proj", "self_attn.k_proj" });
            try m.stack(&s, i, "gate_up", &.{ "mlp.gate_proj", "mlp.up_proj" });
            const root = try s.cast(try s.scalar(@floatCast(1.0 / @sqrt(@as(f64, 2816)))), mx.bf16);
            const scale = try s.binary(mx.c.mlx_multiply, try m.weight(i, "router.scale"), root);
            try m.weights.put(try std.fmt.bufPrint(&buf, "model.layers.{d}.router_norm", .{i}), scale);
        }
        return m;
    }
    pub fn deinit(m: *Model) void {
        m.reset();
        if (m.draft) |*d| d.deinit();
        m.activations.deinit();
        m.kernels.deinit();
        m.weights.deinit();
    }
    pub fn reset(m: *Model) void {
        for (&m.cache) |*cache| cache.deinit();
        if (m.draft) |*d| d.reset();
        m.position = 0;
        m.generation +%= 1;
    }
    pub fn loadDraft(m: *Model, io: std.Io, dir: []const u8) !void {
        return m.loadDraftBits(io, dir, 8);
    }
    pub fn loadDraftBits(m: *Model, io: std.Io, dir: []const u8, bits: i32) !void {
        if (m.position != 0) return error.DraftRequiresEmptyCache;
        const d = try @import("dflash.zig").Draft.init(io, dir, 2816, vocab, 30, bits);
        if (m.draft) |*old| old.deinit();
        m.draft = d;
        m.has_mtp = true;
    }
    pub fn draftAbsorbsOnCommit(_: *Model) bool {
        return true;
    }
    pub fn maxDrafts(m: *Model) usize {
        return if (m.draft) |d| @intCast(d.parsed.value.dflash_config.block_size - 1) else 0;
    }
    pub fn propose(m: *Model, _: A, anchor: i32, output: []i32, _: @import("sampling.zig").Sampling) !void {
        output[0] = anchor;
        if (output.len <= 1) return;
        const d = if (m.draft) |*value| value else return error.MissingDraft;
        if (d.position != m.position or output.len - 1 > m.maxDrafts()) return error.InvalidDraftBlock;
        var s = mx.Scope{};
        defer s.deinit();
        const cfg = d.parsed.value.dflash_config;
        var ids: [128]i32 = @splat(cfg.mask_token_id);
        ids[0] = anchor;
        const embeddings = try s.reshape(try s.binary(mx.c.mlx_multiply, try m.weights.embed(&s, "model.embed_tokens", ids[0..output.len]), try s.cast(try s.scalar(@floatCast(@sqrt(@as(f64, 2816)) * cfg.input_embedding_scale)), mx.bf16)), &.{ 1, @intCast(output.len), 2816 });
        const hidden = try d.forward(&s, embeddings);
        var logits = try s.binary(mx.c.mlx_multiply, try m.weights.linear(&m.kernels, &s, "model.embed_tokens", hidden, false), try s.cast(try s.scalar(cfg.output_multiplier), mx.bf16));
        if (cfg.final_logit_softcapping orelse d.parsed.value.final_logit_softcapping) |cap| if (cap > 0) {
            logits = try @import("prefill_ops.zig").uncompiled(&s, .softcap, &.{ logits, try s.scalar(cap) });
        };
        var selected = mx.c.mlx_array_new();
        const rc = mx.c.mlx_argmax_axis(&selected, logits, -1, false, mx.stream);
        selected = try s.cast(try s.result(rc, selected), mx.c.MLX_INT32);
        try mx.eval(selected);
        @memcpy(output[1..], mx.c.mlx_array_data_int32(selected)[0 .. output.len - 1]);
    }
    fn sliding(i: usize) bool {
        return i % 6 != 5;
    }
    pub fn weight(m: *Model, i: usize, suffix: []const u8) !A {
        var buf: [256]u8 = undefined;
        return m.weights.get(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ i, suffix }));
    }
    pub fn triple(m: *Model, i: usize, suffix: []const u8) ![3]A {
        var buf: [256]u8 = undefined;
        return m.weights.triple(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ i, suffix }));
    }
    fn group(m: *Model, i: usize, suffix: []const u8, bits: i32) !i32 {
        const t = try m.triple(i, suffix);
        return @divExact(mx.dim(t[0], -1) * @divExact(32, bits), mx.dim(t[1], -1));
    }
    fn stack(m: *Model, s: *mx.Scope, i: usize, name: []const u8, members: []const []const u8) !void {
        for ([_][]const u8{ "weight", "scales", "biases" }) |suffix| {
            var arrays: [3]A = undefined;
            var buf: [256]u8 = undefined;
            for (members, 0..) |member, j| arrays[j] = try m.weight(i, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ member, suffix }));
            const value = try s.cat(arrays[0..members.len], 0);
            try mx.eval(value);
            try m.weights.put(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.{s}", .{ i, name, suffix }), value);
        }
    }
    pub fn project(m: *Model, s: *mx.Scope, x: A, weights: [3]A) !A {
        const rows = mx.dim(x, 0);
        const n = mx.dim(weights[0], 0);
        const width = mx.dim(x, 1);
        if (rows < 1 or rows > 16 or @mod(n, 8) != 0 or @mod(width, 64) != 0) return error.InvalidGemmaProjection;
        const gs = @divExact(width, mx.dim(weights[1], -1));
        return (try m.kernels.run(s, src.nemotron_rows_qmv, &.{ x, weights[0], weights[1], weights[2] }, &.{ ti("K", width), ti("N", n), ti("GS", gs), ti("RPS", 4) }, .{ 32 * rows, @divExact(n, 4), 1 }, .{ 32 * rows, if (rows <= 8) 2 else 1, 1 }, &.{.{ .shape = &.{ rows, n } }}))[0];
    }
    pub fn forward(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len == 0 or tokens.len > 16 or m.position > 262144 - tokens.len) return error.ContextLimitExceeded;
        for (tokens) |token| if (token < 0 or token >= vocab) return error.InvalidToken;
        var p = Pass{ .position = m.position, .generation = m.generation, .rows = tokens.len };
        errdefer p.deinit();
        const s = &p.scope;
        const rows: i32 = @intCast(tokens.len);
        var positions: [16]i32 = undefined;
        for (0..tokens.len) |i| positions[i] = m.position + @as(i32, @intCast(i));
        const at = try ops.paddedInts(s, positions[0..tokens.len]);
        const eps = try m.weights.get("eps");
        var h = try s.binary(mx.c.mlx_multiply, try m.weights.embed(s, "model.embed_tokens", tokens), try s.cast(try s.scalar(@floatCast(@sqrt(@as(f64, 2816)))), mx.bf16));
        var normed = try s.rms(h, try m.weight(0, "input_layernorm.weight"));
        var taps: [32]A = undefined;
        var tap_count: usize = 0;
        for (0..30) |i| {
            const local = sliding(i);
            const g = ops.Geometry{ .heads = 16, .kv_heads = if (local) 8 else 2, .head_dim = if (local) 256 else 512, .values_are_keys = !local };
            const qkv = try ops.qkv(&m.kernels, s, g, normed, try m.triple(i, "qkv"), try m.weight(i, "self_attn.q_norm.weight"), try m.weight(i, "self_attn.k_norm.weight"), try m.weights.get(if (local) "inv_local" else "inv_global"), at, eps, try m.group(i, "qkv", 4));
            const keys = if (m.cache[i].keys.ctx != null) m.cache[i].keys else try s.zeros(&.{ 1, g.kv_heads, if (local) 1152 else 1, g.head_dim }, mx.bf16);
            const values = if (m.cache[i].values.ctx != null) m.cache[i].values else try s.zeros(mx.shape(keys), mx.bf16);
            p.records[i] = .{ .keys = try s.reshape(qkv[1], &.{ 1, g.kv_heads, rows, g.head_dim }), .values = try s.reshape(qkv[2], &.{ 1, g.kv_heads, rows, g.head_dim }) };
            const attended = try ops.attention(&m.kernels, s, qkv[0], keys, values, qkv[1], qkv[2], positions[0..tokens.len], if (local) 1024 else 0, if (local) 1152 else 0, 1);
            const out = try m.project(s, try s.reshape(attended, &.{ rows, 16 * g.head_dim }), try m.triple(i, "self_attn.o_proj"));
            const tail = try m.kernels.run(s, src.gemma_attn_tail, &.{ h, out, try m.weight(i, "post_attention_layernorm.weight"), try m.weight(i, "pre_feedforward_layernorm.weight"), try m.weight(i, "pre_feedforward_layernorm_2.weight"), try m.weight(i, "router_norm"), eps }, &.{ ti("D", 2816), ti("T", 256) }, .{ 256 * rows, 1, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ rows, 2816 } }, .{ .shape = &.{ rows, 2816 } }, .{ .shape = &.{ rows, 2816 } }, .{ .shape = &.{ rows, 2816 } } });
            const gu = try m.project(s, tail[1], try m.triple(i, "gate_up"));
            const activated = try m.activations.call(s, .geglu, &.{ try s.slice(gu, 1, 0, 2112), try s.slice(gu, 1, 2112, 4224) });
            const dense = try m.project(s, activated, try m.triple(i, "mlp.down_proj"));
            const routes = try ops.route(&m.kernels, s, try ops.router(&m.kernels, s, tail[3], try m.triple(i, "router.proj"), try m.group(i, "router.proj", 8)), try m.weight(i, "router.per_expert_scale"), 8);
            const expert_act = try ops.gateUp(&m.kernels, s, tail[2], routes[0], 8, try m.triple(i, "experts.switch_glu.gate_proj"), try m.triple(i, "experts.switch_glu.up_proj"), try m.group(i, "experts.switch_glu.gate_proj", 4));
            const expert = try ops.down(&m.kernels, s, expert_act, routes[0], routes[1], 8, try m.triple(i, "experts.switch_glu.down_proj"), try m.group(i, "experts.switch_glu.down_proj", 4));
            const next_weight = if (i < 29) try m.weight(i + 1, "input_layernorm.weight") else try m.weights.get("model.norm.weight");
            const end = try m.kernels.run(s, src.gemma_moe_tail, &.{ tail[0], dense, expert, try m.weight(i, "post_feedforward_layernorm_1.weight"), try m.weight(i, "post_feedforward_layernorm_2.weight"), try m.weight(i, "post_feedforward_layernorm.weight"), try m.weight(i, "layer_scalar"), next_weight, eps }, &.{ ti("D", 2816), ti("T", 256) }, .{ 256 * rows, 1, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ rows, 2816 } }, .{ .shape = &.{ rows, 2816 } } });
            h = end[0];
            normed = end[1];
            if (m.draft) |d| for (d.parsed.value.dflash_config.target_layer_ids) |id| if (id == i) {
                taps[tap_count] = h;
                tap_count += 1;
            };
            if ((i + 1) % 8 == 0) try mx.evalMany(&.{normed}, true);
        }
        p.hidden = normed;
        if (tap_count > 0) p.taps = try s.cat(taps[0..tap_count], -1);
        p.logits = try m.activations.call(s, .softcap, &.{ try m.project(s, normed, try m.weights.triple("model.embed_tokens")), try s.scalar(30) });
        try mx.eval(p.logits);
        return p;
    }
    pub fn prefill(m: *Model, tokens: []const i32) !Pass {
        return @import("gemma_prefill.zig").forward(m, tokens);
    }
    pub fn commit(m: *Model, p: *Pass, keep: usize) !void {
        if (p.position != m.position or p.generation != m.generation or keep == 0 or keep > p.rows) return error.InvalidCommit;
        var s = mx.Scope{};
        defer s.deinit();
        var next: [30]Cache = @splat(.{});
        errdefer for (&next) |*cache| cache.deinit();
        for (p.records, m.cache, &next, 0..) |record, old, *target, i| {
            inline for (.{ "keys", "values" }) |field| {
                const rows = try s.slice(@field(record, field), 2, 0, @intCast(keep));
                const current = @field(old, field);
                const value = if (sliding(i)) blk: {
                    const buffer = if (current.ctx != null) current else try s.zeros(&.{ 1, 8, 1152, 256 }, mx.bf16);
                    const skipped: i32 = @intCast(keep - @min(keep, 1152));
                    const retained = try s.slice(rows, 2, skipped, @intCast(keep));
                    const count = mx.dim(retained, 2);
                    const slot = @mod(m.position + skipped, 1152);
                    const first = @min(count, 1152 - slot);
                    const front = try ringPut(&s, buffer, try s.slice(retained, 2, 0, first), slot);
                    break :blk if (first < count) try ringPut(&s, front, try s.slice(retained, 2, first, count), 0) else front;
                } else if (current.ctx == null) rows else try s.cat(&.{ current, rows }, 2);
                @field(target, field) = try mx.retain(value);
            }
        }
        for (next) |cache| try mx.evalMany(&.{ cache.keys, cache.values }, false);
        if (m.draft) |*d| {
            if (d.position != m.position or p.taps.ctx == null) return error.InvalidDraftContext;
            try d.absorb(try s.slice(p.taps, 0, 0, @intCast(keep)));
        }
        for (&m.cache) |*cache| cache.deinit();
        m.cache = next;
        m.position += @intCast(keep);
        m.generation +%= 1;
    }
    pub fn checkExact(m: *Model, prefix_count: usize) !void {
        defer m.reset();
        {
            m.reset();
            var stale = try m.forward(&.{1});
            defer stale.deinit();
            m.reset();
            try std.testing.expectError(error.InvalidCommit, m.commit(&stale, 1));
        }
        var serial: [4]A = undefined;
        var saved: [30]Cache = @splat(.{});
        defer for (&saved) |*cache| cache.deinit();
        var scope = mx.Scope{};
        defer scope.deinit();
        for (0..2) |run| {
            m.reset();
            var offset: usize = 0;
            while (offset < prefix_count) {
                var tokens: [16]i32 = undefined;
                const count = @min(16, prefix_count - offset);
                for (tokens[0..count], 0..) |*token, j| token.* = @intCast(1000 + (offset + j) % 2000);
                var p = try m.forward(tokens[0..count]);
                defer p.deinit();
                try m.commit(&p, count);
                offset += count;
            }
            if (run == 0) {
                for ([_]i32{ 23, 41, 59, 83 }, 0..) |token, j| {
                    var p = try m.forward(&.{token});
                    defer p.deinit();
                    serial[j] = try scope.own(try mx.retain(p.logits));
                    try m.commit(&p, 1);
                }
                for (m.cache, &saved) |cache, *copy| {
                    copy.keys = try mx.retain(cache.keys);
                    copy.values = try mx.retain(cache.values);
                }
            } else {
                var p = try m.forward(&.{ 23, 41, 59, 83, 97, 101 });
                defer p.deinit();
                for (serial, 0..) |expected, j| try equal(&scope, expected, try scope.slice(p.logits, 0, @intCast(j), @intCast(j + 1)));
                try m.commit(&p, 3);
                try std.testing.expectError(error.InvalidCommit, m.commit(&p, 1));
                var next = try m.forward(&.{83});
                defer next.deinit();
                try equal(&scope, serial[3], next.logits);
                try m.commit(&next, 1);
                for (m.cache, saved) |actual, expected| {
                    try equal(&scope, actual.keys, expected.keys);
                    try equal(&scope, actual.values, expected.values);
                }
            }
        }
        std.debug.print("PASS: Gemma serial/chain logits, partial commit and all 60 cache arrays at prefix {d}.\n", .{prefix_count});
    }
};
fn ringPut(s: *mx.Scope, buffer: A, rows: A, offset: i32) !A {
    var out = mx.c.mlx_array_new();
    const axis: i32 = 2;
    const rc = mx.c.mlx_slice_update_dynamic(&out, buffer, rows, try s.ints(&.{offset}), &axis, 1, mx.stream);
    return s.result(rc, out);
}
fn config(root: std.json.Value) !void {
    if (root != .object) return error.InvalidModelConfig;
    const text = root.object.get("text_config") orelse return error.UnsupportedModel;
    if (text != .object) return error.InvalidModelConfig;
    inline for (.{ .{ "rms_norm_eps", 1e-6 }, .{ "final_logit_softcapping", 30.0 } }) |field| try number(text, field[0], field[1]);
    try string(text, "dtype", "bfloat16");
    try string(text, "hidden_activation", "gelu_pytorch_tanh");
    for ([_][]const u8{ "attention_bias", "use_double_wide_mlp" }) |field| {
        const value = text.object.get(field) orelse return error.InvalidModelConfig;
        if (value != .bool or value.bool) return error.UnsupportedModelGeometry;
    }
    const rope = text.object.get("rope_parameters") orelse return error.InvalidModelConfig;
    if (rope != .object) return error.InvalidModelConfig;
    const global = rope.object.get("full_attention") orelse return error.InvalidModelConfig;
    const local = rope.object.get("sliding_attention") orelse return error.InvalidModelConfig;
    try string(global, "rope_type", "proportional");
    try number(global, "rope_theta", 1000000);
    try number(global, "partial_rotary_factor", 0.25);
    try string(local, "rope_type", "default");
    try number(local, "rope_theta", 10000);
    inline for (.{ .{ "hidden_size", 2816 }, .{ "num_hidden_layers", 30 }, .{ "vocab_size", 262144 }, .{ "num_attention_heads", 16 }, .{ "num_key_value_heads", 8 }, .{ "num_global_key_value_heads", 2 }, .{ "head_dim", 256 }, .{ "global_head_dim", 512 }, .{ "intermediate_size", 2112 }, .{ "moe_intermediate_size", 704 }, .{ "num_experts", 128 }, .{ "top_k_experts", 8 }, .{ "sliding_window", 1024 }, .{ "num_kv_shared_layers", 0 }, .{ "hidden_size_per_layer_input", 0 } }) |field| {
        const value = text.object.get(field[0]) orelse return error.InvalidModelConfig;
        if (value != .integer or value.integer != field[1]) return error.UnsupportedModelGeometry;
    }
    for ([_][]const u8{ "tie_word_embeddings", "enable_moe_block", "attention_k_eq_v" }) |field| {
        const value = text.object.get(field) orelse return error.InvalidModelConfig;
        if (value != .bool or !value.bool) return error.UnsupportedModelGeometry;
    }
    const layers = text.object.get("layer_types") orelse return error.InvalidModelConfig;
    if (layers != .array or layers.array.items.len != 30) return error.UnsupportedModelGeometry;
    for (layers.array.items, 0..) |layer, i| if (layer != .string or !std.mem.eql(u8, layer.string, if (Model.sliding(i)) "sliding_attention" else "full_attention")) return error.UnsupportedModelGeometry;
}
fn number(object: std.json.Value, key: []const u8, expected: f64) !void {
    if (object != .object) return error.InvalidModelConfig;
    const value = object.object.get(key) orelse return error.InvalidModelConfig;
    const actual: f64 = switch (value) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => return error.InvalidModelConfig,
    };
    if (actual != expected) return error.UnsupportedModelGeometry;
}
fn string(object: std.json.Value, key: []const u8, expected: []const u8) !void {
    if (object != .object) return error.InvalidModelConfig;
    const value = object.object.get(key) orelse return error.InvalidModelConfig;
    if (value != .string or !std.mem.eql(u8, value.string, expected)) return error.UnsupportedModelGeometry;
}

pub fn checkModel(io: std.Io, dir: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var model = try Model.init(io, dir);
    defer model.deinit();
    var p = try model.forward(&.{ 1, 2, 3, 4 });
    defer p.deinit();
    const logits = try p.scope.cast(p.logits, mx.f32t);
    const path = try mx.allocator.dupeSentinel(u8, output, 0);
    defer mx.allocator.free(path);
    try mx.check(mx.c.mlx_save(path, logits));
    try model.commit(&p, 2);
    var continuation = try model.forward(&.{ 3, 4 });
    defer continuation.deinit();
    try equal(&continuation.scope, try continuation.scope.slice(p.logits, 0, 2, 4), continuation.logits);
    for (p.records, continuation.records) |full, partial| {
        try equal(&continuation.scope, try continuation.scope.slice(full.keys, 2, 2, 4), partial.keys);
        try equal(&continuation.scope, try continuation.scope.slice(full.values, 2, 2, 4), partial.values);
    }
    std.debug.print("PASS: Gemma partial commit, continuation and every layer's keys/values match the complete chain.\n", .{});
}
fn equal(s: *mx.Scope, a: A, b: A) !void {
    const x = try s.cast(try s.contiguous(a), mx.f32t);
    const y = try s.cast(try s.contiguous(b), mx.f32t);
    try mx.evalMany(&.{ x, y }, false);
    const count = mx.c.mlx_array_size(x);
    if (count != mx.c.mlx_array_size(y) or !std.mem.eql(u8, std.mem.sliceAsBytes(mx.c.mlx_array_data_float32(x)[0..count]), std.mem.sliceAsBytes(mx.c.mlx_array_data_float32(y)[0..count]))) return error.GemmaExactnessMismatch;
}

pub fn checkDraft(io: std.Io, dir: []const u8, drafter: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try Model.init(io, dir);
    defer m.deinit();
    try m.loadDraft(io, drafter);
    try std.Io.Dir.cwd().createDirPath(io, output);
    var buf: [256]u8 = undefined;
    for ([_]usize{ 3, 5, 16, 1, 16 }, [_]usize{ 3, 15, 1, 7, 15 }, 0..) |count, budget, step| {
        var tokens: [16]i32 = undefined;
        const attempted = @min(16, count + 2);
        for (tokens[0..attempted], 0..) |*token, j| token.* = 1000 + m.position + @as(i32, @intCast(j));
        var pass = try m.forward(tokens[0..attempted]);
        defer pass.deinit();
        try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "target-{d}", .{step}), try pass.scope.slice(pass.logits, 0, 0, @intCast(count)));
        try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "taps-{d}", .{step}), try pass.scope.slice(pass.taps, 0, 0, @intCast(count)));
        try m.commit(&pass, count);
        try std.testing.expectError(error.InvalidCommit, m.commit(&pass, 1));
        var proposed: [16]i32 = undefined;
        try m.propose(mx.empty, 42, proposed[0 .. budget + 1], .{});
        try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "proposal-{d}", .{step}), try pass.scope.ints(proposed[0 .. budget + 1]));
        for (m.draft.?.cache, 0..) |cache, i| {
            try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "keys-{d}-{d}", .{ step, i }), cache.keys);
            try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "values-{d}-{d}", .{ step, i }), cache.values);
        }
    }
    for ([_]f64{ 0, 0.8 }) |temperature| {
        const settings = @import("sampling.zig").Sampling{ .temperature = temperature, .seed = 1234, .metal = true };
        m.reset();
        var serial = try @import("serial_generation.zig").generate(&m, &.{ 1000, 1001, 1002, 1003 }, 12, settings, 0, null);
        defer serial.deinit();
        for ([_]usize{ 1, 3, 15 }) |budget| {
            m.reset();
            var drafted = try @import("serial_generation.zig").generate(&m, &.{ 1000, 1001, 1002, 1003 }, 12, settings, budget, null);
            defer drafted.deinit();
            try std.testing.expectEqualSlices(u32, serial.tokens.items, drafted.tokens.items);
        }
    }
    std.debug.print("PASS: Gemma DFlash partial target commits and greedy/seeded generation at budgets 1, 3 and 15.\n", .{});
}
fn saveDraft(s: *mx.Scope, dir: []const u8, name: []const u8, value: A) !void {
    const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ dir, name }, 0);
    defer mx.allocator.free(path);
    const out = try s.cast(value, mx.f32t);
    try mx.eval(out);
    try mx.check(mx.c.mlx_save(path, out));
}
