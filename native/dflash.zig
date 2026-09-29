const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const A = mx.Array;
const c = mx.c;

const Options = struct {
    block_size: i32 = 16,
    target_layer_ids: []const i32,
    mask_token_id: i32,
    input_embedding_scale: f32 = 1,
    output_multiplier: f32 = 1,
    final_logit_softcapping: ?f32 = null,
};
pub const Config = struct {
    hidden_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    intermediate_size: i32,
    vocab_size: i32,
    rms_norm_eps: f32,
    rope_theta: f32 = 10000,
    max_position_embeddings: i32,
    num_target_layers: i32,
    block_size: i32 = 16,
    dflash_config: Options,
    layer_types: []const []const u8 = &.{},
    sliding_window: ?i32 = null,
    is_causal: ?bool = null,
    final_logit_softcapping: ?f32 = null,
    rope_scaling: ?Rope = null,
    rope_parameters: ?Rope = null,
    architectures: []const []const u8 = &.{},
    const Rope = struct { rope_type: []const u8 = "default", type: ?[]const u8 = null, factor: f32 = 1, partial_rotary_factor: f32 = 1, rope_theta: ?f32 = null };

    pub fn validate(v: Config, hidden: i32, vocab: i32, layers: i32) !void {
        for (v.architectures) |name| if (!std.mem.eql(u8, name, "DFlashDraftModel")) return error.UnsupportedDraftArchitecture;
        if (v.hidden_size != hidden or v.vocab_size != vocab or v.num_target_layers != layers or v.num_hidden_layers < 1 or v.num_hidden_layers > 32 or v.num_attention_heads < 1 or v.num_attention_heads > 128 or v.num_key_value_heads < 1 or v.num_key_value_heads > v.num_attention_heads or @mod(v.num_attention_heads, v.num_key_value_heads) != 0 or v.head_dim < 2 or v.head_dim > 512 or @mod(v.head_dim, 2) != 0 or v.intermediate_size < 1 or v.intermediate_size > 65536 or v.max_position_embeddings < 1 or !(v.rms_norm_eps > 0) or !std.math.isFinite(v.rms_norm_eps) or !(v.rope_theta > 0) or !std.math.isFinite(v.rope_theta)) return error.InvalidDraftConfig;
        const d = v.dflash_config;
        if (d.block_size < 2 or d.block_size > 128 or d.mask_token_id < 0 or d.mask_token_id >= vocab or d.target_layer_ids.len == 0 or d.target_layer_ids.len > 32 or !std.math.isFinite(d.input_embedding_scale) or !std.math.isFinite(d.output_multiplier)) return error.InvalidDraftConfig;
        for (d.target_layer_ids, 0..) |id, j| if (id < 0 or id >= layers or (j > 0 and id <= d.target_layer_ids[j - 1])) return error.InvalidDraftConfig;
        if (v.layer_types.len != 0 and v.layer_types.len != v.num_hidden_layers) return error.InvalidDraftConfig;
        for (v.layer_types) |kind| {
            if (std.mem.eql(u8, kind, "sliding_attention")) {
                if ((v.sliding_window orelse 0) < 2) return error.InvalidDraftConfig;
            } else if (!std.mem.eql(u8, kind, "full_attention")) return error.InvalidDraftConfig;
        }
        if (v.rope_parameters orelse v.rope_scaling) |rope| {
            const kind = rope.type orelse rope.rope_type;
            if (!std.mem.eql(u8, kind, "default") and !std.mem.eql(u8, kind, "linear") and !std.mem.eql(u8, kind, "proportional")) return error.UnsupportedDraftRope;
            if (!(rope.factor > 0) or !std.math.isFinite(rope.factor) or !(rope.partial_rotary_factor > 0) or rope.partial_rotary_factor > 1) return error.InvalidDraftConfig;
            if (rope.rope_theta) |theta| if (!(theta > 0) or !std.math.isFinite(theta)) return error.InvalidDraftConfig;
            if (std.mem.eql(u8, kind, "proportional")) {
                const rotated: i32 = @intFromFloat(@as(f32, @floatFromInt(v.head_dim)) * rope.partial_rotary_factor);
                if (rotated < 2 or @mod(rotated, 2) != 0) return error.InvalidDraftConfig;
            }
        }
    }
    fn sliding(v: Config, i: usize) bool {
        return v.layer_types.len > 0 and std.mem.eql(u8, v.layer_types[i], "sliding_attention");
    }
};
pub const Cache = struct {
    keys: A = mx.empty,
    values: A = mx.empty,
    fn deinit(v: *Cache) void {
        mx.free(v.keys);
        mx.free(v.values);
        v.* = .{};
    }
};
const Linear = struct {
    weight: A,
    scales: A = mx.empty,
    biases: A = mx.empty,
    spec: @import("quantization.zig").Spec = .{ .bits = 8 },
    fn apply(l: Linear, s: *mx.Scope, x: A) !A {
        if (l.scales.ctx == null) return s.binary(c.mlx_matmul, x, try s.transpose(l.weight, &.{ 1, 0 }));
        var out = c.mlx_array_new();
        const rc = c.mlx_quantized_matmul(&out, x, l.weight, l.scales, l.biases, true, mx.opt(l.spec.group_size), mx.opt(l.spec.bits), "affine", mx.stream);
        return s.result(rc, out);
    }
};
const Post = struct {
    norm: A,
    gate: Linear,
    up: Linear,
    down: Linear,
    eps: f32,
    fn destroy(raw: ?*anyopaque) callconv(.c) void {
        mx.allocator.destroy(@as(*Post, @ptrCast(@alignCast(raw.?))));
    }
    fn callback(out: [*c]c.mlx_vector_array, ins: c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        return @as(*Post, @ptrCast(@alignCast(raw.?))).graph(out, ins) catch -1;
    }
    fn graph(p: *Post, out: [*c]c.mlx_vector_array, ins: c.mlx_vector_array) !c_int {
        var s = mx.Scope{};
        defer s.deinit();
        var args: [2]A = undefined;
        for (&args, 0..) |*a, j| {
            var value = c.mlx_array_new();
            const rc = c.mlx_vector_array_get(&value, ins, j);
            a.* = try s.result(rc, value);
        }
        const h = try s.binary(c.mlx_add, args[0], args[1]);
        const x = try cp.norm(&s, h, p.norm, p.eps);
        const act = try @import("prefill_ops.zig").uncompiled(&s, .swiglu, &.{ try p.gate.apply(&s, x), try p.up.apply(&s, x) });
        const result = try s.binary(c.mlx_add, h, try p.down.apply(&s, act));
        return c.mlx_vector_array_set_data(out, &result, 1);
    }
};

pub const Draft = struct {
    parsed: std.json.Parsed(Config),
    weights: cp.Store,
    linears: std.StringHashMap(Linear),
    cache: []Cache,
    posts: []c.mlx_closure,
    position: i32 = 0,
    projected_position: i32 = 0,
    pending: A = mx.empty,
    started: bool = false,
    pub fn init(io: std.Io, dir: []const u8, hidden: i32, vocab: i32, layers: i32, bits: i32) !Draft {
        if (bits != 0) try (@import("quantization.zig").Spec{ .bits = bits }).validate();
        var path: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        var parsed = try std.json.parseFromSlice(Config, mx.allocator, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        var owned = false;
        errdefer if (!owned) parsed.deinit();
        const json = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer json.deinit();
        if (!json.value.object.get("dflash_config").?.object.contains("block_size")) parsed.value.dflash_config.block_size = parsed.value.block_size;
        try parsed.value.validate(hidden, vocab, layers);
        const cache = try mx.allocator.alloc(Cache, @intCast(parsed.value.num_hidden_layers));
        errdefer if (!owned) mx.allocator.free(cache);
        @memset(cache, .{});
        const posts = try mx.allocator.alloc(c.mlx_closure, cache.len);
        @memset(posts, .{ .ctx = null });
        var d = Draft{ .parsed = parsed, .weights = cp.Store.init(64), .linears = std.StringHashMap(Linear).init(mx.allocator), .cache = cache, .posts = posts };
        owned = true;
        errdefer d.deinit();
        try d.weights.load(io, dir, "");
        try d.prepare("fc", hidden, hidden * @as(i32, @intCast(parsed.value.dflash_config.target_layer_ids.len)), json.value, bits);
        try d.norm("hidden_norm.weight", hidden);
        try d.norm("norm.weight", hidden);
        for (0..cache.len) |i| {
            const cfg = parsed.value;
            inline for (.{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |suffix| try d.norm(try std.fmt.bufPrint(&path, "layers.{d}.{s}", .{ i, suffix }), hidden);
            inline for (.{ "self_attn.q_norm.weight", "self_attn.k_norm.weight" }) |suffix| try d.norm(try std.fmt.bufPrint(&path, "layers.{d}.{s}", .{ i, suffix }), cfg.head_dim);
            const shapes = [_]struct { []const u8, i32, i32 }{
                .{ "self_attn.q_proj", cfg.num_attention_heads * cfg.head_dim, hidden }, .{ "self_attn.k_proj", cfg.num_key_value_heads * cfg.head_dim, hidden }, .{ "self_attn.v_proj", cfg.num_key_value_heads * cfg.head_dim, hidden }, .{ "self_attn.o_proj", hidden, cfg.num_attention_heads * cfg.head_dim }, .{ "mlp.gate_proj", cfg.intermediate_size, hidden }, .{ "mlp.up_proj", cfg.intermediate_size, hidden }, .{ "mlp.down_proj", hidden, cfg.intermediate_size },
            };
            for (shapes) |shape| try d.prepare(try std.fmt.bufPrint(&path, "layers.{d}.{s}", .{ i, shape[0] }), shape[1], shape[2], json.value, bits);
            const post = Post{ .norm = try d.weight(i, "post_attention_layernorm.weight"), .gate = try d.linear(i, "mlp.gate_proj"), .up = try d.linear(i, "mlp.up_proj"), .down = try d.linear(i, "mlp.down_proj"), .eps = cfg.rms_norm_eps };
            const payload = try mx.allocator.create(Post);
            payload.* = post;
            const fun = c.mlx_closure_new_func_payload(Post.callback, payload, Post.destroy);
            defer _ = c.mlx_closure_free(fun);
            try mx.check(c.mlx_compile(&d.posts[i], fun, false));
        }
        return d;
    }
    pub fn deinit(d: *Draft) void {
        d.reset();
        for (d.posts) |fun| if (fun.ctx != null) {
            _ = c.mlx_closure_free(fun);
        };
        mx.allocator.free(d.posts);
        mx.allocator.free(d.cache);
        var it = d.linears.keyIterator();
        while (it.next()) |key| mx.allocator.free(key.*);
        d.linears.deinit();
        d.weights.deinit();
        d.parsed.deinit();
    }
    pub fn reset(d: *Draft) void {
        for (d.cache) |*cache| cache.deinit();
        mx.free(d.pending);
        d.pending = mx.empty;
        d.position = 0;
        d.projected_position = 0;
        d.started = false;
    }
    fn norm(d: *Draft, name: []const u8, width: i32) !void {
        const w = try d.weights.get(name);
        if (!std.mem.eql(i32, mx.shape(w), &.{width}) or (mx.dtype(w) != mx.bf16 and mx.dtype(w) != c.MLX_FLOAT16 and mx.dtype(w) != mx.f32t)) return error.InvalidTensorShape;
        try mx.eval(w);
    }
    fn prepare(d: *Draft, name: []const u8, n: i32, k: i32, config: std.json.Value, bits: i32) !void {
        var s = mx.Scope{};
        defer s.deinit();
        var l = Linear{ .weight = try d.weights.field(name, "weight") };
        if (mx.dtype(l.weight) == c.MLX_UINT32) {
            l.spec = (try @import("quantization.zig").resolve(config, name)) orelse return error.UnsupportedQuantization;
            l.scales = try d.weights.field(name, "scales");
            l.biases = try d.weights.field(name, "biases");
            if ((mx.dtype(l.scales) != mx.bf16 and mx.dtype(l.scales) != c.MLX_FLOAT16 and mx.dtype(l.scales) != mx.f32t) or mx.dtype(l.biases) != mx.dtype(l.scales)) return error.InvalidTensorDtype;
            const shape = try l.spec.shape(mx.shape(l.weight), mx.shape(l.scales), mx.shape(l.biases));
            if (shape.n != n or shape.k != k) return error.InvalidTensorShape;
        } else {
            if (!std.mem.eql(i32, mx.shape(l.weight), &.{ n, k }) or (mx.dtype(l.weight) != mx.bf16 and mx.dtype(l.weight) != c.MLX_FLOAT16 and mx.dtype(l.weight) != mx.f32t)) return error.InvalidTensorShape;
            if (bits != 0 and @mod(k, 64) == 0) {
                l.spec.bits = bits;
                var quant = c.mlx_vector_array_new();
                defer _ = c.mlx_vector_array_free(quant);
                try mx.check(c.mlx_quantize(&quant, l.weight, mx.opt(64), mx.opt(bits), "affine", mx.empty, mx.stream));
                var values: [3]A = undefined;
                for (&values, 0..) |*v, i| {
                    var a = c.mlx_array_new();
                    const rc = c.mlx_vector_array_get(&a, quant, i);
                    v.* = try s.result(rc, a);
                }
                var buf: [256]u8 = undefined;
                for ([_][]const u8{ "weight", "scales", "biases" }, values) |suffix, v| try d.weights.put(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, suffix }), v);
                l.weight = try d.weights.field(name, "weight");
                l.scales = try d.weights.field(name, "scales");
                l.biases = try d.weights.field(name, "biases");
            }
        }
        try mx.eval(l.weight);
        if (l.scales.ctx != null) try mx.evalMany(&.{ l.scales, l.biases }, false);
        const key = try mx.allocator.dupe(u8, name);
        errdefer mx.allocator.free(key);
        try d.linears.put(key, l);
    }
    fn weight(d: *Draft, i: usize, name: []const u8) !A {
        var buf: [256]u8 = undefined;
        return d.weights.get(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, name }));
    }
    fn linear(d: *Draft, i: usize, name: []const u8) !Linear {
        var buf: [256]u8 = undefined;
        return d.linears.get(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, name })) orelse error.MissingWeight;
    }
    fn rope(d: *Draft, s: *mx.Scope, x: A, offset: i32) !A {
        const cfg = d.parsed.value;
        const r = cfg.rope_parameters orelse cfg.rope_scaling;
        var base = cfg.rope_theta;
        var scale: f32 = 1;
        const dims = cfg.head_dim;
        var freqs = mx.empty;
        if (r) |v| {
            base = v.rope_theta orelse base;
            const kind = v.type orelse v.rope_type;
            if (std.mem.eql(u8, kind, "linear")) scale = 1 / v.factor;
            if (std.mem.eql(u8, kind, "proportional")) {
                const rotated: usize = @intFromFloat(@as(f32, @floatFromInt(dims)) * v.partial_rotary_factor);
                if (rotated % 2 != 0) return error.InvalidDraftConfig;
                var exponents: [256]f32 = undefined;
                for (exponents[0 .. rotated / 2], 0..) |*value, j| value.* = @as(f32, @floatFromInt(j * 2)) / @as(f32, @floatFromInt(dims));
                const powers = try s.binary(c.mlx_multiply, try s.scalar(v.factor), try s.binary(c.mlx_power, try s.scalar(base), try s.data(&exponents, &.{@intCast(rotated / 2)}, mx.f32t)));
                freqs = try s.cat(&.{ powers, try s.binary(c.mlx_add, try s.zeros(&.{@divExact(dims, 2) - @as(i32, @intCast(rotated / 2))}, mx.f32t), try s.scalar(std.math.inf(f32))) }, 0);
            }
        }
        var out = c.mlx_array_new();
        const rc = c.mlx_fast_rope(&out, x, dims, false, .{ .value = base, .has_value = freqs.ctx == null }, scale, offset, freqs, mx.stream);
        return s.result(rc, out);
    }
    fn kv(d: *Draft, s: *mx.Scope, i: usize, x: A, offset: i32) !Cache {
        const cfg = d.parsed.value;
        const rows = mx.dim(x, 1);
        const key = try cp.norm(s, try s.reshape(try (try d.linear(i, "self_attn.k_proj")).apply(s, x), &.{ 1, rows, cfg.num_key_value_heads, cfg.head_dim }), try d.weight(i, "self_attn.k_norm.weight"), cfg.rms_norm_eps);
        return .{ .keys = try d.rope(s, try s.transpose(key, &.{ 0, 2, 1, 3 }), offset), .values = try s.transpose(try s.reshape(try (try d.linear(i, "self_attn.v_proj")).apply(s, x), &.{ 1, rows, cfg.num_key_value_heads, cfg.head_dim }), &.{ 0, 2, 1, 3 }) };
    }
    pub fn absorb(d: *Draft, taps: A) !void {
        const count = mx.dim(taps, 0);
        if (mx.shape(taps).len != 2 or mx.dim(taps, 1) != d.parsed.value.hidden_size * @as(i32, @intCast(d.parsed.value.dflash_config.target_layer_ids.len))) return error.InvalidDraftContext;
        if (count < 1 or count > d.parsed.value.max_position_embeddings - d.position) return error.ContextLimitExceeded;
        var s = mx.Scope{};
        defer s.deinit();
        var value = if (d.pending.ctx == null) taps else try s.cat(&.{ d.pending, taps }, 0);
        var first = d.projected_position;
        if (!d.started) if (d.parsed.value.sliding_window) |window| {
            const skip = @max(0, mx.dim(value, 0) - window + 1);
            if (skip > 0) {
                value = try s.slice(value, 0, skip, mx.dim(value, 0));
                first += skip;
            }
        };
        const next = try mx.retain(try s.contiguous(value));
        errdefer mx.free(next);
        try mx.eval(next);
        mx.free(d.pending);
        d.pending = next;
        d.projected_position = first;
        d.position += count;
    }
    fn flush(d: *Draft) !void {
        if (d.pending.ctx == null) return;
        const end = d.position;
        d.position = d.projected_position;
        defer d.position = end;
        try d.projectContext(d.pending);
        d.projected_position = d.position;
        mx.free(d.pending);
        d.pending = mx.empty;
        d.started = true;
    }
    fn projectContext(d: *Draft, taps: A) !void {
        const cfg = d.parsed.value;
        const count = mx.dim(taps, 0);
        if (count < 1 or count > cfg.max_position_embeddings - d.position) return error.ContextLimitExceeded;
        var s = mx.Scope{};
        defer s.deinit();
        const x = try cp.norm(&s, try d.linears.get("fc").?.apply(&s, try s.reshape(taps, &.{ 1, count, -1 })), try d.weights.get("hidden_norm.weight"), cfg.rms_norm_eps);
        const next = try mx.allocator.alloc(Cache, d.cache.len);
        defer mx.allocator.free(next);
        @memset(next, .{});
        errdefer for (next) |*item| item.deinit();
        for (d.cache, next, 0..) |old, *item, i| {
            const keep = if (cfg.sliding(i)) cfg.sliding_window.? - 1 else count + d.position;
            const skip = @max(0, count - keep);
            const added = try d.kv(&s, i, try s.slice(x, 1, skip, count), d.position + skip);
            inline for (.{ "keys", "values" }) |field| {
                var value = @field(added, field);
                if (@field(old, field).ctx != null) {
                    var previous = @field(old, field);
                    if (cfg.sliding(i) and mx.dim(previous, 2) >= keep) previous = try s.slice(previous, 2, mx.dim(previous, 2) - keep + 1, mx.dim(previous, 2));
                    value = try s.cat(&.{ previous, value }, 2);
                }
                @field(item, field) = try mx.retain(try s.contiguous(value));
            }
            try mx.evalMany(&.{ item.keys, item.values }, false);
        }
        for (d.cache, next) |*old, item| {
            old.deinit();
            old.* = item;
        }
        d.position += count;
    }
    pub fn forward(d: *Draft, s: *mx.Scope, embeddings: A) !A {
        const cfg = d.parsed.value;
        const rows = mx.dim(embeddings, 1);
        if (rows < 2 or rows > cfg.dflash_config.block_size or d.position == 0) return error.InvalidDraftBlock;
        if (rows > cfg.max_position_embeddings - d.position) return error.ContextLimitExceeded;
        try d.flush();
        var h = embeddings;
        for (d.cache, 0..) |cache, i| {
            const x = try cp.norm(s, h, try d.weight(i, "input_layernorm.weight"), cfg.rms_norm_eps);
            var q = try cp.norm(s, try s.reshape(try (try d.linear(i, "self_attn.q_proj")).apply(s, x), &.{ 1, rows, cfg.num_attention_heads, cfg.head_dim }), try d.weight(i, "self_attn.q_norm.weight"), cfg.rms_norm_eps);
            q = try d.rope(s, try s.transpose(q, &.{ 0, 2, 1, 3 }), d.position);
            const prop = try d.kv(s, i, x, d.position);
            const context = mx.dim(cache.keys, 2);
            const total = context + rows;
            const causal = cfg.is_causal orelse cfg.sliding(i);
            var mask = mx.empty;
            if (causal or cfg.sliding(i)) {
                const values = try mx.allocator.alloc(u8, @intCast(rows * total));
                defer mx.allocator.free(values);
                for (0..@intCast(rows)) |r| for (0..@intCast(total)) |col| {
                    const query = context + @as(i32, @intCast(r));
                    const key: i32 = @intCast(col);
                    values[r * @as(usize, @intCast(total)) + col] = @intFromBool(if (key < context) !cfg.sliding(i) or query - key < cfg.sliding_window.? else !causal or key <= query);
                };
                mask = try s.data(values.ptr, &.{ rows, total }, c.MLX_BOOL);
            }
            var attended = c.mlx_array_new();
            const rc = c.mlx_fast_scaled_dot_product_attention(&attended, q, try s.cat(&.{ cache.keys, prop.keys }, 2), try s.cat(&.{ cache.values, prop.values }, 2), 1 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim))), "", mask, mx.empty, false, mx.stream);
            attended = try s.result(rc, attended);
            const out = try (try d.linear(i, "self_attn.o_proj")).apply(s, try s.reshape(try s.transpose(attended, &.{ 0, 2, 1, 3 }), &.{ 1, rows, cfg.num_attention_heads * cfg.head_dim }));
            const inputs = c.mlx_vector_array_new_data(&[_]A{ h, out }, 2);
            defer _ = c.mlx_vector_array_free(inputs);
            var outputs = c.mlx_vector_array_new();
            defer _ = c.mlx_vector_array_free(outputs);
            try mx.check(c.mlx_closure_apply(&outputs, d.posts[i], inputs));
            var value = c.mlx_array_new();
            const status = c.mlx_vector_array_get(&value, outputs, 0);
            h = try s.result(status, value);
        }
        return cp.norm(s, try s.slice(h, 1, 1, rows), try d.weights.get("norm.weight"), cfg.rms_norm_eps);
    }
};

pub fn check(io: std.Io, dir: []const u8, output: []const u8, case: usize) !void {
    try mx.init();
    defer mx.shutdown();
    if (case > 3) return error.InvalidFixture;
    var d = try Draft.init(io, dir, if (case == 3) 2816 else 128, if (case == 3) 262144 else 256, 30, ([_]i32{ 8, 0, 4, 8 })[case]);
    defer d.deinit();
    var inputs = cp.Store.init(64);
    defer inputs.deinit();
    var path: [4096]u8 = undefined;
    try inputs.loadFile(io, try std.fmt.bufPrint(&path, "{s}/fixtures/inputs.safetensors", .{dir}), "", "");
    try std.Io.Dir.cwd().createDirPath(io, output);
    var position: i32 = 0;
    for (0..5) |step| {
        var s = mx.Scope{};
        defer s.deinit();
        const taps = try inputs.get(try std.fmt.bufPrint(&path, "taps-{d}", .{step}));
        const embeddings = try inputs.get(try std.fmt.bufPrint(&path, "embeddings-{d}", .{step}));
        const count = mx.dim(taps, 0);
        if (count > 1) {
            try d.absorb(try s.slice(taps, 0, 0, 1));
            try d.absorb(try s.slice(taps, 0, 1, count));
        } else try d.absorb(taps);
        position += mx.dim(taps, 0);
        try std.testing.expectEqual(position, d.position);
        const hidden = try d.forward(&s, embeddings);
        try save(&s, output, try std.fmt.bufPrint(&path, "hidden-{d}", .{step}), hidden);
        const again = try d.forward(&s, embeddings);
        var same = c.mlx_array_new();
        const rc = c.mlx_array_equal(&same, hidden, again, false, mx.stream);
        same = try s.result(rc, same);
        try mx.eval(same);
        var equal: bool = false;
        try mx.check(c.mlx_array_item_bool(&equal, same));
        try std.testing.expect(equal);
        try std.testing.expectEqual(position, d.position);
        for (d.cache, 0..) |cache, i| {
            try save(&s, output, try std.fmt.bufPrint(&path, "keys-{d}-{d}", .{ step, i }), cache.keys);
            try save(&s, output, try std.fmt.bufPrint(&path, "values-{d}-{d}", .{ step, i }), cache.values);
        }
    }
    d.reset();
    try std.testing.expectEqual(@as(i32, 0), d.position);
    for (d.cache) |cache| try std.testing.expect(cache.keys.ctx == null and cache.values.ctx == null);
    std.debug.print("PASS: DFlash context caches, repeated proposals and reset.\n", .{});
}
fn save(s: *mx.Scope, dir: []const u8, name: []const u8, value: A) !void {
    const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ dir, name }, 0);
    defer mx.allocator.free(path);
    const out = try s.cast(value, mx.f32t);
    try mx.eval(out);
    try mx.check(c.mlx_save(path, out));
}

test "DFlash validates target identity, taps, attention, rotary layout and budgets" {
    const base = Config{ .hidden_size = 128, .num_hidden_layers = 2, .num_attention_heads = 4, .num_key_value_heads = 2, .head_dim = 64, .intermediate_size = 256, .vocab_size = 256, .rms_norm_eps = 1e-6, .max_position_embeddings = 1024, .num_target_layers = 30, .dflash_config = .{ .target_layer_ids = &.{ 4, 14, 24 }, .mask_token_id = 100 } };
    try base.validate(128, 256, 30);
    try std.testing.expectError(error.InvalidDraftConfig, base.validate(256, 256, 30));
    try std.testing.expectError(error.InvalidDraftConfig, base.validate(128, 512, 30));
    try std.testing.expectError(error.InvalidDraftConfig, base.validate(128, 256, 29));
    for ([_][]const i32{ &.{}, &.{-1}, &.{30}, &.{ 4, 4 }, &.{ 24, 14 } }) |ids| {
        var config = base;
        config.dflash_config.target_layer_ids = ids;
        try std.testing.expectError(error.InvalidDraftConfig, config.validate(128, 256, 30));
    }
    inline for (.{ .{ "num_hidden_layers", 0 }, .{ "num_hidden_layers", 33 }, .{ "num_attention_heads", 3 }, .{ "num_key_value_heads", 0 }, .{ "head_dim", 63 }, .{ "intermediate_size", -1 }, .{ "max_position_embeddings", 0 } }) |field| {
        var config = base;
        @field(config, field[0]) = field[1];
        try std.testing.expectError(error.InvalidDraftConfig, config.validate(128, 256, 30));
    }
    for ([_]i32{ 0, 1, 129 }) |block| {
        var config = base;
        config.dflash_config.block_size = block;
        try std.testing.expectError(error.InvalidDraftConfig, config.validate(128, 256, 30));
    }
    var config = base;
    config.layer_types = &.{ "sliding_attention", "full_attention" };
    try std.testing.expectError(error.InvalidDraftConfig, config.validate(128, 256, 30));
    config.sliding_window = 17;
    try config.validate(128, 256, 30);
    config.rope_parameters = .{ .rope_type = "linear", .factor = 0 };
    try std.testing.expectError(error.InvalidDraftConfig, config.validate(128, 256, 30));
    config.rope_parameters = .{ .rope_type = "unknown" };
    try std.testing.expectError(error.UnsupportedDraftRope, config.validate(128, 256, 30));
    config.rope_parameters = null;
    config.architectures = &.{"DFlash2DraftModel"};
    try std.testing.expectError(error.UnsupportedDraftArchitecture, config.validate(128, 256, 30));
}
