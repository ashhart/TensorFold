const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const src = @import("kernel_sources.zig");
const dense = @import("deepseek_dense.zig");
const ops = @import("large_family_ops.zig");
const A = mx.Array;
const c = mx.c;

const Config = struct {
    hidden_size: i32,
    num_hidden_layers: usize,
    vocab_size: i32,
    rms_norm_eps: f32 = 1e-6,
    num_attention_heads: i32,
    head_dim: i32,
    qk_rope_head_dim: i32,
    q_lora_rank: i32,
    o_lora_rank: i32,
    o_groups: i32,
    sliding_window: i32 = 128,
    compress_ratios: []const i32,
    rope_theta: f32 = 10000,
    compress_rope_theta: f32 = 160000,
    rope_scaling: struct { factor: f32 = 1, original_max_position_embeddings: i32 = 0, beta_fast: f64 = 32, beta_slow: f64 = 1, type: []const u8 = "yarn", rope_type: ?[]const u8 = null } = .{},
    index_n_heads: i32,
    index_head_dim: i32,
    index_topk: i32,
    n_routed_experts: i32,
    num_experts_per_tok: i32,
    moe_intermediate_size: i32,
    num_hash_layers: usize = 0,
    routed_scaling_factor: f32 = 1,
    swiglu_limit: f32 = 0,
    hc_mult: i32 = 4,
    hc_eps: f32 = 1e-6,
    hc_sinkhorn_iters: i32 = 20,
    num_nextn_predict_layers: usize = 0,
    num_key_value_heads: i32 = 1,
    n_shared_experts: i32 = 1,
    scoring_func: []const u8 = "sqrtsoftplus",
    eos_token_id: std.json.Value = .null,
    fn validate(g: Config) !void {
        if (g.num_hidden_layers < 1 or g.num_hidden_layers > 128 or g.compress_ratios.len < g.num_hidden_layers or g.hc_mult != 4 or g.num_key_value_heads != 1 or g.n_shared_experts != 1 or !std.mem.eql(u8, g.scoring_func, "sqrtsoftplus")) return error.UnsupportedModelGeometry;
        inline for (.{ "hidden_size", "vocab_size", "num_attention_heads", "head_dim", "qk_rope_head_dim", "q_lora_rank", "o_lora_rank", "o_groups", "sliding_window", "index_n_heads", "index_head_dim", "index_topk", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "hc_sinkhorn_iters" }) |field| if (@field(g, field) < 1 or @field(g, field) > 1048576) return error.UnsupportedModelGeometry;
        for (g.compress_ratios) |r| if (r != 0 and r != 4 and r != 128) return error.UnsupportedCompressionRatio;
        if (g.num_attention_heads > 256 or g.head_dim > 1024 or g.index_n_heads > 256 or g.index_head_dim > 1024 or g.o_groups > 256 or g.o_lora_rank > 65536 or g.q_lora_rank > 65536 or g.moe_intermediate_size > 65536 or g.n_routed_experts > 1024) return error.UnsupportedModelGeometry;
        if (@mod(g.hidden_size, 64) != 0 or @mod(g.head_dim, 64) != 0 or @mod(g.index_head_dim, 64) != 0 or g.qk_rope_head_dim > @min(g.head_dim, g.index_head_dim) or @mod(g.qk_rope_head_dim, 2) != 0 or @mod(g.num_attention_heads * g.head_dim, g.o_groups) != 0 or g.num_experts_per_tok > @min(16, g.n_routed_experts) or g.num_attention_heads > 256 or g.hidden_size > 16384 or g.head_dim > 1024 or g.hc_sinkhorn_iters > 1024) return error.UnsupportedModelGeometry;
        inline for (.{ "rms_norm_eps", "hc_eps", "rope_theta", "compress_rope_theta" }) |field| if (!std.math.isFinite(@field(g, field)) or @field(g, field) <= 0) return error.InvalidModelConfig;
        if (g.hc_eps > 1 or !std.math.isFinite(g.routed_scaling_factor) or !std.math.isFinite(g.swiglu_limit) or g.swiglu_limit < 0) return error.InvalidModelConfig;
        const r = g.rope_scaling;
        const kind = r.rope_type orelse r.type;
        if ((!std.mem.eql(u8, kind, "yarn") and !std.mem.eql(u8, kind, "deepseek_yarn")) or !std.math.isFinite(r.factor) or r.factor < 1 or !std.math.isFinite(r.beta_fast) or !std.math.isFinite(r.beta_slow) or r.beta_fast <= 0 or r.beta_slow <= 0) return error.UnsupportedRotaryGeometry;
    }
};
const Cache = struct {
    keys: A = mx.empty,
    proj: A = mx.empty,
    pool: A = mx.empty,
    ipool: A = mx.empty,
    fn deinit(cache: *Cache) void {
        inline for (comptime std.meta.fieldNames(Cache)) |field| mx.free(@field(cache, field));
        cache.* = .{};
    }
    fn clone(cache: Cache) !Cache {
        var out = Cache{};
        errdefer out.deinit();
        inline for (comptime std.meta.fieldNames(Cache)) |field| if (@field(cache, field).ctx != null) {
            @field(out, field) = try mx.retain(@field(cache, field));
        };
        return out;
    }
};
pub const Pass = struct {
    scope: mx.Scope = .{},
    logits: A = mx.empty,
    hidden: A = mx.empty,
    streams: A = mx.empty,
    records: [16][]Cache = @splat(&.{}),
    rows: usize,
    position: i32,
    generation: u64,
    is_mtp: bool = false,
    pub fn deinit(p: *Pass) void {
        for (p.records[0..p.rows]) |records| mx.allocator.free(records);
        p.scope.deinit();
    }
};
pub const Model = struct {
    weights: cp.Store,
    config: std.json.Parsed(Config),
    kernels: mx.Kernels,
    dispatch: dense.Dense = .{},
    activations: @import("prefill_ops.zig").Ops = .{},
    cache: []Cache,
    position: i32 = 0,
    generation: u64 = 0,
    vocab: i32,
    trace_dir: ?[]const u8 = null,
    mtp_cache: Cache = .{},
    mtp_position: i32 = 0,
    mtp_generation: u64 = 0,
    has_mtp: bool = false,

    pub fn init(io: std.Io, dir: []const u8) !Model {
        var buf: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const root = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer root.deinit();
        if (root.value != .object) return error.InvalidModelConfig;
        try validateFormats(root.value);
        const parsed = try std.json.parseFromValue(Config, mx.allocator, root.value.object.get("text_config") orelse root.value, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        errdefer parsed.deinit();
        try parsed.value.validate();
        const cache = try mx.allocator.alloc(Cache, parsed.value.num_hidden_layers);
        @memset(cache, .{});
        var m = Model{ .weights = cp.Store.init(64), .config = parsed, .kernels = mx.Kernels.init(), .cache = cache, .vocab = parsed.value.vocab_size };
        errdefer {
            m.reset();
            mx.allocator.free(cache);
            m.weights.deinit();
            m.kernels.deinit();
            m.dispatch.deinit();
        }
        try m.weights.load(io, dir, "");
        try m.validateGlobal();
        try m.dispatch.prepare(&m.kernels, &.{.{ .weights = try m.weights.triple("lm_head") }});
        for (0..cache.len) |i| try m.prepare(i);
        m.loadDraft(io, dir) catch |err| if (err != error.FileNotFound) return err;
        return m;
    }
    pub fn deinit(m: *Model) void {
        m.reset();
        mx.allocator.free(m.cache);
        m.weights.deinit();
        m.kernels.deinit();
        m.dispatch.deinit();
        m.activations.deinit();
        m.config.deinit();
    }
    pub fn reset(m: *Model) void {
        for (m.cache) |*cache| cache.deinit();
        m.position = 0;
        m.generation +%= 1;
        m.mtp_cache.deinit();
        m.mtp_position = 0;
        m.mtp_generation +%= 1;
    }
    fn layerRatio(m: *Model, i: usize) i32 {
        return if (i < m.config.value.compress_ratios.len) m.config.value.compress_ratios[i] else 0;
    }
    pub fn loadDraft(m: *Model, io: std.Io, dir: []const u8) !void {
        var path: [4096]u8 = undefined;
        var raw = cp.Store.init(64);
        defer raw.deinit();
        try raw.loadFile(io, try std.fmt.bufPrint(&path, "{s}/mtp.safetensors", .{dir}), "", "");
        var it = raw.arrays.iterator();
        while (it.next()) |entry| {
            if (!std.mem.startsWith(u8, entry.key_ptr.*, "mtp.")) return error.InvalidDraftCheckpoint;
            try m.put(m.cache.len, entry.key_ptr.*[4..], entry.value_ptr.*);
        }
        for ([_][]const u8{ "e_proj", "h_proj" }) |key| try m.dispatch.prepare(&m.kernels, &.{.{ .weights = try m.triple(m.cache.len, key) }});
        try m.prepare(m.cache.len);
        var projections: [7]dense.Projection = undefined;
        for ([_][]const u8{ "e_proj", "h_proj", "attn.x_proj", "attn.wq_b", "attn.wo_b", "ffn.shared_gate_up", "ffn.shared_experts.down_proj" }, &projections) |key, *projection| projection.* = .{ .weights = try m.triple(m.cache.len, key) };
        try m.dispatch.prepareAdditional(&m.kernels, &projections);
        m.mtp_cache.deinit();
        m.mtp_position = 0;
        m.mtp_generation +%= 1;
        m.has_mtp = true;
    }
    pub fn draftHidden(p: *Pass) A {
        return p.streams;
    }
    pub fn isEos(m: *Model, id: i32) bool {
        const eos = m.config.value.eos_token_id;
        if (eos == .integer) return id == eos.integer;
        if (eos == .array) for (eos.array.items) |v| {
            if (v == .integer and v.integer == id) return true;
        };
        return false;
    }
    fn name(buf: []u8, i: usize, suffix: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "model.layers.{d}.{s}", .{ i, suffix });
    }
    fn weight(m: *Model, i: usize, suffix: []const u8) !A {
        var b: [256]u8 = undefined;
        return m.weights.get(try name(&b, i, suffix));
    }
    fn triple(m: *Model, i: usize, suffix: []const u8) ![3]A {
        var b: [256]u8 = undefined;
        return m.weights.triple(try name(&b, i, suffix));
    }
    fn put(m: *Model, i: usize, suffix: []const u8, value: A) !void {
        var b: [256]u8 = undefined;
        try m.weights.put(try name(&b, i, suffix), value);
    }
    fn stack(m: *Model, s: *mx.Scope, i: usize, dest: []const u8, members: []const []const u8) !void {
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "weight", "scales", "biases" }) |field| {
            var values: [4]A = undefined;
            if (members.len > values.len) return error.InvalidProjection;
            for (members, 0..) |member, j| values[j] = try m.weight(i, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ member, field }));
            try m.put(i, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ dest, field }), try s.cat(values[0..members.len], 0));
        }
    }
    fn prepare(m: *Model, i: usize) !void {
        try m.validateLayer(i);
        var s = mx.Scope{};
        defer s.deinit();
        try m.stack(&s, i, "attn.x_proj", &.{ "attn.wq_a", "attn.wkv" });
        try m.stack(&s, i, "ffn.shared_gate_up", &.{ "ffn.shared_experts.gate_proj", "ffn.shared_experts.up_proj" });
        const ratio = m.layerRatio(i);
        if (ratio > 0) try m.stack(&s, i, "attn.cproj", if (ratio == 4) &.{ "attn.compressor.wkv", "attn.compressor.wgate", "attn.indexer.compressor.wkv", "attn.indexer.compressor.wgate" } else &.{ "attn.compressor.wkv", "attn.compressor.wgate" });
        try m.put(i, "router", try s.contiguous(try s.transpose(try s.cast(try m.weight(i, "ffn.gate.weight"), mx.f32t), &.{ 1, 0 })));
        const router = try m.weight(i, "ffn.gate.weight");
        if (mx.dtype(router) == mx.bf16 and @mod(mx.dim(router, 0), 16) == 0 and @mod(mx.dim(router, 1), 32) == 0) {
            const repacked = try s.contiguous(try s.transpose(try s.reshape(try s.transpose(router, &.{ 1, 0 }), &.{ @divExact(mx.dim(router, 1), 32), 8, 4, @divExact(mx.dim(router, 0), 4), 4 }), &.{ 3, 1, 0, 2, 4 }));
            try m.put(i, "router_packed", repacked);
        }
        try m.put(i, "inv", try m.frequencies(&s, ratio));
        if (m.config.value.hidden_size == 4096) for ([_][]const u8{ "attn", "ffn" }) |kind| {
            var buf: [256]u8 = undefined;
            const fnw = try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.fn", .{kind}));
            const prepared = if (mx.dtype(fnw) == mx.bf16) try s.contiguous(try s.transpose(try s.reshape(fnw, &.{ 6, 4, 16, 8, 32, 4 }), &.{ 0, 3, 4, 2, 1, 5 })) else try s.cast(fnw, mx.f32t);
            try m.put(i, try std.fmt.bufPrint(&buf, "{s}_hc.prepared", .{kind}), prepared);
        };
        for ([_][]const u8{ "attn.x_proj", "attn.wq_b", "attn.wo_b", "ffn.shared_gate_up", "ffn.shared_experts.down_proj" }) |key| try m.dispatch.prepare(&m.kernels, &.{.{ .weights = try m.triple(i, key) }});
    }
    fn expectTensor(value: A, dims: []const i32, dtype: c.mlx_dtype) !void {
        if (!std.mem.eql(i32, mx.shape(value), dims)) return error.InvalidTensorShape;
        if (mx.dtype(value) != dtype) return error.InvalidTensorDType;
    }
    fn expectFloat(value: A, dims: []const i32) !void {
        if (mx.dtype(value) != mx.f32t and mx.dtype(value) != mx.bf16) return error.InvalidTensorDType;
        if (!std.mem.eql(i32, mx.shape(value), dims)) return error.InvalidTensorShape;
    }
    fn expectQ(weights: [3]A, n: i32, k: i32) !void {
        if (@mod(k, 64) != 0) return error.UnsupportedProjectionGeometry;
        try expectTensor(weights[0], &.{ n, @divExact(k, 8) }, c.MLX_UINT32);
        try expectTensor(weights[1], &.{ n, @divExact(k, 64) }, mx.bf16);
        try expectTensor(weights[2], &.{ n, @divExact(k, 64) }, mx.bf16);
    }
    fn validateGlobal(m: *Model) !void {
        const g = m.config.value;
        try expectQ(try m.weights.triple("model.embed_tokens"), g.vocab_size, g.hidden_size);
        try expectQ(try m.weights.triple("lm_head"), g.vocab_size, g.hidden_size);
        try expectTensor(try m.weights.get("model.norm.weight"), &.{g.hidden_size}, mx.bf16);
        try expectFloat(try m.weights.get("model.hc_head.fn"), &.{ 4, 4 * g.hidden_size });
        try expectFloat(try m.weights.get("model.hc_head.base"), &.{4});
        try expectFloat(try m.weights.get("model.hc_head.scale"), &.{1});
    }
    fn validateLayer(m: *Model, i: usize) !void {
        const g = m.config.value;
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "attn", "ffn" }) |kind| {
            try expectFloat(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.fn", .{kind})), &.{ 24, 4 * g.hidden_size });
            try expectFloat(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.base", .{kind})), &.{24});
            try expectFloat(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.scale", .{kind})), &.{3});
            try expectTensor(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_norm.weight", .{kind})), &.{g.hidden_size}, mx.bf16);
        }
        const shapes = [_]struct { []const u8, i32, i32 }{
            .{ "attn.wq_a", g.q_lora_rank, g.hidden_size },                              .{ "attn.wkv", g.head_dim, g.hidden_size },                                .{ "attn.wq_b", g.num_attention_heads * g.head_dim, g.q_lora_rank },         .{ "attn.wo_a", g.o_groups * g.o_lora_rank, @divExact(g.num_attention_heads * g.head_dim, g.o_groups) }, .{ "attn.wo_b", g.hidden_size, g.o_groups * g.o_lora_rank },
            .{ "ffn.shared_experts.gate_proj", g.moe_intermediate_size, g.hidden_size }, .{ "ffn.shared_experts.up_proj", g.moe_intermediate_size, g.hidden_size }, .{ "ffn.shared_experts.down_proj", g.hidden_size, g.moe_intermediate_size },
        };
        for (shapes) |shape| try expectQ(try m.triple(i, shape[0]), shape[1], shape[2]);
        try expectTensor(try m.weight(i, "attn.q_norm.weight"), &.{g.q_lora_rank}, mx.bf16);
        try expectTensor(try m.weight(i, "attn.kv_norm.weight"), &.{g.head_dim}, mx.bf16);
        try expectFloat(try m.weight(i, "attn.attn_sink"), &.{g.num_attention_heads});
        try expectFloat(try m.weight(i, "ffn.gate.weight"), &.{ g.n_routed_experts, g.hidden_size });
        if (i < g.num_hash_layers) {
            const table = try m.weight(i, "ffn.gate.tid2eid");
            if (!std.mem.eql(i32, mx.shape(table), &.{ g.vocab_size, g.num_experts_per_tok }) or (mx.dtype(table) != c.MLX_INT64 and mx.dtype(table) != mx.i32t)) return error.InvalidRoutingTable;
            var s = mx.Scope{};
            defer s.deinit();
            const lo = try s.binary(c.mlx_greater_equal, table, try s.ints(&.{0}));
            const hi = try s.binary(c.mlx_less, table, try s.ints(&.{g.n_routed_experts}));
            var valid = c.mlx_array_new();
            const rc = c.mlx_all(&valid, try s.binary(c.mlx_logical_and, lo, hi), false, mx.stream);
            valid = try s.result(rc, valid);
            var ok = false;
            try mx.check(c.mlx_array_item_bool(&ok, valid));
            if (!ok) return error.InvalidRoutingTable;
        } else try expectFloat(try m.weight(i, "ffn.gate.e_score_correction_bias"), &.{g.n_routed_experts});
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |key, j| {
            const n = if (j == 2) g.hidden_size else g.moe_intermediate_size;
            const k = if (j == 2) g.moe_intermediate_size else g.hidden_size;
            try expectTensor(try m.weight(i, try std.fmt.bufPrint(&buf, "ffn.switch_mlp.{s}.weight", .{key})), &.{ g.n_routed_experts, n, @divExact(k, 8) }, c.MLX_UINT32);
            try expectTensor(try m.weight(i, try std.fmt.bufPrint(&buf, "ffn.switch_mlp.{s}.scales", .{key})), &.{ g.n_routed_experts, n, @divExact(k, 32) }, c.MLX_UINT8);
        }
        const ratio = m.layerRatio(i);
        if (ratio != 0) for ([_]bool{ false, true }) |index| {
            if (index and ratio != 4) continue;
            const key = if (index) "attn.indexer.compressor" else "attn.compressor";
            const d = if (index) g.index_head_dim else g.head_dim;
            const w = d * @as(i32, if (ratio == 4) 2 else 1);
            for ([_][]const u8{ "wkv", "wgate" }) |part| try expectQ(try m.triple(i, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ key, part })), w, g.hidden_size);
            try expectFloat(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}.ape", .{key})), &.{ ratio, w });
            try expectTensor(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}.norm.weight", .{key})), &.{d}, mx.bf16);
        };
        if (ratio == 4) {
            try expectQ(try m.triple(i, "attn.indexer.wq_b"), g.index_n_heads * g.index_head_dim, g.q_lora_rank);
            try expectQ(try m.triple(i, "attn.indexer.weights_proj"), g.index_n_heads, g.hidden_size);
        }
        if (i == m.cache.len) {
            for ([_][]const u8{ "e_proj", "h_proj" }) |key| try expectQ(try m.triple(i, key), g.hidden_size, g.hidden_size);
            for ([_][]const u8{ "enorm.weight", "hnorm.weight", "norm.weight" }) |key| try expectTensor(try m.weight(i, key), &.{g.hidden_size}, mx.bf16);
            try expectFloat(try m.weight(i, "hc_head.fn"), &.{ 4, 4 * g.hidden_size });
            try expectFloat(try m.weight(i, "hc_head.base"), &.{4});
            try expectFloat(try m.weight(i, "hc_head.scale"), &.{1});
        }
    }
    fn frequencies(m: *Model, s: *mx.Scope, ratio: i32) !A {
        const g = m.config.value;
        const d = g.qk_rope_head_dim;
        const base = if (ratio > 0) g.compress_rope_theta else g.rope_theta;
        const steps = try mx.allocator.alloc(f32, @intCast(@divExact(d, 2)));
        defer mx.allocator.free(steps);
        for (steps, 0..) |*v, j| v.* = @floatFromInt(j);
        const range = try s.data(steps.ptr, &.{@intCast(steps.len)}, mx.f32t);
        const exponent = try s.binary(c.mlx_divide, try s.binary(c.mlx_multiply, range, try s.scalar(2)), try s.scalar(@floatFromInt(d)));
        const freq = try s.binary(c.mlx_divide, try s.scalar(1), try s.binary(c.mlx_power, try s.scalar(base), exponent));
        const r = g.rope_scaling;
        if (ratio == 0 or r.original_max_position_embeddings <= 0 or r.factor <= 1) return freq;
        const df: f64 = @floatFromInt(d);
        const original: f64 = @floatFromInt(r.original_max_position_embeddings);
        const low = @max(0, @floor(df * @log(original / (r.beta_fast * 2 * std.math.pi)) / (2 * @log(@as(f64, base)))));
        var high = @min(df - 1, @ceil(df * @log(original / (r.beta_slow * 2 * std.math.pi)) / (2 * @log(@as(f64, base)))));
        if (low == high) high += 0.001;
        const ramp = try s.binary(c.mlx_divide, try s.binary(c.mlx_subtract, range, try s.scalar(@floatCast(low))), try s.scalar(@floatCast(high - low)));
        const smooth = try s.binary(c.mlx_subtract, try s.scalar(1), try s.binary(c.mlx_minimum, try s.scalar(1), try s.binary(c.mlx_maximum, ramp, try s.scalar(0))));
        return s.binary(c.mlx_add, try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, freq, try s.scalar(r.factor)), try s.binary(c.mlx_subtract, try s.scalar(1), smooth)), try s.binary(c.mlx_multiply, freq, smooth));
    }
    fn project(m: *Model, s: *mx.Scope, i: usize, key: []const u8, x: A) !A {
        return m.dispatch.apply(&m.kernels, s, x, .{ .weights = try m.triple(i, key) });
    }
    fn stock(s: *mx.Scope, x: A, weights: [3]A) !A {
        var out = c.mlx_array_new();
        const rc = c.mlx_quantized_matmul(&out, x, weights[0], weights[1], weights[2], true, mx.opt(64), mx.opt(4), "affine", mx.stream);
        return s.result(rc, out);
    }
    fn append(s: *mx.Scope, old: A, row: A, limit: i32) !A {
        const all = if (old.ctx == null) row else try s.cat(&.{ old, row }, 0);
        return if (limit > 0 and mx.dim(all, 0) > limit) s.contiguous(try s.slice(all, 0, mx.dim(all, 0) - limit, mx.dim(all, 0))) else all;
    }
    fn hc(m: *Model, s: *mx.Scope, i: usize, kind: []const u8, x: A) ![3]A {
        const g = m.config.value;
        var buf: [256]u8 = undefined;
        const z = try cp.norm(s, try s.reshape(try s.cast(x, mx.f32t), &.{ 1, 4 * g.hidden_size }), mx.empty, g.rms_norm_eps);
        const fnw = try s.cast(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.fn", .{kind})), mx.f32t);
        const base = try s.cast(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.base", .{kind})), mx.f32t);
        const scale = try s.cast(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.scale", .{kind})), mx.f32t);
        const mixes = try s.binary(c.mlx_matmul, z, try s.transpose(fnw, &.{ 1, 0 }));
        const result = try m.kernels.run(s, src.glm_hc_split, &.{ x, mixes, scale, base }, &.{ mx.td("T", mx.bf16), mx.ti("HC", 4), mx.ti("ITERS", g.hc_sinkhorn_iters), mx.ti("D", g.hidden_size), mx.ti("EPS_INT", @intFromFloat(@round(g.hc_eps / 1e-9))) }, .{ 256, 1, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ 1, g.hidden_size } }, .{ .shape = &.{ 1, 4 }, .dtype = mx.f32t }, .{ .shape = &.{ 1, 4, 4 }, .dtype = mx.f32t } });
        return .{ result[0], result[1], result[2] };
    }
    fn expand(s: *mx.Scope, x: A, branch: A, post: A, comb: A) !A {
        const y = try s.binary(c.mlx_multiply, try s.reshape(post, &.{ 1, 4, 1 }), try s.reshape(try s.cast(branch, mx.f32t), &.{ 1, 1, -1 }));
        return s.cast(try s.binary(c.mlx_add, y, try s.binary(c.mlx_matmul, try s.transpose(comb, &.{ 0, 2, 1 }), try s.cast(x, mx.f32t))), mx.bf16);
    }
    fn hcStep(m: *Model, s: *mx.Scope, i: usize, kind: []const u8, x: A, pending: ?[3]A) ![4]A {
        const g = m.config.value;
        var buf: [256]u8 = undefined;
        const fnw = try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.prepared", .{kind}));
        const scale = try s.cast(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.scale", .{kind})), mx.f32t);
        const base = try s.cast(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_hc.base", .{kind})), mx.f32t);
        const norm = try m.weight(i, try std.fmt.bufPrint(&buf, "{s}_norm.weight", .{kind}));
        return ops.hcStep(&m.kernels, s, x, pending, fnw, scale, base, norm, g.rms_norm_eps, g.hc_eps, g.hc_sinkhorn_iters);
    }
    fn pool(m: *Model, s: *mx.Scope, i: usize, proj: A, position: i32, index: bool) !A {
        const g = m.config.value;
        const ratio = m.layerRatio(i);
        const d = if (index) g.index_head_dim else g.head_dim;
        const w = d * @as(i32, if (ratio == 4) 2 else 1);
        const first = @divTrunc(position + 1, ratio) - 1;
        const base = position + 1 - mx.dim(proj, 0);
        const key = if (index) "attn.indexer.compressor" else "attn.compressor";
        var buf: [256]u8 = undefined;
        const ape = try s.cast(try m.weight(i, try std.fmt.bufPrint(&buf, "{s}.ape", .{key})), mx.f32t);
        const norm = try m.weight(i, try std.fmt.bufPrint(&buf, "{s}.norm.weight", .{key}));
        return (try m.kernels.run(s, src.ds4_pool_rows, &.{ proj, ape, norm, try m.weight(i, "inv"), try s.scalar(g.rms_norm_eps), try s.ints(&.{ first, base, 0, if (index) 4 * g.head_dim else 0 }) }, &.{ mx.ti("D", d), mx.ti("R", ratio), mx.ti("OV", @intFromBool(ratio == 4)), mx.ti("WT", mx.dim(proj, 1)), mx.ti("W", w), mx.ti("PE", g.qk_rope_head_dim) }, .{ 32, 1, 1 }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ 1, d } }}))[0];
    }
    fn attention(m: *Model, s: *mx.Scope, i: usize, x: A, cache: *Cache, position: i32) !A {
        const g = m.config.value;
        const positions = try s.ints(&.{position});
        const inv = try m.weight(i, "inv");
        const eps = try s.scalar(g.rms_norm_eps);
        const xp = try m.project(s, i, "attn.x_proj", x);
        const qr = try cp.norm(s, try s.slice(xp, 1, 0, g.q_lora_rank), try m.weight(i, "attn.q_norm.weight"), g.rms_norm_eps);
        const kv = try ops.normRope(&m.kernels, s, try s.slice(xp, 1, g.q_lora_rank, g.q_lora_rank + g.head_dim), try m.weight(i, "attn.kv_norm.weight"), positions, inv, eps, true, false);
        const q = try ops.normRope(&m.kernels, s, try s.reshape(try m.project(s, i, "attn.wq_b", qr), &.{ 1, g.num_attention_heads, g.head_dim }), null, positions, inv, eps, true, false);
        cache.keys = try append(s, cache.keys, kv, g.sliding_window);
        const ratio = m.layerRatio(i);
        if (ratio > 0) {
            const weights = try m.triple(i, "attn.cproj");
            const proj = if (@mod(g.hidden_size, 512) == 0 and @mod(mx.dim(weights[0], 0), 4) == 0) try ops.project(&m.kernels, s, x, weights, 4, true, 4) else try stock(s, try s.cast(x, mx.f32t), weights);
            cache.proj = try append(s, cache.proj, proj, ratio * @as(i32, if (ratio == 4) 2 else 1));
            if (@mod(position + 1, ratio) == 0) {
                cache.pool = try append(s, cache.pool, try m.pool(s, i, cache.proj, position, false), 0);
                if (ratio == 4) cache.ipool = try append(s, cache.ipool, try m.pool(s, i, cache.proj, position, true), 0);
            }
        }
        var keys = cache.keys;
        var pool_keys = mx.empty;
        if (cache.pool.ctx != null) {
            var pooled = cache.pool;
            if (ratio == 4 and mx.dim(pooled, 0) > g.index_topk) {
                const iq = try ops.normRope(&m.kernels, s, try s.reshape(try stock(s, qr, try m.triple(i, "attn.indexer.wq_b")), &.{ 1, g.index_n_heads, g.index_head_dim }), null, positions, inv, eps, false, false);
                const iw = try s.binary(c.mlx_multiply, try stock(s, x, try m.triple(i, "attn.indexer.weights_proj")), try s.cast(try s.scalar(1 / @sqrt(@as(f32, @floatFromInt(g.index_n_heads)))), mx.bf16));
                const dot = try s.binary(c.mlx_matmul, iq, try s.transpose(cache.ipool, &.{ 1, 0 }));
                const relu = try s.binary(c.mlx_maximum, dot, try s.cast(try s.scalar(0), mx.bf16));
                const scaled = try s.binary(c.mlx_multiply, relu, try s.cast(try s.scalar(1 / @sqrt(@as(f32, @floatFromInt(g.index_head_dim)))), mx.bf16));
                const weighted = try s.binary(c.mlx_multiply, try s.cast(scaled, mx.f32t), try s.reshape(try s.cast(iw, mx.f32t), &.{ 1, g.index_n_heads, 1 }));
                var scores = c.mlx_array_new();
                const rc = c.mlx_sum_axis(&scores, weighted, 1, false, mx.stream);
                scores = try s.result(rc, scores);
                pooled = try s.take(pooled, try s.reshape(try topk(s, scores, g.index_topk), &.{g.index_topk}), 0);
            }
            pool_keys = pooled;
            keys = try s.cat(&.{ pooled, keys }, 0);
        }
        const native_attention = g.num_attention_heads == 64 and g.head_dim == 512;
        var attended = c.mlx_array_new();
        if (native_attention) {
            mx.free(attended);
            const count = if (pool_keys.ctx != null) mx.dim(pool_keys, 0) else 0;
            const window = mx.dim(cache.keys, 0);
            attended = (try m.kernels.run(s, src.ds4_attn_split, &.{ q, if (count > 0) pool_keys else try s.slice(cache.keys, 0, 0, 1), try s.ints(&.{0}), try s.ints(&.{count}), cache.keys, try s.ints(&.{ 0, window - 1 }), try s.cast(try m.weight(i, "attn.attn_sink"), mx.f32t), try s.scalar(1 / @sqrt(@as(f32, 512))), try s.ints(&.{ 1, 1, window, position - window + 1 }), inv }, &.{ mx.ti("S", 4), mx.ti("ROT", 1), mx.ti("PE", g.qk_rope_head_dim) }, .{ 128, 64, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ 1, 64, 512 } }}))[0];
        } else {
            const key4 = try s.reshape(keys, &.{ 1, 1, mx.dim(keys, 0), g.head_dim });
            const rc = c.mlx_fast_scaled_dot_product_attention(&attended, try s.reshape(q, &.{ 1, g.num_attention_heads, 1, g.head_dim }), key4, key4, 1 / @sqrt(@as(f32, @floatFromInt(g.head_dim))), "", mx.empty, try s.cast(try m.weight(i, "attn.attn_sink"), mx.bf16), false, mx.stream);
            attended = try s.result(rc, attended);
        }
        const rotated = if (native_attention) attended else try ops.normRope(&m.kernels, s, try s.reshape(attended, &.{ 1, g.num_attention_heads, g.head_dim }), null, positions, inv, eps, false, true);
        const grouped = try s.reshape(rotated, &.{ 1, g.o_groups, -1 });
        const weights = try m.triple(i, "attn.wo_a");
        if (@mod(mx.dim(grouped, 2), 512) == 0 and @mod(mx.dim(weights[0], 0), 4 * g.o_groups) == 0) {
            var batched: [3]A = undefined;
            for (&batched, weights) |*dst, tensor| dst.* = try s.reshape(tensor, &.{ g.o_groups, g.o_lora_rank, -1 });
            return m.project(s, i, "attn.wo_b", try s.reshape(try stock(s, try s.reshape(grouped, &.{ g.o_groups, 1, -1 }), batched), &.{ 1, g.o_groups * g.o_lora_rank }));
        }
        const parts = try mx.allocator.alloc(A, @intCast(g.o_groups));
        defer mx.allocator.free(parts);
        for (parts, 0..) |*part, j| {
            const at: i32 = @intCast(j);
            var w: [3]A = undefined;
            for (&w, weights) |*a, tensor| a.* = try s.slice(tensor, 0, at * g.o_lora_rank, (at + 1) * g.o_lora_rank);
            part.* = try stock(s, try s.reshape(try s.slice(grouped, 1, at, at + 1), &.{ 1, -1 }), w);
        }
        return m.project(s, i, "attn.wo_b", try s.cat(parts, 1));
    }
    fn topk(s: *mx.Scope, scores: A, top: i32) !A {
        var ids = c.mlx_array_new();
        const rc = c.mlx_argpartition_axis(&ids, try s.unary(c.mlx_negative, scores), top - 1, -1, mx.stream);
        ids = try s.result(rc, ids);
        var sorted = c.mlx_array_new();
        const sort_rc = c.mlx_sort_axis(&sorted, try s.slice(ids, 1, 0, top), -1, mx.stream);
        return s.result(sort_rc, sorted);
    }
    fn swiglu(m: *Model, s: *mx.Scope, gate_: A, up_: A) !A {
        if (m.config.value.swiglu_limit != 0) return m.activations.call(s, .clipped_swiglu, &.{ gate_, up_, try s.scalar(m.config.value.swiglu_limit) });
        return m.activations.call(s, .swiglu, &.{ gate_, up_ });
    }
    fn fp4(m: *Model, s: *mx.Scope, i: usize, key: []const u8, x: A, ids: A) !A {
        var buf: [256]u8 = undefined;
        const w = try m.weight(i, try std.fmt.bufPrint(&buf, "ffn.switch_mlp.{s}.weight", .{key}));
        const scales = try m.weight(i, try std.fmt.bufPrint(&buf, "ffn.switch_mlp.{s}.scales", .{key}));
        var out = c.mlx_array_new();
        const rc = c.mlx_gather_qmm(&out, x, w, scales, mx.empty, mx.empty, ids, true, mx.opt(32), mx.opt(4), "mxfp4", false, mx.stream);
        return s.result(rc, out);
    }
    fn moe(m: *Model, s: *mx.Scope, i: usize, x: A, token: i32) !A {
        const g = m.config.value;
        var namebuf: [256]u8 = undefined;
        if (@mod(g.hidden_size, 512) == 0 and @mod(g.moe_intermediate_size, 512) == 0 and g.n_routed_experts < 512 and @mod(g.n_routed_experts, 16) == 0 and m.weights.has(try name(&namebuf, i, "router_packed"))) return m.fusedMoe(s, i, x, token);
        const logits = try s.binary(c.mlx_matmul, try s.cast(x, mx.f32t), try m.weight(i, "router"));
        const scores = try s.unary(c.mlx_sqrt, try s.binary(c.mlx_logaddexp, logits, try s.scalar(0)));
        const ids = if (i < g.num_hash_layers) blk: {
            var sorted = c.mlx_array_new();
            const rc = c.mlx_sort_axis(&sorted, try s.take(try m.weight(i, "ffn.gate.tid2eid"), try s.ints(&.{token}), 0), -1, mx.stream);
            break :blk try s.result(rc, sorted);
        } else try topk(s, try s.binary(c.mlx_add, scores, try s.cast(try m.weight(i, "ffn.gate.e_score_correction_bias"), mx.f32t)), g.num_experts_per_tok);
        var weights = c.mlx_array_new();
        const take_rc = c.mlx_take_along_axis(&weights, scores, ids, -1, mx.stream);
        weights = try s.result(take_rc, weights);
        var total = try s.slice(weights, 1, 0, 1);
        var j: i32 = 1;
        while (j < g.num_experts_per_tok) : (j += 1) total = try s.binary(c.mlx_add, total, try s.slice(weights, 1, j, j + 1));
        weights = try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, weights, try s.binary(c.mlx_add, total, try s.scalar(1e-20))), try s.scalar(g.routed_scaling_factor));
        const h = try s.reshape(x, &.{ 1, 1, 1, g.hidden_size });
        const act = try m.swiglu(s, try m.fp4(s, i, "gate_proj", h, ids), try m.fp4(s, i, "up_proj", h, ids));
        const y = try s.reshape(try s.cast(try m.fp4(s, i, "down_proj", act, ids), mx.f32t), &.{ 1, g.num_experts_per_tok, g.hidden_size });
        var acc = try s.binary(c.mlx_multiply, try s.slice(weights, 1, 0, 1), try s.reshape(try s.slice(y, 1, 0, 1), &.{ 1, g.hidden_size }));
        j = 1;
        while (j < g.num_experts_per_tok) : (j += 1) acc = try s.binary(c.mlx_add, acc, try s.binary(c.mlx_multiply, try s.slice(weights, 1, j, j + 1), try s.reshape(try s.slice(y, 1, j, j + 1), &.{ 1, g.hidden_size })));
        const gu = try m.project(s, i, "ffn.shared_gate_up", x);
        const width = @divExact(mx.dim(gu, 1), 2);
        const shared = try m.project(s, i, "ffn.shared_experts.down_proj", try m.swiglu(s, try s.slice(gu, 1, 0, width), try s.slice(gu, 1, width, 2 * width)));
        return s.binary(c.mlx_add, try s.cast(acc, mx.bf16), shared);
    }
    fn fusedMoe(m: *Model, s: *mx.Scope, i: usize, x: A, token: i32) !A {
        const g = m.config.value;
        const top = g.num_experts_per_tok;
        const hidden = g.hidden_size;
        const width = g.moe_intermediate_size;
        const logits = (try m.kernels.run(s, src.glm_router_tg, &.{ x, try m.weight(i, "router_packed") }, &.{ mx.ti("K", hidden), mx.ti("NE", g.n_routed_experts), mx.ti("RR", 1), mx.ti("C", 8), mx.ti("NT", 1024) }, .{ 1024 * @divExact(g.n_routed_experts, 16), 1, 1 }, .{ 1024, 1, 1 }, &.{.{ .shape = &.{ 1, g.n_routed_experts }, .dtype = mx.f32t }}))[0];
        const hashed = i < g.num_hash_layers;
        const bias = if (hashed) try s.zeros(&.{g.n_routed_experts}, mx.f32t) else try s.cast(try m.weight(i, "ffn.gate.e_score_correction_bias"), mx.f32t);
        const table = if (hashed) try s.cast(try m.weight(i, "ffn.gate.tid2eid"), mx.i32t) else try s.zeros(&.{8}, mx.i32t);
        const routes = try m.kernels.run(s, src.ds4_moe_route, &.{ logits, bias, table, try s.cast(try s.ints(&.{token}), c.MLX_UINT32), try s.scalar(g.routed_scaling_factor) }, &.{ mx.ti("NE", g.n_routed_experts), mx.ti("TOPK", top), mx.ti("MAXR", 16), mx.ti("NT", 512), mx.ti("HASHED", @intFromBool(hashed)) }, .{ 512, 1, 1 }, .{ 512, 1, 1 }, &.{ .{ .shape = &.{ 1, top }, .dtype = mx.f32t }, .{ .shape = &.{@max(top, 8)}, .dtype = mx.i32t }, .{ .shape = &.{ top, 16 }, .dtype = mx.i32t }, .{ .shape = &.{8}, .dtype = mx.i32t } });
        const act = (try m.kernels.run(s, src.ds4_moe_gateup, &.{ x, try m.weight(i, "ffn.switch_mlp.gate_proj.weight"), try m.weight(i, "ffn.switch_mlp.gate_proj.scales"), try m.weight(i, "ffn.switch_mlp.up_proj.weight"), try m.weight(i, "ffn.switch_mlp.up_proj.scales"), routes[0], routes[1], routes[2], routes[3], try s.scalar(g.swiglu_limit) }, &.{ mx.ti("K", hidden), mx.ti("N", width), mx.ti("RPS", 4), mx.ti("TOPK", top), mx.ti("MAXR", 16) }, .{ 32, @divExact(width, 4), top }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ top, width } }}))[0];
        const routed = (try m.kernels.run(s, src.ds4_expert_fp4, &.{ act, try m.weight(i, "ffn.switch_mlp.down_proj.weight"), try m.weight(i, "ffn.switch_mlp.down_proj.scales"), routes[1], routes[2], routes[3] }, &.{ mx.ti("K", width), mx.ti("N", hidden), mx.ti("RPS", 4), mx.ti("TOPK", top), mx.ti("MAXR", 16), mx.ti("PER_PICK", 1) }, .{ 32, @divExact(hidden, 4), top }, .{ 32, 1, 1 }, &.{.{ .shape = &.{ top, hidden } }}))[0];
        const gu = try m.project(s, i, "ffn.shared_gate_up", x);
        const shared = try m.project(s, i, "ffn.shared_experts.down_proj", try m.swiglu(s, try s.slice(gu, 1, 0, width), try s.slice(gu, 1, width, width * 2)));
        return (try m.kernels.run(s, src.ds4_moe_combine, &.{ routed, shared }, &.{ mx.ti("D", hidden), mx.ti("TOPK", top) }, .{ hidden, 1, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ 1, hidden } }}))[0];
    }
    fn head(m: *Model, s: *mx.Scope, x: A, mtp: bool) !A {
        const g = m.config.value;
        const fnw = if (mtp) try m.weight(m.cache.len, "hc_head.fn") else try m.weights.get("model.hc_head.fn");
        const sc = if (mtp) try m.weight(m.cache.len, "hc_head.scale") else try m.weights.get("model.hc_head.scale");
        const base = if (mtp) try m.weight(m.cache.len, "hc_head.base") else try m.weights.get("model.hc_head.base");
        const y = try m.activations.call(s, .deepseek_head, &.{ x, fnw, base, sc, try s.scalar(g.rms_norm_eps), try s.scalar(g.hc_eps) });
        return cp.norm(s, y, if (mtp) try m.weight(m.cache.len, "norm.weight") else try m.weights.get("model.norm.weight"), g.rms_norm_eps);
    }
    pub fn forward(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len == 0 or tokens.len > 16 or m.position > 1048576 - tokens.len) return error.ContextLimitExceeded;
        for (tokens) |token| if (token < 0 or token >= m.vocab) return error.InvalidToken;
        var p = Pass{ .position = m.position, .generation = m.generation, .rows = tokens.len };
        errdefer p.deinit();
        const s = &p.scope;
        const g = m.config.value;
        var logits: [16]A = undefined;
        var hidden: [16]A = undefined;
        var streams: [16]A = undefined;
        for (tokens, 0..) |token, row| {
            const position = m.position + @as(i32, @intCast(row));
            const records = try mx.allocator.dupe(Cache, if (row == 0) m.cache else p.records[row - 1]);
            p.records[row] = records;
            const h = try m.weights.embed(s, "model.embed_tokens", &.{token});
            var x = try s.stack(&.{ h, h, h, h }, 1);
            var pending: ?[3]A = null;
            for (records, 0..) |*cache, i| {
                if (g.hidden_size == 4096) {
                    const a = try m.hcStep(s, i, "attn", x, pending);
                    try m.trace(s, position, i, "attn-input", a[1]);
                    const branch = try m.attention(s, i, a[1], cache, position);
                    try m.trace(s, position, i, "attn-output", branch);
                    const f = try m.hcStep(s, i, "ffn", a[0], .{ branch, a[2], a[3] });
                    try m.trace(s, position, i, "ffn-input", f[1]);
                    x = f[0];
                    const ff = try m.moe(s, i, f[1], token);
                    try m.trace(s, position, i, "ffn-output", ff);
                    pending = .{ ff, f[2], f[3] };
                    continue;
                }
                const a = try m.hc(s, i, "attn", x);
                const ax = try cp.norm(s, a[0], try m.weight(i, "attn_norm.weight"), g.rms_norm_eps);
                try m.trace(s, position, i, "attn-input", ax);
                const branch = try m.attention(s, i, ax, cache, position);
                try m.trace(s, position, i, "attn-output", branch);
                x = try expand(s, x, branch, a[1], a[2]);
                const f = try m.hc(s, i, "ffn", x);
                const fx = try cp.norm(s, f[0], try m.weight(i, "ffn_norm.weight"), g.rms_norm_eps);
                try m.trace(s, position, i, "ffn-input", fx);
                const ff = try m.moe(s, i, fx, token);
                try m.trace(s, position, i, "ffn-output", ff);
                x = try expand(s, x, ff, f[1], f[2]);
                try m.trace(s, position, i, "streams", x);
            }
            if (pending) |pnd| x = (try ops.hcStep(&m.kernels, s, x, pnd, null, mx.empty, mx.empty, mx.empty, g.rms_norm_eps, g.hc_eps, g.hc_sinkhorn_iters))[0];
            streams[row] = x;
            hidden[row] = try m.head(s, x, false);
            logits[row] = try m.dispatch.apply(&m.kernels, s, hidden[row], .{ .weights = try m.weights.triple("lm_head") });
        }
        p.streams = try s.cat(streams[0..tokens.len], 0);
        p.hidden = try s.cat(hidden[0..tokens.len], 0);
        p.logits = try s.cat(logits[0..tokens.len], 0);
        try mx.eval(p.logits);
        return p;
    }
    fn trace(m: *Model, s: *mx.Scope, position: i32, layer: usize, label: []const u8, value: A) !void {
        const dir = m.trace_dir orelse return;
        var path: [4096]u8 = undefined;
        try save(s, try std.fmt.bufPrint(&path, "{s}/trace-{d}-{d}-{s}.npy", .{ dir, position, layer, label }), value);
    }
    pub fn commit(m: *Model, p: *Pass, keep: usize) !void {
        if (p.is_mtp or p.position != m.position or p.generation != m.generation or keep == 0 or keep > p.rows) return error.InvalidCommit;
        const next = try mx.allocator.alloc(Cache, m.cache.len);
        @memset(next, .{});
        errdefer {
            for (next) |*cache| cache.deinit();
            mx.allocator.free(next);
        }
        for (next, p.records[keep - 1]) |*dst, record| dst.* = try record.clone();
        for (next) |cache| inline for (comptime std.meta.fieldNames(Cache)) |field| {
            if (@field(cache, field).ctx != null) try mx.eval(@field(cache, field));
        };
        for (m.cache) |*cache| cache.deinit();
        mx.allocator.free(m.cache);
        m.cache = next;
        m.position += @intCast(keep);
        m.generation +%= 1;
    }
    pub fn forwardMtp(m: *Model, streams: A, tokens: []const i32) !Pass {
        return m.forwardMtpAt(streams, tokens, m.mtp_cache, m.mtp_position);
    }
    fn forwardMtpAt(m: *Model, streams: A, tokens: []const i32, entry: Cache, position: i32) !Pass {
        if (!m.has_mtp) return error.MissingDraftHead;
        const g = m.config.value;
        if (tokens.len == 0 or tokens.len > 16 or position > 1048576 - tokens.len) return error.ContextLimitExceeded;
        if (!std.mem.eql(i32, mx.shape(streams), &.{ @intCast(tokens.len), 4, g.hidden_size }) or mx.dtype(streams) != mx.bf16) return error.InvalidTensorShape;
        for (tokens) |token| if (token < 0 or token >= m.vocab) return error.InvalidToken;
        var p = Pass{ .position = position, .generation = m.mtp_generation, .rows = tokens.len, .is_mtp = true };
        errdefer p.deinit();
        const s = &p.scope;
        const i = m.cache.len;
        var cache = entry;
        var logits: [16]A = undefined;
        var states: [16]A = undefined;
        var hidden: [16]A = undefined;
        for (tokens, 0..) |token, row| {
            const e = try m.project(s, i, "e_proj", try cp.norm(s, try m.weights.embed(s, "model.embed_tokens", &.{token}), try m.weight(i, "enorm.weight"), g.rms_norm_eps));
            const h = try s.reshape(try cp.norm(s, try s.slice(streams, 0, @intCast(row), @intCast(row + 1)), try m.weight(i, "hnorm.weight"), g.rms_norm_eps), &.{ 4, g.hidden_size });
            var x = try s.binary(c.mlx_add, try s.reshape(e, &.{ 1, 1, g.hidden_size }), try s.reshape(try m.project(s, i, "h_proj", h), &.{ 1, 4, g.hidden_size }));
            const a = try m.hc(s, i, "attn", x);
            const ax = try cp.norm(s, a[0], try m.weight(i, "attn_norm.weight"), g.rms_norm_eps);
            x = try expand(s, x, try m.attention(s, i, ax, &cache, position + @as(i32, @intCast(row))), a[1], a[2]);
            const f = try m.hc(s, i, "ffn", x);
            const fx = try cp.norm(s, f[0], try m.weight(i, "ffn_norm.weight"), g.rms_norm_eps);
            x = try expand(s, x, try m.moe(s, i, fx, token), f[1], f[2]);
            states[row] = x;
            hidden[row] = try m.head(s, x, true);
            logits[row] = try m.dispatch.apply(&m.kernels, s, hidden[row], .{ .weights = try m.weights.triple("lm_head") });
            p.records[row] = try mx.allocator.dupe(Cache, &.{cache});
        }
        p.streams = try s.cat(states[0..tokens.len], 0);
        p.hidden = try s.cat(hidden[0..tokens.len], 0);
        p.logits = try s.cat(logits[0..tokens.len], 0);
        try mx.eval(p.logits);
        return p;
    }
    pub fn commitMtp(m: *Model, p: *Pass, keep: usize) !void {
        if (!p.is_mtp or keep == 0 or keep > p.rows or p.position != m.mtp_position or p.generation != m.mtp_generation) return error.InvalidCommit;
        const next = try p.records[keep - 1][0].clone();
        m.mtp_cache.deinit();
        m.mtp_cache = next;
        m.mtp_position += @intCast(keep);
        m.mtp_generation +%= 1;
    }
    pub fn propose(m: *Model, streams: A, first: i32, tokens: []i32, settings: @import("sampling.zig").Sampling) !void {
        if (tokens.len < 1 or tokens.len > 16 or m.mtp_position != m.position - 1) return error.InvalidDraftState;
        tokens[0] = first;
        var cache = try m.mtp_cache.clone();
        defer cache.deinit();
        var row = try mx.retain(streams);
        defer mx.free(row);
        for (tokens[1..], 0..) |*token, j| {
            var p = try m.forwardMtpAt(row, tokens[j..][0..1], cache, m.mtp_position + @as(i32, @intCast(j)));
            defer p.deinit();
            const ids = try @import("sampling.zig").rows(&m.kernels, &p.scope, p.logits, &.{m.position + @as(i32, @intCast(j)) + 1}, settings);
            defer mx.allocator.free(ids);
            token.* = ids[0];
            const next = try p.records[0][0].clone();
            cache.deinit();
            cache = next;
            try mx.replace(&row, p.streams);
        }
    }
    pub fn checkExact(m: *Model, prefix: usize) !void {
        defer m.reset();
        m.reset();
        {
            var stale = try m.forward(&.{1});
            defer stale.deinit();
            m.reset();
            try std.testing.expectError(error.InvalidCommit, m.commit(&stale, 1));
        }
        var expected = mx.empty;
        defer mx.free(expected);
        const saved = try mx.allocator.alloc(Cache, m.cache.len);
        @memset(saved, .{});
        defer {
            for (saved) |*cache| cache.deinit();
            mx.allocator.free(saved);
        }
        for (0..2) |run| {
            m.reset();
            var at: usize = 0;
            while (at < prefix) {
                const count = @min(16, prefix - at);
                var ids: [16]i32 = undefined;
                for (ids[0..count], 0..) |*id, j| id.* = @intCast((at + j) % @as(usize, @intCast(m.vocab)));
                var p = try m.forward(ids[0..count]);
                defer p.deinit();
                try m.commit(&p, count);
                at += count;
            }
            if (run == 0) {
                var p = try m.forward(&.{ 23, 41, 59, 83 });
                defer p.deinit();
                expected = try mx.retain(p.logits);
                try m.commit(&p, 4);
                for (saved, m.cache) |*dst, cache| dst.* = try cache.clone();
            } else {
                var discarded = try m.forward(&.{ 23, 41, 59, 83, 97, 101 });
                defer discarded.deinit();
                try @import("sampling_checks.zig").equal(&discarded.scope, expected, try discarded.scope.slice(discarded.logits, 0, 0, 4));
                try m.commit(&discarded, 3);
                try std.testing.expectError(error.InvalidCommit, m.commit(&discarded, 1));
                var next = try m.forward(&.{83});
                defer next.deinit();
                try @import("sampling_checks.zig").equal(&next.scope, try next.scope.slice(expected, 0, 3, 4), next.logits);
                try m.commit(&next, 1);
                for (saved, m.cache) |old, new| inline for (comptime std.meta.fieldNames(Cache)) |field| {
                    if (@field(old, field).ctx != null) try @import("sampling_checks.zig").equal(&next.scope, @field(old, field), @field(new, field));
                };
            }
        }
        std.debug.print("PASS: DeepSeek rollback, stale passes, sliding keys and compressed pools at prefix {d}.\n", .{prefix});
    }
};

fn save(s: *mx.Scope, path: []const u8, value: A) !void {
    const z = try mx.allocator.dupeSentinel(u8, path, 0);
    defer mx.allocator.free(z);
    const out = try s.cast(value, mx.f32t);
    try mx.eval(out);
    try mx.check(c.mlx_save(z, out));
}
fn validateFormats(root: std.json.Value) !void {
    const block = blk: {
        for ([_]std.json.Value{ root, root.object.get("text_config") orelse .null }) |source| {
            if (source != .object) continue;
            for ([_][]const u8{ "quantization", "quantization_config" }) |key| if (source.object.get(key)) |value| {
                if (value == .object and value.object.count() > 0) break :blk value;
            };
        }
        return error.UnsupportedQuantization;
    };
    try validateFormat(block, false, false);
    var it = block.object.iterator();
    while (it.next()) |entry| if (entry.value_ptr.* == .object) {
        if (entry.value_ptr.object.count() > 0) try validateFormat(entry.value_ptr.*, std.mem.indexOf(u8, entry.key_ptr.*, ".switch_mlp.") != null, true);
    };
}
fn validateFormat(value: std.json.Value, expert: bool, defaults: bool) !void {
    if (value != .object) return error.UnsupportedQuantization;
    const mode = value.object.get("mode") orelse std.json.Value{ .string = "affine" };
    if (mode != .string or !(if (defaults) std.ascii.eqlIgnoreCase(mode.string, if (expert) "mxfp4" else "affine") else std.mem.eql(u8, mode.string, if (expert) "mxfp4" else "affine"))) return error.UnsupportedQuantization;
    const bits = value.object.get("bits") orelse if (defaults) std.json.Value{ .integer = 4 } else return error.UnsupportedQuantization;
    const group = value.object.get("group_size") orelse if (defaults) std.json.Value{ .integer = if (expert) 32 else 64 } else return error.UnsupportedQuantization;
    if (bits != .integer or bits.integer != 4 or group != .integer or group.integer != @as(i64, if (expert) 32 else 64)) return error.UnsupportedQuantization;
}

test "DeepSeek quantization rejects unsupported default and expert formats" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "{\"quantization\":{\"bits\":4,\"group_size\":64,\"mode\":\"affine\",\"model.layers.0.ffn.switch_mlp.up_proj\":{\"bits\":4,\"group_size\":32,\"mode\":\"mxfp4\"}}}",
        "{\"quantization\":{\"bits\":8,\"group_size\":64}}",
        "{\"quantization\":{\"bits\":4,\"group_size\":64,\"model.layers.0.ffn.switch_mlp.up_proj\":{\"bits\":4,\"group_size\":64}}}",
        "{\"quantization\":{\"bits\":4,\"group_size\":32}}",
        "{}",
    }, 0..) |json, i| {
        const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer parsed.deinit();
        if (i == 0) try validateFormats(parsed.value) else try std.testing.expectError(error.UnsupportedQuantization, validateFormats(parsed.value));
    }
}
pub fn checkModel(io: std.Io, dir: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try Model.init(io, dir);
    defer m.deinit();
    try std.Io.Dir.cwd().createDirPath(io, output);
    m.trace_dir = output;
    var path: [4096]u8 = undefined;
    var position: usize = 0;
    {
        var s = mx.Scope{};
        defer s.deinit();
        for (0..m.cache.len) |i| try save(&s, try std.fmt.bufPrint(&path, "{s}/frequencies-{d}.npy", .{ output, i }), try m.weight(i, "inv"));
    }
    var previous_streams = mx.empty;
    defer mx.free(previous_streams);
    while (position < 137) {
        const count = @min(16, 137 - position);
        var tokens: [16]i32 = undefined;
        for (tokens[0..count], 0..) |*token, j| token.* = @intCast((position + j) % 250 + 1);
        var p = try m.forward(tokens[0..count]);
        defer p.deinit();
        for (0..count) |j| {
            const row: i32 = @intCast(j);
            try save(&p.scope, try std.fmt.bufPrint(&path, "{s}/logits-{d}.npy", .{ output, position + j }), try p.scope.slice(p.logits, 0, row, row + 1));
            try save(&p.scope, try std.fmt.bufPrint(&path, "{s}/hidden-{d}.npy", .{ output, position + j }), try p.scope.slice(p.hidden, 0, row, row + 1));
            if (previous_streams.ctx != null) {
                var mtp = try m.forwardMtp(previous_streams, tokens[j..][0..1]);
                defer mtp.deinit();
                try save(&mtp.scope, try std.fmt.bufPrint(&path, "{s}/mtp-streams-{d}.npy", .{ output, position + j }), mtp.streams);
                try save(&mtp.scope, try std.fmt.bufPrint(&path, "{s}/mtp-logits-{d}.npy", .{ output, position + j }), mtp.logits);
                try m.commitMtp(&mtp, 1);
            }
            try mx.replace(&previous_streams, try p.scope.slice(p.streams, 0, row, row + 1));
        }
        try m.commit(&p, count);
        position += count;
        for (m.cache, 0..) |cache, i| inline for (comptime std.meta.fieldNames(Cache)) |field| {
            if (@field(cache, field).ctx != null) try save(&p.scope, try std.fmt.bufPrint(&path, "{s}/cache-{d}-{d}-{s}.npy", .{ output, position, i, field }), @field(cache, field));
        };
    }
    std.debug.print("DeepSeek synthetic backbone traces saved through both compression boundaries.\n", .{});
    m.trace_dir = null;
    for ([_]usize{ 2, 6, 18, 126, 130 }) |prefix| try m.checkExact(prefix);
    for ([_]f64{ 0, 0.8 }, 0..) |temperature, test_id| {
        const sampling = @import("sampling.zig").Sampling{ .seed = 1234, .temperature = temperature, .top_k = 20, .top_p = 0.95 };
        var reference: std.ArrayList(u32) = .empty;
        defer reference.deinit(mx.allocator);
        for ([_]usize{ 0, 1, 3, 7, 15 }) |drafts| {
            m.reset();
            var generated = try @import("serial_generation.zig").generate(&m, &.{ 1, 2, 3, 4 }, 12, sampling, drafts, null);
            defer generated.deinit();
            if (drafts == 0) {
                try reference.appendSlice(mx.allocator, generated.tokens.items);
                var s = mx.Scope{};
                defer s.deinit();
                try save(&s, try std.fmt.bufPrint(&path, "{s}/generated-{d}.npy", .{ output, test_id }), try s.data(generated.tokens.items.ptr, &.{@intCast(generated.tokens.items.len)}, c.MLX_UINT32));
            } else try std.testing.expectEqualSlices(u32, reference.items, generated.tokens.items);
        }
    }
    std.debug.print("PASS: DeepSeek greedy/stochastic MTP generation at draft budgets 0, 1, 3, 7 and 15.\n", .{});
}
