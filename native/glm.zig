const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const src = @import("kernel_sources.zig");
const quant = @import("quantization.zig");
const A = mx.Array;
const c = mx.c;
extern "c" fn setenv([*:0]const u8, [*:0]const u8, c_int) c_int;

const Config = struct {
    hidden_size: i32,
    num_hidden_layers: usize,
    vocab_size: i32,
    rms_norm_eps: f32,
    num_attention_heads: i32,
    q_lora_rank: i32,
    kv_lora_rank: i32,
    qk_nope_head_dim: i32,
    v_head_dim: i32,
    index_n_heads: i32,
    index_head_dim: i32,
    index_topk: i32,
    index_kpool: i32 = 4,
    index_kpool_always_select_tail: bool = true,
    n_routed_experts: i32,
    num_experts_per_tok: i32,
    moe_intermediate_size: i32,
    intermediate_size: i32,
    routed_scaling_factor: f32,
    norm_topk_prob: bool = true,
    swiglu_limit: f32 = 0,
    hc_mult: i32 = 4,
    hc_eps: f32 = 1e-6,
    hc_sinkhorn_iters: i32 = 20,
    qk_rope_head_dim: i32 = 0,
    mla_use_nope: bool = true,
    n_group: i32 = 1,
    topk_group: i32 = 1,
    linear_attn_config: struct { num_heads: i32 = 64, head_dim: i32 = 128, short_conv_kernel_size: i32 = 4, gate_lower_bound: f32 = -5 } = .{},
    eos_token_id: []const i32 = &.{},
    fn validate(g: Config) !void {
        if (g.num_hidden_layers < 1 or g.num_hidden_layers > 128 or g.hidden_size < 64 or g.hidden_size > 16384 or @mod(g.hidden_size, 64) != 0 or g.vocab_size < 1 or g.hc_mult != 4 or g.qk_rope_head_dim != 0 or !g.mla_use_nope or g.n_group != 1 or g.topk_group != 1) return error.UnsupportedModelGeometry;
        inline for (.{ "num_attention_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "v_head_dim", "index_n_heads", "index_head_dim", "index_topk", "index_kpool", "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "intermediate_size", "hc_sinkhorn_iters" }) |key| if (@field(g, key) < 1 or @field(g, key) > 65536) return error.UnsupportedModelGeometry;
        if (g.num_experts_per_tok > g.n_routed_experts or g.num_experts_per_tok > 16 or g.index_topk < g.index_kpool or @mod(g.index_topk, g.index_kpool) != 0 or g.kv_lora_rank > 512 or @mod(g.kv_lora_rank, 32) != 0 or !std.math.isFinite(g.rms_norm_eps) or g.rms_norm_eps <= 0 or !std.math.isFinite(g.hc_eps) or g.hc_eps <= 0) return error.UnsupportedModelGeometry;
        const lin = g.linear_attn_config;
        if (g.hc_eps > 1 or g.hc_sinkhorn_iters > 1024 or g.num_attention_heads > 256 or g.index_n_heads > 256 or g.qk_nope_head_dim > 1024 or g.v_head_dim > 1024 or g.index_head_dim > 1024 or !std.math.isFinite(g.routed_scaling_factor) or !std.math.isFinite(g.swiglu_limit) or !std.math.isFinite(lin.gate_lower_bound)) return error.UnsupportedModelGeometry;
        try (@import("large_family_ops.zig").Kda{ .heads = lin.num_heads, .dims = lin.head_dim, .taps = lin.short_conv_kernel_size, .f_bits = 4, .g_bits = 4 }).validate();
    }
};
const Cache = struct {
    conv: A = mx.empty,
    state: A = mx.empty,
    keys: A = mx.empty,
    ik: A = mx.empty,
    ig: A = mx.empty,
    pool: A = mx.empty,
    pub fn deinit(cache: *Cache) void {
        inline for (comptime std.meta.fieldNames(Cache)) |field| mx.free(@field(cache, field));
        cache.* = .{};
    }
    pub fn clone(cache: Cache) !Cache {
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
    mtp_projection: A = mx.empty,
    mtp_input: A = mx.empty,
    is_mtp: bool = false,
    records: [16][]Cache = @splat(&.{}),
    rows: usize,
    position: i32,
    generation: u64,
    pub fn deinit(p: *Pass) void {
        for (p.records[0..p.rows]) |records| mx.allocator.free(records);
        p.scope.deinit();
    }
};
pub const Model = struct {
    weights: cp.Store,
    formats: std.StringHashMap(quant.Spec),
    splits: std.StringHashMap([][]const u8),
    config: std.json.Parsed(Config),
    kernels: mx.Kernels,
    activations: @import("prefill_ops.zig").Ops = .{},
    cache: []Cache,
    position: i32 = 0,
    generation: u64 = 0,
    vocab: i32,
    mtp_cache: Cache = .{},
    mtp_position: i32 = 0,
    mtp_generation: u64 = 0,
    has_mtp: bool = false,
    trace_dir: ?[]const u8 = null,
    pub fn prepareRuntime() !void {
        if (setenv("MLX_ENABLE_TF32", "0", 1) != 0) return error.RuntimeEnvironment;
    }
    pub fn init(io: std.Io, dir: []const u8) !Model {
        var path: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const root = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer root.deinit();
        if (root.value != .object) return error.InvalidModelConfig;
        const parsed = try std.json.parseFromValue(Config, mx.allocator, root.value.object.get("text_config") orelse root.value, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
        errdefer parsed.deinit();
        try parsed.value.validate();
        const caches = try mx.allocator.alloc(Cache, parsed.value.num_hidden_layers);
        @memset(caches, .{});
        var m = Model{ .weights = cp.Store.init(64), .formats = std.StringHashMap(quant.Spec).init(mx.allocator), .splits = std.StringHashMap([][]const u8).init(mx.allocator), .config = parsed, .kernels = mx.Kernels.init(), .cache = caches, .vocab = parsed.value.vocab_size };
        // parsed is owned here; clean the remaining members on an initialization error.
        errdefer {
            m.reset();
            mx.allocator.free(m.cache);
            m.weights.deinit();
            var it = m.formats.keyIterator();
            while (it.next()) |key| mx.allocator.free(key.*);
            m.formats.deinit();
            m.freeSplits();
            m.kernels.deinit();
        }
        var raw = cp.Store.init(64);
        defer raw.deinit();
        try raw.load(io, dir, "");
        var entries = raw.arrays.iterator();
        while (entries.next()) |entry| {
            const raw_name = entry.key_ptr.*;
            const short = (try canonical(raw_name, parsed.value.num_hidden_layers)) orelse continue;
            defer mx.allocator.free(short);
            if (m.weights.has(short)) return error.DuplicateWeight;
            try m.weights.put(short, entry.value_ptr.*);
            if (std.mem.endsWith(u8, raw_name, ".weight")) {
                const stem = raw_name[0 .. raw_name.len - 7];
                if (raw.has(try std.fmt.bufPrint(&path, "{s}.scales", .{stem}))) {
                    const spec = (try quant.resolve(root.value, stem)) orelse return error.UnsupportedQuantization;
                    try m.formatPut(short[0 .. short.len - 7], spec);
                }
            }
        }
        for (0..m.cache.len) |i| try m.prepare(i);
        var buf: [256]u8 = undefined;
        m.has_mtp = m.weights.has(try name(&buf, m.cache.len, "eh_proj.weight"));
        if (m.has_mtp) try m.prepare(m.cache.len);
        return m;
    }
    pub fn deinit(m: *Model) void {
        m.reset();
        mx.allocator.free(m.cache);
        m.weights.deinit();
        var it = m.formats.keyIterator();
        while (it.next()) |key| mx.allocator.free(key.*);
        m.formats.deinit();
        m.freeSplits();
        m.config.deinit();
        m.kernels.deinit();
        m.activations.deinit();
    }
    pub fn reset(m: *Model) void {
        for (m.cache) |*cache| cache.deinit();
        m.position = 0;
        m.generation +%= 1;
        m.mtp_cache.deinit();
        m.mtp_position = 0;
        m.mtp_generation +%= 1;
    }
    pub fn isEos(m: *Model, id: i32) bool {
        for (m.config.value.eos_token_id) |token| if (token == id) return true;
        return false;
    }
    fn formatPut(m: *Model, key_name: []const u8, spec: quant.Spec) !void {
        if (m.formats.getPtr(key_name)) |old| {
            old.* = spec;
            return;
        }
        const key = try mx.allocator.dupe(u8, key_name);
        errdefer mx.allocator.free(key);
        try m.formats.put(key, spec);
    }
    fn freeSplits(m: *Model) void {
        var it = m.splits.iterator();
        while (it.next()) |entry| {
            mx.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.*) |part| mx.allocator.free(part);
            mx.allocator.free(entry.value_ptr.*);
        }
        m.splits.deinit();
    }
    fn format(m: *Model, key_name: []const u8) !quant.Spec {
        return m.formats.get(key_name) orelse error.UnsupportedQuantization;
    }
    fn name(buf: []u8, i: usize, suffix: []const u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i, suffix });
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
    fn stack(m: *Model, s: *mx.Scope, i: usize, dest: []const u8, members: []const []const u8, axis: i32) !void {
        var b: [256]u8 = undefined;
        const fmt = try m.format(try name(&b, i, members[0]));
        var mixed = false;
        for (members) |member| if (!std.meta.eql(fmt, try m.format(try name(&b, i, member)))) {
            mixed = true;
        };
        if (mixed) {
            if (axis != 0) return error.MixedExpertQuantization;
            const parts = try mx.allocator.alloc([]const u8, members.len);
            var count: usize = 0;
            errdefer {
                for (parts[0..count]) |part| mx.allocator.free(part);
                mx.allocator.free(parts);
            }
            for (members, parts) |member, *part| {
                part.* = try mx.allocator.dupe(u8, try name(&b, i, member));
                count += 1;
            }
            const key = try mx.allocator.dupe(u8, try name(&b, i, dest));
            errdefer mx.allocator.free(key);
            try m.splits.put(key, parts);
            return;
        }
        for ([_][]const u8{ "weight", "scales", "biases" }) |field| {
            const values = try mx.allocator.alloc(A, members.len);
            defer mx.allocator.free(values);
            for (members, values) |member, *value| value.* = try m.weight(i, try std.fmt.bufPrint(&b, "{s}.{s}", .{ member, field }));
            const joined = if (axis == 0) try s.cat(values, 0) else try s.stack(values, 0);
            try mx.eval(joined);
            try m.put(i, try std.fmt.bufPrint(&b, "{s}.{s}", .{ dest, field }), joined);
        }
        try m.formatPut(try name(&b, i, dest), fmt);
    }
    fn prepare(m: *Model, i: usize) !void {
        const g = m.config.value;
        var s = mx.Scope{};
        defer s.deinit();
        var b: [256]u8 = undefined;
        if (m.weights.has(try name(&b, i, "self_attn.q_a_proj.weight"))) {
            try m.stack(&s, i, "self_attn.x_proj", &.{ "self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.indexer.wk", "self_attn.indexer.weights_proj" }, 0);
            try m.stack(&s, i, "self_attn.qr_proj", &.{ "self_attn.q_b_proj", "self_attn.indexer.wq_b" }, 0);
            if (m.weights.has(try name(&b, i, "self_attn.kv_b_proj.weight"))) {
                const fmt = try m.format(try name(&b, i, "self_attn.kv_b_proj"));
                for ([_][]const u8{ "weight", "scales", "biases" }) |field| {
                    const v = try s.reshape(try m.weight(i, try std.fmt.bufPrint(&b, "self_attn.kv_b_proj.{s}", .{field})), &.{ g.num_attention_heads, g.qk_nope_head_dim + g.v_head_dim, -1 });
                    try m.put(i, try std.fmt.bufPrint(&b, "self_attn.wk.{s}", .{field}), try s.contiguous(try s.slice(v, 1, 0, g.qk_nope_head_dim)));
                    try m.put(i, try std.fmt.bufPrint(&b, "self_attn.wv.{s}", .{field}), try s.contiguous(try s.slice(v, 1, g.qk_nope_head_dim, g.qk_nope_head_dim + g.v_head_dim)));
                }
                try m.formatPut(try name(&b, i, "self_attn.wk"), fmt);
                try m.formatPut(try name(&b, i, "self_attn.wv"), fmt);
            } else if (!m.weights.has(try name(&b, i, "self_attn.embed_q.weight")) or !m.weights.has(try name(&b, i, "self_attn.unembed_out.weight"))) return error.UnsupportedAbsorbedLayout;
        } else {
            try m.stack(&s, i, "self_attn.in_proj", &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.f_a_proj", "self_attn.g_a_proj", "self_attn.b_proj" }, 0);
            const lin = g.linear_attn_config;
            const joined = if (m.weights.has(try name(&b, i, "self_attn.conv1d.weight"))) try s.reshape(try m.weight(i, "self_attn.conv1d.weight"), &.{ 3 * lin.num_heads * lin.head_dim, lin.short_conv_kernel_size }) else blk: {
                var conv: [3]A = undefined;
                for ([_]u8{ 'q', 'k', 'v' }, 0..) |part, j| conv[j] = try s.reshape(try m.weight(i, try std.fmt.bufPrint(&b, "self_attn.{c}_conv1d.weight", .{part})), &.{ lin.num_heads * lin.head_dim, lin.short_conv_kernel_size });
                break :blk try s.cat(&conv, 0);
            };
            try m.put(i, "self_attn.conv", try s.contiguous(try s.cast(try s.transpose(joined, &.{ 1, 0 }), mx.f32t)));
            try m.put(i, "self_attn.A", try s.unary(c.mlx_exp, try s.cast(try m.weight(i, "self_attn.A_log"), mx.f32t)));
            try m.put(i, "self_attn.dt", try s.reshape(try s.cast(try m.weight(i, "self_attn.dt_bias"), mx.f32t), &.{-1}));
            try m.put(i, "self_attn.onorm", try s.cast(try m.weight(i, "self_attn.o_norm.weight"), mx.f32t));
        }
        if (m.weights.has(try name(&b, i, "mlp.gate.weight"))) {
            try m.put(i, "mlp.router", try s.contiguous(try s.transpose(try s.cast(try m.weight(i, "mlp.gate.weight"), mx.f32t), &.{ 1, 0 })));
            for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
                const target = try std.fmt.allocPrint(mx.allocator, "mlp.switch_mlp.{s}", .{proj});
                defer mx.allocator.free(target);
                var suffix: [256]u8 = undefined;
                if (m.weights.has(try name(&b, i, try std.fmt.bufPrint(&suffix, "{s}.weight", .{target})))) continue;
                const parts = try mx.allocator.alloc([]const u8, @intCast(g.n_routed_experts));
                defer mx.allocator.free(parts);
                var n: usize = 0;
                defer for (parts[0..n]) |part| mx.allocator.free(part);
                for (parts, 0..) |*part, j| {
                    part.* = try std.fmt.allocPrint(mx.allocator, "mlp.experts.{d}.{s}", .{ j, proj });
                    n += 1;
                }
                try m.stack(&s, i, target, parts, 1);
            }
            if (m.weights.has(try name(&b, i, "mlp.shared_experts.gate_proj.weight"))) try m.stack(&s, i, "mlp.shared_experts.gate_up", &.{ "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj" }, 0);
        } else try m.stack(&s, i, "mlp.gate_up", &.{ "mlp.gate_proj", "mlp.up_proj" }, 0);
    }
    fn qmm(m: *Model, s: *mx.Scope, x: A, key: []const u8, transpose: bool, ids: ?A) !A {
        if (m.splits.get(key)) |parts| {
            const outputs = try mx.allocator.alloc(A, parts.len);
            defer mx.allocator.free(outputs);
            for (parts, outputs) |part, *out| out.* = try m.qmmOne(s, x, part, transpose, ids);
            return s.cat(outputs, @intCast(mx.shape(outputs[0]).len - 1));
        }
        return m.qmmOne(s, x, key, transpose, ids);
    }
    fn qmmOne(m: *Model, s: *mx.Scope, x: A, key: []const u8, transpose: bool, ids: ?A) !A {
        if (!m.formats.contains(key)) {
            if (ids != null) return error.UnsupportedDenseExperts;
            const w = try m.weights.field(key, "weight");
            return s.binary(c.mlx_matmul, x, if (transpose) try s.transpose(w, &.{ 1, 0 }) else w);
        }
        const t = try m.weights.triple(key);
        const f = try m.format(key);
        var out = c.mlx_array_new();
        const rc = if (ids) |ix| c.mlx_gather_qmm(&out, x, t[0], t[1], t[2], mx.empty, ix, transpose, mx.opt(f.group_size), mx.opt(f.bits), "affine", false, mx.stream) else c.mlx_quantized_matmul(&out, x, t[0], t[1], t[2], transpose, mx.opt(f.group_size), mx.opt(f.bits), "affine", mx.stream);
        return s.result(rc, out);
    }
    fn project(m: *Model, s: *mx.Scope, i: usize, key: []const u8, x: A) !A {
        var b: [256]u8 = undefined;
        return m.qmm(s, x, try name(&b, i, key), true, null);
    }
    fn hc(m: *Model, s: *mx.Scope, i: usize, kind: []const u8, x: A) ![3]A {
        var b: [256]u8 = undefined;
        const g = m.config.value;
        if (mx.shape(x).len != 3 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != 4 or mx.dim(x, 2) != g.hidden_size) return error.InvalidTensorShape;
        const rows = mx.dim(x, 0);
        if (rows < 1 or rows > 2048) return error.InvalidTensorShape;
        const z = try cp.norm(s, try s.reshape(try s.cast(x, mx.f32t), &.{ rows, 4 * g.hidden_size }), mx.empty, g.rms_norm_eps);
        const fnw = try s.cast(try m.weight(i, try std.fmt.bufPrint(&b, "hc_{s}_fn", .{kind})), mx.f32t);
        if (!std.mem.eql(i32, mx.shape(fnw), &.{ 24, 4 * g.hidden_size })) return error.InvalidTensorShape;
        const scale = try s.cast(try m.weight(i, try std.fmt.bufPrint(&b, "hc_{s}_scale", .{kind})), mx.f32t);
        const base = try s.cast(try m.weight(i, try std.fmt.bufPrint(&b, "hc_{s}_base", .{kind})), mx.f32t);
        if (c.mlx_array_size(scale) != 3 or c.mlx_array_size(base) != 24) return error.InvalidTensorShape;
        const mixes = try s.binary(c.mlx_matmul, z, try s.transpose(fnw, &.{ 1, 0 }));
        const result = try m.kernels.run(s, src.glm_hc_split, &.{ x, mixes, scale, base }, &.{ mx.td("T", mx.bf16), mx.ti("HC", 4), mx.ti("ITERS", g.hc_sinkhorn_iters), mx.ti("D", g.hidden_size), mx.ti("EPS_INT", @intFromFloat(@round(g.hc_eps / 1e-9))) }, .{ 256 * rows, 1, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ rows, g.hidden_size } }, .{ .shape = &.{ rows, 4 }, .dtype = mx.f32t }, .{ .shape = &.{ rows, 4, 4 }, .dtype = mx.f32t } });
        return .{ result[0], result[1], result[2] };
    }
    fn expand(s: *mx.Scope, x: A, branch: A, post: A, comb: A) !A {
        const rows = mx.dim(x, 0);
        const y = try s.binary(c.mlx_multiply, try s.reshape(post, &.{ rows, 4, 1 }), try s.reshape(try s.cast(branch, mx.f32t), &.{ rows, 1, -1 }));
        return s.cast(try s.binary(c.mlx_add, y, try s.binary(c.mlx_matmul, try s.transpose(comb, &.{ 0, 2, 1 }), try s.cast(x, mx.f32t))), mx.bf16);
    }
    fn kda(m: *Model, s: *mx.Scope, i: usize, x: A, cache: *Cache) !A {
        if (mx.dim(x, 0) > 16) return m.kdaFallback(s, i, x, cache);
        const g = m.config.value.linear_attn_config;
        const width = g.num_heads * g.head_dim;
        var b: [256]u8 = undefined;
        const f = try m.format(try name(&b, i, "self_attn.f_b_proj"));
        const v = try m.format(try name(&b, i, "self_attn.g_b_proj"));
        if (f.group_size != 64 or v.group_size != 64 or (f.bits != 4 and f.bits != 8) or (v.bits != 4 and v.bits != 8)) return m.kdaFallback(s, i, x, cache);
        const fw = try m.triple(i, "self_attn.f_b_proj");
        const gw = try m.triple(i, "self_attn.g_b_proj");
        const result = try @import("large_family_ops.zig").kda(&m.kernels, s, .{ .heads = g.num_heads, .dims = g.head_dim, .taps = g.short_conv_kernel_size, .f_bits = f.bits, .g_bits = v.bits }, .{ try m.project(s, i, "self_attn.in_proj", x), if (cache.conv.ctx != null) cache.conv else try s.zeros(&.{ g.short_conv_kernel_size - 1, 3 * width }, mx.bf16), try m.weight(i, "self_attn.conv"), fw[0], fw[1], fw[2], gw[0], gw[1], gw[2], try m.weight(i, "self_attn.A"), try m.weight(i, "self_attn.dt"), if (cache.state.ctx != null) cache.state else try s.zeros(&.{ 1, g.num_heads, g.head_dim, g.head_dim }, mx.f32t), try m.weight(i, "self_attn.onorm"), try s.scalar(g.gate_lower_bound), try s.scalar(m.config.value.rms_norm_eps) });
        cache.state = result[1];
        cache.conv = result[2];
        return m.project(s, i, "self_attn.o_proj", result[0]);
    }
    fn kdaFallback(m: *Model, s: *mx.Scope, i: usize, x: A, cache: *Cache) !A {
        return (try m.kdaPrefill(s, i, x, cache)).output;
    }
    const KdaResult = struct { output: A, projection: A, convolved: A, q: A, k: A, value: A, decay: A, beta: A, recurrent: A, gated: A };
    fn kdaPrefill(m: *Model, s: *mx.Scope, i: usize, x: A, cache: *Cache) !KdaResult {
        const g = m.config.value.linear_attn_config;
        const h = g.num_heads;
        const d = g.head_dim;
        const width = h * d;
        if (mx.shape(x).len != 2 or mx.dtype(x) != mx.bf16 or mx.dim(x, 1) != m.config.value.hidden_size) return error.InvalidTensorShape;
        const rows = mx.dim(x, 0);
        if (rows < 1 or rows > 2048) return error.InvalidTensorShape;
        if (cache.conv.ctx != null and (!std.mem.eql(i32, mx.shape(cache.conv), &.{ g.short_conv_kernel_size - 1, 3 * width }) or mx.dtype(cache.conv) != mx.bf16)) return error.InvalidTensorShape;
        if (cache.state.ctx != null and (!std.mem.eql(i32, mx.shape(cache.state), &.{ 1, h, d, d }) or mx.dtype(cache.state) != mx.f32t)) return error.InvalidTensorShape;
        const proj = try m.project(s, i, "self_attn.in_proj", x);
        const mixed = try s.slice(proj, 1, 0, 3 * width);
        const conv = if (cache.conv.ctx != null) cache.conv else try s.zeros(&.{ g.short_conv_kernel_size - 1, 3 * width }, mx.bf16);
        const ci = try s.cat(&.{ conv, mixed }, 0);
        const cw = try m.weight(i, "self_attn.conv");
        var acc = try s.binary(c.mlx_multiply, try s.cast(try s.slice(ci, 0, 0, rows), mx.f32t), try s.slice(cw, 0, 0, 1));
        var t: i32 = 1;
        while (t < g.short_conv_kernel_size) : (t += 1) acc = try s.binary(c.mlx_add, acc, try s.binary(c.mlx_multiply, try s.cast(try s.slice(ci, 0, t, t + rows), mx.f32t), try s.slice(cw, 0, t, t + 1)));
        const co = try m.activations.call(s, .silu, &.{try s.cast(acc, mx.bf16)});
        const q0 = try s.reshape(try s.slice(co, 1, 0, width), &.{ 1, rows, h, d });
        const k0 = try s.reshape(try s.slice(co, 1, width, 2 * width), &.{ 1, rows, h, d });
        const value = try s.reshape(try s.slice(co, 1, 2 * width, 3 * width), &.{ 1, rows, h, d });
        const df: f32 = @floatFromInt(d);
        const q = try s.cast(try s.binary(c.mlx_multiply, try cp.norm(s, try s.cast(q0, mx.f32t), mx.empty, 1e-6 / df), try s.scalar(1 / df)), mx.bf16);
        const k = try s.cast(try s.binary(c.mlx_multiply, try cp.norm(s, try s.cast(k0, mx.f32t), mx.empty, 1e-6 / df), try s.scalar(1 / @sqrt(df))), mx.bf16);
        const a = try s.reshape(try s.cast(try m.project(s, i, "self_attn.f_b_proj", try s.slice(proj, 1, 3 * width, 3 * width + d)), mx.f32t), &.{ 1, rows, h, d });
        const decay = try s.unary(c.mlx_exp, try s.binary(c.mlx_multiply, try s.scalar(g.gate_lower_bound), try s.unary(c.mlx_sigmoid, try s.binary(c.mlx_multiply, try s.reshape(try m.weight(i, "self_attn.A"), &.{ h, 1 }), try s.binary(c.mlx_add, a, try s.reshape(try m.weight(i, "self_attn.dt"), &.{ h, d }))))));
        const beta = try s.reshape(try s.unary(c.mlx_sigmoid, try s.slice(proj, 1, 3 * width + 2 * d, 3 * width + 2 * d + h)), &.{ 1, rows, h });
        const entry = if (cache.state.ctx != null) cache.state else try s.zeros(&.{ 1, h, d, d }, mx.f32t);
        const outputs = try m.kernels.run(s, src.glm_gated_delta, &.{ q, k, value, decay, beta, entry, try s.reshape(try s.ints(&.{rows}), &.{}) }, &.{ mx.td("InT", mx.bf16), mx.td("StT", mx.f32t), mx.ti("Dk", d), mx.ti("Dv", d), mx.ti("Hk", h), mx.ti("Hv", h) }, .{ 32, d, h }, .{ 32, 4, 1 }, &.{ .{ .shape = &.{ 1, rows, h, d } }, .{ .shape = &.{ 1, h, d, d }, .dtype = mx.f32t } });
        const next_conv = try s.contiguous(try s.slice(ci, 0, rows, rows + g.short_conv_kernel_size - 1));
        const gate = try s.reshape(try s.cast(try m.project(s, i, "self_attn.g_b_proj", try s.slice(proj, 1, 3 * width + d, 3 * width + 2 * d)), mx.f32t), &.{ rows, h, d });
        const normed = try cp.norm(s, try s.reshape(try s.cast(outputs[0], mx.f32t), &.{ rows, h, d }), try m.weight(i, "self_attn.onorm"), m.config.value.rms_norm_eps);
        const result = try s.reshape(try s.cast(try s.binary(c.mlx_multiply, normed, try s.unary(c.mlx_sigmoid, gate)), mx.bf16), &.{ rows, width });
        const output = try m.project(s, i, "self_attn.o_proj", result);
        cache.state = outputs[1];
        cache.conv = next_conv;
        return .{ .output = output, .projection = proj, .convolved = co, .q = q, .k = k, .value = value, .decay = decay, .beta = beta, .recurrent = outputs[0], .gated = result };
    }
    fn append(s: *mx.Scope, old: A, value: A) !A {
        return if (old.ctx == null) value else s.cat(&.{ old, value }, 0);
    }
    fn mla(m: *Model, s: *mx.Scope, i: usize, x: A, cache: *Cache, position: i32) !A {
        const g = m.config.value;
        var b: [256]u8 = undefined;
        const xp = try m.project(s, i, "self_attn.x_proj", x);
        const a = g.q_lora_rank;
        const b1 = a + g.kv_lora_rank;
        const b2 = b1 + g.index_head_dim;
        const qr = try cp.norm(s, try s.slice(xp, 1, 0, a), try m.weight(i, "self_attn.q_a_layernorm.weight"), g.rms_norm_eps);
        const qp = try m.project(s, i, "self_attn.qr_proj", qr);
        const qn = g.num_attention_heads * g.qk_nope_head_dim;
        const q = try s.reshape(try s.slice(qp, 1, 0, qn), &.{ g.num_attention_heads, 1, g.qk_nope_head_dim });
        const iq = try s.reshape(try s.slice(qp, 1, qn, qn + g.index_n_heads * g.index_head_dim), &.{ 1, g.index_n_heads, g.index_head_dim });
        const lat = try cp.norm(s, try s.slice(xp, 1, a, b1), try m.weight(i, "self_attn.kv_a_layernorm.weight"), g.rms_norm_eps);
        var ln = c.mlx_array_new();
        const rc = c.mlx_fast_layer_norm(&ln, try s.slice(xp, 1, b1, b2), try m.weight(i, "self_attn.indexer.k_norm.weight"), try m.weight(i, "self_attn.indexer.k_norm.bias"), 1e-6, mx.stream);
        const ik = try s.result(rc, ln);
        const ig = try s.binary(c.mlx_matmul, x, try s.transpose(try m.weight(i, "self_attn.indexer.index_kpool_compress_gate"), &.{ 1, 0 }));
        const isc: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(g.index_n_heads))) / @sqrt(@as(f64, @floatFromInt(g.index_head_dim))));
        const iw = try s.binary(c.mlx_multiply, try s.slice(xp, 1, b2, b2 + g.index_n_heads), try s.cast(try s.scalar(isc), mx.bf16));
        cache.keys = try append(s, cache.keys, lat);
        cache.ik = try append(s, cache.ik, ik);
        cache.ig = try append(s, cache.ig, ig);
        const length = position + 1;
        const kp = g.index_kpool;
        if (@mod(length, kp) == 0) {
            const keys = try s.cast(try s.slice(cache.ik, 0, length - kp, length), mx.f32t);
            const logits = try s.binary(c.mlx_add, try s.cast(try s.slice(cache.ig, 0, length - kp, length), mx.f32t), try s.cast(try m.weight(i, "self_attn.indexer.index_kpool_compress_ape"), mx.f32t));
            var top = try s.slice(logits, 0, 0, 1);
            var j: i32 = 1;
            while (j < kp) : (j += 1) top = try s.binary(c.mlx_maximum, top, try s.slice(logits, 0, j, j + 1));
            const exps = try s.unary(c.mlx_exp, try s.binary(c.mlx_subtract, logits, top));
            var total = try s.slice(exps, 0, 0, 1);
            j = 1;
            while (j < kp) : (j += 1) total = try s.binary(c.mlx_add, total, try s.slice(exps, 0, j, j + 1));
            var pooled = try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, try s.slice(exps, 0, 0, 1), total), try s.slice(keys, 0, 0, 1));
            j = 1;
            while (j < kp) : (j += 1) pooled = try s.binary(c.mlx_add, pooled, try s.binary(c.mlx_multiply, try s.binary(c.mlx_divide, try s.slice(exps, 0, j, j + 1), total), try s.slice(keys, 0, j, j + 1)));
            cache.pool = try append(s, cache.pool, try s.cast(pooled, mx.bf16));
        }
        const absorbed = m.weights.has(try name(&b, i, "self_attn.embed_q.weight"));
        const ql = try m.qmm(s, q, try name(&b, i, if (absorbed) "self_attn.embed_q" else "self_attn.wk"), absorbed, null);
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(g.qk_nope_head_dim))));
        const attended = if (length > g.index_topk) blk: {
            const scores = try s.binary(c.mlx_matmul, iq, try s.transpose(cache.pool, &.{ 1, 0 }));
            const weighted = try s.binary(c.mlx_multiply, try s.reshape(iw, &.{ 1, g.index_n_heads, 1 }), try s.binary(c.mlx_maximum, scores, try s.cast(try s.scalar(0), mx.bf16)));
            var sum = c.mlx_array_new();
            const code = c.mlx_sum_axis(&sum, weighted, 1, false, mx.stream);
            const counts = try s.result(code, sum);
            const topn = @divTrunc(g.index_topk, kp);
            const picks = try partition(s, try s.unary(c.mlx_negative, counts), topn);
            const range = try rangeInts(s, kp);
            const starts = try s.binary(c.mlx_multiply, try s.reshape(picks, &.{ topn, 1 }), try s.ints(&.{kp}));
            var ids = try s.cast(try s.reshape(try s.binary(c.mlx_add, starts, try s.reshape(range, &.{ 1, kp })), &.{ 1, -1 }), mx.i32t);
            const tail = @mod(length, kp);
            if (g.index_kpool_always_select_tail and tail > 0) ids = try s.cat(&.{ ids, try s.reshape(try s.binary(c.mlx_add, try rangeInts(s, tail), try s.ints(&.{length - tail})), &.{ 1, tail }) }, 1);
            break :blk try s.reshape(try @import("large_family_ops.zig").indexedAttention(&m.kernels, s, try s.reshape(ql, &.{ 1, g.num_attention_heads, g.kv_lora_rank }), cache.keys, ids, length, scale), &.{ g.num_attention_heads, 1, g.kv_lora_rank });
        } else blk: {
            var out = c.mlx_array_new();
            const keys = try s.reshape(cache.keys, &.{ 1, 1, length, g.kv_lora_rank });
            const code = c.mlx_fast_scaled_dot_product_attention(&out, try s.reshape(ql, &.{ 1, g.num_attention_heads, 1, g.kv_lora_rank }), keys, keys, scale, "", mx.empty, mx.empty, false, mx.stream);
            break :blk try s.reshape(try s.result(code, out), &.{ g.num_attention_heads, 1, g.kv_lora_rank });
        };
        const value = try m.qmm(s, attended, try name(&b, i, if (absorbed) "self_attn.unembed_out" else "self_attn.wv"), true, null);
        return m.project(s, i, "self_attn.o_proj", try s.reshape(value, &.{ 1, g.num_attention_heads * g.v_head_dim }));
    }
    fn activation(m: *Model, s: *mx.Scope, gate: A, up: A) !A {
        const limit = m.config.value.swiglu_limit;
        const g = if (limit > 0) try s.binary(c.mlx_minimum, gate, try s.cast(try s.scalar(limit), mx.dtype(gate))) else gate;
        const u = if (limit > 0) try s.binary(c.mlx_maximum, try s.binary(c.mlx_minimum, up, try s.cast(try s.scalar(limit), mx.dtype(up))), try s.cast(try s.scalar(-limit), mx.dtype(up))) else up;
        return s.binary(c.mlx_multiply, try m.activations.call(s, .silu, &.{g}), u);
    }
    fn dense(m: *Model, s: *mx.Scope, i: usize, prefix: []const u8, x: A) !A {
        var b: [256]u8 = undefined;
        const gu = try m.project(s, i, try std.fmt.bufPrint(&b, "{s}.gate_up", .{prefix}), x);
        const width = @divExact(mx.dim(gu, 1), 2);
        return m.project(s, i, try std.fmt.bufPrint(&b, "{s}.down_proj", .{prefix}), try m.activation(s, try s.slice(gu, 1, 0, width), try s.slice(gu, 1, width, 2 * width)));
    }
    fn mlp(m: *Model, s: *mx.Scope, i: usize, x: A) !A {
        var b: [256]u8 = undefined;
        const g = m.config.value;
        if (!m.weights.has(try name(&b, i, "mlp.gate.weight"))) return m.dense(s, i, "mlp", x);
        const scores = try s.unary(c.mlx_sigmoid, try s.binary(c.mlx_matmul, try s.cast(x, mx.f32t), try m.weight(i, "mlp.router")));
        const ids = try partition(s, try s.unary(c.mlx_negative, try s.binary(c.mlx_add, scores, try s.cast(try m.weight(i, "mlp.gate.e_score_correction_bias"), mx.f32t))), g.num_experts_per_tok);
        var ww = c.mlx_array_new();
        const rc = c.mlx_take_along_axis(&ww, scores, ids, -1, mx.stream);
        var w = try s.result(rc, ww);
        if (g.norm_topk_prob and g.num_experts_per_tok > 1) {
            var sum = c.mlx_array_new();
            const code = c.mlx_sum_axis(&sum, w, -1, true, mx.stream);
            w = try s.binary(c.mlx_divide, w, try s.result(code, sum));
        }
        w = try s.binary(c.mlx_multiply, w, try s.scalar(g.routed_scaling_factor));
        const input = try s.reshape(x, &.{ 1, 1, 1, g.hidden_size });
        const gate = try m.qmm(s, input, try name(&b, i, "mlp.switch_mlp.gate_proj"), true, ids);
        const up = try m.qmm(s, input, try name(&b, i, "mlp.switch_mlp.up_proj"), true, ids);
        const y = try s.reshape(try s.cast(try m.qmm(s, try m.activation(s, gate, up), try name(&b, i, "mlp.switch_mlp.down_proj"), true, ids), mx.f32t), &.{ g.num_experts_per_tok, g.hidden_size });
        var total = try s.binary(c.mlx_multiply, try s.slice(w, 1, 0, 1), try s.slice(y, 0, 0, 1));
        var j: i32 = 1;
        while (j < g.num_experts_per_tok) : (j += 1) total = try s.binary(c.mlx_add, total, try s.binary(c.mlx_multiply, try s.slice(w, 1, j, j + 1), try s.slice(y, 0, j, j + 1)));
        const out = try s.cast(total, mx.bf16);
        return if (m.weights.has(try name(&b, i, "mlp.shared_experts.gate_proj.weight"))) s.binary(c.mlx_add, out, try m.dense(s, i, "mlp.shared_experts", x)) else out;
    }
    fn embedding(m: *Model, s: *mx.Scope, token: i32) !A {
        const embed = try m.weights.triple("embed_tokens");
        const fmt = try m.format("embed_tokens");
        const ids = try s.ints(&.{token});
        var e = c.mlx_array_new();
        const rc = c.mlx_dequantize(&e, try s.take(embed[0], ids, 0), try s.take(embed[1], ids, 0), try s.take(embed[2], ids, 0), mx.opt(fmt.group_size), mx.opt(fmt.bits), "affine", mx.empty, .{ .value = mx.bf16, .has_value = true }, mx.stream);
        return s.result(rc, e);
    }
    pub fn forwardMtp(m: *Model, hidden: A, tokens: []const i32) !Pass {
        return m.forwardMtpAt(hidden, tokens, m.mtp_cache, m.mtp_position);
    }
    fn forwardMtpAt(m: *Model, hidden: A, tokens: []const i32, entry: Cache, position: i32) !Pass {
        if (!m.has_mtp) return error.MissingDraftHead;
        if (tokens.len == 0 or tokens.len > 16 or position > 1048576 - tokens.len) return error.ContextLimitExceeded;
        if (!std.mem.eql(i32, mx.shape(hidden), &.{ @intCast(tokens.len), m.config.value.hidden_size }) or mx.dtype(hidden) != mx.bf16) return error.InvalidTensorShape;
        for (tokens) |token| if (token < 0 or token >= m.vocab) return error.InvalidToken;
        var p = Pass{ .position = position, .generation = m.mtp_generation, .rows = tokens.len, .is_mtp = true };
        errdefer p.deinit();
        const s = &p.scope;
        var cache = entry;
        const i = m.cache.len;
        const eps = m.config.value.rms_norm_eps;
        var logits: [16]A = undefined;
        var states: [16]A = undefined;
        var projected: [16]A = undefined;
        var joined: [16]A = undefined;
        for (tokens, 0..) |token, row| {
            const e = try cp.norm(s, try m.embedding(s, token), try m.weight(i, "enorm.weight"), eps);
            const h = try cp.norm(s, try s.slice(hidden, 0, @intCast(row), @intCast(row + 1)), try m.weight(i, "hnorm.weight"), eps);
            joined[row] = try s.cat(&.{ e, h }, 1);
            var x = try m.project(s, i, "eh_proj", joined[row]);
            projected[row] = x;
            x = try s.binary(c.mlx_add, x, try m.mla(s, i, try cp.norm(s, x, try m.weight(i, "input_layernorm.weight"), eps), &cache, position + @as(i32, @intCast(row))));
            x = try s.binary(c.mlx_add, x, try m.mlp(s, i, try cp.norm(s, x, try m.weight(i, "post_attention_layernorm.weight"), eps)));
            states[row] = x;
            logits[row] = try m.qmm(s, try cp.norm(s, x, try m.weight(i, "shared_head.norm.weight"), eps), "lm_head", true, null);
            try mx.eval(logits[row]);
            p.records[row] = try mx.allocator.dupe(Cache, &.{cache});
        }
        p.hidden = try s.cat(states[0..tokens.len], 0);
        p.mtp_projection = try s.cat(projected[0..tokens.len], 0);
        p.mtp_input = try s.cat(joined[0..tokens.len], 0);
        p.logits = try s.cat(logits[0..tokens.len], 0);
        return p;
    }
    pub fn propose(m: *Model, hidden: A, first: i32, tokens: []i32, settings: @import("sampling.zig").Sampling) !void {
        if (tokens.len < 1 or tokens.len > 16 or m.mtp_position != m.position - 1) return error.InvalidDraftState;
        tokens[0] = first;
        var cache = try m.mtp_cache.clone();
        defer cache.deinit();
        var row = try mx.retain(hidden);
        defer mx.free(row);
        for (tokens[1..], 0..) |*token, j| {
            var pass = try m.forwardMtpAt(row, tokens[j..][0..1], cache, m.mtp_position + @as(i32, @intCast(j)));
            defer pass.deinit();
            const sampled = try @import("sampling.zig").rows(&m.kernels, &pass.scope, pass.logits, &.{m.position + @as(i32, @intCast(j)) + 1}, settings);
            defer mx.allocator.free(sampled);
            token.* = sampled[0];
            const next_cache = try pass.records[0][0].clone();
            cache.deinit();
            cache = next_cache;
            const next_row = try mx.retain(pass.hidden);
            mx.free(row);
            row = next_row;
        }
    }
    pub fn commitMtp(m: *Model, p: *Pass, keep: usize) !void {
        if (!p.is_mtp or keep == 0 or keep > p.rows or p.position != m.mtp_position or p.generation != m.mtp_generation or p.records[keep - 1].len != 1) return error.InvalidCommit;
        const next = try p.records[keep - 1][0].clone();
        m.mtp_cache.deinit();
        m.mtp_cache = next;
        m.mtp_position += @intCast(keep);
        m.mtp_generation +%= 1;
    }
    pub fn forward(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len == 0 or tokens.len > 16 or m.position < 0 or m.position > 1048576 - tokens.len) return error.ContextLimitExceeded;
        for (tokens) |token| if (token < 0 or token >= m.vocab) return error.InvalidToken;
        var p = Pass{ .position = m.position, .generation = m.generation, .rows = tokens.len };
        errdefer p.deinit();
        const s = &p.scope;
        const work = try mx.allocator.dupe(Cache, m.cache);
        defer mx.allocator.free(work);
        var logits: [16]A = undefined;
        var hidden: [16]A = undefined;
        const g = m.config.value;
        for (tokens, 0..) |token, row| {
            const embedded = try m.embedding(s, token);
            var x = try s.reshape(try s.cat(&.{ embedded, embedded, embedded, embedded }, 0), &.{ 1, 4, g.hidden_size });
            for (work, 0..) |*cache, i| {
                const attn = try m.hc(s, i, "attn", x);
                const normed = try cp.norm(s, attn[0], try m.weight(i, "input_layernorm.weight"), g.rms_norm_eps);
                var b: [256]u8 = undefined;
                const a = if (m.weights.has(try name(&b, i, "self_attn.q_a_proj.weight"))) try m.mla(s, i, normed, cache, m.position + @as(i32, @intCast(row))) else try m.kda(s, i, normed, cache);
                const position = m.position + @as(i32, @intCast(row));
                try m.trace(s, position, i, "attn-input", normed);
                try m.trace(s, position, i, "attn-output", a);
                x = try expand(s, x, a, attn[1], attn[2]);
                try m.trace(s, position, i, "attn-expanded", x);
                const ffn = try m.hc(s, i, "ffn", x);
                const ffn_input = try cp.norm(s, ffn[0], try m.weight(i, "post_attention_layernorm.weight"), g.rms_norm_eps);
                const ffn_output = try m.mlp(s, i, ffn_input);
                try m.trace(s, position, i, "ffn-input", ffn_input);
                try m.trace(s, position, i, "ffn-output", ffn_output);
                x = try expand(s, x, ffn_output, ffn[1], ffn[2]);
                if (m.trace_dir) |dir| {
                    var path: [4096]u8 = undefined;
                    try save(s, try std.fmt.bufPrint(&path, "{s}/trace-{d}-{d}.npy", .{ dir, m.position + @as(i32, @intCast(row)), i }), x);
                }
                if ((i + 1) % 8 == 0) try mx.eval(x);
            }
            const wide = try s.reshape(try s.cast(x, mx.f32t), &.{ 4, g.hidden_size });
            var sum = try s.slice(wide, 0, 0, 1);
            for (1..4) |j| sum = try s.binary(c.mlx_add, sum, try s.slice(wide, 0, @intCast(j), @intCast(j + 1)));
            hidden[row] = try cp.norm(s, try s.cast(try s.binary(c.mlx_multiply, sum, try s.scalar(0.25)), mx.bf16), try m.weights.get("norm.weight"), g.rms_norm_eps);
            logits[row] = try m.qmm(s, hidden[row], "lm_head", true, null);
            try mx.eval(logits[row]);
            p.records[row] = try mx.allocator.dupe(Cache, work);
        }
        p.hidden = try s.cat(hidden[0..tokens.len], 0);
        p.logits = try s.cat(logits[0..tokens.len], 0);
        return p;
    }
    fn trace(m: *Model, s: *mx.Scope, position: i32, i: usize, label: []const u8, value: A) !void {
        if (m.trace_dir) |dir| {
            var path: [4096]u8 = undefined;
            try save(s, try std.fmt.bufPrint(&path, "{s}/trace-{d}-{d}-{s}.npy", .{ dir, position, i, label }), value);
        }
    }
    pub fn commit(m: *Model, p: *Pass, keep: usize) !void {
        if (p.is_mtp or keep == 0 or keep > p.rows or p.position != m.position or p.generation != m.generation) return error.InvalidCommit;
        const next = try mx.allocator.alloc(Cache, m.cache.len);
        @memset(next, .{});
        errdefer {
            for (next) |*cache| cache.deinit();
            mx.allocator.free(next);
        }
        for (p.records[keep - 1], next) |record, *cache| cache.* = try record.clone();
        for (m.cache) |*cache| cache.deinit();
        mx.allocator.free(m.cache);
        m.cache = next;
        m.position += @intCast(keep);
        m.generation +%= 1;
    }
    pub fn checkExact(m: *Model, prefix: usize) !void {
        defer m.reset();
        var s = mx.Scope{};
        defer s.deinit();
        var reference: [4]A = undefined;
        var saved: []Cache = &.{};
        defer {
            for (saved) |*cache| cache.deinit();
            mx.allocator.free(saved);
        }
        for (0..2) |run| {
            m.reset();
            var off: usize = 0;
            while (off < prefix) {
                var ids: [16]i32 = undefined;
                const n = @min(16, prefix - off);
                for (ids[0..n], 0..) |*id, j| id.* = @intCast((off + j + 1) % @as(usize, @intCast(m.vocab)));
                var p = try m.forward(ids[0..n]);
                defer p.deinit();
                try m.commit(&p, n);
                off += n;
            }
            if (run == 0) {
                for ([_]i32{ 1, 2, 3, 4 }, 0..) |id, j| {
                    var p = try m.forward(&.{id});
                    defer p.deinit();
                    reference[j] = try s.own(try mx.retain(p.logits));
                    try m.commit(&p, 1);
                }
                saved = try mx.allocator.alloc(Cache, m.cache.len);
                @memset(saved, .{});
                for (m.cache, saved) |cache, *copy| copy.* = try cache.clone();
            } else {
                var p = try m.forward(&.{ 1, 2, 3, 4, 6 });
                defer p.deinit();
                for (reference, 0..) |value, j| try @import("sampling_checks.zig").equal(&s, value, try s.slice(p.logits, 0, @intCast(j), @intCast(j + 1)));
                try m.commit(&p, 3);
                try std.testing.expectError(error.InvalidCommit, m.commit(&p, 1));
                var next = try m.forward(&.{4});
                defer next.deinit();
                try @import("sampling_checks.zig").equal(&s, reference[3], next.logits);
                try m.commit(&next, 1);
                for (m.cache, saved) |actual, want| inline for (comptime std.meta.fieldNames(Cache)) |field| {
                    const x = @field(actual, field);
                    const y = @field(want, field);
                    if ((x.ctx == null) != (y.ctx == null)) return error.CacheMismatch;
                    if (x.ctx != null) try @import("sampling_checks.zig").equal(&s, x, y);
                };
            }
        }
        std.debug.print("PASS: GLM serial/chain logits, partial commit and every cache at prefix {d}.\n", .{prefix});
    }
    pub fn checkMtp(m: *Model) !void {
        if (!m.has_mtp) return;
        defer m.reset();
        var s = mx.Scope{};
        defer s.deinit();
        const hidden = try s.zeros(&.{ 5, m.config.value.hidden_size }, mx.bf16);
        const one = try s.slice(hidden, 0, 0, 1);
        var expected: [4]A = undefined;
        var saved = Cache{};
        defer saved.deinit();
        for (0..2) |run| {
            m.reset();
            for (0..19) |_| {
                var p = try m.forwardMtp(one, &.{1});
                defer p.deinit();
                try m.commitMtp(&p, 1);
            }
            if (run == 0) {
                for ([_]i32{ 1, 2, 3, 4 }, 0..) |id, j| {
                    var p = try m.forwardMtp(one, &.{id});
                    defer p.deinit();
                    expected[j] = try s.own(try mx.retain(p.logits));
                    try m.commitMtp(&p, 1);
                }
                saved = try m.mtp_cache.clone();
            } else {
                var p = try m.forwardMtp(hidden, &.{ 1, 2, 3, 4, 6 });
                defer p.deinit();
                for (expected, 0..) |want, j| try @import("sampling_checks.zig").equal(&s, want, try s.slice(p.logits, 0, @intCast(j), @intCast(j + 1)));
                try std.testing.expectError(error.InvalidCommit, m.commit(&p, 1));
                try m.commitMtp(&p, 3);
                try std.testing.expectError(error.InvalidCommit, m.commitMtp(&p, 1));
                var next = try m.forwardMtp(one, &.{4});
                defer next.deinit();
                try @import("sampling_checks.zig").equal(&s, expected[3], next.logits);
                try m.commitMtp(&next, 1);
                inline for (comptime std.meta.fieldNames(Cache)) |field| {
                    const x = @field(m.mtp_cache, field);
                    const y = @field(saved, field);
                    if ((x.ctx == null) != (y.ctx == null)) return error.CacheMismatch;
                    if (x.ctx != null) try @import("sampling_checks.zig").equal(&s, x, y);
                }
                m.reset();
                try std.testing.expectError(error.InvalidCommit, m.commitMtp(&next, 1));
            }
        }
        std.debug.print("PASS: GLM MTP serial/chain, partial commit, stale passes and every cache.\n", .{});
    }
    pub fn checkGeneration(m: *Model) !void {
        if (!m.has_mtp) return;
        defer m.reset();
        const generation = @import("serial_generation.zig");
        var prompt: [33]i32 = undefined;
        for (&prompt, 0..) |*token, i| token.* = @intCast(i + 1);
        for ([_]@import("sampling.zig").Sampling{ .{ .temperature = 0 }, .{ .seed = 456, .temperature = 0.7, .top_k = 20, .top_p = 0.95, .metal = true } }) |settings| {
            m.reset();
            var serial = try generation.generate(m, &prompt, 12, settings, 0, null);
            defer serial.deinit();
            const saved = try mx.allocator.alloc(Cache, m.cache.len);
            @memset(saved, .{});
            defer {
                for (saved) |*cache| cache.deinit();
                mx.allocator.free(saved);
            }
            for (m.cache, saved) |cache, *copy| copy.* = try cache.clone();
            for ([_]usize{ 1, 3, 15 }) |depth| {
                m.reset();
                var drafted = try generation.generate(m, &prompt, 12, settings, depth, null);
                defer drafted.deinit();
                try std.testing.expectEqualSlices(u32, serial.tokens.items, drafted.tokens.items);
                try std.testing.expect(drafted.drafted > 0);
                var s = mx.Scope{};
                defer s.deinit();
                for (m.cache, saved) |actual, want| inline for (comptime std.meta.fieldNames(Cache)) |field| {
                    const x = @field(actual, field);
                    const y = @field(want, field);
                    if ((x.ctx == null) != (y.ctx == null)) return error.CacheMismatch;
                    if (x.ctx != null) try @import("sampling_checks.zig").equal(&s, x, y);
                };
                try std.testing.expectEqual(m.position - 1, m.mtp_position);
            }
        }
        var s = mx.Scope{};
        defer s.deinit();
        var head = try m.weights.triple("lm_head");
        for (&head) |*a| a.* = try s.own(try mx.retain(a.*));
        defer {
            m.weights.put("lm_head.scales", head[1]) catch unreachable;
            m.weights.put("lm_head.biases", head[2]) catch unreachable;
        }
        try m.weights.put("lm_head.scales", try s.zeros(mx.shape(head[1]), mx.dtype(head[1])));
        try m.weights.put("lm_head.biases", try s.zeros(mx.shape(head[2]), mx.dtype(head[2])));
        for ([_]usize{ 0, 1, 2, 17 }) |limit| {
            m.reset();
            var result = try generation.generate(m, &prompt, limit, .{ .temperature = 0 }, 3, null);
            defer result.deinit();
            try std.testing.expectEqual(limit, result.tokens.items.len);
            try std.testing.expectEqual(result.drafted, result.accepted);
        }
        const eos = m.config.value.eos_token_id;
        defer m.config.value.eos_token_id = eos;
        m.config.value.eos_token_id = &.{0};
        m.reset();
        var stopped = try generation.generate(m, &prompt, 12, .{ .temperature = 0 }, 3, null);
        defer stopped.deinit();
        try std.testing.expectEqualSlices(u32, &.{0}, stopped.tokens.items);
        std.debug.print("PASS: GLM greedy/seeded MTP generation, depths 1/3/15, target cache parity, full acceptance, EOS and token budgets.\n", .{});
    }
};
fn canonical(raw_name: []const u8, layers: usize) !?[]const u8 {
    const short = blk: {
        if (std.mem.startsWith(u8, raw_name, "lm_head.")) break :blk raw_name;
        for ([_][]const u8{ "model.language_model.", "language_model.model.", "language_model." }) |prefix| if (std.mem.startsWith(u8, raw_name, prefix)) break :blk raw_name[prefix.len..];
        return null;
    };
    var result = if (std.mem.startsWith(u8, short, "mtp.0.")) blk: {
        const rest = short[6..];
        break :blk try std.fmt.allocPrint(mx.allocator, "layers.{d}.{s}", .{ layers, if (std.mem.startsWith(u8, rest, "block.")) rest[6..] else if (std.mem.eql(u8, rest, "norm.weight")) "shared_head.norm.weight" else rest });
    } else try mx.allocator.dupe(u8, short);
    errdefer mx.allocator.free(result);
    inline for (.{ .{ ".attn_hc.", ".hc_attn_" }, .{ ".ffn_hc.", ".hc_ffn_" }, .{ ".self_attn.forget_gate.", ".self_attn." } }) |pair| {
        const next = try std.mem.replaceOwned(u8, mx.allocator, result, pair[0], pair[1]);
        mx.allocator.free(result);
        result = next;
    }
    return result;
}
fn partition(s: *mx.Scope, x: A, n: i32) !A {
    var out = c.mlx_array_new();
    const rc = c.mlx_argpartition_axis(&out, x, n - 1, -1, mx.stream);
    return s.slice(try s.result(rc, out), mx.shape(x).len - 1, 0, n);
}
fn rangeInts(s: *mx.Scope, n: i32) !A {
    const ids = try mx.allocator.alloc(i32, @intCast(n));
    defer mx.allocator.free(ids);
    for (ids, 0..) |*id, j| id.* = @intCast(j);
    return s.ints(ids);
}

pub fn checkModel(io: std.Io, dir: []const u8, out_dir: []const u8) !void {
    try Model.prepareRuntime();
    try mx.init();
    defer mx.shutdown();
    var m = try Model.init(io, dir);
    defer m.deinit();
    try std.Io.Dir.cwd().createDirPath(io, out_dir);
    m.trace_dir = out_dir;
    var path: [4096]u8 = undefined;
    for (0..3) |step| {
        var ids: [16]i32 = undefined;
        const count: usize = if (step == 2) 8 else 16;
        for (ids[0..count], 0..) |*id, j| id.* = @intCast(step * 16 + j + 1);
        var p = try m.forward(ids[0..count]);
        defer p.deinit();
        try m.commit(&p, count);
        try save(&p.scope, try std.fmt.bufPrint(&path, "{s}/hidden-{d}.npy", .{ out_dir, step }), p.hidden);
        try save(&p.scope, try std.fmt.bufPrint(&path, "{s}/logits-{d}.npy", .{ out_dir, step }), p.logits);
        if (step == 2) try save(&p.scope, try std.fmt.bufPrint(&path, "{s}/logits.npy", .{out_dir}), p.logits);
        if (m.has_mtp) {
            for (ids[0..count]) |*id| id.* += 1;
            var head = try m.forwardMtp(p.hidden, ids[0..count]);
            defer head.deinit();
            try m.commitMtp(&head, count);
            try save(&head.scope, try std.fmt.bufPrint(&path, "{s}/mtp-projection-{d}.npy", .{ out_dir, step }), head.mtp_projection);
            try save(&head.scope, try std.fmt.bufPrint(&path, "{s}/mtp-input-{d}.npy", .{ out_dir, step }), head.mtp_input);
            if (step == 2) {
                try save(&head.scope, try std.fmt.bufPrint(&path, "{s}/mtp-logits.npy", .{out_dir}), head.logits);
                try save(&head.scope, try std.fmt.bufPrint(&path, "{s}/mtp-hidden.npy", .{out_dir}), head.hidden);
                try save(&head.scope, try std.fmt.bufPrint(&path, "{s}/mtp-projection.npy", .{out_dir}), head.mtp_projection);
                inline for (comptime std.meta.fieldNames(Cache)) |field| if (@field(m.mtp_cache, field).ctx != null) {
                    try save(&head.scope, try std.fmt.bufPrint(&path, "{s}/mtp-{s}.npy", .{ out_dir, field }), @field(m.mtp_cache, field));
                };
            }
        }
    }
    var s = mx.Scope{};
    defer s.deinit();
    for (m.cache, 0..) |cache, i| inline for (comptime std.meta.fieldNames(Cache)) |field| if (@field(cache, field).ctx != null) {
        try save(&s, try std.fmt.bufPrint(&path, "{s}/layer{d}-{s}.npy", .{ out_dir, i, field }), @field(cache, field));
    };
    m.trace_dir = null;
    try m.checkExact(19);
    try m.checkMtp();
    try m.checkGeneration();
    try @import("session_checks.zig").checkSyntheticNeural(&m, io);
}
pub fn checkPrefillKda(io: std.Io, dir: []const u8) !void {
    try Model.prepareRuntime();
    try mx.init();
    defer mx.shutdown();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/kda.json", .{dir}));
    defer mx.allocator.free(bytes);
    const Case = struct { name: []const u8, decode: bool };
    const Group = struct { checkpoint: []const u8, cases: []const Case };
    const groups = try std.json.parseFromSlice([]const Group, mx.allocator, bytes, .{});
    defer groups.deinit();
    if (groups.value.len == 0) return error.EmptyFixtures;
    var count: usize = 0;
    for (groups.value) |group| {
        var m = try Model.init(io, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, group.checkpoint }));
        defer m.deinit();
        if (group.cases.len == 0) return error.EmptyFixtures;
        {
            var s = mx.Scope{};
            defer s.deinit();
            const dims = m.config.value.hidden_size;
            var cache = Cache{};
            try std.testing.expectError(error.InvalidTensorShape, m.kdaPrefill(&s, 0, try s.zeros(&.{ 2049, dims }, mx.bf16), &cache));
            try std.testing.expectError(error.InvalidTensorShape, m.hc(&s, 0, "attn", try s.zeros(&.{ 2049, 4, dims }, mx.bf16)));
            try std.testing.expectError(error.InvalidTensorShape, m.kdaPrefill(&s, 0, try s.zeros(&.{ 1, dims }, mx.f32t), &cache));
            cache.conv = try s.zeros(&.{ 1, 1 }, mx.bf16);
            try std.testing.expectError(error.InvalidTensorShape, m.kdaPrefill(&s, 0, try s.zeros(&.{ 1, dims }, mx.bf16), &cache));
            try std.testing.expect(cache.state.ctx == null);
        }
        for (group.cases) |case| {
            errdefer std.debug.print("GLM prefill KDA fixture failed: {s}/{s}\n", .{ group.checkpoint, case.name });
            var store = cp.Store.init(32);
            defer store.deinit();
            try store.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
            var s = mx.Scope{};
            defer s.deinit();
            const equal = @import("variant_checks.zig").equalBits;
            const x = try store.get("input");
            const hc = try m.hc(&s, 0, "attn", x);
            for (hc, [_][]const u8{ "collapsed", "post", "comb" }) |actual, key| try equal(&s, actual, try store.get(key));
            const normed = try cp.norm(&s, hc[0], try m.weight(0, "input_layernorm.weight"), m.config.value.rms_norm_eps);
            try equal(&s, normed, try store.get("normed"));
            var cache = m.cache[0];
            if (cache.state.ctx != null) {
                try equal(&s, cache.state, try store.get("previous.state"));
                try equal(&s, cache.conv, try store.get("previous.conv"));
            }
            const output = if (case.decode) try m.kda(&s, 0, normed, &cache) else blk: {
                const result = try m.kdaPrefill(&s, 0, normed, &cache);
                inline for (comptime std.meta.fieldNames(Model.KdaResult)) |key| {
                    errdefer std.debug.print("Mismatch in {s}\n", .{key});
                    try equal(&s, @field(result, key), try store.get(key));
                }
                break :blk result.output;
            };
            try equal(&s, output, try store.get("output"));
            try equal(&s, cache.state, try store.get("next.state"));
            try equal(&s, cache.conv, try store.get("next.conv"));
            const expanded = try Model.expand(&s, x, output, hc[1], hc[2]);
            try equal(&s, expanded, try store.get("expanded"));
            const ffn = try m.hc(&s, 0, "ffn", expanded);
            for (ffn, [_][]const u8{ "ffn.collapsed", "ffn.post", "ffn.comb" }) |actual, key| try equal(&s, actual, try store.get(key));
            const saved = try cache.clone();
            m.cache[0].deinit();
            m.cache[0] = saved;
            count += 1;
        }
    }
    std.debug.print("PASS: {d} GLM batched KDA/hyper-connection cases, exact intermediate arrays, mixed projections, caches and decode continuation\n", .{count});
}
fn save(s: *mx.Scope, path: []const u8, value: A) !void {
    const z = try mx.allocator.dupeSentinel(u8, path, 0);
    defer mx.allocator.free(z);
    const v = try s.cast(value, mx.f32t);
    try mx.eval(v);
    try mx.check(c.mlx_save(z, v));
}

test "GLM checkpoint layouts preserve projection and draft-head identities" {
    const cases = .{
        .{ "model.language_model.layers.0.self_attn.f_a_proj.weight", "layers.0.self_attn.f_a_proj.weight" },
        .{ "language_model.model.layers.3.attn_hc.fn", "layers.3.hc_attn_fn" },
        .{ "language_model.model.layers.3.ffn_hc.base", "layers.3.hc_ffn_base" },
        .{ "language_model.model.layers.0.self_attn.forget_gate.f_b_proj.scales", "layers.0.self_attn.f_b_proj.scales" },
        .{ "language_model.mtp.0.block.self_attn.embed_q.weight", "layers.45.self_attn.embed_q.weight" },
        .{ "language_model.mtp.0.eh_proj.weight", "layers.45.eh_proj.weight" },
        .{ "language_model.mtp.0.norm.weight", "layers.45.shared_head.norm.weight" },
        .{ "language_model.lm_head.weight", "lm_head.weight" },
    };
    inline for (cases) |pair| {
        const result = (try canonical(pair[0], 45)).?;
        defer mx.allocator.free(result);
        try std.testing.expectEqualStrings(pair[1], result);
    }
    try std.testing.expectEqual(@as(?[]const u8, null), try canonical("vision_model.proj.weight", 45));
}
