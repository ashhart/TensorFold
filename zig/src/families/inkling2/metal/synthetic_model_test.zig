//! Shared synthetic 48-visit Inkling 2 Mini Metal model builder for GPU parity and tree conformance tests.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const ck = @import("../../../core/checkpoint_metal.zig");
const model_ = @import("model.zig");

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

pub fn syntheticConfig() cfg.Config {
    return .{
        .text = .{ .hidden = 2304, .layers = 24, .vocab = 201024, .unpadded_vocab = 200058, .local = .{ .q = 24, .kv = 8, .dim = 128 }, .global = .{ .q = 24, .kv = 8, .dim = 128 }, .window = 1024, .d_rel = 16, .rel_extent = 1024, .log_floor = 8192, .log_alpha = 0.1, .eps = 1e-6, .qk_clip = 22.627416997969522, .rel_clip = 8, .block = 6, .experts = 128, .topk = 8, .expert_inter = 1152, .route_scale = 8, .gate_cap = 4, .up_cap = 25, .logit_cap = 30, .max_context = 1048576, .loop = .{ .core_start = 6, .core_end = 18, .passes = 3, .gate_rank = 64 } },
        .vision = .{ .hidden = 0, .layers = 0, .heads = 0, .patch = 0, .merge = 0, .grid = 0, .mlp_ratio = 0, .proj_hidden = 0, .text_hidden = 0, .eps = 0 },
        .mtp = .{ .layers = 1, .q_heads = 24, .kv_heads = 8, .head_dim = 128, .inter = 1152, .rope_theta = 10000, .window = 1024, .local = @splat(true), .block = 6, .rank = 64, .taps = .{ 5, 23, 41, 48 } ++ @as([12]u32, @splat(0)), .taps_n = 4 },
        .eos = 1,
        .image_token = cfg.default_image_token,
    };
}

fn makeTensor(buf: mtl.Buffer, off: usize, dtype: ck.DType, dims: []const usize) ck.Tensor {
    var count: usize = 1;
    var t: ck.Tensor = .{ .buffer = buf, .offset = off, .bytes = 0, .dtype = dtype, .rank = dims.len };
    for (dims, 0..) |d, i| {
        t.shape[i] = d;
        count *= d;
    }
    t.bytes = count * dtype.size();
    return t;
}

fn addShard(c: *ck.Checkpoint, device: mtl.Device, bytes: usize) !mtl.Buffer {
    const buf = try device.buffer(bytes, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    try c.shards.append(c.allocator, .{ .buffer = buf, .bytes = bytes });
    return buf;
}

fn fillBf16Rng(out: []u16, rng: std.Random, amp: f32, bias: f32) void {
    for (out) |*v| v.* = bf16(bias + (rng.float(f32) * 2 - 1) * amp);
}

fn fillF32Rng(out: []f32, rng: std.Random, amp: f32, bias: f32) void {
    for (out) |*v| v.* = bias + (rng.float(f32) * 2 - 1) * amp;
}

pub fn buildSyntheticModel(gpa: std.mem.Allocator) !*model_.Model {
    const m = try gpa.create(model_.Model);
    errdefer gpa.destroy(m);
    m.gpa = gpa;
    m.config = syntheticConfig();
    m.device = try mtl.Device.init();
    errdefer m.device.deinit();
    m.queue = try m.device.queue();
    errdefer m.queue.deinit();
    m.checkpoint = ck.Checkpoint.init(gpa);
    errdefer m.checkpoint.deinit();
    m.tfq3_tables = null;
    m.tfq3_maps = .empty;
    m.tfq3_bufs = .empty;
    m.tfq3_pipes = null;
    m.tfq3_bytes = 0;
    var precisions: [24]layer_plan.Precision = undefined;
    for (&precisions, 0..) |*p, i| p.* = if (i < 20) .nvfp4 else .mxfp8;
    m.weight_plan = try layer_plan.build(&m.config.text, &precisions);
    try m.initPipelines();
    errdefer {
        m.sample.deinit();
        for (m.runtime) |p| p.deinit();
        m.norm.deinit();
        for (m.dense) |p| p.deinit();
        m.tree.deinit();
        m.loop.deinit();
        m.experts.deinit();
        m.glue.deinit();
        m.attention.deinit();
    }

    var prng = std.Random.DefaultPrng.init(0x5eed1234);
    const rng = prng.random();
    const embed_bytes = 201024 * 2304 * 2;
    const embed_buf = try addShard(&m.checkpoint, m.device, embed_bytes + 4608);
    fillBf16Rng(embed_buf.slice(u16, 256 * 2304 + 2304), rng, 0.03, 0.002);
    const nv_gu_bytes = 128 * 2304 * 1152;
    const nv_buf = try addShard(&m.checkpoint, m.device, nv_gu_bytes);
    @memset(nv_buf.slice(u64, nv_gu_bytes / 8), 0x3124312431243124);
    const mx_gu_bytes = 128 * 2304 * 2304;
    const mx_buf = try addShard(&m.checkpoint, m.device, mx_gu_bytes);
    @memset(mx_buf.slice(u64, mx_gu_bytes / 8), 0x3031b2b334b536b7);
    const nv_sc_buf = try addShard(&m.checkpoint, m.device, 128 * 2304 * 144);
    @memset(nv_sc_buf.slice(u8, 128 * 2304 * 144), 0x38);
    const mx_sc_buf = try addShard(&m.checkpoint, m.device, 128 * 2304 * 72);
    @memset(mx_sc_buf.slice(u8, 128 * 2304 * 72), 125);
    const misc_buf = try addShard(&m.checkpoint, m.device, 64 << 20);
    const misc_f32_bytes = 2 << 20;
    fillF32Rng(misc_buf.slice(f32, misc_f32_bytes / 4), rng, 0.03, 0);
    const unit_f32 = try addShard(&m.checkpoint, m.device, 4096 * 4);
    for (unit_f32.slice(f32, 4096)) |*v| v.* = 1.0;
    const ones_bf = try addShard(&m.checkpoint, m.device, 4096 * 2);
    fillBf16Rng(ones_bf.slice(u16, 4096), rng, 0.05, 1.0);
    const dense_bytes = 160 << 20;
    const dense_pool = try addShard(&m.checkpoint, m.device, dense_bytes);
    fillBf16Rng(dense_pool.slice(u16, dense_bytes / 2), rng, 0.025, 0);

    m.weights.embed = makeTensor(embed_buf, 0, .bf16, &.{ 201024, 2304 });
    m.weights.unembed = makeTensor(embed_buf, 4608, .bf16, &.{ 201024, 2304 });
    m.weights.embed_norm = makeTensor(ones_bf, 0, .bf16, &.{2304});
    m.weights.norm = makeTensor(ones_bf, 0, .bf16, &.{2304});
    m.weights.output_query = makeTensor(misc_buf, 0, .f32, &.{2304});
    m.weights.cross_queries = makeTensor(misc_buf, 4096, .f32, &.{ 24, 2304 });
    m.weights.cross_kv_queries = makeTensor(misc_buf, 4096 + 24 * 2304 * 4, .f32, &.{ 24, 2304 });
    for (&m.weights.attn_queries, 0..) |*q, i| {
        const g = cfg.isGlobal(@intCast(i + 1));
        q.* = makeTensor(misc_buf, 512 * 1024 + i * 3 * 2304 * 4, .f32, if (g) &.{ 3, 2304 } else &.{2304});
    }
    for (&m.weights.mlp_queries, 0..) |*q, i| q.* = makeTensor(misc_buf, 1024 * 1024 + i * 2304 * 4, .f32, &.{2304});
    var pool_off: usize = 0;
    const allocDense = struct {
        fn next(buf: mtl.Buffer, off: *usize, dims: []const usize) ck.Tensor {
            const t = makeTensor(buf, off.*, .bf16, dims);
            off.* = std.mem.alignForward(usize, off.* + t.bytes, 256);
            return t;
        }
    }.next;
    var shared_wq: [2]ck.Tensor = undefined;
    var shared_wk: [2]ck.Tensor = undefined;
    var shared_wv: [2]ck.Tensor = undefined;
    var shared_wr: [2]ck.Tensor = undefined;
    var shared_wo: [2]ck.Tensor = undefined;
    var shared_gate: [2]ck.Tensor = undefined;
    var shared_router: [2]ck.Tensor = undefined;
    for (0..2) |k| {
        shared_wq[k] = allocDense(dense_pool, &pool_off, &.{ 3072, 2304 });
        shared_wk[k] = allocDense(dense_pool, &pool_off, &.{ 1024, 2304 });
        shared_wv[k] = allocDense(dense_pool, &pool_off, &.{ 1024, 2304 });
        shared_wr[k] = allocDense(dense_pool, &pool_off, &.{ 384, 2304 });
        shared_wo[k] = allocDense(dense_pool, &pool_off, &.{ 2304, 3072 });
        shared_gate[k] = allocDense(dense_pool, &pool_off, &.{ 3072, 2304 });
        shared_router[k] = allocDense(dense_pool, &pool_off, &.{ 128, 2304 });
    }
    for (&m.weights.layers, 0..) |*l, i| {
        const p = m.weight_plan.at(@intCast(i));
        const g = p.kind == .global;
        const k = i % 2;
        l.attn = .{
            .wq = shared_wq[k],
            .wk = shared_wk[k],
            .wv = shared_wv[k],
            .wr = shared_wr[k],
            .wo = shared_wo[k],
            .gate = shared_gate[k],
            .sink = allocDense(dense_pool, &pool_off, &.{ 24, 2304 }),
            .q_scale = makeTensor(ones_bf, 0, .bf16, &.{ 24, 128 }),
            .rel_scale = makeTensor(ones_bf, 0, .bf16, &.{ 24, 16 }),
            .rel_projection = allocDense(dense_pool, &pool_off, &.{ 16, p.extent }),
            .k_conv = if (!g) .{ .static = allocDense(dense_pool, &pool_off, &.{ 1024, 1, 4 }) } else .{ .dynamic = .{ .static_w = allocDense(dense_pool, &pool_off, &.{ 4, 1024 }), .projection = allocDense(dense_pool, &pool_off, &.{ 32, 2304 }) } },
            .v_conv = if (!g) .{ .static = allocDense(dense_pool, &pool_off, &.{ 1024, 1, 4 }) } else .{ .dynamic = .{ .static_w = allocDense(dense_pool, &pool_off, &.{ 4, 1024 }), .projection = allocDense(dense_pool, &pool_off, &.{ 32, 2304 }) } },
            .q_norm = makeTensor(ones_bf, 0, .bf16, &.{2304}),
            .k_norm = makeTensor(ones_bf, 0, .bf16, &.{2304}),
            .v_norm = makeTensor(ones_bf, 0, .bf16, &.{2304}),
        };
        l.attn_conv = allocDense(dense_pool, &pool_off, &.{ 2304, 1, 4 });
        l.mlp_norm = makeTensor(ones_bf, 0, .bf16, &.{2304});
        l.mlp_conv = allocDense(dense_pool, &pool_off, &.{ 2304, 1, 4 });
        l.router_weight = shared_router[k];
        l.router_bias = makeTensor(misc_buf, 1536 * 1024 + i * 128 * 4, .f32, &.{128});
        l.gu = if (p.precision == .nvfp4)
            .{ .weight = makeTensor(nv_buf, 0, .u8, &.{ 128, 2304, 1152 }), .scale = makeTensor(nv_sc_buf, 0, .f8_e4m3, &.{ 128, 2304, 144 }), .amax = makeTensor(unit_f32, 0, .f32, &.{128}), .denominator = makeTensor(unit_f32, 0, .f32, &.{1}) }
        else
            .{ .weight = makeTensor(mx_buf, 0, .f8_e4m3, &.{ 128, 2304, 2304 }), .scale = makeTensor(mx_sc_buf, 0, .u8, &.{ 128, 2304, 72 }) };
        l.down = if (p.precision == .nvfp4)
            .{ .weight = makeTensor(nv_buf, 0, .u8, &.{ 128, 2304, 576 }), .scale = makeTensor(nv_sc_buf, 0, .f8_e4m3, &.{ 128, 2304, 72 }), .amax = makeTensor(unit_f32, 0, .f32, &.{128}), .denominator = makeTensor(unit_f32, 0, .f32, &.{1}) }
        else
            .{ .weight = makeTensor(mx_buf, 0, .f8_e4m3, &.{ 128, 2304, 1152 }), .scale = makeTensor(mx_sc_buf, 0, .u8, &.{ 128, 2304, 36 }) };
        l.tfq3 = null;
        l.controllers = null;
        if (i >= 6 and i < 18) l.controllers = @splat(.{
            .down = makeTensor(dense_pool, 0, .bf16, &.{ 64, 4608 }),
            .forget = makeTensor(dense_pool, 4096, .bf16, &.{ 2304, 64 }),
            .gain = makeTensor(dense_pool, 8192, .bf16, &.{ 2304, 64 }),
        });
    }
    std.debug.assert(pool_off <= dense_bytes);
    return m;
}

