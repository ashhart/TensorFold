//! Deterministic synthetic MLX 4-bit affine g64 Inkling 2 Mini Metal model builder for GPU parity tests.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const ck = @import("../../../core/checkpoint_metal.zig");
const model_ = @import("model.zig");
const affine4 = @import("affine4.zig");
const ref = @import("affine4_ref.zig");

fn syntheticAffine4Config() cfg.Config {
    return .{
        .text = .{ .hidden = 2304, .layers = 24, .vocab = 201024, .unpadded_vocab = 200058, .local = .{ .q = 24, .kv = 8, .dim = 128 }, .global = .{ .q = 24, .kv = 8, .dim = 128 }, .window = 1024, .d_rel = 16, .rel_extent = 1024, .log_floor = 8192, .log_alpha = 0.1, .eps = 1e-6, .qk_clip = 22.627416997969522, .rel_clip = 8, .block = 6, .experts = 128, .topk = 8, .expert_inter = 1152, .route_scale = 8, .gate_cap = 4, .up_cap = 25, .logit_cap = 30, .max_context = 1048576, .loop = .{ .core_start = 6, .core_end = 18, .passes = 3, .gate_rank = 64 } },
        .vision = .{ .hidden = 0, .layers = 0, .heads = 0, .patch = 0, .merge = 0, .grid = 0, .mlp_ratio = 0, .proj_hidden = 0, .text_hidden = 0, .eps = 0 },
        .mtp = .{ .layers = 1, .q_heads = 24, .kv_heads = 8, .head_dim = 128, .inter = 1152, .rope_theta = 10000, .window = 1024, .local = @splat(true), .block = 6, .rank = 64, .taps = .{ 5, 23, 41, 48 } ++ @as([12]u32, @splat(0)), .taps_n = 4 },
        .eos = 1,
        .image_token = cfg.default_image_token,
        .quant = .mlx_affine4,
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
    for (out) |*v| v.* = ref.toBf16(bias + (rng.float(f32) * 2 - 1) * amp);
}

fn fillF32Rng(out: []f32, rng: std.Random, amp: f32, bias: f32) void {
    for (out) |*v| v.* = bias + (rng.float(f32) * 2 - 1) * amp;
}

fn allocQ4(buf: mtl.Buffer, off: *usize, rows: u32, cols: u32, rng: std.Random, scale_amp: f32) affine4.Affine4Tensor {
    const w = makeTensor(buf, off.*, .u32, &.{ rows, cols / 8 });
    off.* = std.mem.alignForward(usize, off.* + w.bytes, 256);
    const s = makeTensor(buf, off.*, .bf16, &.{ rows, cols / 64 });
    off.* = std.mem.alignForward(usize, off.* + s.bytes, 256);
    const b = makeTensor(buf, off.*, .bf16, &.{ rows, cols / 64 });
    off.* = std.mem.alignForward(usize, off.* + b.bytes, 256);
    const w_ptr: [*]u32 = @ptrCast(@alignCast(buf.contents() + w.offset));
    for (w_ptr[0 .. w.bytes / 4]) |*v| v.* = rng.int(u32);
    const s_ptr: [*]u16 = @ptrCast(@alignCast(buf.contents() + s.offset));
    for (s_ptr[0 .. s.bytes / 2]) |*v| v.* = ref.toBf16(rng.float(f32) * scale_amp + scale_amp * 0.25);
    const b_ptr: [*]u16 = @ptrCast(@alignCast(buf.contents() + b.offset));
    for (b_ptr[0 .. b.bytes / 2]) |*v| v.* = ref.toBf16((rng.float(f32) - 0.5) * scale_amp * 12.0);
    return .{ .weight = w, .scales = s, .biases = b, .rows = rows, .cols = cols };
}

pub fn build(gpa: std.mem.Allocator) !*model_.Model {
    const m = try gpa.create(model_.Model);
    errdefer gpa.destroy(m);
    m.gpa = gpa;
    m.config = syntheticAffine4Config();
    m.device = try mtl.Device.init();
    errdefer m.device.deinit();
    m.queue = try m.device.queue();
    errdefer m.queue.deinit();
    m.checkpoint = ck.Checkpoint.init(gpa);
    errdefer m.checkpoint.deinit();
    var precisions: [24]layer_plan.Precision = undefined;
    for (&precisions, 0..) |*p, i| p.* = if (i < 20) .nvfp4 else .mxfp8;
    m.weight_plan = try layer_plan.build(&m.config.text, &precisions);
    try m.initPipelines();
    errdefer {
        m.sample.deinit();
        for (m.runtime) |p| p.deinit();
        if (m.affine4) |p| p.deinit();
        m.norm.deinit();
        for (m.dense) |p| p.deinit();
        if (m.dense_tiled) |tiled| tiled.deinit();
        if (m.dense_mma) |mma| mma.deinit();
        m.tree.deinit();
        m.loop.deinit();
        m.experts.deinit();
        m.glue.deinit();
        m.attention.deinit();
    }

    var prng = std.Random.DefaultPrng.init(0xa4f1_5208);
    const rng = prng.random();

    const vocab: usize = 201024;
    const emb_w_buf = try addShard(&m.checkpoint, m.device, (vocab + 1) * 288 * 4);
    const emb_s_buf = try addShard(&m.checkpoint, m.device, (vocab + 1) * 36 * 2);
    const emb_b_buf = try addShard(&m.checkpoint, m.device, (vocab + 1) * 36 * 2);
    const w_all = emb_w_buf.slice(u32, (vocab + 1) * 288);
    const s_all = emb_s_buf.slice(u16, (vocab + 1) * 36);
    const b_all = emb_b_buf.slice(u16, (vocab + 1) * 36);
    for (0..512) |row| {
        for (0..288) |j| w_all[row * 288 + j] = rng.int(u32);
        for (0..36) |g| {
            s_all[row * 36 + g] = ref.toBf16(rng.float(f32) * 0.004 + 0.001);
            b_all[row * 36 + g] = ref.toBf16((rng.float(f32) - 0.5) * 0.05);
        }
    }
    var copy_row: usize = 512;
    while (copy_row <= vocab) : (copy_row += 512) {
        const take: usize = @min(512, vocab + 1 - copy_row);
        @memcpy(w_all[copy_row * 288 ..][0 .. take * 288], w_all[0 .. take * 288]);
        @memcpy(s_all[copy_row * 36 ..][0 .. take * 36], s_all[0 .. take * 36]);
        @memcpy(b_all[copy_row * 36 ..][0 .. take * 36], b_all[0 .. take * 36]);
    }
    m.weights.embed = .{ .affine4 = .{
        .weight = makeTensor(emb_w_buf, 0, .u32, &.{ vocab, 288 }),
        .scales = makeTensor(emb_s_buf, 0, .bf16, &.{ vocab, 36 }),
        .biases = makeTensor(emb_b_buf, 0, .bf16, &.{ vocab, 36 }),
        .rows = 201024,
        .cols = 2304,
    } };
    m.weights.unembed = .{ .affine4 = .{
        .weight = makeTensor(emb_w_buf, 288 * 4, .u32, &.{ vocab, 288 }),
        .scales = makeTensor(emb_s_buf, 36 * 2, .bf16, &.{ vocab, 36 }),
        .biases = makeTensor(emb_b_buf, 36 * 2, .bf16, &.{ vocab, 36 }),
        .rows = 201024,
        .cols = 2304,
    } };

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
    fillF32Rng(misc_buf.slice(f32, (2 << 20) / 4), rng, 0.03, 0);
    const unit_f32 = try addShard(&m.checkpoint, m.device, 4096 * 4);
    for (unit_f32.slice(f32, 4096)) |*v| v.* = 1.0;
    const ones_bf = try addShard(&m.checkpoint, m.device, 4096 * 2);
    fillBf16Rng(ones_bf.slice(u16, 4096), rng, 0.05, 1.0);
    const dense_bytes = 64 << 20;
    const dense_pool = try addShard(&m.checkpoint, m.device, dense_bytes);
    fillBf16Rng(dense_pool.slice(u16, dense_bytes / 2), rng, 0.025, 0);
    const q4_bytes = 64 << 20;
    const q4_pool = try addShard(&m.checkpoint, m.device, q4_bytes);
    var q4_off: usize = 0;

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
    const allocBf16 = struct {
        fn next(buf: mtl.Buffer, off: *usize, dims: []const usize) ck.Tensor {
            const t = makeTensor(buf, off.*, .bf16, dims);
            off.* = std.mem.alignForward(usize, off.* + t.bytes, 256);
            return t;
        }
    }.next;

    var shared_wq: [2]affine4.Affine4Tensor = undefined;
    var shared_wk: [2]affine4.Affine4Tensor = undefined;
    var shared_wv: [2]affine4.Affine4Tensor = undefined;
    var shared_wr: [2]affine4.Affine4Tensor = undefined;
    var shared_wo: [2]affine4.Affine4Tensor = undefined;
    var shared_gate: [2]affine4.Affine4Tensor = undefined;
    var shared_router: [2]ck.Tensor = undefined;
    for (0..2) |k| {
        shared_wq[k] = allocQ4(q4_pool, &q4_off, 3072, 2304, rng, 0.004);
        shared_wk[k] = allocQ4(q4_pool, &q4_off, 1024, 2304, rng, 0.004);
        shared_wv[k] = allocQ4(q4_pool, &q4_off, 1024, 2304, rng, 0.004);
        shared_wr[k] = allocQ4(q4_pool, &q4_off, 384, 2304, rng, 0.004);
        shared_wo[k] = allocQ4(q4_pool, &q4_off, 2304, 3072, rng, 0.004);
        shared_gate[k] = allocQ4(q4_pool, &q4_off, 3072, 2304, rng, 0.004);
        shared_router[k] = allocBf16(dense_pool, &pool_off, &.{ 128, 2304 });
    }
    for (&m.weights.layers, 0..) |*l, i| {
        const p = m.weight_plan.at(@intCast(i));
        const g = p.kind == .global;
        const k = i % 2;
        l.attn = .{
            .wq = .{ .affine4 = shared_wq[k] },
            .wk = .{ .affine4 = shared_wk[k] },
            .wv = .{ .affine4 = shared_wv[k] },
            .wr = .{ .affine4 = shared_wr[k] },
            .wo = .{ .affine4 = shared_wo[k] },
            .gate = .{ .affine4 = shared_gate[k] },
            .sink = .{ .affine4 = allocQ4(q4_pool, &q4_off, 24, 2304, rng, 0.004) },
            .q_scale = makeTensor(ones_bf, 0, .bf16, &.{ 24, 128 }),
            .rel_scale = makeTensor(ones_bf, 0, .bf16, &.{ 24, 16 }),
            .rel_projection = allocBf16(dense_pool, &pool_off, &.{ 16, p.extent }),
            .k_conv = if (!g) .{ .static = allocBf16(dense_pool, &pool_off, &.{ 1024, 1, 4 }) } else .{ .dynamic = .{ .static_w = allocBf16(dense_pool, &pool_off, &.{ 4, 1024 }), .projection = allocBf16(dense_pool, &pool_off, &.{ 32, 2304 }) } },
            .v_conv = if (!g) .{ .static = allocBf16(dense_pool, &pool_off, &.{ 1024, 1, 4 }) } else .{ .dynamic = .{ .static_w = allocBf16(dense_pool, &pool_off, &.{ 4, 1024 }), .projection = allocBf16(dense_pool, &pool_off, &.{ 32, 2304 }) } },
            .q_norm = makeTensor(ones_bf, 0, .bf16, &.{2304}),
            .k_norm = makeTensor(ones_bf, 0, .bf16, &.{2304}),
            .v_norm = makeTensor(ones_bf, 0, .bf16, &.{2304}),
        };
        l.attn_conv = allocBf16(dense_pool, &pool_off, &.{ 2304, 1, 4 });
        l.mlp_norm = makeTensor(ones_bf, 0, .bf16, &.{2304});
        l.mlp_conv = allocBf16(dense_pool, &pool_off, &.{ 2304, 1, 4 });
        l.router_weight = shared_router[k];
        l.router_bias = makeTensor(misc_buf, 1536 * 1024 + i * 128 * 4, .f32, &.{128});
        l.gu = if (p.precision == .nvfp4)
            .{ .weight = makeTensor(nv_buf, 0, .u32, &.{ 128, 2304, 288 }), .scale = makeTensor(nv_sc_buf, 0, .u8, &.{ 128, 2304, 144 }), .amax = makeTensor(unit_f32, 0, .f32, &.{128}), .denominator = makeTensor(unit_f32, 0, .f32, &.{1}) }
        else
            .{ .weight = makeTensor(mx_buf, 0, .u32, &.{ 128, 2304, 576 }), .scale = makeTensor(mx_sc_buf, 0, .u8, &.{ 128, 2304, 72 }) };
        l.down = if (p.precision == .nvfp4)
            .{ .weight = makeTensor(nv_buf, 0, .u32, &.{ 128, 2304, 144 }), .scale = makeTensor(nv_sc_buf, 0, .u8, &.{ 128, 2304, 72 }), .amax = makeTensor(unit_f32, 0, .f32, &.{128}), .denominator = makeTensor(unit_f32, 0, .f32, &.{1}) }
        else
            .{ .weight = makeTensor(mx_buf, 0, .u32, &.{ 128, 2304, 288 }), .scale = makeTensor(mx_sc_buf, 0, .u8, &.{ 128, 2304, 36 }) };
        l.controllers = null;
        if (i >= 6 and i < 18) l.controllers = @splat(.{
            .down = makeTensor(dense_pool, 0, .bf16, &.{ 64, 4608 }),
            .forget = makeTensor(dense_pool, 4096, .bf16, &.{ 2304, 64 }),
            .gain = makeTensor(dense_pool, 8192, .bf16, &.{ 2304, 64 }),
        });
    }
    std.debug.assert(q4_off <= q4_bytes and pool_off <= dense_bytes);
    return m;
}
