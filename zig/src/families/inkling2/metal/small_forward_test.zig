//! Host geometry checks and synthetic 6-layer Metal GPU bit-exactness tests for Inkling 2 Small EXL3.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const exl3_proj = @import("core").exl3_projection;
const ck = @import("../../../core/checkpoint_metal.zig");
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const state = @import("small_state.zig");
const model_file = @import("small_model.zig");
const fwd = @import("small_forward.zig");

const t = std.testing;

test "inkling2 small: state and forward planning geometry" {
    try t.expectEqual(@as(usize, 4608), state.hidden);
    try t.expectEqual(@as(usize, 6144), state.query_width);
    try t.expectEqual(@as(usize, 1024), state.kv_width);
    try t.expectEqual(@as(usize, 768), state.rel_width);
    try t.expectEqual(@as(usize, 8), state.block_slots);
    try t.expectEqual(@as(usize, 8), state.feature_taps);
    try t.expectEqual(@as(usize, 48), state.layer_slots);
    try t.expectEqual(@as(usize, 8 * 4608 * 2), state.feature_row_bytes);

    try t.expectEqual(@as(?usize, 0), fwd.tapChannel(5));
    try t.expectEqual(@as(?usize, 1), fwd.tapChannel(11));
    try t.expectEqual(@as(?usize, 6), fwd.tapChannel(41));
    try t.expectEqual(@as(?usize, null), fwd.tapChannel(0));
    try t.expectEqual(@as(?usize, null), fwd.tapChannel(47));

    const p1 = fwd.finalPlan(8, false, false);
    try t.expectEqual(@as(usize, 1), p1.work_rows);
    try t.expectEqual(@as(usize, 1), p1.logits_rows);
    try t.expectEqual(@as(usize, 0), p1.capture_rows);

    const p2 = fwd.finalPlan(8, false, true);
    try t.expectEqual(@as(usize, 8), p2.work_rows);
    try t.expectEqual(@as(usize, 1), p2.logits_rows);
    try t.expectEqual(@as(usize, 7), p2.logits_row);
    try t.expectEqual(@as(usize, 8), p2.capture_rows);

    try t.expectEqual(@as(usize, 0), fwd.denseStep(1).variant);
    try t.expectEqual(@as(usize, 3), fwd.denseStep(7).variant);
    try t.expectEqual(@as(usize, 5), fwd.denseStep(64).variant);
}

fn toBf16(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn allocBf16(device: mtl.Device, list: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator, rng: std.Random, dims: []const usize, scale: f32) !ck.Tensor {
    var count: usize = 1;
    for (dims) |d| count *= d;
    const bytes = count * 2;
    const buf = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const dst = buf.slice(u16, count);
    for (dst) |*v| v.* = toBf16((rng.float(f32) * 2.0 - 1.0) * scale);
    try list.append(gpa, buf);
    var shape: [core.safetensors.max_rank]usize = @splat(1);
    for (dims, 0..) |d, i| shape[i] = d;
    return .{ .buffer = buf, .offset = 0, .bytes = bytes, .dtype = .bf16, .shape = shape, .rank = dims.len };
}

fn allocF32(device: mtl.Device, list: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator, rng: std.Random, dims: []const usize, scale: f32) !ck.Tensor {
    var count: usize = 1;
    for (dims) |d| count *= d;
    const bytes = count * 4;
    const buf = try device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const dst = buf.slice(f32, count);
    for (dst) |*v| v.* = (rng.float(f32) * 2.0 - 1.0) * scale;
    try list.append(gpa, buf);
    var shape: [core.safetensors.max_rank]usize = @splat(1);
    for (dims, 0..) |d, i| shape[i] = d;
    return .{ .buffer = buf, .offset = 0, .bytes = bytes, .dtype = .f32, .shape = shape, .rank = dims.len };
}

fn makeExl3Matrix(device: mtl.Device, list: *std.ArrayList(mtl.Buffer), gpa: std.mem.Allocator, rng: std.Random, k: usize, n: usize, bits: u16) !exl3_proj.ExpertDescriptor {
    const halves: u16 = bits * 2;
    const words = (k / 16) * (n / 16) * (@as(usize, halves) * 8);
    const t_buf = try device.buffer(words * 2, mtl.ResourceOptions.shared);
    errdefer t_buf.deinit();
    for (t_buf.slice(u16, words)) |*w| w.* = rng.int(u16);
    try list.append(gpa, t_buf);

    const su_buf = try device.buffer(k * 2, mtl.ResourceOptions.shared);
    errdefer su_buf.deinit();
    for (su_buf.slice(f16, k)) |*s| s.* = @floatCast((rng.float(f32) * 0.2 - 0.1));
    try list.append(gpa, su_buf);

    const sv_buf = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    errdefer sv_buf.deinit();
    for (sv_buf.slice(f16, n)) |*s| s.* = @floatCast((rng.float(f32) * 0.2 - 0.1));
    try list.append(gpa, sv_buf);

    return exl3_proj.descriptor(.{
        .trellis = .{ .buffer = t_buf, .offset = 0 },
        .suh = .{ .buffer = su_buf, .offset = 0 },
        .svh = .{ .buffer = sv_buf, .offset = 0 },
        .k = k,
        .n = n,
        .width = .{ .halves = halves },
        .codebook = .mul1,
        .suh_format = .f16,
        .svh_format = .f16,
    });
}

fn buildSyntheticModel(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !*model_file.Model {
    const n_layers: u32 = 6; // Layers 0..4 local + layer 5 global (tests block boundary, dynamic conv, tap 0 & final).
    const vocab: u32 = 64;
    var prng = std.Random.DefaultPrng.init(0x5094_2026);
    const rng = prng.random();

    const m = try gpa.create(model_file.Model);
    errdefer gpa.destroy(m);
    m.gpa = gpa;
    m.device = device;
    m.queue = queue;
    m.checkpoint = ck.Checkpoint.init(gpa);
    errdefer m.checkpoint.deinit();
    m.expert_buffers = .empty;
    errdefer {
        for (m.expert_buffers.items) |b| b.deinit();
        m.expert_buffers.deinit(gpa);
    }

    m.config = .{
        .quant = .exl3_kortexa_v1,
        .text = .{
            .hidden = state.hidden,
            .layers = n_layers,
            .vocab = vocab,
            .unpadded_vocab = vocab,
            .local = .{ .q = state.heads, .kv = state.kv_heads, .dim = 128 },
            .global = .{ .q = state.heads, .kv = state.kv_heads, .dim = 128 },
            .window = 1024,
            .d_rel = 16,
            .rel_extent = 1024,
            .log_floor = 8192,
            .log_alpha = 0.1,
            .eps = 1e-6,
            .qk_clip = 22.627416997969522,
            .rel_clip = 8,
            .block = 6,
            .experts = state.expert_count,
            .topk = state.experts_per_token,
            .expert_inter = state.intermediate,
            .route_scale = 8,
            .gate_cap = 4,
            .up_cap = 25,
            .logit_cap = 30,
            .max_context = 4096,
            .loop = null,
        },
        .vision = .{
            .hidden = 1,
            .layers = 0,
            .heads = 1,
            .mlp_ratio = 1,
            .patch = 1,
            .grid = 1,
            .merge = 1,
            .proj_hidden = 1,
            .text_hidden = 1,
            .eps = 1e-6,
        },
        .mtp = null,
        .eos = 0,
        .image_token = 1,
    };

    var precs: [n_layers]layer_plan.Precision = @splat(.exl3);
    m.weight_plan = try layer_plan.build(&m.config.text, &precs);

    // Share compact random dense/expert weight buffers across layers to keep test RAM under ~100 MiB.
    var shared_bufs: std.ArrayList(mtl.Buffer) = .empty;
    errdefer {
        for (shared_bufs.items) |b| b.deinit();
        shared_bufs.deinit(gpa);
    }

    const h = state.hidden;
    const q = state.query_width;
    const kv = state.kv_width;
    const rel = state.rel_width;
    const embed = try allocBf16(device, &shared_bufs, gpa, rng, &.{ vocab, h }, 0.2);
    const embed_norm = try allocBf16(device, &shared_bufs, gpa, rng, &.{h}, 0.5);
    const final_norm = try allocBf16(device, &shared_bufs, gpa, rng, &.{h}, 0.5);
    const unembed = try allocBf16(device, &shared_bufs, gpa, rng, &.{ vocab, h }, 0.05);
    const out_q = try allocF32(device, &shared_bufs, gpa, rng, &.{h}, 0.1);
    const attn_q1 = try allocF32(device, &shared_bufs, gpa, rng, &.{h}, 0.1);
    const attn_q3 = try allocF32(device, &shared_bufs, gpa, rng, &.{ 3, h }, 0.1);
    const mlp_q = try allocF32(device, &shared_bufs, gpa, rng, &.{h}, 0.1);

    const wq = try allocBf16(device, &shared_bufs, gpa, rng, &.{ q, h }, 0.02);
    const wk = try allocBf16(device, &shared_bufs, gpa, rng, &.{ kv, h }, 0.02);
    const wv = try allocBf16(device, &shared_bufs, gpa, rng, &.{ kv, h }, 0.02);
    const wr = try allocBf16(device, &shared_bufs, gpa, rng, &.{ rel, h }, 0.02);
    const wo = try allocBf16(device, &shared_bufs, gpa, rng, &.{ h, q }, 0.02);
    const gate = try allocBf16(device, &shared_bufs, gpa, rng, &.{ q, h }, 0.02);
    const sink = try allocBf16(device, &shared_bufs, gpa, rng, &.{ state.heads, h }, 0.02);
    const q_scale = try allocBf16(device, &shared_bufs, gpa, rng, &.{ state.heads, 128 }, 0.5);
    const rel_scale = try allocBf16(device, &shared_bufs, gpa, rng, &.{ state.heads, 16 }, 0.5);
    const rel_proj = try allocBf16(device, &shared_bufs, gpa, rng, &.{ 16, 1024 }, 0.05);
    const kv_sconv = try allocBf16(device, &shared_bufs, gpa, rng, &.{ kv, 1, 4 }, 0.1);
    const kv_dstat = try allocBf16(device, &shared_bufs, gpa, rng, &.{ 4, kv }, 0.1);
    const kv_dproj = try allocBf16(device, &shared_bufs, gpa, rng, &.{ 32, h }, 0.02);
    const h_norm = try allocBf16(device, &shared_bufs, gpa, rng, &.{h}, 0.5);
    const h_sconv = try allocBf16(device, &shared_bufs, gpa, rng, &.{ h, 1, 4 }, 0.1);
    const r_w = try allocBf16(device, &shared_bufs, gpa, rng, &.{ state.expert_count, h }, 0.02);
    const r_b = try allocF32(device, &shared_bufs, gpa, rng, &.{state.expert_count}, 0.1);

    // Three mixed-width EXL3 mul1 expert templates (2-bit, 3-bit, 4-bit) referenced by all 384 descriptors.
    var w13_tpl: [3]exl3_proj.ExpertDescriptor = undefined;
    var w2_tpl: [3]exl3_proj.ExpertDescriptor = undefined;
    const exl3_start = shared_bufs.items.len;
    for ([_]u16{ 2, 3, 4 }, 0..) |bits, bi| {
        w13_tpl[bi] = try makeExl3Matrix(device, &shared_bufs, gpa, rng, h, 2 * state.intermediate, bits);
        w2_tpl[bi] = try makeExl3Matrix(device, &shared_bufs, gpa, rng, state.intermediate, h, bits);
    }
    const exl3_end = shared_bufs.items.len;

    const attn_queries = try gpa.alloc(ck.Tensor, n_layers - 1);
    errdefer gpa.free(attn_queries);
    for (attn_queries, 0..) |*aq, i| aq.* = if (cfg.isGlobal(@intCast(i + 1))) attn_q3 else attn_q1;

    const mlp_queries = try gpa.alloc(ck.Tensor, n_layers);
    errdefer gpa.free(mlp_queries);
    @memset(mlp_queries, mlp_q);

    const layers = try gpa.alloc(model_file.LayerWeights, n_layers);
    errdefer gpa.free(layers);

    var built: usize = 0;
    errdefer for (layers[0..built]) |*lw| {
        lw.experts.w13_descs.deinit();
        lw.experts.w2_descs.deinit();
        gpa.free(lw.experts.shard_ids);
    };

    const desc_bytes = state.expert_count * @sizeOf(exl3_proj.ExpertDescriptor);
    for (layers, 0..) |*lw, i| {
        const global = cfg.isGlobal(@intCast(i));
        const w13_buf = try device.buffer(desc_bytes, mtl.ResourceOptions.shared);
        errdefer w13_buf.deinit();
        const w2_buf = try device.buffer(desc_bytes, mtl.ResourceOptions.shared);
        errdefer w2_buf.deinit();
        const w13_s = w13_buf.slice(exl3_proj.ExpertDescriptor, state.expert_count);
        const w2_s = w2_buf.slice(exl3_proj.ExpertDescriptor, state.expert_count);
        for (0..state.expert_count) |e| {
            w13_s[e] = w13_tpl[(e + i) % 3];
            w2_s[e] = w2_tpl[(e + i) % 3];
        }
        const ids = try gpa.alloc(mtl.objc.Id, exl3_end - exl3_start);
        for (shared_bufs.items[exl3_start..exl3_end], 0..) |b, j| ids[j] = b.id;

        const conv: model_file.Conv = if (global)
            .{ .dynamic = .{ .static_w = kv_dstat, .projection = kv_dproj } }
        else
            .{ .static = kv_sconv };
        lw.* = .{
            .attn = .{
                .wq = wq,
                .wk = wk,
                .wv = wv,
                .wr = wr,
                .wo = wo,
                .gate = gate,
                .sink = sink,
                .q_scale = q_scale,
                .rel_scale = rel_scale,
                .rel_projection = rel_proj,
                .k_conv = conv,
                .v_conv = conv,
                .q_norm = h_norm,
                .k_norm = h_norm,
                .v_norm = h_norm,
            },
            .attn_conv = h_sconv,
            .mlp_norm = h_norm,
            .mlp_conv = h_sconv,
            .router_weight = r_w,
            .router_bias = r_b,
            .experts = .{ .w13_descs = w13_buf, .w2_descs = w2_buf, .shard_ids = ids, .count = state.expert_count },
        };
        built += 1;
    }

    // Transfer ownership of all shared synthetic buffers to m.expert_buffers so deinit frees them once.
    m.expert_buffers = shared_bufs;
    m.weights = .{
        .embed = embed,
        .embed_norm = embed_norm,
        .norm = final_norm,
        .unembed = unembed,
        .output_query = out_q,
        .attn_queries = attn_queries,
        .mlp_queries = mlp_queries,
        .layers = layers,
    };
    m.pipes = try model_file.Pipelines.load(gpa, device);
    return m;
}

test "inkling2 small: 6-layer EXL3 GPU forward chunk == decode, commit/rollback, and shared == solo are bit-exact" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const queue = device.queue() catch {
        device.deinit();
        return error.SkipZigTest;
    };
    const gpa = t.allocator;
    const m = try buildSyntheticModel(gpa, device, queue);
    defer m.deinit();

    const vocab = m.config.text.unpadded_vocab;
    const tokens = [_]u32{ 3, 17, 42, 7, 55, 12 };

    // 1. Sequential 1-token decode baseline (with taps).
    var s1 = try state.Scratch.initFor(device, 8, vocab);
    defer s1.deinit();
    var c_seq = try state.Cache.initFor(device, 64, &m.weight_plan, 8, vocab);
    defer c_seq.deinit();
    const seq_taps = try device.buffer(state.captureBytes(tokens.len), mtl.ResourceOptions.shared);
    defer seq_taps.deinit();
    @memset(seq_taps.slice(u16, tokens.len * state.feature_taps * state.hidden), 0);
    var seq_logits: [tokens.len][64]u32 = undefined;
    for (tokens, 0..) |tok, i| {
        try fwd.run(m, &s1, &c_seq, &.{tok}, .{ .all = true, .taps = .{ .buf = seq_taps, .off = i * state.feature_row_bytes } });
        c_seq.len += 1;
        @memcpy(&seq_logits[i], s1.logits_all.slice(u32, vocab));
        // Confirm non-trivial finite logits.
        var any_nonzero = false;
        for (s1.logits_all.slice(f32, vocab)) |v| {
            try t.expect(std.math.isFinite(v));
            if (v != 0) any_nonzero = true;
        }
        try t.expect(any_nonzero);
    }

    // 2. Single 6-token prompt chunk must match sequential 1-token decode bit-for-bit on all logits and taps.
    var c_chunk = try state.Cache.initFor(device, 64, &m.weight_plan, 8, vocab);
    defer c_chunk.deinit();
    const chunk_taps = try device.buffer(state.captureBytes(tokens.len), mtl.ResourceOptions.shared);
    defer chunk_taps.deinit();
    @memset(chunk_taps.slice(u16, tokens.len * state.feature_taps * state.hidden), 0);
    try fwd.run(m, &s1, &c_chunk, &tokens, .{ .all = true, .taps = .{ .buf = chunk_taps, .off = 0 } });
    c_chunk.len += tokens.len;
    for (0..tokens.len) |i| {
        const got = s1.logits_all.slice(u32, tokens.len * vocab)[i * vocab ..][0..vocab];
        try t.expectEqualSlices(u32, &seq_logits[i], got);
    }
    try t.expectEqualSlices(u16, seq_taps.slice(u16, tokens.len * state.feature_taps * state.hidden), chunk_taps.slice(u16, tokens.len * state.feature_taps * state.hidden));

    // 3. Speculative transaction: prefill 2, begin(4) + commit(2), begin(2) + rollback, then finish 2.
    var c_spec = try state.Cache.initFor(device, 64, &m.weight_plan, 8, vocab);
    defer c_spec.deinit();
    try fwd.run(m, &s1, &c_spec, tokens[0..2], .{ .all = true });
    c_spec.len += 2;
    try c_spec.begin(queue, 4);
    try fwd.run(m, &s1, &c_spec, &.{ tokens[2], tokens[3], 61, 62 }, .{ .all = true });
    try c_spec.commit(queue, 2);
    try c_spec.begin(queue, 2);
    try fwd.run(m, &s1, &c_spec, &.{ 59, 60 }, .{ .all = true });
    try c_spec.rollback(queue);
    try fwd.run(m, &s1, &c_spec, tokens[4..6], .{ .all = true });
    c_spec.len += 2;
    try t.expectEqualSlices(u32, &seq_logits[4], s1.logits_all.slice(u32, 2 * vocab)[0..vocab]);
    try t.expectEqualSlices(u32, &seq_logits[5], s1.logits_all.slice(u32, 2 * vocab)[vocab .. 2 * vocab]);

    // 4. Shared multi-stream batch (stream A: 2 tokens, stream B: 4 tokens) matches solo bit-for-bit.
    var c_a = try state.Cache.initFor(device, 64, &m.weight_plan, 8, vocab);
    defer c_a.deinit();
    var c_b = try state.Cache.initFor(device, 64, &m.weight_plan, 8, vocab);
    defer c_b.deinit();
    try fwd.run(m, &s1, &c_a, tokens[0..2], .{ .all = false });
    c_a.len += 2;
    try fwd.run(m, &s1, &c_b, tokens[0..2], .{ .all = false });
    c_b.len += 2;
    const batch_ids = [_]u32{ tokens[2], tokens[3], tokens[2], tokens[3], tokens[4], tokens[5] };
    try fwd.runShared(m, &s1, &.{ .{ .cache = &c_a, .rows = 2 }, .{ .cache = &c_b, .rows = 4 } }, &batch_ids, .{ .all = true });
    const batch_out = s1.logits_all.slice(u32, 6 * vocab);
    try t.expectEqualSlices(u32, &seq_logits[2], batch_out[0 * vocab ..][0..vocab]);
    try t.expectEqualSlices(u32, &seq_logits[3], batch_out[1 * vocab ..][0..vocab]);
    try t.expectEqualSlices(u32, &seq_logits[2], batch_out[2 * vocab ..][0..vocab]);
    try t.expectEqualSlices(u32, &seq_logits[3], batch_out[3 * vocab ..][0..vocab]);
    try t.expectEqualSlices(u32, &seq_logits[4], batch_out[4 * vocab ..][0..vocab]);
    try t.expectEqualSlices(u32, &seq_logits[5], batch_out[5 * vocab ..][0..vocab]);
}
