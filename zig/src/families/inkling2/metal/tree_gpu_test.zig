//! Real Metal GPU tree compaction, 48-visit differential, and lanes Engine C2/C7 conformance tests.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const layer_plan = @import("../plan.zig");
const tree = @import("tree.zig");
const st = @import("state.zig");
const backend_ = @import("backend.zig");
const synth = @import("synthetic_model_test.zig");
const tp = @import("inkling2_tree_plan");

const gpa = std.testing.allocator;
const taps_ids = [_]u32{ 5, 23, 41, 48 };

fn blitCopy(queue: mtl.Queue, src: mtl.Buffer, src_off: usize, dst: mtl.Buffer, dst_off: usize, bytes: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    e.copy(src, src_off, dst, dst_off, bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

test "Cache beginTree and commitPath compact sliding-wrap KV and all 4 conv histories" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var tree_pipes = try tree.Pipelines.load(device);
    defer tree_pipes.deinit();

    var precisions: [24]layer_plan.Precision = undefined;
    for (&precisions, 0..) |*p, i| p.* = if (i < 20) .nvfp4 else .mxfp8;
    const text_cfg = synth.syntheticConfig().text;
    const wp = try layer_plan.build(&text_cfg, &precisions);
    var cache = try st.Cache.init(device, 1200, &wp);
    defer cache.deinit();
    cache.len = 1086;

    const kv_stage = try device.buffer(st.kvBytes(1200), mtl.ResourceOptions.shared);
    defer kv_stage.deinit();
    const kv_got = try device.buffer(st.kvBytes(1200), mtl.ResourceOptions.shared);
    defer kv_got.deinit();
    const kv_u16 = kv_stage.slice(u16, st.kvBytes(1200) / 2);
    for (kv_u16, 0..) |*v, i| v.* = @truncate((i * 131 + 7) & 0x3fff | 0x3800);
    try blitCopy(queue, kv_stage, 0, cache.kv, 0, st.kvBytes(1200));

    const conv_stage = try device.buffer(st.conv_bytes, mtl.ResourceOptions.shared);
    defer conv_stage.deinit();
    const conv_got = try device.buffer(st.conv_bytes, mtl.ResourceOptions.shared);
    defer conv_got.deinit();
    const conv_u16 = conv_stage.slice(u16, st.conv_bytes / 2);
    for (conv_u16, 0..) |*v, i| v.* = @truncate((i * 73 + 3) & 0x3fff | 0x3800);
    try blitCopy(queue, conv_stage, 0, cache.conv, 0, st.conv_bytes);

    const taps_stage = try device.buffer(st.taps_bytes, mtl.ResourceOptions.shared);
    defer taps_stage.deinit();
    const taps_u16 = taps_stage.slice(u16, st.taps_bytes / 2);
    for (taps_u16, 0..) |*v, i| v.* = @truncate((i * 97 + 19) & 0x3fff | 0x3800);
    try blitCopy(queue, taps_stage, 0, cache.taps, 0, st.taps_bytes);

    const parents = [_]i32{ -1, 0, 0, 1, 2, 4 };
    const path = [_]u32{ 0, 2, 4, 5 };
    try cache.beginTree(6, &parents);
    try cache.commitPath(queue, &path, &tree_pipes);
    try std.testing.expectEqual(@as(usize, 1090), cache.len);
    try std.testing.expect(cache.pending == null);

    try blitCopy(queue, cache.kv, 0, kv_got, 0, st.kvBytes(1200));
    try blitCopy(queue, cache.conv, 0, conv_got, 0, st.conv_bytes);
    const got_kv = kv_got.slice(u16, st.kvBytes(1200) / 2);
    const got_conv = conv_got.slice(u16, st.conv_bytes / 2);

    for (0..st.visit_slots) |slot| {
        const ring = cache.ring(slot);
        for ([_]usize{ cache.key_at[slot], cache.value_at[slot] }) |base| for (0..8) |h| {
            const h_base = base / 2 + h * ring * 128;
            for (path, 0..) |p, j| {
                const src_r = (1086 + p) % ring;
                const dst_r = (1086 + j) % ring;
                try std.testing.expectEqualSlices(u16, kv_u16[h_base + src_r * 128 ..][0..128], got_kv[h_base + dst_r * 128 ..][0..128]);
            }
        };
        const want_src = tp.convHistorySrc(&path);
        for (0..st.conv_states) |cs| {
            const w = st.convWidth(cs);
            for (0..3) |out_r| {
                const dst_off = st.convByteOffset(slot, cs, out_r, 3) / 2;
                const idx = want_src[out_r];
                const want_slice = if (idx < 3)
                    conv_u16[st.convByteOffset(slot, cs, idx, 3) / 2 ..][0..w]
                else
                    taps_u16[st.convByteOffset(slot, cs, idx - 3, st.window_rows) / 2 ..][0..w];
                try std.testing.expectEqualSlices(u16, want_slice, got_conv[dst_off..][0..w]);
            }
        }
    }

    try cache.beginTree(4, &.{ -1, 0, 0, 1 });
    try cache.rollback();
    try blitCopy(queue, cache.conv, 0, conv_got, 0, st.conv_bytes);
    for (0..st.visit_slots) |slot| for (0..st.conv_states) |cs| {
        const w = st.convWidth(cs);
        const want_src = tp.convHistorySrc(&path);
        for (0..3) |out_r| {
            const dst_off = st.convByteOffset(slot, cs, out_r, 3) / 2;
            const idx = want_src[out_r];
            const want_slice = if (idx < 3)
                conv_u16[st.convByteOffset(slot, cs, idx, 3) / 2 ..][0..w]
            else
                taps_u16[st.convByteOffset(slot, cs, idx - 3, st.window_rows) / 2 ..][0..w];
            try std.testing.expectEqualSlices(u16, want_slice, got_conv[dst_off..][0..w]);
        }
    };
}

fn syntheticConfig() cfg.Config {
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

pub fn buildSyntheticModel() !*model_.Model {
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

pub fn expectLogitsBitsEqual(want: []const f32, got: []const f32) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got) |a, b| try std.testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
}

fn checkTreeBranches(target: *backend_.Metal, prompt_len: usize) !void {
    const vocab = target.model.config.text.unpadded_vocab;
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps_ids);
    const prompt = try gpa.alloc(u32, prompt_len);
    defer gpa.free(prompt);
    for (prompt, 0..) |*t, i| t.* = @intCast(5 + (i * 17) % 180);

    var base_s = try lanes.Stream.init(gpa, .{ .id = "base", .prompt = prompt, .max_new = 16 });
    defer base_s.deinit(gpa);
    try b.prefill(&base_s);
    defer b.release(&base_s);
    const tok0 = try b.read(try b.first(&base_s, prompt_len));
    const snap = try backend_.Metal.snapSave(target, &base_s, @intCast(prompt_len));
    defer backend_.Metal.snapDrop(target, snap);

    const parents = [_]i32{ -1, 0, 0, 1, 2, 4 };
    const tokens = [_]u32{ 17, 29, 43, 61, 83 };
    const tree_inputs = [_]u32{ tok0, 17, 29, 43, 61, 83 };
    const positions = [_]u64{ prompt_len + 1, prompt_len + 2, prompt_len + 2, prompt_len + 3, prompt_len + 3, prompt_len + 4 };
    const paths = [_][]const u32{ &.{0}, &.{ 0, 1 }, &.{ 0, 2 }, &.{ 0, 1, 3 }, &.{ 0, 2, 4, 5 } };

    for (paths) |path| {
        var tree_s = try lanes.Stream.init(gpa, .{ .id = "tree", .prompt = prompt, .max_new = 16, .reuse = .{ .saved = snap, .at = @intCast(prompt_len) } });
        defer tree_s.deinit(gpa);
        try b.prefill(&tree_s);
        defer b.release(&tree_s);

        var seq_s = try lanes.Stream.init(gpa, .{ .id = "seq", .prompt = prompt, .max_new = 16, .reuse = .{ .saved = snap, .at = @intCast(prompt_len) } });
        defer seq_s.deinit(gpa);
        try b.prefill(&seq_s);
        defer b.release(&seq_s);

        var sampled: [6]u32 = undefined;
        var drafts: [5]u32 = undefined;
        var out = [_]lanes.backend.Verified{.{ .sampled = &sampled, .drafts = &drafts }};
        const win = [_]lanes.backend.Window{.{ .stream = &tree_s, .pending = tok0, .held = 0, .tokens = &tokens, .parents = &parents, .positions = &positions }};
        try b.verify(&win, &out);
        const verify_logits = try gpa.dupe(f32, target.scratch.logits_all.slice(f32, 6 * vocab));
        defer gpa.free(verify_logits);
        try std.testing.expect(!std.mem.eql(f32, verify_logits[0..vocab], verify_logits[5 * vocab ..][0..vocab]));
        var min_v: f32 = std.math.inf(f32);
        var max_v: f32 = -std.math.inf(f32);
        for (verify_logits[0..256]) |v| {
            try std.testing.expect(std.math.isFinite(v));
            min_v = @min(min_v, v);
            max_v = @max(max_v, v);
        }
        try std.testing.expect(max_v > min_v);

        try b.keep(&win, &.{path});
        const tree_cache = target.caches.get(&tree_s).?;
        const tree_kept_logits = try gpa.dupe(f32, tree_cache.logits.slice(f32, vocab));
        defer gpa.free(tree_kept_logits);
        const f_tree = try b.features(&tree_s, &taps_ids, prompt_len + path[path.len - 1], 1);
        const f_tree_bytes = try gpa.dupe(u8, (@as(mtl.Buffer, .{ .id = @ptrFromInt(f_tree.buffer) })).slice(u8, @intCast(f_tree.offset + f_tree.row_bytes))[@intCast(f_tree.offset)..][0..st.feature_row_bytes]);
        defer gpa.free(f_tree_bytes);

        const seq_cache = target.caches.get(&seq_s).?;
        var seq_tok: u32 = 0;
        for (path, 0..) |node, j| {
            seq_tok = try b.read(try b.queue(&seq_s, .{ .value = tree_inputs[node] }, prompt_len + 1 + j));
            try std.testing.expectEqual(seq_tok, sampled[node]);
            try expectLogitsBitsEqual(seq_cache.logits.slice(f32, vocab), verify_logits[node * vocab ..][0..vocab]);
        }
        try expectLogitsBitsEqual(seq_cache.logits.slice(f32, vocab), tree_kept_logits);
        const f_seq = try b.features(&seq_s, &taps_ids, prompt_len + path.len - 1, 1);
        const f_seq_bytes = (@as(mtl.Buffer, .{ .id = @ptrFromInt(f_seq.buffer) })).slice(u8, @intCast(f_seq.offset + f_seq.row_bytes))[@intCast(f_seq.offset)..][0..st.feature_row_bytes];
        try std.testing.expectEqualSlices(u8, f_seq_bytes, f_tree_bytes);

        const cont_tok: u32 = 97;
        const t_next = try b.read(try b.queue(&tree_s, .{ .value = cont_tok }, prompt_len + path.len + 1));
        const s_next = try b.read(try b.queue(&seq_s, .{ .value = cont_tok }, prompt_len + path.len + 1));
        try std.testing.expectEqual(s_next, t_next);
        try expectLogitsBitsEqual(seq_cache.logits.slice(f32, vocab), tree_cache.logits.slice(f32, vocab));
    }

    var tree_neg = try lanes.Stream.init(gpa, .{ .id = "neg", .prompt = prompt, .max_new = 8, .reuse = .{ .saved = snap, .at = @intCast(prompt_len) } });
    defer tree_neg.deinit(gpa);
    try b.prefill(&tree_neg);
    defer b.release(&tree_neg);
    var neg_sampled: [3]u32 = undefined;
    var neg_drafts: [2]u32 = undefined;
    var neg_out = [_]lanes.backend.Verified{.{ .sampled = &neg_sampled, .drafts = &neg_drafts }};
    try std.testing.expectError(error.BadParents, b.verify(&.{.{ .stream = &tree_neg, .pending = tok0, .held = 0, .tokens = &.{ 1, 2 }, .parents = &.{ -1, 0, 2 }, .positions = &.{ prompt_len + 1, prompt_len + 2, prompt_len + 2 } }}, &neg_out));
    try std.testing.expectError(error.InvalidInklingPosition, b.verify(&.{.{ .stream = &tree_neg, .pending = tok0, .held = 0, .tokens = &.{ 1, 2 }, .parents = &.{ -1, 0, 0 }, .positions = &.{ prompt_len + 1, prompt_len + 2, prompt_len + 3 } }}, &neg_out));
    const ok_win = [_]lanes.backend.Window{.{ .stream = &tree_neg, .pending = tok0, .held = 0, .tokens = &.{ 1, 2 }, .parents = &.{ -1, 0, 0 }, .positions = &.{ prompt_len + 1, prompt_len + 2, prompt_len + 2 } }};
    try b.verify(&ok_win, &neg_out);
    try std.testing.expectError(error.UnsupportedInklingWindow, b.keep(&ok_win, &.{&.{ 0, 1, 2 }}));
}

fn checkMixedBatchAndTreeC2C7(target: *backend_.Metal) !void {
    const vocab = target.model.config.text.unpadded_vocab;
    const b = target.backend();
    const prompt = [_]u32{ 11, 22, 33, 44, 11, 22, 55, 66, 11, 22, 33, 44, 11, 22, 77, 88 };

    var chain_s = try lanes.Stream.init(gpa, .{ .id = "chain_mix", .prompt = &prompt, .max_new = 8 });
    defer chain_s.deinit(gpa);
    var tree_s = try lanes.Stream.init(gpa, .{ .id = "tree_mix", .prompt = &prompt, .max_new = 8 });
    defer tree_s.deinit(gpa);
    var ref_s = try lanes.Stream.init(gpa, .{ .id = "ref_mix", .prompt = &prompt, .max_new = 8 });
    defer ref_s.deinit(gpa);
    try b.prefill(&chain_s);
    defer b.release(&chain_s);
    try b.prefill(&tree_s);
    defer b.release(&tree_s);
    try b.prefill(&ref_s);
    defer b.release(&ref_s);
    const ptok = try b.read(try b.first(&chain_s, prompt.len));
    _ = try b.read(try b.first(&tree_s, prompt.len));
    _ = try b.read(try b.first(&ref_s, prompt.len));

    var s0: [2]u32 = undefined;
    var d0: [1]u32 = undefined;
    var s1: [4]u32 = undefined;
    var d1: [3]u32 = undefined;
    const mix = [_]lanes.backend.Window{
        .{ .stream = &chain_s, .pending = ptok, .held = 0, .tokens = &.{19}, .parents = null, .positions = &.{ prompt.len + 1, prompt.len + 2 } },
        .{ .stream = &tree_s, .pending = ptok, .held = 0, .tokens = &.{ 25, 37, 51 }, .parents = &.{ -1, 0, 0, 2 }, .positions = &.{ prompt.len + 1, prompt.len + 2, prompt.len + 2, prompt.len + 3 } },
    };
    var mix_out = [_]lanes.backend.Verified{ .{ .sampled = &s0, .drafts = &d0 }, .{ .sampled = &s1, .drafts = &d1 } };
    try b.verify(&mix, &mix_out);
    try b.keep(&mix, &.{ &.{ 0, 1 }, &.{ 0, 2, 3 } });

    _ = try b.read(try b.queue(&ref_s, .{ .value = ptok }, prompt.len + 1));
    _ = try b.read(try b.queue(&ref_s, .{ .value = 37 }, prompt.len + 2));
    _ = try b.read(try b.queue(&ref_s, .{ .value = 51 }, prompt.len + 3));
    const tc = target.caches.get(&tree_s).?;
    const rc = target.caches.get(&ref_s).?;
    try expectLogitsBitsEqual(rc.logits.slice(f32, vocab), tc.logits.slice(f32, vocab));
    const nt = try b.read(try b.queue(&tree_s, .{ .value = 71 }, prompt.len + 4));
    const nr = try b.read(try b.queue(&ref_s, .{ .value = 71 }, prompt.len + 4));
    try std.testing.expectEqual(nr, nt);
    try expectLogitsBitsEqual(rc.logits.slice(f32, vocab), tc.logits.slice(f32, vocab));
}

const MiniTreeDrafter = struct {
    oracle: []const u32,
    prompt_len: usize,
    history: std.AutoHashMapUnmanaged(*lanes.Stream, std.ArrayList(u32)) = .empty,
    held_toks: std.AutoHashMapUnmanaged(*lanes.Stream, [5]u32) = .empty,
    held_len: std.AutoHashMapUnmanaged(*lanes.Stream, usize) = .empty,
    saw_non_prefix_rows: bool = false,

    fn drafter(x: *MiniTreeDrafter) lanes.drafter.Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release, .tree = treeFn } };
    }
    fn self(ptr: *anyopaque) *MiniTreeDrafter {
        return @ptrCast(@alignCast(ptr));
    }
    fn taps(_: *anyopaque) []const u32 {
        return &taps_ids;
    }
    fn facts(_: *anyopaque) lanes.drafter.Facts {
        return .{ .depth = 5, .step_ms = 0.01, .tree_nodes = 5, .prior = &.{ 0.9, 0.9, 0.9, 0.9, 0.9 } };
    }
    fn open(ptr: *anyopaque, s: *lanes.Stream) !void {
        const x = self(ptr);
        try x.history.put(gpa, s, .empty);
        try x.held_toks.put(gpa, s, .{ 1, 2, 3, 4, 5 });
        try x.held_len.put(gpa, s, 0);
    }
    fn absorb(ptr: *anyopaque, items: []const lanes.drafter.Absorb) !void {
        const x = self(ptr);
        for (items) |it| {
            if (it.features.space != .device or it.features.dtype != .bf16 or it.features.row_bytes != st.feature_row_bytes) return error.Shape;
            if (it.rows) |rows| {
                if (rows.len != it.follow.len or it.features.rows < rows[rows.len - 1] + 1) return error.Shape;
                for (rows, 0..) |rv, idx| if (rv != idx) {
                    x.saw_non_prefix_rows = true;
                };
            }
            const h = x.history.getPtr(it.stream).?;
            while (h.items.len < it.start) try h.append(gpa, 0);
            h.shrinkRetainingCapacity(@intCast(it.start));
            try h.appendSlice(gpa, it.follow);
        }
    }
    fn hold(ptr: *anyopaque, items: []const lanes.drafter.Hold) !void {
        const x = self(ptr);
        for (items) |it| {
            const p = it.pending;
            const step = it.position -| x.prompt_len;
            const g1 = if (step < x.oracle.len) x.oracle[step] else (p + 1) % 100;
            const g2 = if (step + 1 < x.oracle.len) x.oracle[step + 1] else (p + 2) % 100;
            x.held_toks.getPtr(it.stream).?.* = .{ (g1 + 7) % 100, g1, (g1 + 11) % 100, g2, (g2 + 13) % 100 };
            x.held_len.getPtr(it.stream).?.* = @min(@as(usize, it.depth), 5);
        }
    }
    fn held(ptr: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) !void {
        const x = self(ptr);
        for (streams, out) |s, dst| @memcpy(dst, x.held_toks.get(s).?[0..dst.len]);
    }
    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, alloc: std.mem.Allocator) !?lanes.stream.Held {
        const x = self(ptr);
        const n = x.held_len.get(s) orelse 0;
        if (n < 5) return null;
        const src = x.held_toks.get(s) orelse return null;
        const toks = try alloc.dupe(u32, src[0..5]);
        errdefer alloc.free(toks);
        const ps = try alloc.dupe(i32, &.{ -1, -1, 0, 1, 3 });
        return .{ .count = 5, .tokens = toks, .parents = ps };
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const x = self(ptr);
        if (x.history.fetchRemove(s)) |kv| {
            var list = kv.value;
            list.deinit(gpa);
        }
        _ = x.held_toks.remove(s);
        _ = x.held_len.remove(s);
    }
    fn deinit(x: *MiniTreeDrafter) void {
        var it = x.history.valueIterator();
        while (it.next()) |l| l.deinit(gpa);
        x.history.deinit(gpa);
        x.held_toks.deinit(gpa);
        x.held_len.deinit(gpa);
    }
};

fn checkLanesConformance(target: *backend_.Metal) !void {
    const repeat_prompt = [_]u32{ 11, 22, 33, 11, 22, 44, 11, 22, 55, 11, 22, 33, 11, 22, 44, 11, 22 };
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 1.0 + 0.01 * @as(f64, @floatFromInt(w)) };
    var plain_cfg = try lanes.Config.init(gpa, target.facts(), 16, 16);
    defer plain_cfg.deinit(gpa);

    var plain_clk: lanes.fake.FixedClock = .{};
    var plain_eng = lanes.Engine.init(gpa, &plain_cfg, target.backend(), plain_clk.clock());
    defer plain_eng.deinit();
    var ps = try lanes.Stream.init(gpa, .{ .id = "c_plain", .prompt = &repeat_prompt, .max_new = 10 });
    defer ps.deinit(gpa);
    try plain_eng.addStream(&ps);
    while (plain_eng.activeCount() > 0) try plain_eng.step();

    var drafter: MiniTreeDrafter = .{ .oracle = ps.emitted(), .prompt_len = repeat_prompt.len };
    defer drafter.deinit();
    var wrapped = try lanes.drafted.Drafted.init(gpa, target.backend(), drafter.drafter());
    defer wrapped.deinit();
    var tree_model = wrapped.facts(target.facts());
    tree_model.window_costs = &costs;
    var tree_cfg = try lanes.Config.init(gpa, tree_model, 16, 16);
    defer tree_cfg.deinit(gpa);
    tree_cfg.fill_lanes = 8;

    var half_s = try lanes.Stream.init(gpa, .{ .id = "c_snap", .prompt = repeat_prompt[0..8], .max_new = 10 });
    defer half_s.deinit(gpa);
    try target.backend().prefill(&half_s);
    defer target.backend().release(&half_s);
    const snap = try backend_.Metal.snapSave(target, &half_s, 8);
    defer backend_.Metal.snapDrop(target, snap);

    var tree_clk: lanes.fake.FixedClock = .{};
    var tree_eng = lanes.Engine.init(gpa, &tree_cfg, wrapped.backend(), tree_clk.clock());
    defer tree_eng.deinit();
    var ts = try lanes.Stream.init(gpa, .{ .id = "c_tree", .prompt = &repeat_prompt, .max_new = 10, .drafts = true });
    defer ts.deinit(gpa);
    try tree_eng.addStream(&ts);
    while (tree_eng.activeCount() > 0) try tree_eng.step();

    var rs = try lanes.Stream.init(gpa, .{ .id = "c_resumed", .prompt = &repeat_prompt, .max_new = 10, .drafts = true, .reuse = .{ .saved = snap, .at = 8 } });
    defer rs.deinit(gpa);
    try tree_eng.addStream(&rs);
    while (tree_eng.activeCount() > 0) try tree_eng.step();

    try std.testing.expectEqualSlices(u32, ps.emitted(), ts.emitted());
    try std.testing.expectEqualSlices(u32, ps.emitted(), rs.emitted());
    try std.testing.expectEqual(@as(u32, 8), rs.cached);
    try std.testing.expect(ts.drafted > 0 and ts.accepted > 0 and rs.drafted > 0);
    try std.testing.expect(drafter.saw_non_prefix_rows);
}

test "48-visit Inkling 2 Mini Metal verify tree matches sequential replay bit-for-bit and passes C2/C7" {
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    dev.deinit();
    const m = try synth.buildSyntheticModel(gpa);
    defer m.deinit();
    const target = try backend_.Metal.init(gpa, m, .{ .capacity = 256, .streams = 4, .budget = 8 * st.cacheBytes(256) });
    defer target.deinit();
    try target.warm();
    try checkTreeBranches(target, 12);
    try checkTreeBranches(target, 68);
    try checkMixedBatchAndTreeC2C7(target);
    try checkLanesConformance(target);
}
