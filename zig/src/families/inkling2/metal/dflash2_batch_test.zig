//! Metal parity tests for batched multi-stream DFlash 2 drafting (N>1 vs N=1 serial, BF16 and Q4).
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const lanes = @import("lanes");
const sources = @import("kernel_sources");
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;
const fwd = @import("forward.zig");
const dense_mma = @import("dense.zig");
const dflash2 = @import("dflash2.zig");
const w = @import("dflash2_weights.zig");
const prepared = @import("dflash2_prepared.zig");
const q4 = @import("draft_head_q4.zig");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn bf16(v: f32) u16 {
    return core.affine4.asBf16(v);
}

fn fillBf16Tile(rng: std.Random, slice: []u16, cols: usize, scale: f32) void {
    const rows = slice.len / cols;
    const tile_rows = @min(rows, 32);
    const tile_elems = tile_rows * cols;
    for (slice[0..tile_elems]) |*v| v.* = bf16((rng.float(f32) * 2.0 - 1.0) * scale);
    var r: usize = tile_rows;
    while (r < rows) : (r += tile_rows) {
        const take = @min(tile_rows, rows - r);
        @memcpy(slice[r * cols ..][0 .. take * cols], slice[0 .. take * cols]);
        for (0..take) |j| {
            const row = r + j;
            const c = (row * 7) % cols;
            const delta: u16 = @intCast((row % 11) + 1);
            slice[row * cols + c] = bf16((@as(f32, @floatFromInt(delta)) - 6.0) * 0.05 * scale);
        }
    }
}

fn makeTensor(buf: mtl.Buffer, off: usize, dims: []const usize) ck.Tensor {
    var shape: [core.safetensors.max_rank]usize = @splat(1);
    @memcpy(shape[0..dims.len], dims);
    var count: usize = 1;
    for (dims) |d| count *= d;
    return .{ .dtype = .bf16, .rank = dims.len, .shape = shape, .bytes = count * 2, .buffer = buf, .offset = off };
}

fn loadDensePipelines(device: mtl.Device) ![6]mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, sources.spark25_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    var out: [6]mtl.Pipeline = undefined;
    var made: usize = 0;
    errdefer for (out[0..made]) |p| p.deinit();
    for ([_]u32{ 1, 2, 4, 8, 16, 32 }, 0..) |rows, i| {
        var buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrintSentinel(&buf, "sp_matvec_{d}", .{rows}, 0);
        out[i] = try mtl.Pipeline.init(device, lib, name, false);
        made += 1;
    }
    return out;
}

const SyntheticRig = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    model: Model,
    vocab_buf: mtl.Buffer,
    cb_buf: mtl.Buffer,
    weight_bufs: std.ArrayList(mtl.Buffer) = .empty,

    fn init(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !SyntheticRig {
        var rig: SyntheticRig = .{
            .gpa = gpa,
            .device = device,
            .queue = queue,
            .model = undefined,
            .vocab_buf = undefined,
            .cb_buf = undefined,
        };
        rig.model.device = device;
        rig.model.queue = queue;
        rig.model.config.text.unpadded_vocab = w.valid_vocab;
        rig.model.dense = try loadDensePipelines(device);
        errdefer for (rig.model.dense) |p| p.deinit();
        rig.model.dense_mma = try dense_mma.Mma.load(device);
        errdefer if (rig.model.dense_mma) |m| m.deinit();
        var prng = std.Random.DefaultPrng.init(0xdf2_ba7c);
        const rng = prng.random();
        rig.vocab_buf = try device.buffer(w.padded_vocab * w.hidden * 2, opts);
        errdefer rig.vocab_buf.deinit();
        fillBf16Tile(rng, rig.vocab_buf.slice(u16, w.padded_vocab * w.hidden), w.hidden, 0.25);
        rig.cb_buf = try device.buffer(2 * w.padded_vocab * w.rank * 2, opts);
        errdefer rig.cb_buf.deinit();
        fillBf16Tile(rng, rig.cb_buf.slice(u16, 2 * w.padded_vocab * w.rank), w.rank, 0.2);
        const embed_norm_buf = try rig.allocFilled(w.hidden, rng, 1.0);
        rig.model.weights.embed = makeTensor(rig.vocab_buf, 0, &.{ w.padded_vocab, w.hidden });
        rig.model.weights.unembed = makeTensor(rig.vocab_buf, 0, &.{ w.padded_vocab, w.hidden });
        rig.model.weights.embed_norm = makeTensor(embed_norm_buf, 0, &.{w.hidden});
        return rig;
    }

    fn deinit(rig: *SyntheticRig) void {
        for (rig.weight_bufs.items) |b| b.deinit();
        rig.weight_bufs.deinit(rig.gpa);
        rig.cb_buf.deinit();
        rig.vocab_buf.deinit();
        if (rig.model.dense_mma) |m| m.deinit();
        for (rig.model.dense) |p| p.deinit();
    }

    fn allocFilled(rig: *SyntheticRig, count: usize, rng: std.Random, scale: f32) !mtl.Buffer {
        const buf = try rig.device.buffer(count * 2, opts);
        errdefer buf.deinit();
        try rig.weight_bufs.append(rig.gpa, buf);
        for (buf.slice(u16, count)) |*v| v.* = bf16((rng.float(f32) * 2.0 - 1.0) * scale);
        return buf;
    }

    fn makeLinear(rig: *SyntheticRig, n: usize, k: usize, rng: std.Random, copies: ?*prepared.Copies, order: ?core.lane_projection.RegOrder, key_seed: usize) !w.Linear {
        const buf = try rig.allocFilled(n * k, rng, 0.2);
        const t = makeTensor(buf, 0, &.{ n, k });
        const key = @intFromPtr(buf.contents()) +% key_seed;
        if (copies) |cp| {
            try cp.addBf16(rig.device, order, key, n, k, buf.slice(u16, n * k));
            return .{ .n = n, .k = k, .key = key, .bf16 = null };
        }
        return .{ .n = n, .k = k, .key = key, .bf16 = t };
    }

    fn makeDrafter(rig: *SyntheticRig, bits: u8, use_tu: bool, max_streams: usize) !*dflash2.Drafter {
        var prng = std.Random.DefaultPrng.init(0xdf2_9001);
        const rng = prng.random();
        var copies: ?prepared.Copies = if (bits == 4) .{ .gpa = rig.gpa } else null;
        errdefer if (copies) |*c| c.deinit();
        const order = if (bits == 4 and use_tu) try core.lane_projection.RegOrder.init(rig.device) else null;
        defer if (order) |o| o.deinit();
        if (bits == 4 and !use_tu) copies.?.simd = try core.row_projection.simd.Pipelines.load(rig.device);
        const cp: ?*prepared.Copies = if (copies) |*c| c else null;

        const fc = try rig.makeLinear(w.hidden, w.taps_n * w.hidden, rng, cp, order, 1);
        const hidden_proj = try rig.makeLinear(w.rank, w.hidden, rng, cp, order, 2);
        const q_proj = try rig.makeLinear(w.query_width, w.hidden, rng, cp, order, 3);
        const k_proj = try rig.makeLinear(w.kv_width, w.hidden, rng, cp, order, 4);
        const v_proj = try rig.makeLinear(w.kv_width, w.hidden, rng, cp, order, 5);
        const o_proj = try rig.makeLinear(w.hidden, w.query_width, rng, cp, order, 6);
        const gate_proj = try rig.makeLinear(w.intermediate, w.hidden, rng, cp, order, 7);
        const up_proj = try rig.makeLinear(w.intermediate, w.hidden, rng, cp, order, 8);
        const down_proj = try rig.makeLinear(w.hidden, w.intermediate, rng, cp, order, 9);
        const attn_cproj = try rig.makeLinear(w.dyn_width, w.hidden, rng, cp, order, 10);
        const mlp_cproj = try rig.makeLinear(w.dyn_width, w.hidden, rng, cp, order, 11);

        const hnorm_b = try rig.allocFilled(w.hidden, rng, 0.5);
        const fnorm_b = try rig.allocFilled(w.hidden, rng, 0.5);
        var layers: [w.layers_n]w.Layer = undefined;
        for (&layers) |*l| {
            l.* = .{
                .in_norm = makeTensor(try rig.allocFilled(w.hidden, rng, 0.5), 0, &.{w.hidden}),
                .post_norm = makeTensor(try rig.allocFilled(w.hidden, rng, 0.5), 0, &.{w.hidden}),
                .q_norm = makeTensor(try rig.allocFilled(w.head_dim, rng, 0.5), 0, &.{w.head_dim}),
                .k_norm = makeTensor(try rig.allocFilled(w.head_dim, rng, 0.5), 0, &.{w.head_dim}),
                .q_proj = q_proj,
                .k_proj = k_proj,
                .v_proj = v_proj,
                .o_proj = o_proj,
                .gate_proj = gate_proj,
                .up_proj = up_proj,
                .down_proj = down_proj,
                .attn_conv = .{ .base = makeTensor(try rig.allocFilled(4 * w.hidden, rng, 0.2), 0, &.{ 2, 2, w.hidden }), .proj = attn_cproj },
                .mlp_conv = .{ .base = makeTensor(try rig.allocFilled(4 * w.hidden, rng, 0.2), 0, &.{ 2, 2, w.hidden }), .proj = mlp_cproj },
            };
        }
        const loaded: w.Loaded = .{
            .checkpoint = ck.Checkpoint.init(rig.gpa),
            .config = .{},
            .weights = .{
                .fc = fc,
                .hidden_norm = makeTensor(hnorm_b, 0, &.{w.hidden}),
                .norm = makeTensor(fnorm_b, 0, &.{w.hidden}),
                .hidden_proj = hidden_proj,
                .pred_cb = makeTensor(rig.cb_buf, 0, &.{ w.padded_vocab, w.rank }),
                .succ_cb = makeTensor(rig.cb_buf, w.padded_vocab * w.rank * 2, &.{ w.padded_vocab, w.rank }),
                .layers = layers,
            },
            .bits = bits,
            .copies = copies,
        };
        return dflash2.Drafter.testing.init(rig.gpa, &rig.model, loaded, max_streams);
    }
};

fn readRingBytes(device: mtl.Device, queue: mtl.Queue, df: *const dflash2.Drafter, s: *lanes.Stream, layer: usize, is_v: bool, dst: []u8) !void {
    const src = dflash2.Drafter.testing.ring(df, s, layer, is_v) orelse return error.MissingRing;
    const staging = try device.buffer(dst.len, opts);
    defer staging.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const b = cb.blit();
    b.copy(src, 0, staging, 0, dst.len);
    b.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    @memcpy(dst, staging.contents()[0..dst.len]);
}

fn fillTapRows(buf: mtl.Buffer, start_row: usize, rows: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const row_elems = w.taps_n * w.hidden;
    const dst = buf.slice(u16, (start_row + rows) * row_elems)[start_row * row_elems ..];
    for (dst) |*v| v.* = bf16((rng.float(f32) * 2.0 - 1.0) * 0.25);
}

fn verifyProjectionExactness(device: mtl.Device, queue: mtl.Queue, dense: *const Model, n: usize, k: usize, bits: u8, use_tu: bool) !void {
    const gpa = std.testing.allocator;
    const max_rows: usize = 64;
    var prng = std.Random.DefaultPrng.init(0xdf2_1111 +% n +% k +% bits);
    const rng = prng.random();
    const x_buf = try device.buffer(max_rows * k * 2, opts);
    defer x_buf.deinit();
    for (x_buf.slice(u16, max_rows * k)) |*v| v.* = bf16((rng.float(f32) * 2.0 - 1.0) * 0.3);
    const w_buf = try device.buffer(n * k * 2, opts);
    defer w_buf.deinit();
    for (w_buf.slice(u16, n * k)) |*v| v.* = bf16((rng.float(f32) * 2.0 - 1.0) * 0.2);

    var copies: ?prepared.Copies = if (bits == 4) .{ .gpa = gpa } else null;
    defer if (copies) |*c| c.deinit();
    if (bits == 4) {
        const order = if (use_tu) try core.lane_projection.RegOrder.init(device) else null;
        defer if (order) |o| o.deinit();
        if (!use_tu) copies.?.simd = try core.row_projection.simd.Pipelines.load(device);
        try copies.?.addBf16(device, order, 1, n, k, w_buf.slice(u16, n * k));
    }

    const solo_buf = try device.buffer(max_rows * n * 2, opts);
    defer solo_buf.deinit();
    const batch_buf = try device.buffer(max_rows * n * 2, opts);
    defer batch_buf.deinit();
    const w_tensor = makeTensor(w_buf, 0, &.{ n, k });

    // Run 8-row slices one at a time, plus a 7-row tail slice for rows 1..7.
    {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        var s: usize = 0;
        while (s < max_rows) : (s += 8) {
            const x_ref = dflash2.Drafter.testing;
            _ = x_ref;
            if (bits == 4) {
                try copies.?.project(e, 1, n, k, .{ .buf = x_buf, .off = s * k * 2 }, .{ .buf = solo_buf, .off = s * n * 2 }, 8);
            } else {
                fwd.dense(dense, e, w_tensor, .{ .buf = x_buf, .off = s * k * 2 }, .{ .buf = solo_buf, .off = s * n * 2 }, 8);
            }
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
    }

    const want = solo_buf.slice(u16, max_rows * n);
    for ([_]usize{ 7, 16, 24, 32, 64 }) |rows| {
        const x_off = if (rows == 7) k * 2 else 0;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        if (bits == 4) {
            try copies.?.project(e, 1, n, k, .{ .buf = x_buf, .off = x_off }, .{ .buf = batch_buf, .off = 0 }, rows);
        } else {
            fwd.dense(dense, e, w_tensor, .{ .buf = x_buf, .off = x_off }, .{ .buf = batch_buf, .off = 0 }, rows);
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        const got = batch_buf.slice(u16, rows * n);
        if (rows == 7) {
            try std.testing.expectEqualSlices(u16, want[n .. 8 * n], got);
        } else {
            try std.testing.expectEqualSlices(u16, want[0 .. rows * n], got);
        }
    }
}

test "DFlash2 batched linear projections match single-stream rows bit-for-bit across BF16 and Q4" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var dense_model: Model = undefined;
    dense_model.device = device;
    dense_model.queue = queue;
    dense_model.dense = try loadDensePipelines(device);
    defer for (dense_model.dense) |p| p.deinit();

    try verifyProjectionExactness(device, queue, &dense_model, 256, 2304, 0, false);
    try verifyProjectionExactness(device, queue, &dense_model, 576, 2304, 0, false);
    try verifyProjectionExactness(device, queue, &dense_model, 2304, 2304, 0, false);
    try verifyProjectionExactness(device, queue, &dense_model, 256, 2304, 4, false);
    try verifyProjectionExactness(device, queue, &dense_model, 576, 2304, 4, false);
    if (device.tensorUnits()) {
        try verifyProjectionExactness(device, queue, &dense_model, 256, 2304, 4, true);
        try verifyProjectionExactness(device, queue, &dense_model, 576, 2304, 4, true);
        try verifyProjectionExactness(device, queue, &dense_model, 4096, 2304, 4, true);
    }
}

fn runParityMode(rig: *SyntheticRig, bits: u8, use_tu: bool, count: usize, head_q4: bool) !void {
    const gpa = rig.gpa;
    const solo = try rig.makeDrafter(bits, use_tu, 1);
    defer solo.deinit();
    const batch = try rig.makeDrafter(bits, use_tu, dflash2.max_batch_streams);
    defer batch.deinit();
    if (head_q4) {
        solo.loaded.draft_head_q4 = try q4.init(rig.device, rig.model.weights.unembed.host(u16), w.valid_vocab, w.hidden);
        batch.loaded.draft_head_q4 = try q4.init(rig.device, rig.model.weights.unembed.host(u16), w.valid_vocab, w.hidden);
    }

    const solo_dr = solo.drafter();
    const batch_dr = batch.drafter();
    try std.testing.expect(batch_dr.facts().batched);
    try std.testing.expect(batch_dr.facts().fixed_block);

    const row_bytes = w.taps_n * w.hidden * 2;
    const ring_bytes = w.ring_size * w.kv_width * 2;
    const tap_buf = try rig.device.buffer(3200 * row_bytes, opts);
    defer tap_buf.deinit();
    fillTapRows(tap_buf, 0, 3200, 0xdf2_aa01);

    const Case = struct { ctx_rows: usize, prompt_len: usize, pending: u32, gap_rows: usize, depth: usize };
    const base_cases = [_]Case{
        .{ .ctx_rows = 0, .prompt_len = 1, .pending = 17, .gap_rows = 2, .depth = 7 },
        .{ .ctx_rows = 17, .prompt_len = 18, .pending = 403, .gap_rows = 5, .depth = 5 },
        .{ .ctx_rows = 300, .prompt_len = 301, .pending = 1234, .gap_rows = 1, .depth = 3 },
        .{ .ctx_rows = 2500, .prompt_len = 2501, .pending = 9876, .gap_rows = 4, .depth = 7 },
    };
    const streams = try gpa.alloc(*lanes.Stream, count);
    defer gpa.free(streams);
    const solo_streams = try gpa.alloc(*lanes.Stream, count);
    defer gpa.free(solo_streams);
    var dummy_prompt: [16]u32 = @splat(11);
    for (0..count) |i| {
        const c = base_cases[i % base_cases.len];
        const plen = c.prompt_len;
        const s_solo = try gpa.create(lanes.Stream);
        s_solo.* = try lanes.Stream.init(gpa, .{ .id = "solo", .prompt = &dummy_prompt, .max_new = 16 });
        s_solo.prompt_len = plen;
        solo_streams[i] = s_solo;
        try solo_dr.open(s_solo);

        const s_batch = try gpa.create(lanes.Stream);
        s_batch.* = try lanes.Stream.init(gpa, .{ .id = "batch", .prompt = &dummy_prompt, .max_new = 16 });
        s_batch.prompt_len = plen;
        streams[i] = s_batch;
        try batch_dr.open(s_batch);
    }
    defer for (0..count) |i| {
        solo_dr.release(solo_streams[i]);
        solo_streams[i].deinit(gpa);
        gpa.destroy(solo_streams[i]);
        batch_dr.release(streams[i]);
        streams[i].deinit(gpa);
        gpa.destroy(streams[i]);
    };

    var wants = try gpa.alloc([7]u32, count);
    defer gpa.free(wants);
    var gots = try gpa.alloc([7]u32, count);
    defer gpa.free(gots);
    const dummy_follow: [3000]u32 = @splat(1);

    // 1. Run each stream solo (N=1) and record held drafts and rings.
    var want_k = try gpa.alloc(u8, count * w.layers_n * ring_bytes);
    defer gpa.free(want_k);
    var want_v = try gpa.alloc(u8, count * w.layers_n * ring_bytes);
    defer gpa.free(want_v);
    for (0..count) |i| {
        const c = base_cases[i % base_cases.len];
        if (c.ctx_rows > 0) {
            try solo_dr.absorb(&.{.{
                .stream = solo_streams[i],
                .features = .{ .buffer = @intFromPtr(tap_buf.id), .offset = (i * 11) * row_bytes, .rows = @intCast(c.ctx_rows), .row_bytes = row_bytes, .space = .device, .dtype = .bf16 },
                .start = 0,
                .follow = dummy_follow[0..c.ctx_rows],
                .prompt = true,
            }});
        }
        try solo_dr.hold(&.{.{ .stream = solo_streams[i], .pending = c.pending +% @as(u32, @intCast(i)), .position = c.ctx_rows + 1, .depth = @intCast(c.depth) }});
        try solo_dr.held(&.{solo_streams[i]}, &.{wants[i][0..c.depth]});
        for (0..w.layers_n) |layer| {
            const off = (i * w.layers_n + layer) * ring_bytes;
            try readRingBytes(rig.device, rig.queue, solo, solo_streams[i], layer, false, want_k[off..][0..ring_bytes]);
            try readRingBytes(rig.device, rig.queue, solo, solo_streams[i], layer, true, want_v[off..][0..ring_bytes]);
        }
    }

    // 2. Run all streams batched (N>1) in a single absorb + single hold.
    var absorb_items = std.ArrayList(lanes.drafter.Absorb).empty;
    defer absorb_items.deinit(gpa);
    for (0..count) |i| {
        const c = base_cases[i % base_cases.len];
        if (c.ctx_rows > 0) {
            try absorb_items.append(gpa, .{
                .stream = streams[i],
                .features = .{ .buffer = @intFromPtr(tap_buf.id), .offset = (i * 11) * row_bytes, .rows = @intCast(c.ctx_rows), .row_bytes = row_bytes, .space = .device, .dtype = .bf16 },
                .start = 0,
                .follow = dummy_follow[0..c.ctx_rows],
                .prompt = true,
            });
        }
    }
    if (absorb_items.items.len > 0) try batch_dr.absorb(absorb_items.items);

    const hold_items = try gpa.alloc(lanes.drafter.Hold, count);
    defer gpa.free(hold_items);
    const held_outs = try gpa.alloc([]u32, count);
    defer gpa.free(held_outs);
    for (0..count) |i| {
        const c = base_cases[i % base_cases.len];
        hold_items[i] = .{ .stream = streams[i], .pending = c.pending +% @as(u32, @intCast(i)), .position = c.ctx_rows + 1, .depth = @intCast(c.depth) };
        held_outs[i] = gots[i][0..c.depth];
    }
    try batch_dr.hold(hold_items);
    try batch_dr.held(streams, held_outs);

    const got_ring = try gpa.alloc(u8, ring_bytes);
    defer gpa.free(got_ring);
    for (0..count) |i| {
        const c = base_cases[i % base_cases.len];
        try std.testing.expectEqualSlices(u32, wants[i][0..c.depth], gots[i][0..c.depth]);
        for (0..w.layers_n) |layer| {
            const off = (i * w.layers_n + layer) * ring_bytes;
            try readRingBytes(rig.device, rig.queue, batch, streams[i], layer, false, got_ring);
            try std.testing.expectEqualSlices(u8, want_k[off..][0..ring_bytes], got_ring);
            try readRingBytes(rig.device, rig.queue, batch, streams[i], layer, true, got_ring);
            try std.testing.expectEqualSlices(u8, want_v[off..][0..ring_bytes], got_ring);
        }
    }

    // 3. Follow-up incremental live decode absorb (with non-contiguous offsets/gaps) + hold!
    absorb_items.clearRetainingCapacity();
    var tap_cursor: usize = 20;
    for (0..count) |i| {
        const c = base_cases[i % base_cases.len];
        const rows = (i % 3) + 1;
        const src_off = tap_cursor * row_bytes;
        tap_cursor += rows + c.gap_rows;
        try solo_dr.absorb(&.{.{
            .stream = solo_streams[i],
            .features = .{ .buffer = @intFromPtr(tap_buf.id), .offset = src_off, .rows = @intCast(rows), .row_bytes = row_bytes, .space = .device, .dtype = .bf16 },
            .start = c.ctx_rows,
            .follow = dummy_follow[0..rows],
            .prompt = false,
        }});
        try solo_dr.hold(&.{.{ .stream = solo_streams[i], .pending = c.pending +% 55, .position = c.ctx_rows + rows + 1, .depth = 7 }});
        try solo_dr.held(&.{solo_streams[i]}, &.{&wants[i]});

        try absorb_items.append(gpa, .{
            .stream = streams[i],
            .features = .{ .buffer = @intFromPtr(tap_buf.id), .offset = src_off, .rows = @intCast(rows), .row_bytes = row_bytes, .space = .device, .dtype = .bf16 },
            .start = c.ctx_rows,
            .follow = dummy_follow[0..rows],
            .prompt = false,
        });
        hold_items[i] = .{ .stream = streams[i], .pending = c.pending +% 55, .position = c.ctx_rows + rows + 1, .depth = 7 };
        held_outs[i] = &gots[i];
    }
    try batch_dr.absorb(absorb_items.items);
    try batch_dr.hold(hold_items);
    try batch_dr.held(streams, held_outs);
    for (0..count) |i| {
        try std.testing.expectEqualSlices(u32, &wants[i], &gots[i]);
        for (0..w.layers_n) |layer| {
            try readRingBytes(rig.device, rig.queue, solo, solo_streams[i], layer, false, got_ring);
            const batch_k = try gpa.alloc(u8, ring_bytes);
            defer gpa.free(batch_k);
            try readRingBytes(rig.device, rig.queue, batch, streams[i], layer, false, batch_k);
            try std.testing.expectEqualSlices(u8, got_ring, batch_k);
            try readRingBytes(rig.device, rig.queue, solo, solo_streams[i], layer, true, got_ring);
            try readRingBytes(rig.device, rig.queue, batch, streams[i], layer, true, batch_k);
            try std.testing.expectEqualSlices(u8, got_ring, batch_k);
        }
    }
}

test "DFlash2 batched N=4 and N=9 match N=1 serial bit-for-bit across BF16 and Q4" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var rig = try SyntheticRig.init(std.testing.allocator, device, queue);
    defer rig.deinit();

    try runParityMode(&rig, 0, false, 4, false);
    try runParityMode(&rig, 0, false, 4, true);
    try runParityMode(&rig, 4, false, 4, false);
    if (device.tensorUnits()) {
        try runParityMode(&rig, 4, true, 4, false);
        try runParityMode(&rig, 4, true, 4, true);
        try runParityMode(&rig, 4, true, 9, false);
        try runParityMode(&rig, 4, true, 9, true);
    } else {
        try runParityMode(&rig, 0, false, 9, false);
    }
}

test "DFlash2 encodeBlocks microbenchmark across bf16, q4 and q4+head_q4" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var rig = try SyntheticRig.init(std.testing.allocator, device, queue);
    defer rig.deinit();

    var b1: f64 = 0;
    var b4: f64 = 0;
    {
        const df_bf16 = try rig.makeDrafter(0, device.tensorUnits(), 4);
        defer df_bf16.deinit();
        b1 = try dflash2.Drafter.testing.benchmarkEncode(df_bf16, 1, 4);
        b4 = try dflash2.Drafter.testing.benchmarkEncode(df_bf16, 4, 4);
    }

    const df_q4 = try rig.makeDrafter(4, device.tensorUnits(), 4);
    defer df_q4.deinit();
    const q1 = try dflash2.Drafter.testing.benchmarkEncode(df_q4, 1, 4);
    const q4_nohead = try dflash2.Drafter.testing.benchmarkEncode(df_q4, 4, 4);

    df_q4.loaded.draft_head_q4 = try q4.init(rig.device, rig.model.weights.unembed.host(u16), w.valid_vocab, w.hidden);
    const qh1 = try dflash2.Drafter.testing.benchmarkEncode(df_q4, 1, 4);
    const qh4 = try dflash2.Drafter.testing.benchmarkEncode(df_q4, 4, 4);

    std.debug.print("\nDFlash2 encodeBlocks ({s}): bf16 c=1:{d:.3}ms c=4:{d:.3}ms | q4 c=1:{d:.3}ms c=4:{d:.3}ms | q4+head_q4 c=1:{d:.3}ms c=4:{d:.3}ms\n", .{
        device.name(), b1, b4, q1, q4_nohead, qh1, qh4,
    });
}
