//! Metal tests for DFlash 2 prompt-cache snapshots: multi-stream slot >= 1, wrap-around, warm == cold, zero leaks.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const api = @import("engine_api");
const sources = @import("kernel_sources");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const model_file = @import("model.zig");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");
const backend = @import("backend.zig");
const w = @import("dflash2_weights.zig");
const k = @import("dflash2_kernels.zig");
const state = @import("dflash2_state.zig");
const dflash2 = @import("dflash2.zig");

const gpa = std.testing.allocator;

const SyntheticEnv = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    model: model_file.Model,
    drafter: dflash2.Drafter,
    vocab_buf: mtl.Buffer,
    cb_buf: mtl.Buffer,
    layer_buf: mtl.Buffer,
    taps_buf: mtl.Buffer,

    fn makeTensor(buf: mtl.Buffer, off: usize, dims: []const usize) ck.Tensor {
        var count: usize = 1;
        var shape: @TypeOf(@as(ck.Tensor, undefined).shape) = @splat(1);
        for (dims, 0..) |d, i| {
            shape[i] = d;
            count *= d;
        }
        return .{ .buffer = buf, .offset = off, .bytes = count * 2, .dtype = .bf16, .shape = shape, .rank = dims.len };
    }

    fn bf16FromF32(v: f32) u16 {
        const bits: u32 = @bitCast(v);
        return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
    }

    fn init(tap_rows: usize) !SyntheticEnv {
        const device = mtl.Device.init() catch return error.SkipZigTest;
        errdefer device.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();

        const lib = try mtl.Library.fromSource(device, sources.spark25_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        var dense: [6]mtl.Pipeline = undefined;
        var made_dense: usize = 0;
        errdefer for (dense[0..made_dense]) |p| p.deinit();
        for ([_]u32{ 1, 2, 4, 8, 16, 32 }, 0..) |rows, i| {
            var buf: [32]u8 = undefined;
            const name = try std.fmt.bufPrintSentinel(&buf, "sp_matvec_{d}", .{rows}, 0);
            dense[i] = try mtl.Pipeline.init(device, lib, name, false);
            made_dense += 1;
        }

        const seed_bytes: usize = 4 << 20;
        const seed_buf = try device.buffer(seed_bytes, mtl.ResourceOptions.shared);
        defer seed_buf.deinit();
        var prng = std.Random.DefaultPrng.init(20261124);
        const rng = prng.random();
        for (seed_buf.slice(u16, seed_bytes / 2)) |*dst| {
            dst.* = bf16FromF32((rng.float(f32) - 0.5) * 0.08);
        }

        const vocab_bytes = w.padded_vocab * w.hidden * 2;
        const cb_bytes = w.padded_vocab * w.rank * 2;
        const layer_bytes = 128 << 20;
        const vocab_buf = try device.buffer(vocab_bytes, mtl.ResourceOptions.private);
        errdefer vocab_buf.deinit();
        const cb_buf = try device.buffer(cb_bytes, mtl.ResourceOptions.private);
        errdefer cb_buf.deinit();
        const layer_buf = try device.buffer(layer_bytes, mtl.ResourceOptions.private);
        errdefer layer_buf.deinit();

        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = queue.commandBuffer();
            const e = cb.blit();
            for ([_]struct { dst: mtl.Buffer, total: usize }{
                .{ .dst = vocab_buf, .total = vocab_bytes },
                .{ .dst = cb_buf, .total = cb_bytes },
                .{ .dst = layer_buf, .total = layer_bytes },
            }) |target| {
                var off: usize = 0;
                var phase: usize = 0;
                while (off < target.total) {
                    const src_off = (phase * 65536) % (seed_bytes / 2);
                    const n = @min(seed_bytes - src_off, target.total - off);
                    e.copy(seed_buf, src_off, target.dst, off, n);
                    off += n;
                    phase += 1;
                }
            }
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailed;
        }

        const taps_bytes = tap_rows * w.taps_n * w.hidden * 2;
        const taps_buf = try device.buffer(taps_bytes, mtl.ResourceOptions.shared);
        errdefer taps_buf.deinit();
        for (taps_buf.slice(u16, taps_bytes / 2)) |*dst| {
            dst.* = bf16FromF32((rng.float(f32) - 0.5) * 0.25);
        }

        var env: SyntheticEnv = .{
            .device = device,
            .queue = queue,
            .model = undefined,
            .drafter = undefined,
            .vocab_buf = vocab_buf,
            .cb_buf = cb_buf,
            .layer_buf = layer_buf,
            .taps_buf = taps_buf,
        };
        env.model = .{
            .gpa = gpa,
            .device = device,
            .queue = queue,
            .config = .{
                .text = .{
                    .hidden = 2304,
                    .layers = 24,
                    .vocab = 201024,
                    .unpadded_vocab = 200058,
                    .local = .{ .q = 24, .kv = 8, .dim = 128 },
                    .global = .{ .q = 24, .kv = 8, .dim = 128 },
                    .window = 1024,
                    .d_rel = 16,
                    .rel_extent = 1024,
                    .log_floor = 8192,
                    .log_alpha = 0.1,
                    .eps = 1e-6,
                    .qk_clip = 22.627416997969522,
                    .rel_clip = 8,
                    .block = 6,
                    .experts = 128,
                    .topk = 8,
                    .expert_inter = 1152,
                    .route_scale = 8,
                    .gate_cap = 4,
                    .up_cap = 25,
                    .logit_cap = 30,
                    .max_context = 1048576,
                    .loop = .{ .core_start = 6, .core_end = 18, .passes = 3, .gate_rank = 64 },
                },
                .vision = undefined,
                .mtp = null,
                .eos = 200055,
                .image_token = 200054,
            },
            .checkpoint = ck.Checkpoint.init(gpa),
            .weights = undefined,
            .weight_plan = undefined,
            .attention = undefined,
            .glue = undefined,
            .experts = undefined,
            .loop = undefined,
            .tree = undefined,
            .dense = dense,
            .norm = dense[0],
            .dense_mma = null,
            .dense_tiled = null,
            .runtime = undefined,
            .sample = undefined,
        };
        env.model.weights.embed = makeTensor(vocab_buf, 0, &.{ w.padded_vocab, w.hidden });
        env.model.weights.embed_norm = makeTensor(layer_buf, 0, &.{w.hidden});
        env.model.weights.unembed = makeTensor(vocab_buf, 0, &.{ w.padded_vocab, w.hidden });

        var precision: [24]layer_plan.Precision = @splat(.nvfp4);
        for (precision[4..]) |*p| p.* = .mxfp8;
        env.model.weight_plan = try layer_plan.build(&env.model.config.text, &precision);

        var off: usize = 40 << 20;
        const allocTensor = struct {
            fn next(buf: mtl.Buffer, cursor: *usize, dims: []const usize) ck.Tensor {
                const t = makeTensor(buf, cursor.*, dims);
                cursor.* = std.mem.alignForward(usize, cursor.* + t.bytes, 256);
                return t;
            }
        }.next;
        var layers: [w.layers_n]w.Layer = undefined;
        for (&layers, 0..) |*l, i| {
            const base_off = i * 4096;
            l.* = .{
                .in_norm = makeTensor(layer_buf, base_off, &.{w.hidden}),
                .post_norm = makeTensor(layer_buf, base_off + 256, &.{w.hidden}),
                .q_norm = makeTensor(layer_buf, base_off + 512, &.{w.head_dim}),
                .k_norm = makeTensor(layer_buf, base_off + 768, &.{w.head_dim}),
                .q_proj = makeTensor(layer_buf, base_off + 1024, &.{ w.query_width, w.hidden }),
                .k_proj = makeTensor(layer_buf, base_off + 2048, &.{ w.kv_width, w.hidden }),
                .v_proj = makeTensor(layer_buf, base_off + 4096, &.{ w.kv_width, w.hidden }),
                .o_proj = makeTensor(layer_buf, base_off + 8192, &.{ w.hidden, w.query_width }),
                .gate_proj = makeTensor(layer_buf, base_off + 16384, &.{ w.intermediate, w.hidden }),
                .up_proj = makeTensor(layer_buf, base_off + 32768, &.{ w.intermediate, w.hidden }),
                .down_proj = makeTensor(layer_buf, base_off + 65536, &.{ w.hidden, w.intermediate }),
                .attn_conv = .{
                    .base = makeTensor(layer_buf, base_off + 131072, &.{ 2, 2, w.hidden }),
                    .proj = makeTensor(layer_buf, base_off + 196608, &.{ w.dyn_width, w.hidden }),
                },
                .mlp_conv = .{
                    .base = makeTensor(layer_buf, base_off + 262144, &.{ 2, 2, w.hidden }),
                    .proj = makeTensor(layer_buf, base_off + 327680, &.{ w.dyn_width, w.hidden }),
                },
            };
        }
        const weights: w.Weights = .{
            .fc = allocTensor(layer_buf, &off, &.{ w.hidden, w.taps_n * w.hidden }),
            .hidden_norm = allocTensor(layer_buf, &off, &.{w.hidden}),
            .norm = allocTensor(layer_buf, &off, &.{w.hidden}),
            .hidden_proj = allocTensor(layer_buf, &off, &.{ w.rank, w.hidden }),
            .pred_cb = makeTensor(cb_buf, 0, &.{ w.padded_vocab, w.rank }),
            .succ_cb = makeTensor(cb_buf, 0, &.{ w.padded_vocab, w.rank }),
            .layers = layers,
        };

        const pipes = try k.Pipelines.load(device);
        errdefer {
            var owned = pipes;
            owned.deinit();
        }
        const scratch = try state.Scratch.init(device);
        env.drafter = .{
            .gpa = gpa,
            .target = &env.model,
            .loaded = .{ .checkpoint = ck.Checkpoint.init(gpa), .config = .{}, .weights = weights },
            .pipelines = pipes,
            .scratch = scratch,
        };
        return env;
    }

    fn deinit(env: *SyntheticEnv) void {
        env.drafter.settle() catch {};
        var it = env.drafter.contexts.valueIterator();
        while (it.next()) |lane| lane.deinit();
        env.drafter.contexts.deinit(gpa);
        env.drafter.scratch.deinit();
        env.drafter.pipelines.deinit();
        env.drafter.loaded.deinit();
        env.model.checkpoint.deinit();
        for (env.model.dense) |p| p.deinit();
        env.taps_buf.deinit();
        env.layer_buf.deinit();
        env.cb_buf.deinit();
        env.vocab_buf.deinit();
        env.queue.deinit();
        env.device.deinit();
    }
};

fn readRingActive(queue: mtl.Queue, device: mtl.Device, lane: *const state.Lane, lo: u64, end: u64) ![]u8 {
    const rows: usize = @intCast(end - lo);
    const total = rows * state.layer_row_bytes;
    const flat = try device.buffer(total, mtl.ResourceOptions.shared);
    defer flat.deinit();
    const start_slot: usize = @intCast(lo % w.ring_size);
    const first_rows = @min(rows, w.ring_size - start_slot);
    const second_rows = rows - first_rows;
    const first_bytes = first_rows * state.row_bytes;
    const second_bytes = second_rows * state.row_bytes;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var at: usize = 0;
    for (0..w.layers_n) |i| {
        for ([_]mtl.Buffer{ lane.k[i], lane.v[i] }) |rb| {
            e.copy(rb, start_slot * state.row_bytes, flat, at, first_bytes);
            if (second_bytes > 0) e.copy(rb, 0, flat, at + first_bytes, second_bytes);
            at += first_bytes + second_bytes;
        }
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    return try gpa.dupe(u8, flat.slice(u8, total));
}

fn absorbSlice(dr: lanes.drafter.Drafter, s: *lanes.Stream, taps: mtl.Buffer, src_base: usize, start: u64, rows: usize, prompt_mode: bool) !void {
    const follow = try gpa.alloc(u32, rows);
    defer gpa.free(follow);
    for (follow, 0..) |*f, i| f.* = @intCast((start + i + 1) % w.valid_vocab);
    var pos: usize = 0;
    while (pos < rows) {
        const n = @min(st.prompt_rows, rows - pos);
        const f: lanes.backend.Features = .{
            .buffer = @intFromPtr(taps.id),
            .offset = (src_base + @as(usize, @intCast(start)) + pos) * state.tap_row_bytes,
            .rows = @intCast(n),
            .row_bytes = state.tap_row_bytes,
            .space = .device,
            .dtype = .bf16,
        };
        try dr.absorb(&.{.{ .stream = s, .features = f, .start = start + pos, .follow = follow[pos..][0..n], .prompt = prompt_mode }});
        pos += n;
    }
}

test "DFlash2 multi-stream save/restore at slot >= 1 preserves slot 0 and matches cold bit-identically" {
    var env = try SyntheticEnv.init(2400);
    defer env.deinit();
    const dr = env.drafter.drafter();

    const len1: usize = 2310;
    const snap1_at: u32 = 2296;
    const len2: usize = 2260;
    const snap2_at: u32 = 2240;
    const ids = try gpa.alloc(u32, len1);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(10 + (i * 13) % 500);
    const marks1 = [_]u32{snap1_at};
    const marks2 = [_]u32{snap2_at};

    var slot0 = try lanes.Stream.init(gpa, .{ .id = "slot0", .prompt = ids[0..120], .max_new = 16 });
    defer slot0.deinit(gpa);
    var slot1 = try lanes.Stream.init(gpa, .{ .id = "slot1", .prompt = ids, .max_new = 16, .reuse = .{ .marks = &marks1 } });
    defer slot1.deinit(gpa);
    var slot2 = try lanes.Stream.init(gpa, .{ .id = "slot2", .prompt = ids[0..len2], .max_new = 16, .reuse = .{ .marks = &marks2 } });
    defer slot2.deinit(gpa);
    var cold1 = try lanes.Stream.init(gpa, .{ .id = "cold1", .prompt = ids, .max_new = 16, .reuse = .{ .marks = &marks1 } });
    defer cold1.deinit(gpa);
    var cold2 = try lanes.Stream.init(gpa, .{ .id = "cold2", .prompt = ids[0..len2], .max_new = 16, .reuse = .{ .marks = &marks2 } });
    defer cold2.deinit(gpa);

    try dr.open(&slot0);
    defer dr.release(&slot0);
    try dr.open(&slot1);
    var slot1_open = true;
    defer if (slot1_open) dr.release(&slot1);
    try dr.open(&slot2);
    var slot2_open = true;
    defer if (slot2_open) dr.release(&slot2);
    try dr.open(&cold1);
    defer dr.release(&cold1);
    try dr.open(&cold2);
    defer dr.release(&cold2);

    // Populate slot 0 and hold drafts so we can prove peer saves/restores never clobber slot 0.
    try absorbSlice(dr, &slot0, env.taps_buf, 0, 0, 120, true);
    try dr.hold(&.{.{ .stream = &slot0, .pending = 11, .position = 121, .depth = 7 }});
    var slot0_held_before: [7]u32 = undefined;
    try dr.held(&.{&slot0}, &.{&slot0_held_before});
    const slot0_ring_before = try readRingActive(env.queue, env.device, env.drafter.contexts.getPtr(&slot0).?, 0, 120);
    defer gpa.free(slot0_ring_before);

    // Slot 1 saves across ring_size = 2176; slot 2 saves with an unabsorbed tail tap at a non-zero shared offset.
    try absorbSlice(dr, &slot1, env.taps_buf, 0, 0, snap1_at, true);
    try absorbSlice(dr, &cold1, env.taps_buf, 0, 0, snap1_at, true);
    try absorbSlice(dr, &slot2, env.taps_buf, 32, 0, snap2_at - 1, true);
    try absorbSlice(dr, &cold2, env.taps_buf, 32, 0, snap2_at, true);

    var snap1 = (try env.drafter.saveSnap(&slot1, snap1_at, .{}, 0)).?;
    defer snap1.free();
    try std.testing.expectEqual(@as(u64, snap1_at - w.window_size), snap1.from);
    try std.testing.expectEqual(@as(u64, snap1_at), snap1.end);

    const shared_tap: st.Tap = .{
        .buf = env.taps_buf,
        .off = (32 + snap2_at - 2) * state.tap_row_bytes,
        .start = snap2_at - 2,
        .rows = 2,
        .gen = 5,
    };
    var snap2 = (try env.drafter.saveSnap(&slot2, snap2_at, shared_tap, 5)).?;
    defer snap2.free();
    try std.testing.expectEqual(@as(u64, snap2_at - 1 - w.window_size), snap2.from);
    try std.testing.expectEqual(@as(u64, snap2_at - 1), snap2.end);
    try std.testing.expect(snap2.tail_off != null);

    // Release slots 1 and 2 and open fresh warm streams at slot >= 1 while slot 0 remains live.
    dr.release(&slot1);
    slot1_open = false;
    dr.release(&slot2);
    slot2_open = false;

    var warm1 = try lanes.Stream.init(gpa, .{ .id = "warm1", .prompt = ids, .max_new = 16 });
    defer warm1.deinit(gpa);
    var warm2 = try lanes.Stream.init(gpa, .{ .id = "warm2", .prompt = ids[0..len2], .max_new = 16 });
    defer warm2.deinit(gpa);
    try dr.open(&warm1);
    defer dr.release(&warm1);
    try dr.open(&warm2);
    defer dr.release(&warm2);

    try env.drafter.restoreSnap(&warm1, &snap1, snap1_at);
    try env.drafter.restoreSnap(&warm2, &snap2, snap2_at);

    // Verify slot 0 ring bytes, metadata, and held drafts are completely untouched.
    const slot0_lane = env.drafter.contexts.getPtr(&slot0).?;
    try std.testing.expectEqual(@as(u64, 0), slot0_lane.from);
    try std.testing.expectEqual(@as(u64, 120), slot0_lane.end);
    try std.testing.expectEqual(@as(u32, 7), slot0_lane.held_len);
    var slot0_held_after: [7]u32 = undefined;
    try dr.held(&.{&slot0}, &.{&slot0_held_after});
    try std.testing.expectEqualSlices(u32, &slot0_held_before, &slot0_held_after);
    const slot0_ring_after = try readRingActive(env.queue, env.device, slot0_lane, 0, 120);
    defer gpa.free(slot0_ring_after);
    try std.testing.expectEqualSlices(u8, slot0_ring_before, slot0_ring_after);

    // Continue both restored streams alongside their cold references through prompt tail, hold, and decode.
    try absorbSlice(dr, &cold1, env.taps_buf, 0, snap1_at, len1 - snap1_at, true);
    try absorbSlice(dr, &warm1, env.taps_buf, 0, snap1_at, len1 - snap1_at, true);
    try absorbSlice(dr, &cold2, env.taps_buf, 32, snap2_at, len2 - snap2_at, true);
    try absorbSlice(dr, &warm2, env.taps_buf, 32, snap2_at, len2 - snap2_at, true);
    try dr.hold(&.{
        .{ .stream = &cold1, .pending = 42, .position = len1 + 1, .depth = 7 },
        .{ .stream = &warm1, .pending = 42, .position = len1 + 1, .depth = 7 },
        .{ .stream = &cold2, .pending = 63, .position = len2 + 1, .depth = 7 },
        .{ .stream = &warm2, .pending = 63, .position = len2 + 1, .depth = 7 },
    });

    var h_c1: [7]u32 = undefined;
    var h_w1: [7]u32 = undefined;
    var h_c2: [7]u32 = undefined;
    var h_w2: [7]u32 = undefined;
    try dr.held(&.{ &cold1, &warm1, &cold2, &warm2 }, &.{ &h_c1, &h_w1, &h_c2, &h_w2 });
    try std.testing.expectEqualSlices(u32, &h_c1, &h_w1);
    try std.testing.expectEqualSlices(u32, &h_c2, &h_w2);

    const lo1: u64 = (len1 + 1) - w.window_size;
    const c1_ring = try readRingActive(env.queue, env.device, env.drafter.contexts.getPtr(&cold1).?, lo1, len1);
    defer gpa.free(c1_ring);
    const w1_ring = try readRingActive(env.queue, env.device, env.drafter.contexts.getPtr(&warm1).?, lo1, len1);
    defer gpa.free(w1_ring);
    try std.testing.expectEqualSlices(u8, c1_ring, w1_ring);

    const lo2: u64 = (len2 + 1) - w.window_size;
    const c2_ring = try readRingActive(env.queue, env.device, env.drafter.contexts.getPtr(&cold2).?, lo2, len2);
    defer gpa.free(c2_ring);
    const w2_ring = try readRingActive(env.queue, env.device, env.drafter.contexts.getPtr(&warm2).?, lo2, len2);
    defer gpa.free(w2_ring);
    try std.testing.expectEqualSlices(u8, c2_ring, w2_ring);

    try absorbSlice(dr, &cold1, env.taps_buf, 0, len1, 3, false);
    try absorbSlice(dr, &warm1, env.taps_buf, 0, len1, 3, false);
    try absorbSlice(dr, &cold2, env.taps_buf, 32, len2, 3, false);
    try absorbSlice(dr, &warm2, env.taps_buf, 32, len2, 3, false);
    try dr.hold(&.{
        .{ .stream = &cold1, .pending = 77, .position = len1 + 4, .depth = 7 },
        .{ .stream = &warm1, .pending = 77, .position = len1 + 4, .depth = 7 },
        .{ .stream = &cold2, .pending = 88, .position = len2 + 4, .depth = 7 },
        .{ .stream = &warm2, .pending = 88, .position = len2 + 4, .depth = 7 },
    });
    try dr.held(&.{ &cold1, &warm1, &cold2, &warm2 }, &.{ &h_c1, &h_w1, &h_c2, &h_w2 });
    try std.testing.expectEqualSlices(u32, &h_c1, &h_w1);
    try std.testing.expectEqualSlices(u32, &h_c2, &h_w2);
}

test "DFlash2 prompt marks below prompt_len - window stay intact for shorter resumed prompts" {
    var env = try SyntheticEnv.init(2500);
    defer env.deinit();
    const dr = env.drafter.drafter();

    const long_len: usize = 2500;
    const short_len: usize = 300;
    const mark: u32 = 256;
    const long_ids = try gpa.alloc(u32, long_len);
    defer gpa.free(long_ids);
    for (long_ids, 0..) |*id, i| id.* = @intCast(50 + i % 100);
    const marks = [_]u32{mark};

    var slot0 = try lanes.Stream.init(gpa, .{ .id = "slot0", .prompt = long_ids[0..64], .max_new = 4 });
    defer slot0.deinit(gpa);
    var saver = try lanes.Stream.init(gpa, .{ .id = "saver", .prompt = long_ids, .max_new = 8, .reuse = .{ .marks = &marks } });
    defer saver.deinit(gpa);
    var warm = try lanes.Stream.init(gpa, .{ .id = "warm", .prompt = long_ids[0..short_len], .max_new = 8 });
    defer warm.deinit(gpa);
    var cold = try lanes.Stream.init(gpa, .{ .id = "cold", .prompt = long_ids[0..short_len], .max_new = 8 });
    defer cold.deinit(gpa);

    try dr.open(&slot0);
    defer dr.release(&slot0);
    try dr.open(&saver);
    defer dr.release(&saver);
    try dr.open(&warm);
    defer dr.release(&warm);
    try dr.open(&cold);
    defer dr.release(&cold);

    try absorbSlice(dr, &saver, env.taps_buf, 0, 0, mark, true);
    var snap = (try env.drafter.saveSnap(&saver, mark, .{}, 0)).?;
    defer snap.free();
    try std.testing.expectEqual(@as(u64, 0), snap.from);
    try std.testing.expectEqual(@as(u64, mark), snap.end);

    try env.drafter.restoreSnap(&warm, &snap, mark);
    try absorbSlice(dr, &warm, env.taps_buf, 0, mark, short_len - mark, true);
    try absorbSlice(dr, &cold, env.taps_buf, 0, 0, short_len, true);

    try dr.hold(&.{
        .{ .stream = &cold, .pending = 19, .position = short_len + 1, .depth = 7 },
        .{ .stream = &warm, .pending = 19, .position = short_len + 1, .depth = 7 },
    });
    var cold_held: [7]u32 = undefined;
    var warm_held: [7]u32 = undefined;
    try dr.held(&.{ &cold, &warm }, &.{ &cold_held, &warm_held });
    try std.testing.expectEqualSlices(u32, &cold_held, &warm_held);
}

test "backend.Metal snapSave, restoreSnap, and snapDrop with DFlash2 at slot >= 1 leak zero memory" {
    var env = try SyntheticEnv.init(160);
    defer env.deinit();
    const m = try backend.Metal.init(gpa, &env.model, .{ .capacity = 512, .streams = 3, .budget = 2 << 30 });
    defer m.deinit();
    m.drafter = .{ .dflash2 = &env.drafter };
    m.prepared = true;

    const ids: [128]u32 = @splat(1);
    var s0 = try lanes.Stream.init(gpa, .{ .id = "s0", .prompt = &ids, .max_new = 4 });
    defer s0.deinit(gpa);
    var s1 = try lanes.Stream.init(gpa, .{ .id = "s1", .prompt = &ids, .max_new = 4 });
    defer s1.deinit(gpa);
    var s2 = try lanes.Stream.init(gpa, .{ .id = "s2", .prompt = &ids, .max_new = 4 });
    defer s2.deinit(gpa);
    var caches: [3]*st.Cache = undefined;
    for ([_]*lanes.Stream{ &s0, &s1, &s2 }, 0..) |stream, idx| {
        const cache = try m.createCache(256);
        errdefer {
            cache.deinit();
            gpa.destroy(cache);
        }
        try m.caches.put(gpa, stream, cache);
        caches[idx] = cache;
    }
    const c0 = caches[0];
    const c1 = caches[1];
    const c2 = caches[2];
    c0.len = 64;
    c1.len = 128;

    const dr = env.drafter.drafter();
    try dr.open(&s0);
    defer dr.release(&s0);
    try dr.open(&s1);
    defer dr.release(&s1);
    try dr.open(&s2);
    defer dr.release(&s2);
    try absorbSlice(dr, &s0, env.taps_buf, 0, 0, 64, true);
    try absorbSlice(dr, &s1, env.taps_buf, 16, 0, 128, true);

    const want_bytes = snapshot.bytes(128) + state.Snap.bytes(128);
    try std.testing.expectEqual(want_bytes, backend.Metal.snapBytes(m, 128));
    try std.testing.expect(want_bytes > snapshot.bytes(128));

    // Store keeps and drops snapshots saved from slot 1 via snapSave and snapDrop under std.testing.allocator.
    const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .planned = true, .grid = st.prompt_rows, .min_prompt = 0 };
    var store = api.prompt_cache.Store.init(gpa, .{
        .ptr = m,
        .vtable = &.{ .bytes = backend.Metal.snapBytes, .save = backend.Metal.snapSave, .restore = backend.Metal.snapRestore, .drop = backend.Metal.snapDrop },
    }, rules, want_bytes);
    defer store.deinit();
    try std.testing.expect(store.keep(&ids, 128, &s1, &.{}, &.{}));
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    const saved = store.entries.items[0].saved;

    // Restore into slot 2 (s2) while slot 0 (s0) stays live; verify both target cache and DFlash2 lane.
    try std.testing.expectError(error.BackendRestores, backend.Metal.snapRestore(m, &s2, saved));
    const snap_ptr: *const backend.SavedSnap = @ptrCast(@alignCast(saved));
    try m.restoreSnap(c2, &s2, snap_ptr);
    try std.testing.expectEqual(@as(usize, 128), c2.len);
    try std.testing.expectEqual(@as(u64, 128), env.drafter.contexts.getPtr(&s2).?.end);
    try std.testing.expectEqual(@as(u64, 64), env.drafter.contexts.getPtr(&s0).?.end);

    // Negative controls: active-lane restore refusal, out-of-step at_pos refusal, and drafter-less snapshot refusal.
    try std.testing.expectError(error.DFlash2SnapshotGeometry, m.restoreSnap(c2, &s2, snap_ptr));
    c2.len = 0;
    env.drafter.resetLane(&s2);
    try std.testing.expectError(error.DFlash2SnapshotGeometry, env.drafter.restoreSnap(&s2, &snap_ptr.drafter.?.dflash2, 129));
    try std.testing.expectEqual(@as(u64, 0), env.drafter.contexts.getPtr(&s2).?.end);

    m.drafter = null;
    const plain_saved = try backend.Metal.snapSave(m, &s1, 128);
    defer backend.Metal.snapDrop(m, plain_saved);
    m.drafter = .{ .dflash2 = &env.drafter };
    const plain_ptr: *const backend.SavedSnap = @ptrCast(@alignCast(plain_saved));
    try std.testing.expectError(error.DFlash2SnapshotMissing, m.restoreSnap(c2, &s2, plain_ptr));
    env.drafter.resetLane(&s2);
    try std.testing.expectEqual(@as(u64, 0), env.drafter.contexts.getPtr(&s2).?.end);
    try std.testing.expectEqual(@as(u64, 64), env.drafter.contexts.getPtr(&s0).?.end);
}
