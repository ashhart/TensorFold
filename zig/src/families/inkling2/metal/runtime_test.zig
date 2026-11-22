//! Host-only schedule, cache-layout, accepted-history and compact-snapshot checks.
const std = @import("std");
const mtl = @import("metal");
const ap = @import("attention_plan.zig");
const cfg = @import("../config.zig");
const layer_plan = @import("../plan.zig");
const state = @import("state.zig");
const snapshot = @import("snapshot.zig");
const model = @import("model.zig");
const fwd = @import("forward.zig");
const backend = @import("backend.zig");

fn mini() cfg.Config {
    return .{
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
    };
}

fn schedule() layer_plan.Plan {
    const c = mini();
    var precision: [24]layer_plan.Precision = @splat(.nvfp4);
    for (precision[4..]) |*p| p.* = .mxfp8;
    return layer_plan.build(&c.text, &precision) catch unreachable;
}

test "Mini admits only its fixed geometry and expands 24 weights to 48 visits" {
    const c = mini();
    try model.admitConfig(&c);
    const p = schedule();
    try std.testing.expectEqual(@as(u32, 48), p.n_visits);
    try std.testing.expectEqual(@as(u32, 8), p.visitCount(.global));
    try std.testing.expectEqual(@as(u32, 5), p.final_sources);
    for (p.visits[0..p.n_visits], 0..) |v, i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)), v.cache_slot);
        try std.testing.expect(v.layer < 24);
    }
    for ([_]usize{ 18, 30 }) |i| {
        try std.testing.expect(p.visits[i].pass > 1);
        try std.testing.expectEqual(@as(u32, 6), p.visits[i].layer);
        try std.testing.expect(p.visits[i].hasMessages());
    }
    try std.testing.expect(!p.visits[0].hasMessages());
    try std.testing.expect(!(p.visits[17].pass > 1 and p.visits[17].layer == 6));
}

test "Every visit and convolution role has disjoint persistent and staged bytes" {
    var previous_persistent: usize = 0;
    for (0..state.visit_slots) |slot| for (0..state.conv_states) |cs| {
        const off = state.convByteOffset(slot, cs, 0, 3);
        const len = 3 * state.convWidth(cs) * 2;
        if (slot != 0 or cs != 0) try std.testing.expect(off >= previous_persistent);
        previous_persistent = off + len;
        for (0..4) |accepted| {
            const source = state.convByteOffset(slot, cs, accepted, 3 + state.window_rows);
            try std.testing.expect(source + len <= state.taps_bytes);
        }
    };
    try std.testing.expectEqual(state.conv_bytes, previous_persistent);
    // Decode acceptance k preserves exactly old[k], old[k+1], old[k+2] through raw[k..k+3].
    for (0..state.window_rows + 1) |accepted| {
        const old = state.convByteOffset(47, 3, 0, 3 + state.window_rows);
        const selected = state.convByteOffset(47, 3, accepted, 3 + state.window_rows);
        try std.testing.expectEqual(old + accepted * state.hidden * 2, selected);
    }
    try std.testing.expectEqual(@as(usize, 48 * 19 * 6656 * 2), state.taps_bytes);
}

test "Compact snapshots charge global prefixes and sliding physical rings" {
    const one_layer_pair: u64 = state.kv_width * 2 * 2;
    for ([_]usize{ 0, 1, 1024, state.sliding_ring, state.sliding_ring + 1 }) |at| {
        const n: u64 = @intCast(at);
        const live = @as(u64, 8) * n + @as(u64, 40) * @as(u64, @intCast(@min(at, state.sliding_ring)));
        const kv = live * one_layer_pair;
        const expected = kv + state.conv_bytes;
        try std.testing.expectEqual(expected, snapshot.bytes(at));
    }
    try std.testing.expect(snapshot.bytes(4096) < snapshot.bytes(8192));
    try std.testing.expectEqual(state.conv_bytes, state.visit_slots * 3 * state.conv_row_values * 2);
}

test "DSpark feature rows are 4x2304 BF16 and ranges expose only live accepted rows" {
    try std.testing.expectEqual(@as(usize, 4 * 2304 * 2), state.feature_row_bytes);
    try std.testing.expectEqual(@as(usize, 7 * 4 * 2304 * 2), state.captureBytes(7));
    const off = 3 * state.feature_row_bytes;
    const tp = state.Tap{ .buf = null, .off = off, .start = 100, .rows = 4, .gen = 9 };
    try std.testing.expectEqual(off, try state.featureOffset(tp, 100, 4, 9));
    try std.testing.expectEqual(off + 2 * state.feature_row_bytes, try state.featureOffset(tp, 102, 2, 9));
    try std.testing.expectError(error.StaleInkling2Taps, state.featureOffset(tp, 100, 1, 10));
    try std.testing.expectError(error.Inkling2TapsNotHeld, state.featureOffset(tp, 99, 1, 9));
    try std.testing.expectError(error.Inkling2TapsNotHeld, state.featureOffset(tp, 102, 3, 9));
    try std.testing.expectError(error.Inkling2TapsNotHeld, state.featureOffset(tp, 104, 1, 9));
    try std.testing.expectError(error.Inkling2TapsNotHeld, state.featureOffset(tp, 100, 0, 9));
    var accepted = tp;
    accepted.rows = 2;
    try std.testing.expectEqual(off + state.feature_row_bytes, try state.featureOffset(accepted, 101, 1, 9));
    try std.testing.expectError(error.Inkling2TapsNotHeld, state.featureOffset(accepted, 102, 1, 9));
    accepted.gen = null; // the copied prompt-last row survives later scratch generations
    try std.testing.expectEqual(off, try state.featureOffset(accepted, 100, 1, 1000));
}

test "tap48 captures every final-norm row but last-row-only logits keep their exact source" {
    const captured = fwd.finalPlan(7, false, true);
    try std.testing.expectEqual(@as(usize, 7), captured.work_rows);
    try std.testing.expectEqual(@as(usize, 7), captured.capture_rows);
    try std.testing.expectEqual(@as(usize, 1), captured.logits_rows);
    try std.testing.expectEqual(@as(usize, 6), captured.logits_row);
    try std.testing.expectEqual(@as(usize, 6 * state.hidden * 2), captured.logits_row * state.hidden * 2);
    const all = fwd.finalPlan(7, true, true);
    try std.testing.expectEqual(@as(usize, 7), all.logits_rows);
    try std.testing.expectEqual(@as(usize, 0), all.logits_row);
}

test "no-draft defaults retain KIM2's single last-row final norm and no capture" {
    const options: fwd.Options = .{};
    try std.testing.expect(options.taps == null);
    const plain = fwd.finalPlan(7, false, options.taps != null);
    try std.testing.expectEqual(@as(usize, 1), plain.work_rows);
    try std.testing.expectEqual(@as(usize, 0), plain.capture_rows);
    try std.testing.expectEqual(@as(usize, 1), plain.logits_rows);
    try std.testing.expectEqual(@as(usize, 0), plain.logits_row);
    const all = fwd.finalPlan(7, true, false);
    try std.testing.expectEqual(@as(usize, 7), all.work_rows);
    try std.testing.expectEqual(@as(usize, 7), all.logits_rows);
    try std.testing.expectEqual(@as(usize, 0), all.capture_rows);
}

test "prepare_features requires exactly Mini's checkpoint tap list" {
    var c = mini();
    c.mtp = .{
        .layers = 3,
        .q_heads = 24,
        .kv_heads = 8,
        .head_dim = 128,
        .inter = 9216,
        .rope_theta = 1e6,
        .window = 1024,
        .local = .{ true, false, true, false, false, false, false, false },
        .block = 8,
        .rank = 256,
        .taps = .{ 5, 23, 41, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        .taps_n = 4,
    };
    try backend.validateTaps(&c, &.{ 5, 23, 41, 48 });
    try std.testing.expectError(error.UnsupportedInkling2Taps, backend.validateTaps(&c, &.{ 5, 23, 41 }));
    try std.testing.expectError(error.UnsupportedInkling2Taps, backend.validateTaps(&c, &.{ 5, 11, 41, 48 }));
    c.mtp.?.taps[1] = 22;
    try std.testing.expectError(error.UnsupportedInkling2Taps, backend.validateTaps(&c, &.{ 5, 23, 41, 48 }));
    c.mtp = null;
    try std.testing.expectError(error.UnsupportedInkling2Taps, backend.validateTaps(&c, &.{ 5, 23, 41, 48 }));
}

/// Every head's first `rows` rows of every slot's keys and values, then the convolution states, to or from `flat`.
fn moveLive(queue: mtl.Queue, cache: *const state.Cache, flat: mtl.Buffer, rows: usize, into: bool) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    const n = rows * ap.head_dim * 2;
    var at: usize = 0;
    for (0..state.visit_slots) |slot| for ([_]usize{ cache.key_at[slot], cache.value_at[slot] }) |base| for (0..state.kv_width / ap.head_dim) |h| {
        const off = base + h * cache.ring(slot) * ap.head_dim * 2;
        if (into) e.copy(flat, at, cache.kv, off, n) else e.copy(cache.kv, off, flat, at, n);
        at += n;
    };
    if (into) e.copy(flat, at, cache.conv, 0, state.conv_bytes) else e.copy(cache.conv, 0, flat, at, state.conv_bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

test "A snapshot restores each head's live rows into a cache of another capacity" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    const queue = try device.queue();
    defer queue.deinit();
    const p = schedule();
    const rows = 100;
    var from = try state.Cache.init(device, 200, &p);
    defer from.deinit();
    var to = try state.Cache.init(device, 500, &p);
    defer to.deinit();
    try std.testing.expect(from.ring(5) != to.ring(5) and from.globals[5] and rows < from.ring(0));
    const size = state.visit_slots * 2 * state.kv_width * 2 * rows + state.conv_bytes;
    const want = try device.buffer(size, mtl.ResourceOptions.shared);
    defer want.deinit();
    const got = try device.buffer(size, mtl.ResourceOptions.shared);
    defer got.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().bytes(want.slice(u8, size));
    try moveLive(queue, &from, want, rows, true);
    from.len = rows;
    var snap = try snapshot.save(device, queue, &from);
    defer snap.deinit();
    try snapshot.restore(&to, queue, &snap);
    try std.testing.expectEqual(@as(usize, rows), to.len);
    try moveLive(queue, &to, got, rows, false);
    try std.testing.expect(std.mem.eql(u8, want.slice(u8, size), got.slice(u8, size)));
}
