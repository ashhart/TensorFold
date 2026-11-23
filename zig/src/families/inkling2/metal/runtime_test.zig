//! Host-only schedule, cache-layout, accepted-history and compact-snapshot checks.
const std = @import("std");
const builtin = @import("builtin");
const mtl = @import("metal");
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
            const source = state.convByteOffset(slot, cs, accepted, 3 + state.max_rows);
            try std.testing.expect(source + len <= state.taps_bytes);
        }
    };
    try std.testing.expectEqual(state.conv_bytes, previous_persistent);
    // Decode acceptance k preserves exactly old[k], old[k+1], old[k+2] through raw[k..k+3].
    for (0..state.window_rows + 1) |accepted| {
        const old = state.convByteOffset(47, 3, 0, 3 + state.max_rows);
        const selected = state.convByteOffset(47, 3, accepted, 3 + state.max_rows);
        try std.testing.expectEqual(old + accepted * state.hidden * 2, selected);
    }
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

test "Model.scanNvfp4Weights merges all 20 NVFP4 layers and skips null globals" {
    if (!builtin.os.tag.isDarwin()) return error.SkipZigTest;
    var w: model.Weights = undefined;
    for (0..24) |i| {
        w.layers[i].gu = .{ .weight = undefined, .scale = undefined, .amax = null, .denominator = null };
        w.layers[i].down = .{ .weight = undefined, .scale = undefined, .amax = null, .denominator = null };
    }
    const empty = model.Model.scanNvfp4Weights(&w);
    try std.testing.expectEqual(@as(usize, 0), empty.experts);
    try std.testing.expect(empty.finite_globals and empty.normal_range and empty.two_prod_exact and empty.finite_scales);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const amax_buf = try device.buffer(16, mtl.ResourceOptions.shared);
    defer amax_buf.deinit();
    const den_buf = try device.buffer(8, mtl.ResourceOptions.shared);
    defer den_buf.deinit();
    const sc_buf = try device.buffer(16, mtl.ResourceOptions.shared);
    defer sc_buf.deinit();
    @memcpy(amax_buf.slice(f32, 2), &[_]f32{ 2688.125, 2688.5 });
    den_buf.slice(f32, 1)[0] = 2695.75 * 64.0;
    @memcpy(sc_buf.slice(u8, 4), &[_]u8{ 0x38, 0x39, 0x3a, 0x3b });
    const mat = model.ExpertMatrix{
        .weight = undefined,
        .scale = .{ .buffer = sc_buf, .offset = 0, .bytes = 4, .dtype = .f8_e4m3 },
        .amax = .{ .buffer = amax_buf, .offset = 0, .bytes = 8, .dtype = .f32 },
        .denominator = .{ .buffer = den_buf, .offset = 0, .bytes = 4, .dtype = .f32 },
    };
    w.layers[0].gu = mat;
    w.layers[19].down = mat;
    // Layer 20 is MXFP8 and must be ignored by the 20-layer NVFP4 scan even if populated.
    w.layers[20].gu = mat;
    const ok = model.Model.scanNvfp4Weights(&w);
    try std.testing.expectEqual(@as(usize, 4), ok.experts);
    try std.testing.expect(ok.finite_globals and ok.normal_range and ok.two_prod_exact and ok.finite_scales);
    amax_buf.slice(f32, 2)[0] = 0x1.0p-118;
    sc_buf.slice(u8, 4)[0] = 0x7f;
    const bad = model.Model.scanNvfp4Weights(&w);
    try std.testing.expect(!bad.normal_range and !bad.finite_scales and bad.usesThreeProdFallback());
    try std.testing.expectEqual(@as(usize, 2), bad.unsafe_phi_experts);
    try std.testing.expectEqual(@as(usize, 2), bad.nan_scales);
    amax_buf.slice(f32, 2)[0] = 1.0 + 1.0 / 64.0;
    amax_buf.slice(f32, 2)[1] = 1.0 + 1.0 / 64.0;
    den_buf.slice(f32, 1)[0] = 1.5;
    sc_buf.slice(u8, 4)[0] = 0x38;
    const tie = model.Model.scanNvfp4Weights(&w);
    try std.testing.expect(tie.normal_range and !tie.two_prod_exact and tie.usesThreeProdFallback());
    try std.testing.expectEqual(@as(usize, 4), tie.tie_fallback_experts);
}
