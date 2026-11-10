//! Host tests for target-coupled keyed-Gumbel draft draws in Qwen CUDA MTP (solo, tree, and batched shared rounds).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const mtp_draw = @import("cuda_mtp_draw.zig");
const sim_mod = @import("cuda_mtp_coupled_sim.zig");
const rig_mod = @import("cuda_mtp_coupled_rig.zig");
const t = std.testing;

const Sim = sim_mod.Sim;
const Rig = rig_mod.Rig;
const Target = rig_mod.Target;
const runEndToEnd = rig_mod.runEndToEnd;
const vocab = sim_mod.vocab;
const hidden = sim_mod.hidden;
const init_state = sim_mod.init_state;
const stepState = sim_mod.stepState;
const drawState = sim_mod.drawState;
const writeLogits = sim_mod.writeLogits;
const bf16ToF32 = sim_mod.bf16ToF32;

test "MTP coupled: solo chain sampled drafts match target draws across multi-step holds and high positions" {
    for ([_]u64{ 0, 1, 65536, 0x1_0000_0000, 0x1_0000_0009 }) |pos| {
        const m = mtp_draw.metaOf(pos);
        try t.expectEqual(@as(u32, @truncate(pos)), @as(u32, @bitCast(m)) +% 1);
    }
    const cases = [_]lanes.Sampling{
        .{ .seed = 42, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 99, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 777, .temperature = 0.85, .top_k = 12, .top_p = 0.9, .min_p = 0.08 },
    };
    for (cases) |samp| {
        const res = try runEndToEnd(&.{samp}, .{ .drafts = true, .batch = false, .max_new = 24 });
        try t.expect(res.drafted[0] >= 15);
        try t.expectEqual(res.drafted[0], res.accepted[0]);
        try t.expect(Sim.draw_rows_launches > 0);
        try t.expectEqual(@as(usize, 0), Sim.argmax_launches);
    }
    var rig: Rig = .{};
    try rig.init(t.allocator, false, false);
    defer rig.deinit();
    const tap = try cuda.DeviceBuffer.alloc(&rig.drv, engine.tap_rows * hidden * 2);
    defer {
        var tmp = tap;
        tmp.free();
    }
    const samp: lanes.Sampling = .{ .seed = 31415, .temperature = 0.9, .top_k = 0 };
    var s = try lanes.Stream.init(t.allocator, .{ .id = "hi", .prompt = &.{ 3, 9 }, .max_new = 4, .sampling = samp });
    defer s.deinit(t.allocator);
    try rig.d.drafter().open(&s);
    const high_start: u64 = 70_000;
    var st = stepState(init_state, 3, high_start + 1);
    Target.writeTap(tap, high_start, st);
    st = stepState(st, 9, high_start + 2);
    Target.writeTap(tap, high_start + 1, st);
    try rig.d.drafter().absorb(&.{.{
        .stream = &s,
        .features = .{ .buffer = tap.ptr, .offset = high_start, .rows = 2, .row_bytes = hidden * 2, .space = .device, .dtype = .bf16 },
        .start = high_start,
        .follow = &.{ 9, 14 },
    }});
    try rig.d.drafter().hold(&.{.{ .stream = &s, .pending = 14, .position = high_start + 3, .depth = 3, .sampling = samp }});
    var got: [3]u32 = undefined;
    try rig.d.drafter().held(&.{&s}, &.{&got});
    var cur = stepState(st, 14, high_start + 3);
    for (got, 0..) |tok, k| {
        const want = try drawState(t.allocator, cur, samp, high_start + 3 + k);
        try t.expectEqual(want, tok);
        cur = stepState(cur, tok, high_start + 4 + k);
    }
}

test "MTP coupled: batched shared rounds match serial drafts and couple mixed sampled and greedy streams" {
    const mixed = [_]?lanes.Sampling{
        .{ .seed = 11, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        null,
        .{ .seed = 22, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 33, .temperature = 0.0 },
    };
    const b_res = try runEndToEnd(&mixed, .{ .drafts = true, .batch = true, .max_new = 20 });
    try t.expect(b_res.batch_passes > 0);
    try t.expect(Sim.draw_rows_launches > 0 and Sim.argmax_launches > 0);
    for (0..mixed.len) |i| {
        try t.expect(b_res.drafted[i] >= 12);
        try t.expectEqual(b_res.drafted[i], b_res.accepted[i]);
    }
    const s_res = try runEndToEnd(&mixed, .{ .drafts = true, .batch = false, .max_new = 20 });
    try t.expectEqualSlices([24]u32, &s_res.replies, &b_res.replies);
    try t.expectEqualSlices(u64, &s_res.accepted, &b_res.accepted);
    try t.expectEqualSlices(u64, &s_res.drafted, &b_res.drafted);
    var wide_held: [2][17][3]u32 = undefined;
    for ([_]bool{ false, true }, 0..) |use_batch, bi| {
        var rig: Rig = .{};
        try rig.init(t.allocator, false, use_batch);
        defer rig.deinit();
        const tap = try cuda.DeviceBuffer.alloc(&rig.drv, engine.tap_rows * hidden * 2);
        defer {
            var tmp = tap;
            tmp.free();
        }
        var streams: [17]lanes.Stream = undefined;
        var opened: usize = 0;
        defer for (streams[0..opened]) |*s| s.deinit(t.allocator);
        var absorbs: [17]lanes.drafter.Absorb = undefined;
        var holds: [17]lanes.drafter.Hold = undefined;
        var follows: [17][1]u32 = undefined;
        for (&streams, 0..) |*s, i| {
            const samp: ?lanes.Sampling = if (i % 5 == 3) null else .{ .seed = 500 + i, .temperature = 0.8, .top_k = if (i % 2 == 0) 20 else 0 };
            s.* = try lanes.Stream.init(t.allocator, .{ .id = "w", .prompt = &.{@intCast(1 + i)}, .max_new = 4, .sampling = samp });
            opened += 1;
            try rig.d.drafter().open(s);
            const slot: u64 = i * 4;
            Target.writeTap(tap, slot, stepState(init_state, @intCast(1 + i), 1));
            follows[i] = .{@intCast(2 + i)};
            absorbs[i] = .{
                .stream = s,
                .features = .{ .buffer = tap.ptr, .offset = slot, .rows = 1, .row_bytes = hidden * 2, .space = .device, .dtype = .bf16 },
                .start = slot,
                .follow = &follows[i],
            };
            holds[i] = .{ .stream = s, .pending = follows[i][0], .position = slot + 2, .depth = 3, .sampling = samp };
        }
        try rig.d.drafter().absorb(&absorbs);
        try rig.d.drafter().hold(&holds);
        if (use_batch) try t.expectEqual(@as(u64, 6), rig.d.batch_passes);
        for (&streams, 0..) |*s, i| try rig.d.drafter().held(&.{s}, &.{&wide_held[bi][i]});
    }
    try t.expectEqual(wide_held[0], wide_held[1]);
}

test "MTP coupled: tree mode anchors rank 0 to sample.cu and orders runner-up branches by Gumbel race" {
    var parents = [_]i32{ -1, -1, -1, -1, 0, 0, 4 };
    var depths = [_]u8{ 0, 0, 0, 0, 1, 1, 2 };
    var tree_ranks = [_]u8{ 0, 1, 2, 3, 0, 1, 0 };
    const shape: lanes.shape.Shape = .{ .parents = &parents, .depths = &depths, .ranks = &tree_ranks, .expected = 1.0 };
    for ([_]lanes.Sampling{
        .{ .seed = 7, .temperature = 0.9, .top_k = 20, .top_p = 0.92 },
        .{ .seed = 13, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 29, .temperature = 1.2, .top_k = 0, .top_p = 1.0 },
        .{ .seed = 47, .temperature = 1.2, .top_k = 20, .top_p = 1.0 },
    }) |samp| {
        var rig: Rig = .{};
        try rig.init(t.allocator, true, false);
        defer rig.deinit();
        const tap = try cuda.DeviceBuffer.alloc(&rig.drv, engine.tap_rows * hidden * 2);
        defer {
            var tmp = tap;
            tmp.free();
        }
        var s = try lanes.Stream.init(t.allocator, .{ .id = "tree", .prompt = &.{5}, .max_new = 4, .sampling = samp });
        defer s.deinit(t.allocator);
        try rig.d.drafter().open(&s);
        const st0 = stepState(init_state, 5, 1);
        Target.writeTap(tap, 0, st0);
        try rig.d.drafter().absorb(&.{.{
            .stream = &s,
            .features = .{ .buffer = tap.ptr, .offset = 0, .rows = 1, .row_bytes = hidden * 2, .space = .device, .dtype = .bf16 },
            .start = 0,
            .follow = &.{12},
        }});
        try rig.d.drafter().hold(&.{.{ .stream = &s, .pending = 12, .position = 2, .depth = 3, .lanes = &shape, .sampling = samp }});
        var got: [7]u32 = undefined;
        try rig.d.drafter().held(&.{&s}, &.{&got});
        const st1 = stepState(st0, 12, 2);
        const want0 = try drawState(t.allocator, st1, samp, 2);
        try t.expectEqual(want0, got[0]);
        for (1..4) |j| {
            try t.expect(got[j] != got[0]);
            for (1..j) |prev| try t.expect(got[j] != got[prev]);
        }
        {
            const rl = cuda.sample.rule(samp);
            var words: [vocab]u16 = undefined;
            writeLogits(st1, &words);
            const full = rl.top_k == 0 or rl.top_k > lanes.gpu_rule.candidates;
            const u1_v = if (full) lanes.gpu_full.raceUniform(rl.seed, 2, got[1]) else lanes.gpu_rule.uniform24(rl.seed, 2, got[1]);
            const u2_v = if (full) lanes.gpu_full.raceUniform(rl.seed, 2, got[2]) else lanes.gpu_rule.uniform24(rl.seed, 2, got[2]);
            const u3_v = if (full) lanes.gpu_full.raceUniform(rl.seed, 2, got[3]) else lanes.gpu_rule.uniform24(rl.seed, 2, got[3]);
            const s1 = bf16ToF32(words[got[1]]) * rl.inv_t - @log(-@log(u1_v));
            const s2 = bf16ToF32(words[got[2]]) * rl.inv_t - @log(-@log(u2_v));
            const s3 = bf16ToF32(words[got[3]]) * rl.inv_t - @log(-@log(u3_v));
            if (samp.top_p >= 1.0) try t.expect(s1 >= s2 and s2 >= s3);
        }
        const st2 = stepState(st1, got[0], 3);
        const want1 = try drawState(t.allocator, st2, samp, 3);
        try t.expectEqual(want1, got[4]);
        try t.expect(got[5] != got[4]);
        const st3 = stepState(st2, got[4], 4);
        const want2 = try drawState(t.allocator, st3, samp, 4);
        try t.expectEqual(want2, got[6]);
    }
    var w_row: [43]u32 = @splat(0xFFFF_FFFF);
    w_row[0] = 0;
    w_row[1] = 0;
    const flat_logits = [_]f32{ 4.0, 3.99, 3.98, 3.97, 3.96, 3.95, 0.5, 0.4 };
    for (flat_logits, 0..) |lv, j| {
        w_row[2 + j] = @intCast(10 + j);
        w_row[2 + 20 + j] = @bitCast(lv);
    }
    var saw_reordered = false;
    for (1..32) |sd| {
        const r_full = mtp_draw.rankTreeRow(cuda.sample.rule(@as(lanes.Sampling, .{ .seed = sd, .temperature = 1.0, .top_k = 0 })), 9, 10, &w_row);
        if (r_full[1] != 11 or r_full[2] != 12) saw_reordered = true;
        const r_cap = mtp_draw.rankTreeRow(cuda.sample.rule(@as(lanes.Sampling, .{ .seed = sd, .temperature = 1.0, .top_k = 2 })), 9, 10, &w_row);
        try t.expectEqual(@as(u32, 11), r_cap[1]);
        const r_topp = mtp_draw.rankTreeRow(cuda.sample.rule(@as(lanes.Sampling, .{ .seed = sd, .temperature = 1.0, .top_k = 0, .top_p = 0.25 })), 9, 10, &w_row);
        try t.expectEqual(@as(u32, 11), r_topp[1]);
        const r_minp = mtp_draw.rankTreeRow(cuda.sample.rule(@as(lanes.Sampling, .{ .seed = sd, .temperature = 1.0, .top_k = 0, .min_p = 0.1 })), 9, 10, &w_row);
        try t.expect(r_minp[1] < 16 and r_minp[2] < 16 and r_minp[3] < 16);
    }
    try t.expect(saw_reordered);
}

test "MTP coupled: greedy solo batched and tree streams keep legacy argmax with zero rule uploads" {
    const greedy_pair = [_]?lanes.Sampling{ null, .{ .seed = 88, .temperature = 0.0 } };
    for ([_]bool{ false, true }) |trees| {
        const res = try runEndToEnd(if (trees) greedy_pair[0..1] else &greedy_pair, .{ .drafts = true, .trees = trees, .batch = !trees, .max_new = 16 });
        try t.expectEqual(@as(usize, 0), Sim.draw_rows_launches);
        try t.expectEqual(@as(usize, 0), Sim.rule_uploads);
        try t.expect(Sim.argmax_launches > 0);
        try t.expect(res.drafted[0] >= 10);
        if (!trees) try t.expectEqual(res.drafted[0], res.accepted[0]);
    }
}

test "MTP coupled: emitted replies equal no-draft baseline for greedy and seeded sampling under matching or diverging head" {
    const cells = [_]?lanes.Sampling{
        null,
        .{ .seed = 5, .temperature = 0.0 },
        .{ .seed = 101, .temperature = 0.7, .top_k = 20, .top_p = 0.95 },
        .{ .seed = 202, .temperature = 1.0, .top_k = 0, .top_p = 1.0 },
    };
    const base_4 = try runEndToEnd(&cells, .{ .drafts = false, .max_new = 24 });
    for ([_]u64{ 0, 0xCAFE_BABE_DEAD_BEEF }) |bias| {
        const batched_4 = try runEndToEnd(&cells, .{ .drafts = true, .batch = true, .head_bias = bias, .max_new = 24 });
        try t.expectEqualSlices([24]u32, &base_4.replies, &batched_4.replies);
        const serial_4 = try runEndToEnd(&cells, .{ .drafts = true, .batch = false, .head_bias = bias, .max_new = 24 });
        try t.expectEqualSlices([24]u32, &base_4.replies, &serial_4.replies);
        for (cells, 0..) |samp, i| {
            const base_1 = try runEndToEnd(&.{samp}, .{ .drafts = false, .prompt_seed = i, .max_new = 24 });
            const solo_1 = try runEndToEnd(&.{samp}, .{ .drafts = true, .batch = false, .head_bias = bias, .prompt_seed = i, .max_new = 24 });
            const tree_1 = try runEndToEnd(&.{samp}, .{ .drafts = true, .trees = true, .batch = false, .head_bias = bias, .prompt_seed = i, .max_new = 24 });
            try t.expectEqualSlices(u32, &base_4.replies[i], &base_1.replies[0]);
            try t.expectEqualSlices(u32, &base_1.replies[0], &solo_1.replies[0]);
            try t.expectEqualSlices(u32, &base_1.replies[0], &tree_1.replies[0]);
            if (bias != 0 and samp != null and samp.?.temperature > 0) try t.expect(solo_1.accepted[0] < solo_1.drafted[0]);
        }
    }
    try t.expectEqual(@as(usize, 0), Sim.live_allocs.load(.monotonic));
}
