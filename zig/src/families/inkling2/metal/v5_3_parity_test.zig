//! Real Metal GPU bit-exact parity tests between serial/backup baseline and V5-3 across rows 1..256.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const fwd_ops = @import("forward_ops.zig");
const model_ = @import("model.zig");
const snapshot = @import("snapshot.zig");
const backend_ = @import("backend.zig");
const synth = @import("synthetic_model_test.zig");
const histories = @import("v5_3_history_test.zig");
const base_ref = @import("v5_3_baseline_test.zig");

const gpa = std.testing.allocator;
const taps_ids = [_]u32{ 5, 23, 41, 48 };

fn readPrivateBuffer(queue: mtl.Queue, src: mtl.Buffer, dst: mtl.Buffer, bytes: usize) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    e.copy(src, 0, dst, 0, bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

fn fillTokens(out: []u32, seed: usize) void {
    for (out, 0..) |*t, i| t.* = @intCast(17 + (i * 41 + seed * 131) % 19000);
}

fn expectConvAndTapsEqual(queue: mtl.Queue, a_conv: mtl.Buffer, b_conv: mtl.Buffer, stage_a: mtl.Buffer, stage_b: mtl.Buffer, a_taps: mtl.Buffer, b_taps: mtl.Buffer, rows: usize) !void {
    const tap_bytes = st.captureBytes(rows);
    try std.testing.expectEqualSlices(u8, a_taps.slice(u8, tap_bytes), b_taps.slice(u8, tap_bytes));
    try readPrivateBuffer(queue, a_conv, stage_a, st.conv_bytes);
    try readPrivateBuffer(queue, b_conv, stage_b, st.conv_bytes);
    try std.testing.expectEqualSlices(u8, stage_a.slice(u8, st.conv_bytes), stage_b.slice(u8, st.conv_bytes));
}

fn checkRowWidths(m: *model_.Model, stage_a: mtl.Buffer, stage_b: mtl.Buffer) !void {
    const vocab = m.config.text.unpadded_vocab;
    var s_base = try st.Scratch.init(m.device, st.prompt_rows);
    defer s_base.deinit();
    var s_v5 = try st.Scratch.init(m.device, st.prompt_rows);
    defer s_v5.deinit();
    s_base.taps = try m.device.buffer(st.captureBytes(st.prompt_rows), mtl.ResourceOptions.shared);
    s_v5.taps = try m.device.buffer(st.captureBytes(st.prompt_rows), mtl.ResourceOptions.shared);

    var bb = [_]base_ref.BaselineBuffers{try base_ref.BaselineBuffers.init(m.device)};
    defer bb[0].deinit();

    var prefix: [68]u32 = undefined;
    fillTokens(&prefix, 1);

    for ([_]usize{ 1, 2, 7, 8, 15, 16, 32, 256 }) |rows| {
        const cap = if (rows == 256) @as(usize, 320) else @as(usize, 128);
        var c_base = try st.Cache.init(m.device, cap, &m.weight_plan);
        defer c_base.deinit();
        var c_v5 = try st.Cache.init(m.device, cap, &m.weight_plan);
        defer c_v5.deinit();

        const pre_n: usize = if (rows == 256) 17 else 68;
        try base_ref.runBaselineSerial(m, &s_base, &bb, &.{.{ .cache = &c_base, .rows = pre_n }}, prefix[0..pre_n], .{ .all = true, .taps = .{ .buf = s_base.taps.? } });
        c_base.len += pre_n;
        try fwd.run(m, &s_v5, &c_v5, prefix[0..pre_n], .{ .all = true, .taps = .{ .buf = s_v5.taps.? } });
        c_v5.len += pre_n;

        const ids = try gpa.alloc(u32, rows);
        defer gpa.free(ids);
        fillTokens(ids, rows + 3);

        try base_ref.runBaselineSerial(m, &s_base, &bb, &.{.{ .cache = &c_base, .rows = rows }}, ids, .{ .all = true, .taps = .{ .buf = s_base.taps.? } });
        c_base.len += rows;
        try fwd.run(m, &s_v5, &c_v5, ids, .{ .all = true, .taps = .{ .buf = s_v5.taps.? } });
        c_v5.len += rows;

        const l_bytes = rows * vocab * 4;
        try std.testing.expectEqualSlices(u8, s_base.logits_all.slice(u8, l_bytes), s_v5.logits_all.slice(u8, l_bytes));
        try expectConvAndTapsEqual(m.queue, c_base.conv, c_v5.conv, stage_a, stage_b, s_base.taps.?, s_v5.taps.?, rows);
    }
}

fn checkAcceptRejectAndRollback(m: *model_.Model, stage_a: mtl.Buffer, stage_b: mtl.Buffer) !void {
    const vocab = m.config.text.unpadded_vocab;
    var s_base = try st.Scratch.init(m.device, st.prompt_rows);
    defer s_base.deinit();
    var s_v5 = try st.Scratch.init(m.device, st.prompt_rows);
    defer s_v5.deinit();
    s_base.taps = try m.device.buffer(st.captureBytes(st.prompt_rows), mtl.ResourceOptions.shared);
    s_v5.taps = try m.device.buffer(st.captureBytes(st.prompt_rows), mtl.ResourceOptions.shared);

    var bb = [_]base_ref.BaselineBuffers{try base_ref.BaselineBuffers.init(m.device)};
    defer bb[0].deinit();

    var prompt: [68]u32 = undefined;
    fillTokens(&prompt, 9);
    var win_toks: [16]u32 = undefined;
    fillTokens(&win_toks, 21);
    const cont_toks = [_]u32{ 313, 617 };

    const pending_a = try m.device.buffer(st.taps_bytes, mtl.ResourceOptions.shared);
    defer pending_a.deinit();
    const pending_b = try m.device.buffer(st.taps_bytes, mtl.ResourceOptions.shared);
    defer pending_b.deinit();

    const Case = struct { win: usize, acc: usize };
    for ([_]Case{
        .{ .win = 1, .acc = 1 },
        .{ .win = 1, .acc = 0 },
        .{ .win = 2, .acc = 2 },
        .{ .win = 2, .acc = 1 },
        .{ .win = 2, .acc = 0 },
        .{ .win = 7, .acc = 7 },
        .{ .win = 7, .acc = 2 },
        .{ .win = 7, .acc = 0 },
        .{ .win = 15, .acc = 15 },
        .{ .win = 15, .acc = 2 },
        .{ .win = 15, .acc = 0 },
        .{ .win = 16, .acc = 0 },
        .{ .win = 8, .acc = 8 },
        .{ .win = 8, .acc = 1 },
        .{ .win = 16, .acc = 16 },
        .{ .win = 16, .acc = 3 },
        .{ .win = 16, .acc = 7 },
        .{ .win = 8, .acc = 0 },
    }) |case| {
        var c_base = try st.Cache.init(m.device, 128, &m.weight_plan);
        defer c_base.deinit();
        var c_v5 = try st.Cache.init(m.device, 128, &m.weight_plan);
        defer c_v5.deinit();

        try base_ref.runBaselineSerial(m, &s_base, &bb, &.{.{ .cache = &c_base, .rows = prompt.len }}, &prompt, .{ .all = true, .taps = .{ .buf = s_base.taps.? } });
        c_base.len += prompt.len;
        try fwd.run(m, &s_v5, &c_v5, &prompt, .{ .all = true, .taps = .{ .buf = s_v5.taps.? } });
        c_v5.len += prompt.len;

        try base_ref.baselineBegin(&c_base, m.queue, &bb[0], case.win);
        try base_ref.runBaselineSerial(m, &s_base, &bb, &.{.{ .cache = &c_base, .rows = case.win }}, win_toks[0..case.win], .{ .all = true, .taps = .{ .buf = s_base.taps.? } });

        try histories.poison(m.queue, c_v5.taps);
        try c_v5.begin(case.win);
        try fwd.runShared(m, &s_v5, &.{.{ .cache = &c_v5, .rows = case.win }}, win_toks[0..case.win], .{ .all = true, .taps = .{ .buf = s_v5.taps.? } });
        try std.testing.expectEqualSlices(u8, s_base.logits_all.slice(u8, case.win * vocab * 4), s_v5.logits_all.slice(u8, case.win * vocab * 4));
        try std.testing.expectEqualSlices(u8, s_base.taps.?.slice(u8, st.captureBytes(case.win)), s_v5.taps.?.slice(u8, st.captureBytes(case.win)));

        try histories.expectPending(m.queue, bb[0].taps_3plus, c_v5.taps, pending_a, pending_b, case.win);
        if (case.acc == 0) {
            try base_ref.baselineRollback(&c_base, m.queue, &bb[0]);
            try c_v5.rollback();
        } else {
            try base_ref.baselineCommit(&c_base, m.queue, &bb[0], case.acc);
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = m.queue.commandBuffer();
            const e = cb.compute(.concurrent);
            fwd_ops.commitHistories(m, e, &[_]fwd.Segment{.{ .cache = &c_v5, .rows = case.win }}, &.{case.acc});
            e.end();
            cb.commit();
            cb.wait();
            try std.testing.expect(cb.failure() == null);
            try c_v5.commit(case.acc);
        }
        try readPrivateBuffer(m.queue, c_base.conv, stage_a, st.conv_bytes);
        try readPrivateBuffer(m.queue, c_v5.conv, stage_b, st.conv_bytes);
        try histories.expectCommitted(stage_a, stage_b);

        try base_ref.runBaselineSerial(m, &s_base, &bb, &.{.{ .cache = &c_base, .rows = cont_toks.len }}, &cont_toks, .{ .all = true, .taps = .{ .buf = s_base.taps.? } });
        c_base.len += cont_toks.len;
        try fwd.run(m, &s_v5, &c_v5, &cont_toks, .{ .all = true, .taps = .{ .buf = s_v5.taps.? } });
        c_v5.len += cont_toks.len;

        try std.testing.expectEqualSlices(u8, s_base.logits_all.slice(u8, cont_toks.len * vocab * 4), s_v5.logits_all.slice(u8, cont_toks.len * vocab * 4));
        try expectConvAndTapsEqual(m.queue, c_base.conv, c_v5.conv, stage_a, stage_b, s_base.taps.?, s_v5.taps.?, cont_toks.len);
    }
}

fn checkFourStreamsAndResumed(m: *model_.Model, stage_a: mtl.Buffer, stage_b: mtl.Buffer) !void {
    const vocab = m.config.text.unpadded_vocab;
    const target = try backend_.Metal.init(gpa, m, .{ .capacity = 256, .streams = 4, .budget = 16 * st.cacheBytes(256) });
    defer target.deinit();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps_ids);

    var s_base = try st.Scratch.init(m.device, st.prompt_rows);
    defer s_base.deinit();
    s_base.taps = try m.device.buffer(st.captureBytes(st.prompt_rows), mtl.ResourceOptions.shared);
    var bb = [_]base_ref.BaselineBuffers{try base_ref.BaselineBuffers.init(m.device)};
    defer bb[0].deinit();

    var p0: [64]u32 = undefined;
    fillTokens(&p0, 31);
    var p1: [68]u32 = undefined;
    fillTokens(&p1, 41);
    var p2: [70]u32 = undefined;
    fillTokens(&p2, 51);
    var p3: [72]u32 = undefined;
    fillTokens(&p3, 61);

    var snap_owner = try lanes.Stream.init(gpa, .{ .id = "snap_src", .prompt = &p0, .max_new = 32 });
    defer snap_owner.deinit(gpa);
    try b.prefill(&snap_owner);
    defer b.release(&snap_owner);
    const snap = try backend_.Metal.snapSave(target, &snap_owner, 64);
    defer backend_.Metal.snapDrop(target, snap);

    var p0_ext: [67]u32 = undefined;
    @memcpy(p0_ext[0..64], &p0);
    p0_ext[64..67].* = .{ 501, 502, 503 };
    const prompts = [4][]const u32{ &p0_ext, &p1, &p2, &p3 };

    var solo_caches: [4]st.Cache = undefined;
    var made: usize = 0;
    defer for (solo_caches[0..made]) |*c| c.deinit();
    for (0..4) |i| {
        solo_caches[i] = try st.Cache.init(m.device, 256, &m.weight_plan);
        made += 1;
        try base_ref.runBaselineSerial(m, &s_base, &bb, &.{.{ .cache = &solo_caches[i], .rows = prompts[i].len }}, prompts[i], .{ .all = true, .taps = .{ .buf = s_base.taps.? } });
        solo_caches[i].len += prompts[i].len;
    }

    var streams: [4]lanes.Stream = undefined;
    var opened: usize = 0;
    defer for (streams[0..opened]) |*sm| {
        b.release(sm);
        sm.deinit(gpa);
    };
    for (0..4) |i| {
        const reuse: lanes.stream.Reuse = if (i == 0) .{ .saved = snap, .at = 64 } else .{};
        streams[i] = try lanes.Stream.init(gpa, .{ .id = "v53_s", .prompt = prompts[i], .max_new = 32, .reuse = reuse });
        opened += 1;
        try b.prefill(&streams[i]);
    }
    try std.testing.expectEqual(@as(u32, 64), streams[0].cached);

    var drafts_buf: [8]u32 = undefined;
    fillTokens(&drafts_buf, 99);
    const win_specs = [_]struct { rows: usize, acc: usize }{
        .{ .rows = 2, .acc = 1 },
        .{ .rows = 4, .acc = 4 },
        .{ .rows = 8, .acc = 3 },
    };
    for (win_specs) |ws| {
        var wins: [4]lanes.backend.Window = undefined;
        var outs: [4]lanes.backend.Verified = undefined;
        var samp_mem: [4][8]u32 = undefined;
        var echo_mem: [4][7]u32 = undefined;
        var pos_mem: [4][8]u64 = undefined;
        var paths_mem: [4][8]u32 = undefined;
        var paths: [4][]const u32 = undefined;
        var base_logits: [4][]u8 = undefined;
        defer for (0..4) |i| gpa.free(base_logits[i]);
        var base_taps: [4][]u8 = undefined;
        defer for (0..4) |i| gpa.free(base_taps[i]);

        for (0..4) |i| {
            const c_solo = &solo_caches[i];
            const start = c_solo.len;
            const pend: u32 = @intCast(700 + i * 17 + start);
            var wids: [8]u32 = undefined;
            wids[0] = pend;
            @memcpy(wids[1..ws.rows], drafts_buf[0 .. ws.rows - 1]);
            for (0..ws.rows) |r| pos_mem[i][r] = start + r + 1;
            for (0..ws.acc) |r| paths_mem[i][r] = @intCast(r);
            paths[i] = paths_mem[i][0..ws.acc];

            try base_ref.baselineBegin(c_solo, m.queue, &bb[0], ws.rows);
            try base_ref.runBaselineSerial(m, &s_base, &bb, &.{.{ .cache = c_solo, .rows = ws.rows }}, wids[0..ws.rows], .{ .all = true, .taps = .{ .buf = s_base.taps.? } });
            base_logits[i] = try gpa.dupe(u8, s_base.logits_all.slice(u8, ws.rows * vocab * 4));
            base_taps[i] = try gpa.dupe(u8, s_base.taps.?.slice(u8, st.captureBytes(ws.rows)));
            try base_ref.baselineCommit(c_solo, m.queue, &bb[0], ws.acc);

            wins[i] = .{ .stream = &streams[i], .pending = pend, .held = 0, .tokens = drafts_buf[0 .. ws.rows - 1], .parents = null, .positions = pos_mem[i][0..ws.rows] };
            outs[i] = .{ .sampled = samp_mem[i][0..ws.rows], .drafts = echo_mem[i][0 .. ws.rows - 1] };
        }

        try b.verify(&wins, &outs);
        var base: usize = 0;
        for (0..4) |i| {
            const c_v5 = target.caches.get(&streams[i]).?;
            const got_l = target.scratch.logits_all.slice(u8, 4 * ws.rows * vocab * 4)[base * vocab * 4 ..][0 .. ws.rows * vocab * 4];
            try std.testing.expectEqualSlices(u8, base_logits[i], got_l);
            const f = try b.features(&streams[i], &taps_ids, c_v5.len, @intCast(ws.rows));
            const tap_buf: mtl.Buffer = .{ .id = @ptrFromInt(f.buffer) };
            const got_t = tap_buf.slice(u8, @intCast(f.offset + st.captureBytes(ws.rows)))[@intCast(f.offset)..][0..st.captureBytes(ws.rows)];
            try std.testing.expectEqualSlices(u8, base_taps[i], got_t);
            base += ws.rows;
        }
        // Issue async keep and verify commit_cb is pending before settling.
        try b.keep(&wins, &paths);
        try std.testing.expect(target.commit_cb != null);
        for (0..4) |i| {
            const c_v5 = target.caches.get(&streams[i]).?;
            const accepted = paths[i].len;
            _ = try b.features(&streams[i], &taps_ids, c_v5.len - 1, 1);
            if (accepted < ws.rows) {
                try std.testing.expectError(error.Inkling2TapsNotHeld, b.features(&streams[i], &taps_ids, c_v5.len, 1));
            }
        }
        if (ws.rows != 8) {
            try target.settleCommit();
            for (0..4) |i| {
                const c_v5 = target.caches.get(&streams[i]).?;
                try readPrivateBuffer(m.queue, solo_caches[i].conv, stage_a, st.conv_bytes);
                try readPrivateBuffer(m.queue, c_v5.conv, stage_b, st.conv_bytes);
                try std.testing.expectEqualSlices(u8, stage_a.slice(u8, st.conv_bytes), stage_b.slice(u8, st.conv_bytes));
            }
        }
    }
    // With the final ws.rows=8 async keep still in flight, queueFn must settle commit_cb before reading conv.
    try std.testing.expect(target.commit_cb != null);
    const c0 = target.caches.get(&streams[0]).?;
    const pos0 = c0.len + 1;
    const next_tok = try b.read(try b.queue(&streams[0], .{ .value = 809 }, pos0));
    try std.testing.expect(target.commit_cb == null);
    try std.testing.expect(next_tok < vocab);
    try base_ref.runBaselineSerial(m, &s_base, &bb, &.{.{ .cache = &solo_caches[0], .rows = 1 }}, &.{809}, .{ .all = true, .taps = .{ .buf = s_base.taps.? } });
    solo_caches[0].len += 1;
    try std.testing.expectEqualSlices(u8, s_base.logits_all.slice(u8, vocab * 4), c0.logits.slice(u8, vocab * 4));
    try readPrivateBuffer(m.queue, solo_caches[0].conv, stage_a, st.conv_bytes);
    try readPrivateBuffer(m.queue, c0.conv, stage_b, st.conv_bytes);
    try std.testing.expectEqualSlices(u8, stage_a.slice(u8, st.conv_bytes), stage_b.slice(u8, st.conv_bytes));
    var final_snap = try snapshot.save(m.device, m.queue, c0);
    defer final_snap.deinit();
    try std.testing.expectEqual(solo_caches[0].len, final_snap.at);
}

fn expectStreamPoisoned(m: *model_.Model, target: *backend_.Metal, b: lanes.backend.Backend, sm: *lanes.Stream, cache: *st.Cache) !void {
    const len = cache.len;
    try std.testing.expect(cache.commit_failed);
    try std.testing.expect(cache.tap.buf == null);
    try std.testing.expectError(error.GpuFailed, b.queue(sm, .{ .value = 55 }, len + 1));
    var samp = [_]u32{ 0, 0 };
    var dr = [_]u32{0};
    const pos = [_]u64{ len + 1, len + 2 };
    const win = [_]lanes.backend.Window{.{ .stream = sm, .pending = 55, .held = 0, .tokens = &.{56}, .parents = null, .positions = &pos }};
    var out = [_]lanes.backend.Verified{.{ .sampled = &samp, .drafts = &dr }};
    try std.testing.expectError(error.GpuFailed, b.verify(&win, &out));
    try std.testing.expectError(error.GpuFailed, b.keep(&win, &.{&.{0}}));
    try std.testing.expectError(error.GpuFailed, b.features(sm, &taps_ids, len - 1, 1));
    try std.testing.expectError(error.GpuFailed, backend_.Metal.snapSave(target, sm, @intCast(len)));
    try std.testing.expectError(error.GpuFailed, snapshot.save(m.device, m.queue, cache));
    try std.testing.expectError(error.GpuFailed, b.first(sm, len));
    try std.testing.expectError(error.GpuFailed, b.firstRow(sm));
    try std.testing.expectError(error.GpuFailed, b.prefill(sm));
    try std.testing.expectError(error.GpuFailed, target.feed(cache, &.{55}, null, null));
    try std.testing.expectError(error.GpuFailed, fwd.run(m, &target.scratch, cache, &.{55}, .{}));
}

fn checkAsyncCommitFailure(m: *model_.Model) !void {
    const target = try backend_.Metal.init(gpa, m, .{ .capacity = 128, .streams = 4, .budget = 16 * st.cacheBytes(128) });
    defer target.deinit();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &taps_ids);

    var p0: [8]u32 = undefined;
    fillTokens(&p0, 101);
    var p1: [8]u32 = undefined;
    fillTokens(&p1, 102);
    var p2: [8]u32 = undefined;
    fillTokens(&p2, 103);
    var p3: [8]u32 = undefined;
    fillTokens(&p3, 104);

    // Case 1: async keep failure observed by a different stream's prefill poisons every cache in that batch.
    {
        var s0 = try lanes.Stream.init(gpa, .{ .id = "ok_peer", .prompt = &p0, .max_new = 32 });
        defer {
            b.release(&s0);
            s0.deinit(gpa);
        }
        var s1 = try lanes.Stream.init(gpa, .{ .id = "fail_a", .prompt = &p1, .max_new = 32 });
        defer {
            b.release(&s1);
            s1.deinit(gpa);
        }
        var s2 = try lanes.Stream.init(gpa, .{ .id = "fail_b", .prompt = &p2, .max_new = 32 });
        defer {
            b.release(&s2);
            s2.deinit(gpa);
        }
        var s3 = try lanes.Stream.init(gpa, .{ .id = "observer_prefill", .prompt = &p3, .max_new = 32 });
        defer {
            b.release(&s3);
            s3.deinit(gpa);
        }

        try b.prefill(&s0);
        try b.prefill(&s1);
        try b.prefill(&s2);
        const c0 = target.caches.get(&s0).?;
        const c1 = target.caches.get(&s1).?;
        const c2 = target.caches.get(&s2).?;

        var samp1 = [_]u32{ 0, 0, 0 };
        var samp2 = [_]u32{ 0, 0, 0 };
        var dr1 = [_]u32{ 0, 0 };
        var dr2 = [_]u32{ 0, 0 };
        const pos1 = [_]u64{ c1.len + 1, c1.len + 2, c1.len + 3 };
        const pos2 = [_]u64{ c2.len + 1, c2.len + 2, c2.len + 3 };
        const wins = [_]lanes.backend.Window{
            .{ .stream = &s1, .pending = 201, .held = 0, .tokens = &.{ 202, 203 }, .parents = null, .positions = &pos1 },
            .{ .stream = &s2, .pending = 301, .held = 0, .tokens = &.{ 302, 303 }, .parents = null, .positions = &pos2 },
        };
        var outs = [_]lanes.backend.Verified{
            .{ .sampled = &samp1, .drafts = &dr1 },
            .{ .sampled = &samp2, .drafts = &dr2 },
        };
        try b.verify(&wins, &outs);
        try b.keep(&wins, &.{ &.{ 0, 1 }, &.{ 0, 1, 2 } });
        try std.testing.expect(target.commit_cb != null);
        try std.testing.expect(target.commit_active_seq != 0);
        try std.testing.expectEqual(target.commit_active_seq, c1.commit_seq);
        try std.testing.expectEqual(target.commit_active_seq, c2.commit_seq);
        try std.testing.expect(c0.commit_seq != target.commit_active_seq);

        target.test_fail_commit = true;
        try std.testing.expectError(error.GpuFailed, b.prefill(&s3));
        try std.testing.expect(target.commit_cb == null);
        try std.testing.expectEqual(@as(u64, 0), target.commit_active_seq);
        try std.testing.expect(!target.caches.contains(&s3));

        try expectStreamPoisoned(m, target, b, &s1, c1);
        try expectStreamPoisoned(m, target, b, &s2, c2);

        // Mixed verify with healthy s0 and poisoned s1 fails closed without leaving s0 open.
        var s0_samp = [_]u32{0};
        var s1_samp = [_]u32{0};
        var empty_dr: [0]u32 = .{};
        const mix = [_]lanes.backend.Window{
            .{ .stream = &s0, .pending = 401, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{c0.len + 1} },
            .{ .stream = &s1, .pending = 402, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{c1.len + 1} },
        };
        var mix_out = [_]lanes.backend.Verified{
            .{ .sampled = &s0_samp, .drafts = &empty_dr },
            .{ .sampled = &s1_samp, .drafts = &empty_dr },
        };
        try std.testing.expectError(error.GpuFailed, b.verify(&mix, &mix_out));
        try std.testing.expectEqual(@as(?usize, null), c0.pending);

        // Unrelated healthy stream s0 still decodes and snapshots cleanly.
        try std.testing.expect(!c0.commit_failed);
        _ = try b.read(try b.queue(&s0, .{ .value = 401 }, c0.len + 1));
        const ok_snap = try backend_.Metal.snapSave(target, &s0, @intCast(c0.len));
        backend_.Metal.snapDrop(target, ok_snap);
    }

    // Case 2: async keep failure observed during releaseFn still poisons remaining batch peers.
    {
        var r1 = try lanes.Stream.init(gpa, .{ .id = "rel_a", .prompt = &p1, .max_new = 32 });
        defer {
            b.release(&r1);
            r1.deinit(gpa);
        }
        var r2 = try lanes.Stream.init(gpa, .{ .id = "rel_b", .prompt = &p2, .max_new = 32 });
        defer {
            b.release(&r2);
            r2.deinit(gpa);
        }
        try b.prefill(&r1);
        try b.prefill(&r2);
        const c1 = target.caches.get(&r1).?;
        const c2 = target.caches.get(&r2).?;

        var samp1 = [_]u32{ 0, 0 };
        var samp2 = [_]u32{ 0, 0 };
        var dr1 = [_]u32{0};
        var dr2 = [_]u32{0};
        const pos1 = [_]u64{ c1.len + 1, c1.len + 2 };
        const pos2 = [_]u64{ c2.len + 1, c2.len + 2 };
        const wins = [_]lanes.backend.Window{
            .{ .stream = &r1, .pending = 501, .held = 0, .tokens = &.{502}, .parents = null, .positions = &pos1 },
            .{ .stream = &r2, .pending = 601, .held = 0, .tokens = &.{602}, .parents = null, .positions = &pos2 },
        };
        var outs = [_]lanes.backend.Verified{
            .{ .sampled = &samp1, .drafts = &dr1 },
            .{ .sampled = &samp2, .drafts = &dr2 },
        };
        try b.verify(&wins, &outs);
        try b.keep(&wins, &.{ &.{ 0, 1 }, &.{0} });
        try std.testing.expect(target.commit_cb != null);

        target.test_fail_commit = true;
        b.release(&r1);
        try std.testing.expect(target.commit_cb == null);
        try expectStreamPoisoned(m, target, b, &r2, c2);
    }
}

test "V5-3 matches serial/backup baseline raw logits, taps, and conv histories across rows 1..256, 4 streams, resumed, and partial-reject" {
    const dev = mtl.Device.init() catch return error.SkipZigTest;
    dev.deinit();
    const m = try synth.buildSyntheticModel(gpa);
    defer m.deinit();
    const stage_a = try m.device.buffer(st.conv_bytes, mtl.ResourceOptions.shared);
    defer stage_a.deinit();
    const stage_b = try m.device.buffer(st.conv_bytes, mtl.ResourceOptions.shared);
    defer stage_b.deinit();
    try checkRowWidths(m, stage_a, stage_b);
    try checkAcceptRejectAndRollback(m, stage_a, stage_b);
    try checkFourStreamsAndResumed(m, stage_a, stage_b);
    try checkAsyncCommitFailure(m);
}
