//! Real-device controls for tap capture and physical hold grouping.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const c = q.config;
const st = q.state;
/// 3x9 + 5x1 physical passes: same-kernel reference and synthetic grid controls, not a speed measurement.
pub fn splitHold(gpa: std.mem.Allocator, io: std.Io, d: *q.dspark.Drafter, ids: []const u32) !void {
    const depths = [8]u32{ 9, 1, 1, 1, 1, 1, 1, 1 };
    const groups = [2][2]usize{ .{ 0, 3 }, .{ 3, 8 } }; // the lanes each physical pass holds
    const real_grid = d.holds;
    const real_len = d.holds_len;
    const step_ms = d.step_ms;
    const hold_ms = d.hold_ms;
    const spent_ms = d.hold_spent_ms;
    defer {
        d.holds = real_grid;
        d.holds_len = real_len;
        d.step_ms = step_ms;
        d.hold_ms = hold_ms;
        d.hold_spent_ms = spent_ms;
    }
    var pending: [8]u32 = undefined;
    for (&pending, 0..) |*p, i| p.* = ids[(i * 131 + 5) % ids.len];
    const drafter = d.drafter();
    var streams: [8]tf.lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    while (made < streams.len) : (made += 1) streams[made] = try tf.lanes.Stream.init(gpa, .{ .id = "split", .prompt = &.{1}, .max_new = 1 });
    var opened: usize = 0;
    defer for (streams[0..opened]) |*s| drafter.release(s);
    while (opened < streams.len) : (opened += 1) {
        try drafter.open(&streams[opened]);
        if ((d.lanes.get(&streams[opened]) orelse return error.LaneNotOpened).cap != 64 + d.config.block) return error.LaneNotAtCapacity64;
    }
    // Reference groups use the same kernels on separate lanes.
    var refs: [8]*q.dspark.Lane = undefined;
    var ref_n: usize = 0;
    defer for (refs[0..ref_n]) |l| d.dropLane(l);
    while (ref_n < refs.len) : (ref_n += 1) refs[ref_n] = try d.newLane(64);
    var blocks: [8]q.dspark_block.Block = undefined;
    for (&blocks, refs, 0..) |*b, l, i| b.* = .{ .lane = l, .pending = pending[i], .anchor = 0, .depth = depths[i] };
    for (groups) |g| _ = try q.dspark_block.hold(d, blocks[g[0]..g[1]], null);
    for (d.holds[0..d.holds_len]) |*h| h.ms = 1e4 * (1 + 0.25 * @as(f64, @floatFromInt(h.streams))) * (1 + 0.5 * @as(f64, @floatFromInt(h.depth)));
    const synthetic = d.holds;
    _ = q.dspark_block.takeHoldMs(d);
    var items: [8]tf.lanes.drafter.Hold = undefined;
    for (&items, 0..) |*it, i| it.* = .{ .stream = &streams[i], .pending = pending[i], .position = 1, .depth = depths[i] };
    const t0 = std.Io.Clock.awake.now(io);
    try drafter.hold(&items);
    const wall = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    // nothing pending (one 8-block signalled pass would be), and no lane points into a signalled pass
    if (d.pending != null or d.pass_len != 0) return error.SplitHoldLeftAPassPending;
    for (&streams) |*s| if (d.lanes.get(s).?.slot != null) return error.SplitHoldLaneInASignalledPass;
    const taken = q.dspark_block.takeHoldMs(d);
    if (!(taken > 0) or q.dspark_block.takeHoldMs(d) != 0) return error.SplitHoldTimeNotTakenOnce;
    if (taken > wall) return error.SplitHoldTimeExceedsWall;
    // each lane's picks: its depth of them, the same as the reference pass of its group
    for (&streams, refs, depths, 0..) |*s, ref, depth, i| {
        var got: [16]u32 = undefined;
        try drafter.held(&.{s}, &.{got[0..depth]});
        if (!std.mem.eql(u32, got[0..depth], ref.held[0..depth])) return error.SplitHoldPicksDiffer;
        if (depth == 1 and i > 0) if (drafter.held(&.{s}, &.{got[0 .. depth + 1]})) |_| return error.SplitHoldHeldMoreThanAsked else |err| if (err != error.DraftHeldFewer) return err;
    }
    // the grid: participating shapes stepped down by about a tenth, the controls exactly as set
    if (d.holds_len != real_len) return error.GridResized;
    for (synthetic[0..real_len], d.holds[0..real_len]) |before, now| {
        const ratio = now.ms / before.ms;
        const ok = switch (before.streams) {
            2, 8 => ratio > 0.9 and ratio < 0.91, // one group's step
            4 => ratio > 0.81 and ratio < 0.83, // both groups' steps
            else => now.ms == before.ms, // 1 and 16 blocks: in neither pass
        };
        if (!ok) return error.SplitHoldGridStepsWrong;
    }
    std.debug.print("split hold, 8 lanes at depths [9, 1 x 7]: passes of 3x9 and 5x1 rows, both synchronous; {d:.2} ms GPU taken once ({d:.2} ms wall); picks equal the reference groups'; 2-, 8- and 4-block passes repriced once, once and twice, 1- and 16-block passes untouched\n", .{ taken, wall });
}
fn untouched(words: []const u16) bool {
    for (words) |v| if (v != 0x7fc0) return false;
    return true;
}

/// The forward capturing into a stage, into none, and beside a stream with none.
fn stageTaps(gpa: std.mem.Allocator, m: *q.Model, scratch: *st.Scratch, ids: []const u32, record: bool, taps: st.Taps, x: []const u8, logits: []const u8) !void {
    const n = ids.len;
    const row_words = taps.count * c.hidden;
    const want = try gpa.dupe(u16, taps.buf.slice(u16, n * row_words));
    defer gpa.free(want);
    const stage = try m.device.buffer((n + 2) * row_words * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer stage.deinit();
    const got = stage.slice(u16, (n + 2) * row_words);
    const scratch_taps = taps.buf.slice(u16, n * row_words);
    const place: q.forward.Place = .{ .buf = stage, .off = row_words * 2 };
    for ([_]q.forward.Tap{ .{ .stage = place }, .none }, 0..) |tap, mode| {
        @memset(got, 0x7fc0);
        @memset(scratch_taps, 0x7fc0);
        var cache = try st.Cache.init(gpa, m.device, &m.config.attention, n + 16);
        defer cache.deinit();
        try q.forward.run(m, scratch, &.{.{ .cache = &cache, .rows = n, .tap = tap }}, ids, record, .last);
        if (!std.mem.eql(u8, x, scratch.x.contents()[0..x.len]) or !std.mem.eql(u8, logits, scratch.logits.contents()[0..logits.len])) return error.TargetTapOutputMismatch;
        if (!untouched(scratch_taps)) return error.StagedForwardWroteScratchTaps;
        if (mode == 1) {
            if (!untouched(got)) return error.UnboundForwardWroteStage;
            continue;
        }
        if (!std.mem.eql(u16, want, got[row_words..][0 .. n * row_words]) or !untouched(got[0..row_words]) or !untouched(got[(n + 1) * row_words ..])) return error.StagedTapsDiffer;
    }
    if (2 * n > st.batch_rows or 2 * n > scratch.rows) return;
    // two streams in one recorded call: the first captured into the stage, the second nowhere
    @memset(got, 0x7fc0);
    var a = try st.Cache.init(gpa, m.device, &m.config.attention, n + 16);
    defer a.deinit();
    var b = try st.Cache.init(gpa, m.device, &m.config.attention, n + 16);
    defer b.deinit();
    var both: [32]u32 = undefined;
    @memcpy(both[0..n], ids);
    @memcpy(both[n..][0..n], ids);
    try q.forward.run(m, scratch, &.{ .{ .cache = &a, .rows = n, .tap = .{ .stage = place } }, .{ .cache = &b, .rows = n, .tap = .none } }, both[0 .. 2 * n], true, .all);
    if (!std.mem.eql(u16, want, got[row_words..][0 .. n * row_words]) or !untouched(got[0..row_words]) or !untouched(got[(n + 1) * row_words ..])) return error.SharedStagedTapsDiffer;
}

/// Untapped target outputs versus capture/restore, including duplicate final taps and every admitted column.
pub fn targetTaps(gpa: std.mem.Allocator, m: *q.Model, scratch: *st.Scratch, ids: []const u32) !void {
    const original = scratch.taps.?;
    defer scratch.taps = original;
    for ([_]usize{ 1, 4, 16, ids.len }, 0..) |n, shape| {
        if (n > ids.len or (shape == 3 and (n == 1 or n == 4 or n == 16))) continue;
        scratch.taps = null;
        var plain = try st.Cache.init(gpa, m.device, &m.config.attention, n + 16);
        defer plain.deinit();
        const record = n <= st.batch_rows;
        try q.forward.run(m, scratch, &.{.{ .cache = &plain, .rows = n }}, ids[0..n], record, .last);
        const x = try gpa.dupe(u8, scratch.x.contents()[0 .. n * c.hidden * 2]);
        defer gpa.free(x);
        const logits = try gpa.dupe(u8, scratch.logits.contents()[0 .. c.vocab * 2]);
        defer gpa.free(logits);
        for (0..3) |layout| {
            var taps = original;
            if (layout == 1) {
                taps.count = 3;
                taps.layers[0..3].* = .{ 0, c.layers - 1, c.layers - 1 };
            } else if (layout == 2) {
                taps.count = st.max_taps;
                for (0..st.max_taps) |i| taps.layers[i] = @intCast(if (i >= st.max_taps - 2) c.layers - 1 else i);
            }
            taps.table_at = n * taps.count * c.hidden * 2;
            taps.buf = try m.device.buffer(taps.table_at + scratch.rows * @sizeOf(u64), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            defer {
                scratch.taps = null;
                taps.buf.deinit();
            }
            scratch.taps = taps;
            var captured = try st.Cache.init(gpa, m.device, &m.config.attention, n + 16);
            defer captured.deinit();
            try q.forward.run(m, scratch, &.{.{ .cache = &captured, .rows = n }}, ids[0..n], record, .last);
            if (!std.mem.eql(u8, x, scratch.x.contents()[0..x.len]) or !std.mem.eql(u8, logits, scratch.logits.contents()[0..logits.len])) return error.TargetTapOutputMismatch;
            const data = taps.buf.slice(u16, n * taps.count * c.hidden);
            for (taps.layers[0..taps.count], 0..) |layer, i| for (taps.layers[0..i], 0..) |other, j| {
                if (layer != other) continue;
                for (0..n) |row| if (!std.mem.eql(u16, data[(row * taps.count + i) * c.hidden ..][0..c.hidden], data[(row * taps.count + j) * c.hidden ..][0..c.hidden])) return error.RepeatedTapMismatch;
            };
            try stageTaps(gpa, m, scratch, ids[0..n], record, taps, x, logits);
        }
        std.debug.print("target taps: {d} rows, released/duplicate/max-column layouts preserve all normalized and logit bytes\n", .{n});
    }
}
