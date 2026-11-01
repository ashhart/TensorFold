//! The target's tap capture on the real weights: the same normalized and logit bytes, the taps where they were sent.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const c = q.config;
const st = q.state;

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
    var both: [st.batch_rows]u32 = undefined;
    @memcpy(both[0..n], ids);
    @memcpy(both[n..][0..n], ids);
    try q.forward.run(m, scratch, &.{ .{ .cache = &a, .rows = n, .tap = .{ .stage = place } }, .{ .cache = &b, .rows = n, .tap = .none } }, both[0 .. 2 * n], true, .all);
    if (!std.mem.eql(u16, want, got[row_words..][0 .. n * row_words]) or !untouched(got[0..row_words]) or !untouched(got[(n + 1) * row_words ..])) return error.SharedStagedTapsDiffer;
}

/// Untapped target outputs versus the capture, including duplicate final taps and every admitted column.
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
