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

/// Planted W1/W2 drive native scan/pick failures through waited, event-waited target and host-held consumers.
pub fn nonfinite(gpa: std.mem.Allocator, d: *q.dspark.Drafter) !void {
    try d.settle();
    const m = d.target;
    const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const rank: usize = d.config.rank;
    const bf16 = d.w.w2 == .bf16;
    const old_w1 = d.w.w1;
    const old_w2 = d.w.w2;
    const grid = d.holds;
    const grid_len = d.holds_len;
    const step_ms = d.step_ms;
    const hold_ms = d.hold_ms;
    const spent_ms = d.hold_spent_ms;
    defer {
        d.w.w1 = old_w1;
        d.w.w2 = old_w2;
        d.holds = grid;
        d.holds_len = grid_len;
        d.step_ms = step_ms;
        d.hold_ms = hold_ms;
        d.hold_spent_ms = spent_ms;
    }
    // Depth one reads only W1[0]; all its factors are one, so an isolated infinity cannot become 0*Inf.
    const w1 = try m.device.buffer(rank * 2, shared);
    defer w1.deinit();
    @memset(w1.slice(u16, rank), 0x3f80);
    const w2 = try m.device.buffer(c.vocab * rank * @as(usize, if (bf16) 2 else 1), shared);
    defer w2.deinit();
    @memset(w2.contents()[0..w2.length()], 0);
    const groups = c.vocab * rank / 64;
    const affine: ?mtl.Buffer = if (bf16) null else try m.device.buffer(groups * 4, shared);
    defer if (affine) |buf| buf.deinit();
    if (affine) |buf| @memset(buf.contents()[0..buf.length()], 0);
    d.w.w1 = w1;
    d.w.w2 = if (bf16) .{ .bf16 = .{ .w = w2, .n = c.vocab, .k = rank } } else .{ .codes = .{
        .w = w2,
        .scales = affine.?,
        .biases = affine.?,
        .b_off = groups * 2,
        .n = c.vocab,
        .k = rank,
        .bits = 8,
    } };
    var scratch = try st.Scratch.init(gpa, m.device, &m.config.attention, 1);
    defer scratch.deinit();
    const dr = d.drafter();
    var failures: usize = 0;
    // The final finite case also proves that neither the bad word nor the held flag survives the next pass.
    for ([_]u16{ 0, 0x7f80, 0xff80, 0x7fc0, 0 }) |bits| {
        if (bf16) w2.slice(u16, c.vocab * rank)[512 * rank] = bits else affine.?.slice(u16, groups * 2)[groups + 512 * rank / 64] = bits;
        const invalid = bits != 0;
        var stream = try tf.lanes.Stream.init(gpa, .{ .id = "nonfinite", .prompt = &.{0}, .max_new = 1 });
        defer stream.deinit(gpa);
        try dr.open(&stream);
        defer dr.release(&stream);
        const lane = d.lanes.get(&stream) orelse return error.LaneNotOpened;
        const waited_bad = if (q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = 0, .anchor = 0, .depth = 1 }}, null)) |_| false else |err| blk: {
            if (err != error.DraftNonfinite) return err;
            break :blk true;
        };
        // The only nonfinite input is W2's planted row, not a broken target/backbone fixture.
        for (d.s.logits.slice(u16, c.vocab)) |v| if (v & 0x7f80 == 0x7f80) return error.NonfiniteControlBaseLogit;
        const expected = if (!waited_bad) lane.held[0] else 0;
        try dr.hold(&.{.{ .stream = &stream, .pending = 0, .position = 1, .depth = 1 }});
        const held = q.dspark.Drafter.device(d, &stream) orelse return error.NonfiniteControlNotDeviceHeld;
        var cache = try st.Cache.init(gpa, m.device, &m.config.attention, 16);
        defer cache.deinit();
        const target_bad = if (q.forward.runHeld(m, &scratch, &.{.{ .cache = &cache, .rows = 1, .tap = .none }}, &.{0}, false, .last, &.{.{ .row = 0, .src = held.src, .off = held.off, .stride = held.stride, .count = 1 }}, .{ .event = held.wait.event, .value = held.wait.value })) |_| false else |err| blk: {
            if (err != error.DraftNonfinite) return err;
            break :blk true;
        };
        var got: [1]u32 = undefined;
        const held_bad = if (dr.held(&.{&stream}, &.{&got})) |_| false else |err| blk: {
            if (err != error.DraftNonfinite) return err;
            break :blk true;
        };
        if (target_bad and cache.len != 0) return error.NonfiniteControlTargetCommitted;
        if (!invalid and !waited_bad and !held_bad and (got[0] != expected or scratch.ids.slice(u32, 1)[0] != expected)) return error.NonfiniteControlFinitePickChanged;
        const ok = waited_bad == invalid and target_bad == invalid and held_bad == invalid;
        failures += @intFromBool(!ok);
        std.debug.print("{s} Markov producer {s} bits 0x{x}: waited refused {}, held target refused {}, held host refused {} (expected {})\n", .{ if (ok) "PASS" else "FAIL", if (bf16) "bf16" else "8bit", bits, waited_bad, target_bad, held_bad, invalid });
    }
    if (failures != 0) return error.MarkovNonfiniteControlFailed;
}
