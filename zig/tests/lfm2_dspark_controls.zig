//! Real-device controls for tap capture and physical hold grouping.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm2_moe;
const c = q.config;
const st = q.state;
/// Ragged logical cohorts keep their requested prefixes across bounded, configured-width query passes.
pub fn splitHold(gpa: std.mem.Allocator, io: std.Io, d: *q.dspark.Drafter, ids: []const u32, count: usize) !void {
    var depths: [16]u32 = @splat(1);
    depths[0] = 9;
    const real_grid = d.holds;
    const real_len = d.holds_len;
    const step_ms = d.step_ms;
    const hold_ms = d.hold_ms;
    const spent_ms = d.hold_spent_ms;
    const encode_ms = d.encode_cpu_ms;
    defer {
        d.holds = real_grid;
        d.holds_len = real_len;
        d.step_ms = step_ms;
        d.hold_ms = hold_ms;
        d.hold_spent_ms = spent_ms;
        d.encode_cpu_ms = encode_ms;
    }
    var pending: [16]u32 = undefined;
    for (pending[0..count], 0..) |*p, i| p.* = ids[(i * 131 + 5) % ids.len];
    const drafter = d.drafter();
    var streams: [16]tf.lanes.Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(gpa);
    while (made < count) : (made += 1) streams[made] = try tf.lanes.Stream.init(gpa, .{ .id = "split", .prompt = &.{1}, .max_new = 1 });
    var opened: usize = 0;
    defer for (streams[0..opened]) |*s| drafter.release(s);
    while (opened < count) : (opened += 1) {
        try drafter.open(&streams[opened]);
        if ((d.lanes.get(&streams[opened]) orelse return error.LaneNotOpened).cap != 64 + d.config.block) return error.LaneNotAtCapacity64;
    }
    // Reference groups use the same kernels on separate lanes.
    var refs: [16]*q.dspark.Lane = undefined;
    var ref_n: usize = 0;
    defer for (refs[0..ref_n]) |l| d.dropLane(l);
    while (ref_n < count) : (ref_n += 1) refs[ref_n] = try d.newLane(64);
    var blocks: [16]q.dspark_block.Block = undefined;
    for (blocks[0..count], refs[0..count], 0..) |*b, l, i| b.* = .{ .lane = l, .pending = pending[i], .anchor = 0, .depth = depths[i] };
    var at: usize = 0;
    while (at < count) {
        const end = @min(count, at + q.dspark.pass_rows / d.config.block);
        _ = try q.dspark_block.hold(d, blocks[at..end], null);
        at = end;
    }
    _ = q.dspark_block.takeHoldMs(d);
    var items: [16]tf.lanes.drafter.Hold = undefined;
    for (items[0..count], 0..) |*it, i| it.* = .{ .stream = &streams[i], .pending = pending[i], .position = 1, .depth = depths[i] };
    const t0 = std.Io.Clock.awake.now(io);
    try drafter.hold(items[0..count]);
    const wall = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    // A logical cohort larger than one physical pass must leave no pending pass or borrowed lane slots.
    if (d.pending != null or d.pass_len != 0) return error.SplitHoldLeftAPassPending;
    for (streams[0..count]) |*s| if (d.lanes.get(s).?.slot != null) return error.SplitHoldLaneInASignalledPass;
    const gpu_ms = d.hold_spent_ms;
    const cpu_ms = d.encode_cpu_ms;
    const taken = q.dspark_block.takeHoldMs(d);
    if (!(taken > 0) or q.dspark_block.takeHoldMs(d) != 0) return error.SplitHoldTimeNotTakenOnce;
    if (taken != cpu_ms + gpu_ms or !(cpu_ms > 0) or !(gpu_ms > 0)) return error.SplitHoldComponentsNotTaken;
    if (taken > wall) return error.SplitHoldTimeExceedsWall;
    // each lane's picks: its depth of them, the same as the reference pass of its group
    for (streams[0..count], refs[0..count], depths[0..count], 0..) |*s, ref, depth, i| {
        var got: [16]u32 = undefined;
        try drafter.held(&.{s}, &.{got[0..depth]});
        if (!std.mem.eql(u32, got[0..depth], ref.held[0..depth])) return error.SplitHoldPicksDiffer;
        if (depth == 1 and i > 0) if (drafter.held(&.{s}, &.{got[0 .. depth + 1]})) |_| return error.SplitHoldHeldMoreThanAsked else |err| if (err != error.DraftHeldFewer) return err;
    }
    std.debug.print("split hold, {d} logical lanes: configured {d}-row query blocks in bounded synchronous passes; {d:.3} ms CPU-encode + {d:.3} ms GPU = {d:.3} ms taken once ({d:.3} ms wall); requested picks equal independent same-regime references\n", .{ count, d.config.block, cpu_ms, gpu_ms, taken, wall });
}
fn untouched(words: []const u16) bool {
    for (words) |v| if (v != 0x7fc0) return false;
    return true;
}

/// Table-only capture must reject raw/table destinations before encoding, and match generic capture.
fn tableOnlyTaps(gpa: std.mem.Allocator, m: *q.Model, scratch: *st.Scratch, ids: []const u32, record: bool, want: []const u16, x: []const u8, logits: []const u8) !void {
    const original = scratch.taps.?;
    defer scratch.taps = original;
    var taps = original;
    taps.table_at = 0;
    taps.buf = try m.device.buffer(scratch.rows * @sizeOf(u64), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    taps.gpu_base = taps.buf.gpuAddress();
    defer taps.buf.deinit();
    scratch.taps = taps;
    const width = taps.count * c.hidden * 2;
    const stage = try m.device.buffer((ids.len + 2) * width, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer stage.deinit();
    const stage_base = stage.gpuAddress();
    @memset(stage.slice(u16, stage.length() / 2), 0x7fc0);
    var cache = try st.Cache.init(gpa, m.device, &m.config.attention, ids.len + 16);
    defer cache.deinit();
    const spent = q.forward.gpu_seconds;
    for ([_]q.forward.Tap{ .scratch, .{ .stage = .{ .buf = taps.buf, .gpu_base = taps.gpu_base, .off = 0 } } }) |tap| {
        if (q.forward.run(m, scratch, &.{.{ .cache = &cache, .rows = ids.len, .tap = tap }}, ids, record, .last)) |_| return error.InvalidTapDestinationAccepted else |err| if (err != error.BadLfmTapDestination) return err;
        if (cache.len != 0 or q.forward.gpu_seconds != spent or !untouched(stage.slice(u16, stage.length() / 2))) return error.InvalidTapDestinationEncoded;
    }
    try q.forward.run(m, scratch, &.{.{ .cache = &cache, .rows = ids.len, .tap = .{ .stage = .{ .buf = stage, .gpu_base = stage_base, .off = width } } }}, ids, record, .last);
    if (!std.mem.eql(u16, want, stage.slice(u16, stage.length() / 2)[width / 2 ..][0..want.len])) return error.TableOnlyTapsDiffer;
    if (!untouched(stage.slice(u16, width / 2)) or !untouched(stage.slice(u16, stage.length() / 2)[(ids.len + 1) * (width / 2) ..])) return error.TableOnlyTapGuardChanged;
    if (!std.mem.eql(u8, x, scratch.x.contents()[0..x.len]) or !std.mem.eql(u8, logits, scratch.logits.contents()[0..logits.len])) return error.TableOnlyTargetOutputMismatch;
}

/// The forward capturing into a stage, into none, and beside a stream with none.
fn stageTaps(gpa: std.mem.Allocator, m: *q.Model, scratch: *st.Scratch, ids: []const u32, record: bool, taps: st.Taps, x: []const u8, logits: []const u8) !void {
    const n = ids.len;
    const row_words = taps.count * c.hidden;
    const want = try gpa.dupe(u16, taps.buf.slice(u16, n * row_words));
    defer gpa.free(want);
    // The independent table-only stage must not overlap this control's stage.
    std.debug.print("target tap control: {d} rows, {d} columns, table-only stage\n", .{ n, taps.count });
    try tableOnlyTaps(gpa, m, scratch, ids, record, want, x, logits);
    const stage = try m.device.buffer((n + 2) * row_words * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer stage.deinit();
    const got = stage.slice(u16, (n + 2) * row_words);
    const scratch_taps = taps.buf.slice(u16, n * row_words);
    const place: q.forward.Place = .{ .buf = stage, .gpu_base = stage.gpuAddress(), .off = row_words * 2 };
    for ([_]q.forward.Tap{ .{ .stage = place }, .none }, 0..) |tap, mode| {
        @memset(got, 0x7fc0);
        @memset(scratch_taps, 0x7fc0);
        var cache = try st.Cache.init(gpa, m.device, &m.config.attention, n + 16);
        defer cache.deinit();
        std.debug.print("target tap control: {d} rows, {d} columns, stage mode {d}\n", .{ n, taps.count, mode });
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
pub fn targetTaps(gpa: std.mem.Allocator, m: *q.Model, scratch: *st.Scratch, ids: []const u32, layers: []const u32) !void {
    const twin = m.kernels.twin;
    defer m.kernels.twin = twin;
    try targetTapMode(gpa, m, scratch, ids, layers);
    if (twin) {
        m.kernels.twin = false;
        try targetTapMode(gpa, m, scratch, ids, layers);
    }
}

fn targetTapMode(gpa: std.mem.Allocator, m: *q.Model, scratch: *st.Scratch, ids: []const u32, layers: []const u32) !void {
    const original = scratch.taps;
    defer scratch.taps = original;
    for ([_]usize{ 1, 4, 8, 16, ids.len }, 0..) |n, shape| {
        if (n > ids.len or (shape == 4 and (n == 1 or n == 4 or n == 8 or n == 16))) continue;
        scratch.taps = null;
        const record = n <= st.batch_rows;
        {
            var plain = try st.Cache.init(gpa, m.device, &m.config.attention, n + 16);
            defer plain.deinit();
            std.debug.print("target tap control: {d} rows, twin {}, no capture\n", .{ n, m.kernels.twin });
            try q.forward.run(m, scratch, &.{.{ .cache = &plain, .rows = n }}, ids[0..n], record, .last);
        }
        const x = try gpa.dupe(u8, scratch.x.contents()[0 .. n * c.hidden * 2]);
        defer gpa.free(x);
        const logits = try gpa.dupe(u8, scratch.logits.contents()[0 .. c.vocab * 2]);
        defer gpa.free(logits);
        for (0..3) |layout| {
            var count: u32 = @intCast(layers.len);
            var tap_layers: [st.max_taps]u32 = @splat(0);
            @memcpy(tap_layers[0..layers.len], layers);
            if (layout == 1) {
                count = 3;
                tap_layers[0..3].* = .{ 0, c.layers - 1, c.layers - 1 };
            } else if (layout == 2) {
                count = st.max_taps;
                for (0..st.max_taps) |i| tap_layers[i] = @intCast(if (i >= st.max_taps - 2) c.layers - 1 else i);
            }
            const table_at = n * count * c.hidden * 2;
            const buf = try m.device.buffer(table_at + scratch.rows * @sizeOf(u64), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            const taps: st.Taps = .{ .layers = tap_layers, .count = count, .buf = buf, .gpu_base = buf.gpuAddress(), .table_at = table_at };
            defer {
                scratch.taps = null;
                taps.buf.deinit();
            }
            scratch.taps = taps;
            {
                var captured = try st.Cache.init(gpa, m.device, &m.config.attention, n + 16);
                defer captured.deinit();
                std.debug.print("target tap control: {d} rows, twin {}, {d} columns, scratch capture\n", .{ n, m.kernels.twin, count });
                try q.forward.run(m, scratch, &.{.{ .cache = &captured, .rows = n }}, ids[0..n], record, .last);
            }
            if (!std.mem.eql(u8, x, scratch.x.contents()[0..x.len]) or !std.mem.eql(u8, logits, scratch.logits.contents()[0..logits.len])) return error.TargetTapOutputMismatch;
            const data = taps.buf.slice(u16, n * taps.count * c.hidden);
            for (taps.layers[0..taps.count], 0..) |layer, i| for (taps.layers[0..i], 0..) |other, j| {
                if (layer != other) continue;
                for (0..n) |row| if (!std.mem.eql(u16, data[(row * taps.count + i) * c.hidden ..][0..c.hidden], data[(row * taps.count + j) * c.hidden ..][0..c.hidden])) return error.RepeatedTapMismatch;
            };
            try stageTaps(gpa, m, scratch, ids[0..n], record, taps, x, logits);
        }
        std.debug.print("target taps: {d} rows, twin {}, released/duplicate/max-column layouts preserve all normalized and logit bytes\n", .{ n, m.kernels.twin });
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
    const encode_ms = d.encode_cpu_ms;
    defer {
        d.w.w1 = old_w1;
        d.w.w2 = old_w2;
        d.holds = grid;
        d.holds_len = grid_len;
        d.step_ms = step_ms;
        d.hold_ms = hold_ms;
        d.hold_spent_ms = spent_ms;
        d.encode_cpu_ms = encode_ms;
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

/// A shorter retained prefix cannot remove the future mask rows of a bidirectional query block.
pub fn fixedBlockPrefix(gpa: std.mem.Allocator, d: *q.dspark.Drafter, lane: *q.dspark.Lane, pending: u32, anchor: u64) !void {
    try d.settle();
    _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = pending, .anchor = anchor, .depth = d.config.block }}, null);
    const expected = try gpa.dupe(u8, d.s.logits.contents()[0 .. c.vocab * 2]);
    defer gpa.free(expected);
    const first = lane.held[0];
    const has_confidence = lane.conf_len > 0;
    const confidence: u32 = @bitCast(lane.conf[0]);
    for ([_]u32{ 1, @min(3, d.config.block), d.config.block }) |depth| {
        _ = try q.dspark_block.hold(d, &.{.{ .lane = lane, .pending = pending, .anchor = anchor, .depth = depth }}, null);
        if (!std.mem.eql(u8, expected, d.s.logits.contents()[0..expected.len])) return error.DraftPrefixDependsOnGrant;
        if (lane.held[0] != first) return error.DraftFirstProposalDependsOnGrant;
        if (has_confidence and (lane.conf_len == 0 or @as(u32, @bitCast(lane.conf[0])) != confidence)) return error.DraftConfidenceDependsOnGrant;
    }
    std.debug.print("fixed query block: {d} rows; retained prefixes 1/3/{d} preserve all first-row base-logit bytes, first proposal and confidence\n", .{ d.config.block, d.config.block });
}
