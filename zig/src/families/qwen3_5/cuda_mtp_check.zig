//! GPU regressions for the MTP drafter: absorbs in chunks, a tree's lanes against their paths chained, a poisoned ring.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const Mtp = @import("cuda_mtp.zig").Mtp;
const solo = @import("cuda_solo.zig");
const retained_rows = @import("cuda_cache.zig").tap_window;
const prompt_kv_tile = @import("cuda_mtp_absorb.zig").prompt_kv_tile;

/// Compare target-normalized rows with MTP's tap-plus-norm path and resulting K/V state.
pub fn normalizedInputCheck(gpa: std.mem.Allocator, d: *Mtp, tap: u64, normalized: u64, start: u64, rows: usize, follow: []const u32) !bool {
    const e = d.e;
    if (e.g.bf16) {
        std.debug.print("SKIP MTP direct normalized input: BF16 target uses shifted RMSNorm, so the fast path is disabled\n", .{});
        return true;
    }
    if (rows == 0 or follow.len < rows) return error.BadDraftRows;
    const row_bytes = e.g.hidden * 2;
    const bytes = rows * row_bytes;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const from_tap = try a.alloc(u8, bytes);
    const from_target = try a.alloc(u8, bytes);
    const raw_tap = try a.alloc(u8, bytes);
    var at: usize = 0;
    while (at < rows) {
        const n = @min(prompt_kv_tile, rows - at);
        try e.run(d.ring_fns.gather_taps, .{ n, 1, 1 }, .{ 256, 1, 1 }, .{ tap, @as(c_int, @intCast(e.tapRows())), @as(c_longlong, @intCast(start + at)), @as(c_int, @intCast(e.g.hidden)), d.taps_in });
        try download(d, raw_tap[at * row_bytes ..][0 .. n * row_bytes], d.taps_in);
        try d.norm(d.taps_in, 0, e.final_norm, d.hin, n);
        try download(d, from_tap[at * row_bytes ..][0 .. n * row_bytes], d.hin);
        at += n;
    }
    try download(d, from_target, normalized);
    const norm_equal = std.mem.eql(u8, from_tap, from_target);
    if (!norm_equal) reportMismatch("normalized", from_tap, from_target, rows, row_bytes);
    var populated: usize = 0;
    for (0..rows * row_bytes / 2) |i| {
        if (raw_tap[i * 2] != 0 or raw_tap[i * 2 + 1] != 0) populated += 1;
    }
    std.debug.print("  MTP tap source populated: {d}/{d} bf16 entries\n", .{ populated, rows * row_bytes / 2 });

    const reference = try d.create();
    defer d.drop(reference);
    const direct = try d.create();
    defer d.drop(direct);
    const skipped = @min(rows / 2, 17);
    const source_start = start + skipped;
    const source_rows = rows - skipped;
    const normalized_source = normalized + skipped * e.g.hidden * 2;
    try d.absorbRole(reference, tap, source_start, source_rows, null, follow[skipped..][0..source_rows], true, false, null);
    try d.absorbNormalizedRole(direct, .{ .buffer = normalized_source, .start = source_start, .rows = @intCast(source_rows), .row_bytes = @intCast(e.g.hidden * 2) }, start, rows, follow[0..rows], true, false, null);
    const ring_bytes = 2 * d.capacity * d.kvWidth() * 2;
    const ref_ring = try a.alloc(u8, ring_bytes);
    const direct_ring = try a.alloc(u8, ring_bytes);
    try download(d, ref_ring, reference.ring.ptr);
    try download(d, direct_ring, direct.ring.ptr);
    const kv_equal = sameLiveRing(d, ref_ring, direct_ring, source_start, start + @as(u64, @intCast(rows)));
    const ok = populated > 0 and norm_equal and kv_equal;
    std.debug.print("{s} MTP target final-norm reuse: raw residual {s}, normalized bytes {s}, resulting K/V bytes {s}, {d} rows after a {d}-row prefix skip\n", .{
        if (ok) "PASS" else "FAIL",
        if (populated > 0) "captured" else "all zero",
        if (norm_equal) "equal" else "differ",
        if (kv_equal) "equal" else "differ",
        source_rows,
        skipped,
    });
    return ok;
}

fn reportMismatch(name: []const u8, want: []const u8, got: []const u8, rows: usize, row_bytes: usize) void {
    var first: ?usize = null;
    var differs: usize = 0;
    for (0..rows * row_bytes / 2) |i| {
        const byte = i * 2;
        if (want[byte] != got[byte] or want[byte + 1] != got[byte + 1]) {
            if (first == null) first = byte;
            differs += 1;
        }
    }
    if (first) |byte| {
        const want_value = @as(u16, want[byte]) | (@as(u16, want[byte + 1]) << 8);
        const got_value = @as(u16, got[byte]) | (@as(u16, got[byte + 1]) << 8);
        std.debug.print("  {s} mismatch: {d}/{d} elements; first at row {d}, channel {d}: {x:0>4} != {x:0>4}\n", .{
            name,
            differs,
            rows * row_bytes / 2,
            byte / row_bytes,
            (byte % row_bytes) / 2,
            want_value,
            got_value,
        });
    }
}

/// Siblings 0 and 1 at depth 0, six siblings 3 to 8 at depth 1, lane 9 at depth 2 under 3 and 0; each has a child.
const parents = [_]i32{ -1, -1, -1, 0, 0, 0, 0, 1, 1, 3, 4, 5, 6, 7, 8, 9 };
const ranks = [_]u8{ 0, 1, 2, 0, 1, 2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
const depths = [_]u8{ 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3 };
const internal = [_]usize{ 0, 1, 3, 4, 5, 6, 7, 8, 9 };

/// A lane's rows from a pass: its normed output, its key and its value.
const Rows = struct { state: []u8, k: []u8, v: []u8 };

/// Absorbs `ids` from the target's tap ring (`tap`, filled by forwards over them), then checks the tree twice.
pub fn run(gpa: std.mem.Allocator, d: *Mtp, tap: u64, ids: []const u32) !bool {
    const e = d.e;
    const g = e.g;
    const prompt_rows = solo.passRows(e);
    d.trees = true;
    var ok = try kvOnlyProjectionCheck(gpa, d, tap, ids);
    ok = try denseMlpCheck(gpa, d) and ok;
    const ctx = try d.create();
    defer d.drop(ctx);
    try d.absorb(ctx, tap, 0, ids.len - 1, null, ids[1..]);
    const pos0 = ctx.end;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    // Compare the stream's target chunks with direct absorbs of exactly the suffix each chunk retains.
    const apart = try d.create();
    defer d.drop(apart);
    const reference = try d.create();
    defer d.drop(reference);
    const end: usize = @intCast(pos0);
    const source_start = end - e.tapRows();
    const second_start = source_start + prompt_rows;
    const first_retained_start = second_start - retained_rows;
    try d.absorb(reference, tap, first_retained_start, retained_rows, null, ids[first_retained_start + 1 ..][0..retained_rows]);
    try d.absorb(reference, tap, second_start, end - second_start, null, ids[second_start + 1 ..][0 .. end - second_start]);

    var from = source_start;
    while (from < end) {
        const n = @min(prompt_rows, end - from);
        try d.absorb(apart, tap, from, n, null, ids[from + 1 ..][0..n]);
        from += n;
    }
    const ring = 2 * d.capacity * d.kvWidth() * 2;
    const whole = try a.alloc(u8, ring + g.hidden * 2);
    const parts = try a.alloc(u8, ring + g.hidden * 2);
    try download(d, whole[0..ring], reference.ring.ptr);
    try download(d, whole[ring..], reference.last.ptr);
    try download(d, parts[0..ring], apart.ring.ptr);
    try download(d, parts[ring..], apart.last.ptr);
    const live_begin = @max(reference.begin, apart.begin);
    const live_end = @min(reference.end, apart.end);
    const live_rows = if (live_begin <= live_end) live_end - live_begin else 0;
    const live_ring_equal = sameLiveRing(d, whole[0..ring], parts[0..ring], live_begin, live_end);
    const expected_live_rows = @min(d.capacity, 2 * retained_rows);
    const live_count_equal = live_rows == @as(u64, @intCast(expected_live_rows));
    const state_equal = std.mem.eql(u8, whole[ring..], parts[ring..]);
    const candidates_equal = std.mem.eql(u32, &reference.cands, &apart.cands);
    ok = live_count_equal and live_ring_equal and state_equal and candidates_equal and ok;
    if (!live_count_equal or !live_ring_equal or !state_equal or !candidates_equal) {
        std.debug.print("  details: live rows {d}/{d}, ring {s}, final state {s}, candidates {s}\n", .{
            live_rows,
            expected_live_rows,
            if (live_ring_equal) "equal" else "differ",
            if (state_equal) "equal" else "differ",
            if (candidates_equal) "equal" else "differ",
        });
    }
    std.debug.print("{s} {d} target tap rows in chunks of at most {d} vs their retained suffixes: {d} live KV rows, final state and candidates\n", .{ if (ok) "PASS" else "FAIL", pos0, prompt_rows, live_rows });
    var p = parents;
    var r = ranks;
    var dp = depths;
    const shape: lanes.shape.Shape = .{ .parents = &p, .ranks = &r, .depths = &dp, .expected = 0 };
    const hidden = g.hidden * 2;
    const kw = d.kvWidth() * 2;

    try d.holdTree(ctx, pos0, &shape);
    const tree = try lanesRows(a, d, hidden, kw);
    const held = ctx.held;

    // the ring at and past the committed end poisoned: a tree pass reads none of it, so nothing may change
    const junk = try a.alloc(u8, 16 * kw);
    @memset(junk, 0x7f);
    for (0..16) |i| {
        const slot = (pos0 + i) % d.capacity;
        try upload(d, d.ringK(ctx) + slot * kw, junk[0..kw]);
        try upload(d, d.ringV(ctx) + slot * kw, junk[0..kw]);
    }
    try d.holdTree(ctx, pos0, &shape);
    const poisoned = try lanesRows(a, d, hidden, kw);
    ok = report("poisoned ring suffix, tree rows", poisoned, tree, &internal) and ok;
    if (!std.mem.eql(u32, &ctx.held, &held)) {
        std.debug.print("FAIL poisoned ring suffix: the lanes' tokens changed\n", .{});
        ok = false;
    }

    // each internal lane's path chained from the newest absorbed row, one row a slot, against the tree's rows
    var chain: Rows = .{ .state = try a.alloc(u8, tree.state.len), .k = try a.alloc(u8, tree.k.len), .v = try a.alloc(u8, tree.v.len) };
    for (internal) |lane| {
        var path: [4]usize = undefined;
        var n: usize = 0;
        var at: i32 = @intCast(lane);
        while (at >= 0) : (at = parents[@intCast(at)]) {
            path[n] = @intCast(at);
            n += 1;
        }
        std.mem.reverse(usize, path[0..n]);
        for (path[0..n], 0..) |step, k| {
            const src = if (k == 0) ctx.last.ptr else d.x;
            try d.scratch.copyFrom(d.hin - d.scratch.ptr, src, hidden, e.stream.handle);
            try d.putIds(&.{held[step]});
            try d.layer(ctx, 1, pos0 + k, .ring);
        }
        try download(d, chain.state[lane * hidden ..][0..hidden], d.x);
        try download(d, chain.k[lane * kw ..][0..kw], d.kb);
        try download(d, chain.v[lane * kw ..][0..kw], d.vb);
    }
    ok = report("six siblings a depth, tree rows vs each lane's path chained", tree, chain, &internal) and ok;
    // batched shared rounds against one stream at a time, on the same tap ring
    ok = try @import("cuda_mtp_batch_check.zig").run(gpa, d, tap, ids) and ok;
    ok = try @import("cuda_mtp_yarn_check.zig").run(gpa, d, tap, ids) and ok;
    ok = try @import("cuda_mtp_batch_check.zig").runTreeMultiStream(gpa, d, tap, ids) and ok;
    return ok;
}

/// KV-only prompt absorption must write byte-identical K/V rows to the original Q/K/V proposal path.
fn kvOnlyProjectionCheck(gpa: std.mem.Allocator, d: *Mtp, tap: u64, ids: []const u32) !bool {
    const previous_role = d.projection_role;
    defer d.projection_role = previous_role;
    d.projection_role = .prompt;
    var ok = true;
    for ([_]usize{ 1, engine.most_verify, engine.most_verify * 2, engine.most_verify * 4 }) |requested_rows| {
        const rows = @min(requested_rows, ids.len -| 1);
        if (rows == 0) return error.BadDraftRows;
        ok = try kvOnlyRowsCheck(gpa, d, tap, ids, rows) and ok;
    }
    return ok;
}

fn kvOnlyRowsCheck(gpa: std.mem.Allocator, d: *Mtp, tap: u64, ids: []const u32, rows: usize) !bool {
    const e = d.e;
    const g = e.g;
    const kw_bytes = d.kvWidth() * 2;
    const kv_only = try d.create();
    defer d.drop(kv_only);
    const proposal = try d.create();
    defer d.drop(proposal);
    // Context setup uses shared MTP scratch, so stage the measured input only after both contexts are ready.
    try e.run(d.ring_fns.gather_taps, .{ rows, 1, 1 }, .{ 256, 1, 1 }, .{ tap, @as(c_int, @intCast(e.tapRows())), @as(c_longlong, 0), @as(c_int, @intCast(g.hidden)), d.taps_in });
    try d.norm(d.taps_in, 0, e.final_norm, d.hin, rows);
    try d.putIds(ids[1 .. rows + 1]);

    var kv_start = try cuda.Event.init(e.d, true);
    defer kv_start.deinit();
    var kv_end = try cuda.Event.init(e.d, true);
    defer kv_end.deinit();
    try kv_start.record(e.stream);
    try d.layerModeWithIds(kv_only, d.ids, rows, 0, .ring, false);
    try kv_end.record(e.stream);
    const kv_sync_start = std.Io.Clock.awake.now(d.io);
    try kv_end.synchronize();
    const kv_sync_ns = kv_sync_start.durationTo(std.Io.Clock.awake.now(d.io)).toNanoseconds();
    const kv_device_ms = try cuda.Event.elapsedMs(kv_start, kv_end);

    var proposal_start = try cuda.Event.init(e.d, true);
    defer proposal_start.deinit();
    var proposal_end = try cuda.Event.init(e.d, true);
    defer proposal_end.deinit();
    try proposal_start.record(e.stream);
    try d.layerModeWithIds(proposal, d.ids, rows, 0, .ring, true);
    try proposal_end.record(e.stream);
    const proposal_sync_start = std.Io.Clock.awake.now(d.io);
    try proposal_end.synchronize();
    const proposal_sync_ns = proposal_sync_start.durationTo(std.Io.Clock.awake.now(d.io)).toNanoseconds();
    const proposal_device_ms = try cuda.Event.elapsedMs(proposal_start, proposal_end);

    const a = try gpa.alloc(u8, rows * kw_bytes * 2);
    defer gpa.free(a);
    const b = try gpa.alloc(u8, rows * kw_bytes * 2);
    defer gpa.free(b);
    try download(d, a[0 .. rows * kw_bytes], d.ringK(kv_only));
    try download(d, a[rows * kw_bytes ..], d.ringV(kv_only));
    try download(d, b[0 .. rows * kw_bytes], d.ringK(proposal));
    try download(d, b[rows * kw_bytes ..], d.ringV(proposal));
    const same = std.mem.eql(u8, a, b);
    std.debug.print("{s} MTP K/V ring match for {d} prompt rows: KV-only cuda_stream_ms={d:.3}, explicit_sync_wait_ms={d:.3}; full-proposal cuda_stream_ms={d:.3}, explicit_sync_wait_ms={d:.3}\n", .{
        if (same) "PASS" else "FAIL",
        rows,
        kv_device_ms,
        @as(f64, @floatFromInt(@max(kv_sync_ns, 0))) / 1e6,
        proposal_device_ms,
        @as(f64, @floatFromInt(@max(proposal_sync_ns, 0))) / 1e6,
    });
    return same;
}

/// Compare only the positions each context can attend to; slots outside the live interval may be stale.
fn sameLiveRing(d: *Mtp, got: []const u8, want: []const u8, begin: u64, end: u64) bool {
    const row_bytes = d.kvWidth() * 2;
    const one_ring_bytes = d.capacity * row_bytes;
    if (got.len != 2 * one_ring_bytes or want.len != got.len or begin > end) return false;
    var position = begin;
    const capacity: u64 = @intCast(d.capacity);
    while (position < end) : (position += 1) {
        const slot: usize = @intCast(position % capacity);
        const offset = slot * row_bytes;
        if (!std.mem.eql(u8, got[offset..][0..row_bytes], want[offset..][0..row_bytes]) or
            !std.mem.eql(u8, got[one_ring_bytes + offset ..][0..row_bytes], want[one_ring_bytes + offset ..][0..row_bytes])) return false;
    }
    return true;
}

/// Checks dense MTP batch SwiGLU rows against the serial packed-weight reference.
fn denseMlpCheck(gpa: std.mem.Allocator, d: *Mtp) !bool {
    const dense = switch (d.w.mlp) {
        .dense => |m| m,
        .moe => {
            std.debug.print("SKIP dense MTP SwiGLU GPU check: this checkpoint has routed MTP weights\n", .{});
            return true;
        },
    };
    const e = d.e;
    const hidden = e.g.hidden;
    const rows: usize = 4;
    const bytes = rows * hidden * 2;
    const input = try gpa.alloc(u8, bytes);
    defer gpa.free(input);
    for (0..rows * hidden) |i| {
        const value = @as(f32, @floatFromInt(@as(i32, @intCast(i % 127)) - 63)) / 128.0;
        const bf16_bits: u16 = @import("core").affine4_host.bf16of(value);
        std.mem.writeInt(u16, input[i * 2 ..][0..2], bf16_bits, .little);
    }
    try d.scratch.fill8(0, e.stream.handle);
    try upload(d, d.hin, input);
    const batch_out = try gpa.alloc(u8, bytes);
    defer gpa.free(batch_out);
    const serial_out = try gpa.alloc(u8, bytes);
    defer gpa.free(serial_out);

    try d.mlp(d.hin, d.hn, rows);
    for (0..rows) |row| {
        const offset = row * hidden * 2;
        const x = d.hin + offset;
        try d.proj(dense.gate, x, e.gate, 1);
        try d.proj(dense.up, x, e.up, 1);
        const elements = e.g.intermediate;
        try e.run(e.fns.mlp_act, .{ (elements + 1023) / 1024, 1, 1 }, .{ 256, 1, 1 }, .{ e.gate, e.up, e.act, @as(c_longlong, @intCast(elements)), @as(u64, 0), @as(c_int, @intCast(elements)), @as(u64, 0) });
        try d.proj(dense.down, e.act, d.r + offset, 1);
    }
    try download(d, batch_out, d.hn);
    try download(d, serial_out, d.r);
    const same = std.mem.eql(u8, batch_out, serial_out);
    if (!same) {
        var differing: usize = 0;
        var max_abs: f64 = 0;
        var first: ?struct { index: usize, batch: u16, serial: u16 } = null;
        for (0..rows * hidden) |i| {
            const got = std.mem.readInt(u16, batch_out[i * 2 ..][0..2], .little);
            const want = std.mem.readInt(u16, serial_out[i * 2 ..][0..2], .little);
            if (got == want) continue;
            differing += 1;
            if (first == null) first = .{ .index = i, .batch = got, .serial = want };
            const got_f32: f32 = @bitCast(@as(u32, got) << 16);
            const want_f32: f32 = @bitCast(@as(u32, want) << 16);
            max_abs = @max(max_abs, @abs(@as(f64, got_f32) - @as(f64, want_f32)));
        }
        const f = first.?;
        std.debug.print("dense MTP GPU comparison: {d}/{d} BF16 outputs differ, max_abs={e}, first index={d} batched=0x{x:0>4} serial=0x{x:0>4}\n", .{ differing, rows * hidden, max_abs, f.index, f.batch, f.serial });
    }
    std.debug.print("{s} dense MTP SwiGLU: four batched affine-4 rows equal the serial packed-weight reference\n", .{if (same) "PASS" else "FAIL"});
    return same;
}

fn lanesRows(a: std.mem.Allocator, d: *Mtp, hidden: usize, kw: usize) !Rows {
    const n = parents.len;
    const out: Rows = .{ .state = try a.alloc(u8, n * hidden), .k = try a.alloc(u8, n * kw), .v = try a.alloc(u8, n * kw) };
    try download(d, out.state, d.lane_state);
    try download(d, out.k, d.lane_k);
    try download(d, out.v, d.lane_v);
    return out;
}

/// PASS when every internal lane's state, key and value bytes are equal; else the first lane that differs.
fn report(what: []const u8, got: Rows, want: Rows, lanes_: []const usize) bool {
    const hidden = got.state.len / parents.len;
    const kw = got.k.len / parents.len;
    for (lanes_) |l| {
        const same = std.mem.eql(u8, got.state[l * hidden ..][0..hidden], want.state[l * hidden ..][0..hidden]) and
            std.mem.eql(u8, got.k[l * kw ..][0..kw], want.k[l * kw ..][0..kw]) and std.mem.eql(u8, got.v[l * kw ..][0..kw], want.v[l * kw ..][0..kw]);
        if (!same) {
            std.debug.print("FAIL {s}: lane {d} (depth {d}) differs\n", .{ what, l, depths[l] });
            return false;
        }
    }
    std.debug.print("PASS {s}: {d} lanes byte-equal\n", .{ what, lanes_.len });
    return true;
}

fn upload(d: *Mtp, dst: u64, bytes: []const u8) !void {
    const e = d.e;
    try cuda.copy_source.enqueue(e.d, dst, bytes, e.stream);
    try e.stream.synchronize();
}

fn download(d: *Mtp, out: []u8, src: u64) !void {
    const e = d.e;
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(out.ptr, src, out.len, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
}
