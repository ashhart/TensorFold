//! TF_DSV41_CONE_CHECK=<sizes>: the cone prefill against the full path on slot 0 (logits, kept state, ids, time).
const std = @import("std");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;

const vocab = 129280;
const log = std.log.scoped(.cone_check);

fn envText(name: [:0]const u8) ?[]const u8 {
    return (@import("sys").environment.Process{}).get(name);
}

fn argmax(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = @intCast(i);
    };
    return best;
}

fn seconds(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds())) / 1e9;
}

/// The slot's kept buffers and cursors, copied out after a prefill.
const Kept = struct { storage: []u8, live: []u8, taps: []u8, position: u32, taps_row: u32, taps_n: u32, taps_rows_last: u32, taps_base: u32 };

fn capture(m: *ds.Model, a: std.mem.Allocator) !Kept {
    const state = &m.caches[0];
    return .{
        .storage = try a.dupe(u8, state.storage.contents()[0..state.plan.packedBytes()]),
        .live = try a.dupe(u8, state.live.contents()[0..state.plan.mutableBytes()]),
        .taps = try a.dupe(u8, m.taps.contents()[0..m.taps.length()]),
        .position = state.position,
        .taps_row = m.taps_row,
        .taps_n = m.taps_n,
        .taps_rows_last = m.taps_rows_last,
        .taps_base = m.taps_base,
    };
}

/// One path's run: the prefill's last row, its kept state, the greedy ids after it.
const Run = struct {
    logits: []f32,
    snap: Kept,
    picks: []u32,
    steps: []f32,
    prefill_s: f64,
};

fn prefill(m: *ds.Model, a: std.mem.Allocator, io: std.Io, ids: []const u32) !struct { []f32, f64 } {
    const rows = m.workspace.plan.rows;
    try m.reset();
    // beyond the position the buffers hold the last run's rows: start both paths from zeros
    const state = &m.caches[0];
    @memset(state.storage.contents()[0..state.plan.packedBytes()], 0);
    @memset(state.live.contents()[0..state.plan.mutableBytes()], 0);
    @memset(m.taps.contents()[0..m.taps.length()], 0);
    const t0 = seconds(io);
    var at: usize = 0;
    var last: []f32 = &.{};
    while (at < ids.len) {
        const end = @min(at + rows, ids.len);
        if (end == ids.len) last = try a.dupe(f32, try m.forward(ids[at..end], .last)) else _ = try m.forward(ids[at..end], .none);
        at = end;
    }
    return .{ last, seconds(io) - t0 };
}

fn run(m: *ds.Model, a: std.mem.Allocator, io: std.Io, ids: []const u32, decode_steps: usize, mode: u32) !Run {
    ds.forward_cone.force(m, mode);
    const pre = try prefill(m, a, io, ids);
    const snap = try capture(m, a);
    const picks = try a.alloc(u32, decode_steps);
    const steps = try a.alloc(f32, decode_steps * vocab);
    var logits: []const f32 = pre[0];
    for (0..decode_steps) |k| {
        picks[k] = argmax(logits);
        logits = try m.forward(&[1]u32{picks[k]}, .last);
        @memcpy(steps[k * vocab ..][0..vocab], logits);
    }
    return .{ .logits = pre[0], .snap = snap, .picks = picks, .steps = steps, .prefill_s = pre[1] };
}

/// Five-row verify rounds after the prompt, rolled back to 1-4 rows as the served no-re-feed keep does; digest of all.
fn rollbackRun(m: *ds.Model, a: std.mem.Allocator, io: std.Io, ids: []const u32, mode: u32, rounds: usize) !struct { [8]u8, usize } {
    ds.forward_cone.force(m, mode);
    const pre = try prefill(m, a, io, ids);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var pend = argmax(pre[0]);
    h.update(std.mem.asBytes(&pend));
    for (0..rounds) |r| {
        const kept: u32 = 1 + @as(u32, @intCast(r % 4));
        var feeds: [5]u32 = undefined;
        feeds[0] = pend;
        for (feeds[1..], 1..) |*f, k| f.* = @intCast(1000 + (pend * 31 + k * 977 + r * 13) % 100000);
        m.commitSlot(0) catch {};
        try m.beginSlot(0);
        const lg = try m.forwardOn(0, &feeds, .all);
        h.update(std.mem.sliceAsBytes(lg[0 .. 5 * vocab]));
        pend = argmax(lg[(kept - 1) * vocab ..][0..vocab]);
        try m.rollbackSlot(0);
        try m.restoreRing(0, kept, 5);
        try m.portKeep(0, kept);
        m.commitSlot(0) catch {};
        h.update(std.mem.asBytes(&pend));
    }
    for (0..32) |_| {
        const lg = try m.forward(&[1]u32{pend}, .last);
        h.update(std.mem.sliceAsBytes(lg));
        pend = argmax(lg);
        h.update(std.mem.asBytes(&pend));
    }
    var out: [32]u8 = undefined;
    h.final(&out);
    return .{ out[0..8].*, m.caches[0].position };
}

fn diffBytes(want: []const u8, got: []const u8, first: *usize) usize {
    var n: usize = 0;
    for (want, got, 0..) |x, y, i| if (x != y) {
        if (n == 0) first.* = i;
        n += 1;
    };
    return n;
}

fn diffFloats(want: []const f32, got: []const f32) struct { usize, f32 } {
    var n: usize = 0;
    var worst: f32 = 0;
    for (want, got) |x, y| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
        n += 1;
        worst = @max(worst, @abs(x - y));
    };
    return .{ n, worst };
}

fn sha(picks: []const u32) [8]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(picks), &h, .{});
    return h[0..8].*;
}

fn compare(m: *const ds.Model, n: usize, mode: u32, full: Run, cone: Run, decode_steps: usize) !void {
    const lg = diffFloats(full.logits, cone.logits);
    var steps_bad: usize = 0;
    var steps_worst: f32 = 0;
    for (0..decode_steps) |k| {
        const d = diffFloats(full.steps[k * vocab ..][0..vocab], cone.steps[k * vocab ..][0..vocab]);
        steps_bad += @intFromBool(d[0] != 0);
        steps_worst = @max(steps_worst, d[1]);
    }
    var first: usize = 0;
    const st = diffBytes(full.snap.storage, cone.snap.storage, &first);
    const st_first = first;
    first = 0;
    const lv = diffBytes(full.snap.live, cone.snap.live, &first);
    const lv_first = first;
    first = 0;
    const tp = diffBytes(full.snap.taps, cone.snap.taps, &first);
    const ring_bytes = @as(usize, m.caches[0].plan.layers) * ds.kv.cache.window * ds.kv.cache.ring_stride;
    const layer_bytes = ds.kv.cache.window * ds.kv.cache.ring_stride;
    regions(m.caches[0].plan, &full.snap, &cone.snap);
    log.info("n={d} mode {d}: last-token logits differ in {d}/{d} values (worst {e}), argmax {d}/{d}", .{ n, mode, lg[0], vocab, lg[1], argmax(full.logits), argmax(cone.logits) });
    log.info("n={d} mode {d}: greedy {d} tokens sha full {x} cone {x}, {d} step rows differ (worst {e})", .{ n, mode, decode_steps, sha(full.picks), sha(cone.picks), steps_bad, steps_worst });
    log.info("n={d}: storage {d} of {d} bytes differ (first {d}), live {d} of {d} (first {d}, ring layer {d}, ring {d} B), taps {d} of {d}; pos {d}/{d} taps_row {d}/{d} taps_n {d}/{d} last {d}/{d} base {d}/{d}", .{
        n,                   st,                 full.snap.storage.len, st_first,           lv,                 full.snap.live.len, lv_first,         lv_first / layer_bytes,   ring_bytes,               tp,
        full.snap.taps.len,  full.snap.position, cone.snap.position,    full.snap.taps_row, cone.snap.taps_row, full.snap.taps_n,   cone.snap.taps_n, full.snap.taps_rows_last, cone.snap.taps_rows_last, full.snap.taps_base,
        cone.snap.taps_base,
    });
    log.info("n={d} mode {d}: prefill full {d:.2} s ({d:.0} tok/s), cone {d:.2} s ({d:.0} tok/s), x{d:.2}", .{
        n,
        mode,
        full.prefill_s,
        @as(f64, @floatFromInt(n)) / full.prefill_s,
        cone.prefill_s,
        @as(f64, @floatFromInt(n)) / cone.prefill_s,
        full.prefill_s / cone.prefill_s,
    });
    if (lg[0] != 0 or steps_bad != 0 or !std.mem.eql(u32, full.picks, cone.picks) or
        full.snap.position != cone.snap.position or full.snap.taps_row != cone.snap.taps_row or
        full.snap.taps_n != cone.snap.taps_n or full.snap.taps_rows_last != cone.snap.taps_rows_last or
        full.snap.taps_base != cone.snap.taps_base) return error.ConeMismatch;
}

/// Where the kept state differs: storage by owner and kind, live by ring layer, then the scratch behind the rings.
fn regions(plan: ds.kv.cache.Plan, full: *const Kept, cone: *const Kept) void {
    const cache = ds.kv.cache;
    var off: usize = 0;
    for (cache.ratios, 0..) |r, o| {
        const rows = plan.capacity / r;
        inline for (.{ .{ "main", cache.main_stride }, .{ "index", cache.index_stride } }) |part| {
            const len = rows * part[1];
            var first: usize = 0;
            const bad = diffBytes(full.storage[off..][0..len], cone.storage[off..][0..len], &first);
            if (bad != 0) log.info("  storage owner layer {d} {s}: {d} bytes differ, first row {d} of {d}", .{ cache.owners[o], part[0], bad, first / part[1], rows });
            off += len;
        }
    }
    const ring = cache.window * cache.ring_stride;
    for (0..plan.layers) |l| {
        var first: usize = 0;
        const bad = diffBytes(full.live[l * ring ..][0..ring], cone.live[l * ring ..][0..ring], &first);
        if (bad != 0) log.info("  live ring layer {d}: {d} bytes differ, first slot {d}", .{ l, bad, first / cache.ring_stride });
    }
    const part = cache.Plan.partialOffset(plan);
    var first: usize = 0;
    const pbad = diffBytes(full.live[part..][0..cache.Plan.partialBytes(plan)], cone.live[part..][0..cache.Plan.partialBytes(plan)], &first);
    if (pbad != 0) log.info("  live partial scratch: {d} bytes differ, first at {d}", .{ pbad, first });
    for (cache.owners, 0..) |owner, o| {
        const at = plan.captureOffset() + o * cache.captureBytes();
        const bad = diffBytes(full.live[at..][0..cache.captureBytes()], cone.live[at..][0..cache.captureBytes()], &first);
        if (bad != 0) log.info("  live capture owner layer {d}: {d} of {d} bytes differ, first slot {d}", .{ owner, bad, cache.captureBytes(), first / (2 * cache.capture_width * 4) });
    }
}

fn parseIds(a: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,[]");
    while (it.next()) |word| try out.append(a, try std.fmt.parseInt(u32, word, 10));
    if (out.items.len == 0) return error.EmptyPrompt;
    return out.toOwnedSlice(a);
}

fn tokens(a: std.mem.Allocator, io: std.Io, n: usize) ![]u32 {
    const out = try a.alloc(u32, n);
    if (envText("TF_DSV41_CONE_TOKENS")) |path| {
        const src = try parseIds(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)));
        for (out, 0..) |*t, i| t.* = src[i % src.len];
        return out;
    }
    var x: u64 = 0x9e3779b97f4a7c15;
    for (out) |*t| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        t.* = @intCast(1000 + (x >> 33) % 100000);
    }
    return out;
}

/// Runs the check for the sizes in `spec`; a size the slot cannot hold is skipped.
pub fn runSizes(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, spec: []const u8) !void {
    const restore_mode = try @import("sys").environment.read(u32, "TF_DSV41_CONE", 2);
    defer ds.forward_cone.force(m, restore_mode);
    const decode_steps: usize = if (envText("TF_DSV41_CONE_STEPS")) |s| try std.fmt.parseInt(usize, s, 10) else 128;
    var it = std.mem.tokenizeScalar(u8, spec, ',');
    // warm both paths: the first prefill pays pipeline builds and page-in
    {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const ids = try tokens(a, io, 5000);
        for ([_]u32{ 0, 1, 2 }) |mode| _ = try run(m, a, io, ids, 2, mode);
    }
    while (it.next()) |word| {
        const n = try std.fmt.parseInt(usize, word, 10);
        if (n + decode_steps + 1 > m.caches[0].plan.capacity) {
            log.err("n={d} + {d} steps does not fit the context ({d}): skipped", .{ n, decode_steps, m.caches[0].plan.capacity });
            continue;
        }
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const ids = try tokens(a, io, n);
        const full = try run(m, a, io, ids, decode_steps, 0);
        for ([_]u32{ 1, 2 }) |mode| try compare(m, n, mode, full, try run(m, a, io, ids, decode_steps, mode), decode_steps);
        const rb_full = try rollbackRun(m, a, io, ids, 0, 24);
        for ([_]u32{ 1, 2 }) |mode| {
            const rb = try rollbackRun(m, a, io, ids, mode, 24);
            log.info("n={d} mode {d}: rollback rounds digest full {x} cone {x} ({s}), position {d}/{d}", .{ n, mode, rb_full[0], rb[0], if (std.mem.eql(u8, &rb_full[0], &rb[0])) "equal" else "DIFFER", rb_full[1], rb[1] });
            if (!std.mem.eql(u8, &rb_full[0], &rb[0]) or rb_full[1] != rb[1]) return error.ConeRollbackMismatch;
        }
        // the second timing of each path, state warm
        const full2 = try run(m, a, io, ids, 1, 0);
        for ([_]u32{ 1, 2 }) |mode| {
            const cone2 = try run(m, a, io, ids, 1, mode);
            log.info("n={d} mode {d}: second prefill full {d:.2} s ({d:.0} tok/s), cone {d:.2} s ({d:.0} tok/s), x{d:.2}", .{
                n,
                mode,
                full2.prefill_s,
                @as(f64, @floatFromInt(n)) / full2.prefill_s,
                cone2.prefill_s,
                @as(f64, @floatFromInt(n)) / cone2.prefill_s,
                full2.prefill_s / cone2.prefill_s,
            });
        }
    }
    ds.forward_cone.force(m, 0);
    try m.reset();
    log.info("done", .{});
}
