//! Qwen2B Metal's stepped prompt pass (#3021) on host-backed caches with a sentinel forward: cuts, state, decodes.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const st = @import("families/qwen3_5/state.zig");
const snapshot = @import("families/qwen3_5/snapshot.zig");
const prefill = @import("families/qwen3_5/prefill.zig");
const be = lanes.backend;
const t = std.testing;
const a = t.allocator;

/// The cuts of main's whole loop (prefill.zig's run before #3021), copied as they were, `mark` carried.
fn mainCuts(total: usize, from: usize, cap: usize, planned: []const u32, kept_marks: []const u32) ![][2]usize {
    var out: std.ArrayList([2]usize) = .empty;
    var at = from;
    var mark: usize = 0;
    while (at < total) {
        var end = @min(total, at + cap);
        for (planned) |boundary| if (boundary > at) {
            end = @min(end, boundary);
            break;
        };
        while (mark < kept_marks.len and kept_marks[mark] <= at) mark += 1;
        if (mark < kept_marks.len) end = @min(end, kept_marks[mark]);
        try out.append(a, .{ at, end });
        at = end;
    }
    return out.toOwnedSlice(a);
}

/// A forward stand-in whose state depends on every token in order: the GDN recurrences, the conv rows and the logits.
fn sentinel(cache: *st.Cache, ids: []const u32) void {
    for (ids) |id| {
        for (cache.blocks, 0..) |block, layer| if (block == .delta) {
            const state = block.delta.recurrence.slice(u32, st.delta_bytes / 4);
            state[0] = state[0] *% 31 +% id +% @as(u32, @intCast(layer));
            state[state.len - 1] +%= state[0];
            const conv = block.delta.conv.slice(u16, st.conv_bytes / 2);
            conv[0] +%= @truncate(id);
        };
        const logits = cache.logits.slice(u32, 1);
        logits[0] = logits[0] *% 131 +% id;
        cache.len += 1;
    }
}

/// promptChunk's stand-in: each chunk logged by stream, then the sentinel forward.
const Rec = struct {
    gpa: std.mem.Allocator = a,
    log: *std.ArrayList(Event),
    id: u8,
    fn advance(r: Rec, cache: *st.Cache, ids: []const u32) !void {
        try r.log.append(a, .{ .chunk = .{ .id = r.id, .at = cache.len, .end = cache.len + ids.len } });
        sentinel(cache, ids);
    }
};

const Event = union(enum) { chunk: struct { id: u8, at: usize, end: usize }, decode: u8 };

fn seeded(device: mtl.Device, capacity: usize) !st.Cache {
    var cache = try st.Cache.init(a, device, capacity);
    @memset(cache.logits.contents()[0 .. 4 * 2], 0);
    for (cache.blocks) |block| if (block == .delta) {
        @memset(block.delta.recurrence.contents()[0..st.delta_bytes], 0);
        @memset(block.delta.conv.contents()[0..st.conv_bytes], 0);
    };
    return cache;
}

/// The bytes the sentinel writes, and the length, equal in two caches.
fn expectSame(x: *const st.Cache, y: *const st.Cache) !void {
    try t.expectEqual(x.len, y.len);
    try t.expectEqualSlices(u8, x.logits.contents()[0..4], y.logits.contents()[0..4]);
    for (x.blocks, y.blocks) |p, q| if (p == .delta) {
        try t.expectEqualSlices(u8, p.delta.recurrence.contents()[0..st.delta_bytes], q.delta.recurrence.contents()[0..st.delta_bytes]);
        try t.expectEqualSlices(u8, p.delta.conv.contents()[0..st.conv_bytes], q.delta.conv.contents()[0..st.conv_bytes]);
    };
}

fn cutsOf(log: []const Event, id: u8) ![][2]usize {
    var out: std.ArrayList([2]usize) = .empty;
    for (log) |e| if (e == .chunk and e.chunk.id == id) try out.append(a, .{ e.chunk.at, e.chunk.end });
    return out.toOwnedSlice(a);
}

/// A 29-token prompt at an 8-row chunk: planned starts at 3 and 20, a kept mark at 11.
const len = 29;
const rows = 8;
const chunks = [_]u32{ 3, 20 };
const marks = [_]u32{11};

fn prompt() [len]u32 {
    var ids: [len]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i * 17 % 101 + 1);
    return ids;
}

test "Qwen2B Metal steps: one call runs one chunk at today's cut, and the steps leave the whole pass's state" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    const ids = prompt();
    const want = try mainCuts(len, 0, rows, &chunks, &marks);
    defer a.free(want);
    try t.expectEqualSlices([2]usize, &.{ .{ 0, 3 }, .{ 3, 11 }, .{ 11, 19 }, .{ 19, 20 }, .{ 20, 28 }, .{ 28, 29 } }, want);
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(a);
    // the whole pass, as prefillFn runs it
    var whole = try seeded(device, 64);
    defer whole.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "w", .prompt = &ids, .max_new = 4, .chunks = &chunks, .reuse = .{ .marks = &marks } });
    defer s.deinit(a);
    try prefill.run(&whole, &s, rows, Rec{ .log = &log, .id = 'w' }, Rec.advance);
    // the same prompt a chunk a call, as prefillStepFn runs it
    var stepped = try seeded(device, 64);
    defer stepped.deinit();
    var u = try lanes.Stream.init(a, .{ .id = "s", .prompt = &ids, .max_new = 4, .chunks = &chunks, .reuse = .{ .marks = &marks } });
    defer u.deinit(a);
    var at = prefill.open(&stepped, &u);
    for (want, 0..) |w, i| {
        const before = log.items.len;
        try t.expectEqual(i + 1 == want.len, try prefill.step(&stepped, &u, &at, rows, Rec{ .log = &log, .id = 's' }, Rec.advance));
        try t.expectEqual(before + 1, log.items.len); // one chunk a call
        try t.expectEqual(@as(u32, @intCast(w[1])), u.fill_at.load(.monotonic));
    }
    const got_whole = try cutsOf(log.items, 'w');
    defer a.free(got_whole);
    const got_steps = try cutsOf(log.items, 's');
    defer a.free(got_steps);
    try t.expectEqualSlices([2]usize, want, got_whole);
    try t.expectEqualSlices([2]usize, want, got_steps);
    try t.expectEqualSlices(u32, s.prefill_widths.items, u.prefill_widths.items);
    try expectSame(&whole, &stepped);
    // a call after the last chunk has nothing left: done, and no chunk runs
    try t.expect(try prefill.step(&stepped, &u, &at, rows, Rec{ .log = &log, .id = 's' }, Rec.advance));
    try t.expectEqual(want.len, got_steps.len);
}

test "Qwen2B Metal steps: a pass resumed from a kept state cuts as main's loop does, and a cancel keeps the last chunk's end" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    const ids = prompt();
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(a);
    // the state after rows 0..11, kept
    var source = try seeded(device, 64);
    defer source.deinit();
    sentinel(&source, ids[0..11]);
    const snap = try snapshot.save(a, &source, 11);
    defer snapshot.drop(a, snap);
    var resumed = try seeded(device, 64);
    defer resumed.deinit();
    var s = try lanes.Stream.init(a, .{ .id = "r", .prompt = &ids, .max_new = 4, .chunks = &chunks, .reuse = .{ .saved = @ptrCast(snap), .at = 11, .marks = &marks } });
    defer s.deinit(a);
    var at = prefill.open(&resumed, &s);
    try t.expectEqual(@as(usize, 11), at);
    try t.expectEqual(@as(u32, 11), s.cached);
    while (!try prefill.step(&resumed, &s, &at, rows, Rec{ .log = &log, .id = 'r' }, Rec.advance)) {}
    const want = try mainCuts(len, 11, rows, &chunks, &marks);
    defer a.free(want);
    const got = try cutsOf(log.items, 'r');
    defer a.free(got);
    try t.expectEqualSlices([2]usize, want, got);
    var fresh = try seeded(device, 64);
    defer fresh.deinit();
    sentinel(&fresh, &ids);
    try expectSame(&fresh, &resumed);

    // a cancel asked before the fourth chunk: the rows in (0..19) are kept and no chunk runs
    const Flags = struct {
        var kept: ?u32 = null;
        var cancelled = false;
        fn keep(_: *anyopaque, _: *lanes.Stream, row: u32) void {
            kept = row;
        }
        fn check(_: *anyopaque) bool {
            return cancelled;
        }
    };
    var dummy: u8 = 0;
    var cut = try seeded(device, 64);
    defer cut.deinit();
    var c = try lanes.Stream.init(a, .{ .id = "c", .prompt = &ids, .max_new = 4, .chunks = &chunks, .reuse = .{ .marks = &marks }, .cancel_check = .{ .ptr = &dummy, .check = Flags.check } });
    defer c.deinit(a);
    c.reuse.hook = .{ .ptr = &dummy, .at = Flags.keep };
    var row = prefill.open(&cut, &c);
    for (0..3) |_| _ = try prefill.step(&cut, &c, &row, rows, Rec{ .log = &log, .id = 'c' }, Rec.advance);
    Flags.cancelled = true;
    Flags.kept = null;
    try t.expectError(error.Cancelled, prefill.step(&cut, &c, &row, rows, Rec{ .log = &log, .id = 'c' }, Rec.advance));
    try t.expectEqual(@as(?u32, 19), Flags.kept);
    try t.expectEqual(@as(usize, 19), cut.len);
    // cancel_prefill between calls keeps the same row (cancelPrefillFn: keepInterruptedPrefill at the fill's row)
    Flags.kept = null;
    c.keepInterruptedPrefill(@intCast(row));
    try t.expectEqual(@as(?u32, 19), Flags.kept);
}

/// The lanes fake with Qwen2B's stepping in front: each stream's host cache, its prompt a chunk a call at 4 rows.
const Q = struct {
    fake: lanes.fake.Fake,
    device: mtl.Device,
    inner: be.Backend = undefined,
    passes: std.AutoHashMapUnmanaged(*lanes.Stream, struct { cache: *st.Cache, at: usize }) = .empty,
    log: std.ArrayList(Event) = .empty,
    vt: be.Backend.VTable = undefined,

    var current: *Q = undefined;

    fn init(x: *Q) be.Backend {
        current = x;
        x.inner = x.fake.backend();
        x.vt = x.inner.vtable.*;
        x.vt.prefill_step = prefillStep;
        x.vt.reports_fill = true;
        x.vt.queue = queue;
        x.vt.verify = verify;
        return .{ .ptr = x.inner.ptr, .vtable = &x.vt };
    }

    fn deinit(x: *Q) void {
        var it = x.passes.valueIterator();
        while (it.next()) |p| {
            p.cache.deinit();
            a.destroy(p.cache);
        }
        x.passes.deinit(a);
        x.log.deinit(a);
        x.fake.deinit();
    }

    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const x = current;
        const got = try x.passes.getOrPut(a, s);
        if (!got.found_existing) {
            const cache = try a.create(st.Cache);
            cache.* = try seeded(x.device, 64);
            got.value_ptr.* = .{ .cache = cache, .at = prefill.open(cache, s) };
        }
        const p = got.value_ptr;
        const done = try prefill.step(p.cache, s, &p.at, 4, Rec{ .log = &x.log, .id = s.id[0] }, Rec.advance);
        if (done) try x.inner.vtable.prefill(ptr, s); // the fake's own state: its tokens follow the whole prompt
        return done;
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        try current.log.append(a, .{ .decode = s.id[0] });
        return current.inner.vtable.queue(ptr, s, feed, position);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        for (windows) |w| try current.log.append(a, .{ .decode = w.stream.id[0] });
        return current.inner.vtable.verify(ptr, windows, out);
    }
};

test "Qwen2B Metal steps: another stream decodes between two chunks, and the long prompt's state is the whole pass's" {
    var allocations: mtl.Allocations = .{};
    const device: mtl.Device = .{ .allocations = &allocations };
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*cost, w| cost.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.config.Config.init(a, .{ .exact_width = 16, .gpu_tokens = true, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8 }, 16, 15);
    defer cfg.deinit(a);
    var q: Q = .{ .fake = .{ .gpa = a }, .device = device };
    defer q.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(a, &cfg, q.init(), clock.clock());
    defer engine.deinit();
    try t.expect(engine.fills()); // the lane host steps this backend's prompts (lane_host_fill.zig)
    const short = [_]u32{ 5, 6, 7 };
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var x = try lanes.Stream.init(a, .{ .id = "a", .prompt = &short, .max_new = 20, .drafts = false });
    defer x.deinit(a);
    var y = try lanes.Stream.init(a, .{ .id = "b", .prompt = &long, .max_new = 4, .drafts = false });
    defer y.deinit(a);
    while (!try engine.fillStream(&x, q.fake.prefill_count == 0)) {}
    var first = true;
    while (!try engine.fillStream(&y, first)) {
        first = false;
        try engine.step(); // stream a's round between b's chunks
    }
    var seen: [3]usize = undefined;
    var n: usize = 0;
    for (q.log.items, 0..) |e, i| if (e == .chunk and e.chunk.id == 'b') {
        seen[n] = i;
        n += 1;
    };
    try t.expectEqual(@as(usize, 3), n);
    for (seen[0..2], seen[1..3]) |from, to| {
        const between = for (q.log.items[from..to]) |e| {
            if (e == .decode and e.decode == 'a') break true;
        } else false;
        try t.expect(between);
    }
    while (engine.activeCount() > 0) try engine.step();
    // b's cache after three stepped chunks with decodes between them: the whole prompt's state
    var whole = try seeded(device, 64);
    defer whole.deinit();
    sentinel(&whole, &long);
    try expectSame(&whole, q.passes.get(&y).?.cache);
}
