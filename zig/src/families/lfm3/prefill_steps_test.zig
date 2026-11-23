//! LFM3 Metal's stepped prompt pass on recording fakes: today's cuts, a decode between chunks, cancel and yield.
const std = @import("std");
const lanes = @import("lanes");
const steps = @import("prefill_steps.zig");
const be = lanes.backend;
const t = std.testing;
const gpa = t.allocator;

const Event = union(enum) { chunk: struct { id: u8, at: usize, end: usize }, keep: u32, decode: u8 };

/// The cuts of main's whole loop (backend.zig's prefillFn before #3021), copied as they were, `mark` carried.
fn mainCuts(a: std.mem.Allocator, total: usize, from: usize, rows: usize, planned: []const u32, kept_marks: []const u32) ![][2]usize {
    var out: std.ArrayList([2]usize) = .empty;
    var at = from;
    var mark: usize = 0;
    while (at < total) {
        var end: usize = @min(total, at + rows);
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

/// Records each chunk run and each kept mark, as backend.zig's Run does on the device.
const Rec = struct {
    log: *std.ArrayList(Event),
    pub fn run(r: Rec, s: *lanes.Stream, at: usize, end: usize) !void {
        try r.log.append(gpa, .{ .chunk = .{ .id = s.id[0], .at = at, .end = end } });
    }
    pub fn keep(r: Rec, _: *lanes.Stream, at: u32) void {
        r.log.append(gpa, .{ .keep = at }) catch @panic("out of memory");
    }
};

/// A 29-token prompt at an 8-row cap: planned starts at 3 and 20, a kept mark at 11.
const len = 29;
const cap = 8;
const chunks = [_]u32{ 3, 20 };
const marks = [_]u32{11};

fn prompt() [len]u32 {
    var ids: [len]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i % 97 + 3);
    return ids;
}

/// The row a stream's reuse hook last kept, and whether its cancel or yield check answers yes.
const Flags = struct {
    var kept: ?u32 = null;
    var cancelled = false;
    var yielding = false;
    fn at(_: *anyopaque, _: *lanes.Stream, row: u32) void {
        kept = row;
    }
    fn cancel(_: *anyopaque) bool {
        return cancelled;
    }
    fn yields(_: *anyopaque) bool {
        return yielding;
    }
    fn reset() void {
        kept = null;
        cancelled = false;
        yielding = false;
    }
};
var dummy: u8 = 0;

fn checked(ids: []const u32) !lanes.Stream {
    var s = try lanes.Stream.init(gpa, .{ .id = "x", .prompt = ids, .max_new = 4, .chunks = &chunks, .reuse = .{ .marks = &marks }, .cancel_check = .{ .ptr = &dummy, .check = Flags.cancel }, .yield_check = .{ .ptr = &dummy, .check = Flags.yields } });
    s.reuse.hook = .{ .ptr = &dummy, .at = Flags.at };
    return s;
}

test "LFM3 Metal steps: one call runs one chunk at today's cut, and the calls' chunks are the whole loop's" {
    const ids = prompt();
    const want = try mainCuts(gpa, len, 0, cap, &chunks, &marks);
    defer gpa.free(want);
    try t.expectEqualSlices([2]usize, &.{ .{ 0, 3 }, .{ 3, 11 }, .{ 11, 19 }, .{ 19, 20 }, .{ 20, 28 }, .{ 28, 29 } }, want);
    for ([_]bool{ true, false }) |whole| { // the whole pass (prefillFn) and the lane host's steps
        var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &ids, .max_new = 4, .chunks = &chunks, .reuse = .{ .marks = &marks } });
        defer s.deinit(gpa);
        var log: std.ArrayList(Event) = .empty;
        defer log.deinit(gpa);
        var at: usize = 0;
        for (want, 0..) |w, i| {
            const before = log.items.len;
            const done = try steps.step(Rec{ .log = &log }, &s, &at, cap, whole);
            try t.expectEqual(i + 1 == want.len, done);
            try t.expectEqual(Event{ .chunk = .{ .id = 's', .at = w[0], .end = w[1] } }, log.items[before]); // today's chunk
            var n: usize = 0;
            for (log.items[before..]) |e| n += @intFromBool(e == .chunk);
            try t.expectEqual(@as(usize, 1), n); // and only one a call
            try t.expectEqual(@as(u32, @intCast(w[1])), s.fill_at.load(.monotonic));
        }
        // the mark at 11 is kept right after the chunk that reaches it, as main's loop kept it
        try t.expectEqual(@as(usize, 7), log.items.len);
        try t.expectEqual(Event{ .keep = 11 }, log.items[2]);
    }
    // a pass resumed from a kept state at 11 cuts as main's loop does from there
    const resumed = try mainCuts(gpa, len, 11, cap, &chunks, &marks);
    defer gpa.free(resumed);
    var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &ids, .max_new = 4, .chunks = &chunks, .reuse = .{ .marks = &marks } });
    defer s.deinit(gpa);
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var at: usize = 11;
    while (!try steps.step(Rec{ .log = &log }, &s, &at, cap, false)) {}
    var got: std.ArrayList([2]usize) = .empty;
    defer got.deinit(gpa);
    for (log.items) |e| if (e == .chunk) try got.append(gpa, .{ e.chunk.at, e.chunk.end });
    try t.expectEqualSlices([2]usize, resumed, got.items);
}

test "LFM3 Metal steps: a cancel between chunks keeps the state at the last chunk's end" {
    const ids = prompt();
    Flags.reset();
    defer Flags.reset();
    var s = try checked(&ids);
    defer s.deinit(gpa);
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var at: usize = 0;
    for (0..3) |_| _ = try steps.step(Rec{ .log = &log }, &s, &at, cap, false);
    Flags.kept = null;
    steps.cancel(&s, at); // the host's cancel_prefill
    try t.expectEqual(@as(?u32, 19), Flags.kept); // rows 0..19 are in; no chunk ran for the cancel
    Flags.kept = null;
    steps.cancel(&s, null); // no pass in progress: nothing kept
    try t.expectEqual(@as(?u32, null), Flags.kept);
    // a step asked after the cancel keeps the same rows and runs nothing, as prefill_control.before does
    Flags.cancelled = true;
    try t.expectError(error.Cancelled, steps.step(Rec{ .log = &log }, &s, &at, cap, false));
    try t.expectEqual(@as(?u32, 19), Flags.kept);
    var n: usize = 0;
    for (log.items) |e| n += @intFromBool(e == .chunk);
    try t.expectEqual(@as(usize, 3), n);
}

test "LFM3 Metal steps: the whole pass still yields at a chunk start with its prefix kept; a step leaves it to the host" {
    const ids = prompt();
    Flags.reset();
    defer Flags.reset();
    var s = try checked(&ids);
    defer s.deinit(gpa);
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var at: usize = 0;
    for (0..3) |_| _ = try steps.step(Rec{ .log = &log }, &s, &at, cap, true);
    Flags.yielding = true;
    Flags.kept = null;
    try t.expectError(error.Yielded, steps.step(Rec{ .log = &log }, &s, &at, cap, true));
    try t.expectEqual(@as(?u32, 19), Flags.kept); // prefill_control.before kept the prefix, as main's loop did
    try t.expectEqual(@as(usize, 19), at);
    // the lane host schedules a stepped pass itself: a yield check does not stop the next chunk
    try t.expect(!try steps.step(Rec{ .log = &log }, &s, &at, cap, false));
    try t.expectEqual(@as(usize, 20), at);
}

/// The lanes fake with LFM3's stepping in front: its prompt passes a chunk a call at `rows` (4 here), logged.
const Lfm = struct {
    fake: lanes.fake.Fake,
    inner: be.Backend = undefined,
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, usize) = .empty,
    log: std.ArrayList(Event) = .empty,
    vt: be.Backend.VTable = undefined,

    var current: *Lfm = undefined;
    const rows = 4;

    fn init(x: *Lfm) be.Backend {
        current = x;
        x.inner = x.fake.backend();
        x.vt = x.inner.vtable.*;
        x.vt.prefill_step = prefillStep;
        x.vt.reports_fill = true;
        x.vt.queue = queue;
        x.vt.verify = verify;
        return .{ .ptr = x.inner.ptr, .vtable = &x.vt };
    }

    fn deinit(x: *Lfm) void {
        x.fills.deinit(gpa);
        x.log.deinit(gpa);
        x.fake.deinit();
    }

    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const x = current;
        const got = try x.fills.getOrPut(gpa, s);
        if (!got.found_existing) got.value_ptr.* = 0;
        const done = try steps.step(Rec{ .log = &x.log }, s, got.value_ptr, rows, false);
        if (done) {
            _ = x.fills.remove(s);
            try x.inner.vtable.prefill(ptr, s); // the fake's own state: its tokens follow the whole prompt
        }
        return done;
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        try current.log.append(gpa, .{ .decode = s.id[0] });
        return current.inner.vtable.queue(ptr, s, feed, position);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        for (windows) |w| try current.log.append(gpa, .{ .decode = w.stream.id[0] });
        return current.inner.vtable.verify(ptr, windows, out);
    }
};

test "LFM3 Metal steps: another stream decodes between two chunks of a long prompt" {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.config.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8 }, 16, 15);
    defer cfg.deinit(gpa);
    var lfm: Lfm = .{ .fake = .{ .gpa = gpa } };
    defer lfm.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, lfm.init(), clock.clock());
    defer engine.deinit();
    try t.expect(engine.fills()); // the lane host steps this backend's prompts (lane_host_fill.zig)
    const short = [_]u32{ 5, 6, 7 };
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &short, .max_new = 20, .drafts = false });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &long, .max_new = 4, .drafts = false });
    defer b.deinit(gpa);
    while (!try engine.fillStream(&a, lfm.fake.prefill_count == 0)) {}
    var first = true;
    while (!try engine.fillStream(&b, first)) {
        first = false;
        try engine.step(); // stream a's round between b's chunks
    }
    // b's three chunks, each followed by a's decode before the next
    var seen: [3]usize = undefined;
    var n: usize = 0;
    for (lfm.log.items, 0..) |e, i| if (e == .chunk and e.chunk.id == 'b') {
        seen[n] = i;
        n += 1;
    };
    try t.expectEqual(@as(usize, 3), n);
    for (seen[0..2], seen[1..3]) |from, to| {
        const between = for (lfm.log.items[from..to]) |e| {
            if (e == .decode and e.decode == 'a') break true;
        } else false;
        try t.expect(between);
    }
    while (engine.activeCount() > 0) try engine.step();
}
