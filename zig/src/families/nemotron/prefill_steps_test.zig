//! Nemotron Metal's stepped prompt pass on recording fakes: today's cuts, a decode between chunks, the head's rows.
const std = @import("std");
const lanes = @import("lanes");
const pass = @import("prefill_steps.zig");
const flight = @import("prompt_flight.zig");
const be = lanes.backend;
const t = std.testing;
const gpa = t.allocator;

const Kind = enum(u2) { prefill, step, verify, draft };
const Buffer = struct {
    pub fn wait(_: Buffer) void {}
    pub fn release(_: Buffer) void {}
    pub fn failure(_: Buffer) ?[]const u8 {
        return null;
    }
    pub fn gpuSeconds(_: Buffer) f64 {
        return 0.002;
    }
};
const Flight = struct { end: u64, cb: Buffer, kind: Kind, prompt: ?flight.Prompt = null };

/// The flight owner prompt_flight.zig drives, as prompt_flight_test.zig's: submissions queue, `land` waits.
const Queue = struct {
    gpa: std.mem.Allocator = gpa,
    next: u64 = 1,
    landed: u64 = 0,
    last_ms: f64 = 0,
    gpu_ms: f64 = 0,
    timing: struct { ms: [4]f64 = @splat(0), count: [4]usize = @splat(0) } = .{},
    flights: std.ArrayList(Flight) = .empty,
    drains: usize = 0,
    pub fn submit(q: *Queue, kind: Kind, end: u64, _: anytype) !void {
        try q.flights.append(q.gpa, .{ .end = end, .kind = kind, .cb = .{} });
    }
    pub fn land(q: *Queue) !void {
        try flight.land(q);
    }
    pub fn drain(q: *Queue) !void {
        q.drains += 1;
        while (q.flights.items.len > 0) try q.land();
    }
    fn deinit(q: *Queue) void {
        q.flights.deinit(q.gpa);
    }
};

const Event = union(enum) { chunk: struct { id: u8, at: usize, rows: usize }, load: u8, decode: u8, keep: u32 };

/// The device side as prefill_metal.zig's Ops: each chunk's rows and the head's absorbed rows recorded.
const Ops = struct {
    log: *std.ArrayList(Event),
    map: *pass.Steps,
    queue: *Queue,
    cap_rows: usize = 8,
    decide: ?usize = null,
    head: ?*std.ArrayList([2]usize) = null, // the MTP head's absorbed rows, [start, end) a chunk
    loads_on_empty: *bool, // every load found the queue landed (the shared prompt buffer free)

    pub fn steps(o: Ops) *pass.Steps {
        return o.map;
    }
    pub fn cap(o: Ops) usize {
        return o.cap_rows;
    }
    pub fn decision(o: Ops, _: *lanes.Stream) ?usize {
        return o.decide;
    }
    pub fn grow(_: Ops, _: *lanes.Stream, _: usize) !void {}
    pub fn body(o: Ops, s: *lanes.Stream, at: usize, rows: usize) struct { rows: usize } {
        o.log.append(gpa, .{ .chunk = .{ .id = s.id[0], .at = at, .rows = rows } }) catch @panic("out of memory");
        if (o.head) |h| h.append(gpa, .{ at, at + pass.absorbed(at, rows, s.prompt_len) }) catch @panic("out of memory");
        return .{ .rows = rows };
    }
    pub fn load(o: Ops, s: *lanes.Stream) void {
        if (o.queue.flights.items.len != 0) o.loads_on_empty.* = false;
        o.map.ids = s;
        o.log.append(gpa, .{ .load = s.id[0] }) catch @panic("out of memory");
    }
    pub fn open(o: Ops, s: *lanes.Stream) !usize {
        if (s.isCancelled()) return error.Cancelled;
        try o.queue.drain();
        o.load(s);
        return 0;
    }
    pub fn finish(_: Ops, _: *lanes.Stream) !void {}
};

/// main's cuts for the 29-token prompt: starts 3 and 20, decision row 12, cap 8 (as prompt_flight_test.zig).
const want = [_][2]usize{ .{ 0, 3 }, .{ 3, 8 }, .{ 11, 2 }, .{ 13, 7 }, .{ 20, 8 }, .{ 28, 1 } };

fn chunksOf(log: []const Event) ![][2]usize {
    var out: std.ArrayList([2]usize) = .empty;
    for (log) |e| if (e == .chunk) try out.append(gpa, .{ e.chunk.at, e.chunk.rows });
    return out.toOwnedSlice(gpa);
}

test "Nemotron Metal steps: one call submits one chunk at today's cut, and the calls' chunks are the whole pass's" {
    const ids: [29]u32 = @splat(7);
    for ([_]bool{ false, true }) |stepped| {
        var s = try lanes.Stream.init(gpa, .{ .id = "s", .prompt = &ids, .max_new = 1, .chunks = &.{ 3, 20 } });
        defer s.deinit(gpa);
        var q: Queue = .{};
        defer q.deinit();
        var log: std.ArrayList(Event) = .empty;
        defer log.deinit(gpa);
        var map: pass.Steps = .{};
        defer map.at.deinit(gpa);
        var empty = true;
        const ops: Ops = .{ .log = &log, .map = &map, .queue = &q, .decide = 12, .loads_on_empty = &empty };
        if (stepped) {
            for (want, 0..) |w, i| {
                const before = log.items.len;
                try t.expectEqual(i + 1 == want.len, try pass.step(&q, ops, &s));
                const got = try chunksOf(log.items[before..]);
                defer gpa.free(got);
                try t.expectEqualSlices([2]usize, &.{w}, got); // exactly one chunk, today's
                try t.expectEqual(@as(u32, @intCast(w[0])), s.fill_at.load(.monotonic)); // landed rows only
            }
            try t.expect(empty); // each step loaded this prompt's ids only once the queue had landed
            try t.expectEqual(@as(usize, 0), map.at.count());
        } else try pass.whole(&q, ops, &s);
        const got = try chunksOf(log.items);
        defer gpa.free(got);
        try t.expectEqualSlices([2]usize, &want, got);
        try q.drain();
        try t.expectEqualSlices(u32, &.{ 3, 8, 2, 7, 8, 1 }, s.prefill_widths.items);
    }
}

test "Nemotron Metal steps: the MTP head absorbs every prompt row but the last, once, chunk by chunk" {
    const ids: [29]u32 = @splat(7);
    var s = try lanes.Stream.init(gpa, .{ .id = "h", .prompt = &ids, .max_new = 1, .chunks = &.{ 3, 20 } });
    defer s.deinit(gpa);
    var q: Queue = .{};
    defer q.deinit();
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var map: pass.Steps = .{};
    defer map.at.deinit(gpa);
    var head: std.ArrayList([2]usize) = .empty;
    defer head.deinit(gpa);
    var empty = true;
    const ops: Ops = .{ .log = &log, .map = &map, .queue = &q, .decide = 12, .head = &head, .loads_on_empty = &empty };
    while (!try pass.step(&q, ops, &s)) {}
    // each chunk's own encode absorbs its rows: contiguous, 0..28 (row 28's next token is the first draw)
    var at: usize = 0;
    for (head.items) |h| {
        try t.expectEqual(at, h[0]);
        at = h[1];
    }
    try t.expectEqual(@as(usize, 28), at);
    try t.expectEqual(want.len, head.items.len);
}

const Kept = struct {
    log: *std.ArrayList(Event),
    published: ?u32 = null,
    fn at(ptr: *anyopaque, s: *lanes.Stream, row: u32) void {
        const k: *Kept = @ptrCast(@alignCast(ptr));
        k.published = s.fill_at.load(.monotonic);
        k.log.append(gpa, .{ .keep = row }) catch @panic("out of memory");
    }
};

test "Nemotron Metal steps: a cancel between chunks lands the queue and keeps the state at the last chunk's end" {
    const ids: [29]u32 = @splat(7);
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var kept: Kept = .{ .log = &log };
    var s = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = &ids, .max_new = 1, .chunks = &.{ 3, 20 }, .reuse = .{ .hook = .{ .ptr = &kept, .at = Kept.at } } });
    defer s.deinit(gpa);
    var q: Queue = .{};
    defer q.deinit();
    var map: pass.Steps = .{};
    defer map.at.deinit(gpa);
    var empty = true;
    const ops: Ops = .{ .log = &log, .map = &map, .queue = &q, .decide = 12, .loads_on_empty = &empty };
    for (0..2) |_| try t.expect(!try pass.step(&q, ops, &s));
    try t.expectEqual(@as(usize, 1), q.flights.items.len); // the second chunk still queued
    try pass.cancel(&q, ops, &s);
    try t.expectEqual(@as(usize, 0), q.flights.items.len);
    try t.expectEqual(Event{ .keep = 11 }, log.items[log.items.len - 1]);
    try t.expectEqual(@as(?u32, 11), kept.published);
    try t.expectEqual(@as(u32, 11), s.fill_at.load(.monotonic));
}

test "Nemotron Metal steps: consecutive chunks retain the prompt ids and one queued chunk" {
    const ids: [29]u32 = @splat(7);
    var s = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &ids, .max_new = 1 });
    defer s.deinit(gpa);
    var q: Queue = .{};
    defer q.deinit();
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var map: pass.Steps = .{};
    defer map.at.deinit(gpa);
    var empty = true;
    const ops: Ops = .{ .log = &log, .map = &map, .queue = &q, .loads_on_empty = &empty };
    try t.expect(!try pass.step(&q, ops, &s));
    const drains = q.drains;
    for (0..2) |_| {
        try t.expect(!try pass.step(&q, ops, &s));
        try t.expectEqual(drains, q.drains);
        try t.expectEqual(@as(usize, 1), q.flights.items.len);
    }
    var loads: usize = 0;
    for (log.items) |e| if (e == .load) {
        loads += 1;
    };
    try t.expectEqual(@as(usize, 1), loads);
    try t.expectEqualSlices(u32, &.{ 8, 8 }, s.prefill_widths.items);
    try t.expect(empty);
}

test "Nemotron Metal steps: changing the prompt owner drains before each id reload" {
    const ids: [29]u32 = @splat(7);
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &ids, .max_new = 1 });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &ids, .max_new = 1 });
    defer b.deinit(gpa);
    var q: Queue = .{};
    defer q.deinit();
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var map: pass.Steps = .{};
    defer map.at.deinit(gpa);
    var empty = true;
    const ops: Ops = .{ .log = &log, .map = &map, .queue = &q, .loads_on_empty = &empty };
    for ([_]*lanes.Stream{ &a, &a, &b, &a }) |s| try t.expect(!try pass.step(&q, ops, s));
    var loaded: std.ArrayList(u8) = .empty;
    defer loaded.deinit(gpa);
    for (log.items) |e| if (e == .load) {
        try loaded.append(gpa, e.load);
    };
    try t.expectEqualSlices(u8, "aba", loaded.items);
    try t.expectEqual(@as(usize, 3), q.drains);
    try t.expect(empty);
}

test "Nemotron Metal steps: cancellation inside a step publishes and keeps its landed tail" {
    const Stop = struct {
        fn yes(_: *anyopaque) bool {
            return true;
        }
    };
    const ids: [29]u32 = @splat(7);
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var kept: Kept = .{ .log = &log };
    var s = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = &ids, .max_new = 1, .reuse = .{ .hook = .{ .ptr = &kept, .at = Kept.at } } });
    defer s.deinit(gpa);
    var q: Queue = .{};
    defer q.deinit();
    var map: pass.Steps = .{};
    defer map.at.deinit(gpa);
    var empty = true;
    const ops: Ops = .{ .log = &log, .map = &map, .queue = &q, .loads_on_empty = &empty };
    for (0..2) |_| try t.expect(!try pass.step(&q, ops, &s));
    var stop: u8 = 0;
    s.cancel_check = .{ .ptr = &stop, .check = Stop.yes };
    try t.expectError(error.Cancelled, pass.step(&q, ops, &s));
    try t.expectEqual(@as(?u32, 16), kept.published);
    try t.expectEqual(@as(u32, 16), s.fill_at.load(.monotonic));
    try t.expectEqual(@as(usize, 0), q.flights.items.len);
    try t.expectEqual(@as(usize, 0), map.at.count());
    map.forget(&s);
    try t.expectEqual(@as(?*lanes.Stream, null), map.ids);
}

/// The lanes fake with Nemotron's stepping in front (cap 4), every chunk, load and decode logged.
const Gem = struct {
    fake: lanes.fake.Fake,
    inner: be.Backend = undefined,
    vt: be.Backend.VTable = undefined,
    q: Queue = .{},
    map: pass.Steps = .{},
    log: std.ArrayList(Event) = .empty,
    empty: bool = true,

    var current: *Gem = undefined;

    fn init(x: *Gem) be.Backend {
        current = x;
        x.inner = x.fake.backend();
        x.vt = x.inner.vtable.*;
        x.vt.prefill_step = prefillStep;
        x.vt.reports_fill = true;
        x.vt.queue = queue;
        x.vt.verify = verify;
        return .{ .ptr = x.inner.ptr, .vtable = &x.vt };
    }
    fn deinit(x: *Gem) void {
        x.q.deinit();
        x.map.at.deinit(gpa);
        x.log.deinit(gpa);
        x.fake.deinit();
    }
    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const x = current;
        const done = try pass.step(&x.q, Ops{ .log = &x.log, .map = &x.map, .queue = &x.q, .cap_rows = 4, .loads_on_empty = &x.empty }, s);
        if (done) {
            try x.q.drain();
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

test "Nemotron Metal steps: another stream decodes between two chunks of a long prompt" {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.config.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8 }, 16, 15);
    defer cfg.deinit(gpa);
    var gem: Gem = .{ .fake = .{ .gpa = gpa } };
    defer gem.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, gem.init(), clock.clock());
    defer engine.deinit();
    try t.expect(engine.fills()); // the lane host now steps this backend's prompts (lane_host_fill.zig)
    const short = [_]u32{ 5, 6, 7 };
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &short, .max_new = 20, .drafts = false });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &long, .max_new = 4, .drafts = false });
    defer b.deinit(gpa);
    while (!try engine.fillStream(&a, gem.fake.prefill_count == 0)) {}
    var first = true;
    while (!try engine.fillStream(&b, first)) {
        first = false;
        try engine.step(); // stream a's round between b's chunks
    }
    var seen: [3]usize = undefined;
    var n: usize = 0;
    for (gem.log.items, 0..) |e, i| if (e == .chunk and e.chunk.id == 'b') {
        seen[n] = i;
        n += 1;
    };
    try t.expectEqual(@as(usize, 3), n);
    for (seen[0..2], seen[1..3]) |from, to| {
        const between = for (gem.log.items[from..to]) |e| {
            if (e == .decode and e.decode == 'a') break true;
        } else false;
        try t.expect(between);
    }
    try t.expect(gem.empty); // no chunk's ids went into the prompt buffer while another chunk was queued
    while (engine.activeCount() > 0) try engine.step();
}
