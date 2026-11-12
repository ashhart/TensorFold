//! Gemma 4 Metal's stepped prompt pass on a recording fake: today's cuts, a decode between chunks, every chunk's taps.
const std = @import("std");
const lanes = @import("lanes");
const steps = @import("prefill_steps.zig");
const be = lanes.backend;
const t = std.testing;
const gpa = t.allocator;

const Span = struct { at: usize, n: usize };
const Event = union(enum) { chunk: struct { id: u8, at: usize, end: usize }, keep: u32, decode: u8, absorb: struct { start: u64, rows: u32 } };

/// The cuts of main's whole loop (backend.zig before #3021), copied as they were: every stepped pass must equal them.
fn mainCuts(a: std.mem.Allocator, len: usize, from: usize, rows: usize, chunks: []const u32, marks: []const u32, spans: []const Span) ![][2]usize {
    var out: std.ArrayList([2]usize) = .empty;
    var at = from;
    while (at < len) {
        var end = @min(len, at + rows);
        for ([_][]const u32{ chunks, marks }) |cuts| for (cuts) |x| if (x > at and x < end) {
            end = x;
        };
        for (spans) |span| if (span.at < end and span.at + span.n > end) {
            end = if (span.at > at) span.at else span.at + span.n;
        };
        try out.append(a, .{ at, end });
        at = end;
    }
    return out.toOwnedSlice(a);
}

/// Records each chunk run and each kept mark, as Run does on the device.
const Rec = struct {
    log: *std.ArrayList(Event),
    pub fn run(r: Rec, s: *lanes.Stream, at: usize, end: usize) !void {
        try r.log.append(gpa, .{ .chunk = .{ .id = s.id[0], .at = at, .end = end } });
    }
    pub fn keep(r: Rec, _: *lanes.Stream, at: u32) void {
        r.log.append(gpa, .{ .keep = at }) catch @panic("out of memory");
    }
};

/// A 1,300-token prompt: a planned start at 700, a kept mark at 900, an image over the 512-row cut (480..580).
const prompt_len = 1300;
const planned = [_]u32{700};
const kept = [_]u32{900};
const images = [_]Span{.{ .at = 480, .n = 100 }};

fn stream(prompt: []const u32, id: []const u8) !lanes.Stream {
    return lanes.Stream.init(gpa, .{ .id = id, .prompt = prompt, .max_new = 4, .chunks = &planned, .reuse = .{ .marks = &kept } });
}

test "Gemma Metal steps: one call runs one chunk at today's cut, and the calls' chunks are the whole loop's" {
    var ids: [prompt_len]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i % 97 + 3);
    const want = try mainCuts(gpa, prompt_len, 0, 512, &planned, &kept, &images);
    defer gpa.free(want);
    try t.expectEqualSlices([2]usize, &.{ .{ 0, 480 }, .{ 480, 700 }, .{ 700, 900 }, .{ 900, 1300 } }, want);
    for ([_]bool{ false, true }) |later| { // the whole pass (Run with a sink) and a drafting stream's steps
        var s = try stream(&ids, "s");
        defer s.deinit(gpa);
        var log: std.ArrayList(Event) = .empty;
        defer log.deinit(gpa);
        var f: steps.Fill = .{ .at = 0 };
        for (want, 0..) |w, i| {
            const before = log.items.len;
            const done = try steps.step(Rec{ .log = &log }, &s, &f, 512, &images, later);
            try t.expectEqual(i + 1 == want.len, done);
            const ran = for (log.items[before..]) |e| {
                if (e == .chunk) break e.chunk;
            } else return error.NoChunk;
            try t.expectEqual(w, [2]usize{ ran.at, ran.end }); // exactly today's chunk
            var n: usize = 0;
            for (log.items[before..]) |e| n += @intFromBool(e == .chunk);
            try t.expectEqual(@as(usize, 1), n); // and only one a call
            try t.expectEqual(@as(u32, @intCast(w[1])), s.fill_at.load(.monotonic));
        }
        // the mark at 900 is kept before the chunk after it in both: with `later` at that next call's start
        try t.expectEqual(@as(usize, 5), log.items.len);
        try t.expectEqual(Event{ .keep = 900 }, log.items[3]);
    }
    // a resumed pass from the planned start at 700 cuts as the whole loop does from there
    const resumed = try mainCuts(gpa, prompt_len, 700, 512, &planned, &kept, &images);
    defer gpa.free(resumed);
    var s = try stream(&ids, "s");
    defer s.deinit(gpa);
    var log: std.ArrayList(Event) = .empty;
    defer log.deinit(gpa);
    var f: steps.Fill = .{ .at = 700 };
    while (!try steps.step(Rec{ .log = &log }, &s, &f, 512, &images, false)) {}
    var got: std.ArrayList([2]usize) = .empty;
    defer got.deinit(gpa);
    for (log.items) |e| if (e == .chunk) try got.append(gpa, .{ e.chunk.at, e.chunk.end });
    try t.expectEqualSlices([2]usize, resumed, got.items);
}

/// The lanes fake with Gemma's stepping in front: its prompt passes a chunk a call at `rows` (4 here), logged.
const Gem = struct {
    fake: lanes.fake.Fake,
    inner: be.Backend = undefined,
    fills: std.AutoHashMapUnmanaged(*lanes.Stream, steps.Fill) = .empty,
    log: std.ArrayList(Event) = .empty,
    held: std.AutoHashMapUnmanaged(*lanes.Stream, steps.Held) = .empty,
    prompt_gen: u64 = 0,
    vt: be.Backend.VTable = undefined,

    var current: *Gem = undefined;
    const rows = 4;

    fn init(x: *Gem) be.Backend {
        current = x;
        x.inner = x.fake.backend();
        x.vt = x.inner.vtable.*;
        x.vt.prefill_step = prefillStep;
        x.vt.reports_fill = true;
        x.vt.queue = queue;
        x.vt.verify = verify;
        x.vt.features = features;
        return .{ .ptr = x.inner.ptr, .vtable = &x.vt };
    }

    fn deinit(x: *Gem) void {
        x.fills.deinit(gpa);
        x.held.deinit(gpa);
        x.log.deinit(gpa);
        x.fake.deinit();
    }

    const Run = struct {
        pub fn run(_: Run, s: *lanes.Stream, at: usize, end: usize) !void {
            const x = current;
            x.prompt_gen += 1;
            try x.held.put(gpa, s, .{ .start = at, .rows = end - at, .gen = x.prompt_gen }); // the scratch's rows
            try x.log.append(gpa, .{ .chunk = .{ .id = s.id[0], .at = at, .end = end } });
        }
        pub fn keep(_: Run, _: *lanes.Stream, at: u32) void {
            current.log.append(gpa, .{ .keep = at }) catch @panic("out of memory");
        }
    };

    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const x = current;
        const got = try x.fills.getOrPut(gpa, s);
        if (!got.found_existing) got.value_ptr.* = .{ .at = 0 };
        const done = try steps.step(Run{}, s, got.value_ptr, rows, &[_]Span{}, s.drafts);
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

    /// featuresFn's rule: the latest chunk's rows, while no later chunk has rewritten the scratch.
    fn features(_: *anyopaque, s: *lanes.Stream, _: []const u32, start: u64, count: u32) anyerror!be.Features {
        const x = current;
        const h = x.held.get(s) orelse return error.GemmaTapsNotHeld;
        if (!steps.holds(h, x.prompt_gen, start, count)) return error.GemmaTapsNotHeld;
        return .{ .buffer = h.start, .rows = count, .row_bytes = 2, .space = .host, .dtype = .bf16 };
    }
};

test "Gemma Metal steps: another stream decodes between two chunks of a long prompt" {
    var costs: [16]lanes.config.Cost = undefined;
    for (&costs, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 5.0 + 0.8 * @as(f64, @floatFromInt(w)) };
    var cfg = try lanes.config.Config.init(gpa, .{ .exact_width = 16, .gpu_tokens = true, .window_costs = &costs, .hidden_rows = true, .batch_rows = 32, .max_streams = 8 }, 16, 15);
    defer cfg.deinit(gpa);
    var gem: Gem = .{ .fake = .{ .gpa = gpa } };
    defer gem.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, gem.init(), clock.clock());
    defer engine.deinit();
    try t.expect(engine.fills()); // the lane host steps this backend's prompts (lane_host_fill.zig), as it steps CUDA's
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
    // b's three chunks, each followed by a's decode before the next
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
    while (engine.activeCount() > 0) try engine.step();
}

/// A drafter that records the rows it absorbs and the chunk each came from (Features.buffer: the chunk's start).
const Head = struct {
    log: *std.ArrayList(Event),
    fn drafter(x: *Head) lanes.drafter.Drafter {
        return .{ .ptr = x, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release } };
    }
    fn taps(_: *anyopaque) []const u32 {
        return &.{ 3, 7 };
    }
    fn facts(_: *anyopaque) lanes.drafter.Facts {
        return .{ .depth = 2, .step_ms = 0.5 };
    }
    fn open(_: *anyopaque, _: *lanes.Stream) anyerror!void {}
    fn absorb(ptr: *anyopaque, items: []const lanes.drafter.Absorb) anyerror!void {
        const x: *Head = @ptrCast(@alignCast(ptr));
        for (items) |it| {
            try t.expectEqual(it.start, it.features.buffer); // the rows came from the chunk that starts there
            try x.log.append(gpa, .{ .absorb = .{ .start = it.start, .rows = it.features.rows } });
        }
    }
    fn hold(_: *anyopaque, _: []const lanes.drafter.Hold) anyerror!void {}
    fn held(_: *anyopaque, _: []const *lanes.Stream, out: []const []u32) anyerror!void {
        for (out) |o| @memset(o, 0);
    }
    fn release(_: *anyopaque, _: *lanes.Stream) void {}
};

test "Gemma Metal steps: a drafter absorbs every chunk's tapped rows, and a mark keeps after its chunk's absorb" {
    var gem: Gem = .{ .fake = .{ .gpa = gpa } };
    defer gem.deinit();
    var head: Head = .{ .log = &gem.log };
    var wrapped = try lanes.drafted.Drafted.init(gpa, gem.init(), head.drafter());
    defer wrapped.deinit();
    const backend = wrapped.backend();
    try t.expect(backend.vtable.prefill_step != null); // the wrapper steps: the target steps, reports fill, serves taps
    const long = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var s = try lanes.Stream.init(gpa, .{ .id = "d", .prompt = &long, .max_new = 4, .reuse = .{ .marks = &.{8} } });
    defer s.deinit(gpa);
    defer backend.vtable.release(backend.ptr, &s);
    while (!try backend.vtable.prefill_step.?(backend.ptr, &s, 0)) {}
    const want = [_]Event{
        .{ .chunk = .{ .id = 'd', .at = 0, .end = 4 } }, .{ .absorb = .{ .start = 0, .rows = 4 } },
        .{ .chunk = .{ .id = 'd', .at = 4, .end = 8 } }, .{ .absorb = .{ .start = 4, .rows = 4 } },
        .{ .keep = 8 }, // the state at 8 is kept once the drafter holds rows 4..8: its context comes with it
        .{ .chunk = .{ .id = 'd', .at = 8, .end = 12 } }, .{ .absorb = .{ .start = 8, .rows = 3 } }, // the last: later
    };
    try t.expectEqual(want.len, gem.log.items.len);
    for (want, gem.log.items) |w, g| try t.expectEqual(w, g);
}
