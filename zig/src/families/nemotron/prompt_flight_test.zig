//! Actual Nemotron cuts, lookahead and completion accounting, with recording command buffers and no GPU.
const std = @import("std");
const lanes = @import("lanes");
const workflow = @import("prompt_flight.zig");
const a = std.testing.allocator;
const t = std.testing;
const Kind = enum(u2) { prefill, step, verify, draft };
const State = struct { waits: usize = 0, releases: usize = 0 };
const Buffer = struct {
    state: *State,
    pub fn wait(b: Buffer) void {
        b.state.waits += 1;
    }
    pub fn release(b: Buffer) void {
        b.state.releases += 1;
    }
    pub fn failure(_: Buffer) ?[]const u8 {
        return null;
    }
    pub fn gpuSeconds(_: Buffer) f64 {
        return 0.002;
    }
};
const Flight = struct { end: u64, cb: Buffer, kind: Kind, prompt: ?workflow.Prompt = null };
const Fake = struct {
    gpa: std.mem.Allocator = a,
    next: u64 = 1,
    landed: u64 = 0,
    last_ms: f64 = 0,
    gpu_ms: f64 = 0,
    timing: struct { ms: [4]f64 = @splat(0), count: [4]usize = @splat(0) } = .{},
    states: [32]State = @splat(.{}),
    submitted: usize = 0,
    flights: std.ArrayList(Flight) = .empty,
    fail_submit: bool = false,
    pub fn submit(f: *Fake, kind: Kind, end: u64, _: anytype) !void {
        if (f.fail_submit) return error.SubmitFailed;
        try f.flights.append(f.gpa, .{ .end = end, .kind = kind, .cb = .{ .state = &f.states[f.submitted] } });
        f.submitted += 1;
    }
    pub fn land(f: *Fake) !void {
        try workflow.land(f);
    }
    fn drain(f: *Fake) !void {
        while (f.flights.items.len > 0) try f.land();
    }
    fn deinit(f: *Fake) void {
        f.flights.deinit(f.gpa);
    }
};

fn fill(f: *Fake, s: *lanes.Stream, cap: usize, decision: ?usize) !void {
    var at: usize = 0;
    var next: usize = 0;
    while (at < s.prompt_len) {
        try workflow.checkpoint(s);
        const n = workflow.rows(s, at, cap, decision, &next);
        try workflow.submit(f, s, .{ .rows = n });
        at += n;
    }
}

test "Nemotron prefill widths: foreground work interrupts a warm pass with its queued flight still owned" {
    const Stop = struct {
        f: *Fake,
        fn ready(ptr: *anyopaque) bool {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            return p.f.submitted >= 2;
        }
    };
    const ids: [24]u32 = @splat(7);
    var f: Fake = .{};
    defer f.deinit();
    var stop: Stop = .{ .f = &f };
    var s = try lanes.Stream.init(a, .{ .id = "warm", .prompt = &ids, .max_new = 0, .yield_check = .{ .ptr = &stop, .check = Stop.ready } });
    defer s.deinit(a);
    try t.expectError(error.Yielded, fill(&f, &s, 8, null));
    try t.expectEqual(@as(usize, 2), f.submitted);
    try t.expectEqualSlices(u32, &.{8}, s.prefill_widths.items);
    try t.expectEqual(@as(usize, 1), f.flights.items.len);
    try t.expect(f.flights.items[0].prompt.?.stream == &s);
    try f.drain();
    try t.expectEqualSlices(u32, &.{ 8, 8 }, s.prefill_widths.items);
    for (f.states[0..2]) |cb| {
        try t.expectEqual(@as(usize, 1), cb.waits);
        try t.expectEqual(@as(usize, 1), cb.releases);
    }
    var foreground = try lanes.Stream.init(a, .{ .id = "foreground", .prompt = &ids, .max_new = 1 });
    defer foreground.deinit(a);
    try fill(&f, &foreground, 8, null);
    try f.drain();
    try t.expectEqualSlices(u32, &.{ 8, 8, 8 }, foreground.prefill_widths.items);
}

test "Nemotron prefill widths: a pending cancel wins over yield and submits no chunk" {
    const Stop = struct {
        fn yes(_: *anyopaque) bool {
            return true;
        }
    };
    var dummy: u8 = 0;
    var s = try lanes.Stream.init(a, .{ .id = "cancel", .prompt = &.{ 1, 2, 3 }, .max_new = 0, .cancel_check = .{ .ptr = &dummy, .check = Stop.yes }, .yield_check = .{ .ptr = &dummy, .check = Stop.yes } });
    defer s.deinit(a);
    var f: Fake = .{};
    defer f.deinit();
    try t.expectError(error.Cancelled, fill(&f, &s, 8, null));
    try t.expectEqual(@as(usize, 0), f.submitted);
    try t.expectEqual(@as(usize, 0), f.flights.items.len);
}

test "Nemotron prefill widths: actual requested cuts, decision row and cap land in order with one chunk ahead" {
    const ids: [29]u32 = @splat(7);
    var s = try lanes.Stream.init(a, .{ .id = "cut", .prompt = &ids, .max_new = 1, .chunks = &.{ 3, 20 } });
    defer s.deinit(a);
    var f: Fake = .{};
    defer f.deinit();
    try fill(&f, &s, 8, 12);
    try t.expectEqualSlices(u32, &.{ 3, 8, 2, 7, 8 }, s.prefill_widths.items);
    try t.expectEqual(@as(usize, 1), f.flights.items.len);
    try t.expectEqual(@as(usize, 0), f.states[5].waits);
    try f.drain();
    try t.expectEqualSlices(u32, &.{ 3, 8, 2, 7, 8, 1 }, s.prefill_widths.items);
    for (f.states[0..6]) |cb| {
        try t.expectEqual(@as(usize, 1), cb.waits);
        try t.expectEqual(@as(usize, 1), cb.releases);
    }
    try t.expectEqual(@as(usize, 6), f.timing.count[0]);
    try t.expectEqual(@as(f64, 12), f.gpu_ms);
}

test "Nemotron prefill widths: one pass waits once; verify and draft buffers never count as prompt passes" {
    var s = try lanes.Stream.init(a, .{ .id = "one", .prompt = &.{ 1, 2, 3, 4, 5, 6, 7 }, .max_new = 1 });
    defer s.deinit(a);
    var f: Fake = .{};
    defer f.deinit();
    try fill(&f, &s, 2048, null);
    try t.expectEqual(@as(usize, 0), s.prefill_widths.items.len);
    try f.submit(.verify, 9, .{});
    try f.submit(.draft, 10, .{});
    try f.drain();
    try t.expectEqualSlices(u32, &.{7}, s.prefill_widths.items);
    try t.expectEqualSlices(usize, &.{ 1, 0, 1, 1 }, &f.timing.count);
    try t.expectEqual(@as(u64, 10), f.landed);
    for (f.states[0..3]) |cb| try t.expectEqual(@as(usize, 1), cb.waits);
}

test "Nemotron prefill widths: interleaved flights retain their request owner and failed submission appends nothing" {
    var first = try lanes.Stream.init(a, .{ .id = "first", .prompt = &.{ 1, 2, 3 }, .max_new = 1 });
    defer first.deinit(a);
    var second = try lanes.Stream.init(a, .{ .id = "second", .prompt = &.{ 4, 5 }, .max_new = 1 });
    defer second.deinit(a);
    var f: Fake = .{};
    defer f.deinit();
    try workflow.submit(&f, &first, .{ .rows = @as(usize, 3) });
    try workflow.submit(&f, &second, .{ .rows = @as(usize, 2) });
    try t.expectEqualSlices(u32, &.{3}, first.prefill_widths.items);
    try t.expectEqual(@as(usize, 0), second.prefill_widths.items.len);
    f.fail_submit = true;
    try t.expectError(error.SubmitFailed, workflow.submit(&f, &second, .{ .rows = @as(usize, 1) }));
    try t.expectEqual(@as(usize, 1), f.flights.items.len);
    try f.drain();
    try t.expectEqualSlices(u32, &.{3}, first.prefill_widths.items);
    try t.expectEqualSlices(u32, &.{2}, second.prefill_widths.items);
}

test "Nemotron prefill widths: failed release drain forgets only its owner before the stream can be freed" {
    var first = try lanes.Stream.init(a, .{ .id = "released", .prompt = &.{ 1, 2, 3 }, .max_new = 1 });
    defer first.deinit(a);
    var second = try lanes.Stream.init(a, .{ .id = "retained", .prompt = &.{ 4, 5 }, .max_new = 1 });
    defer second.deinit(a);
    var f: Fake = .{};
    defer f.deinit();
    try f.submit(.prefill, 1, .{});
    f.flights.items[0].prompt = .{ .stream = &first, .rows = 3 };
    try f.submit(.prefill, 2, .{});
    f.flights.items[1].prompt = .{ .stream = &second, .rows = 2 };
    workflow.forget(&f, &first);
    try t.expect(f.flights.items[0].prompt == null);
    try t.expect(f.flights.items[1].prompt.?.stream == &second);
    try t.expectEqual(@as(usize, 0), f.states[0].waits);
    try f.drain();
    try t.expectEqual(@as(usize, 0), first.prefill_widths.items.len);
    try t.expectEqualSlices(u32, &.{2}, second.prefill_widths.items);
}
