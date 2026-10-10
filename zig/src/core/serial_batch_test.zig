//! The shared host bounds draft batches, routes plain/sampled/custom-stop requests and cancels between callbacks.
const std = @import("std");
const api = @import("engine_api.zig");
const serial = @import("serial_host.zig");
const io = std.testing.io;
const Fixture = struct {
    next: u32 = 100,
    draws: usize = 0,
    batches: usize = 0,
    budget: u32 = 0,
    bad: u8 = 0,
    slow: bool = false,
    started: std.atomic.Value(bool) = .init(false),
    words: [serial.max_round + 1]u32 = undefined,
    fn self(ptr: *anyopaque) *@This() {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        self(ptr).next = 100;
    }
    fn prompt(_: *anyopaque, _: []const u32, _: bool) !void {}
    fn draw(ptr: *anyopaque, _: ?api.Sampling) !u32 {
        const f = self(ptr);
        f.draws += 1;
        return f.next;
    }
    fn advance(ptr: *anyopaque, token: u32) !void {
        if (self(ptr).next != token) return error.BadAdvance;
        self(ptr).next += 1;
    }
    fn batch(ptr: *anyopaque, budget: u32, eos: []const u32) !serial.Batch {
        const f = self(ptr);
        f.batches += 1;
        f.budget = @max(f.budget, budget);
        f.started.store(true, .release);
        if (f.slow) try std.Io.sleep(io, .fromMilliseconds(30), .awake);
        if (f.bad == 1) return .{ .tokens = &.{} };
        if (f.bad == 2) return .{ .tokens = &f.words };
        if (f.bad == 3) {
            f.words[0] = eos[0];
            f.words[1] = 104;
            return .{ .tokens = f.words[0..2] };
        }
        var count: usize = 0;
        while (count < @min(budget, 16)) {
            const token = f.next;
            f.words[count] = token;
            f.next += 1;
            count += 1;
            if (std.mem.indexOfScalar(u32, eos, token) != null) break;
        }
        return .{ .tokens = f.words[0..count], .stats = .{ .rounds = 1, .drafted = count - 1, .accepted = count - 1, .min_rows = 2 } };
    }
    fn driver(f: *Fixture) serial.Driver {
        return .{ .ctx = f, .info = .{ .name = "fixture", .context_window = 128, .prefill_step = 16 }, .reset = reset, .prompt_chunk = prompt, .draw = draw, .advance = advance, .draft_batch = batch };
    }
};
const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: [32]u32 = undefined,
    count: usize = 0,
    reason: ?api.Reason = null,
    stats: api.Stats = .{},
    message: []const u8 = "",
    token_events: usize = 0,
    cancel_engine: ?api.Engine = null,
    fn event(ptr: *anyopaque, id: api.Id, event_value: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (event_value.*) {
            .prefilled => {},
            .logprobs => {},
            .tokens => |ids| {
                @memcpy(b.tokens[b.count..][0..ids.len], ids);
                b.count += ids.len;
                b.token_events += 1;
                if (b.cancel_engine) |engine| engine.cancel(id);
            },
            .finished => |finish| {
                b.reason = finish.reason;
                b.stats = finish.stats;
                b.message = finish.message;
            },
        }
    }
    fn wait(b: *Box) !void {
        for (0..2000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.reason != null;
            b.mutex.unlock(io);
            if (done) return;
            try std.Io.sleep(io, .fromMilliseconds(1), .awake);
        }
        return error.HostTimeout;
    }
};
fn stop(_: *anyopaque, tokens: []const u32) bool {
    return tokens.len == 4;
}
test "draft batch routing, bounded delivery and EOS preserve shared host events" {
    const a = std.testing.allocator;
    for (0..6) |mode| {
        var f = Fixture{};
        var b = Box{};
        const h = try serial.Host.init(a, io, f.driver());
        defer h.deinit();
        try std.testing.expect(!h.engine().info().plain_only);
        const request = api.Request{ .prompt = &.{1}, .max_tokens = 20, .drafts = mode != 1, .sampling = if (mode == 2) .{ .seed = 7 } else null, .stop = if (mode == 3) .{ .ctx = &f, .check = stop } else null, .eos = if (mode == 4) &.{103} else &.{} };
        if (mode == 5) f.bad = 1;
        try h.engine().submit(1, &request, .{ .ctx = &b, .event = Box.event });
        try b.wait();
        if (mode == 5) {
            try std.testing.expectEqual(api.Reason.failed, b.reason.?);
            try std.testing.expectEqualStrings("BadDraftBatch", b.message);
            continue;
        }
        const count: usize = if (mode == 3 or mode == 4) 4 else 20;
        try std.testing.expectEqual(count, b.count);
        for (b.tokens[0..count], 0..) |token, i| try std.testing.expectEqual(@as(u32, @intCast(100 + i)), token);
        if (mode == 1 or mode == 2 or mode == 3) {
            try std.testing.expectEqual(@as(usize, 0), f.batches);
            try std.testing.expectEqual(count, f.draws);
        } else {
            try std.testing.expectEqual(@as(usize, 0), f.draws);
            try std.testing.expectEqual(request.max_tokens, f.budget);
            try std.testing.expectEqual(@as(u32, 2), b.stats.min_rows);
        }
        try std.testing.expectEqual(if (count == 4) api.Reason.stop else api.Reason.length, b.reason.?);
    }
}

test "draft cancellation publishes a complete returned round and resets the next request" {
    var f = Fixture{};
    const h = try serial.Host.init(std.testing.allocator, io, f.driver());
    defer h.deinit();
    var canceled = Box{ .cancel_engine = h.engine() };
    const request = api.Request{ .prompt = &.{1}, .max_tokens = 20 };
    try h.engine().submit(3, &request, .{ .ctx = &canceled, .event = Box.event });
    try canceled.wait();
    try std.testing.expectEqual(api.Reason.cancelled, canceled.reason.?);
    try std.testing.expectEqual(@as(usize, 16), canceled.count);
    try std.testing.expectEqual(@as(usize, 1), canceled.token_events);
    try std.testing.expectEqual(@as(usize, 1), f.batches);
    var next = Box{};
    const short = api.Request{ .prompt = &.{1}, .max_tokens = 3 };
    try h.engine().submit(4, &short, .{ .ctx = &next, .event = Box.event });
    try next.wait();
    try std.testing.expectEqual(api.Reason.length, next.reason.?);
    try std.testing.expectEqualSlices(u32, &.{ 100, 101, 102 }, next.tokens[0..next.count]);
}

test "a returned draft round cannot include tokens after EOS" {
    var f = Fixture{ .bad = 3 };
    var b = Box{};
    const h = try serial.Host.init(std.testing.allocator, io, f.driver());
    defer h.deinit();
    const request = api.Request{ .prompt = &.{1}, .max_tokens = 20, .eos = &.{103} };
    try h.engine().submit(5, &request, .{ .ctx = &b, .event = Box.event });
    try b.wait();
    try std.testing.expectEqual(api.Reason.failed, b.reason.?);
    try std.testing.expectEqualStrings("BadDraftBatch", b.message);
    try std.testing.expectEqual(@as(usize, 0), b.count);
}
test "invalid oversized batches and canceled active callbacks emit no stale tokens" {
    const a = std.testing.allocator;
    for ([_]bool{ false, true }) |cancel| {
        var f = Fixture{ .bad = if (cancel) 0 else 2, .slow = cancel };
        var b = Box{};
        const h = try serial.Host.init(a, io, f.driver());
        defer h.deinit();
        const request = api.Request{ .prompt = &.{1}, .max_tokens = 20 };
        try h.engine().submit(2, &request, .{ .ctx = &b, .event = Box.event });
        if (cancel) {
            while (!f.started.load(.acquire)) try std.Io.sleep(io, .fromMilliseconds(1), .awake);
            h.engine().cancel(2);
        }
        try b.wait();
        try std.testing.expectEqual(@as(usize, 0), b.count);
        try std.testing.expectEqual(if (cancel) api.Reason.cancelled else api.Reason.failed, b.reason.?);
    }
}
