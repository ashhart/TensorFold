//! Driver fixtures check serialized requests, cancellation of queued work and reset isolation.
const std = @import("std");
const api = @import("engine_api.zig");
const serial = @import("serial_host.zig");
const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    prefilled: bool = false,
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |ids| b.tokens.appendSlice(std.testing.allocator, ids) catch {},
            .prefilled => b.prefilled = true,
            .finished => |f| b.done = f.reason,
        }
    }
    fn wait(b: *Box) !api.Reason {
        for (0..2000) |_| {
            b.mutex.lockUncancelable(std.testing.io);
            const value = b.done;
            b.mutex.unlock(std.testing.io);
            if (value) |reason| return reason;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.NoCompletion;
    }
};
const Target = struct {
    position: u32 = 0,
    chunk_count: u32 = 0,
    resets: u32 = 0,
    mutex: std.Io.Mutex = .init,
    blocked: bool = false,
    release: bool = true,
    fn self(ptr: *anyopaque) *Target {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        const t = self(ptr);
        t.position = 0;
        t.chunk_count = 0;
        t.resets += 1;
    }
    fn chunk(ptr: *anyopaque, ids: []const u32, _: bool) !void {
        const t = self(ptr);
        t.position += @intCast(ids.len);
        t.chunk_count += 1;
        for (0..2000) |_| {
            t.mutex.lockUncancelable(std.testing.io);
            t.blocked = !t.release;
            const ready = t.release;
            t.mutex.unlock(std.testing.io);
            if (ready) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.TargetBlocked;
    }
    fn draw(ptr: *anyopaque, _: ?api.Sampling) !u32 {
        return self(ptr).position;
    }
    fn advance(ptr: *anyopaque, _: u32) !void {
        self(ptr).position += 1;
    }
    fn driver(t: *Target) serial.Driver {
        return .{ .ctx = t, .info = .{ .name = "fixture", .context_window = 128, .prefill_step = 2 }, .reset = reset, .prompt_chunk = chunk, .draw = draw, .advance = advance };
    }
    fn waitBlocked(t: *Target) !void {
        for (0..2000) |_| {
            t.mutex.lockUncancelable(std.testing.io);
            const blocked = t.blocked;
            t.mutex.unlock(std.testing.io);
            if (blocked) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.NotBlocked;
    }
    fn unblock(t: *Target) void {
        t.mutex.lockUncancelable(std.testing.io);
        t.release = true;
        t.mutex.unlock(std.testing.io);
    }
};

test "serial host uses prompt cuts and resets independent queued requests" {
    const a = std.testing.allocator;
    var target: Target = .{};
    var b1: Box = .{};
    defer b1.tokens.deinit(a);
    var b2: Box = .{};
    defer b2.tokens.deinit(a);
    const h = try serial.Host.init(a, std.testing.io, target.driver());
    defer {
        target.unblock();
        h.deinit();
    }
    try std.testing.expect(h.engine().info().plain_only);
    const r1 = api.Request{ .prompt = &.{ 1, 2, 3 }, .max_tokens = 3, .chunks = &.{1} };
    const r2 = api.Request{ .prompt = &.{4}, .max_tokens = 2 };
    try h.engine().submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try h.engine().submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, try b1.wait());
    try std.testing.expectEqual(api.Reason.length, try b2.wait());
    try std.testing.expectEqualSlices(u32, &.{ 3, 4, 5 }, b1.tokens.items);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, b2.tokens.items);
    try std.testing.expect(b1.prefilled and b2.prefilled);
    try std.testing.expectEqual(@as(u32, 2), target.resets);
    var status: api.Status = undefined;
    h.engine().status(&status, &.{});
    try std.testing.expectEqual(@as(u64, 0), status.generation_tokens);
}

test "canceling queued work leaves the running driver and request lifetime intact" {
    const a = std.testing.allocator;
    var target = Target{ .release = false };
    var b1: Box = .{};
    defer b1.tokens.deinit(a);
    var b2: Box = .{};
    defer b2.tokens.deinit(a);
    const h = try serial.Host.init(a, std.testing.io, target.driver());
    defer {
        target.unblock();
        h.deinit();
    }
    const r1 = api.Request{ .prompt = &.{1}, .max_tokens = 2 };
    const r2 = api.Request{ .prompt = &.{2}, .max_tokens = 2 };
    try h.engine().submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try target.waitBlocked();
    try h.engine().submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    h.engine().cancel(2);
    try std.testing.expectEqual(api.Reason.cancelled, try b2.wait());
    target.unblock();
    try std.testing.expectEqual(api.Reason.length, try b1.wait());
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, b1.tokens.items);
    try std.testing.expectEqual(@as(usize, 0), b2.tokens.items.len);
    try std.testing.expectEqual(@as(u32, 1), target.resets);
}

const Spans = struct {
    calls: std.ArrayList([3]u32) = .empty, // (decoded, start, rows) a chunk
    position: u32 = 0,
    fn self(ptr: *anyopaque) *Spans {
        return @ptrCast(@alignCast(ptr));
    }
    fn reset(ptr: *anyopaque) !void {
        self(ptr).position = 0;
    }
    fn record(ptr: *anyopaque, ids: []const u32, decoded: u32) !void {
        const s = self(ptr);
        try s.calls.append(std.testing.allocator, .{ decoded, s.position, @intCast(ids.len) });
        s.position += @intCast(ids.len);
    }
    fn prompt(ptr: *anyopaque, ids: []const u32, _: bool) !void {
        try record(ptr, ids, 0);
    }
    fn decode(ptr: *anyopaque, ids: []const u32, _: bool) !void {
        try record(ptr, ids, 1);
    }
    fn draw(_: *anyopaque, _: ?api.Sampling) !u32 {
        return 0;
    }
    fn advance(ptr: *anyopaque, _: u32) !void {
        self(ptr).position += 1;
    }
    fn driver(s: *Spans) serial.Driver {
        return .{ .ctx = s, .info = .{ .name = "spans", .context_window = 128, .prefill_step = 4 }, .reset = reset, .prompt_chunk = prompt, .draw = draw, .advance = advance, .decode_chunk = decode };
    }
};

test "reply spans prefill as decoded rows, cutting the prompt's chunks at their edges" {
    const a = std.testing.allocator;
    var s: Spans = .{};
    defer s.calls.deinit(a);
    var b: Box = .{};
    defer b.tokens.deinit(a);
    const h = try serial.Host.init(a, std.testing.io, s.driver());
    defer h.deinit();
    var ids: [20]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i);
    const request = api.Request{ .prompt = &ids, .max_tokens = 1, .decode_spans = &.{ .{ 5, 9 }, .{ 14, 16 } } };
    try h.engine().submit(1, &request, .{ .ctx = &b, .event = Box.event });
    _ = try b.wait();
    const want = [_][3]u32{ .{ 0, 0, 4 }, .{ 0, 4, 1 }, .{ 1, 5, 4 }, .{ 0, 9, 4 }, .{ 0, 13, 1 }, .{ 1, 14, 2 }, .{ 0, 16, 4 } };
    try std.testing.expectEqualSlices([3]u32, &want, s.calls.items);
}
