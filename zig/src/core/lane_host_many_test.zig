//! Several queued requests in one prompt pass (a backend with prefill_many): the replies a pass alone would give.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const gpa = std.testing.allocator;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }
    fn wait(b: *Box) !Reason {
        const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            const now = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
            if (now - began > 2 * std.time.ns_per_s) return error.RequestTimedOut;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// Each reply the fake target's own continuation of its prompt.
fn expectOwn(prompt: []const u32, tokens: []const u32) !void {
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, prompt);
    for (tokens) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

/// `n` requests queued before the host starts (`sampled`: one the fake refuses), replies checked, batched passes.
const Passes = struct { passes: usize, streams: usize };

fn serve(n: usize, memory_lanes: ?usize, sampled: ?usize) !Passes {
    return serveOn(n, memory_lanes, sampled, false);
}

fn serveOn(n: usize, memory_lanes: ?usize, sampled: ?usize, steps: bool) !Passes {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = memory_lanes, .refuse_sampled = sampled != null };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, if (steps) target.batchedSteps() else target.batched(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    var prompts: [6][5]u32 = undefined;
    var requests: [6]Request = undefined;
    var boxes: [6]Box = @splat(.{});
    defer for (&boxes) |*b| b.tokens.deinit(gpa);
    const e = host.engine();
    for (0..n) |i| {
        prompts[i] = .{ 2, 7, 1, 8, @intCast(10 + i) };
        requests[i] = .{ .prompt = &prompts[i], .max_tokens = 12 };
        if (sampled == i) requests[i].sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 5 };
        try e.submit(@intCast(i + 1), &requests[i], .{ .ctx = &boxes[i], .event = Box.event });
    }
    try host.start();
    defer host.stop();
    for (0..n) |i| {
        if (sampled == i) {
            try std.testing.expectEqual(Reason.failed, try boxes[i].wait());
            continue;
        }
        try std.testing.expectEqual(Reason.length, try boxes[i].wait());
        try std.testing.expectEqual(@as(usize, 12), boxes[i].tokens.items.len);
        try expectOwn(&prompts[i], boxes[i].tokens.items);
    }
    return .{ .passes = target.many_passes, .streams = target.many_streams };
}

test "queued requests share one prompt pass, each reply the target's own continuation" {
    const r = try serve(4, null, null);
    try std.testing.expectEqual(@as(usize, 1), r.passes);
    try std.testing.expectEqual(@as(usize, 4), r.streams);
}

test "a backend that also fills by chunks: short queued prompts still share one pass" {
    const r = try serveOn(4, null, null, true);
    try std.testing.expectEqual(@as(usize, 1), r.passes);
    try std.testing.expectEqual(@as(usize, 4), r.streams);
}

test "more requests than lanes: the free lanes' prompts in one pass, the rest after" {
    const r = try serve(6, null, null);
    try std.testing.expect(r.passes >= 1);
    try std.testing.expect(r.streams >= 4 and r.streams <= 6);
}

test "a backend without room for them all: they go one at a time, a later pass only as many as fit, same replies" {
    const r = try serve(4, 2, null);
    try std.testing.expect(r.streams <= 2 * r.passes); // the fake refuses a pass past its room: none took more
}

test "one request the backend refuses fails alone: the pass is retried a stream at a time, the others' replies kept" {
    const r = try serve(4, null, 2);
    try std.testing.expectEqual(@as(usize, 0), r.passes);
}

const LateFailure = struct {
    bad: *lanes.Stream,
    raised: bool = false,

    fn hook(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const f: *LateFailure = @ptrCast(@alignCast(ctx));
        if (s == f.bad) f.raised = true;
    }
    fn check(ctx: *anyopaque) bool {
        const f: *LateFailure = @ptrCast(@alignCast(ctx));
        return f.raised;
    }
};

/// Rollback includes the failing participant, but leaves an already-live stream alone.
fn rollbackLate(yielding: bool) !void {
    for (0..3) |bad| {
        var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = 4, .prefill_chunks = 2 };
        defer target.deinit();
        const b = target.batched();
        var incumbent = try lanes.Stream.init(gpa, .{ .id = "incumbent", .prompt = &.{ 8, 9 }, .max_new = 1 });
        defer incumbent.deinit(gpa);
        try b.prefill(&incumbent);
        defer b.release(&incumbent);
        var streams: [3]lanes.Stream = undefined;
        var ss: [3]*lanes.Stream = undefined;
        for (&streams, &ss) |*s, *p| {
            s.* = try lanes.Stream.init(gpa, .{ .id = "batch", .prompt = &.{ 1, 2, 3 }, .max_new = 1 });
            p.* = s;
        }
        defer for (&streams) |*s| s.deinit(gpa);
        var fail: LateFailure = .{ .bad = &streams[bad] };
        const check: lanes.stream.CancelCheck = .{ .ptr = &fail, .check = LateFailure.check };
        if (yielding) streams[bad].yield_check = check else streams[bad].cancel_check = check;
        target.prefill_hook = LateFailure.hook;
        target.prefill_hook_ctx = &fail;
        try std.testing.expectError(if (yielding) error.Yielded else error.Cancelled, b.vtable.prefill_many.?(b.ptr, &ss));
        try std.testing.expect(fail.raised);
        try std.testing.expectEqual(@as(usize, 1), target.lanes.count());
        try std.testing.expect(target.lanes.contains(&incumbent));
        target.prefill_hook = null;
        fail.raised = false;
        for (ss) |s| try b.prefill(s);
        try std.testing.expectEqual(@as(usize, 4), target.lanes.count());
        for (ss) |s| b.release(s);
        try std.testing.expectEqual(@as(usize, 1), target.lanes.count());
    }
}

test "a cancelled batch rolls back every participant including the failing stream, preserving an existing lane" {
    try rollbackLate(false);
}

test "a yielded batch rolls back every participant including the failing stream, preserving an existing lane" {
    try rollbackLate(true);
}

const OpenFailure = struct {
    fn bad(s: *lanes.Stream) bool {
        return s.prompt()[4] == 11;
    }
    fn first(ctx: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        if (bad(s)) return error.GpuFailed;
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ctx));
        return f.backend().first(s, position);
    }
    fn draft(ctx: *anyopaque, requests: []const lanes.backend.DraftRequest) !void {
        for (requests) |r| if (bad(r.stream)) return error.GpuFailed;
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ctx));
        return f.backend().draft(requests);
    }
};

test "a successful prompt batch followed by one first-token or draft error leaves the other replies intact" {
    for ([_]bool{ false, true }) |draft| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 3 }, 8, 7);
        defer cfg.deinit(gpa);
        var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = 3 };
        defer target.deinit();
        var table = target.batched().vtable.*;
        if (draft) table.draft = OpenFailure.draft else table.first = OpenFailure.first;
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, .{ .ptr = &target, .vtable = &table }, clock.clock());
        defer core.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
        var prompts: [3][5]u32 = undefined;
        var requests: [3]Request = undefined;
        var boxes: [3]Box = @splat(.{});
        defer for (&boxes) |*b| b.tokens.deinit(gpa);
        const e = host.engine();
        for (0..3) |i| {
            prompts[i] = .{ 2, 7, 1, 8, @intCast(10 + i) };
            requests[i] = .{ .prompt = &prompts[i], .max_tokens = 12 };
            try e.submit(@intCast(i + 1), &requests[i], .{ .ctx = &boxes[i], .event = Box.event });
        }
        {
            try host.start();
            defer host.stop();
            for (0..3) |i| {
                try std.testing.expectEqual(if (i == 1) Reason.failed else Reason.length, try boxes[i].wait());
                if (i != 1) {
                    try std.testing.expectEqual(@as(usize, 12), boxes[i].tokens.items.len);
                    try expectOwn(&prompts[i], boxes[i].tokens.items);
                }
            }
        }
        try std.testing.expectEqual(@as(usize, 1), target.many_passes);
        try std.testing.expectEqual(@as(usize, 3), target.many_streams);
        try std.testing.expectEqual(@as(usize, 0), target.lanes.count());
        try std.testing.expectEqual(@as(usize, 0), core.activeCount());
    }
}
