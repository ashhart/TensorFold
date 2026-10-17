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
    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
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
fn serve(n: usize, memory_lanes: ?usize, sampled: ?usize) !struct { passes: usize, streams: usize } {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = memory_lanes, .refuse_sampled = sampled != null };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.batched(), clock.clock());
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
            try std.testing.expectEqual(Reason.failed, boxes[i].wait());
            continue;
        }
        try std.testing.expectEqual(Reason.length, boxes[i].wait());
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
