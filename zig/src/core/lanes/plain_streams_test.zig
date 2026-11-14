//! Plain cross-stream sharing has its own capability, without widening a stream's verification window.
const std = @import("std");
const setup = @import("config.zig");
const Engine = @import("engine.zig").Engine;
const be = @import("backend.zig");
const sm = @import("stream.zig");
const fake = @import("fake.zig");
const a = std.testing.allocator;

const Watch = struct {
    var inner_verify: *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void = undefined;
    var inner_queue: *const fn (*anyopaque, *sm.Stream, be.Feed, u64) anyerror!u64 = undefined;
    var widest: usize = 0;
    var queues: usize = 0;
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        widest = @max(widest, windows.len);
        for (windows) |w| try std.testing.expectEqual(@as(usize, 1), w.rows());
        try inner_verify(ptr, windows, out);
    }
    fn queue(ptr: *anyopaque, s: *sm.Stream, feed: be.Feed, position: u64) !u64 {
        queues += 1;
        return inner_queue(ptr, s, feed, position);
    }
};
const Result = struct { tokens: [4][32]u32 = undefined, widest: usize, queues: usize };
fn run(n: usize, sharing: bool, sampled: bool, drafts: bool, staggered: bool) !Result {
    var target: fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var backend = target.backend();
    var hooks = backend.vtable.*;
    Watch.inner_verify = hooks.verify;
    Watch.inner_queue = hooks.queue;
    Watch.widest = 0;
    Watch.queues = 0;
    hooks.verify = Watch.verify;
    hooks.queue = Watch.queue;
    backend.vtable = &hooks;
    var cfg = try setup.Config.init(a, .{ .plain_share = sharing, .gpu_tokens = true, .hidden_rows = true, .batch_rows = 4, .max_streams = 4 }, 16, 15);
    defer cfg.deinit(a);
    var clock: fake.FixedClock = .{};
    var engine = Engine.init(a, &cfg, backend, clock.clock());
    defer engine.deinit();
    var streams: [4]sm.Stream = undefined;
    const prompt = [_]u32{ 1, 5, 9, 2, 8, 3, 7 };
    for (streams[0..n], 0..) |*s, i| s.* = try sm.Stream.init(a, .{ .id = "plain", .prompt = prompt[0 .. 3 + i], .max_new = 32, .drafts = drafts, .sampling = if (sampled) .{ .seed = 7 + i, .temperature = 0.7, .top_k = 20, .top_p = 0.95 } else null });
    defer for (streams[0..n]) |*s| s.deinit(a);
    for (streams[0..n], 0..) |*s, i| {
        try engine.addStream(s);
        if (staggered and i == 0) for (0..4) |_| try engine.step();
    }
    var steps: usize = 0;
    while (engine.activeCount() > 0) : (steps += 1) {
        if (steps > 256) return error.RoundLoopDidNotFinish;
        try engine.step();
    }
    var result: Result = .{ .widest = Watch.widest, .queues = Watch.queues };
    for (streams[0..n], 0..) |*s, i| {
        try std.testing.expectEqual(@as(usize, 32), s.emitted().len);
        @memcpy(&result.tokens[i], s.emitted());
    }
    try std.testing.expectEqual(@as(usize, 0), target.drafts_calls);
    try std.testing.expectEqual(@as(u64, 0), engine.drafted);
    return result;
}

test "plain sharing: width one may share streams without increasing drafts or copies" {
    var c = try setup.Config.init(a, .{ .plain_share = true, .gpu_tokens = true, .hidden_rows = true, .batch_rows = 4, .max_streams = 4 }, 16, 15);
    defer c.deinit(a);
    try std.testing.expect(c.family_streams and c.pipelined);
    try std.testing.expectEqual(@as(u32, 4), c.batch_streams);
    try std.testing.expectEqual(@as(u32, 1), c.family_width);
    try std.testing.expectEqual(@as(u32, 1), c.base_width);
    try std.testing.expectEqual(@as(u32, 0), c.max_copy);
    try std.testing.expectEqual(@as(u32, 0), c.most_drafts);
    try std.testing.expect(!c.family_mtp);
    for ([_]setup.Model{
        .{ .hidden_rows = true },
        .{ .plain_share = true },
        .{ .plain_share = true, .hidden_rows = true, .streams_exact = false },
    }) |m| {
        var off = try setup.Config.init(a, m, 16, 15);
        defer off.deinit(a);
        try std.testing.expect(!off.family_streams);
        try std.testing.expectEqual(@as(u32, 1), off.batch_streams);
    }
}

test "plain sharing: synchronous streams equal pipelined solo and keep the lone path" {
    for ([_]bool{ false, true }) |sampled| for ([_]bool{ false, true }) |drafts| {
        for ([_]usize{ 1, 2, 3, 4 }) |n| for ([_]bool{ false, true }) |staggered| {
            const before = try run(n, false, sampled, drafts, staggered);
            const after = try run(n, true, sampled, drafts, staggered);
            for (0..n) |i| try std.testing.expectEqualSlices(u32, &before.tokens[i], &after.tokens[i]);
            if (n > 1) try std.testing.expectEqual(n, after.widest) else {
                try std.testing.expectEqual(@as(usize, 0), after.widest);
                try std.testing.expectEqual(before.queues, after.queues);
            }
        };
    };
}
