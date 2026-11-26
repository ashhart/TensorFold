//! A request's grammar the server compiled reaches the lane thread as is: no compile there, the same masked reply.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const fake = lanes.grammar_fake;
const a = std.testing.allocator;

const Box = struct {
    reason: api.Reason = .failed,
    tokens: std.ArrayList(u32) = .empty,
    done: std.atomic.Value(bool) = .init(false),
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(a, t) catch {},
            .finished => |f| {
                b.reason = f.reason;
                b.done.store(true, .release);
            },
            else => {},
        }
    }
    fn wait(b: *Box) !api.Reason {
        const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
        while (!b.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began > 2 * std.time.ns_per_s) return error.RequestTimedOut;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return b.reason;
    }
};

/// One reply under the request's structure; returns its tokens (owned) after `finished`.
fn reply(host: *api.LaneHost, id: api.Id, st: api.Structure) ![]u32 {
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 8, .structure = st };
    var box: Box = .{};
    try host.engine().submit(id, &request, .{ .ctx = &box, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, try box.wait());
    return box.tokens.toOwnedSlice(a);
}

test "a compiled grammar handed in compiles nothing on the lane thread and masks the reply as a compile does" {
    const old = fake.install();
    defer _ = lanes.grammar.swapLibrary(old);
    var c = fake.compiler();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
    try host.start();
    defer host.stop();
    var err: [512]u8 = undefined;
    const text = "{\"type\":\"object\"}";
    const g = try c.compile(std.testing.io, .json_schema, text, &err); // the server's compile
    defer g.free();
    const handed = try reply(&host, 1, .{ .kind = .json_schema, .text = text, .compiler = &c, .compiled = g });
    defer a.free(handed);
    try std.testing.expectEqual(@as(usize, 1), fake.compiles);
    try std.testing.expectEqual(@as(usize, 0), fake.frees); // the handle stays the server's
    const compiled = try reply(&host, 2, .{ .kind = .json_schema, .text = text, .compiler = &c });
    defer a.free(compiled);
    try std.testing.expectEqual(@as(usize, 2), fake.compiles); // without a handle the lane thread compiles, as before
    try std.testing.expectEqual(@as(usize, 8), handed.len);
    try std.testing.expectEqualSlices(u32, compiled, handed);
    try std.testing.expectEqual(@as(usize, 0), fake.arms + fake.narrows); // grammar_budget off: never consulted
    const budgeted = try reply(&host, 3, .{ .kind = .json_schema, .text = text, .compiler = &c, .compiled = g, .budget = true });
    defer a.free(budgeted);
    try std.testing.expectEqual(@as(usize, 1), fake.arms); // armed once, before its first token
    try std.testing.expect(fake.narrows > 0);
    try std.testing.expectEqualSlices(u32, handed, budgeted); // the stand-in narrows nothing, so nothing else changed
}

test "lane_stream.open frees its reply grammar when Stream.init fails with OutOfMemory" {
    const lane_stream = @import("lane_stream.zig");
    const Tracked = struct {
        var live: usize = 0;
        var created: usize = 0;
        var freed: usize = 0;
        var base: lanes.grammar.Library = undefined;
        fn matcher(g: *anyopaque) callconv(.c) ?*anyopaque {
            const m = base.matcher(g) orelse return null;
            live += 1;
            created += 1;
            return m;
        }
        fn matcherFree(m: *anyopaque) callconv(.c) void {
            freed += 1;
            live -= 1;
            base.matcher_free(m);
        }
    };
    const old = fake.install();
    defer _ = lanes.grammar.swapLibrary(old);
    const base_lib = fake.compiler().f;
    Tracked.live = 0;
    Tracked.created = 0;
    Tracked.freed = 0;
    Tracked.base = base_lib;
    var lib = base_lib;
    lib.matcher = Tracked.matcher;
    lib.matcher_free = Tracked.matcherFree;
    _ = lanes.grammar.swapLibrary(lib);
    var c = fake.compiler();
    c.f = lib;
    var err: [512]u8 = undefined;
    const text = "{\"type\":\"object\"}";
    const compiled = try c.compile(std.testing.io, .json_schema, text, &err);
    defer compiled.free();

    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();

    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const request: api.Request = .{
        .prompt = &prompt,
        .max_tokens = 8,
        .structure = .{ .kind = .json_schema, .text = text, .compiler = &c, .compiled = compiled, .budget = true },
    };
    const FailSink = struct {
        finished: usize = 0,
        reason: ?api.Reason = null,
        message: []const u8 = "",
        fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
            const s: *@This() = @ptrCast(@alignCast(ctx));
            if (e.* == .finished) {
                s.finished += 1;
                s.reason = e.finished.reason;
                s.message = e.finished.message;
            }
        }
    };

    var saw_stream_oom = false;
    var opened_once = false;
    var fail_index: usize = 0;
    while (fail_index < 32) : (fail_index += 1) {
        var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = fail_index });
        const gpa = failing.allocator();
        var host = api.LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
        defer host.stop();
        const job = gpa.create(api.LaneHost.Job) catch continue;
        var sink: FailSink = .{};
        job.* = .{ .host = &host, .id = fail_index + 1, .request = &request, .sink = .{ .ctx = &sink, .event = FailSink.event } };
        const before_arms = fake.arms;
        const opened = lane_stream.open(&host, job);
        if (opened) {
            opened_once = true;
            try std.testing.expect(job.started);
            try std.testing.expect(job.stream.grammar != null);
            host.finish(job, .cancelled, "");
            try std.testing.expectEqual(@as(usize, 0), Tracked.live);
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            break;
        }
        try std.testing.expectEqual(@as(usize, 1), sink.finished);
        try std.testing.expectEqual(api.Reason.failed, sink.reason.?);
        if (fake.arms > before_arms) {
            // Constraint armed and returned before Stream.init failed with OutOfMemory.
            saw_stream_oom = true;
            try std.testing.expectEqualStrings("out of memory", sink.message);
        }
        try std.testing.expectEqual(@as(usize, 0), Tracked.live);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    }
    try std.testing.expect(saw_stream_oom);
    try std.testing.expect(opened_once);
    try std.testing.expectEqual(Tracked.created, Tracked.freed);
}
