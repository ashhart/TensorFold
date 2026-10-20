//! Remaining-work scheduling respects progress, stable ties and bounded admission age.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const Host = @import("lane_host.zig").LaneHost;
const fill = @import("lane_host_fill.zig");
const t = std.testing;
const a = t.allocator;
fn noop(_: *anyopaque, _: api.Id, _: *const api.Event) void {}
fn job(h: *Host, request: *const api.Request, cached: u32, progress: u32, began: bool) Host.Job {
    var result: Host.Job = .{ .host = h, .id = 1, .request = request, .sink = .{ .ctx = h, .event = noop } };
    result.fill_began = began;
    result.stream.prompt_len = request.prompt.len;
    result.stream.cached = cached;
    result.stream.fill_at = .init(progress);
    return result;
}
test "prefill remaining: progress reports cached floors and stale cursors preserve stable priority" {
    var target: lanes.fake.Fake = .{ .gpa = a };
    var core: lanes.Engine = undefined;
    var vt = target.stepped().vtable.*;
    core.backend = .{ .ptr = &target, .vtable = &vt };
    var host = Host.init(a, t.io, &core, .{ .lanes = 2 });
    defer host.filling.deinit(a);
    const tokens: [32]u32 = @splat(3);
    const long: api.Request = .{ .prompt = &tokens, .max_tokens = 1 };
    const short: api.Request = .{ .prompt = tokens[0..16], .max_tokens = 1 };
    const Case = struct { cached: u32, progress: u32, began: bool, reports: bool, winner: usize };
    const cases = [_]Case{
        .{ .cached = 0, .progress = 28, .began = true, .reports = true, .winner = 0 },
        .{ .cached = 0, .progress = 28, .began = true, .reports = false, .winner = 1 },
        .{ .cached = 0, .progress = 28, .began = false, .reports = true, .winner = 1 },
        .{ .cached = 30, .progress = 2, .began = true, .reports = true, .winner = 0 },
        .{ .cached = 30, .progress = 0, .began = true, .reports = false, .winner = 0 },
        .{ .cached = 0, .progress = 100, .began = true, .reports = true, .winner = 0 },
        .{ .cached = 0, .progress = 16, .began = true, .reports = true, .winner = 0 },
    };
    for (cases) |c| {
        vt.reports_fill = c.reports;
        var first = job(&host, &long, c.cached, c.progress, c.began);
        var second = job(&host, &short, 0, 0, false);
        host.filling.clearRetainingCapacity();
        try host.filling.appendSlice(a, &.{ &first, &second });
        try t.expectEqual(c.winner, fill.next(&host));
        try t.expectEqual(@as(usize, if (c.winner == 0) 0 else 1), first.fill_wait);
        try t.expectEqual(@as(usize, if (c.winner == 1) 0 else 1), second.fill_wait);
    }
}
test "prefill remaining: foreground preference yields to aging and keeps equal-age ties stable" {
    var target: lanes.fake.Fake = .{ .gpa = a };
    var core: lanes.Engine = undefined;
    core.backend = target.stepped();
    var host = Host.init(a, t.io, &core, .{ .lanes = 3 });
    defer host.filling.deinit(a);
    const tokens: [32]u32 = @splat(3);
    const background: api.Request = .{ .prompt = &tokens, .max_tokens = 1, .background = true };
    const foreground: api.Request = .{ .prompt = tokens[0..16], .max_tokens = 1 };
    var first = job(&host, &background, 0, 31, true);
    var second = job(&host, &foreground, 0, 0, true);
    var third = job(&host, &background, 0, 0, false);
    try host.filling.appendSlice(a, &.{ &first, &second, &third });
    try t.expectEqual(@as(usize, 1), fill.next(&host));
    first.fill_wait = fill.preference_rounds;
    third.fill_wait = fill.preference_rounds;
    try t.expectEqual(@as(usize, 0), fill.next(&host));
    try t.expectEqual(@as(usize, 2), fill.next(&host));
    var served = false;
    for (0..fill.preference_rounds + 3) |_| if (fill.next(&host) == 0) {
        served = true;
        break;
    };
    try t.expect(served);
}
const Watch = struct {
    engine: api.Engine,
    later: *const api.Request,
    order: [2]api.Id = @splat(0),
    count: usize = 0,
    submitted: bool = false,
    done: std.atomic.Value(u32) = .init(0),
    fn event(ptr: *anyopaque, id: api.Id, value: *const api.Event) void {
        const w: *Watch = @ptrCast(@alignCast(ptr));
        switch (value.*) {
            .prefilled => {
                w.order[w.count] = id;
                w.count += 1;
            },
            .finished => _ = w.done.fetchAdd(1, .release),
            else => {},
        }
    }
    fn arrive(ptr: *anyopaque, _: *lanes.Stream, chunk: usize) void {
        const w: *Watch = @ptrCast(@alignCast(ptr));
        if (w.submitted or chunk != 2) return;
        w.engine.submit(2, w.later, .{ .ctx = w, .event = event }) catch return;
        w.submitted = true;
    }
};
test "prefill remaining: a long prompt with one chunk left finishes before a new shorter prompt" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = Host.init(a, t.io, &core, .{ .lanes = 2 });
    const tokens: [32]u32 = @splat(3);
    const long: api.Request = .{ .prompt = &tokens, .max_tokens = 2 };
    const short: api.Request = .{ .prompt = tokens[0..16], .max_tokens = 2 };
    var watch: Watch = .{ .engine = host.engine(), .later = &short };
    target.prefill_hook = Watch.arrive;
    target.prefill_hook_ctx = &watch;
    try watch.engine.submit(1, &long, .{ .ctx = &watch, .event = Watch.event });
    try host.start();
    defer host.stop();
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (watch.done.load(.acquire) != 2) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() >= until) return error.TestUnexpectedResult;
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    try t.expect(watch.submitted);
    try t.expectEqual(@as(usize, 2), watch.count);
    try t.expectEqualSlices(api.Id, &.{ 1, 2 }, &watch.order);
}
