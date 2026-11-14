//! The observed row residual and the next-draft allocator use the same head batching switch.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const env = @import("sys").env;
const be = @import("backend.zig");
const fake = @import("fake.zig");
const Stream = @import("stream.zig").Stream;
const Engine = @import("engine.zig").Engine;
const Config = @import("config.zig").Config;

const Clock = struct {
    target: fake.Fake = .{ .gpa = a, .head_accepts = 8 },
    step: f64,
    extra: f64,
    batched: bool,
    sum: u32 = 0,
    firsts: u32 = 0,
    deepest: u32 = 0,
    fn self(ptr: *anyopaque) *@This() {
        return @ptrCast(@alignCast(ptr));
    }
    fn clock(c: *@This()) be.Clock {
        return .{ .ptr = c, .vtable = &.{ .start = start, .elapsed_ms = elapsed } };
    }
    fn start(ptr: *anyopaque) void {
        const c = self(ptr);
        c.sum = 0;
        c.firsts = 0;
        c.deepest = 0;
    }
    fn elapsed(ptr: *anyopaque, _: be.Mark) f64 {
        const c = self(ptr);
        const steps = if (c.batched) c.deepest else c.sum;
        const extra = if (c.batched) c.deepest -| 1 else c.sum - c.firsts;
        return 12 + c.step * @as(f64, @floatFromInt(steps)) + c.extra * @as(f64, @floatFromInt(extra));
    }
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const target: *fake.Fake = @ptrCast(@alignCast(ptr));
        const c: *@This() = @fieldParentPtr("target", target);
        for (requests) |r| {
            c.sum += r.depth;
            c.firsts += @intFromBool(r.depth > 0);
            c.deepest = @max(c.deepest, r.depth);
        }
        try target.backend().vtable.draft(ptr, requests);
    }
};

fn run(batched: bool, step: f64, extra: f64) !void {
    var cfg = try Config.init(a, .{ .exact_width = 8, .mtp = true, .speculate = true, .drafts = 2, .draft_prior = &.{ 1, 1 }, .mtp_step_ms = step, .draft_step_ms = extra, .draft_streams = true, .plain_guard = true, .hidden_rows = true, .window_costs = &.{ .{ .width = 1, .ms = 10 }, .{ .width = 8, .ms = 10 } } }, 8, 7);
    defer cfg.deinit(a);
    var clock: Clock = .{ .batched = batched, .step = step, .extra = extra };
    defer clock.target.deinit();
    var backend = clock.target.backend();
    var vtable = backend.vtable.*;
    vtable.draft = Clock.draft;
    backend.vtable = &vtable;
    var e = Engine.init(a, &cfg, backend, clock.clock());
    defer e.deinit();
    try e.rule.observeShared(2, 2, 12, false, 0, true, false);
    var streams: [2]Stream = undefined;
    var made: usize = 0;
    defer for (streams[0..made]) |*s| s.deinit(a);
    for (&streams) |*s| {
        s.* = try Stream.init(a, .{ .id = "heads", .prompt = &.{ 1, 2, 3 }, .max_new = 64 });
        made += 1;
        try e.addStream(s);
    }
    var measured_rounds: usize = 0;
    for (0..3) |_| {
        errdefer std.debug.print("head test: batched={any} step={d} extra={d} firsts={d} sum={d} deepest={d} rows={d} price={any}\n", .{ batched, step, extra, clock.firsts, clock.sum, clock.deepest, clock.target.last_rows, e.rule.sharedPrice(2, false) });
        try e.step();
        try t.expect(clock.firsts == 2 and clock.sum > clock.deepest);
        try t.expectEqual(@as(u32, if (batched or step < 1) 4 else 3), clock.sum);
        measured_rounds += @intFromBool(clock.target.last_rows > 2);
        try t.expectApproxEqAbs(@as(f64, 2), e.rule.sharedPrice(2, false).?.plain, 1e-9);
        try t.expectApproxEqAbs(@as(f64, 0), e.rule.sharedPrice(2, false).?.row, 1e-9);
    }
    try t.expect(measured_rounds >= 2);
}

test "shared prices: unbatched priced heads cannot leak into the learned row slope" {
    const variable = "FZ_PRICE_BATCHED";
    const old = if (std.c.getenv(variable)) |v| try a.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (old) |v| {
            _ = env.setenv(variable, v, 1);
            a.free(v);
        } else _ = env.unsetenv(variable);
    }
    for ([_]bool{ false, true }) |batched| {
        try t.expectEqual(@as(c_int, 0), env.setenv(variable, if (batched) "1" else "0", 1));
        try run(batched, 3, 2);
        try run(batched, 0.2, 0.1);
    }
}
