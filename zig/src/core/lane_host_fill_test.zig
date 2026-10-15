//! The order a lane host fills waiting prompts in: foreground first, then the shortest (lane_host_fill.zig).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

test "a lane host fills the shortest waiting prompt first, so a short request waits on no long one" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Order = struct {
        mutex: std.Io.Mutex = .init,
        prefilled: [2]Id = .{ 0, 0 },
        count: usize = 0,
        done: usize = 0,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(std.testing.io);
            defer o.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => {
                    o.prefilled[o.count] = id;
                    o.count += 1;
                },
                .finished => o.done += 1,
                else => {},
            }
        }
    };
    var order: Order = .{};
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const short = [_]u32{ 2, 7, 1, 8 };
    const first: Request = .{ .prompt = &long, .max_tokens = 4 };
    const second: Request = .{ .prompt = &short, .max_tokens = 4 };
    const e = host.engine();
    // both wait before the engine thread starts: it admits them together and picks between them
    try e.submit(1, &first, .{ .ctx = &order, .event = Order.event });
    try e.submit(2, &second, .{ .ctx = &order, .event = Order.event });
    try host.start();
    defer host.stop();
    while (true) {
        order.mutex.lockUncancelable(std.testing.io);
        const d = order.done;
        order.mutex.unlock(std.testing.io);
        if (d == 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqualSlices(Id, &.{ 2, 1 }, &order.prefilled);
}

test "a foreground prompt arriving while a background one fills takes the next chunk" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const Order = struct {
        mutex: std.Io.Mutex = .init,
        prefilled: [2]Id = .{ 0, 0 },
        count: usize = 0,
        done: usize = 0,
        fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.mutex.lockUncancelable(std.testing.io);
            defer o.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => {
                    o.prefilled[o.count] = id;
                    o.count += 1;
                },
                .finished => o.done += 1,
                else => {},
            }
        }
    };
    var order: Order = .{};
    const short = [_]u32{ 2, 7, 1, 8 };
    const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
    const background: Request = .{ .prompt = &short, .max_tokens = 4, .background = true };
    const foreground: Request = .{ .prompt = &long, .max_tokens = 4 };
    const e = host.engine();
    // the background prompt's first chunk sends the foreground request: it is admitted while the other fills
    const Arrive = struct {
        engine: api.Engine,
        request: *const Request,
        order: *Order,
        sent: bool = false,
        fn call(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
            const a: *@This() = @ptrCast(@alignCast(ctx));
            if (a.sent) return;
            a.sent = true;
            a.engine.submit(2, a.request, .{ .ctx = a.order, .event = Order.event }) catch {};
        }
    };
    var arrive: Arrive = .{ .engine = e, .request = &foreground, .order = &order };
    target.prefill_hook = Arrive.call;
    target.prefill_hook_ctx = &arrive;
    try e.submit(1, &background, .{ .ctx = &order, .event = Order.event });
    try host.start();
    defer host.stop();
    while (true) {
        order.mutex.lockUncancelable(std.testing.io);
        const d = order.done;
        order.mutex.unlock(std.testing.io);
        if (d == 2) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqualSlices(Id, &.{ 2, 1 }, &order.prefilled);
}
