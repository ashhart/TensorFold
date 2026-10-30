//! Recurring idle-unload behavior under the synchronized fake-engine fixture.
const std = @import("std");
const api = @import("engine_api");
const cli = @import("cli.zig");
const idle_unload = @import("idle_unload.zig");

test "the manager stays off for 0 and negative idle seconds" {
    var args: cli.Args = .{};
    args.idle_unload = 0;
    const re: idle_unload.Reopen = .{ .ctx = undefined, .open = undefined, .close_held = undefined, .replace = undefined };
    const m = idle_unload.start(std.testing.allocator, std.testing.io, args, re, "d", "t", null);
    try std.testing.expect(m == null);
    args.idle_unload = -5;
    try std.testing.expect(idle_unload.start(std.testing.allocator, std.testing.io, args, re, "d", "t", null) == null);
}

test "a one-second window parks the engine without a request; the timer does not re-close" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitParked(3000); // the window passed with no request: the park landed
    try std.testing.expectEqual(@as(u32, 1), h.closes.load(.acquire));
    h.sleepMs(1100); // a second window: the engine is already closed, the park does not re-close
    try std.testing.expectEqual(@as(u32, 1), h.closes.load(.acquire));
}

test "active work prevents the park; its end restarts the window" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    h.man.?.begin();
    h.sleepMs(1400); // a whole window with a request in flight: no park
    try std.testing.expectEqual(@as(u32, 0), h.closes.load(.acquire));
    h.man.?.end();
    try h.waitParked(3000); // the end re-armed the deadline: the park lands after the fresh window
    try std.testing.expectEqual(@as(u32, 1), h.closes.load(.acquire));
}

test "the wake reloads: the candidate opens, the stub is freed, and a second cycle works" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitUnloaded(3000);
    var problem: []const u8 = "";
    try h.reload(&problem);
    try std.testing.expectEqual(@as(u32, 1), h.opens.load(.acquire));
    try std.testing.expectEqual(@as(u32, 1), h.replaces.load(.acquire));
    try std.testing.expect(!h.man.?.isParked());
    try h.waitUnloaded(3000); // a second complete cycle
    try h.reload(&problem);
    try std.testing.expectEqual(@as(u32, 2), h.opens.load(.acquire));
    try std.testing.expectEqual(@as(u32, 2), h.closes.load(.acquire));
}

test "concurrent wake requests produce exactly one reload; both find the engine resident" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitUnloaded(3000);
    var a_err: ?anyerror = null;
    var b_err: ?anyerror = null;
    const Wa = struct {
        fn run(x: *Harness, e: *?anyerror) void {
            var p: []const u8 = "";
            x.reload(&p) catch |err| {
                e.* = err;
            };
        }
    };
    {
        const ta = try std.Thread.spawn(.{}, Wa.run, .{ &h, &a_err });
        defer ta.join();
        const tb = try std.Thread.spawn(.{}, Wa.run, .{ &h, &b_err });
        defer tb.join();
    }
    try std.testing.expectEqual(@as(u32, 1), h.opens.load(.acquire)); // exactly one reload opened the engine
    try std.testing.expect(a_err == null and b_err == null);
    try std.testing.expect(!h.man.?.isParked());
}

test "a failed open cleans up and leaves the server parked and retryable" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    h.fail_open = true;
    try h.waitUnloaded(3000);
    var problem: []const u8 = "";
    try std.testing.expectError(error.EngineOpenFailed, h.reload(&problem));
    try std.testing.expect(h.man.?.isParked()); // still parked: the next request retries
    h.fail_open = false;
    try h.reload(&problem); // the retry succeeds
    try std.testing.expectEqual(@as(u32, 1), h.opens.load(.acquire));
    try std.testing.expect(!h.man.?.isParked());
}

test "the parked stub refuses submits and reports unhealthy" {
    const Harness = @import("idle_unload_test.zig").Harness;
    var h: Harness = .{};
    defer h.deinit();
    try h.setup(1);
    try h.waitParked(3000);
    const engine = h.srv.?.engine;
    var probe: api.Request = .{ .prompt = &.{}, .max_tokens = 0 };
    try std.testing.expectError(error.Closed, engine.submit(1, &probe, .{ .ctx = undefined, .event = undefined }));
    var status: api.Status = .{};
    engine.status(&status, &.{});
    try std.testing.expect(!status.healthy);
    try std.testing.expect(engine.memory(false) == null);
}
