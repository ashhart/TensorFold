//! Status reads stay lifetime-safe: a finishing job has left the host's lists before its event fires.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Engine = api.Engine;

const short_prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };

// The finished event fires inside `finish`, before the host frees the job: this is the racing poll.
test "status at a request's finished event no longer counts the job that just ended" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 2 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Watch = struct {
        engine: Engine = undefined,
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        running_at_finish: usize = 0,
        fill_at_finish: bool = false,

        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            w.mutex.lockUncancelable(std.testing.io);
            defer w.mutex.unlock(std.testing.io);
            switch (e.*) {
                .finished => |f| {
                    var status: api.Status = .{};
                    w.engine.status(&status, &.{}); // the poll that raced with this completion
                    w.running_at_finish = status.running;
                    w.fill_at_finish = status.fill_tokens_left != null;
                    w.done = f.reason;
                },
                else => {},
            }
        }
        fn wait(w: *@This()) Reason {
            while (true) {
                w.mutex.lockUncancelable(std.testing.io);
                const d = w.done;
                w.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    var watch: Watch = .{};
    watch.engine = host.engine();
    const request: Request = .{ .prompt = &short_prompt, .max_tokens = 4 };
    try watch.engine.submit(1, &request, .{ .ctx = &watch, .event = Watch.event });
    try std.testing.expectEqual(Reason.length, watch.wait());
    try std.testing.expectEqual(@as(usize, 0), watch.running_at_finish); // it left `admitted` before the event
    try std.testing.expect(!watch.fill_at_finish); // and its fill progress with it
    var status: api.Status = .{};
    watch.engine.status(&status, &.{});
    try std.testing.expectEqual(@as(u32, 0), status.running); // the host is idle: nothing is listed
}

// A second request runs while the first completes: the poll at its finished event counts the survivor.
test "status at one request's finished event still counts the other request running" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 2 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const Watch = struct {
        engine: Engine = undefined,
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        running_at_finish: usize = 0,
        fill_at_finish: bool = false,

        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            w.mutex.lockUncancelable(std.testing.io);
            defer w.mutex.unlock(std.testing.io);
            switch (e.*) {
                .finished => |f| {
                    var status: api.Status = .{};
                    w.engine.status(&status, &.{});
                    w.running_at_finish = status.running;
                    w.fill_at_finish = status.fill_tokens_left != null;
                    w.done = f.reason;
                },
                else => {},
            }
        }
        fn wait(w: *@This()) Reason {
            while (true) {
                w.mutex.lockUncancelable(std.testing.io);
                const d = w.done;
                w.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    var first: Watch = .{};
    first.engine = host.engine();
    var second: Watch = .{};
    second.engine = host.engine();
    const quick: Request = .{ .prompt = &short_prompt, .max_tokens = 1 };
    const long: Request = .{ .prompt = &short_prompt, .max_tokens = 100000 };
    try second.engine.submit(1, &long, .{ .ctx = &second, .event = Watch.event });
    try first.engine.submit(2, &quick, .{ .ctx = &first, .event = Watch.event });
    try std.testing.expectEqual(Reason.length, first.wait());
    try std.testing.expectEqual(@as(usize, 1), first.running_at_finish); // only the long one
    try std.testing.expect(!first.fill_at_finish); // the long one decodes: it fills nothing
    var status: api.Status = .{};
    second.engine.status(&status, &.{});
    try std.testing.expectEqual(@as(u32, 1), status.running);
    second.engine.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, second.wait());
}

test "status polls through a cancellation, a failed admission and a requeue stay sane" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .memory_lanes = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    try host.start();
    defer host.stop();
    const e = host.engine();
    const Poller = struct {
        engine: Engine,
        stop: std.atomic.Value(bool) = .init(false),
        failed: std.atomic.Value(bool) = .init(false),

        fn poll(ctx: *anyopaque) void {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            while (!p.stop.load(.acquire)) {
                var status: api.Status = .{};
                p.engine.status(&status, &.{});
                if (status.running > 2) p.failed.store(true, .release); // only two lanes exist
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    var poller: Poller = .{ .engine = e };
    const thread = try std.Thread.spawn(.{}, Poller.poll, .{&poller});
    defer {
        poller.stop.store(true, .release);
        thread.join();
    }
    const running: Request = .{ .prompt = &short_prompt, .max_tokens = 100000 };
    const small: Request = .{ .prompt = &short_prompt, .max_tokens = 4 };
    var a = try gpa.create(Box);
    defer gpa.destroy(a);
    a.* = .{};
    try e.submit(1, &running, .{ .ctx = a, .event = Box.event });
    var b = try gpa.create(Box);
    defer gpa.destroy(b);
    b.* = .{};
    try e.submit(2, &small, .{ .ctx = b, .event = Box.event }); // no memory for a second lane: it waits
    std.Io.sleep(std.testing.io, .fromMilliseconds(20), .awake) catch {};
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait()); // admitted once the lane freed, then ran to its end
    try std.testing.expect(!poller.failed.load(.acquire));
    var status: api.Status = .{};
    e.status(&status, &.{});
    try std.testing.expectEqual(@as(u32, 0), status.running);
    try std.testing.expect(status.fill_prompt_tokens == null and status.fill_tokens_left == null);
}

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    message: []const u8 = "",
    tokens: usize = 0,
    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| {
                b.done = f.reason;
                b.message = f.message;
            },
            else => {},
        }
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};
