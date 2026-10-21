//! The lane host's decision hooks: queued passes on an idle engine, in the decider's own words.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Id = api.Id;
const Event = api.Event;
const Reason = api.Reason;
const Request = api.Request;

/// A decider that echoes its task: score gives each label its id as a logit; decide adds the branch's length.
const Record = struct {
    core: *lanes.Engine,
    calls: usize = 0,
    idle: bool = true, // activeCount at every pass: false once a pass ran beside a live stream
    live_ok: bool = false, // what `ready` answers when the host asks beside live work (false: idle only)
    fail: ?error{ Failed, Refused } = null,

    fn ready(ctx: *anyopaque) bool {
        const r: *Record = @ptrCast(@alignCast(ctx));
        return r.live_ok;
    }

    fn run(ctx: *anyopaque, task: *api.DecisionTask) error{ Failed, Refused }!void {
        const r: *Record = @ptrCast(@alignCast(ctx));
        defer r.calls += 1;
        r.idle = r.idle and r.core.activeCount() == 0;
        if (r.fail) |e| return e;
        switch (task.*) {
            .score => |*s| {
                for (s.labels, s.logits) |label, *x| x.* = @floatFromInt(label);
                s.out = if (s.mass) 42.5 else 0;
            },
            .decide => |d| {
                var at: usize = 0;
                for (d.branches, d.counts) |b, n| {
                    for (d.labels[at..][0..n], d.logits[at..][0..n]) |label, *x| x.* = @floatFromInt(label + b.len);
                    d.read += b.len;
                    at += n;
                }
                d.read += d.trunk.len;
            },
        }
    }
};

fn config(gpa: std.mem.Allocator) !lanes.Config {
    return lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
}

test "a lane host without a decider keeps score/decide absent; one with it serves both" {
    const gpa = std.testing.allocator;
    var cfg = try config(gpa);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    try std.testing.expect(host.engine().vtable.score == null);
    try std.testing.expect(host.engine().vtable.decide == null);
    var record = Record{ .core = &core };
    host.decider = .{ .ctx = &record, .run = Record.run };
    const e = host.engine();
    try std.testing.expect(e.vtable.score != null);
    try std.testing.expect(e.vtable.decide != null);
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const labels = [_]u32{ 5, 9 };
    var logits = [_]f64{ 0, 0 };
    try std.testing.expectEqual(@as(f64, 42.5), try e.score(&prompt, &labels, &logits, true));
    try std.testing.expectEqualSlices(f64, &.{ 5, 9 }, &logits);
    const branches = [_][]const u32{ &.{ 4, 4 }, &.{6} };
    const counts = [_]usize{ 1, 1 };
    const wanted = [_]u32{ 30, 40 };
    var out = [_]f64{ 0, 0 };
    var d = api.Decision{ .trunk = &prompt, .branches = &branches, .counts = &counts, .labels = &wanted, .logits = &out };
    try e.decide(&d);
    try std.testing.expectEqualSlices(f64, &.{ 32, 41 }, &out);
    try std.testing.expectEqual(@as(usize, prompt.len + 3), d.read);
    try std.testing.expectEqual(@as(usize, 2), record.calls);
    try std.testing.expect(record.idle);
}

test "a decider's refusal and failure reach the requester" {
    const gpa = std.testing.allocator;
    var cfg = try config(gpa);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var record = Record{ .core = &core };
    host.decider = .{ .ctx = &record, .run = Record.run };
    try host.start();
    defer host.stop();
    const e = host.engine();
    const prompt = [_]u32{ 2, 7, 1, 8 };
    const labels = [_]u32{5};
    var logits = [_]f64{0};
    const branches = [_][]const u32{&.{4}};
    const counts = [_]usize{1};
    var d = api.Decision{ .trunk = &prompt, .branches = &branches, .counts = &counts, .labels = &labels, .logits = &logits };
    record.fail = error.Refused;
    try std.testing.expectError(error.Refused, e.decide(&d));
    try std.testing.expectError(error.Failed, e.score(&prompt, &labels, &logits, false)); // a score has no Refused
    record.fail = error.Failed;
    try std.testing.expectError(error.Failed, e.decide(&d));
    try std.testing.expectError(error.Failed, e.score(&prompt, &labels, &logits, false));
}

test "a queued decision runs once the live stream ends, and the text reply still lands" {
    const gpa = std.testing.allocator;
    var cfg = try config(gpa);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var record = Record{ .core = &core };
    host.decider = .{ .ctx = &record, .run = Record.run };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var box: Box = .{};
    const request: Request = .{ .prompt = &prompt, .max_tokens = 64 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    const labels = [_]u32{5};
    var logits = [_]f64{0};
    // whatever round the reply is on, the pass waits for it to leave the engine
    try std.testing.expectEqual(@as(f64, 0), try e.score(&prompt, &labels, &logits, false));
    try std.testing.expectEqualSlices(f64, &.{5}, &logits);
    while (true) {
        box.mutex.lockUncancelable(std.testing.io);
        const done = box.done;
        box.mutex.unlock(std.testing.io);
        if (done) |r| {
            try std.testing.expectEqual(Reason.length, r);
            break;
        }
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqual(@as(usize, 64), box.tokens);
    try std.testing.expect(record.idle);
    try std.testing.expectEqual(@as(usize, 1), record.calls);
}

test "a decider that can take a slot beside the rounds answers a queued decision while a reply is live" {
    const gpa = std.testing.allocator;
    var cfg = try config(gpa);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 1 });
    var record = Record{ .core = &core, .live_ok = true };
    host.decider = .{ .ctx = &record, .run = Record.run, .ready = Record.ready };
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        done: ?Reason = null,
        tokens: usize = 0,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .tokens => |t| b.tokens += t.len,
                .finished => |f| b.done = f.reason,
                else => {},
            }
        }
        fn read(b: *@This()) struct { done: ?Reason, tokens: usize } {
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            return .{ .done = b.done, .tokens = b.tokens };
        }
    };
    const prompt = [_]u32{ 2, 7, 1, 8 };
    var box: Box = .{};
    const request: Request = .{ .prompt = &prompt, .max_tokens = 1_000_000 };
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &box, .event = Box.event });
    // the reply is live (its first tokens landed) before the decision is queued: it must not wait for the end
    while (box.read().tokens == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    const labels = [_]u32{5};
    var logits = [_]f64{0};
    try std.testing.expectEqual(@as(f64, 0), try e.score(&prompt, &labels, &logits, false));
    try std.testing.expectEqualSlices(f64, &.{5}, &logits);
    try std.testing.expect(!record.idle); // the pass ran beside the reply, not after it
    try std.testing.expectEqual(@as(usize, 1), record.calls);
    e.cancel(1);
    while (box.read().done == null) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expectEqual(Reason.cancelled, box.read().done.?);
}
