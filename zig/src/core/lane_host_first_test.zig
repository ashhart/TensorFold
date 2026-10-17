//! Actual lane-host delivery keeps terminal jobs alive through the callback and never requeues an emitted first token.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const a = std.testing.allocator;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: usize = 0,
    logprobs: usize = 0,
    prefilled: usize = 0,
    finishes: usize = 0,
    reason: ?api.Reason = null,
    fn event(ptr: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (ev.*) {
            .tokens => |t| b.tokens += t.len,
            .logprobs => |r| b.logprobs += r.len,
            .prefilled => b.prefilled += 1,
            .finished => |f| {
                b.reason = f.reason;
                b.finishes += 1;
            },
            else => {},
        }
    }
    fn wait(b: *Box) !void {
        for (0..1000) |_| {
            b.mutex.lockUncancelable(std.testing.io);
            const done = b.reason != null;
            b.mutex.unlock(std.testing.io);
            if (done) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.TestTimedOut;
    }
};

const Target = struct {
    fake: lanes.fake.Fake = .{ .gpa = a },
    table: lanes.backend.Backend.VTable = undefined,
    drafts: usize = 0,
    releases: usize = 0,
    lane_memory: bool,
    observed: *Box,
    delivered_before_draft: bool = false,
    fn backend(t: *Target) lanes.backend.Backend {
        t.table = t.fake.backend().vtable.*;
        t.table.draft = draft;
        t.table.release = release;
        return .{ .ptr = &t.fake, .vtable = &t.table, .first_before_draft = true };
    }
    fn self(ptr: *anyopaque) *Target {
        const fake: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return @fieldParentPtr("fake", fake);
    }
    fn draft(ptr: *anyopaque, _: []const lanes.backend.DraftRequest) !void {
        const t = self(ptr);
        t.drafts += 1;
        t.observed.mutex.lockUncancelable(std.testing.io);
        t.delivered_before_draft = t.observed.tokens == 1 and t.observed.logprobs == 1;
        t.observed.mutex.unlock(std.testing.io);
        return if (t.lane_memory) error.LaneMemory else error.InitialDraftFailed;
    }
    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const t = self(ptr);
        t.releases += 1;
        t.fake.backend().release(s);
    }
};

fn check(lane_memory: bool) !void {
    var one: Box = .{};
    var other: Box = .{};
    var t: Target = .{ .lane_memory = lane_memory, .observed = &other };
    defer t.fake.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .hidden_rows = true, .mtp = true, .speculate = true, .drafts = 7 }, 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, t.backend(), clock.clock());
    defer core.deinit();
    var h = LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    const e = h.engine();
    const terminal: api.Request = .{ .prompt = &.{ 3, 1, 4 }, .max_tokens = 1, .logprobs = 2 };
    const failing: api.Request = .{ .prompt = &.{ 3, 1, 4 }, .max_tokens = 8, .logprobs = 2 };
    try e.submit(1, &terminal, .{ .ctx = &one, .event = Box.event });
    try e.submit(2, &failing, .{ .ctx = &other, .event = Box.event });
    try h.start();
    var stopped = false;
    defer if (!stopped) h.stop();
    try one.wait();
    try other.wait();
    h.stop();
    stopped = true;
    try std.testing.expectEqual(api.Reason.length, one.reason.?);
    try std.testing.expectEqual(api.Reason.failed, other.reason.?);
    try std.testing.expectEqual(@as(usize, 1), one.tokens);
    try std.testing.expectEqual(@as(usize, 1), other.tokens);
    try std.testing.expectEqual(@as(usize, 1), one.logprobs);
    try std.testing.expectEqual(@as(usize, 1), other.logprobs);
    try std.testing.expectEqual(@as(usize, 1), one.prefilled);
    try std.testing.expectEqual(@as(usize, 1), other.prefilled);
    try std.testing.expectEqual(@as(usize, 1), one.finishes);
    try std.testing.expectEqual(@as(usize, 1), other.finishes);
    try std.testing.expectEqual(@as(usize, 1), t.drafts);
    try std.testing.expectEqual(@as(usize, 2), t.releases);
    try std.testing.expect(t.delivered_before_draft);
}

test "first token and logprob sent once; terminal job survives another job's draft error" {
    try check(false);
}

test "LaneMemory after the committed handoff fails instead of resetting and requeueing the reply" {
    try check(true);
}

test "the host's sink can cancel on its first delivered token before drafting" {
    const Cancel = struct {
        fake: lanes.fake.Fake = .{ .gpa = a },
        host: *LaneHost = undefined,
        done: std.atomic.Value(bool) = .init(false),
        tokens: usize = 0,
        finishes: usize = 0,
        drafts: usize = 0,
        releases: usize = 0,
        reason: ?api.Reason = null,
        fn event(ctx: *anyopaque, id: api.Id, e: *const api.Event) void {
            const c: *@This() = @ptrCast(@alignCast(ctx));
            switch (e.*) {
                .tokens => |ts| {
                    c.tokens += ts.len;
                    c.host.engine().cancel(id);
                },
                .finished => |f| {
                    c.finishes += 1;
                    c.reason = f.reason;
                    c.done.store(true, .release);
                },
                else => {},
            }
        }
        fn self(ctx: *anyopaque) *@This() {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ctx));
            return @fieldParentPtr("fake", f);
        }
        fn draft(ctx: *anyopaque, _: []const lanes.backend.DraftRequest) !void {
            self(ctx).drafts += 1;
            return error.UnexpectedDraft;
        }
        fn release(ctx: *anyopaque, s: *lanes.Stream) void {
            const c = self(ctx);
            c.releases += 1;
            c.fake.backend().release(s);
        }
    };
    var c: Cancel = .{};
    defer c.fake.deinit();
    var be = c.fake.backend();
    var vt = be.vtable.*;
    vt.draft = Cancel.draft;
    vt.release = Cancel.release;
    be.vtable = &vt;
    be.first_before_draft = true;
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .hidden_rows = true, .mtp = true, .speculate = true, .drafts = 7 }, 8, 7);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, be, clock.clock());
    defer core.deinit();
    var host = LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
    c.host = &host;
    const req: api.Request = .{ .prompt = &.{ 3, 1, 4 }, .max_tokens = 8 };
    try host.engine().submit(1, &req, .{ .ctx = &c, .event = Cancel.event });
    try host.start();
    for (0..2000) |_| {
        if (c.done.load(.acquire)) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    host.stop();
    try std.testing.expect(c.done.load(.acquire));
    try std.testing.expectEqual(api.Reason.cancelled, c.reason.?);
    try std.testing.expectEqual(@as(usize, 1), c.tokens);
    try std.testing.expectEqual(@as(usize, 1), c.finishes);
    try std.testing.expectEqual(@as(usize, 0), c.drafts);
    try std.testing.expectEqual(@as(usize, 1), c.releases);
}
