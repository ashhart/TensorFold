//! The real lane worker keeps copy-only jobs exact without requesting positive neural depths.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const a = std.testing.allocator;

const Target = struct {
    fake: lanes.fake.Fake = .{ .gpa = a, .probe_prompt = 120, .cycle_after = 0 },
    table: lanes.backend.Backend.VTable = undefined,
    positive: usize = 0,

    fn backend(t: *Target) lanes.backend.Backend {
        t.table = t.fake.backend().vtable.*;
        t.table.draft = draft;
        return .{ .ptr = &t.fake, .vtable = &t.table, .first_before_draft = true };
    }

    fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) !void {
        const inner: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const t: *Target = @fieldParentPtr("fake", inner);
        for (requests) |r| t.positive += @intFromBool(r.depth > 0);
        try inner.backend().draft(requests);
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?api.Reason = null,
    ids: [80]u32 = undefined,
    used: usize = 0,

    fn event(ptr: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ptr));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (ev.*) {
            .tokens => |tokens| for (tokens) |t| {
                if (b.used < b.ids.len) b.ids[b.used] = t;
                b.used += 1;
            },
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }

    fn wait(b: *Box) !void {
        for (0..10000) |_| {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |reason| return std.testing.expectEqual(api.Reason.length, reason);
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.WorkerTimedOut;
    }
};

test "root copy: the production lane host serves copied requests without positive head depths" {
    var m: lanes.Model = .{ .exact_width = 16, .batch_rows = 32, .hidden_rows = true, .context_copies = true, .copy_match = 3, .mtp = true, .speculate = true, .speculate_early = true, .drafts = 6, .draft_streams = true };
    m.root_copies = true;
    var cfg = try lanes.Config.init(a, m, 16, 15);
    defer cfg.deinit(a);
    var target: Target = .{};
    defer target.fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    host.min_match = 3;
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    var prompt: [120]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(11 + i % 3);
    const req: api.Request = .{ .prompt = &prompt, .max_tokens = 80 };
    var boxes: [2]Box = .{ .{}, .{} };
    for (&boxes, 1..) |*b, id| try host.engine().submit(@intCast(id), &req, .{ .ctx = b, .event = Box.event });
    for (&boxes) |*b| try b.wait();
    host.stop();
    stopped = true;
    for (&boxes) |*b| {
        try std.testing.expectEqual(@as(usize, 80), b.used);
        for (b.ids, 0..) |token, i| try std.testing.expectEqual(@as(u32, @intCast(11 + i % 3)), token);
    }
    try std.testing.expectEqual(@as(usize, 0), target.positive);
}
