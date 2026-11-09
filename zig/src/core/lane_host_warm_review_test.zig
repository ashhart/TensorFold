//! Warm queue bounds and a disk-held foreground head through the production lane host.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const ends = @import("lane_host_end.zig");
const a = std.testing.allocator;

const Box = struct {
    done: std.atomic.Value(bool) = .init(false),
    reason: api.Reason = .failed,
    host: ?*api.LaneHost = null,
    locked_callback: std.atomic.Value(bool) = .init(false),
    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        if (e.* != .finished) return;
        const b: *Box = @ptrCast(@alignCast(ptr));
        if (b.host) |h| {
            if (h.mutex.tryLock()) h.unlock() else b.locked_callback.store(true, .release);
        }
        b.reason = e.finished.reason;
        b.done.store(true, .release);
    }
    fn wait(b: *Box) !void {
        for (0..3000) |_| {
            if (b.done.load(.acquire)) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.WarmTimedOut;
    }
};

fn close(h: *api.LaneHost) void {
    if (h.thread == null) ends.closeAll(h);
    h.stop();
}

const Chunks = struct {
    count: std.atomic.Value(usize) = .init(0),
    fn hook(ptr: *anyopaque, s: *lanes.Stream, _: usize) void {
        if (s.max_new == 0) _ = @as(*Chunks, @ptrCast(@alignCast(ptr))).count.fetchAdd(1, .monotonic);
    }
};

const Mode = enum { cap, conversation, foreground, late, kept, kept_cuts, kept_modes, kept_expired };
fn queued(mode: Mode) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 3 };
    defer target.deinit();
    var chunks: Chunks = .{};
    target.prefill_hook = Chunks.hook;
    target.prefill_hook_ctx = &chunks;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.warm_quiet = std.time.ns_per_ms;
    var snapshots: @import("prompt_cache_fake.zig").Fake = .{ .gpa = a };
    var store = pc.Store.init(a, snapshots.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    defer close(&host);
    var words: [6][16]u32 = undefined;
    var requests: [6]api.Request = undefined;
    var replies: [6]Box = @splat(.{});
    const endpoint = mode == .kept or mode == .kept_cuts or mode == .kept_modes or mode == .kept_expired;
    const n: usize = if (endpoint) 1 else if (mode == .foreground or mode == .late) 2 else 6;
    for (&words, 0..) |*row, i| for (row, 0..) |*v, j| {
        v.* = @intCast(j + 1 + if (mode == .cap) 100 * i else 0);
    };
    for (requests[0..n], 0..) |*r, i| r.* = .{ .prompt = words[i][0..if (mode == .cap or endpoint) 16 else 4 + 2 * i], .max_tokens = 0, .background = true };
    if (mode == .foreground) {
        requests[1].max_tokens = 1;
        requests[1].background = false;
    }
    if (mode == .late) {
        requests[0] = .{ .prompt = &words[0], .max_tokens = 1 };
        requests[1].prompt = words[1][0..4];
    }
    if (endpoint) {
        host.cache = &store;
        _ = snapshots.pass(&store, &words[0], .{});
        if (mode == .kept_cuts) store.rules.cut_sensitive = true;
        try std.testing.expect(store.keepBorn(&words[0], 16, 16, null, if (mode == .kept_cuts) &.{ 4, 8, 16 } else &.{}, &.{}));
        if (mode == .kept_cuts) requests[0].chunks = &.{ 4, 12, 16 };
        if (mode == .kept_modes) {
            host.decoded_rows = true;
            requests[0].decode_spans = &.{.{ 0, 4 }};
        }
        if (mode == .kept_expired) store.entries.items[0].expired = true;
    }
    for (replies[0..n]) |*b| b.host = &host;
    for (requests[0..n], replies[0..n], 0..) |*r, *b, i| try host.engine().submit(i + 1, r, .{ .ctx = b, .event = Box.event });
    const wanted: usize = switch (mode) {
        .cap => 2,
        else => 1,
    };
    try std.testing.expectEqual(wanted, host.queued.items.len);
    if (mode == .cap) {
        try std.testing.expectEqual(@as(api.Id, 5), host.queued.items[0].id);
        try std.testing.expectEqual(@as(api.Id, 6), host.queued.items[1].id);
    } else if (mode == .conversation or mode == .foreground) {
        try std.testing.expectEqual(@as(api.Id, n), host.queued.items[0].id);
    } else if (mode == .late) try std.testing.expectEqual(@as(api.Id, 1), host.queued.items[0].id);
    try host.start();
    for (replies[0..n]) |*b| {
        try b.wait();
        try std.testing.expect(!b.locked_callback.load(.acquire));
    }
    try std.testing.expectEqual(@as(usize, switch (mode) {
        .cap => 6,
        .conversation, .kept_cuts, .kept_modes, .kept_expired => 3,
        else => 0,
    }), chunks.count.load(.acquire));
    if (mode == .kept) try std.testing.expectEqual(@as(u64, 1), store.counts.kept);
}

test "warm review: the lane-count cap drops oldest unrelated warm requests" {
    try queued(.cap);
}
test "warm review: only the newest queued turn of a conversation runs" {
    try queued(.conversation);
}
test "warm review: a foreground continuation supersedes an older queued warm pass" {
    try queued(.foreground);
}
test "warm review: a late warm pass stays out behind its queued foreground continuation" {
    try queued(.late);
}
test "warm review: an exact end already kept skips all backend prefill work" {
    try queued(.kept);
}

const Gate = struct {
    var entered: std.atomic.Value(bool) = .init(false);
    var release: std.atomic.Value(bool) = .init(false);
    var left: std.atomic.Value(bool) = .init(false);
    fn read(_: *@import("rows_prefetch.zig").Prefetch, _: *@import("rows_prefetch.zig").Read) anyerror!void {
        entered.store(true, .release);
        while (!release.load(.acquire)) try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        left.store(true, .release);
        return error.PlantedReadFailure;
    }
};

test "warm review: a fetch-held foreground head never spins a deferred warm job" {
    const fixture = @import("rows_async_fixture.zig");
    const x = try fixture.Fixture.open("warm-held", .evicted, true, 1 << 20);
    defer x.close();
    const words: [20]u32 = @splat(7);
    try x.keep(&words, 8);
    x.s.remove(0);
    try std.testing.expect(x.r.states.items.len > 0);
    const reader = try @import("rows_prefetch.zig").make(x.r, std.testing.io, .{});
    Gate.entered.store(false, .release);
    Gate.release.store(false, .release);
    Gate.left.store(false, .release);
    reader.reader = Gate.read;
    defer {
        Gate.release.store(true, .release);
        fixture.Gate.wait(&Gate.left) catch unreachable;
    }
    const name = reader.start(x.r, 0) orelse return error.MissingRead;
    try fixture.Gate.wait(&Gate.entered);
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    defer close(&host);
    host.cache = &x.s;
    host.warm_quiet = std.time.ns_per_ms;
    const foreground: api.Request = .{ .prompt = &words, .max_tokens = 1 };
    const warm: api.Request = .{ .prompt = &.{ 1, 2, 3, 4 }, .max_tokens = 0, .background = true };
    var f: Box = .{};
    var w: Box = .{};
    try host.engine().submit(1, &foreground, .{ .ctx = &f, .event = Box.event });
    try host.engine().submit(2, &warm, .{ .ctx = &w, .event = Box.event });
    host.queued.items[0].fetch = .{ .looked = true, .name = name };
    for (0..100) |_| switch (host.takeJob()) {
        .none => {},
        .job => |job| {
            _ = host.setup(job);
        },
        .failed => return error.UnexpectedFailure,
    };
    try std.testing.expect(host.load.requeued <= 1);
    try std.testing.expectEqual(@as(usize, 0), host.admitted.items.len);
}

test "warm review: a deferred oldest warm pass cannot overfill the queue" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 3 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    defer close(&host);
    host.warm_quiet = 100 * std.time.ns_per_ms;
    var replies: [4]Box = @splat(.{});
    const first: api.Request = .{ .prompt = &.{ 1, 2, 3, 4 }, .max_tokens = 0, .background = true };
    const second: api.Request = .{ .prompt = &.{ 5, 6, 7, 8 }, .max_tokens = 0, .background = true };
    const third: api.Request = .{ .prompt = &.{ 9, 10, 11, 12 }, .max_tokens = 0, .background = true };
    const real: api.Request = .{ .prompt = &.{ 13, 14, 15, 16 }, .max_tokens = 1 };
    try host.start();
    try host.engine().submit(1, &first, .{ .ctx = &replies[0], .event = Box.event });
    for (0..1000) |_| {
        host.lock();
        const taken = host.admitted.items.len == 1;
        host.unlock();
        if (taken) break;
        try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
    }
    host.lock();
    const taken = host.admitted.items.len == 1;
    host.unlock();
    try std.testing.expect(taken);
    try host.engine().submit(2, &second, .{ .ctx = &replies[1], .event = Box.event });
    try host.engine().submit(3, &third, .{ .ctx = &replies[2], .event = Box.event });
    try host.engine().submit(4, &real, .{ .ctx = &replies[3], .event = Box.event });
    for (&replies) |*b| try b.wait();
    try std.testing.expectEqual(api.Reason.cancelled, replies[0].reason);
    for (replies[1..]) |*b| try std.testing.expectEqual(api.Reason.length, b.reason);
}

test "warm review: a kept end with different arithmetic cuts still runs" {
    try queued(.kept_cuts);
}
test "warm review: a kept end with different decoded intervals still runs" {
    try queued(.kept_modes);
}
test "warm review: an expired kept end still runs" {
    try queued(.kept_expired);
}
