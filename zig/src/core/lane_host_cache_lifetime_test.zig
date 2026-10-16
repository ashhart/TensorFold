//! Lookup ownership is returned on cancellation and on failures before a stream can be initialized.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const a = std.testing.allocator;

const Box = struct {
    reason: api.Reason = .failed,
    done: std.atomic.Value(bool) = .init(false),
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        if (e.* == .finished) {
            b.reason = e.finished.reason;
            b.done.store(true, .release);
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

const Snaps = struct {
    f: *lanes.fake.Fake,
    store: *pc.Store = undefined,
    host: *api.LaneHost = undefined,
    live: usize = 0,
    fn of(ctx: *anyopaque) *Snaps {
        return @ptrCast(@alignCast(ctx));
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(ctx: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        const s = of(ctx);
        const saved = try s.f.save(@ptrCast(@alignCast(owner.?)), at);
        s.live += 1;
        return saved;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ctx: *anyopaque, saved: pc.Saved) void {
        const s = of(ctx);
        s.live -= 1;
        s.f.drop(saved);
    }
    fn cancel(ctx: *anyopaque, stream: *lanes.Stream, _: usize) void {
        const s = of(ctx);
        const saved = stream.reuse.saved orelse return;
        s.store.expired(saved);
        s.host.engine().cancel(2);
    }
};

fn check(cancelled: bool) !void {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var snaps: Snaps = .{ .f = &target };
    var store = pc.Store.init(a, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
    host.cache = &store;
    snaps.store = &store;
    snaps.host = &host;
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const first: api.Request = .{ .prompt = &prompt, .max_tokens = 2, .history_len = 4 };
    var first_box: Box = .{};
    try host.engine().submit(1, &first, .{ .ctx = &first_box, .event = Box.event });
    try std.testing.expectEqual(api.Reason.length, try first_box.wait());
    try std.testing.expectEqual(@as(usize, 1), snaps.live);
    if (cancelled) {
        target.prefill_hook = Snaps.cancel;
        target.prefill_hook_ctx = &snaps;
    }
    const second: api.Request = .{ .prompt = &prompt, .max_tokens = 2, .history_len = 4, .structure = if (cancelled) null else .{ .kind = .json } };
    var second_box: Box = .{};
    try host.engine().submit(2, &second, .{ .ctx = &second_box, .event = Box.event });
    try std.testing.expectEqual(if (cancelled) api.Reason.cancelled else api.Reason.failed, try second_box.wait());
    if (!cancelled) {
        const entry = store.find(&prompt, &.{}, &.{}).?;
        try std.testing.expectEqual(@as(u32, 0), entry.borrowers);
        store.expired(entry.saved);
        store.collectExpired();
    }
    try std.testing.expectEqual(@as(usize, 0), snaps.live);
    try std.testing.expectEqual(@as(u64, 0), store.held);
}

test "a cancelled prefill releases its invalidated restore lookup" {
    try check(true);
}

test "failed stream initialization releases its restore lookup" {
    try check(false);
}
