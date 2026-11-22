//! A stepped pass's restored prompt-cache entry goes back to the store after its first call, not at the pass's end.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const fill = @import("lane_host_fill.zig");
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
    prompt: []const u32 = &.{},
    at_second: ?u32 = null, // the entry's borrowers when the pass makes its second call
    fn of(ctx: *anyopaque) *Snaps {
        return @ptrCast(@alignCast(ctx));
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(ctx: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
        return of(ctx).f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }
    fn drop(ctx: *anyopaque, saved: pc.Saved) void {
        of(ctx).f.drop(saved);
    }
    /// The fake target's step: a real backend restores in its first call (cached set there), then each chunk.
    fn hook(ctx: *anyopaque, s: *lanes.Stream, filled: usize) void {
        const x = of(ctx);
        if (s.reuse.saved == null and filled == 0) return;
        if (filled == 0) s.cached = s.reuse.at;
        if (filled == 1) if (x.store.find(x.prompt, &.{}, &.{})) |e| {
            x.at_second = e.borrowers;
        };
    }
};

test "a stepped pass gives its restored entry back after its first call" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var snaps: Snaps = .{ .f = &target, .prompt = &prompt };
    var store = pc.Store.init(a, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
    host.cache = &store;
    snaps.store = &store;
    try host.start();
    defer host.stop();
    const first: api.Request = .{ .prompt = &prompt, .max_tokens = 2, .history_len = 8 };
    var first_box: Box = .{};
    try host.engine().submit(1, &first, .{ .ctx = &first_box, .event = Box.event });
    _ = try first_box.wait();
    target.prefill_hook = Snaps.hook;
    target.prefill_hook_ctx = &snaps;
    var second_box: Box = .{};
    try host.engine().submit(2, &first, .{ .ctx = &second_box, .event = Box.event });
    _ = try second_box.wait();
    try std.testing.expectEqual(@as(?u32, 0), snaps.at_second); // no longer pinned while the pass's later chunks run
}

test "a pass's lookup is reported once its restore is settled: restored, refused, or none offered" {
    var saved: u8 = 0;
    const prompt = [_]u32{ 1, 2, 3 };
    var s = try lanes.Stream.init(a, .{ .id = "s", .prompt = &prompt, .max_new = 1, .reuse = .{ .saved = &saved, .at = 2 } });
    defer s.deinit(a);
    try std.testing.expect(!fill.restoreSettled(&s)); // not yet restored: the entry stays borrowed
    s.cached = 2;
    try std.testing.expect(fill.restoreSettled(&s));
    s.cached = 0;
    s.reuse_failed = true;
    try std.testing.expect(fill.restoreSettled(&s));
}
