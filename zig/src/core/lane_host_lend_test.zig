//! A lent kept state evicted before its job's first call: the job is looked up again and counted from what it fills.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const fill = @import("lane_host_fill.zig");
const a = std.testing.allocator;

const Box = struct {
    done: std.atomic.Value(bool) = .init(false),
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        if (e.* == .finished) @as(*Box, @ptrCast(@alignCast(ctx))).done.store(true, .release);
    }
    fn wait(b: *Box) !void {
        const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
        while (!b.done.load(.acquire)) {
            if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began > 2 * std.time.ns_per_s) return error.RequestTimedOut;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

const Snaps = struct {
    f: *lanes.fake.Fake,
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
    /// A backend's first call restores what the stream was offered.
    fn hook(_: *anyopaque, s: *lanes.Stream, filled: usize) void {
        if (filled == 0 and s.reuse.saved != null) s.cached = s.reuse.at;
    }
};

const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

test "a lent state evicted before the first call: the job fills cold, marks and promise planned again, counted" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var snaps: Snaps = .{ .f = &target };
    var store = pc.Store.init(a, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    const req: api.Request = .{ .prompt = &prompt, .max_tokens = 2, .history_len = 16 };
    { // the prompt's state kept at 16
        var h = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 });
        h.cache = &store;
        try h.start();
        defer h.stop();
        var box: Box = .{};
        try h.engine().submit(1, &req, .{ .ctx = &box, .event = Box.event });
        try box.wait();
    }
    target.prefill_hook = Snaps.hook;
    target.prefill_hook_ctx = &snaps;
    var h = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 1 }); // its steps run on this thread
    h.cache = &store;
    defer h.stop();
    var box: Box = .{};
    try h.engine().submit(2, &req, .{ .ctx = &box, .event = Box.event });
    try std.testing.expect(h.admitOne());
    const job = h.filling.items[0];
    try std.testing.expect(job.lent != null); // admitted on the kept state, lent back until its first call
    try std.testing.expectEqual(@as(u32, 16), job.stream.reuse.at);
    try std.testing.expectEqual(@as(usize, 0), job.marks.len); // a resumed pass keeps nothing up to its state
    // evicted before its first call
    const e = for (store.entries.items) |x| {
        if (x.serial == job.lent.?) break x;
    } else return error.NotKept;
    store.remove(std.mem.indexOfScalar(*pc.Entry, store.entries.items, e).?);
    const rows = fill.step(&h, 0);
    // its first call filled from the start: the step counts the rows it ran, not none
    try std.testing.expect(rows != std.math.maxInt(u64));
    try std.testing.expectEqual(@as(u64, job.stream.fill_at.load(.monotonic)), rows);
    try std.testing.expectEqual(@as(u32, 0), job.stream.reuse.at);
    try std.testing.expect(job.load.sure and job.load.hint == 0); // the load ledger counts its chunks
    try std.testing.expect(std.mem.indexOfScalar(u32, job.marks, 16) != null); // a cold pass keeps its state again
    try std.testing.expect(job.cache_promise.slots > 0); // and is promised the room for it
    try h.start(); // the rest of the request on the lane thread
    try box.wait();
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len); // kept again at 16
}
