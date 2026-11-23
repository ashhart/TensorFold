//! Filling streams admitted together: none holds its kept state borrowed while another's first call makes room.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const a = std.testing.allocator;

const Box = struct {
    done: std.atomic.Value(bool) = .init(false),
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        if (e.* == .finished) b.done.store(true, .release);
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
    store: *pc.Store = undefined,
    prompts: [2][]const u32,
    beside: u32 = 0, // the most borrowers the other prompt's entry had while a pass made its first call
    firsts: usize = 0,
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
    /// A pass's first call, where a backend opens its cache and makes room: the other prompt's entry must be free.
    fn hook(ctx: *anyopaque, s: *lanes.Stream, filled: usize) void {
        const x = of(ctx);
        if (filled != 0 or s.reuse.saved == null) return;
        s.cached = s.reuse.at;
        x.firsts += 1;
        const other = if (std.mem.eql(u32, s.prompt()[0..4], x.prompts[0][0..4])) x.prompts[1] else x.prompts[0];
        if (x.store.find(other, &.{}, &.{})) |e| x.beside = @max(x.beside, e.borrowers);
    }
};

fn host(core: *lanes.Engine, store: *pc.Store, n: u32) api.LaneHost {
    var h = api.LaneHost.init(a, std.testing.io, core, .{ .lanes = n });
    h.cache = store;
    return h;
}

test "streams admitted together hold no kept state borrowed while another's first call makes room" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    const pa = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const pb = [_]u32{ 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32 };
    var snaps: Snaps = .{ .f = &target, .prompts = .{ &pa, &pb } };
    var store = pc.Store.init(a, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    snaps.store = &store;
    const reqs = [2]api.Request{ .{ .prompt = &pa, .max_tokens = 2, .history_len = 8 }, .{ .prompt = &pb, .max_tokens = 2, .history_len = 8 } };
    { // each prompt's state kept, one at a time
        var h = host(&core, &store, 1);
        try h.start();
        defer h.stop();
        for (&reqs, 1..) |*r, id| {
            var box: Box = .{};
            try h.engine().submit(id, r, .{ .ctx = &box, .event = Box.event });
            try box.wait();
        }
    }
    target.prefill_hook = Snaps.hook;
    target.prefill_hook_ctx = &snaps;
    var h = host(&core, &store, 2);
    var boxes: [2]Box = .{ .{}, .{} };
    for (&reqs, &boxes, 3..) |*r, *box, id| try h.engine().submit(id, r, .{ .ctx = box, .event = Box.event }); // both queued at once
    try h.start();
    defer h.stop();
    for (&boxes) |*box| try box.wait();
    try std.testing.expectEqual(@as(usize, 2), snaps.firsts); // both resumed their kept states
    try std.testing.expectEqual(@as(u32, 0), snaps.beside); // the other's entry was not pinned beside either first call
}
