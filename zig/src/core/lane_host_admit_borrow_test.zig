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

/// Two prompts with kept states (a, b) and one without (miss), shorter than b past b's kept rows.
const pa = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
const pb: [40]u32 = blk: { // long past its kept state: the shorter miss fills first
    var x: [40]u32 = undefined;
    for (&x, 21..) |*v, i| v.* = @intCast(i);
    break :blk x;
};
const miss = [_]u32{ 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52 };

/// What the passes' first calls saw.
const Seen = struct {
    firsts: usize = 0,
    resumed: usize = 0,
    beside: u32 = 0, // the most borrowers another prompt's kept state had while a pass made its first call
    filling: usize = 0, // admitted fills when the first of the first calls ran
};

const Snaps = struct {
    f: *lanes.fake.Fake,
    store: *pc.Store = undefined,
    host: ?*api.LaneHost = null,
    seen: Seen = .{},
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
    /// A pass's first call, where a backend opens its cache and makes room: no other kept state may be pinned.
    fn hook(ctx: *anyopaque, s: *lanes.Stream, filled: usize) void {
        const x = of(ctx);
        if (filled != 0) return;
        if (x.seen.firsts == 0) x.seen.filling = x.host.?.filling.items.len;
        x.seen.firsts += 1;
        if (s.reuse.saved != null) {
            s.cached = s.reuse.at; // the backend restores in its first call
            x.seen.resumed += 1;
        }
        for ([_][]const u32{ &pa, &pb }) |other| if (!std.mem.eql(u32, s.prompt()[0..4], other[0..4])) {
            if (x.store.find(other, &.{}, &.{})) |e| x.seen.beside = @max(x.seen.beside, e.borrowers);
        };
    }
};

fn host(core: *lanes.Engine, store: *pc.Store, n: u32) api.LaneHost {
    var h = api.LaneHost.init(a, std.testing.io, core, .{ .lanes = n });
    h.cache = store;
    return h;
}

/// `a` and `b` kept one at a time, then `queued` submitted together before a two-lane host starts.
fn run(queued: []const []const u32) !Seen {
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
    snaps.store = &store;
    {
        var h = host(&core, &store, 1);
        try h.start();
        defer h.stop();
        for ([_][]const u32{ &pa, &pb }, 1..) |p, id| {
            const r: api.Request = .{ .prompt = p, .max_tokens = 2, .history_len = 8 };
            var box: Box = .{};
            try h.engine().submit(id, &r, .{ .ctx = &box, .event = Box.event });
            try box.wait();
        }
    }
    target.prefill_hook = Snaps.hook;
    target.prefill_hook_ctx = &snaps;
    var h = host(&core, &store, 2);
    snaps.host = &h;
    var reqs: [2]api.Request = undefined;
    var boxes: [2]Box = .{ .{}, .{} };
    for (queued, reqs[0..queued.len], boxes[0..queued.len], 3..) |p, *r, *box, id| {
        r.* = .{ .prompt = p, .max_tokens = 2, .history_len = 8 };
        try h.engine().submit(id, r, .{ .ctx = box, .event = Box.event }); // queued before the host starts
    }
    try h.start();
    defer h.stop();
    for (boxes[0..queued.len]) |*box| try box.wait();
    return snaps.seen;
}

test "streams admitted together hold no kept state borrowed while another's first call makes room" {
    const seen = try run(&.{ &pa, &pb });
    try std.testing.expectEqual(@as(usize, 2), seen.resumed); // both resumed their kept states, borrowed again at their calls
    try std.testing.expectEqual(@as(u32, 0), seen.beside);
}

test "a cache miss queued ahead of a hit opens with the hit's kept state unpinned" {
    const seen = try run(&.{ &miss, &pb });
    try std.testing.expectEqual(@as(usize, 2), seen.firsts);
    try std.testing.expectEqual(@as(usize, 1), seen.resumed); // the hit still resumes
    try std.testing.expectEqual(@as(u32, 0), seen.beside); // the old admission hold let the miss admit the hit borrowed beside it
}

test "a prompt queued behind a hit not yet begun is admitted with it: no new admission wait" {
    const seen = try run(&.{ &pb, &miss });
    try std.testing.expectEqual(@as(usize, 2), seen.filling); // both admitted before either's first call, as on main
    try std.testing.expectEqual(@as(u32, 0), seen.beside);
}

test "the store borrows an entry again by its serial, and not once it has expired" {
    var target: lanes.fake.Fake = .{ .gpa = a };
    defer target.deinit();
    var snaps: Snaps = .{ .f = &target };
    var store = pc.Store.init(a, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    var x: u8 = 0;
    const e = try a.create(pc.Entry);
    e.* = .{ .tokens = try a.dupe(u32, &pa), .at = 8, .saved = &x, .bytes = 1, .born = 12, .used = 0, .last = &.{}, .serial = 7 };
    try store.entries.append(a, e);
    defer {
        a.free(e.tokens);
        a.destroy(e);
        store.entries.clearRetainingCapacity();
    }
    try std.testing.expect(store.reborrow(8) == null); // no entry has that serial
    try std.testing.expectEqual(e, store.reborrow(7).?);
    try std.testing.expectEqual(@as(u32, 1), e.borrowers);
    e.borrowers = 0;
    e.expired = true; // gone from lookups: a lent job fills from the start instead
    try std.testing.expect(store.reborrow(7) == null);
    try std.testing.expectEqual(@as(u32, 0), e.borrowers);
}
