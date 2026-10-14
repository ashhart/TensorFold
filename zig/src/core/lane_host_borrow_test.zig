//! A queued prompt's looked-up state outlives later admissions: another request's reservation cannot free it first.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const a = std.testing.allocator;

const Reply = struct {
    mutex: std.Io.Mutex = .init,
    ids: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    reason: ?api.Reason = null,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        r.mutex.lockUncancelable(std.testing.io);
        defer r.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |c| r.cached = c,
            .tokens => |t| r.ids.appendSlice(a, t) catch {},
            .finished => |f| r.reason = f.reason,
            else => {},
        }
    }

    fn wait(r: *Reply) !api.Reason {
        const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
        while (true) {
            r.mutex.lockUncancelable(std.testing.io);
            const done = r.reason;
            r.mutex.unlock(std.testing.io);
            if (done) |reason| return reason;
            const now = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
            if (now - began > 2 * std.time.ns_per_s) return error.RequestTimedOut;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// The fake target's states, each as large as the whole budget, so the cache holds one at a time.
const Snaps = struct {
    f: *lanes.fake.Fake,

    fn of(ctx: *anyopaque) *Snaps {
        return @ptrCast(@alignCast(ctx));
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 100;
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
};

/// A `len`-token prompt's first chunk sends `request`, admitted while that prompt still waits to restore its state.
const Late = struct {
    engine: api.Engine,
    request: *const api.Request,
    reply: *Reply,
    len: usize,
    sent: bool = false,

    fn call(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
        const l: *Late = @ptrCast(@alignCast(ctx));
        if (l.sent or s.prompt().len != l.len) return;
        l.sent = true;
        l.engine.submit(3, l.request, .{ .ctx = l.reply, .event = Reply.event }) catch {};
    }
};

/// The fake target's greedy reply to `prompt` alone, `n` tokens.
fn fresh(prompt: []const u32, n: usize) ![]u32 {
    var h: std.ArrayList(u32) = .empty;
    defer h.deinit(a);
    try h.appendSlice(a, prompt);
    for (0..n) |_| try h.append(a, lanes.fake.next(h.items, null, h.items.len));
    return a.dupe(u32, h.items[prompt.len..]);
}

test "a queued prompt's kept state survives a later request's reservation until its own pass restores it" {
    var cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(a);
    var target: lanes.fake.Fake = .{ .gpa = a, .prefill_chunks = 3 }; // a prompt restores its state at its last chunk
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.stepped(), clock.clock());
    defer core.deinit();
    var snaps: Snaps = .{ .f = &target };
    const vtable: pc.Snapshots.VTable = .{
        .bytes = Snaps.bytes,
        .save = Snaps.save,
        .restore = Snaps.restore,
        .drop = Snaps.drop,
    };
    var store = pc.Store.init(a, .{ .ptr = &snaps, .vtable = &vtable }, .{ .min_prompt = 0 }, 100);
    defer store.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const e = host.engine();
    const pa = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const p2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }; // extends A's kept prefix
    const p3 = [_]u32{ 50, 51, 52, 53, 54, 55 }; // shares nothing, and its history mark needs A's room
    const first: api.Request = .{ .prompt = &pa, .max_tokens = 2, .history_len = 4 };
    const second: api.Request = .{ .prompt = &p2, .max_tokens = 8, .history_len = 6 };
    const third: api.Request = .{ .prompt = &p3, .max_tokens = 8, .history_len = 4 };
    var r1: Reply = .{};
    defer r1.ids.deinit(a);
    var r2: Reply = .{};
    defer r2.ids.deinit(a);
    var r3: Reply = .{};
    defer r3.ids.deinit(a);
    try e.submit(1, &first, .{ .ctx = &r1, .event = Reply.event });
    try std.testing.expectEqual(api.Reason.length, try r1.wait());
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len); // A, at 4 tokens, fills the budget
    var late: Late = .{ .engine = e, .request = &third, .reply = &r3, .len = p2.len };
    target.prefill_hook = Late.call;
    target.prefill_hook_ctx = &late;
    try e.submit(2, &second, .{ .ctx = &r2, .event = Reply.event });
    try std.testing.expectEqual(api.Reason.length, try r2.wait());
    try std.testing.expectEqual(api.Reason.length, try r3.wait());
    try std.testing.expect(late.sent);
    try std.testing.expectEqual(@as(?u32, 4), r2.cached); // it resumed A, which the third request's lookup left alone
    const want2 = try fresh(&p2, 8);
    defer a.free(want2);
    try std.testing.expectEqualSlices(u32, want2, r2.ids.items);
    const want3 = try fresh(&p3, 8);
    defer a.free(want3);
    try std.testing.expectEqualSlices(u32, want3, r3.ids.items);
}
