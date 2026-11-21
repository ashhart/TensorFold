//! #1594 step 2b, the producer's side under the reservation guard: a refused keep at the cut costs only the state.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const fixture = @import("lane_remote_test.zig");
const Exp = fixture.Exp;
const Kept = fixture.Kept;
const six = fixture.six;
const t = std.testing;

/// Kept's copies, sized by position: a fresh copy at the cut outweighs the state below it that it could displace.
const Sized = struct {
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 60 + at;
    }
};

/// Each pass's first chunk in turn: where it resumed, on which copy, and how many kept states were borrowed then.
const Passes = struct {
    store: *const pc.Store,
    n: usize = 0,
    at: [2]u32 = .{ 0, 0 },
    remote: [2]bool = .{ false, false },
    borrowed: [2]u32 = .{ 0, 0 },

    fn chunk(ctx: *anyopaque, s: *lanes.Stream, n: usize) void {
        const p: *Passes = @ptrCast(@alignCast(ctx));
        if (n != 0 or p.n >= 2) return;
        var borrowed: u32 = 0;
        for (p.store.entries.items) |e| borrowed += e.borrowers;
        p.at[p.n], p.remote[p.n], p.borrowed[p.n] = .{ s.reuse.at, s.reuse.remote, borrowed };
        p.n += 1;
    }
};

/// What one request heard.
const Reply = struct {
    tokens: usize = 0,
    done: std.atomic.Value(bool) = .init(false),

    fn event(ctx: *anyopaque, _: api.Id, ev: *const api.Event) void {
        const r: *Reply = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .tokens => |ts| r.tokens += ts.len,
            .finished => r.done.store(true, .release),
            else => {},
        }
    }
};

var none: u8 = 0;

test "handoff: a repeat whose keep at the cut is refused leaks no borrow and no promise, and the remote copy serves" {
    // the remote copy at the cut and the engine's own state below it fill all but one byte of the budget
    var store = pc.Store.init(t.allocator, .{ .ptr = &none, .vtable = &.{ .bytes = Sized.bytes, .save = Kept.save, .restore = Kept.restore, .drop = Kept.drop } }, .{ .min_prompt = 0 }, 63 + 65 + 1);
    defer store.deinit();
    try t.expect(store.keep(&six, 3, null, &.{}, &.{}));
    store.keep_remote = true;
    try t.expect(store.keep(&six, 5, null, &.{}, &.{}));
    store.keep_remote = false;
    var cfg = try lanes.Config.init(t.allocator, .{}, 1, 0);
    defer cfg.deinit(t.allocator);
    var passes: Passes = .{ .store = &store };
    var fake: lanes.fake.Fake = .{ .gpa = t.allocator, .prefill_hook = Passes.chunk, .prefill_hook_ctx = &passes };
    defer fake.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(t.allocator, &cfg, fake.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(t.allocator, t.io, &core, .{ .lanes = 1 });
    host.cache = &store;
    var e: Exp = .{ .fake = &fake };
    host.exporter = e.exporter();
    var handoff: Reply = .{};
    var chat: Reply = .{};
    // both before the engine thread runs: the chat waits in the queue through the hand-off, protecting the remote copy
    try host.engine().submit(1, &.{ .prompt = &six, .max_tokens = 1, .handoff = fixture.id }, .{ .ctx = &handoff, .event = Reply.event });
    try host.engine().submit(2, &.{ .prompt = &six, .max_tokens = 1 }, .{ .ctx = &chat, .event = Reply.event });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    const until = std.Io.Clock.awake.now(t.io).toNanoseconds() + 5 * std.time.ns_per_s;
    while (!handoff.done.load(.acquire) or !chat.done.load(.acquire)) {
        if (std.Io.Clock.awake.now(t.io).toNanoseconds() > until) return error.RemoteFixtureTimeout;
        try std.Io.sleep(t.io, .fromMilliseconds(1), .awake);
    }
    host.lock();
    const admitted = host.admitted.items.len; // each job left the list before its reply finished
    host.unlock();
    try t.expectEqual(@as(usize, 0), admitted); // no job is left whose promise a later admission counts
    host.stop(); // the engine thread is joined: every job has ended
    stopped = true;
    try t.expectEqual(@as(usize, 2), passes.n);
    try t.expect(passes.at[0] == 3 and !passes.remote[0]); // the hand-off resumed below its cut on its own state
    try t.expectEqual(@as(u32, 1), passes.borrowed[0]); // borrowed from below() while its pass ran
    try t.expectEqual(@as(u64, 1), store.counts.failed); // the keep at the cut, refused under the guard
    try t.expectEqual(@as(u64, 0), store.counts.evicted); // refused before it evicted the state below the cut
    try t.expectEqual(@as(u32, 1), e.calls); // the export still went out, at the cut
    try t.expectEqual(@as(?usize, 5), e.tokens);
    try t.expect(e.why == null);
    try t.expectEqual(@as(usize, 1), handoff.tokens); // and the hand-off's reply went on
    try t.expect(passes.at[1] == 5 and passes.remote[1]); // the remote copy at the cut served the chat
    try t.expectEqual(@as(usize, 1), chat.tokens);
    try fixture.unborrowed(&store); // the hand-off's borrow below the cut and the chat's of the copy, both let go
    var held: u64 = 0;
    for (store.entries.items) |x| held += x.bytes;
    try t.expectEqual(held, store.held);
    const copy = store.find(&six, &.{}, &.{}) orelse return error.TestRemoteCopyGone;
    try t.expect(copy.remote and copy.at == 5);
}
