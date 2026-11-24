//! TF_KV_SHARE's store on a lane host (no host cache): moved weights and a flush drop its states, asks read it.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const radix = @import("prompt_radix.zig");
const learn = @import("lane_host_learn.zig");
const rows = @import("lane_host_rows.zig");
const LaneHost = api.LaneHost;
const t = std.testing;
const gpa = t.allocator;

/// A state is a byte; `live` counts the saved ones not yet dropped.
const Family = struct {
    live: usize = 0,
    byte: u8 = 0,

    fn of(p: *anyopaque) *Family {
        return @ptrCast(@alignCast(p));
    }
    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    fn save(p: *anyopaque, _: ?*anyopaque, _: u32) anyerror!radix.Saved {
        of(p).live += 1;
        return @ptrCast(&of(p).byte);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: radix.Saved) anyerror!void {}
    fn drop(p: *anyopaque, _: radix.Saved) void {
        of(p).live -= 1;
    }
    fn snapshots(f: *Family) radix.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } };
    }
};

/// Eight one-byte pages, each with its holders (a stand-in for the core pool: engine_api cannot import it).
const Pool = struct {
    refs: [8]u32 = @splat(0),

    fn of(p: *anyopaque) *Pool {
        return @ptrCast(@alignCast(p));
    }
    fn retain(p: *anyopaque, id: u32) void {
        of(p).refs[id] += 1;
    }
    fn release(p: *anyopaque, id: u32) void {
        of(p).refs[id] -= 1;
    }
    fn holders(p: *anyopaque, id: u32) u32 {
        return of(p).refs[id];
    }
    fn available(p: *anyopaque) usize {
        var n: usize = 0;
        for (of(p).refs) |r| n += @intFromBool(r == 0);
        return n;
    }
    fn pages(p: *Pool) radix.Pages {
        return .{ .ptr = p, .bytes = 1, .vtable = &.{ .retain = retain, .release = release, .holders = holders, .available = available } };
    }
};

/// A learner whose one step ends its lesson, reporting `changed` weights or not.
const Learner = struct {
    changed: bool,
    sink: ?api.LearnSink = null,

    fn begin(ctx: *anyopaque, _: *const api.LearnRequest, sink: api.LearnSink) anyerror!void {
        const l: *Learner = @ptrCast(@alignCast(ctx));
        l.sink = sink;
    }
    fn step(ctx: *anyopaque) api.Learner.Step {
        const l: *Learner = @ptrCast(@alignCast(ctx));
        l.sink.?.event(l.sink.?.ctx, &.{ .done = .{} });
        return .{ .done = true, .changed = l.changed };
    }
    fn abort(_: *anyopaque) void {}
    fn ignore(_: *anyopaque, _: *const api.LearnEvent) void {}
};

/// A host that never starts (its engine-thread calls made here), over a pool of 8 four-token pages.
const Fixture = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake = .{ .gpa = gpa },
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: LaneHost,
    pool: Pool = .{},
    family: Family = .{},
    store: radix.Store,

    fn init(f: *Fixture) !void {
        f.* = .{ .cfg = undefined, .core = undefined, .host = undefined, .store = undefined };
        f.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        f.core = lanes.Engine.init(gpa, &f.cfg, f.target.backend(), f.clock.clock());
        f.host = LaneHost.init(gpa, t.io, &f.core, .{ .lanes = 2 });
        f.store = try radix.Store.init(gpa, f.family.snapshots(), f.pool.pages(), .{ .page = 4 });
        f.store.limit(.{ .pages = 8, .snaps = 4 });
        f.host.disk.shared = &f.store;
    }

    fn deinit(f: *Fixture) void {
        f.host.stop(); // never started: its lists freed, its asks closed
        f.store.deinit();
        f.core.deinit();
        f.target.deinit();
        f.cfg.deinit(gpa);
    }

    /// The store keeps `words`' first 8 tokens: two pages, and a state.
    fn keep(f: *Fixture, words: []const u32) !void {
        const ids = [2]u32{ 0, 1 }; // the stream's pages
        for (ids) |id| Pool.retain(&f.pool, id);
        var path: [2]u32 = undefined;
        var owner: u8 = 0;
        try t.expect(f.store.keep(.of(words), 8, &owner, &ids, &path).held);
        for (ids) |id| Pool.release(&f.pool, id); // the stream ends: the store's holds remain
        try t.expectEqual(@as(usize, 6), Pool.available(&f.pool));
        try t.expectEqual(@as(usize, 1), f.family.live);
    }

    /// One lesson stepped to its end by a learner reporting `changed`.
    fn lesson(f: *Fixture, changed: bool) !void {
        var l: Learner = .{ .changed = changed };
        f.host.learner = .{ .ctx = &l, .begin = Learner.begin, .step = Learner.step, .abort = Learner.abort };
        defer f.host.learner = null;
        const request: api.LearnRequest = .{};
        try f.host.lessons.append(gpa, .{ .request = &request, .sink = .{ .ctx = &l, .event = Learner.ignore } });
        learn.step(&f.host);
        try t.expect(!f.host.lesson_open);
    }

    /// Where `words` would resume, as GET /v1/imprint asks it.
    fn where(f: *Fixture, words: []const u32) !u32 {
        var a: api.ImprintAsk = .{ .op = .where, .prompt = words };
        try t.expect(rows.ask(&f.host, &a));
        rows.answer(&f.host);
        try t.expect(a.done.load(.acquire));
        try t.expectEqual(@as(u32, @intCast(words.len)), a.out.tokens);
        return a.out.ram;
    }
};

const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

test "TF_KV_SHARE lane host: a learn step that moves weights drops the store's kept pages and states" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    try f.keep(&prompt);
    try t.expectEqual(@as(u32, 8), try f.where(&prompt));
    try f.lesson(false); // control: weights that did not move keep the state
    try t.expect(f.store.find(.of(&prompt)) != null);
    try t.expectEqual(@as(usize, 1), f.family.live);
    try f.lesson(true);
    try t.expect(f.store.find(.of(&prompt)) == null); // a later taught request prefills cold, on the new weights
    try t.expectEqual(@as(usize, 0), f.family.live);
    try t.expectEqual(@as(usize, 0), f.store.tree.heldPages());
    try t.expectEqual(@as(usize, 8), Pool.available(&f.pool));
    try t.expectEqual(@as(u32, 0), try f.where(&prompt));
    try f.keep(&prompt); // the cleared store keeps again
    try t.expectEqual(@as(u32, 8), try f.where(&prompt));
}

test "TF_KV_SHARE lane host: a flush is taken without a host cache and drops the store's kept pages and states" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    try f.keep(&prompt);
    var flush: api.ImprintAsk = .{ .op = .flush };
    try t.expect(rows.ask(&f.host, &flush)); // not 501: the store keeps states
    rows.answer(&f.host);
    try t.expect(flush.done.load(.acquire));
    try t.expectEqual(@as(u32, 1), flush.dropped);
    try t.expectEqual(@as(u64, 2 + 1), flush.ram_freed); // two one-byte pages and the state's byte
    try t.expect(!flush.disk);
    try t.expectEqual(@as(usize, 0), f.family.live);
    try t.expectEqual(@as(usize, 8), Pool.available(&f.pool));
    try t.expectEqual(@as(u32, 0), try f.where(&prompt));
    // control: with neither store the flush is refused (501), as before
    f.host.disk.shared = null;
    var none: api.ImprintAsk = .{ .op = .flush };
    try t.expect(!rows.ask(&f.host, &none));
}
