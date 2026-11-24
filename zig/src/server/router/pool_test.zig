//! pool.zig with /bin/sh children: least recently used stops, busy waits and their timeout, crashes, failures, stops.
const std = @import("std");
const tc = @import("test_child.zig");
const catalog = @import("catalog.zig");
const pool_mod = @import("pool.zig");
const t = std.testing;

const Fixture = struct {
    arena: std.heap.ArenaAllocator = .init(t.allocator),
    tmp: std.testing.TmpDir,
    children: std.ArrayList(tc.Child) = .empty,
    pool: ?*pool_mod.Pool = null,

    fn init() Fixture {
        return .{ .tmp = std.testing.tmpDir(.{}) };
    }

    fn deinit(f: *Fixture) void {
        if (f.pool) |p| {
            p.stopAll();
            p.deinit();
        }
        for (f.children.items) |c| c.server.stop();
        f.tmp.cleanup();
        f.arena.deinit();
    }

    /// Fake children named ``ids`` and a pool over them, with ``extra`` entries after them.
    fn start(f: *Fixture, ids: []const []const u8, extra: []const catalog.Entry, options: pool_mod.Options) !*pool_mod.Pool {
        const a = f.arena.allocator();
        var entries: std.ArrayList(catalog.Entry) = .empty;
        for (ids) |id| {
            const c = try tc.start(a, id, &.{}, &.{});
            try f.children.append(a, c);
            try entries.append(a, c.entry);
        }
        try entries.appendSlice(a, extra);
        f.pool = try tc.pool(a, &f.tmp, entries.items, options);
        return f.pool.?;
    }
};

fn lease(p: *pool_mod.Pool, a: std.mem.Allocator, i: usize) !pool_mod.Lease {
    return switch (p.acquire(a, i)) {
        .lease => |l| l,
        .refused => |r| {
            std.debug.print("refused: {any}\n", .{r});
            return error.TestUnexpectedResult;
        },
    };
}

test "serve --router: past --max-loaded the least recently used idle child stops, and a busy one is never stopped" {
    var f: Fixture = .init();
    defer f.deinit();
    const a = f.arena.allocator();
    const p = try f.start(&.{ "alpha", "beta", "gamma" }, &.{}, .{ .max_loaded = 2, .log_dir = "" });
    p.release(try lease(p, a, 0));
    p.release(try lease(p, a, 1));
    p.release(try lease(p, a, 0)); // alpha is now the more recent
    const alpha = tc.pid(p, 0).?;
    const beta = tc.pid(p, 1).?;
    const held = try lease(p, a, 2);
    try t.expect(tc.gone(beta) and !tc.gone(alpha));
    try t.expect(tc.pid(p, 1) == null and tc.pid(p, 2) != null);
    // alpha is idle and gamma is held: beta's start stops alpha, not the busy gamma
    p.release(try lease(p, a, 1));
    try t.expect(tc.gone(alpha) and tc.pid(p, 2) != null);
    p.release(held);
    try t.expectEqual(@as(usize, 2), (try p.loaded(a)).len);
}

const Waiter = struct {
    p: *pool_mod.Pool,
    a: std.mem.Allocator,
    got: ?pool_mod.Acquired = null,

    fn run(w: *Waiter) void {
        w.got = w.p.acquire(w.a, 1);
    }
};

test "serve --router: with every loaded model busy a request waits for one to finish, and past its bound gets busy" {
    var f: Fixture = .init();
    defer f.deinit();
    const a = f.arena.allocator();
    const p = try f.start(&.{ "alpha", "beta" }, &.{}, .{ .max_loaded = 1, .busy_wait_s = 0.3, .log_dir = "" });
    const held = try lease(p, a, 0);
    const t0 = tc.now();
    const refused = p.acquire(a, 1);
    try t.expect(refused == .refused and refused.refused == .busy);
    try t.expect(tc.since(t0) >= 0.3 and tc.since(t0) < 5);
    try t.expect(tc.pid(p, 0) != null and tc.pid(p, 1) == null);
    p.options.busy_wait_s = 10;
    var w: Waiter = .{ .p = p, .a = a };
    const thread = try std.Thread.spawn(.{}, Waiter.run, .{&w});
    @import("../benchmark_fake.zig").nap(300);
    try t.expect(w.got == null);
    const alpha = tc.pid(p, 0).?;
    p.release(held);
    thread.join();
    try t.expect(w.got.? == .lease);
    try t.expect(tc.gone(alpha) and tc.pid(p, 1) != null);
    p.release(w.got.?.lease);
}

test "serve --router: a child that dies is unloaded, a failed exchange learns it, and the next request starts it again" {
    var f: Fixture = .init();
    defer f.deinit();
    const a = f.arena.allocator();
    const p = try f.start(&.{"alpha"}, &.{}, .{ .log_dir = "" });
    const first = try lease(p, a, 0);
    const before = tc.pid(p, 0).?;
    try std.posix.kill(before, .KILL);
    try t.expect(p.lost(first));
    p.release(first);
    try t.expect(tc.pid(p, 0) == null);
    const again = try lease(p, a, 0);
    const after = tc.pid(p, 0).?;
    try t.expect(after != before and !std.mem.eql(u8, &first.key, &again.key));
    try t.expect(!p.lost(again)); // alive: a failed exchange alone does not unload it
    p.release(again);
}

test "serve --router: a load that exits names the child's last line, and one that never serves times out" {
    var f: Fixture = .init();
    defer f.deinit();
    const a = f.arena.allocator();
    const broken = try tc.failing(a, "broken", "tensorfold: example checkpoint is incomplete");
    const mute = try tc.silent(a, "mute", 0.5);
    const p = try f.start(&.{}, &.{ broken, mute }, .{ .log_dir = "" });
    const failed = p.acquire(a, 0);
    try t.expect(failed == .refused and failed.refused == .failed);
    try t.expect(std.mem.indexOf(u8, failed.refused.failed, "example checkpoint is incomplete") != null);
    const t0 = tc.now();
    const late = p.acquire(a, 1);
    try t.expect(late == .refused and late.refused == .timeout);
    try t.expect(tc.since(t0) >= 0.5 and tc.since(t0) < 8);
    try t.expectEqual(@as(usize, 0), (try p.loaded(a)).len);
}

const Loader = struct {
    p: *pool_mod.Pool,
    a: std.mem.Allocator,
    got: ?pool_mod.Acquired = null,

    fn run(l: *Loader) void {
        l.got = l.p.acquire(l.a, 1);
    }
};

test "serve --router: stopping ends a load in progress, stops every child and refuses new leases" {
    var f: Fixture = .init();
    defer f.deinit();
    const a = f.arena.allocator();
    const mute = try tc.silent(a, "mute", 30);
    const p = try f.start(&.{"alpha"}, &.{mute}, .{ .max_loaded = 2, .log_dir = "" });
    p.release(try lease(p, a, 0));
    const alpha = tc.pid(p, 0).?;
    var l: Loader = .{ .p = p, .a = a };
    const thread = try std.Thread.spawn(.{}, Loader.run, .{&l});
    const Loading = struct {
        fn check(q: *pool_mod.Pool) bool {
            return tc.pid(q, 1) == null and tc.state(q, 1) == .loading;
        }
    };
    try t.expect(tc.within(5_000, p, Loading.check));
    const t0 = tc.now();
    p.stopAll();
    thread.join();
    try t.expect(tc.since(t0) < 8);
    try t.expect(l.got.? == .refused and l.got.?.refused == .stopping);
    try t.expect(tc.gone(alpha));
    const after = p.acquire(a, 0);
    try t.expect(after == .refused and after.refused == .stopping);
}

test "serve --router: a child idle past --idle-unload is stopped, and a leased one is kept" {
    var f: Fixture = .init();
    defer f.deinit();
    const a = f.arena.allocator();
    const p = try f.start(&.{ "alpha", "beta" }, &.{}, .{ .max_loaded = 2, .idle_s = 1, .log_dir = "" });
    try p.startReaper();
    p.release(try lease(p, a, 0));
    const held = try lease(p, a, 1);
    const alpha = tc.pid(p, 0).?;
    const Unloaded = struct {
        fn check(q: *pool_mod.Pool) bool {
            return tc.state(q, 0) == .unloaded;
        }
    };
    try t.expect(tc.within(5_000, p, Unloaded.check));
    try t.expect(tc.gone(alpha) and tc.pid(p, 1) != null);
    p.release(held);
}
