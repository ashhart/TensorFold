//! GLM expiry coupled to the actual cache; no GPU or checkpoint is needed for snapshot ownership and pool metadata.
const std = @import("std");
const pc = @import("../../core/prompt_cache.zig");
const slot_mod = @import("slots.zig");
const kept = @import("kept.zig");
const pool = @import("pool.zig");
const Snap = @import("snapshot.zig").Snap;
const Engine = @import("engine.zig").Engine;
const control = @import("ep_control.zig");
const Kind = @import("mirror.zig").Kind;
const fabric = @import("fabric");
const a = std.testing.allocator;

const Fixture = struct {
    engine: Engine = undefined,
    slots: [2]slot_mod.Slot = undefined,
    sl: slot_mod.Slots = undefined,
    store: pc.Store = undefined,
    next: ?*Snap = null,
    drops: usize = 0,
    critical: bool = false,
    early_drop: bool = false,
    ctl: ?*control.Control = null,
    peer_error: ?anyerror = null,

    fn init(f: *Fixture) void {
        f.engine.c = .{};
        f.engine.c.run = 0;
        f.engine.c.mtp = 0;
        for (&f.slots) |*s| s.* = .{ .s = undefined, .held = undefined, .last = undefined };
        f.sl = .{ .gpa = a, .e = &f.engine, .slots = &f.slots, .rows = undefined, .ids = undefined, .lasts = undefined, .picks = undefined, .ring = undefined, .pool = .{ .base = undefined, .blocks = 16, .kpool = 4 } };
        f.store = pc.Store.init(a, .{ .ptr = f, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_prompt = 0 }, 1024);
        f.sl.expire = .{ .ctx = &f.store, .f = expired };
    }

    fn deinit(f: *Fixture) void {
        f.store.deinit();
        f.sl.snaps.deinit(a);
    }

    fn bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }

    fn save(ctx: *anyopaque, _: ?*anyopaque, _: u32) !pc.Saved {
        const f: *Fixture = @ptrCast(@alignCast(ctx));
        const snap = f.next orelse return error.NoSnapshot;
        f.next = null;
        return snap;
    }

    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
        return error.BackendRestores;
    }

    fn drop(ctx: *anyopaque, saved: pc.Saved) void {
        const f: *Fixture = @ptrCast(@alignCast(ctx));
        const snap: *Snap = @ptrCast(@alignCast(saved));
        f.early_drop = f.early_drop or f.critical;
        if (f.ctl) |c| c.sendCommand(@backingInt(Kind.drop), &.{snap.id}) catch |err| {
            f.peer_error = err;
        };
        _ = f.sl.snaps.remove(snap.id);
        f.drops += 1;
        a.destroy(snap);
    }

    fn expired(ctx: *anyopaque, saved: pc.Saved) void {
        const s: *pc.Store = @ptrCast(@alignCast(ctx));
        s.expired(saved);
    }

    fn keep(f: *Fixture, prompt: []const u32, at: u32, home: ?u32, held: ?[2]u32) !*Snap {
        const snap = try a.create(Snap);
        errdefer a.destroy(snap);
        snap.* = .{ .id = @intCast(f.sl.snaps.count() + 1), .at = at, .buf = undefined, .bytes = 1, .home = home, .held = held };
        try f.sl.snaps.put(a, snap.id, snap);
        errdefer _ = f.sl.snaps.remove(snap.id);
        f.next = snap;
        try std.testing.expect(f.store.keep(prompt, at, null, &.{}, &.{}));
        return snap;
    }
};

test "GLM expiry: evicting the selected restore prefix falls back without a dangling snapshot" {
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const prompt: [1501]u32 = @splat(7);
    const snap = try f.keep(&prompt, 1500, null, .{ 2, 6 });
    const entry = (try f.store.lookup(arena.allocator(), &prompt, 1500, &.{}, &.{}, &.{})).entry.?;
    f.slots[1].region = .{ .start = 12, .blocks = 4 };
    f.critical = true;
    try pool.reserve(&f.sl, 0, 3000, snap);
    f.critical = false;
    try std.testing.expect(snap.stale and snap.held == null);
    try std.testing.expect(f.store.find(&prompt, &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(usize, 0), f.drops);
    try std.testing.expectEqual(@as(u32, 1500), entry.at);
    f.store.resumed(entry, &prompt, false);
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    try std.testing.expect(!f.early_drop);
    try std.testing.expectEqual(@as(u64, 0), f.store.held);
    try std.testing.expectEqual(@as(u64, 1), f.store.counts.evicted);
    try std.testing.expectEqual(@as(u64, 1), f.store.counts.failed);
    try std.testing.expect((try f.store.lookup(arena.allocator(), &prompt, 1500, &.{}, &.{}, &.{})).entry == null);
}

test "GLM expiry: multiple overwritten snapshots do not drop inside the backend operation" {
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    const prompt: [129]u32 = @splat(7);
    _ = try f.keep(&prompt, 64, 0, null);
    _ = try f.keep(&prompt, 128, 0, null);
    f.critical = true;
    kept.overwrite(&f.sl, 0, 32);
    kept.overwrite(&f.sl, 0, 16);
    try std.testing.expectEqual(@as(usize, 0), f.drops);
    try std.testing.expect(f.store.find(&prompt, &.{}, &.{}) == null);
    f.critical = false;
    f.store.collectExpired();
    try std.testing.expectEqual(@as(usize, 2), f.drops);
    try std.testing.expectEqual(@as(u64, 2), f.store.counts.evicted);
    try std.testing.expectEqual(@as(usize, 0), f.sl.snaps.count());
    try std.testing.expect(!f.early_drop);
}

test "GLM expiry: peer drops wait until a mirrored chunk can finish" {
    const cluster = try fabric.fake.Cluster.init(a, 2, control.BYTES, .immediate, null);
    defer cluster.deinit();
    var failed = std.atomic.Value(bool).init(false);
    var c0 = try control.Control.init(a, cluster.endpoint(0), 0, &failed);
    defer c0.deinit(a);
    var c1 = try control.Control.init(a, cluster.endpoint(1), 0, &failed);
    defer c1.deinit(a);
    c0.limit = 24_000_000;
    const Peer = struct {
        c: *control.Control,
        entered: std.atomic.Value(bool) = .init(false),
        complete: std.atomic.Value(bool) = .init(false),
        bad: std.atomic.Value(bool) = .init(false),
        drops: usize = 0,
        fn run(p: *@This()) void {
            const cmd = (p.c.waitCommand() catch {
                p.bad.store(true, .release);
                return;
            }) orelse return;
            if (cmd != .lanes or cmd.lanes.kind != @backingInt(Kind.chunk)) {
                p.bad.store(true, .release);
                return;
            }
            p.entered.store(true, .release);
            while (!p.complete.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            while (p.drops < 2) {
                const drop_cmd = (p.c.waitCommand() catch {
                    p.bad.store(true, .release);
                    return;
                }) orelse return;
                if (drop_cmd != .lanes or drop_cmd.lanes.kind != @backingInt(Kind.drop) or drop_cmd.lanes.words.len != 1 or drop_cmd.lanes.words[0] != p.drops + 1) {
                    p.bad.store(true, .release);
                    return;
                }
                p.drops += 1;
            }
        }
    };
    var peer: Peer = .{ .c = &c1 };
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    f.ctl = &c0;
    const prompt: [129]u32 = @splat(7);
    _ = try f.keep(&prompt, 64, 0, null);
    _ = try f.keep(&prompt, 128, 0, null);
    try c0.sendCommand(@backingInt(Kind.chunk), &.{ 0, 32, 16 });
    const thread = try std.Thread.spawn(.{}, Peer.run, .{&peer});
    var joined = false;
    defer {
        peer.complete.store(true, .release);
        c1.stop.store(true, .release);
        if (!joined) thread.join();
    }
    const began = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    while (!peer.entered.load(.acquire)) {
        if (peer.bad.load(.acquire) or std.Io.Clock.awake.now(std.testing.io).toNanoseconds() - began > std.time.ns_per_s) return error.PeerDidNotEnter;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    f.critical = true;
    kept.overwrite(&f.sl, 0, 32);
    try std.testing.expectEqual(@as(usize, 0), f.drops);
    f.critical = false;
    peer.complete.store(true, .release);
    f.store.collectExpired();
    try std.testing.expect(f.peer_error == null and !f.early_drop);
    thread.join();
    joined = true;
    try std.testing.expectEqual(@as(usize, 2), peer.drops);
    try std.testing.expect(!peer.bad.load(.acquire));
    f.ctl = null;
}
