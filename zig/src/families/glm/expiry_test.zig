//! GLM expiry coupled to the actual cache; no GPU or checkpoint is needed for snapshot ownership and pool metadata.
const std = @import("std");
const pc = @import("../../core/prompt_cache.zig");
const platform = @import("sys");
const slot_mod = @import("slots.zig");
const kept = @import("kept.zig");
const pool = @import("pool.zig");
const Snap = @import("snapshot.zig").Snap;
const Engine = @import("engine.zig").Engine;
const Ep = std.meta.Child(std.meta.Child(blk: {
    const info = @typeInfo(Engine).@"struct";
    for (info.field_names, info.field_types) |name, field_type| if (std.mem.eql(u8, name, "ep")) break :blk field_type;
    @compileError("Engine.ep type is unavailable");
}));
const control = @import("ep_control.zig");
const mirror = @import("mirror.zig");
const Kind = mirror.Kind;
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

const PeerDisk = struct {
    const Space = struct {
        fn free(_: [:0]const u8) ?u64 {
            return 1 << 30;
        }
        fn clock() ?u64 {
            return 100;
        }
    };

    temp: std.testing.TmpDir = undefined,
    root: [:0]const u8 = &.{},
    dir: [:0]const u8 = &.{},
    cluster: *fabric.fake.Cluster = undefined,
    failed: std.atomic.Value(bool) = .init(false),
    c: [2]control.Control = undefined,
    ep: Ep = undefined,
    f: Fixture = .{},
    snap: Snap = undefined,
    wins: [2]slot_mod.Win = undefined,
    drafts: [2]slot_mod.Draft = undefined,

    fn init(p: *PeerDisk) !void {
        p.temp = std.testing.tmpDir(.{});
        p.root = try p.temp.dir.realPathFileAlloc(std.testing.io, ".", a);
        p.dir = try std.fmt.allocPrintSentinel(a, "{s}/0000000000000001", .{p.root}, 0);
        try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(p.dir, 0o700));
        p.cluster = try fabric.fake.Cluster.init(a, 2, control.BYTES, .immediate, null);
        p.c[0] = try control.Control.init(a, p.cluster.endpoint(0), 0, &p.failed);
        p.c[1] = try control.Control.init(a, p.cluster.endpoint(1), 0, &p.failed);
        for (&p.c) |*c| c.limit = 24_000_000;
        p.ep = undefined;
        p.ep.rank = 1;
        p.ep.ctl = p.c[1];
        p.f.init();
        p.f.engine.ep = &p.ep;
        p.f.sl.learned = p.dir;
        p.f.sl.disk.admission = .{ .floor = 0, .free = Space.free, .clock = Space.clock };
        p.snap = .{ .id = 9, .at = 0, .buf = undefined, .bytes = 1 };
        try p.f.sl.snaps.put(a, p.snap.id, &p.snap);
    }

    fn deinit(p: *PeerDisk) void {
        _ = p.f.sl.snaps.remove(p.snap.id);
        p.f.deinit();
        for (&p.c) |*c| c.deinit(a);
        p.cluster.deinit();
        a.free(p.dir);
        a.free(p.root);
        p.temp.cleanup();
    }

    fn apply(p: *PeerDisk, kind: Kind, words: []const u32) !void {
        return p.applyFrom(kind, words, "/tf-kf81-distinct-leader");
    }

    fn applyLegacy(p: *PeerDisk, kind: Kind, words: []const u32) !void {
        return p.applyFrom(kind, words, null);
    }

    fn applyFrom(p: *PeerDisk, kind: Kind, words: []const u32, root: ?[]const u8) !void {
        var out: [5 + 1 + (1100 + 3) / 4]u32 = undefined;
        var sent: []const u32 = words;
        if (root) |leader_root| {
            @memcpy(out[0..words.len], words);
            out[words.len] = @intCast(leader_root.len);
            const packed_len = (leader_root.len + 3) / 4;
            out[words.len + packed_len] = 0;
            const packed_bytes = std.mem.sliceAsBytes(out[words.len + 1 ..][0..packed_len]);
            @memcpy(packed_bytes[0..leader_root.len], leader_root);
            sent = out[0 .. words.len + 1 + packed_len];
        }
        try p.c[0].sendCommand(@backingInt(kind), sent);
        const cmd = (try p.f.sl.e.ep.?.ctl.waitCommand()).?;
        if (cmd != .lanes) return error.PeerOutOfStep;
        if (@hasDecl(mirror, "applyDisk")) return mirror.applyDisk(&p.f.sl, cmd.lanes.kind, cmd.lanes.words);
        return p.applyExisting(kind, cmd.lanes.words);
    }

    fn applyExisting(p: *PeerDisk, kind: Kind, w: []const u32) !void {
        const need: usize = switch (kind) {
            .disk_need => 3,
            .disk_reserve => 5,
            .disk_finish, .disk_other_next => 3,
            .disk_other_remove, .forget => 2,
            else => return error.CommandOutOfStep,
        };
        if (w.len < need) return error.CommandOutOfStep;
        const ctl = &p.f.sl.e.ep.?.ctl;
        const dirs = @import("lanes").learned_dirs;
        const dir = p.f.sl.learned orelse return error.NotLearning;
        switch (kind) {
            .disk_need => {
                const snap = p.f.sl.snaps.get(w[0]) orelse return ctl.replyValue(std.math.maxInt(u64));
                const extra = @as(u64, w[1]) | @as(u64, w[2]) << 32;
                try ctl.replyValue(p.f.sl.disk.need(dir, snap.bytes, extra));
            },
            .disk_reserve => {
                const snap = p.f.sl.snaps.get(w[0]) orelse return ctl.reply(false);
                const key = @as(u64, w[1]) | @as(u64, w[2]) << 32;
                const extra = @as(u64, w[3]) | @as(u64, w[4]) << 32;
                p.f.sl.disk.reserve(dir, key, snap.bytes, extra) catch return ctl.reply(false);
                try ctl.reply(true);
            },
            .disk_finish => {
                const key = @as(u64, w[0]) | @as(u64, w[1]) << 32;
                p.f.sl.disk.finish(dir, key, w[2] != 0) catch return ctl.reply(false);
                try ctl.reply(true);
            },
            .disk_other_remove => {
                const id = @as(u64, w[0]) | @as(u64, w[1]) << 32;
                dirs.removeOther(dirs.rootOf(dir), dir, id) catch return ctl.reply(false);
                try ctl.reply(true);
            },
            .disk_other_next => {
                const id = @as(u64, w[0]) | @as(u64, w[1]) << 32;
                const found = dirs.next(dirs.rootOf(dir), dir, if (w[2] != 0) id else null) catch return ctl.replyValue(std.math.maxInt(u64));
                if (found) |value| try ctl.replyValue(value) else try ctl.reply(false);
            },
            .forget => {
                var buf: [1100]u8 = undefined;
                const file = try @import("snapshot.zig").path(&buf, dir, @as(u64, w[0]) | @as(u64, w[1]) << 32, 1);
                dirs.unlink(file) catch return ctl.reply(false);
                try ctl.reply(true);
            },
            else => unreachable,
        }
    }

    fn followerExit(p: *PeerDisk) void {
        if (@hasDecl(mirror, "releaseDisk")) mirror.releaseDisk(&p.f.sl);
    }

    fn locked(p: *PeerDisk) !void {
        const lease = platform.folder_lock.hold(a, p.root, true) catch |err| {
            try std.testing.expectEqual(error.Busy, err);
            return;
        };
        lease.release();
        return error.FollowerRootNotLeased;
    }

    fn released(p: *PeerDisk) !void {
        const lease = try platform.folder_lock.hold(a, p.root, true);
        lease.release();
    }

    fn makeIdentity(p: *PeerDisk, id: u64) !void {
        var path: [1100]u8 = undefined;
        const dir = try std.fmt.bufPrintSentinel(&path, "{s}/{x:0>16}", .{ p.root, id }, 0);
        try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(dir, 0o700));
    }

    fn expectIdentityGone(p: *PeerDisk, id: u64) !void {
        var path: [1100]u8 = undefined;
        const dir = try std.fmt.bufPrintSentinel(&path, "{s}/{x:0>16}", .{ p.root, id }, 0);
        const fd = platform.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, @as(platform.c.mode_t, 0));
        if (fd >= 0) {
            _ = platform.c.close(fd);
            return error.LearnedIdentityRemains;
        }
        try std.testing.expectEqual(.NOENT, platform.c.errno(fd));
    }
};

test "GLM expiry: a distinct follower root stays leased from peer planning through disk finish" {
    var leader_temp = std.testing.tmpDir(.{});
    defer leader_temp.cleanup();
    const leader_root = try leader_temp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(leader_root);
    const leader = try platform.folder_lock.hold(a, leader_root, true);
    defer leader.release();
    var p: PeerDisk = .{};
    try p.init();
    defer p.deinit();
    try p.makeIdentity(2);
    try p.applyFrom(.disk_need, &.{ 9, 0, 0 }, leader_root);
    try std.testing.expectEqual(@as(u64, 0), try p.c[0].waitReplyValue());
    try p.locked();
    try p.apply(.disk_reserve, &.{ 9, 77, 0, 0, 0 });
    try std.testing.expect(try p.c[0].waitReply());
    try p.locked();
    try p.apply(.disk_other_remove, &.{ 2, 0 });
    try std.testing.expect(try p.c[0].waitReply());
    try p.expectIdentityGone(2);
    try p.locked();
    // The same transaction root reuses its active owner instead of recursively acquiring a second lease.
    try p.apply(.forget, &.{ 66, 0 });
    try std.testing.expect(try p.c[0].waitReply());
    try p.locked();
    try p.apply(.disk_finish, &.{ 77, 0, 1 });
    try std.testing.expect(try p.c[0].waitReply());
    try p.released();
    {
        const contender = try platform.folder_lock.hold(a, p.root, true);
        defer contender.release();
        try p.applyFrom(.disk_need, &.{ 9, 0, 0 }, leader_root);
        try std.testing.expectEqual(std.math.maxInt(u64), try p.c[0].waitReplyValue());
    }
    try p.released();
}

test "GLM expiry: an identical leader and follower root reuses the active transaction lease" {
    var p: PeerDisk = .{};
    try p.init();
    defer p.deinit();
    const leader = try platform.folder_lock.hold(a, p.root, true);
    defer leader.release();
    try p.applyFrom(.disk_need, &.{ 9, 0, 0 }, p.root);
    try std.testing.expectEqual(@as(u64, 0), try p.c[0].waitReplyValue());
    try p.apply(.disk_reserve, &.{ 9, 77, 0, 0, 0 });
    try std.testing.expect(try p.c[0].waitReply());
    try p.apply(.disk_finish, &.{ 77, 0, 1 });
    try std.testing.expect(try p.c[0].waitReply());
    try p.applyLegacy(.disk_need, &.{ 9, 0, 0 }); // a legacy leader has no root proof: preserve its old shared-root behavior
    try std.testing.expectEqual(@as(u64, 0), try p.c[0].waitReplyValue());
    try p.applyLegacy(.disk_finish, &.{ 77, 0, 0 });
    try std.testing.expect(try p.c[0].waitReply());
}

test "GLM expiry: a peer command error releases the follower root lease" {
    var p: PeerDisk = .{};
    try p.init();
    defer p.deinit();
    try p.apply(.disk_need, &.{ 9, 0, 0 });
    try std.testing.expectEqual(@as(u64, 0), try p.c[0].waitReplyValue());
    try p.locked();
    try std.testing.expectError(error.CommandOutOfStep, p.applyLegacy(.disk_other_next, &.{}));
    try p.released();
    try p.apply(.disk_finish, &.{ 77, 0, 1 });
    try std.testing.expect(!try p.c[0].waitReply());
}

test "GLM expiry: failed rollback deletion retains the lease through disk finish" {
    var p: PeerDisk = .{};
    try p.init();
    defer p.deinit();
    try p.apply(.disk_need, &.{ 9, 0, 0 });
    try std.testing.expectEqual(@as(u64, 0), try p.c[0].waitReplyValue());
    try p.apply(.disk_reserve, &.{ 9, 77, 0, 0, 0 });
    try std.testing.expect(try p.c[0].waitReply());
    var path: [1100]u8 = undefined;
    const file = try @import("snapshot.zig").path(&path, p.dir, 66, 1);
    try std.testing.expectEqual(@as(c_int, 0), platform.c.mkdir(file, 0o700));
    defer _ = platform.c.rmdir(file);
    try p.apply(.forget, &.{ 66, 0 });
    try std.testing.expect(!try p.c[0].waitReply());
    try p.locked();
    try p.apply(.disk_finish, &.{ 77, 0, 0 });
    try std.testing.expect(try p.c[0].waitReply());
    try p.released();
    try std.testing.expect(p.f.sl.disk.pending == null);
}

test "GLM expiry: follower exit releases an active peer disk root lease" {
    var p: PeerDisk = .{};
    try p.init();
    defer p.deinit();
    try p.apply(.disk_need, &.{ 9, 0, 0 });
    try std.testing.expectEqual(@as(u64, 0), try p.c[0].waitReplyValue());
    try p.locked();
    p.followerExit();
    try p.released();
}

test "GLM expiry: a failed relocation keeps its selected restore borrow alive until reported" {
    const log_level = std.testing.log_level;
    std.testing.log_level = .err;
    defer std.testing.log_level = log_level;
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
    const Failed = struct {
        pub fn move(_: @This(), _: *slot_mod.Slots, _: *const Snap, _: u32) !void {
            return error.PoolMoveNotDrained;
        }
    };
    try std.testing.expectError(error.PoolMoveNotDrained, pool.reserveWith(&f.sl, 0, 3000, snap, Failed{}));
    f.critical = false;
    try std.testing.expect(!snap.stale and snap.held != null and f.slots[0].region == null);
    try std.testing.expect(f.store.find(&prompt, &.{}, &.{}) != null);
    try std.testing.expectEqual(@as(usize, 0), f.drops);
    try std.testing.expectEqual(@as(u32, 1500), entry.at);
    f.store.resumed(entry, &prompt, false);
    try std.testing.expectEqual(@as(usize, 1), f.drops);
    try std.testing.expect(!f.early_drop);
    try std.testing.expectEqual(@as(u64, 0), f.store.held);
    try std.testing.expectEqual(@as(u64, 0), f.store.counts.evicted);
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

test "GLM expiry: a long continuing turn relocates its own prefix instead of evicting it" {
    for ([_]struct { at: u32, old: ?u32, len: u32, previous: u32, start: u32 }{
        .{ .at = 99180, .old = null, .len = 101398, .previous = 99409, .start = 440 },
        .{ .at = 107824, .old = 107418, .len = 113214, .previous = 109844, .start = 400 },
        .{ .at = 187000, .old = null, .len = 188000, .previous = 187001, .start = 100 },
    }) |case| {
        var f: Fixture = .{};
        f.init();
        defer f.deinit();
        f.sl.pool.?.blocks = 1024;
        f.engine.c.kv_lora = 3;
        f.engine.c.i_dim = 2;
        try std.testing.expect(case.start + (case.previous + 49152 + 17 + 4 + 255) / 256 <= 1024);
        try std.testing.expect(case.start + (case.len + 49152 + 17 + 4 + 255) / 256 > 1024);
        const prompt = try a.alloc(u32, case.len);
        defer a.free(prompt);
        for (prompt, 0..) |*t, i| t.* = @intCast(i % 317);
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const blocks = (case.at + 4 + 255) / 256;
        if (case.old) |old| _ = try f.keep(prompt, old, null, .{ case.start, blocks });
        const own = try f.keep(prompt, case.at, null, .{ case.start, blocks });
        const entry = (try f.store.lookup(arena.allocator(), prompt, case.len - 19, &.{}, &.{}, &.{})).entry.?;
        const Ref = struct {
            data: []u8,
            pub fn addr(r: @This()) [*]u8 {
                return r.data.ptr;
            }
        };
        const Part = struct { keys: Ref, ik: Ref, ig: Ref, pool: Ref };
        const Mover = struct {
            parts: []Part,
            pub fn move(m: @This(), sl: *slot_mod.Slots, snap: *const Snap, start: u32) !void {
                @import("pool_move.zig").prefix(&sl.e.c, m.parts, snap.held.?[0], start, snap.at);
            }
        };
        var parts: [1]Part = undefined;
        var buffers: [4][]u8 = undefined;
        for (&buffers, 0..) |*bytes, k| bytes.* = try a.alloc(u8, 262144 * @as(usize, if (k == 0) 6 else 4));
        defer for (buffers) |bytes| a.free(bytes);
        parts[0] = .{ .keys = .{ .data = buffers[0] }, .ik = .{ .data = buffers[1] }, .ig = .{ .data = buffers[2] }, .pool = .{ .data = buffers[3] } };
        for (buffers, 0..) |bytes, k| {
            const stride: usize = if (k == 0) 6 else 4;
            @memset(bytes, 0xab);
            const rows = own.at / @as(u32, if (k == 3) 4 else 1) + @intFromBool(k == 3);
            const start: usize = @as(usize, case.start) * 256 / @as(usize, if (k == 3) 4 else 1) * stride;
            for (bytes[start..][0 .. rows * stride], 0..) |*b, j| b.* = @truncate((j * 17) ^ (j >> 8) ^ (j >> 16) ^ k);
        }
        try pool.reserveWith(&f.sl, 0, case.len + 49152 + 17, own, Mover{ .parts = &parts });
        try std.testing.expect(kept.usable(&f.sl, own) and own.held != null);
        try std.testing.expectEqual(@as(u32, 0), f.slots[0].region.?.start);
        try std.testing.expectEqual(@as(u32, 0), own.held.?[0]);
        for (buffers, 0..) |bytes, k| {
            const stride: usize = if (k == 0) 6 else 4;
            const rows = prompt.len / @as(usize, if (k == 3) 4 else 1) + @intFromBool(k == 3);
            const restored = own.at / @as(u32, if (k == 3) 4 else 1) + @intFromBool(k == 3);
            for (bytes[0 .. restored * stride], 0..) |b, j| try std.testing.expectEqual(@as(u8, @truncate((j * 17) ^ (j >> 8) ^ (j >> 16) ^ k)), b);
            for (bytes[restored * stride .. rows * stride], restored * stride..) |*b, j| b.* = @truncate((j * 17) ^ (j >> 8) ^ (j >> 16) ^ k);
            for (bytes[0 .. rows * stride], 0..) |b, j| try std.testing.expectEqual(@as(u8, @truncate((j * 17) ^ (j >> 8) ^ (j >> 16) ^ k)), b);
        }
        f.store.resumed(entry, prompt, true);
        try std.testing.expectEqual(@as(u64, 1), f.store.counts.hits);
        try std.testing.expectEqual(@as(u64, 0), f.store.counts.failed);
    }
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
