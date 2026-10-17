//! A served model's buffers in one residency set, kept wired by the server's idle keepalive, inside its load limit.
const std = @import("std");
const mtl = @import("metal");
const cache_fit = @import("cache_fit.zig");

/// macOS unwires GPU memory after ~1-2 s idle and a request re-wires it all: keepalive ticks use the set.
pub const Resident = struct {
    held: Held(mtl.ResidencySet, mtl.Buffer),
    work: mtl.Queue,
    scratch: mtl.Buffer, // inside the set: the keepalive's tick copies 16 bytes within it

    /// One set of `buffers` that `work` holds, requested resident at once; null (logged for `name`) past the 70% limit.
    pub fn hold(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, work: mtl.Queue, buffers: []const mtl.Buffer, name: []const u8) !?Resident {
        return holdUnder(gpa, io, device, work, buffers, name, null);
    }

    /// As hold, under a family's validated load limit when supplied, keeping the margin inside it.
    pub fn holdUnder(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, work: mtl.Queue, buffers: []const mtl.Buffer, name: []const u8, limit: ?u64) !?Resident {
        const set = try device.residencySet(buffers.len + 1);
        errdefer set.deinit();
        const scratch = try device.buffer(64, mtl.ResourceOptions.shared);
        errdefer scratch.deinit();
        var held = try Held(mtl.ResidencySet, mtl.Buffer).initUnder(gpa, io, set, buffers, scratch, name, cache_fit.ram(), limit) orelse {
            scratch.deinit();
            set.deinit();
            return null;
        };
        errdefer held.deinit();
        set.requestResidency();
        work.addResidencySet(set);
        if (limit) |n| {
            std.log.info("{s}: {d:.1} GiB kept wired between requests, inside the {d:.1} GiB load limit", .{ name, cache_fit.gibs(held.bytes), cache_fit.gibs(@min(n, held.ram.?)) });
        } else std.log.info("{s}: {d:.1} GiB kept wired between requests, inside 70% of {d:.0} GiB of RAM", .{ name, cache_fit.gibs(held.bytes), cache_fit.gibs(held.ram.?) });
        return .{ .held = held, .work = work, .scratch = scratch };
    }

    /// Buffers made after the set join it at the next flush, if they fit beside what it holds; else they stay out.
    pub fn add(r: *Resident, buffers: []const mtl.Buffer) void {
        _ = r.held.join(buffers);
    }

    /// `add`, committed now: buffers the next round reads (a stream's slot), so no round mixes them with the set's.
    pub fn addNow(r: *Resident, buffers: []const mtl.Buffer) void {
        if (r.held.join(buffers)) r.held.flush();
    }

    /// A buffer about to be freed leaves the set now: the set holds its own reference until a commit drops it.
    pub fn remove(r: *Resident, buffer: mtl.Buffer) void {
        r.held.leave(buffer);
    }

    /// The hook a family's allocator calls for buffers it makes or frees after the set was built; `r` must not move.
    pub fn follow(r: *Resident) mtl.ResidencySet.Follow {
        return .{ .ctx = r, .join = joinOne, .leave = leaveOne };
    }

    fn joinOne(ctx: *anyopaque, buffer: mtl.Buffer) void {
        const r: *Resident = @ptrCast(@alignCast(ctx));
        r.add(&.{buffer});
    }

    fn leaveOne(ctx: *anyopaque, buffer: mtl.Buffer) void {
        const r: *Resident = @ptrCast(@alignCast(ctx));
        r.remove(buffer);
    }

    fn flushFn(ctx: *anyopaque) void {
        const r: *Resident = @ptrCast(@alignCast(ctx));
        r.held.flush();
    }

    /// The idle keepalive's target: waiting joins committed, then a tiny command buffer using the set; `r` stays put.
    pub fn target(r: *Resident) mtl.keepalive.Target {
        return .{ .queue = r.work, .sets = @as(*const [1]mtl.ResidencySet, &r.held.set), .scratch = r.scratch, .flush = .{ .ctx = r, .run = flushFn } };
    }

    pub fn deinit(r: *Resident) void {
        r.work.removeResidencySet(r.held.set);
        r.held.deinit();
        r.held.set.deinit();
        r.scratch.deinit();
    }
};

/// What a residency set holds and what waits to join it, inside its load limit (`Set` is a counting fake in tests).
pub fn Held(comptime Set: type, comptime Buf: type) type {
    return struct {
        const Self = @This();
        const Member = struct { bytes: u64, joined: bool };

        gpa: std.mem.Allocator,
        io: std.Io,
        set: Set,
        name: []const u8,
        ram: ?u64,
        limit: ?u64 = null,
        bytes: u64 = 0, // the set's buffers and those waiting to join it: what the bound counts
        members: std.AutoHashMapUnmanaged(usize, Member) = .empty,
        joining: std.ArrayList(Buf) = .empty, // added and committed at the next flush, never during a request
        mutex: std.Io.Mutex = .init, // the engine's thread joins and leaves, the keepalive's ticker flushes

        /// `buffers` and `extra` in `set`, committed, if together they fit; else null and nothing is added.
        pub fn init(gpa: std.mem.Allocator, io: std.Io, set: Set, buffers: []const Buf, extra: Buf, name: []const u8, ram: ?u64) !?Self {
            return initUnder(gpa, io, set, buffers, extra, name, ram, null);
        }

        /// A family-selected ceiling applies at creation and to every later join.
        pub fn initUnder(gpa: std.mem.Allocator, io: std.Io, set: Set, buffers: []const Buf, extra: Buf, name: []const u8, ram: ?u64, limit: ?u64) !?Self {
            const bytes = sum(buffers) + extra.length();
            if (!mtl.ResidencySet.fitsUnder(0, bytes, ram, limit)) {
                mtl.ResidencySet.refusedUnder(name, 0, bytes, ram, limit);
                return null;
            }
            var h: Self = .{ .gpa = gpa, .io = io, .set = set, .name = name, .ram = ram, .limit = limit, .bytes = bytes };
            errdefer h.members.deinit(gpa);
            try h.members.ensureTotalCapacity(gpa, @intCast(buffers.len + 1));
            for (buffers) |b| h.place(b);
            h.place(extra);
            h.set.commit();
            return h;
        }

        fn place(h: *Self, b: Buf) void {
            const gop = h.members.getOrPutAssumeCapacity(key(b));
            if (gop.found_existing) return;
            gop.value_ptr.* = .{ .bytes = b.length(), .joined = true };
            h.set.add(b);
        }

        /// `buffers` (those not yet members) wait to join, if they fit beside the set; false: refused, logged.
        pub fn join(h: *Self, buffers: []const Buf) bool {
            h.mutex.lockUncancelable(h.io);
            defer h.mutex.unlock(h.io);
            var more: u64 = 0;
            for (buffers) |b| if (!h.members.contains(key(b))) {
                more += b.length();
            };
            if (more == 0) return true;
            if (!mtl.ResidencySet.fitsUnder(h.bytes, more, h.ram, h.limit)) {
                mtl.ResidencySet.refusedUnder(h.name, h.bytes, more, h.ram, h.limit);
                return false;
            }
            h.members.ensureUnusedCapacity(h.gpa, @intCast(buffers.len)) catch return false;
            h.joining.ensureUnusedCapacity(h.gpa, buffers.len) catch return false;
            for (buffers) |b| {
                const gop = h.members.getOrPutAssumeCapacity(key(b));
                if (gop.found_existing) continue;
                gop.value_ptr.* = .{ .bytes = b.length(), .joined = false };
                h.joining.appendAssumeCapacity(b);
            }
            h.bytes += more;
            return true;
        }

        /// `b` leaves before its owner frees it: out of the set with a commit of its own, or off the waiting list.
        pub fn leave(h: *Self, b: Buf) void {
            h.mutex.lockUncancelable(h.io);
            defer h.mutex.unlock(h.io);
            const m = h.members.fetchRemove(key(b)) orelse return;
            h.bytes -= m.value.bytes;
            if (m.value.joined) {
                h.set.remove(b);
                h.set.commit(); // nothing else is staged, so this drops only the set's reference to `b`
                return;
            }
            for (h.joining.items, 0..) |w, i| if (key(w) == key(b)) {
                _ = h.joining.swapRemove(i);
                break;
            };
        }

        /// The waiting buffers added and committed: the keepalive's tick, while no request runs.
        pub fn flush(h: *Self) void {
            h.mutex.lockUncancelable(h.io);
            defer h.mutex.unlock(h.io);
            if (h.joining.items.len == 0) return;
            for (h.joining.items) |b| {
                h.set.add(b);
                h.members.getPtr(key(b)).?.joined = true;
            }
            h.set.commit();
            h.joining.clearRetainingCapacity();
        }

        pub fn deinit(h: *Self) void {
            h.members.deinit(h.gpa);
            h.joining.deinit(h.gpa);
        }

        fn key(b: Buf) usize {
            return @intFromPtr(b.id);
        }

        fn sum(buffers: []const Buf) u64 {
            var n: u64 = 0;
            for (buffers) |b| n += b.length();
            return n;
        }
    };
}

/// A residency set that counts what it is asked: the CPU tests' stand-in for Metal's.
const Fake = struct {
    adds: u32 = 0,
    removes: u32 = 0,
    commits: u32 = 0,

    fn add(f: *Fake, _: Buffer) void {
        f.adds += 1;
    }

    fn remove(f: *Fake, _: Buffer) void {
        f.removes += 1;
    }

    fn commit(f: *Fake) void {
        f.commits += 1;
    }

    const Buffer = struct {
        id: *const u8,
        len: u64,

        fn length(b: Buffer) u64 {
            return b.len;
        }
    };
};

const FakeSet = struct {
    f: *Fake,

    pub fn add(s: FakeSet, b: Fake.Buffer) void {
        s.f.add(b);
    }

    pub fn remove(s: FakeSet, b: Fake.Buffer) void {
        s.f.remove(b);
    }

    pub fn commit(s: FakeSet) void {
        s.f.commit();
    }
};

test "a set is built only inside the 70% limit, and never when RAM could not be read" {
    const GiB = cache_fit.GiB;
    const ids: [4]u8 = .{ 0, 1, 2, 3 };
    const big = [_]Fake.Buffer{ .{ .id = &ids[0], .len = 100 * GiB }, .{ .id = &ids[1], .len = 72 * GiB } };
    const scratch: Fake.Buffer = .{ .id = &ids[3], .len = 64 };
    var f: Fake = .{};
    const H = Held(FakeSet, Fake.Buffer);
    try std.testing.expect((try H.init(std.testing.allocator, std.testing.io, .{ .f = &f }, &big, scratch, "test", 240 * GiB)) == null);
    try std.testing.expect((try H.init(std.testing.allocator, std.testing.io, .{ .f = &f }, &big, scratch, "test", null)) == null);
    try std.testing.expectEqual(Fake{}, f); // a refused set adds and commits nothing
    var h = (try H.init(std.testing.allocator, std.testing.io, .{ .f = &f }, &big, scratch, "test", 256 * GiB)).?;
    defer h.deinit();
    try std.testing.expectEqual(Fake{ .adds = 3, .commits = 1 }, f);
    try std.testing.expectEqual(172 * GiB + 64, h.bytes);
}

test "buffers made later wait for a flush inside the limit, and leave at once before they are freed" {
    const GiB = cache_fit.GiB;
    const ids: [8]u8 = @splat(0);
    const weights = [_]Fake.Buffer{.{ .id = &ids[0], .len = 172 * GiB }};
    const scratch: Fake.Buffer = .{ .id = &ids[1], .len = 64 };
    const state: Fake.Buffer = .{ .id = &ids[2], .len = 2 * GiB };
    const partials: Fake.Buffer = .{ .id = &ids[3], .len = 4 * GiB };
    const late: Fake.Buffer = .{ .id = &ids[4], .len = GiB };
    var f: Fake = .{};
    var h = (try Held(FakeSet, Fake.Buffer).init(std.testing.allocator, std.testing.io, .{ .f = &f }, &weights, scratch, "test", 256 * GiB)).?;
    defer h.deinit();
    f = .{};
    try std.testing.expect(h.join(&.{state})); // 174 GiB with the margin: inside 179.2
    try std.testing.expect(h.join(&.{state})); // a member already: counted once
    try std.testing.expectEqual(172 * GiB + 64 + 2 * GiB, h.bytes);
    try std.testing.expectEqual(Fake{}, f); // nothing touches the set until the flush
    try std.testing.expect(!h.join(&.{partials})); // 4 GiB more passes the limit: refused, not counted
    try std.testing.expectEqual(172 * GiB + 64 + 2 * GiB, h.bytes);
    h.flush();
    try std.testing.expectEqual(Fake{ .adds = 1, .commits = 1 }, f);
    h.flush(); // nothing waits: no commit
    try std.testing.expectEqual(@as(u32, 1), f.commits);
    try std.testing.expect(h.join(&.{late}));
    h.leave(late); // freed before any flush: off the waiting list, the set untouched
    try std.testing.expectEqual(Fake{ .adds = 1, .commits = 1 }, f);
    h.leave(state); // in the set: removed with a commit of its own, so its memory goes when its owner frees it
    try std.testing.expectEqual(Fake{ .adds = 1, .removes = 1, .commits = 2 }, f);
    h.leave(partials); // never a member: nothing
    try std.testing.expectEqual(172 * GiB + 64, h.bytes);
    h.flush();
    try std.testing.expectEqual(Fake{ .adds = 1, .removes = 1, .commits = 2 }, f);
    try std.testing.expect(h.join(&.{partials})); // room again once the state left
    const slot: Fake.Buffer = .{ .id = &ids[5], .len = GiB };
    f = .{};
    if (h.join(&.{slot})) h.flush(); // Resident.addNow: a slot's commit takes what waits with it
    try std.testing.expectEqual(Fake{ .adds = 2, .commits = 1 }, f);
}

test "an operator load ceiling holds the shared host and stays in force for later buffers" {
    const GiB = cache_fit.GiB;
    const ids: [4]u8 = @splat(0);
    const buffers = [_]Fake.Buffer{.{ .id = &ids[0], .len = 194 * GiB }};
    const scratch: Fake.Buffer = .{ .id = &ids[1], .len = 64 };
    const state: Fake.Buffer = .{ .id = &ids[2], .len = 3 * GiB };
    const more: Fake.Buffer = .{ .id = &ids[3], .len = GiB };
    const cap: u64 = 214_700_000_000;
    const H = Held(FakeSet, Fake.Buffer);
    var f: Fake = .{};
    try std.testing.expect((try H.init(std.testing.allocator, std.testing.io, .{ .f = &f }, &buffers, scratch, "test", 256 * GiB)) == null);
    try std.testing.expect((try H.initUnder(std.testing.allocator, std.testing.io, .{ .f = &f }, &buffers, scratch, "test", null, cap)) == null);
    try std.testing.expectEqual(Fake{}, f);
    const opened = try H.initUnder(std.testing.allocator, std.testing.io, .{ .f = &f }, &buffers, scratch, "test", 256 * GiB, cap);
    try std.testing.expect(opened != null);
    var h = opened.?;
    defer h.deinit();
    try std.testing.expectEqual(Fake{ .adds = 2, .commits = 1 }, f);
    try std.testing.expectEqual(cap, h.limit.?);
    f = .{};
    try std.testing.expect(h.join(&.{state}));
    try std.testing.expect(!h.join(&.{more}));
    try std.testing.expectEqual(197 * GiB + 64, h.bytes);
    try std.testing.expectEqual(Fake{}, f);
    h.flush();
    try std.testing.expectEqual(Fake{ .adds = 1, .commits = 1 }, f);
    h.leave(state);
    try std.testing.expectEqual(Fake{ .adds = 1, .removes = 1, .commits = 2 }, f);
    try std.testing.expect(h.join(&.{more}));
    try std.testing.expectEqual(195 * GiB + 64, h.bytes);
}

test "an explicit residency ceiling keeps the margin, RAM backstop and overflow checks" {
    const gib: u64 = 1 << 30;
    const ram = 256 * gib;
    const cap: u64 = 214_700_000_000;
    for ([_]u64{ 1787, 1798, 1826, 1940 }) |tenths| {
        const bytes = tenths * gib / 10;
        try std.testing.expect(!mtl.ResidencySet.fits(0, bytes, ram));
        try std.testing.expect(mtl.ResidencySet.fitsUnder(0, bytes, ram, cap));
    }
    try std.testing.expect(mtl.ResidencySet.fitsUnder(0, cap - mtl.ResidencySet.margin, ram, cap));
    try std.testing.expect(!mtl.ResidencySet.fitsUnder(0, cap - mtl.ResidencySet.margin + 1, ram, cap));
    try std.testing.expect(!mtl.ResidencySet.fitsUnder(0, 1, null, cap));
    try std.testing.expect(!mtl.ResidencySet.fitsUnder(0, 1, ram, 0));
    try std.testing.expect(mtl.ResidencySet.fitsUnder(0, 8 * gib, ram, 10 * gib));
    try std.testing.expect(!mtl.ResidencySet.fitsUnder(0, 8 * gib + 1, ram, 10 * gib));
    try std.testing.expect(mtl.ResidencySet.fitsUnder(0, 254 * gib, ram, 512 * gib));
    try std.testing.expect(!mtl.ResidencySet.fitsUnder(0, 254 * gib + 1, ram, 512 * gib));
    const most = std.math.maxInt(u64);
    try std.testing.expect(!mtl.ResidencySet.fitsUnder(most - 16, 32, most, most));
}
