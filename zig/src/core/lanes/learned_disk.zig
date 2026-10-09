//! Reserve learned-state writes above a free-disk floor, with bounded backoff after refusals or write failures.
const std = @import("std");
const builtin = @import("builtin");
const Count = if (builtin.os.tag == .linux) u64 else u32;
const Stat = extern struct {
    bsize: u64,
    frsize: u64,
    blocks: Count,
    bfree: Count,
    bavail: Count,
    files: Count,
    ffree: Count,
    favail: Count,
    fsid: u64,
    flag: u64,
    namemax: u64,
    tail: [if (builtin.os.tag == .linux) 24 else 0]u8,
};
extern "c" fn statvfs(path: [*:0]const u8, result: *Stat) c_int;

pub fn available(dir: [:0]const u8) ?u64 {
    var stat: Stat = undefined;
    if (statvfs(dir, &stat) != 0) return null;
    return std.math.mul(u64, stat.frsize, stat.bavail) catch null;
}

fn now() ?u64 {
    var t: std.c.timespec = undefined;
    if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t) != 0 or t.sec < 0) return null;
    return @intCast(t.sec);
}

pub const Admission = struct {
    floor: u64 = 4 << 30,
    reserved: u64 = 0,
    failures: u8 = 0,
    retry_at: u64 = 0,
    failed_free: ?u64 = null,
    free: *const fn ([:0]const u8) ?u64 = available,
    clock: *const fn () ?u64 = now,

    pub const Result = enum { ready, refused, quiet };
    pub fn waiting(a: *const Admission, dir: [:0]const u8) bool {
        const tick = a.clock() orelse return true;
        const free = a.free(dir);
        const increased = if (free) |n| if (a.failed_free) |old| n > old else false else false;
        return tick < a.retry_at and !increased;
    }
    pub fn shortfall(a: *const Admission, dir: [:0]const u8, bytes: u64) ?u64 {
        _ = a.clock() orelse return null;
        const need = std.math.add(u64, a.floor, a.reserved) catch return null;
        const full = std.math.add(u64, need, bytes) catch return null;
        return full -| (a.free(dir) orelse return null);
    }
    pub fn refuse(a: *Admission, dir: [:0]const u8) void {
        _ = a.block(a.free(dir));
    }

    pub fn reserve(a: *Admission, dir: [:0]const u8, bytes: u64) Result {
        const space = a.free(dir);
        const changed = if (space) |n| if (a.failed_free) |old| n > old else false else false;
        const tick = a.clock() orelse return a.block(space);
        if (tick < a.retry_at and !changed) return .quiet;
        const need = std.math.add(u64, a.floor, a.reserved) catch return a.block(space);
        const full = std.math.add(u64, need, bytes) catch return a.block(space);
        if (space == null or space.? < full) return a.block(space);
        a.reserved += bytes;
        return .ready;
    }

    fn block(a: *Admission, space: ?u64) Result {
        a.failed_free = space;
        a.failures = @min(a.failures +| 1, 7);
        const wait = @min(@as(u64, 1) << @intCast(a.failures - 1), 60);
        a.retry_at = (a.clock() orelse 0) +| wait;
        return .refused;
    }

    pub fn finish(a: *Admission, dir: [:0]const u8, bytes: u64, success: bool) void {
        std.debug.assert(a.reserved >= bytes);
        a.reserved -= bytes;
        if (success) {
            a.failures = 0;
            a.retry_at = 0;
            a.failed_free = null;
        } else _ = a.block(a.free(dir));
    }
};

test "disk floor, unknown space, overflow, reservations and bounded write-failure retries" {
    const Fake = struct {
        var space: ?u64 = 110;
        var tick: u64 = 100;
        fn free(_: [:0]const u8) ?u64 {
            return space;
        }
        fn clock() ?u64 {
            return tick;
        }
    };
    var a: Admission = .{ .floor = 100, .free = Fake.free, .clock = Fake.clock };
    try std.testing.expectEqual(Admission.Result.ready, a.reserve(".", 10));
    try std.testing.expectEqual(Admission.Result.refused, a.reserve(".", 1));
    a.finish(".", 10, false);
    try std.testing.expectEqual(Admission.Result.quiet, a.reserve(".", 10));
    Fake.space = 120;
    try std.testing.expectEqual(Admission.Result.ready, a.reserve(".", 10));
    a.finish(".", 10, true);
    Fake.space = null;
    try std.testing.expectEqual(Admission.Result.refused, a.reserve(".", 1));
    Fake.tick += 100;
    Fake.space = std.math.maxInt(u64);
    try std.testing.expectEqual(Admission.Result.refused, a.reserve(".", std.math.maxInt(u64)));
    for (0..20) |_| {
        Fake.tick += 100;
        _ = a.reserve(".", std.math.maxInt(u64));
    }
    try std.testing.expectEqual(Fake.tick + 60, a.retry_at);
    try std.testing.expectEqual(@as(u64, 0), a.reserved);
}

test "real statvfs fails closed for a missing directory" {
    try std.testing.expect(available("/this-directory-does-not-exist") == null);
    try std.testing.expect(available(".") != null);
    try std.testing.expectEqual(@as(usize, if (builtin.os.tag == .linux) 112 else 64), @sizeOf(Stat));
}
