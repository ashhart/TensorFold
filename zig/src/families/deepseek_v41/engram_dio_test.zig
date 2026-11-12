//! Linux Engram shards on a disk whose direct reads need 4 KiB, or 512 bytes as today: the same rows in every mode.
const std = @import("std");
const testing = std.testing;
const linux = std.os.linux;
const er = @import("engram_rows.zig");
const dio = @import("engram_dio.zig");
const fixture = @import("engram_fixture.zig");

const o_direct: usize = @as(u32, @bitCast(linux.O{ .DIRECT = true }));

fn isDirect(fd: linux.fd_t) bool {
    return linux.fcntl(fd, linux.F.GETFL, 0) & o_direct != 0;
}

/// A disk whose direct reads start and end on `unit` (0: anywhere); any other fails with EINVAL, as the kernel's does.
const Disk = struct {
    var unit: u64 = 0;
    var refused: std.atomic.Value(u32) = .init(0);
    var taken: std.atomic.Value(u32) = .init(0); // direct reads it served

    fn pread(fd: linux.fd_t, buf: [*]u8, len: usize, off: i64) usize {
        if (isDirect(fd)) {
            if (unit != 0 and (@as(u64, @intCast(off)) % unit != 0 or len % unit != 0)) {
                _ = refused.fetchAdd(1, .monotonic);
                return @bitCast(-@as(isize, @backingInt(linux.E.INVAL)));
            }
            _ = taken.fetchAdd(1, .monotonic);
        }
        return linux.pread(fd, buf, len, off);
    }
};

const Mode = enum { sync, pool, uring, aio };

fn openRows(a: std.mem.Allocator, root: []const u8, mode: Mode) !*er.Rows {
    const r = try er.Rows.open(a, root, 0, 2, &.{1}, 4096);
    errdefer r.close();
    if (mode == .aio) {
        try r.startAio();
        r.aio.?.poll = false;
        return r;
    }
    if (r.aio) |*x| {
        x.deinit();
        r.aio = null;
    }
    if (mode != .uring) if (r.ring) |*g| {
        g.deinit();
        r.ring = null;
    };
    if (mode == .pool and r.pool == null) try r.startPool(8);
    if (mode == .sync) r.stopPool();
    return r;
}

test "engram rows: a disk whose direct reads need 4 KiB reads its shard buffered, every row the same in every mode" {
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var lut_sha: [32]u8 = undefined;
    const fx = try fixture.parseFx(arena.allocator(), &lut_sha);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.writeShard(a, io, tmp.dir, fx);
    const root = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    dio.test_pread = Disk.pread;
    defer dio.test_pread = null;
    defer Disk.unit = 0;
    // each mode's extents on today's disk: a 512-byte disk and a 4 KiB one read the same
    var base: [4][2]u64 = undefined;
    for ([_]u64{ 0, 512, 4096 }) |unit| for ([_]Mode{ .sync, .pool, .uring, .aio }, &base) |mode, *b| {
        Disk.unit = unit;
        Disk.refused.store(0, .monotonic);
        Disk.taken.store(0, .monotonic);
        const r = try openRows(a, root, mode);
        defer r.close();
        const t = &r.tables[0];
        if (unit == 0 and !t.direct) return error.SkipZigTest; // this file system refuses O_DIRECT: nothing to refuse
        for (fx.sets) |s| {
            const out = try a.alloc(u16, s.idx.len * 256);
            defer a.free(out);
            try r.rows(0, s.idx, 256, out);
            try testing.expectEqual(s.sha, fixture.sha(out));
        }
        if (unit == 0) b.* = .{ r.stats.reads, r.stats.bytes } else try testing.expectEqual(b.*, [2]u64{ r.stats.reads, r.stats.bytes });
        // the 4 KiB disk refused one read, the check at open, and served none direct; the others refused none
        const four = unit == 4096;
        try testing.expectEqual(!four, t.direct);
        try testing.expectEqual(!four, isDirect(t.fd));
        try testing.expectEqual(@as(u32, @intFromBool(four)), Disk.refused.load(.monotonic));
        // open's check and header read, then the serial and threaded reads of the rows
        const taken = Disk.taken.load(.monotonic);
        if (four) try testing.expectEqual(@as(u32, 0), taken) else if (mode == .sync or mode == .pool) try testing.expect(taken > 2);
    };
}
