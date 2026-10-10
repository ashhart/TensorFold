//! A peer's disk reservation covers its snapshot half until rank 0 has committed or removed the shared index entry.
const std = @import("std");
const disk = @import("lanes").learned_disk;
pub const State = struct {
    admission: disk.Admission = .{},
    cap: u64 = std.math.maxInt(u64),
    pending: ?struct { key: u64, bytes: u64 } = null,
    pub fn need(s: *const State, dir: [:0]const u8, payload: u64, extra: u64) u64 {
        if (s.pending != null or s.admission.waiting(dir)) return std.math.maxInt(u64);
        const bytes = std.math.add(u64, payload, extra) catch return std.math.maxInt(u64);
        const floor_need = s.admission.shortfall(dir, bytes) orelse return std.math.maxInt(u64);
        const used_bytes = @import("lanes").learned_dirs.total(@import("lanes").learned_dirs.rootOf(dir)) catch return std.math.maxInt(u64);
        return @max(floor_need, (std.math.add(u64, used_bytes, bytes) catch return std.math.maxInt(u64)) -| s.cap);
    }
    pub fn reserve(s: *State, dir: [:0]const u8, key: u64, payload: u64, extra: u64) !void {
        if (s.pending != null) return error.PeerDiskBusy;
        const bytes = std.math.add(u64, payload, extra) catch return error.PeerDiskFull;
        if (s.need(dir, payload, extra) != 0) return error.PeerDiskFull;
        if (s.admission.reserve(dir, bytes) != .ready) return error.PeerDiskFull;
        s.pending = .{ .key = key, .bytes = bytes };
    }
    pub fn finish(s: *State, dir: [:0]const u8, key: u64, success: bool) !void {
        const pending = s.pending orelse return;
        if (pending.key != key) return error.PeerDiskOutOfStep;
        s.admission.finish(dir, pending.bytes, success);
        s.pending = null;
    }
};
test "a peer floor or in-flight reservation prevents writes and failure backs off until space changes" {
    const Fake = struct {
        var free_bytes: ?u64 = 109;
        var tick: u64 = 10;
        fn free(_: [:0]const u8) ?u64 {
            return free_bytes;
        }
        fn clock() ?u64 {
            return tick;
        }
    };
    var s: State = .{ .admission = .{ .floor = 100, .free = Fake.free, .clock = Fake.clock } };
    try std.testing.expectEqual(@as(u64, 1), s.need(".", 8, 2));
    try std.testing.expectError(error.PeerDiskFull, s.reserve(".", 1, 8, 2));
    try std.testing.expect(s.pending == null);
    Fake.free_bytes = 110;
    try s.reserve(".", 1, 8, 2);
    try std.testing.expectEqual(std.math.maxInt(u64), s.need(".", 1, 0));
    try std.testing.expectError(error.PeerDiskOutOfStep, s.finish(".", 2, true));
    try s.finish(".", 1, false);
    try std.testing.expectEqual(@as(u64, 0), s.admission.reserved);
    try std.testing.expectError(error.PeerDiskFull, s.reserve(".", 1, 8, 2));
    Fake.free_bytes = 120;
    try s.reserve(".", 1, 8, 2);
    try s.finish(".", 1, true);
    try std.testing.expectEqual(@as(u8, 0), s.admission.failures);
}

test "lost reservation acknowledgement rolls back idempotently and each rank enforces its own cap" {
    const Fake = struct {
        fn free(_: [:0]const u8) ?u64 {
            return 1 << 30;
        }
        fn clock() ?u64 {
            return 100;
        }
    };
    var s: State = .{ .admission = .{ .floor = 0, .free = Fake.free, .clock = Fake.clock }, .cap = 9 };
    try std.testing.expectEqual(@as(u64, 1), s.need(".", 8, 2));
    try std.testing.expectError(error.PeerDiskFull, s.reserve(".", 1, 8, 2));
    s.cap = 100;
    try s.reserve(".", 1, 8, 2);
    // The caller cannot know whether the reserve reply arrived, so rollback also accepts no pending operation.
    try s.finish(".", 1, false);
    try s.finish(".", 1, false);
    try std.testing.expectEqual(@as(u64, 0), s.admission.reserved);
    s.admission.retry_at = 0;
    try s.reserve(".", 2, 8, 2);
    try std.testing.expectError(error.PeerDiskOutOfStep, s.finish(".", 1, false));
    try s.finish(".", 2, true);
}

test "each rank includes actual other-identity files in its independent cap before reservation" {
    const Fake = struct {
        fn free(_: [:0]const u8) ?u64 {
            return 1 << 30;
        }
        fn clock() ?u64 {
            return 100;
        }
    };
    const a = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const root = try temp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(root);
    var one: [1200]u8 = undefined;
    var two: [1200]u8 = undefined;
    var filename: [1400]u8 = undefined;
    const current = try std.fmt.bufPrintSentinel(&one, "{s}/0000000000000001", .{root}, 0);
    const other = try std.fmt.bufPrintSentinel(&two, "{s}/0000000000000002", .{root}, 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(current, 0o700));
    try std.testing.expectEqual(@as(c_int, 0), std.c.mkdir(other, 0o700));
    const fd = std.c.open(try std.fmt.bufPrintSentinel(&filename, "{s}/state.bin", .{other}, 0), .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestCreate;
    try std.testing.expectEqual(@as(c_int, 0), std.c.ftruncate(fd, 96));
    _ = std.c.close(fd);
    var rank0: State = .{ .admission = .{ .floor = 0, .free = Fake.free, .clock = Fake.clock }, .cap = 200 };
    var rank1: State = .{ .admission = .{ .floor = 0, .free = Fake.free, .clock = Fake.clock }, .cap = 100 };
    try std.testing.expectEqual(@as(u64, 0), rank0.need(current, 8, 2));
    try std.testing.expectEqual(@as(u64, 6), rank1.need(current, 8, 2));
    try std.testing.expectError(error.PeerDiskFull, rank1.reserve(current, 3, 8, 2));
    try std.testing.expect(rank1.pending == null);
    try @import("lanes").learned_dirs.removeOther(root, current, 2);
    try rank1.reserve(current, 3, 8, 2);
    try rank1.finish(current, 3, true);
    try std.testing.expectEqual(@as(u64, 0), rank1.admission.reserved);
}
