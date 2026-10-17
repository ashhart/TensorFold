const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const snap = @import("cuda_snap.zig");
const ranks = @import("cuda_rank_engine.zig");
const wire = @import("cuda_rank_wire.zig");

// Virtual device addresses; these callbacks never open CUDA or dereference device memory.
const Device = struct {
    var next: u64 = 0x100000;
    var copies: usize = 0;

    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = next;
        next += n + 256;
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        return 0;
    }
    fn fill(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn copy(_: u64, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        copies += 1;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemsetD8Async = fill;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuStreamSynchronize = sync;
        return d;
    }
};

/// Exercise the actual lane open and explicit snapshot entry points with a CPU control peer.
pub fn run(comptime Cuda: type, comptime open: anytype, comptime restore: anytype, comptime drop: anytype, explicit: bool, drafting: bool) !void {
    const a = std.testing.allocator;
    Device.next = 0x100000;
    Device.copies = 0;
    var driver = Device.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.gpa = a;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.eos = 7;
    engine.chunk = 16;
    engine.round_graph = .{};
    var pair: [2]std.posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    const peer: wire.Link = .{ .fd = pair[1] };
    // A successful create, followed by a successful restore if the lane attempts it.
    try peer.sendHeader(.{ .kind = .ack, .sequence = 1, .object = 1 }, wire.deadline(1000));
    try peer.sendHeader(.{ .kind = .ack, .sequence = 2, .object = 1 }, wire.deadline(1000));
    var rank: ranks.Session = .{ .gpa = a, .rank = 0, .link = .{ .fd = pair[0] }, .collective = undefined, .engine = &engine };
    rank.collective.failed = false;
    var ids = [_]u32{1};
    var head: mtp.Head = undefined;
    head.ids = &ids;
    var self: Cuda = undefined;
    self.gpa = a;
    self.context = 1024;
    self.engine = &engine;
    self.stream = stream;
    self.ranks = &rank;
    self.head = if (drafting) &head else null;
    self.lanes = .empty;
    defer self.lanes.deinit(a);
    var kept: snap.Snap = .{ .rank_id = 99, .buf = undefined, .at = 2, .history = .{ 8, 9 }, .head = false };
    kept.buf.ptr = 0x100;
    var s = try lanes.Stream.init(a, .{ .id = "restore", .prompt = &.{ 1, 2, 3, 4 }, .max_new = 1, .drafts = drafting, .reuse = if (explicit) .{} else .{ .saved = &kept, .at = 2 } });
    defer s.deinit(a);
    const lane = try open(&self, &s);
    defer {
        rank.stopped = true;
        drop(&self, lane);
    }
    if (explicit) {
        if (drafting) {
            try std.testing.expectError(error.NoHeadState, restore(&self, &s, &kept));
        } else try restore(&self, &s, &kept);
    }
    const created = try peer.receiveHeader(wire.deadline(1000));
    try std.testing.expectEqual(wire.Kind.create, created.kind);
    try std.testing.expectEqual(@as(u64, 1), created.object);
    if (drafting) {
        try std.testing.expectEqual(@as(usize, 0), lane.seq.fed);
        try std.testing.expectEqual(@as(usize, 0), lane.draft.?.len);
        try std.testing.expectEqual(@as(usize, 0), lane.filled.?);
        try std.testing.expectEqual(@as(u32, 0), s.cached);
        try std.testing.expectEqual(!explicit, s.reuse_failed);
        try std.testing.expectEqual(@as(u64, 2), rank.next);
        try std.testing.expectEqual(@as(usize, 0), Device.copies);
        try std.testing.expectEqual([2]i64{ 7, 7 }, lane.seq.history);
    } else {
        const resumed = try peer.receiveHeader(wire.deadline(1000));
        try std.testing.expectEqual(wire.Kind.restore, resumed.kind);
        try std.testing.expectEqual(@as(u64, 1), resumed.object);
        try std.testing.expectEqual(kept.rank_id, resumed.arg);
        try std.testing.expectEqual(@as(usize, 2), lane.seq.fed);
        try std.testing.expect(lane.draft == null);
        try std.testing.expect(!s.reuse_failed);
        try std.testing.expectEqual(@as(u64, 3), rank.next);
        try std.testing.expect(Device.copies > 0);
        try std.testing.expectEqual([2]i64{ 8, 9 }, lane.seq.history);
    }
}

/// Each case runs independently in the CUDA family's standard host test root.
pub fn Cases(comptime Cuda: type, comptime open: anytype, comptime restore: anytype, comptime drop: anytype) type {
    return struct {
        test "rank cache restore: drafted open refuses a plain snapshot before rank restore" {
            try run(Cuda, open, restore, drop, false, true);
        }

        test "rank cache restore: explicit drafted restore refuses a plain snapshot" {
            try run(Cuda, open, restore, drop, true, true);
        }

        test "rank cache restore: plain open retains the rank restore command" {
            try run(Cuda, open, restore, drop, false, false);
        }

        test "rank cache restore: explicit plain restore retains the rank command" {
            try run(Cuda, open, restore, drop, true, false);
        }
    };
}
