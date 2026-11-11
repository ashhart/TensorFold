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

pub fn fakeDriver() cuda.Driver {
    Device.next = 0x100000;
    Device.copies = 0;
    return Device.driver();
}

/// Exercise the actual lane open and explicit snapshot entry points with a CPU control peer.
pub fn run(comptime Cuda: type, comptime open: anytype, comptime restore: anytype, comptime drop: anytype, explicit: bool, drafting: bool, with_head: bool) !void {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the peer rank is a Unix socket pair
    const a = std.testing.allocator;
    Device.next = 0x100000;
    Device.copies = 0;
    var driver = Device.driver();
    var stream: cuda.Stream = .{ .d = &driver, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.exl3_prefill = .{};
    engine.exl3_chain = .{};
    engine.gpa = a;
    engine.driver = &driver;
    engine.stream = &stream;
    engine.eos = 7;
    engine.chunk = 16;
    engine.round_graph = .{};
    engine.heads = .full;
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
    self.exl3 = false;
    self.kv = .{};
    self.lanes = .empty;
    self.parts = .{};
    defer self.lanes.deinit(a);
    var kept: snap.Snap = .{ .rank_id = 99, .buf = undefined, .at = 2, .history = .{ 8, 9 }, .head = with_head };
    kept.buf.ptr = 0x100;
    var s = try lanes.Stream.init(a, .{ .id = "restore", .prompt = &.{ 1, 2, 3, 4 }, .max_new = 1, .drafts = drafting, .reuse = if (explicit) .{} else .{ .saved = &kept, .at = 2 } });
    defer s.deinit(a);
    const lane = try open(&self, &s);
    defer {
        rank.stopped = true;
        drop(&self, lane);
    }
    const refused = drafting and !with_head; // a state without the head's rows can't serve a drafting stream
    if (explicit) {
        if (refused) {
            try std.testing.expectError(error.NoHeadState, restore(&self, &s, &kept));
        } else try restore(&self, &s, &kept);
    }
    const created = try peer.receiveHeader(wire.deadline(1000));
    try std.testing.expectEqual(wire.Kind.create, created.kind);
    try std.testing.expectEqual(@as(u64, 1), created.object);
    if (refused) {
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
        if (drafting) {
            try std.testing.expectEqual(@as(usize, 2), lane.draft.?.len);
            try std.testing.expectEqual(@as(usize, 0), lane.draft.?.drafted);
        } else try std.testing.expect(lane.draft == null);
        try std.testing.expect(!s.reuse_failed);
        try std.testing.expectEqual(@as(u64, 3), rank.next);
        try std.testing.expect(Device.copies > 0);
        try std.testing.expectEqual([2]i64{ 8, 9 }, lane.seq.history);
    }
}

/// A follower rank's real frame loop over a virtual device: `frames`, then `check` on its sequence and window.
fn follower(comptime Cuda: type, frames: []const wire.Header, setup: *const fn (*prompt.Seq, *mtp.Draft) void, check: *const fn (*ranks.Session, *prompt.Seq, *mtp.Draft) anyerror!void, want: ?anyerror) !void {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest; // the leader is a Unix socket pair
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
    engine.heads = .full;
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
    const leader: wire.Link = .{ .fd = pair[1] };
    for (frames) |h| try leader.sendHeader(h, wire.deadline(1000));
    var rank: ranks.Session = .{ .gpa = a, .rank = 1, .link = .{ .fd = pair[0] }, .collective = undefined, .engine = &engine, .drafting = true };
    rank.collective.failed = false;
    rank.collective.comm = null;
    defer rank.seqs.deinit(a);
    defer rank.windows.deinit(a);
    defer rank.saved.deinit(a);
    const q = try a.create(prompt.Seq);
    defer a.destroy(q);
    q.* = try engine.newSeq(64);
    defer q.deinit(a);
    q.rank_id = 1;
    q.fed = 2;
    q.history = .{ 8, 9 };
    try rank.seqs.put(a, 1, q);
    try rank.windows.put(a, 1, try mtp.Draft.init(a, &driver, stream, 64, 1, engine.chunk, .full));
    const w = rank.windows.getPtr(1).?;
    defer w.deinit();
    defer w.log = null;
    setup(q, w);
    var self: Cuda = undefined;
    self.set = null;
    self.stream = stream;
    const tri = self.tri();
    const done = rank.follow(@TypeOf(tri), tri, 0);
    if (want) |err| {
        try std.testing.expectError(err, done);
    } else try done;
    try check(&rank, q, w);
}

/// Each case runs independently in the CUDA family's standard host test root.
pub fn Cases(comptime Cuda: type, comptime open: anytype, comptime restore: anytype, comptime drop: anytype) type {
    return struct {
        test "rank cache restore: drafted open refuses a plain snapshot before rank restore" {
            try run(Cuda, open, restore, drop, false, true, false);
        }

        test "rank cache restore: explicit drafted restore refuses a plain snapshot" {
            try run(Cuda, open, restore, drop, true, true, false);
        }

        test "rank cache restore: drafted open resumes a snapshot with the head's rows on every rank" {
            try run(Cuda, open, restore, drop, false, true, true);
        }

        test "rank cache restore: explicit drafted restore resumes a snapshot with the head's rows" {
            try run(Cuda, open, restore, drop, true, true, true);
        }

        test "rank cache restore: plain open retains the rank restore command" {
            try run(Cuda, open, restore, drop, false, false, false);
        }

        test "rank cache restore: explicit plain restore retains the rank command" {
            try run(Cuda, open, restore, drop, true, false, false);
        }

        test "rank cache restore: a plain stream resumes a snapshot with the head's rows" {
            try run(Cuda, open, restore, drop, false, false, true);
        }

        test "rank cache save: a follower's window flushes before its copy, and an open one refuses" {
            const frames = [_]wire.Header{.{ .kind = .save, .sequence = 1, .object = 1, .arg = 5 }};
            try follower(Cuda, &frames, struct {
                fn setup(_: *prompt.Seq, w: *mtp.Draft) void {
                    w.log = w.kc; // flushLate refuses an open window with pending rows before any launch
                    w.pending = 1;
                    w.open = true;
                }
            }.setup, struct {
                fn check(rank: *ranks.Session, _: *prompt.Seq, _: *mtp.Draft) !void {
                    try std.testing.expectEqual(@as(usize, 0), Device.copies);
                    try std.testing.expectEqual(@as(u32, 0), rank.saved.count());
                }
            }.check, error.CannotRewind);
        }

        test "rank cache restore: a follower saves, and a restore starts its window over" {
            const frames = [_]wire.Header{
                .{ .kind = .save, .sequence = 1, .object = 1, .arg = 5 },
                .{ .kind = .restore, .sequence = 2, .object = 1, .arg = 5 },
                .{ .kind = .stop, .sequence = 3 },
            };
            try follower(Cuda, &frames, struct {
                fn setup(_: *prompt.Seq, w: *mtp.Draft) void {
                    w.len = 9; // the follower's window never holds the head's rows, and its counters don't outlive a restore
                    w.rows = 1;
                }
            }.setup, struct {
                fn check(rank: *ranks.Session, q: *prompt.Seq, w: *mtp.Draft) !void {
                    try std.testing.expect(Device.copies > 0);
                    try std.testing.expectEqual(@as(u32, 1), rank.saved.count());
                    try std.testing.expectEqual(@as(usize, 2), q.fed);
                    try std.testing.expectEqual([2]i64{ 8, 9 }, q.history);
                    try std.testing.expectEqual(@as(usize, 0), w.rows);
                    try std.testing.expectEqual(@as(usize, 0), w.pending);
                    try std.testing.expectEqual(@as(usize, 0), w.len);
                    var it = rank.saved.valueIterator();
                    while (it.next()) |kept| snap.drop(rank.gpa, rank.engine.stream.*, kept.*);
                }
            }.check, null);
        }
    };
}
