//! Real warm-up over fake NCCL and CUDA calls; the outer escape makes the unbounded control fail, not hang.
const std = @import("std");
const cuda = @import("cuda");
const warm = @import("cuda_warm_peers.zig");
const t = std.testing;

const Fake = struct {
    const Stage = enum { posting, stream, cleanup };
    var stage: Stage = .posting;
    var ms = std.atomic.Value(u64).init(0);
    var entered = std.atomic.Value(bool).init(false);
    var aborted = std.atomic.Value(bool).init(false);
    var escaped = std.atomic.Value(bool).init(false);
    var done = std.atomic.Value(bool).init(false);
    var replies: ?u64 = null;
    var stops: [16]?[]const u8 = @splat(null);
    var mine: usize = 0;
    var world: usize = 2;
    var order: u8 = 0;
    var abort_count: usize = 0;
    var syncs: usize = 0;
    var message: [2048]u8 = undefined;
    var message_len: usize = 0;
    var allocated: usize = 0;
    var freed: usize = 0;
    var posts: usize = 0;

    fn reset(rank: usize, count: usize, where: Stage, answer: ?u64) void {
        ms.store(0, .release);
        entered.store(false, .release);
        aborted.store(false, .release);
        escaped.store(false, .release);
        done.store(false, .release);
        stops = @splat(null);
        stage = where;
        replies = answer;
        mine = rank;
        world = count;
        order = 0;
        abort_count = 0;
        syncs = 0;
        message_len = 0;
        allocated = 0;
        freed = 0;
        posts = 0;
    }
    fn pause() void {
        std.Io.sleep(t.io, .fromMilliseconds(1), .awake) catch {};
    }
    fn wait() void {
        entered.store(true, .release);
        while (!aborted.load(.acquire) and !escaped.load(.acquire)) {
            if (replies) |at| if (ms.load(.acquire) >= at) return;
            pause();
        }
    }
    fn escape() void {
        for (0..500) |_| {
            if (done.load(.acquire)) return;
            if (@hasDecl(warm, "warmWith")) pause() else (Clock{}).pause();
        }
        escaped.store(true, .release);
    }
    const Clock = struct {
        pub fn now(_: @This()) u64 {
            return ms.load(.acquire);
        }
        pub fn pause(_: @This()) void {
            if (entered.load(.acquire)) {
                const now_ms = ms.load(.acquire);
                if (replies == null or now_ms < replies.?) _ = ms.fetchAdd(1, .acq_rel);
            }
            Fake.pause();
        }
    };
    fn alloc(ptr: *u64, n: usize) callconv(.c) cuda.abi.Result {
        ptr.* = 0x4000;
        allocated = n;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        freed += 1;
        return 0;
    }
    fn stream(ptr: *cuda.abi.Stream, flags: c_uint) callconv(.c) cuda.abi.Result {
        std.debug.assert(flags == 0);
        ptr.* = null;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        if (stage == .stream or (stage == .cleanup and syncs > 1)) wait();
        return 0;
    }
    fn destroy(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn start() callconv(.c) cuda.nccl.Result {
        return 0;
    }
    fn end() callconv(.c) cuda.nccl.Result {
        if (stage == .posting) wait();
        return 0;
    }
    fn transfer(_: u64, n: usize, _: cuda.nccl.DataType, peer: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        std.debug.assert(n == 4 << 20 and peer != mine);
        posts += 1;
        return 0;
    }
    fn gathered(_: u64, _: u64, n: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        std.debug.assert(n == 4 << 20);
        posts += 1;
        return 0;
    }
    fn abort(_: cuda.nccl.Comm) callconv(.c) cuda.nccl.Result {
        std.debug.assert(order == 1);
        order = 2;
        abort_count += 1;
        aborted.store(true, .release);
        return 0;
    }
    const Watch = struct {
        pub fn fail(name: []const u8) void {
            std.debug.assert(message_len > 0 and order == 0);
            order = 1;
            for (0..world) |rank| if (rank != mine) {
                stops[rank] = name;
            };
        }
        pub fn stop() void {
            std.debug.assert(order == 2 and aborted.load(.acquire));
            stops[mine] = "PeerWarmupTimedOut";
            order = 3;
        }
    };
    fn log(text: []const u8) void {
        @memcpy(message[0..text.len], text);
        message_len = text.len;
    }
    const Action = struct {
        g: *cuda.tp_group.Group,
        peers: warm.Peers,
        pub fn expired(a: @This()) void {
            if (@hasDecl(warm, "timeoutWith")) warm.timeoutWith(a.g, a.peers, &.{}, Watch, log);
        }
    };
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuStreamCreate = stream;
        d.api.cuStreamSynchronize = sync;
        d.api.cuStreamDestroy_v2 = destroy;
        return d;
    }
    fn group() cuda.tp_group.Group {
        var g: cuda.tp_group.Group = .{ .rank = mine, .world = world, .comm = null, .lib = undefined };
        g.lib.api.ncclGroupStart = start;
        g.lib.api.ncclGroupEnd = end;
        g.lib.api.ncclSend = transfer;
        g.lib.api.ncclRecv = transfer;
        g.lib.api.ncclAllGather = gathered;
        g.lib.api.ncclCommAbort = abort;
        return g;
    }
    fn run(rank: usize, count: usize, peers: warm.Peers, where: Stage, answer: ?u64) !void {
        reset(rank, count, where, answer);
        var g = group();
        const d = driver();
        const guard = try std.Thread.spawn(.{}, escape, .{});
        defer {
            done.store(true, .release);
            guard.join();
        }
        if (@hasDecl(warm, "warmWith")) {
            try warm.warmWith(&d, &g, peers, Clock{}, 100, Action{ .g = &g, .peers = peers });
        } else {
            try warm.warm(&d, &g, peers);
        }
    }
};

test "GLM warm peers: a missing peer in posting or stream wait aborts once and stops every rank by name" {
    for ([_]Fake.Stage{ .posting, .stream, .cleanup }) |stage| for (0..2) |rank| {
        try t.expectError(error.PeerWarmupTimedOut, Fake.run(rank, 2, .every, stage, null));
        try t.expect(!Fake.escaped.load(.acquire));
        try t.expectEqual(@as(u64, 100), Fake.ms.load(.acquire));
        try t.expectEqual(@as(usize, 1), Fake.abort_count);
        try t.expectEqual(@as(u8, 3), Fake.order);
        for (Fake.stops[0..2]) |cause| try t.expectEqualStrings("PeerWarmupTimedOut", cause orelse "still waiting");
        var expected: [512]u8 = undefined;
        const line = try std.fmt.bufPrint(&expected, "rank {d}: peer warm-up: 4 MiB send/recv with rank(s) {d} not finished after 300 s (NCCL transport unknown); PeerWarmupTimedOut; stopping every rank", .{ rank, 1 - rank });
        try t.expectEqualStrings(line, Fake.message[0..Fake.message_len]);
        try t.expectEqual(@as(usize, 16 << 20), Fake.allocated);
        try t.expectEqual(@as(usize, 1), Fake.freed);
        try t.expectEqual(@as(usize, 2), Fake.posts);
    };
}

test "GLM warm peers: a peer answering just inside the bound passes without abort or stopping another rank" {
    for ([_]Fake.Stage{ .posting, .stream }) |stage| for (0..2) |rank| {
        try Fake.run(rank, 2, .every, stage, 99);
        try t.expect(!Fake.escaped.load(.acquire));
        if (@hasDecl(warm, "warmWith")) try t.expectEqual(@as(u64, 99), Fake.ms.load(.acquire));
        try t.expectEqual(@as(usize, 0), Fake.abort_count);
        try t.expectEqual(@as(usize, 0), Fake.message_len);
        for (Fake.stops[0..2]) |cause| try t.expect(cause == null);
        try t.expectEqual(@as(usize, 1), Fake.freed);
    };
}

test "GLM warm peers: timeout names only selected ring peers and observed NCCL transport" {
    if (!@hasDecl(warm, "timeoutLine")) return error.SkipZigTest;
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try warm.timeoutLine(&w, 1, 4, .neighbours, 5);
    try t.expectEqualStrings("rank 1: peer warm-up: 4 MiB send/recv with rank(s) 0,2 not finished after 300 s (NCCL transport P2P+NET); PeerWarmupTimedOut; stopping every rank", w.buffered());
    w = std.Io.Writer.fixed(&buf);
    try warm.timeoutLine(&w, 1, 4, .every, 0);
    try t.expect(std.mem.indexOf(u8, w.buffered(), "rank(s) 0,2,3") != null);
}

test "GLM warm peers: gather's all-gather that never completes stops every rank, named as the all-gather" {
    for (0..2) |rank| {
        try t.expectError(error.PeerWarmupTimedOut, Fake.run(rank, 2, .gather, .stream, null));
        try t.expect(!Fake.escaped.load(.acquire));
        try t.expectEqual(@as(u64, 100), Fake.ms.load(.acquire));
        try t.expectEqual(@as(usize, 1), Fake.abort_count);
        for (Fake.stops[0..2]) |cause| try t.expectEqualStrings("PeerWarmupTimedOut", cause orelse "still waiting");
        var expected: [512]u8 = undefined;
        const line = try std.fmt.bufPrint(&expected, "rank {d}: peer warm-up: 4 MiB all-gather not finished after 300 s (NCCL transport unknown); PeerWarmupTimedOut; stopping every rank", .{rank});
        try t.expectEqualStrings(line, Fake.message[0..Fake.message_len]);
        try t.expectEqual(@as(usize, 1), Fake.posts);
    }
}
