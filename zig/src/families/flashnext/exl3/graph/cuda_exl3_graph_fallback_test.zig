//! The capture-failure fallback: a failed capture costs one eager round; two ranks decide it together.

const std = @import("std");
const windows = @import("builtin").os.tag == .windows;
const collective = @import("../../cuda_collective.zig");
const wire = @import("../../cuda_rank_wire.zig");
const gfall = @import("cuda_exl3_graph_fallback.zig");

const FakeGraph = struct {
    var live: std.atomic.Value(i32) = .init(0);
    id: u32,

    fn make(id: u32) FakeGraph {
        _ = live.fetchAdd(1, .monotonic);
        return .{ .id = id };
    }

    pub fn deinit(_: *FakeGraph) void {
        _ = live.fetchSub(1, .monotonic);
    }
};

/// One rank's capture attempts: `fail[i]` is whether its i-th attempt fails.
const Attempt = struct {
    fail: []const bool,
    calls: *usize,

    pub fn run(a: Attempt) ?FakeGraph {
        defer a.calls.* += 1;
        return if (a.fail[a.calls.*]) null else FakeGraph.make(@intCast(a.calls.*));
    }
};

/// A rank's side of the wire: `graphAgree` is the exchange `Collective.graphAgree` makes.
const Rank = struct {
    link: wire.Link,
    step: u64 = 0,

    pub fn graphAgree(r: *Rank, signature: u64, vote: collective.GraphVote) !collective.GraphDecision {
        r.step += 1;
        return collective.graphExchange(r.link, r.step, signature, vote, wire.deadline(2000));
    }
};

fn socketPair() ![2]std.posix.socket_t {
    var pair: [2]std.posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    errdefer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    return pair;
}

const rounds = 4;

/// What a rank saw of each round: whether it kept a graph.
const Outcome = struct { kept: [rounds]bool = @splat(false), err: ?anyerror = null };

fn rankMain(link: wire.Link, fail: []const bool, out: *Outcome) void {
    var rank: Rank = .{ .link = link };
    var calls: usize = 0;
    for (0..rounds) |round| {
        const got = gfall.agreedCapture(&rank, 0x5eed + round, 7, Attempt{ .fail = fail, .calls = &calls }) catch |err| {
            out.err = err;
            return;
        };
        out.kept[round] = got != null;
        if (got) |g| {
            var graph = g;
            graph.deinit(); // the round's replay is done with it
        }
    }
}

fn runPair(fail0: []const bool, fail1: []const bool) !struct { Outcome, Outcome } {
    const pair = try socketPair();
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    var peer: Outcome = .{};
    var mine: Outcome = .{};
    const thread = try std.Thread.spawn(.{}, rankMain, .{ wire.Link{ .fd = pair[1] }, fail1, &peer });
    rankMain(.{ .fd = pair[0] }, fail0, &mine);
    thread.join();
    return .{ mine, peer };
}

test "exl3: both ranks keep a capture only when both captured" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    FakeGraph.live.store(0, .monotonic);
    const none = [_]bool{ false, false, false, false };
    const second = [_]bool{ false, true, false, false };
    const cases = [_][2][]const bool{ .{ &none, &none }, .{ &second, &none }, .{ &none, &second }, .{ &second, &second } };
    for (cases) |c| {
        const a, const b = try runPair(c[0], c[1]);
        try std.testing.expectEqual(@as(?anyerror, null), a.err);
        try std.testing.expectEqual(@as(?anyerror, null), b.err);
        // lockstep: both ranks decided every round the same way
        try std.testing.expectEqualSlices(bool, &a.kept, &b.kept);
        // round 2 failed on at least one rank in three of the cases: both ran it eagerly; later rounds are not poisoned
        const failed = c[0][1] or c[1][1];
        try std.testing.expectEqualSlices(bool, &.{ true, !failed, true, true }, &a.kept);
        // no capture is left behind on either rank
        try std.testing.expectEqual(@as(i32, 0), FakeGraph.live.load(.monotonic));
    }
}

test "exl3: a rank that fails every capture leaves its peer eager every round, the communicator untouched" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    FakeGraph.live.store(0, .monotonic);
    const all = [_]bool{ true, true, true, true };
    const none = [_]bool{ false, false, false, false };
    const a, const b = try runPair(&all, &none);
    try std.testing.expectEqual(@as(?anyerror, null), a.err);
    try std.testing.expectEqual(@as(?anyerror, null), b.err);
    try std.testing.expectEqualSlices(bool, &.{ false, false, false, false }, &a.kept);
    try std.testing.expectEqualSlices(bool, &a.kept, &b.kept);
    try std.testing.expectEqual(@as(i32, 0), FakeGraph.live.load(.monotonic));
}

test "exl3: a lost peer during the capture agreement drops the local capture and reports the loss" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    FakeGraph.live.store(0, .monotonic);
    const pair = try socketPair();
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    _ = std.c.shutdown(pair[1], std.posix.SHUT.RDWR);
    var rank: Rank = .{ .link = .{ .fd = pair[0] } };
    var calls: usize = 0;
    const none = [_]bool{false};
    try std.testing.expectError(error.RankDisconnected, gfall.agreedCapture(&rank, 1, 7, Attempt{ .fail = &none, .calls = &calls }));
    try std.testing.expectEqual(@as(i32, 0), FakeGraph.live.load(.monotonic));
}

const cuda = @import("cuda");
const memory = @import("cuda_exl3_graph_memory.zig");
const gcache = @import("cuda_exl3_graph_cache.zig");
const gkey = @import("cuda_exl3_graph_key.zig");
const grun = @import("cuda_exl3_graph_run.zig");
const TestDriver = @import("cuda_exl3_graph_test_driver.zig").GraphTestDriver;
const admission_key: gkey.Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };

const AdmissionPass = struct {
    io: std.Io = std.testing.io,
    stream: cuda.Stream,
    m: struct { collective: ?*collective.Collective },
    prepared: bool = false,
    chunks: usize = 0,

    pub fn chunk(p: *@This(), _: []const u32, _: usize, _: ?[]u8) !void {
        p.chunks += 1;
    }
};

fn noHeadroom() !u64 {
    return 0;
}

fn queryFailure() !u64 {
    return error.CudaFailed;
}

fn admitTest() !void {
    gcache.clear();
    gfall.clear();
    try memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
    gcache.seen[0] = .{ .key = admission_key, .n = 1000, .used = 0 };
}

test "exl3: decode graph headroom refusal runs eagerly and leaves unexpected query errors visible" {
    try admitTest();
    defer {
        gcache.clear();
        gfall.clear();
    }
    memory.setFreeMemoryHook(noHeadroom);
    defer memory.setFreeMemoryHook(null);
    var driver = TestDriver.driver();
    TestDriver.launches = 0;
    var pass: AdmissionPass = .{ .stream = cuda.Stream.borrowed(&driver, null), .m = .{ .collective = null } };
    try grun.run(admission_key, &pass, &.{1}, 0);
    try std.testing.expectEqual(@as(usize, 1), pass.chunks);
    try std.testing.expect(pass.prepared);
    try std.testing.expectEqual(@as(usize, 0), TestDriver.launches);
    try std.testing.expectEqual(@as(u64, 0), memory.charged());
    try grun.run(admission_key, &pass, &.{1}, 0);
    try std.testing.expectEqual(@as(usize, 2), pass.chunks);
    try admitTest();
    memory.setFreeMemoryHook(queryFailure);
    try std.testing.expectError(error.CudaFailed, grun.run(admission_key, &pass, &.{1}, 0));
    try std.testing.expectEqual(@as(usize, 2), pass.chunks);
    try std.testing.expectEqual(@as(u64, 0), memory.charged());
}

test "exl3: exhausted TP graph admission runs eagerly without poisoning the next collective step" {
    if (windows) return error.SkipZigTest; // The rank wire explicitly refuses Windows.
    try admitTest();
    defer {
        gcache.clear();
        gfall.clear();
    }
    memory.setFreeMemoryHook(noHeadroom);
    defer memory.setFreeMemoryHook(null);
    const pair = try socketPair();
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const Peer = struct {
        fn run(link: wire.Link) !void {
            var c = TestDriver.control(link);
            c.timeout_ms = 5000;
            const signature = gkey.graphSignature(admission_key, 0);
            const capture = try c.graphAgree(signature, .{ .state = .capture });
            try std.testing.expectEqual(collective.GraphChoice.capture, capture.choice);
            const refused = try c.graphAgree(signature ^ 0x61646d6974, .{ .state = .ready, .cohort = capture.cohort });
            try std.testing.expectEqual(collective.GraphChoice.eager, refused.choice);
            try std.testing.expectEqual(@as(u64, 0), try c.graphVictim(signature ^ 0x76696374696d, 0));
            const quiet = try c.graphAgree(signature ^ 0x6c6f63616c7175, .{ .state = .ready, .cohort = capture.cohort });
            try std.testing.expectEqual(collective.GraphChoice.replay, quiet.choice);
            try std.testing.expectEqual(@as(u64, 0), try c.graphVictim(signature ^ 0x6c6f63616c646f, 0));
            const next = try c.graphAgree(signature, .{ .state = .eager });
            try std.testing.expectEqual(collective.GraphChoice.eager, next.choice);
        }
        fn main(link: wire.Link, err: *?anyerror) void {
            run(link) catch |e| {
                err.* = e;
            };
        }
    };
    var peer_error: ?anyerror = null;
    const thread = try std.Thread.spawn(.{}, Peer.main, .{ wire.Link{ .fd = pair[1] }, &peer_error });
    var joined = false;
    defer if (!joined) thread.join();
    var control = TestDriver.control(.{ .fd = pair[0] });
    control.timeout_ms = 5000;
    var driver = TestDriver.driver();
    var pass: AdmissionPass = .{ .stream = cuda.Stream.borrowed(&driver, null), .m = .{ .collective = &control } };
    try grun.run(admission_key, &pass, &.{1}, 0);
    try std.testing.expect(!control.failed);
    try grun.run(admission_key, &pass, &.{1}, 0);
    thread.join();
    joined = true;
    try std.testing.expectEqual(@as(?anyerror, null), peer_error);
    try std.testing.expectEqual(@as(usize, 2), pass.chunks);
    try std.testing.expect(!control.failed);
    try std.testing.expectEqual(@as(u64, 0), memory.charged());
}
