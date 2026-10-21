//! One two-rank NCCL owner. Nonblocking initialization and collectives observe the control peer and a deadline.
const std = @import("std");
const cuda = @import("cuda");
const wire = @import("cuda_rank_wire.zig");
const nccl = cuda.nccl;

const Advanced = struct {
    ncclCommInitRankConfig: *const fn (*nccl.Comm, c_int, nccl.UniqueId, c_int, *nccl.Config) callconv(.c) nccl.Result,
    ncclCommAbort: *const fn (nccl.Comm) callconv(.c) nccl.Result,
    ncclCommGetAsyncError: *const fn (nccl.Comm, *nccl.Result) callconv(.c) nccl.Result,
};

/// How long a wait polls before it sleeps between polls (a decode step's work ends within a few milliseconds; each
/// 50 us sleep can add up to ~100 us to the step).
const spin_ns: u64 = 3 * std.time.ns_per_ms;

pub const GraphState = enum(u32) { eager, capture, pending, ready, failed, timed_out };
pub const GraphChoice = enum { eager, capture, replay };
pub const GraphVote = struct { state: GraphState, cohort: u64 = 0 };
pub const GraphDecision = struct { choice: GraphChoice, cohort: u64 };

pub fn selectGraphVictim(local: u64, peer: u64) u64 {
    if (local == 0) return peer;
    if (peer == 0) return local;
    return @min(local, peer);
}

test "exl3: selective retirement agrees stable union victims for asymmetric cohort sets" {
    var left = [_]u64{ 3, 7, 11 };
    var right = [_]u64{ 5, 7, 11 };
    for ([_]u64{ 3, 5 }) |expected| {
        var local: u64 = 0;
        var peer: u64 = 0;
        for (left) |cohort| local = selectGraphVictim(local, cohort);
        for (right) |cohort| peer = selectGraphVictim(peer, cohort);
        const victim = selectGraphVictim(local, peer);
        try std.testing.expectEqual(expected, victim);
        try std.testing.expectEqual(victim, selectGraphVictim(peer, local));
        for (&left) |*cohort| if (cohort.* == victim) {
            cohort.* = 0;
        };
        for (&right) |*cohort| if (cohort.* == victim) {
            cohort.* = 0;
        };
    }
    try std.testing.expectEqual(@as(u64, 7), left[1]);
    try std.testing.expectEqual(@as(u64, 11), right[2]);
    try std.testing.expectEqual(@as(u64, 0), selectGraphVictim(0, 0));
}

fn chooseGraph(local: GraphVote, peer: GraphVote, step: u64) !GraphDecision {
    if (local.state == .timed_out or peer.state == .timed_out) return error.GraphInstantiationTimeout;
    if (local.state == .failed or peer.state == .failed) return error.GraphInstantiationFailed;
    if (local.state == .eager or peer.state == .eager or local.state == .pending or peer.state == .pending)
        return .{ .choice = .eager, .cohort = 0 };
    if (local.state == .ready and peer.state == .ready and local.cohort != 0 and local.cohort == peer.cohort)
        return .{ .choice = .replay, .cohort = local.cohort };
    return .{ .choice = .capture, .cohort = step };
}

fn graphExchange(link: wire.Link, step: u64, signature: u64, vote: GraphVote, until: u64) !GraphDecision {
    const magic: u32 = 0x47520000;
    try link.sendHeader(.{ .kind = .ready, .sequence = step, .object = vote.cohort, .arg = signature, .flags = magic | @backingInt(vote.state) }, until);
    const peer = try link.receiveHeader(until);
    if (peer.kind != .ready or peer.sequence != step or peer.arg != signature or peer.bytes != 0 or peer.flags & 0xffff0000 != magic)
        return error.GraphRendezvousMismatch;
    const state = std.enums.fromInt(GraphState, peer.flags & 0xffff) orelse return error.GraphRendezvousMismatch;
    return chooseGraph(vote, .{ .state = state, .cohort = peer.object }, step);
}

pub const Collective = struct {
    lib: nccl.Library,
    api: Advanced,
    comm: nccl.Comm = null,
    link: wire.Link,
    timeout_ms: u64,
    failed: bool = false,
    graph_step: u64 = 0,

    /// `link` has completed the source/settings handshake; it stays owned by the caller.
    pub fn init(link: wire.Link, rank: usize, timeout_ms: u64) !Collective {
        if (rank > 1 or timeout_ms == 0) return error.BadRank;
        var c: Collective = .{ .lib = try nccl.Library.open(), .api = undefined, .link = link, .timeout_ms = timeout_ms };
        errdefer c.lib.close();
        if (try c.lib.version() < 23000) return error.NcclVersion;
        const info = @typeInfo(Advanced).@"struct";
        inline for (info.field_names, info.field_types) |name, T| @field(c.api, name) = c.lib.lib.lookup(T, name) orelse return error.MissingSymbol;
        var id: nccl.UniqueId = undefined;
        const until = wire.deadline(timeout_ms);
        if (rank == 0) {
            try c.lib.check(c.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(std.mem.asBytes(&id), until);
        } else try link.receive(std.mem.asBytes(&id), until);
        var config: nccl.Config = .{};
        // TF_FLASHNEXT_NCCL_BLOCKING=1: a blocking communicator (NCCL enqueues on the calling thread; a nonblocking
        // one hands each collective to its own thread, ~100 us of host time an eager collective)
        if (std.c.getenv("TF_FLASHNEXT_NCCL_BLOCKING")) |v| if (std.mem.eql(u8, std.mem.span(v), "1")) {
            config.blocking = 1;
        };
        errdefer c.abort();
        try c.submitted(c.api.ncclCommInitRankConfig(&c.comm, 2, id, @intCast(rank), &config));
        try c.wait(null, until);
        return c;
    }

    pub fn abort(c: *Collective) void {
        c.failed = true;
        if (c.comm) |comm| {
            _ = c.api.ncclCommAbort(comm);
            c.comm = null;
        }
    }

    pub fn deinit(c: *Collective) void {
        // Abort is local and also releases an idle communicator if its peer exited first.
        c.abort();
        c.lib.close();
    }

    fn submitted(c: *Collective, result: nccl.Result) !void {
        if (result == 0 or result == 7) return;
        return c.lib.check(result, "NCCL nonblocking operation");
    }

    fn wait(c: *Collective, stream: ?cuda.Stream, until: u64) !void {
        errdefer c.abort();
        var polls: u32 = 0;
        var spin_until: u64 = 0;
        while (true) {
            try c.link.checkPeer();
            if (wire.now() >= until) return error.RankTimeout;
            var state: nccl.Result = 0;
            try c.lib.check(c.api.ncclCommGetAsyncError(c.comm, &state), "ncclCommGetAsyncError");
            if (state != 0 and state != 7) return c.lib.check(state, "NCCL asynchronous operation");
            if (state == 0) {
                if (stream) |s| {
                    if (try s.done()) return;
                } else return;
            }
            // a nonblocking enqueue lands within tens of microseconds: poll that long before sleeping (each sleep
            // of 50 us cost every eager or captured collective ~100 us of host time)
            polls += 1;
            if (polls == 1) spin_until = wire.now() + spin_ns;
            if (wire.now() < spin_until) std.atomic.spinLoopHint() else wire.pause();
        }
    }

    /// Wait at a host-visible boundary while retaining peer, asynchronous-error and timeout checks.
    pub fn finish(c: *Collective, stream: cuda.Stream) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        try c.wait(stream, wire.deadline(c.timeout_ms));
    }

    pub fn failGraph(c: *Collective) void {
        c.abort();
        _ = std.c.shutdown(c.link.fd, std.posix.SHUT.RDWR);
    }

    pub fn graphTimeoutMs(c: *const Collective) u64 {
        return @min(c.timeout_ms, 30_000);
    }

    pub fn graphAgree(c: *Collective, signature: u64, vote: GraphVote) !GraphDecision {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        errdefer c.failGraph();
        c.graph_step = try std.math.add(u64, c.graph_step, 1);
        return graphExchange(c.link, c.graph_step, signature, vote, wire.deadline(c.graphTimeoutMs()));
    }

    pub fn graphVictim(c: *Collective, signature: u64, candidate: u64) !u64 {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        errdefer c.failGraph();
        c.graph_step = try std.math.add(u64, c.graph_step, 1);
        const until = wire.deadline(c.graphTimeoutMs());
        try c.link.sendHeader(.{ .kind = .ready, .sequence = c.graph_step, .object = candidate, .arg = signature, .flags = 0x47560000 }, until);
        const peer = try c.link.receiveHeader(until);
        if (peer.kind != .ready or peer.sequence != c.graph_step or peer.arg != signature or peer.bytes != 0 or peer.flags != 0x47560000)
            return error.GraphRendezvousMismatch;
        return selectGraphVictim(candidate, peer.object);
    }

    /// Disjoint BF16 vocabulary columns are concatenated without an arithmetic reduction.
    pub fn gather(c: *Collective, stream: cuda.Stream, local: u64, out: u64, count: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllGather(local, out, count, .bf16, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }

    pub fn broadcast(c: *Collective, stream: cuda.Stream, values: u64, bytes: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclBroadcast(values, values, bytes, .u8, 0, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }

    /// FP32 sum is enqueued on the compute stream; later kernels on that stream observe its result.
    pub fn sum(c: *Collective, stream: cuda.Stream, values: u64, count: usize) !void {
        if (c.failed or c.comm == null) return error.RankDisconnected;
        if (count == 0) return;
        errdefer c.abort();
        try c.link.checkPeer();
        try c.submitted(c.lib.api.ncclAllReduce(values, values, count, .f32, .sum, c.comm, stream.handle));
        try c.wait(null, wire.deadline(c.timeout_ms));
    }
};

test "exl3: graph readiness requires matching completed capture cohorts" {
    const states = std.enums.values(GraphState);
    for (states) |local| for (states) |peer| {
        const left: GraphVote = .{ .state = local, .cohort = 7 };
        const right: GraphVote = .{ .state = peer, .cohort = 7 };
        if (local == .timed_out or peer == .timed_out) {
            try std.testing.expectError(error.GraphInstantiationTimeout, chooseGraph(left, right, 10));
        } else if (local == .failed or peer == .failed) {
            try std.testing.expectError(error.GraphInstantiationFailed, chooseGraph(left, right, 10));
        } else {
            const choice: GraphChoice = if (local == .eager or peer == .eager or local == .pending or peer == .pending)
                .eager
            else if (local == .ready and peer == .ready) .replay else .capture;
            const decision = try chooseGraph(left, right, 10);
            try std.testing.expectEqual(choice, decision.choice);
            try std.testing.expectEqualDeep(decision, try chooseGraph(right, left, 10));
        }
    };
    try std.testing.expectEqual(GraphChoice.capture, (try chooseGraph(.{ .state = .ready, .cohort = 7 }, .{ .state = .ready, .cohort = 8 }, 10)).choice);
}

fn graphTestPair() ![2]std.posix.socket_t {
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

test "exl3: graph rendezvous keeps a delayed rank eager until both workers finish" {
    const pair = try graphTestPair();
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const Peer = struct {
        fn run(link: wire.Link) void {
            const votes = [_]GraphVote{ .{ .state = .pending, .cohort = 3 }, .{ .state = .ready, .cohort = 3 }, .{ .state = .ready, .cohort = 3 } };
            for (votes, 1..) |vote, step| {
                const decision = graphExchange(link, step, 42, vote, wire.deadline(1000)) catch @panic("peer rendezvous failed");
                std.debug.assert(decision.choice == (if (step < 3) GraphChoice.eager else GraphChoice.replay));
            }
        }
    };
    const thread = try std.Thread.spawn(.{}, Peer.run, .{wire.Link{ .fd = pair[1] }});
    defer thread.join();
    const votes = [_]GraphVote{ .{ .state = .ready, .cohort = 3 }, .{ .state = .pending, .cohort = 3 }, .{ .state = .ready, .cohort = 3 } };
    for (votes, 1..) |vote, step| {
        const decision = try graphExchange(.{ .fd = pair[0] }, step, 42, vote, wire.deadline(1000));
        try std.testing.expectEqual(if (step < 3) GraphChoice.eager else GraphChoice.replay, decision.choice);
    }
}

test "exl3: graph rendezvous rejects stale frames failures timeouts and disconnected peers" {
    const pair = try graphTestPair();
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const local: wire.Link = .{ .fd = pair[0] };
    const peer: wire.Link = .{ .fd = pair[1] };
    const vote: GraphVote = .{ .state = .ready, .cohort = 2 };
    for ([_]wire.Header{
        .{ .kind = .ack, .sequence = 1, .arg = 42, .flags = 0x47520003 },
        .{ .kind = .ready, .sequence = 2, .arg = 42, .flags = 0x47520003 },
        .{ .kind = .ready, .sequence = 1, .arg = 43, .flags = 0x47520003 },
        .{ .kind = .ready, .sequence = 1, .arg = 42, .flags = 0x475200ff },
    }) |header| {
        try peer.sendHeader(header, wire.deadline(1000));
        try std.testing.expectError(error.GraphRendezvousMismatch, graphExchange(local, 1, 42, vote, wire.deadline(1000)));
        _ = try peer.receiveHeader(wire.deadline(1000));
    }
    try peer.sendHeader(.{ .kind = .ready, .sequence = 1, .arg = 42, .flags = 0x47520004 }, wire.deadline(1000));
    try std.testing.expectError(error.GraphInstantiationFailed, graphExchange(local, 1, 42, vote, wire.deadline(1000)));
    _ = try peer.receiveHeader(wire.deadline(1000));
    try std.testing.expectError(error.RankTimeout, graphExchange(local, 1, 42, vote, wire.deadline(5)));
    _ = std.c.shutdown(peer.fd, std.posix.SHUT.RDWR);
    try std.testing.expectError(error.RankDisconnected, graphExchange(local, 1, 42, vote, wire.deadline(1000)));
}

test "exl3: graph rendezvous failure aborts the communicator and disconnects its control peer" {
    const pair = try graphTestPair();
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const Abort = struct {
        var calls: usize = 0;
        fn run(_: nccl.Comm) callconv(.c) nccl.Result {
            calls += 1;
            return 0;
        }
    };
    Abort.calls = 0;
    var c: Collective = undefined;
    c.comm = @ptrFromInt(1);
    c.api.ncclCommAbort = Abort.run;
    c.link = .{ .fd = pair[0] };
    c.failed = false;
    c.timeout_ms = 5;
    c.graph_step = 0;
    try std.testing.expectError(error.RankTimeout, c.graphAgree(42, .{ .state = .pending }));
    try std.testing.expect(c.failed and c.comm == null);
    try std.testing.expectEqual(@as(usize, 1), Abort.calls);
    try std.testing.expectError(error.RankDisconnected, c.graphAgree(42, .{ .state = .ready }));
    const peer: wire.Link = .{ .fd = pair[1] };
    _ = try peer.receiveHeader(wire.deadline(1000));
    try std.testing.expectError(error.RankDisconnected, peer.checkPeer());
}
