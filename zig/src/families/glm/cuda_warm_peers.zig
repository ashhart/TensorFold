//! Connect GLM's explicit prompt-exchange peers before weights consume the registration budget.
const std = @import("std");
const cuda = @import("cuda");
const deadline = @import("cuda_warm_deadline.zig");

pub const bytes = 4 << 20;

/// What the exchange uses: every peer's send/receive (split), the neighbours' (ring), or all-gathers alone (gather).
pub const Peers = enum {
    every,
    neighbours,
    gather,

    pub fn of(split: bool, ring: bool) Peers {
        return if (!split) .gather else if (ring) .neighbours else .every;
    }
};

pub fn warm(io: std.Io, d: *const cuda.Driver, g: *cuda.tp_group.Group, peers: Peers, diag: *const cuda.p2p_report.Session) !void {
    return warmWith(d, g, peers, Clock{ .io = io }, deadline.timeout_ms, Action{ .g = g, .peers = peers, .diag = diag });
}

/// Production uses 300 s; host fixtures shorten only this injected clock/deadline and termination action.
pub fn warmWith(d: *const cuda.Driver, g: *cuda.tp_group.Group, peers: Peers, clock: anytype, limit_ms: u64, action: anytype) !void {
    if (g.world <= 1) return;
    try deadline.run(clock, limit_ms, action, Work{ .d = d, .g = g, .peers = peers }, Work.run);
}

const Work = struct {
    d: *const cuda.Driver,
    g: *cuda.tp_group.Group,
    peers: Peers,
    fn run(a: Work) !void {
        const d = a.d;
        const g = a.g;
        const peers = a.peers;
        var buf = try cuda.DeviceBuffer.alloc(d, 2 * bytes * g.world);
        defer buf.free();
        var stream = try cuda.Stream.init(d, false);
        defer stream.deinit();
        try post(g, stream, buf.ptr, peers);
        try stream.synchronize();
    }
};

const Clock = struct {
    io: std.Io,
    pub fn now(c: Clock) u64 {
        return @intCast(@divTrunc(std.Io.Clock.awake.now(c.io).toNanoseconds(), std.time.ns_per_ms));
    }
    pub fn pause(c: Clock) void {
        std.Io.sleep(c.io, .fromMilliseconds(10), .awake) catch {};
    }
};

const Action = struct {
    g: *cuda.tp_group.Group,
    peers: Peers,
    diag: *const cuda.p2p_report.Session,
    pub fn expired(a: Action) void {
        timeoutWith(a.g, a.peers, a.diag, cuda.tp_watch, log);
    }
};

fn log(text: []const u8) void {
    std.log.err("{s}", .{text});
}

/// Publish the named cause before abort can release a blocked NCCL call with a less specific error.
pub fn timeoutWith(g: *cuda.tp_group.Group, peers: Peers, diag: *const cuda.p2p_report.Session, watch: anytype, sink: cuda.startup_log.Sink) void {
    var raw: [64 << 10]u8 = undefined;
    const how = cuda.startup_watch.transport(diag.capture.read(&raw), @intCast(g.rank));
    var buf: [2048]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    timeoutLine(&w, g.rank, g.world, peers, how) catch {};
    sink(w.buffered());
    watch.fail("PeerWarmupTimedOut");
    g.abort();
    watch.stop();
}

pub fn timeoutLine(w: *std.Io.Writer, rank: usize, world: usize, peers: Peers, how: u3) !void {
    if (peers == .gather) {
        try w.print("rank {d}: peer warm-up: 4 MiB all-gather", .{rank});
    } else {
        try w.print("rank {d}: peer warm-up: 4 MiB send/recv with rank(s) ", .{rank});
        const plan: cuda.relay_plan.Relay = .{ .rank = rank, .world = world, .n = 0 };
        var first = true;
        for (0..world) |peer| {
            if (peer == rank or (peers == .neighbours and peer != plan.next() and peer != plan.prev())) continue;
            try w.print("{s}{d}", .{ if (first) "" else ",", peer });
            first = false;
        }
    }
    try w.print(" not finished after {d} s (NCCL transport {s}); PeerWarmupTimedOut; stopping every rank", .{ deadline.timeout_ms / 1000, cuda.startup_watch.transportName(how) });
}

/// The 4 MiB send/receive, to neighbours only for ring; gather connects its all-gather and posts no send/receive.
pub fn post(g: *const cuda.tp_group.Group, stream: cuda.Stream, buffer: u64, peers: Peers) !void {
    if (g.world == 0 or g.rank >= g.world) return error.BadWorld;
    if (g.world == 1) return;
    // NCCL picks a receive's port by channel, not by peer: on a switchless ring some face a rank without the peer
    if (peers == .gather) return g.lib.check(g.lib.api.ncclAllGather(buffer + g.rank * bytes, buffer, bytes, .u8, g.comm, stream.handle), "ncclAllGather");
    const plan: cuda.relay_plan.Relay = .{ .rank = g.rank, .world = g.world, .n = 0 };
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = posting: {
        for (0..g.world) |peer| {
            if (peer == g.rank or (peers == .neighbours and peer != plan.next() and peer != plan.prev())) continue;
            g.send(stream, buffer + peer * bytes, bytes, peer) catch |err| break :posting err;
            g.recv(stream, buffer + (g.world + peer) * bytes, bytes, peer) catch |err| break :posting err;
        }
        break :posting {};
    };
    const closed = g.lib.api.ncclGroupEnd();
    // split's warm-up is its first exchange with every rank; from four ranks a ring has some that are not neighbours
    const refuse = peers == .every and g.world >= 4;
    if (closed != 0 and refuse) refuseSplit(g.rank); // before check, which stops every rank and hands the others its name
    const ended = g.lib.check(closed, if (refuse) "REFUSE TF_GLM_PREFILL_REDUCE=split at ncclGroupEnd" else "ncclGroupEnd");
    try posted;
    try ended;
}

/// The latest rank that refused split, which a test build keeps instead of printing the refusal.
pub var refused_rank: ?usize = null;

fn refuseSplit(rank: usize) void {
    if (@import("builtin").is_test) {
        refused_rank = rank;
        return;
    }
    std.debug.print("REFUSE TF_GLM_PREFILL_REDUCE=split: rank {d} could not finish its start-up exchange with every other rank; split sends every rank's rows to every other, and a fabric where some ranks reach only their neighbours (a ring of four without a switch) needs TF_GLM_PREFILL_REDUCE=gather\n", .{rank});
}
