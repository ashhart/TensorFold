//! Connect GLM's explicit prompt-exchange peers before weights consume the registration budget.
const std = @import("std");
const cuda = @import("cuda");
const deadline = @import("cuda_warm_deadline.zig");

pub const bytes = 4 << 20;

pub fn warm(io: std.Io, d: *const cuda.Driver, g: *cuda.tp_group.Group, ring: bool, diag: *const cuda.p2p_report.Session) !void {
    return warmWith(d, g, ring, Clock{ .io = io }, deadline.timeout_ms, Action{ .g = g, .ring = ring, .diag = diag });
}

/// Production uses 300 s; host fixtures shorten only this injected clock/deadline and termination action.
pub fn warmWith(d: *const cuda.Driver, g: *cuda.tp_group.Group, ring: bool, clock: anytype, limit_ms: u64, action: anytype) !void {
    if (g.world <= 1) return;
    try deadline.run(clock, limit_ms, action, Work{ .d = d, .g = g, .ring = ring }, Work.run);
}

const Work = struct {
    d: *const cuda.Driver,
    g: *cuda.tp_group.Group,
    ring: bool,
    fn run(a: Work) !void {
        const d = a.d;
        const g = a.g;
        const ring = a.ring;
        var buf = try cuda.DeviceBuffer.alloc(d, 2 * bytes * g.world);
        defer buf.free();
        var stream = try cuda.Stream.init(d, false);
        defer stream.deinit();
        try post(g, stream, buf.ptr, ring);
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
    ring: bool,
    diag: *const cuda.p2p_report.Session,
    pub fn expired(a: Action) void {
        timeoutWith(a.g, a.ring, a.diag, cuda.tp_watch, log);
    }
};

fn log(text: []const u8) void {
    std.log.err("{s}", .{text});
}

/// Publish the named cause before abort can release a blocked NCCL call with a less specific error.
pub fn timeoutWith(g: *cuda.tp_group.Group, ring: bool, diag: *const cuda.p2p_report.Session, watch: anytype, sink: cuda.startup_log.Sink) void {
    var raw: [64 << 10]u8 = undefined;
    const how = cuda.startup_watch.transport(diag.capture.read(&raw), @intCast(g.rank));
    var buf: [2048]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    timeoutLine(&w, g.rank, g.world, ring, how) catch {};
    sink(w.buffered());
    watch.fail("PeerWarmupTimedOut");
    g.abort();
    watch.stop();
}

pub fn timeoutLine(w: *std.Io.Writer, rank: usize, world: usize, ring: bool, how: u3) !void {
    try w.print("rank {d}: peer warm-up: 4 MiB send/recv with rank(s) ", .{rank});
    const plan: cuda.relay_plan.Relay = .{ .rank = rank, .world = world, .n = 0 };
    var first = true;
    for (0..world) |peer| {
        if (peer == rank or (ring and peer != plan.next() and peer != plan.prev())) continue;
        try w.print("{s}{d}", .{ if (first) "" else ",", peer });
        first = false;
    }
    try w.print(" not finished after {d} s (NCCL transport {s}); PeerWarmupTimedOut; stopping every rank", .{ deadline.timeout_ms / 1000, cuda.startup_watch.transportName(how) });
}

/// The existing 4 MiB send/receive, restricted to neighbours only when ring exchange was selected.
pub fn post(g: *const cuda.tp_group.Group, stream: cuda.Stream, buffer: u64, ring: bool) !void {
    if (g.world == 0 or g.rank >= g.world) return error.BadWorld;
    if (g.world == 1) return;
    const plan: cuda.relay_plan.Relay = .{ .rank = g.rank, .world = g.world, .n = 0 };
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = posting: {
        for (0..g.world) |peer| {
            if (peer == g.rank or (ring and peer != plan.next() and peer != plan.prev())) continue;
            g.send(stream, buffer + peer * bytes, bytes, peer) catch |err| break :posting err;
            g.recv(stream, buffer + (g.world + peer) * bytes, bytes, peer) catch |err| break :posting err;
        }
        break :posting {};
    };
    const ended = g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    try posted;
    try ended;
}
