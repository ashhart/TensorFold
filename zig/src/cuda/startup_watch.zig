//! Startup progress is observed independently of NCCL and CUDA; it never controls the work it names.
const std = @import("std");
const ncclog = @import("p2p_nccl_log.zig");

pub const Phase = enum(u3) {
    nccl_init,
    peer_warmup,
    weights,
    calibration,
    ready,
    following,
    failed,
    pub fn name(p: Phase) []const u8 {
        return switch (p) {
            .nccl_init => "NCCL init",
            .peer_warmup => "peer warm-up",
            .weights => "weights",
            .calibration => "calibration",
            .ready => "ready",
            .following => "ready to follow rank 0",
            .failed => "stopped before ready",
        };
    }
    fn peer(p: Phase) bool {
        return p == .nccl_init or p == .peer_warmup;
    }
};
pub const Event = struct { world: usize = 1, rank: usize, phase: Phase, elapsed_s: u64 = 0, transport: u3 = 0 };
pub const Sink = *const fn (bool, Event) void;
fn log(waiting: bool, e: Event) void {
    if (!waiting) {
        const entering = if (e.phase == .following or e.phase == .failed) "" else "entering ";
        std.log.info("rank {d} startup: {s}{s}", .{ e.rank, entering, e.phase.name() });
        return;
    }
    const kinds = [_][]const u8{ "unknown", "P2P", "SHM", "P2P+SHM", "NET", "P2P+NET", "SHM+NET", "P2P+SHM+NET" };
    std.log.warn("rank {d} startup: still in {s} after {d} s; transport {s}; {s}waiting without changing the transfer", .{ e.rank, e.phase.name(), e.elapsed_s, kinds[e.transport], advice(e) });
}

/// NCCL's direct P2P setting is relevant only to an observed P2P link on a two-rank start.
pub fn advice(e: Event) []const u8 {
    return if (e.world == 2 and e.transport & 1 != 0) "NCCL_P2P_DISABLE=1 is one thing to try; " else "";
}

/// Only the watchdog writes its cadence fields; phase publication is one coherent atomic timestamp and tag.
pub const Monitor = struct {
    rank: usize,
    world: usize,
    state: std.atomic.Value(u64) = .init(@backingInt(Phase.ready)),
    done: std.atomic.Value(bool) = .init(false),
    capture: ncclog.Capture = .{},
    sink: Sink = log,
    seen: ?u64 = null,
    last_ms: u64 = 0,

    pub fn enter(m: *Monitor, phase: Phase, now_ms: u64) void {
        if (m.world < 2) return;
        m.state.store((now_ms << 3) | @backingInt(phase), .release);
        m.sink(false, .{ .world = m.world, .rank = m.rank, .phase = phase });
    }
    pub fn poll(m: *Monitor, now_ms: u64) void {
        if (m.world < 2) return;
        const state = m.state.load(.acquire);
        const phase: Phase = @fromBackingInt(@intCast(@as(u3, @truncate(state))));
        const since = state >> 3;
        if (m.seen == null or m.seen.? != state) {
            m.seen = state;
            m.last_ms = since;
        }
        if (!phase.peer() or now_ms < since or now_ms - since < 60_000 or now_ms - m.last_ms < 60_000) return;
        var buf: [64 << 10]u8 = undefined;
        const how = transport(m.capture.read(&buf), @intCast(m.rank));
        if (m.state.load(.acquire) != state or m.done.load(.acquire)) return;
        m.last_ms = now_ms;
        m.sink(true, .{ .world = m.world, .rank = m.rank, .phase = phase, .elapsed_s = (now_ms - since) / 1000, .transport = how });
    }
};

/// This loop has no phase-held mutex, driver handle or communicator; services supply only a clock and sleep.
pub fn drive(m: *Monitor, clock: anytype) void {
    while (!m.done.load(.acquire)) {
        m.poll(clock.now());
        clock.pause();
    }
}

pub fn transport(text: []const u8, rank: u32) u3 {
    var bits: u3 = 0;
    const summary = ncclog.parse(text, rank);
    for (summary.list()) |l| bits |= switch (l.kind) {
        .p2p => 1,
        .shm => 2,
        .net => 4,
    };
    return bits;
}

const Clock = struct {
    io: std.Io,
    pub fn now(c: Clock) u64 {
        return @intCast(@divTrunc(std.Io.Clock.awake.now(c.io).toNanoseconds(), std.time.ns_per_ms));
    }
    pub fn pause(c: Clock) void {
        std.Io.sleep(c.io, .fromMilliseconds(100), .awake) catch {};
    }
};

/// The owner keeps this value at a stable address until end joins the observer, including failed starts.
pub const Watcher = struct {
    monitor: Monitor,
    io: std.Io,
    thread: ?std.Thread = null,
    pub fn init(io: std.Io, rank: usize, world: usize, capture: ncclog.Capture) Watcher {
        return .{ .io = io, .monitor = .{ .rank = rank, .world = world, .capture = capture } };
    }
    pub fn start(w: *Watcher) void {
        if (w.monitor.world < 2) return;
        w.thread = std.Thread.spawn(.{}, run, .{w}) catch {
            std.log.warn("rank {d} startup: progress watcher unavailable; phase entries remain enabled", .{w.monitor.rank});
            return;
        };
    }
    fn run(w: *Watcher) void {
        drive(&w.monitor, Clock{ .io = w.io });
    }
    pub fn enter(w: *Watcher, phase: Phase) void {
        if (w.monitor.world < 2) return;
        w.monitor.enter(phase, (Clock{ .io = w.io }).now());
    }
    pub fn end(w: *Watcher) void {
        w.monitor.done.store(true, .release);
        if (w.thread) |t| t.join();
        w.thread = null;
    }
};

test {
    _ = @import("startup_watch_test.zig");
}
