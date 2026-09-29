const std = @import("std");
const Cancellation = @import("cancellation.zig").Cancellation;

fn now(io: std.Io) i64 {
    return std.Io.Clock.awake.now(io).toMilliseconds();
}

var interrupted: std.atomic.Value(bool) = .init(false);
pub const Signals = struct {
    old_int: std.posix.Sigaction,
    old_term: std.posix.Sigaction,

    pub fn install() Signals {
        interrupted.store(false, .release);
        var result: Signals = undefined;
        const action = std.posix.Sigaction{ .handler = .{ .handler = onSignal }, .mask = std.posix.sigemptyset(), .flags = 0 };
        std.posix.sigaction(.INT, &action, &result.old_int);
        std.posix.sigaction(.TERM, &action, &result.old_term);
        return result;
    }
    pub fn deinit(s: *Signals) void {
        std.posix.sigaction(.INT, &s.old_int, null);
        std.posix.sigaction(.TERM, &s.old_term, null);
    }
    fn onSignal(_: std.posix.SIG) callconv(.c) void {
        interrupted.store(true, .release);
    }
};

const Reason = enum(u8) { none, timeout, shutdown, disconnected };

pub const Client = struct {
    owner: *Registry = undefined,
    socket: ?std.posix.fd_t = null,
    deadline: ?i64 = null,
    reason: std.atomic.Value(Reason) = .init(.none),
    interrupted_socket: bool = false,

    pub fn cancellation(c: *Client) Cancellation {
        return .{ .context = c, .callback = check };
    }
    fn check(context: ?*anyopaque) !void {
        const c: *Client = @ptrCast(@alignCast(context.?));
        try c.checkAt(now(c.owner.io));
        const fd = c.socket orelse return error.RequestCancelled;
        var byte: [1]u8 = undefined;
        const result = std.c.recv(fd, &byte, 1, std.posix.MSG.PEEK | std.posix.MSG.DONTWAIT);
        if (result > 0) return;
        if (result < 0) switch (std.posix.errno(result)) {
            .AGAIN, .INTR => return,
            else => {},
        };
        c.mark(.disconnected);
        return error.RequestCancelled;
    }
    fn mark(c: *Client, reason: Reason) void {
        _ = c.reason.cmpxchgStrong(.none, reason, .acq_rel, .acquire);
    }
    fn checkAt(c: *Client, instant: i64) !void {
        if (c.owner.stopping.load(.acquire)) c.mark(.shutdown);
        if (c.deadline) |deadline| if (instant >= deadline) c.mark(.timeout);
        return switch (c.reason.load(.acquire)) {
            .none => {},
            .timeout => error.RequestTimedOut,
            .shutdown => error.ServerStopping,
            .disconnected => error.RequestCancelled,
        };
    }
};

pub const Registry = struct {
    io: std.Io,
    timeout_ms: i64 = 0,
    shutdown_grace_ms: i64 = 5000,
    stopping: std.atomic.Value(bool) = .init(false),
    finished: std.atomic.Value(bool) = .init(false),
    mutex: std.Io.Mutex = .init,
    clients: [32]Client = @splat(.{}),
    stop_time: ?i64 = null,

    pub fn acquire(r: *Registry, socket: std.posix.fd_t) ?*Client {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        if (r.stopping.load(.acquire)) return null;
        for (&r.clients) |*client| if (client.socket == null) {
            client.* = .{ .owner = r, .socket = socket, .deadline = if (r.timeout_ms == 0) null else now(r.io) + r.timeout_ms };
            return client;
        };
        return null;
    }
    // Release before closing the descriptor, so the watchdog cannot touch a reused fd.
    pub fn release(r: *Registry, client: *Client) void {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        client.socket = null;
    }
    pub fn stop(r: *Registry) void {
        r.stopping.store(true, .release);
    }
    pub fn waitStopped(r: *Registry) anyerror!void {
        while (!r.stopping.load(.acquire)) try std.Io.sleep(r.io, .fromMilliseconds(10), .awake);
    }
    pub fn watch(r: *Registry) void {
        while (!r.finished.load(.acquire)) {
            if (interrupted.load(.acquire)) r.stop();
            r.enforce(now(r.io));
            std.Io.sleep(r.io, .fromMilliseconds(10), .awake) catch return;
        }
    }
    fn enforce(r: *Registry, instant: i64) void {
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        if (r.stopping.load(.acquire) and r.stop_time == null) r.stop_time = instant;
        for (&r.clients) |*client| if (client.socket) |fd| {
            client.checkAt(instant) catch {};
            const interrupt_at = switch (client.reason.load(.acquire)) {
                .none => continue,
                .timeout => client.deadline.? + 250,
                .shutdown => r.stop_time.? + r.shutdown_grace_ms,
                .disconnected => instant,
            };
            if (instant >= interrupt_at and !client.interrupted_socket) {
                _ = std.c.shutdown(fd, 2);
                client.interrupted_socket = true;
            }
        };
    }
};

pub fn seconds(text: []const u8) !i64 {
    const value = std.fmt.parseFloat(f64, text) catch return error.InvalidTimeout;
    if (!std.math.isFinite(value) or value < 0 or value > 86400) return error.InvalidTimeout;
    return @intFromFloat(@ceil(value * 1000));
}

test "request deadline is absolute and cancellation remains sticky" {
    var registry = Registry{ .io = std.testing.io };
    var client = Client{ .owner = &registry, .deadline = 100 };
    try client.checkAt(99);
    try std.testing.expectError(error.RequestTimedOut, client.checkAt(100));
    registry.stop();
    try std.testing.expectError(error.RequestTimedOut, client.checkAt(101));
    var other = Client{ .owner = &registry };
    try std.testing.expectError(error.ServerStopping, other.checkAt(0));
}

test "connection slots survive cancellation until worker acknowledgement" {
    var registry = Registry{ .io = std.testing.io };
    var held: [32]*Client = undefined;
    for (&held, 0..) |*slot, index| slot.* = registry.acquire(@intCast(index + 100)).?;
    held[0].mark(.timeout);
    try std.testing.expectEqual(null, registry.acquire(999));
    registry.release(held[0]);
    const recycled = registry.acquire(999).?;
    try std.testing.expectEqual(Reason.none, recycled.reason.load(.acquire));
    registry.stop();
    registry.release(recycled);
    try std.testing.expectEqual(null, registry.acquire(999));
}

test "timeout flags accept finite nonnegative seconds and preserve subsecond limits" {
    try std.testing.expectEqual(@as(i64, 0), try seconds("0"));
    try std.testing.expectEqual(@as(i64, 250), try seconds("0.25"));
    try std.testing.expectEqual(@as(i64, 1), try seconds("0.0001"));
    for ([_][]const u8{ "-1", "nan", "inf", "86401", "invalid" }) |value| try std.testing.expectError(error.InvalidTimeout, seconds(value));
}
