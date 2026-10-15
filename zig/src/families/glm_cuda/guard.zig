//! A rank's watchdog: exits at the MemAvailable floor (TF_MEM_FLOOR_GIB, 2 GiB) or when its peer is gone; ``tight``.

const std = @import("std");
const posix = std.posix;
const cuda = @import("cuda");

const gib: f64 = 1 << 30;
const POLLRDHUP: i16 = 0x2000;

pub const Guard = struct {
    floor: u64,
    tight_at: u64,
    fd: posix.socket_t,
    nccl: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    running: std.atomic.Value(bool) = .init(true),
    tight: std.atomic.Value(bool) = .init(false),
    available: std.atomic.Value(u64) = .init(0),
    lowest: std.atomic.Value(u64) = .init(std.math.maxInt(u64)),
    thread: ?std.Thread = null,

    /// Watches this process's memory, the link's socket ``fd`` and ``comm`` (kept alive by TCP keepalives).
    pub fn start(gpa: std.mem.Allocator, fd: posix.socket_t, nccl: *const cuda.nccl.Library, comm: cuda.nccl.Comm) !*Guard {
        const floor = floorBytes();
        const g = try gpa.create(Guard);
        g.* = .{ .floor = floor, .tight_at = floor + (3 << 30), .fd = fd, .nccl = nccl, .comm = comm };
        keepAlive(fd);
        g.thread = try std.Thread.spawn(.{}, loop, .{g});
        std.log.info("memory guard: exit below {d:.1} GiB available, no new requests below {d:.1} GiB", .{ toGib(g.floor), toGib(g.tight_at) });
        return g;
    }

    /// Stopped before the link closes (a closing link is no lost peer).
    pub fn stop(g: *Guard, gpa: std.mem.Allocator) void {
        g.running.store(false, .release);
        if (g.thread) |t| t.join();
        gpa.destroy(g);
    }

    pub fn isTight(g: *const Guard) bool {
        return g.tight.load(.acquire);
    }
};

fn loop(g: *Guard) void {
    const tick: std.c.timespec = .{ .sec = 0, .nsec = 100_000_000 };
    while (g.running.load(.acquire)) {
        _ = std.c.nanosleep(&tick, null);
        if (available()) |a| {
            g.available.store(a, .release);
            _ = g.lowest.fetchMin(a, .monotonic);
            if (a < g.floor) die(75, "memory guard: {d:.2} GiB available, under the {d:.1} GiB floor: exiting to free this rank's memory before the machine stalls", .{ toGib(a), toGib(g.floor) });
            const was = g.tight.load(.monotonic);
            const now = a < g.tight_at;
            if (now != was) {
                g.tight.store(now, .release);
                if (now) std.log.warn("memory guard: {d:.2} GiB available: no new requests until it is back over {d:.1} GiB", .{ toGib(a), toGib(g.tight_at) }) else std.log.info("memory guard: {d:.2} GiB available again: requests admitted", .{toGib(a)});
            }
        }
        if (peerGone(g.fd)) {
            // a stop frame may be in flight before the peer's close: give this rank's own shutdown time to stop us
            const grace: std.c.timespec = .{ .sec = 3, .nsec = 0 };
            _ = std.c.nanosleep(&grace, null);
            if (!g.running.load(.acquire)) return;
            die(76, "memory guard: the peer rank is gone (control link closed): exiting rather than waiting on it", .{});
        }
        var r: cuda.nccl.Result = 0;
        if (g.nccl.api.ncclCommGetAsyncError(g.comm, &r) == 0 and r != 0 and r != 7) // 7: ncclInProgress
            die(77, "memory guard: NCCL reports error {d} ({s}): the peer rank is gone; exiting", .{ r, std.mem.span(g.nccl.api.ncclGetErrorString(r) orelse "?") });
    }
}

fn die(code: u8, comptime fmt: []const u8, args: anytype) noreturn {
    std.log.err(fmt, args);
    std.c._exit(code);
}

fn peerGone(fd: posix.socket_t) bool {
    var fds = [_]posix.pollfd{.{ .fd = fd, .events = POLLRDHUP, .revents = 0 }};
    const n = posix.poll(&fds, 0) catch return false;
    return n > 0 and fds[0].revents & (POLLRDHUP | posix.POLL.HUP | posix.POLL.ERR) != 0;
}

/// A peer that stops answering (its machine stalled, no FIN sent) errors the socket within about 11 s.
fn keepAlive(fd: posix.socket_t) void {
    const on: c_int = 1;
    const idle: c_int = 5;
    const intvl: c_int = 2;
    const cnt: c_int = 3;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, std.mem.asBytes(&on)) catch {};
    posix.setsockopt(fd, posix.IPPROTO.TCP, 4, std.mem.asBytes(&idle)) catch {}; // TCP_KEEPIDLE
    posix.setsockopt(fd, posix.IPPROTO.TCP, 5, std.mem.asBytes(&intvl)) catch {}; // TCP_KEEPINTVL
    posix.setsockopt(fd, posix.IPPROTO.TCP, 6, std.mem.asBytes(&cnt)) catch {}; // TCP_KEEPCNT
}

/// MemAvailable in bytes (null: unreadable).
pub fn available() ?u64 {
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var buf: [4096]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return null;
    return parseAvailable(buf[0..@intCast(n)]);
}

fn parseAvailable(text: []const u8) ?u64 {
    var rows = std.mem.tokenizeScalar(u8, text, '\n');
    while (rows.next()) |row| {
        if (!std.mem.startsWith(u8, row, "MemAvailable:")) continue;
        var words = std.mem.tokenizeAny(u8, row["MemAvailable:".len..], " \t");
        const kib = std.fmt.parseInt(u64, words.next() orelse return null, 10) catch return null;
        return kib * 1024;
    }
    return null;
}

fn floorBytes() u64 {
    const v = std.c.getenv("TF_MEM_FLOOR_GIB") orelse return 2 << 30;
    const g = std.fmt.parseFloat(f64, std.mem.span(v)) catch return 2 << 30;
    return @intFromFloat(@max(0.5, g) * gib);
}

fn toGib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / gib;
}

test "MemAvailable parsed in bytes" {
    try std.testing.expectEqual(@as(?u64, 3 * 1024), parseAvailable("MemTotal: 10 kB\nMemFree: 1 kB\nMemAvailable:       3 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), parseAvailable("MemTotal: 10 kB\n"));
}
