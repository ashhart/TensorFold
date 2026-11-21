//! Opening the ranks' liveness lines: rank 0 takes one line from each other rank, and each line proves the group's key.
const std = @import("std");
const platform = @import("sys");
const posix = platform.posix;
const auth = platform.rank_auth;

pub const max_lines = 16;

/// Milliseconds rank 0 waits for the next rank's line, and a rank waits for rank 0's side of the proof.
pub const wait_ms = 300_000;

/// A stream socket whose peer's host vanishing (no FIN) shows within about 20 s: 10 s idle, then 3 probes 3 s apart.
pub fn socket() !posix.socket_t {
    // macOS refuses the close-on-exec bit in a socket's type; ranks run on Linux and Windows, its host tests anywhere
    const cloexec = if (@import("builtin").os.tag.isDarwin()) 0 else posix.SOCK.CLOEXEC;
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM | cloexec, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    const one: c_int = 1;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, std.mem.asBytes(&one)) catch {};
    if (@import("builtin").os.tag == .linux) for ([_][2]u32{ .{ posix.TCP.KEEPIDLE, 10 }, .{ posix.TCP.KEEPINTVL, 3 }, .{ posix.TCP.KEEPCNT, 3 } }) |o| {
        const v: c_int = @intCast(o[1]);
        posix.setsockopt(fd, posix.IPPROTO.TCP, o[0], std.mem.asBytes(&v)) catch {};
    };
    return fd;
}

/// Rank 0: a line from each other rank into `lines`, each within `ms` of the last and each proving `secret`.
pub fn gather(server: posix.socket_t, secret: *const auth.Secret, world: usize, lines: *[max_lines]posix.socket_t, ms: u64) !void {
    var seen: u32 = 1; // a bit for each rank whose line is in; rank 0 has none
    var count: usize = 0;
    var until = auth.nowMs() + ms;
    while (count < world - 1) {
        const left = until -| auth.nowMs();
        if (left == 0) return error.RankNeverCame;
        var fds = [_]posix.pollfd{.{ .fd = server, .events = posix.POLL.IN, .revents = 0 }};
        if ((posix.poll(&fds, @intCast(@min(left, 1000))) catch 0) == 0) continue;
        const rc = posix.system.accept(server, null, null);
        if (posix.errno(rc) != .SUCCESS) return error.AcceptFailed;
        const fd: posix.socket_t = @intCast(rc);
        // a connection that proves no key, or claims a rank already in, is closed: no line, and sent no NCCL id
        const who = auth.accept(fd, secret, &.{.liveness}, left) catch |e| {
            var at: [64]u8 = undefined;
            std.log.warn("tp: a connection from {s} to the liveness port proved no key and is ignored: {s}", .{ auth.peerText(fd, &at), auth.explain(e) orelse @errorName(e) });
            _ = posix.system.close(fd);
            continue;
        };
        if (who.world != world or seen & (@as(u32, 1) << @intCast(who.rank)) != 0) {
            std.log.warn("tp: a second line for rank {d} of {d} is ignored", .{ who.rank, who.world });
            _ = posix.system.close(fd);
            continue;
        }
        seen |= @as(u32, 1) << @intCast(who.rank);
        lines[count] = fd;
        count += 1;
        until = auth.nowMs() + ms;
    }
}

/// Another rank: its line to rank 0 at `addr`, the key proven both ways, and how many connects it took (100 ms apart).
pub fn reach(io: std.Io, addr: *const posix.sockaddr.in, secret: *const auth.Secret, rank: usize, world: usize) !struct { posix.socket_t, usize } {
    var tries: usize = 0;
    while (true) : (tries += 1) {
        const fd = try socket();
        if (posix.errno(posix.system.connect(fd, @ptrCast(addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) {
            _ = auth.connect(fd, secret, .liveness, @intCast(rank), @intCast(world), wait_ms) catch |e| {
                _ = posix.system.close(fd);
                return e;
            };
            return .{ fd, tries + 1 };
        }
        _ = posix.system.close(fd);
        if (tries >= 3000) return error.RankNeverCame;
        std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    }
}

const Listener = struct { fd: posix.socket_t, addr: posix.sockaddr.in };

/// Rank 0's listening socket on a loopback port of the system's choosing.
fn testListener() !Listener {
    const fd = try socket();
    errdefer _ = posix.system.close(fd);
    var addr: posix.sockaddr.in = .{ .port = 0, .addr = std.mem.nativeToBig(u32, 0x7f000001) };
    var len: posix.socklen_t = @sizeOf(posix.sockaddr.in);
    if (posix.errno(posix.system.bind(fd, @ptrCast(&addr), len)) != .SUCCESS) return error.BindFailed;
    if (posix.errno(posix.system.listen(fd, 16)) != .SUCCESS) return error.BindFailed;
    if (posix.errno(posix.system.getsockname(fd, @ptrCast(&addr), &len)) != .SUCCESS) return error.BindFailed;
    return .{ .fd = fd, .addr = addr };
}

const TestRank = struct {
    fd: ?posix.socket_t = null,
    err: ?anyerror = null,

    fn run(addr: posix.sockaddr.in, secret: auth.Secret, rank: usize, world: usize, out: *TestRank) void {
        const line = reach(std.testing.io, &addr, &secret, rank, world) catch |e| {
            out.err = e;
            return;
        };
        out.fd = line[0];
    }
};

/// Bytes a peer of `fd` sent before it closed, up to `buf.len`.
fn testRead(fd: posix.socket_t, buf: []u8) usize {
    var at: usize = 0;
    while (at < buf.len) {
        const rc = posix.system.read(fd, buf[at..].ptr, buf.len - at);
        if (posix.errno(rc) != .SUCCESS or rc == 0) break;
        at += @intCast(rc);
    }
    return at;
}

test "security: a connection that proves no key is no liveness line and is sent nothing" {
    if (platform.windows) return error.SkipZigTest;
    const key = try auth.Secret.fromBytes("the group's rank secret for the tests");
    const l = try testListener();
    defer _ = posix.system.close(l.fd);
    // a stranger first, with the ready record a rank sends: before the proof the first connection was rank 1's line
    const stranger = try socket();
    defer _ = posix.system.close(stranger);
    try std.testing.expectEqual(posix.E.SUCCESS, posix.errno(posix.system.connect(stranger, @ptrCast(&l.addr), @sizeOf(posix.sockaddr.in))));
    const record: [64]u8 = @splat(1);
    try std.testing.expectEqual(@as(isize, record.len), posix.system.write(stranger, &record, record.len));
    var rank: TestRank = .{};
    const thread = try std.Thread.spawn(.{}, TestRank.run, .{ l.addr, key, 1, 2, &rank });
    var lines: [max_lines]posix.socket_t = undefined;
    try gather(l.fd, &key, 2, &lines, 5000);
    thread.join();
    defer _ = posix.system.close(lines[0]);
    try std.testing.expectEqual(@as(?anyerror, null), rank.err);
    defer _ = posix.system.close(rank.fd.?);
    // the line rank 0 holds is the rank's: what rank 0 writes on it (its NCCL id, in the watcher) reaches the rank only
    try std.testing.expectEqual(@as(isize, 2), posix.system.write(lines[0], "id", 2));
    var got: [2]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 2), testRead(rank.fd.?, &got));
    try std.testing.expectEqualStrings("id", &got);
    var none: [8]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), testRead(stranger, &none));
}

test "security: a rank with another key and a second line for one rank are not lines" {
    if (platform.windows) return error.SkipZigTest;
    const key = try auth.Secret.fromBytes("the group's rank secret for the tests");
    const other = try auth.Secret.fromBytes("another group's secret for the tests");
    const l = try testListener();
    defer _ = posix.system.close(l.fd);
    var ranks: [4]TestRank = @splat(.{});
    var threads: [4]std.Thread = undefined;
    // rank 1 with another key, rank 1 with the group's, a second rank 1, then rank 2
    const who = [_]struct { auth.Secret, usize }{ .{ other, 1 }, .{ key, 1 }, .{ key, 1 }, .{ key, 2 } };
    var lines: [max_lines]posix.socket_t = undefined;
    const Lead = struct {
        fn run(server: posix.socket_t, secret: auth.Secret, out: *[max_lines]posix.socket_t, err: *?anyerror) void {
            gather(server, &secret, 3, out, 5000) catch |e| {
                err.* = e;
            };
        }
    };
    var lead_err: ?anyerror = null;
    const lead = try std.Thread.spawn(.{}, Lead.run, .{ l.fd, key, &lines, &lead_err });
    for (who, 0..) |w, i| {
        threads[i] = try std.Thread.spawn(.{}, TestRank.run, .{ l.addr, w[0], w[1], 3, &ranks[i] });
        threads[i].join(); // one at a time: rank 0 meets them in this order
    }
    lead.join();
    try std.testing.expectEqual(@as(?anyerror, null), lead_err);
    try std.testing.expectEqual(@as(?anyerror, error.RankAuthWrongKey), ranks[0].err);
    for ([_]usize{ 1, 2, 3 }) |i| try std.testing.expectEqual(@as(?anyerror, null), ranks[i].err);
    // the second rank 1 proved the key, so its connect returned; rank 0 then closed it and holds two lines
    var end: [1]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), testRead(ranks[2].fd.?, &end));
    for (ranks) |r| if (r.fd) |fd| {
        _ = posix.system.close(fd);
    };
    for (lines[0..2]) |fd| _ = posix.system.close(fd);
    const late = try testListener();
    defer _ = posix.system.close(late.fd);
    try std.testing.expectError(error.RankNeverCame, gather(late.fd, &key, 2, &lines, 200));
}
