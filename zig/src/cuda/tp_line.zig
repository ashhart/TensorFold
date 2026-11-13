//! The ranks' liveness lines: rank 0 takes one from each other rank; each proves the group's key, then seals records.
const std = @import("std");
const platform = @import("sys");
const posix = platform.posix;
const auth = platform.rank_auth;

pub const max_lines = 16;

/// Milliseconds rank 0 waits for the next rank's line, and a rank waits for rank 0's side of the proof.
pub const wait_ms = 300_000;

/// A keyed group's line after its proof: every record on it crosses sealed. Senders take turns; one thread reads.
pub const Sealed = struct {
    seal: auth.seal.Seal,
    arriving: auth.seal.Arriving = .{},
    sending: std.atomic.Value(bool) = .init(false),
};

/// A line's seal under its proof's `session`; null on a group with no key, whose lines stay on one host as plain bytes.
fn sealOf(secret: *const auth.Secret, session: *const auth.Key, connector: bool) !?*Sealed {
    if (!secret.keyed) return null;
    const s = try std.heap.page_allocator.create(Sealed);
    s.* = .{ .seal = .init(session, connector) };
    return s;
}

/// `bytes` on a line: one sealed record, or as they are on a line with no seal. A line that fails is its watcher's.
pub fn write(fd: posix.socket_t, sealed: ?*Sealed, bytes: []const u8) void {
    const s = sealed orelse return writeAll(fd, bytes);
    while (s.sending.swap(true, .acquire)) std.atomic.spinLoopHint();
    defer s.sending.store(false, .release);
    var wire: [auth.seal.max_record]u8 = undefined;
    writeAll(fd, wire[0..s.seal.seal(bytes, &wire)]);
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => at += @intCast(rc),
            .INTR => {},
            else => return,
        }
    }
}

/// Fills `out` from a line. False once the line closed or failed, or carried a record that does not open under the key.
pub fn read(fd: posix.socket_t, sealed: ?*Sealed, out: []u8) bool {
    var at: usize = 0;
    while (at < out.len) {
        const s = sealed orelse {
            at += readSome(fd, out[at..]) orelse return false;
            continue;
        };
        at += s.seal.take(out[at..]);
        if (at == out.len) break;
        const want = s.arriving.want() catch return refused(fd);
        const n = readSome(fd, want) orelse return false;
        s.arriving.fed(&s.seal, n) catch return refused(fd);
    }
    return true;
}

/// What `fd` has now, up to `out.len` bytes; null once it closed or failed.
fn readSome(fd: posix.socket_t, out: []u8) ?usize {
    while (true) {
        const rc = posix.system.read(fd, out.ptr, out.len);
        switch (posix.errno(rc)) {
            .SUCCESS => return if (rc == 0) null else @intCast(rc),
            .INTR => {},
            else => return null,
        }
    }
}

fn refused(fd: posix.socket_t) bool {
    if (!@import("builtin").is_test) std.log.warn("tp: a record on a liveness line did not open under the group's key, so the line is closed", .{});
    _ = posix.system.shutdown(fd, posix.SHUT.RDWR);
    return false;
}

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

/// Rank 0: a line from each other rank into `lines` and its seal into `seals`, each within `ms` of the last.
pub fn gather(server: posix.socket_t, secret: *const auth.Secret, world: usize, lines: *[max_lines]posix.socket_t, seals: *[max_lines]?*Sealed, ms: u64) !void {
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
        seals[count] = sealOf(secret, &who.session, false) catch {
            _ = posix.system.close(fd);
            continue;
        };
        seen |= @as(u32, 1) << @intCast(who.rank);
        lines[count] = fd;
        count += 1;
        until = auth.nowMs() + ms;
    }
}

/// Another rank: its line to rank 0 at `addr` with the key proven both ways, its seal, and its connects (100 ms apart).
pub fn reach(io: std.Io, addr: *const posix.sockaddr.in, secret: *const auth.Secret, rank: usize, world: usize) !struct { posix.socket_t, ?*Sealed, usize } {
    var tries: usize = 0;
    while (true) : (tries += 1) {
        const fd = try socket();
        if (posix.errno(posix.system.connect(fd, @ptrCast(addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) {
            errdefer _ = posix.system.close(fd);
            const session = try auth.connect(fd, secret, .liveness, @intCast(rank), @intCast(world), wait_ms);
            return .{ fd, try sealOf(secret, &session, true), tries + 1 };
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
    seal: ?*Sealed = null,
    err: ?anyerror = null,

    fn run(addr: posix.sockaddr.in, secret: auth.Secret, rank: usize, world: usize, out: *TestRank) void {
        const line = reach(std.testing.io, &addr, &secret, rank, world) catch |e| {
            out.err = e;
            return;
        };
        out.fd = line[0];
        out.seal = line[1];
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
    var seals: [max_lines]?*Sealed = @splat(null);
    try gather(l.fd, &key, 2, &lines, &seals, 5000);
    thread.join();
    defer _ = posix.system.close(lines[0]);
    try std.testing.expectEqual(@as(?anyerror, null), rank.err);
    defer _ = posix.system.close(rank.fd.?);
    // the line rank 0 holds is the rank's: what rank 0 writes on it (its NCCL id, in the watcher) reaches the rank only
    write(lines[0], seals[0], "id");
    var got: [2]u8 = undefined;
    try std.testing.expect(read(rank.fd.?, rank.seal, &got));
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
            var seals: [max_lines]?*Sealed = @splat(null);
            gather(server, &secret, 3, out, &seals, 5000) catch |e| {
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
    var none: [max_lines]?*Sealed = @splat(null);
    try std.testing.expectError(error.RankNeverCame, gather(late.fd, &key, 2, &lines, &none, 200));
}

/// A rank's line to rank 0 over loopback with both ends' seals: rank 0's end first.
fn testLine(secret: auth.Secret) !struct { [2]posix.socket_t, [2]?*Sealed } {
    const l = try testListener();
    defer _ = posix.system.close(l.fd);
    var rank: TestRank = .{};
    const thread = try std.Thread.spawn(.{}, TestRank.run, .{ l.addr, secret, 1, 2, &rank });
    var lines: [max_lines]posix.socket_t = undefined;
    var seals: [max_lines]?*Sealed = @splat(null);
    try gather(l.fd, &secret, 2, &lines, &seals, 5000);
    thread.join();
    if (rank.err) |e| return e;
    return .{ .{ lines[0], rank.fd.? }, .{ seals[0], rank.seal } };
}

test "security: a keyed group's line shows the network none of rank 0's NCCL id, and a record lost on the way ends it" {
    if (platform.windows) return error.SkipZigTest;
    const id = "ncclUniqueId: 192.0.2.7:41233 magic 5f1a";
    const fds, const seals = try testLine(try auth.Secret.fromBytes("the group's rank secret for the tests"));
    defer for (fds) |fd| {
        _ = posix.system.close(fd);
    };
    // both ways, as the ranks use it: rank 0's id to the rank, the rank's ready record back
    var got: [id.len]u8 = undefined;
    write(fds[0], seals[0], id);
    try std.testing.expect(read(fds[1], seals[1], &got));
    try std.testing.expectEqualStrings(id, &got);
    write(fds[1], seals[1], "ready");
    try std.testing.expect(read(fds[0], seals[0], got[0..5]));
    try std.testing.expectEqualStrings("ready", got[0..5]);
    // the same id again, taken off the wire as a host on the path would see it: a length, sealed bytes and a tag
    write(fds[0], seals[0], id);
    var wire: [auth.seal.head_len + id.len + auth.seal.tag_len]u8 = undefined;
    try std.testing.expectEqual(wire.len, testRead(fds[1], &wire));
    try std.testing.expect(std.mem.indexOf(u8, &wire, "ncclUniqueId") == null);
    try std.testing.expect(std.mem.indexOf(u8, &wire, "41233") == null);
    // that record never reached the rank's reader, so the next one is out of place: it does not open, and the line ends
    write(fds[0], seals[0], id);
    try std.testing.expect(!read(fds[1], seals[1], &got));
    var end: [1]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), testRead(fds[0], &end));
}

test "a line with no key carries its records as they are, on one host" {
    if (platform.windows) return error.SkipZigTest;
    const fds, const seals = try testLine(.none);
    defer for (fds) |fd| {
        _ = posix.system.close(fd);
    };
    try std.testing.expectEqual(@as(?*Sealed, null), seals[0]);
    try std.testing.expectEqual(@as(?*Sealed, null), seals[1]);
    write(fds[0], seals[0], "ready");
    var wire: [5]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 5), testRead(fds[1], &wire));
    try std.testing.expectEqualStrings("ready", &wire);
}
