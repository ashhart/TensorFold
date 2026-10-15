//! Ranks' liveness lines: a TCP line from each rank to rank 0, open before NCCL, carrying ready and error records.
const std = @import("std");
const posix = std.posix;

/// One record on a line: ready, or an error and its name.
const Record = extern struct { kind: u32, rank: u32, text: [56]u8 = @splat(0) };
const ready_kind: u32 = 1;
const error_kind: u32 = 2;
const max_lines = 16;

/// Set by a rank that ends on purpose, before it tells the others: a line closing after that is expected.
var ending = std.atomic.Value(bool).init(false);
var ready_peers = std.atomic.Value(usize).init(0);
var lines: [max_lines]posix.socket_t = undefined;
var line_count: usize = 0;
var my_rank: usize = 0;
var peer_digest: [max_lines][32]u8 = undefined; // each line's ready record: its rank's checkpoint digest
var peer_rank: [max_lines]u32 = undefined;

/// This rank ends on purpose (rank 0 sends its end header, or a rank leaves on one).
pub fn ended() void {
    ending.store(true, .release);
}

/// NCCL_COMM_ID's host (IPv4) at its port plus one.
fn address() !posix.sockaddr.in {
    const id = std.mem.span(std.c.getenv("NCCL_COMM_ID") orelse return error.NoCommId);
    const colon = std.mem.lastIndexOfScalar(u8, id, ':') orelse return error.NoCommId;
    const port = (std.fmt.parseInt(u16, id[colon + 1 ..], 10) catch return error.NoCommId) +% 1;
    const ip = std.Io.net.IpAddress.parse(id[0..colon], port) catch return error.NoCommId;
    if (ip != .ip4) return error.NoCommId;
    return .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.readInt(u32, &ip.ip4.bytes, .little) };
}

/// A stream socket whose peer's host vanishing (no FIN) shows within about 20 s: 10 s idle, then 3 probes 3 s apart.
fn socket() !posix.socket_t {
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM | posix.SOCK.CLOEXEC, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(rc);
    const one: c_int = 1;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.KEEPALIVE, std.mem.asBytes(&one)) catch {};
    for ([_][2]u32{ .{ posix.TCP.KEEPIDLE, 10 }, .{ posix.TCP.KEEPINTVL, 3 }, .{ posix.TCP.KEEPCNT, 3 } }) |o| {
        const v: c_int = @intCast(o[1]);
        posix.setsockopt(fd, posix.IPPROTO.TCP, o[0], std.mem.asBytes(&v)) catch {};
    }
    return fd;
}

/// Rank 0 takes a line from each other rank within 5 minutes, and they connect to it; then every line is watched.
pub fn start(io: std.Io, rank: usize, world: usize) !void {
    if (world < 2) return;
    if (world > max_lines + 1) return error.TooManyRanks;
    my_rank = rank;
    var addr = try address();
    if (rank == 0) {
        const server = try socket();
        defer _ = posix.system.close(server);
        const one: c_int = 1;
        posix.setsockopt(server, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one)) catch return error.SocketFailed;
        if (posix.errno(posix.system.bind(server, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.BindFailed;
        if (posix.errno(posix.system.listen(server, 16)) != .SUCCESS) return error.BindFailed;
        for (1..world) |_| {
            var fds = [_]posix.pollfd{.{ .fd = server, .events = posix.POLL.IN, .revents = 0 }};
            if ((posix.poll(&fds, 300_000) catch 0) == 0) return error.RankNeverCame;
            const rc = posix.system.accept(server, null, null);
            if (posix.errno(rc) != .SUCCESS) return error.AcceptFailed;
            lines[line_count] = @intCast(rc);
            line_count += 1;
        }
    } else {
        var tries: usize = 0;
        while (true) : (tries += 1) {
            const fd = try socket();
            if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) {
                lines[0] = fd;
                line_count = 1;
                break;
            }
            _ = posix.system.close(fd);
            if (tries >= 3000) return error.RankNeverCame;
            std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
        }
    }
    for (lines[0..line_count], 0..) |fd, i| (try std.Thread.spawn(.{}, watch, .{ fd, i })).detach();
}

fn send(r: Record) void {
    for (lines[0..line_count]) |fd| {
        const b = std.mem.asBytes(&r);
        var at: usize = 0;
        while (at < b.len) {
            const rc = posix.system.write(fd, b[at..].ptr, b.len - at);
            switch (posix.errno(rc)) {
                .SUCCESS => at += @intCast(rc),
                .INTR => {},
                else => break,
            }
        }
    }
}

/// This rank loaded `digest`'s bytes: it tells the others, waits for theirs, and stops if any rank's bytes differ.
pub fn ready(io: std.Io, digest: [32]u8) !void {
    if (line_count == 0) return;
    var r: Record = .{ .kind = ready_kind, .rank = @intCast(my_rank) };
    @memcpy(r.text[0..32], &digest);
    send(r);
    var waited: usize = 0;
    while (ready_peers.load(.acquire) < line_count) : (waited += 1) {
        if (waited >= 90_000) return error.RankNeverReady; // 15 minutes
        std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
    }
    for (peer_digest[0..line_count], peer_rank[0..line_count]) |d, rank| if (!std.mem.eql(u8, &d, &digest)) {
        std.log.err("rank {d}'s checkpoint bytes differ from rank {d}'s (every byte hashed): every rank stops", .{ rank, my_rank });
        return error.CheckpointBytesDiffer;
    };
}

/// This rank stops with `name`: every other rank hears it and stops with it (the caller exits next).
pub fn fail(name: []const u8) void {
    if (line_count == 0) return;
    ending.store(true, .release);
    var r: Record = .{ .kind = error_kind, .rank = @intCast(my_rank) };
    @memcpy(r.text[0..@min(name.len, r.text.len)], name[0..@min(name.len, r.text.len)]);
    send(r);
}

/// Ends this process at once: a stream may hold kernels that wait on the gone rank, which a normal exit would join.
fn stop() noreturn {
    std.c._exit(1);
}

/// A line's records; it closing or carrying an error stops this rank unless it ends on purpose.
fn watch(fd: posix.socket_t, line: usize) void {
    var r: Record = undefined;
    while (true) {
        const b = std.mem.asBytes(&r);
        var at: usize = 0;
        while (at < b.len) {
            const rc = posix.system.read(fd, b[at..].ptr, b.len - at);
            switch (posix.errno(rc)) {
                .SUCCESS => if (rc == 0) break else {
                    at += @intCast(rc);
                },
                .INTR => {},
                else => break,
            }
        }
        if (at < b.len) break;
        if (r.kind == ready_kind) {
            @memcpy(&peer_digest[line], r.text[0..32]);
            peer_rank[line] = r.rank;
            _ = ready_peers.fetchAdd(1, .acq_rel);
            continue;
        }
        if (r.kind == error_kind and !ending.load(.acquire)) {
            const text = std.mem.sliceTo(&r.text, 0);
            std.log.err("rank {d}: rank {d} stopped ({s}); this rank stops too", .{ my_rank, r.rank, text });
            stop();
        }
    }
    if (ending.load(.acquire)) return;
    std.log.err("rank {d}: another rank's process ended (its line closed); this rank stops instead of waiting on it", .{my_rank});
    stop();
}

test "a record is 64 bytes, and a name past 56 bytes is cut" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Record));
    var r: Record = .{ .kind = error_kind, .rank = 1 };
    const name: [70]u8 = @splat('x');
    @memcpy(r.text[0..@min(name.len, r.text.len)], name[0..@min(name.len, r.text.len)]);
    try std.testing.expectEqual(@as(usize, 56), std.mem.sliceTo(&r.text, 0).len);
}
