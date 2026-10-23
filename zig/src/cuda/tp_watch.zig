//! Ranks' liveness lines: a TCP line from each rank to rank 0, open before NCCL: rank 0's NCCL id, ready, error, wake.
const std = @import("std");
const platform = @import("sys");
const posix = @import("sys").posix;

/// One record on a line: ready, an error and its name, rank 0's wake, or a part of rank 0's NCCL id.
const Record = extern struct { kind: u32, rank: u32, text: [56]u8 = @splat(0) };
const ready_kind: u32 = 1;
const error_kind: u32 = 2;
const wake_kind: u32 = 3;
const id_kind: u32 = 4;
const max_lines = 16;
const id_len = 128; // ncclUniqueId's bytes
const id_part = 48; // an id record's bytes, from text[8]; its part's number is text[0]
const id_parts = (id_len + id_part - 1) / id_part;
const id_all: u32 = (1 << id_parts) - 1;

/// Set by a rank that ends on purpose, before it tells the others: a line closing after that is expected.
var ending = std.atomic.Value(bool).init(false);
var ready_peers = std.atomic.Value(usize).init(0);
var lines: [max_lines]posix.socket_t = undefined;
var line_count: usize = 0;
var my_rank: usize = 0;
var peer_digest: [max_lines][32]u8 = undefined; // each line's ready record: its rank's checkpoint digest
var peer_rank: [max_lines]u32 = undefined;
var wakes = std.atomic.Value(u32).init(0); // wake records this rank's line carried (another rank's)
var slept: u32 = 0; // the wakes this rank's sleeps took
var watch_io: std.Io = undefined; // the watchers' futex wakes
var id_bytes: [id_len]u8 = undefined; // rank 0's NCCL id as this rank's line carried it
var id_seen = std.atomic.Value(u32).init(0); // a bit for each part of it taken

/// This rank ends on purpose (rank 0 sends its end header, or a rank leaves on one).
pub fn ended() void {
    ending.store(true, .release);
}

/// NCCL_COMM_ID's host (IPv4) at its port plus one.
fn address() !posix.sockaddr.in {
    const id = std.mem.span(platform.c.getenv("NCCL_COMM_ID") orelse return error.NoCommId);
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
    watch_io = io;
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
        std.log.err("rank {d}'s checkpoint identity differs from rank {d}'s: every rank stops", .{ rank, my_rank });
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

/// Whether this rank has lines (more than one rank).
pub fn open() bool {
    return line_count > 0;
}

/// Rank 0: every other rank's sleep returns (its next step follows on NCCL).
pub fn wake() void {
    send(.{ .kind = wake_kind, .rank = @intCast(my_rank) });
}

/// Until rank 0's next wake record, which may already be here: a wait that spins neither CUDA nor NCCL.
pub fn sleep(io: std.Io) void {
    if (line_count == 0) return;
    slept += 1;
    while (true) {
        const n = wakes.load(.acquire);
        if (n >= slept) return;
        io.futexWaitUncancelable(u32, &wakes.raw, n);
    }
}

/// Rank 0: its NCCL id to every other rank, before NCCL starts there (as the fabric's bootstrap broadcasts its id).
pub fn sendId(id: *const [id_len]u8) void {
    for (0..id_parts) |part| send(idRecord(id, part));
}

/// A rank other than 0: rank 0's NCCL id from this rank's line, within 15 minutes.
pub fn takeId(io: std.Io) ![id_len]u8 {
    var waited: usize = 0;
    while (id_seen.load(.acquire) != id_all) : (waited += 1) {
        if (waited >= 90_000) return error.RankNeverSentId;
        std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
    }
    return id_bytes;
}

fn idRecord(id: *const [id_len]u8, part: usize) Record {
    var r: Record = .{ .kind = id_kind, .rank = @intCast(my_rank) };
    const at = part * id_part;
    const n = @min(id_part, id_len - at);
    r.text[0] = @intCast(part);
    @memcpy(r.text[8..][0..n], id[at..][0..n]);
    return r;
}

/// A part of rank 0's id into id_bytes; a part past the id is not taken.
fn takePart(r: *const Record) void {
    const part = r.text[0];
    if (part >= id_parts) return;
    const at = @as(usize, part) * id_part;
    const n = @min(id_part, id_len - at);
    @memcpy(id_bytes[at..][0..n], r.text[8..][0..n]);
    _ = id_seen.fetchOr(@as(u32, 1) << @intCast(part), .acq_rel);
}

/// A rank's transport failed: with lines open every rank hears `name` and this one ends now; with none, nothing.
pub fn fatal(name: []const u8) void {
    if (line_count == 0) return;
    fail(name);
    stop();
}

/// Ends this process at once: a stream may hold kernels that wait on the gone rank, which a normal exit would join.
fn stop() noreturn {
    platform.c._exit(1);
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
        if (r.kind == wake_kind) {
            _ = wakes.fetchAdd(1, .release);
            watch_io.futexWake(u32, &wakes.raw, 1);
            continue;
        }
        if (r.kind == id_kind) {
            takePart(&r);
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

const SleepTest = struct {
    var joined = false;
    const Guard = struct {
        done: std.atomic.Value(bool) = .init(false),
        expired: std.atomic.Value(bool) = .init(false),
        target: u32,
        limit: usize,

        fn run(g: *Guard) void {
            for (0..g.limit) |_| {
                if (g.done.load(.acquire)) return;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
            g.expired.store(true, .release);
            wakes.store(g.target, .release);
            std.testing.io.futexWake(u32, &wakes.raw, 1);
        }
    };

    fn late(fd: c_int, bytes: []const u8) void {
        std.Io.sleep(std.testing.io, .fromMilliseconds(50), .awake) catch {};
        _ = posix.system.write(fd, bytes.ptr, bytes.len);
    }

    fn firstWake(first: u32) !void {
        for (0..2000) |_| {
            if (wakes.load(.acquire) != first) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.WakeTimeout;
    }

    fn check(mode: u8) !void {
        var pair: [2]c_int = undefined;
        if (ProcessTest.socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketFailed;
        defer _ = posix.system.close(pair[0]);
        var peer_open = true;
        defer if (peer_open) {
            _ = posix.system.close(pair[1]);
        };
        const was_ending = ending.load(.acquire);
        const was_count = line_count;
        const was_slept = slept;
        const first = wakes.load(.acquire);
        defer {
            ending.store(was_ending, .release);
            line_count = was_count;
            slept = was_slept;
            wakes.store(first, .release);
        }
        watch_io = std.testing.io;
        line_count = 1;
        slept = first;
        ending.store(false, .release);
        joined = false;
        const watcher = try std.Thread.spawn(.{}, watch, .{ pair[0], 0 });
        defer {
            ending.store(true, .release);
            _ = posix.system.close(pair[1]);
            peer_open = false;
            watcher.join();
            joined = true;
        }
        if (mode == 1) return error.InjectedParentFailure;
        var guard: Guard = .{ .target = first + 2, .limit = if (mode == 2) 100 else 2000 };
        const timer = try std.Thread.spawn(.{}, Guard.run, .{&guard});
        defer {
            guard.done.store(true, .release);
            timer.join();
        }
        const rec: Record = .{ .kind = wake_kind, .rank = 0 };
        const record = std.mem.asBytes(&rec);
        try std.testing.expectEqual(@as(isize, record.len), posix.system.write(pair[1], record.ptr, record.len));
        try firstWake(first);
        if (guard.expired.load(.acquire)) return error.WakeTimeout;
        sleep(std.testing.io);
        if (guard.expired.load(.acquire)) return error.WakeTimeout;
        const sender: ?std.Thread = if (mode == 2) null else try std.Thread.spawn(.{}, late, .{ pair[1], record[0..] });
        defer if (sender) |t| t.join();
        sleep(std.testing.io);
        if (guard.expired.load(.acquire)) return error.WakeTimeout;
        try std.testing.expectEqual(first + 2, wakes.load(.acquire));
    }
};

test "a sleep takes a wake record that came before it, and waits for one that comes after" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    try SleepTest.check(0);
    try std.testing.expect(SleepTest.joined);
}

test "a lost wake and an early failure both stop and join the test watcher" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const before = wakes.load(.acquire);
    const count = line_count;
    try std.testing.expectError(error.InjectedParentFailure, SleepTest.check(1));
    try std.testing.expect(SleepTest.joined);
    try std.testing.expectError(error.WakeTimeout, SleepTest.check(2));
    try std.testing.expect(SleepTest.joined);
    try std.testing.expectEqual(before, wakes.load(.acquire));
    try std.testing.expectEqual(count, line_count);
}

const IdTest = struct {
    /// The watcher took every part within 2 s (takeId alone would wait 15 minutes).
    fn taken() !void {
        for (0..2000) |_| {
            if (id_seen.load(.acquire) == id_all) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.IdTimeout;
    }

    fn bytes(seed: u8) [id_len]u8 {
        var id: [id_len]u8 = undefined;
        for (&id, 0..) |*b, i| b.* = @as(u8, @truncate(i *% 37)) +% seed;
        return id;
    }
};

test "a rank takes rank 0's NCCL id from its line in any order, not a part past the id, and as sendId sends it" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var pair: [2]c_int = undefined;
    if (ProcessTest.socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketFailed;
    defer _ = posix.system.close(pair[0]);
    var peer_open = true;
    defer if (peer_open) {
        _ = posix.system.close(pair[1]);
    };
    const was_ending = ending.load(.acquire);
    const was_seen = id_seen.load(.acquire);
    const was_count = line_count;
    defer {
        ending.store(was_ending, .release);
        id_seen.store(was_seen, .release);
        line_count = was_count;
    }
    id_seen.store(0, .release);
    ending.store(false, .release);
    const watcher = try std.Thread.spawn(.{}, watch, .{ pair[0], 0 });
    defer {
        ending.store(true, .release);
        _ = posix.system.close(pair[1]);
        peer_open = false;
        watcher.join();
    }
    const id = IdTest.bytes(11);
    var past = idRecord(&id, 1);
    past.text[0] = id_parts;
    @memset(past.text[8..], 0xff);
    for ([_]Record{ idRecord(&id, 2), past, idRecord(&id, 0), idRecord(&id, 1) }) |r| {
        const b = std.mem.asBytes(&r);
        try std.testing.expectEqual(@as(isize, b.len), posix.system.write(pair[1], b.ptr, b.len));
    }
    try IdTest.taken();
    const got = try takeId(std.testing.io);
    try std.testing.expectEqualSlices(u8, &id, &got);
    id_seen.store(0, .release);
    lines[0] = pair[1];
    line_count = 1;
    const next = IdTest.bytes(5);
    sendId(&next);
    try IdTest.taken();
    const sent = try takeId(std.testing.io);
    try std.testing.expectEqualSlices(u8, &next, &sent);
}

const ProcessTest = struct {
    var last_pid: c_int = 0;
    var last_fd: c_int = -1;
    var last_reaped = false;
    extern "c" fn fork() c_int;
    extern "c" fn socketpair(c_int, c_int, c_int, *[2]c_int) c_int;
    extern "c" fn kill(c_int, c_int) c_int;
    fn check(mode: u8) !void {
        var pair: [2]c_int = undefined;
        if (socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketFailed;
        const pid = fork();
        if (pid < 0) {
            _ = posix.system.close(pair[0]);
            _ = posix.system.close(pair[1]);
            return error.ForkFailed;
        }
        if (pid == 0) {
            _ = posix.system.close(pair[1]);
            ending.store(mode == 2, .release);
            watch(pair[0], 0);
            std.c._exit(0);
        }
        _ = posix.system.close(pair[0]);
        last_pid = pid;
        last_fd = pair[1];
        last_reaped = false;
        defer {
            if (last_fd >= 0) {
                _ = posix.system.close(last_fd);
                last_fd = -1;
            }
            if (!last_reaped) {
                _ = kill(pid, 9);
                var cleanup: c_int = 0;
                _ = std.c.waitpid(pid, &cleanup, 0);
                last_reaped = true;
            }
        }
        if (mode == 3) return error.InjectedParentFailure;
        if (mode == 1) {
            var record: Record = .{ .kind = error_kind, .rank = 1 };
            @memcpy(record.text[0..8], "peer OOM");
            const data = std.mem.asBytes(&record);
            try std.testing.expectEqual(@as(isize, data.len), posix.system.write(pair[1], data.ptr, data.len));
        }
        _ = posix.system.close(pair[1]);
        last_fd = -1;
        var status: c_int = 0;
        for (0..2000) |_| {
            if (std.c.waitpid(pid, &status, 1) == pid) {
                last_reaped = true;
                try std.testing.expectEqual(@as(c_int, 0), status & 0x7f);
                try std.testing.expectEqual(@as(c_int, if (mode == 2) 0 else 1), (status >> 8) & 0xff);
                return;
            }
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        _ = kill(pid, 9);
        _ = std.c.waitpid(pid, &status, 0);
        last_reaped = true;
        return error.PeerDidNotStop;
    }
};
test "the real peer watcher exits on a closed line and on an error record, and returns for an intentional end" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    try ProcessTest.check(0);
    try ProcessTest.check(1);
    try ProcessTest.check(2);
}

test "a failing parent check still closes the line, kills and reaps the watcher's child" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    defer if (!ProcessTest.last_reaped) {
        if (ProcessTest.last_fd >= 0) _ = posix.system.close(ProcessTest.last_fd);
        _ = ProcessTest.kill(ProcessTest.last_pid, 9);
        var status: c_int = 0;
        _ = std.c.waitpid(ProcessTest.last_pid, &status, 0);
    };
    try std.testing.expectError(error.InjectedParentFailure, ProcessTest.check(3));
    try std.testing.expect(ProcessTest.last_reaped);
    var status: c_int = 0;
    try std.testing.expectEqual(@as(c_int, -1), std.c.waitpid(ProcessTest.last_pid, &status, 1));
}

const id_test = @import("tp_watch_id_test.zig");

test "tp id coverage: rank zero fans all 128 bytes to every TP2 TP3 TP4 and TP8 follower" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    const was_count = line_count;
    const was_rank = my_rank;
    var saved: [max_lines]posix.socket_t = undefined;
    @memcpy(saved[0..was_count], lines[0..was_count]);
    defer {
        @memcpy(lines[0..was_count], saved[0..was_count]);
        line_count = was_count;
        my_rank = was_rank;
    }
    for ([_]usize{ 2, 3, 4, 8 }) |world| {
        var pairs: [7][2]c_int = undefined;
        var opened: usize = 0;
        defer for (pairs[0..opened]) |pair| {
            _ = posix.system.close(pair[0]);
            _ = posix.system.close(pair[1]);
        };
        for (0..world - 1) |i| {
            if (ProcessTest.socketpair(posix.AF.UNIX, posix.SOCK.STREAM, 0, &pairs[i]) != 0) return error.SocketFailed;
            opened += 1;
            lines[i] = pairs[i][0];
        }
        line_count = world - 1;
        my_rank = 0;
        const id = IdTest.bytes(@intCast(world));
        sendId(&id);
        for (pairs[0..opened]) |pair| {
            var records: [3]Record = undefined;
            try id_test.receive(pair[1], &records);
            try id_test.verify(&records, &id);
        }
    }
}

test "tp id coverage: the wire oracle rejects missing parts and changed ID bytes" {
    const was_rank = my_rank;
    defer my_rank = was_rank;
    my_rank = 0;
    const id = IdTest.bytes(71);
    var records = [_]Record{ idRecord(&id, 0), idRecord(&id, 1), idRecord(&id, 2) };
    try id_test.verify(&records, &id);
    try std.testing.expectError(error.IdPartsMissing, id_test.verify(records[0..2], &id));
    records[1].text[8] ^= 1;
    try std.testing.expectError(error.BadIdBytes, id_test.verify(&records, &id));
    records[1].text[8] ^= 1;
    records[2].text[0] = 1;
    try std.testing.expectError(error.BadIdPart, id_test.verify(&records, &id));
}
