//! carveout_lend.zig's host tests: a fake daemon on a real SEQPACKET socket, and a driver that cannot answer.

const std = @import("std");
const linux = std.os.linux;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const carveout = @import("carveout.zig");
const lend = @import("carveout_lend.zig");
const claim = lend.claim;
const slab = lend.slab;
const reply_max = lend.reply_max;

/// The fake daemon's own sockaddr_un and control buffer, not the client's: the family, then a 108-byte path.
const SockaddrUn = extern struct { family: linux.sa_family_t = linux.AF.UNIX, path: [108]u8 };
const control_max = 64;

/// What the fake daemon replies with, and whether it waits for the client's close afterwards.
const Script = struct {
    payload: []const u8,
    lent: ?linux.fd_t = null, // the descriptor to attach, if any
    extra: ?linux.fd_t = null, // a second descriptor, to prove the client closes what it does not keep
    wait_close: bool = false,
    close_after_reply: bool = false, // the daemon closes once it has refused, as the real one does
    silent: bool = false, // the daemon accepts and reads the request but never answers; it waits for the client's close
};

/// What the fake daemon saw: the request bytes, and whether the client closed the connection.
const Observed = struct {
    request: [512]u8 = undefined,
    request_len: usize = 0,
    closed: bool = false,
    failed: ?anyerror = null,
};

/// A fake daemon: a listening SEQPACKET socket under the test's tmp dir, one connection, one scripted reply.
const Daemon = struct {
    buf: [128]u8 = undefined,
    path: [:0]const u8 = "",
    listener: linux.fd_t = -1,
    script: Script = .{ .payload = "" },
    observed: Observed = .{},
    thread: ?std.Thread = null,

    fn start(d: *Daemon, tmp: *std.testing.TmpDir, name: []const u8, script: Script) !void {
        d.script = script;
        d.observed = .{};
        d.path = try std.fmt.bufPrintSentinel(&d.buf, ".zig-cache/tmp/{s}/{s}.sock", .{ tmp.sub_path, name }, 0);
        var addr: SockaddrUn = .{ .path = @splat(0) };
        @memcpy(addr.path[0..d.path.len], d.path);
        const rc = linux.socket(linux.AF.UNIX, linux.SOCK.SEQPACKET | linux.SOCK.CLOEXEC, 0);
        if (linux.errno(rc) != .SUCCESS) return error.SocketFailed;
        d.listener = @intCast(rc);
        errdefer _ = linux.close(d.listener);
        const len: linux.socklen_t = @intCast(@offsetOf(SockaddrUn, "path") + d.path.len + 1);
        if (linux.errno(linux.bind(d.listener, @ptrCast(&addr), len)) != .SUCCESS) return error.BindFailed;
        if (linux.errno(linux.listen(d.listener, 4)) != .SUCCESS) return error.ListenFailed;
        d.thread = try std.Thread.spawn(.{}, serve, .{d});
    }

    fn stop(d: *Daemon) void {
        if (d.thread) |th| th.join();
        d.thread = null;
        if (d.listener >= 0) _ = linux.close(d.listener);
        d.listener = -1;
    }
};

/// The daemon's side: accept, read the request, reply, and (when asked) wait for the client's close.
fn serve(d: *Daemon) void {
    const conn = acceptWithin(d.listener, 5000) orelse {
        d.observed.failed = error.NoConnection;
        return;
    };
    defer _ = linux.close(conn);
    const n = linux.read(conn, &d.observed.request, d.observed.request.len);
    if (linux.errno(n) != .SUCCESS) {
        d.observed.failed = error.ReadFailed;
        return;
    }
    d.observed.request_len = @intCast(n);
    if (!d.script.silent) sendReply(conn, d.script);
    if (d.script.close_after_reply) return; // the real daemon closes after a refusal; the reply is already queued
    if (!d.script.wait_close and !d.script.silent) return;
    var fds = [_]linux.pollfd{.{ .fd = conn, .events = linux.POLL.IN, .revents = 0 }};
    const r = linux.poll(&fds, 1, 5000);
    if (linux.errno(r) != .SUCCESS or r == 0) {
        d.observed.failed = error.NoClose;
        return;
    }
    var one: [1]u8 = undefined;
    const c = linux.read(conn, &one, 1);
    d.observed.closed = linux.errno(c) == .SUCCESS and c == 0;
}

/// The next connection, or null when none arrives within `ms`: a failing test must not hang the suite.
fn acceptWithin(listener: linux.fd_t, ms: i32) ?linux.fd_t {
    var fds = [_]linux.pollfd{.{ .fd = listener, .events = linux.POLL.IN, .revents = 0 }};
    const r = linux.poll(&fds, 1, ms);
    if (linux.errno(r) != .SUCCESS or r == 0) return null;
    const rc = linux.accept(listener, null, null);
    if (linux.errno(rc) != .SUCCESS) return null;
    return @intCast(rc);
}

/// The scripted reply, with its descriptors in the ancillary data.
fn sendReply(conn: linux.fd_t, script: Script) void {
    var control: [control_max]u8 align(linux.cmsg_align) = @splat(0);
    var iov = [_]std.posix.iovec_const{.{ .base = script.payload.ptr, .len = script.payload.len }};
    var msg: linux.msghdr_const = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = null, .controllen = 0, .flags = 0 };
    var fds: [2]linux.fd_t = undefined;
    var n: usize = 0;
    if (script.lent) |l| {
        fds[n] = l;
        n += 1;
    }
    if (script.extra) |e| {
        fds[n] = e;
        n += 1;
    }
    if (n > 0) {
        const h: *linux.cmsghdr = @ptrCast(&control);
        h.* = .{ .len = @sizeOf(linux.cmsghdr) + n * @sizeOf(linux.fd_t), .level = linux.SOL.SOCKET, .type = linux.SCM.RIGHTS };
        @memcpy(control[@sizeOf(linux.cmsghdr)..][0 .. n * @sizeOf(linux.fd_t)], std.mem.sliceAsBytes(fds[0..n]));
        msg.control = &control;
        msg.controllen = std.mem.alignForward(usize, h.len, linux.cmsg_align);
    }
    _ = linux.sendmsg(conn, &msg, 0);
}

/// A descriptor the daemon can lend: a memfd holding `marker`, so the client can prove it is the same file.
fn memfd(marker: []const u8) !linux.fd_t {
    const rc = linux.memfd_create("lend", 0);
    if (linux.errno(rc) != .SUCCESS) return error.MemfdFailed;
    const fd: linux.fd_t = @intCast(rc);
    errdefer _ = linux.close(fd);
    // The whole reservation's size, so the length check passes for any grant; a memfd is sparse and costs no memory.
    if (linux.errno(linux.ftruncate(fd, 2145386496)) != .SUCCESS) return error.MemfdFailed;
    if (linux.errno(linux.pwrite(fd, marker.ptr, marker.len, 0)) != .SUCCESS) return error.MemfdFailed;
    return fd;
}

/// The descriptors this process holds, so a test can prove one was closed rather than leaked.
fn openFds() usize {
    const rc = linux.open("/proc/self/fd", .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return 0;
    const fd: linux.fd_t = @intCast(rc);
    defer _ = linux.close(fd);
    var buf: [2048]u8 = undefined;
    var n: usize = 0;
    while (true) {
        const r = linux.getdents64(fd, &buf, buf.len);
        if (linux.errno(r) != .SUCCESS) break;
        const len: usize = @intCast(r);
        if (len == 0) break;
        var off: usize = 0;
        while (off < len) {
            const e: *const linux.dirent64 = @ptrCast(@alignCast(buf[off..].ptr));
            if (e.reclen == 0) break;
            const name: [*:0]const u8 = @ptrCast(&e.name);
            if (name[0] != '.') n += 1;
            off += e.reclen;
        }
    }
    return n;
}

test "a granted claim carries the size the daemon granted, and the descriptor by SCM_RIGHTS" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("lend");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "granted", .{ .payload = "{\"ok\": true, \"base\": 10739515392, \"size\": 4194304}", .lent = lent });
    defer d.stop();
    const s = try slab(d.path, "kindlingai_1", 1 << 20);
    defer _ = linux.close(s.sock);
    defer _ = linux.close(s.fd);
    try t.expectEqual(@as(usize, 4 << 20), s.size); // the grant, not the 1 MiB ask
    var marker: [4]u8 = undefined;
    const n = linux.read(s.fd, &marker, marker.len);
    try t.expectEqual(linux.E.SUCCESS, linux.errno(n));
    try t.expectEqualStrings("lend", &marker); // the daemon's own file, read through the received descriptor
    d.stop();
    var parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, d.observed.request[0..d.observed.request_len], .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try t.expectEqualStrings("kindlingai_1", obj.get("key").?.string);
    try t.expectEqualStrings("alloc", obj.get("op").?.string);
    try t.expectEqual(@as(i64, 2 << 20), obj.get("size").?.integer); // the 1 MiB ask, rounded up to a whole slice
}

test "a refused claim is DaemonRefused" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    const reply = "{\"ok\": false, \"error\": \"not enough free carveout\", \"base\": 1, \"size\": 2, \"free\": 3, \"largest\": 4}";
    try d.start(&tmp, "refused", .{ .payload = reply });
    defer d.stop();
    try t.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 1 << 20));
}

test "a reply that is not the expected JSON is DaemonRefused" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("json");
    defer _ = linux.close(lent);
    // Each carries a descriptor, so the refusal is the JSON's and not a missing fd's.
    const payloads = [_][]const u8{
        "not json at all",
        "[1, 2, 3]",
        "{\"base\": 1, \"size\": 2097152}",
        "{\"ok\": true, \"size\": \"2097152\"}",
        "{\"ok\": true, \"size\": -1}",
        "{\"ok\": true, \"size\": 2097152.5}",
        "{\"ok\": \"yes\", \"size\": 2097152}",
        "{\"ok\": false, \"size\": 2097152}",
        "{\"ok\": false, \"error\": \"unknown op\"}",
    };
    for (payloads, 0..) |payload, i| {
        var name: [8]u8 = undefined;
        var d: Daemon = .{};
        try d.start(&tmp, try std.fmt.bufPrint(&name, "m{d}", .{i}), .{ .payload = payload, .lent = lent });
        defer d.stop();
        try t.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 1 << 20));
        d.stop();
    }
}

test "a zero-byte grant is SliceTooSmall even when nothing was asked for" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("none");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "zero", .{ .payload = "{\"ok\": true, \"size\": 0}", .lent = lent });
    defer d.stop();
    try t.expectError(error.SliceTooSmall, slab(d.path, "kindlingai_1", 0));
}

test "a reply longer than the client's buffer is DaemonRefused, not a truncated parse" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("long");
    defer _ = linux.close(lent);
    // A complete grant followed by whitespace: a truncated read still parses, so only the truncation check refuses it.
    var payload: [reply_max + 64]u8 = undefined;
    const head = "{\"ok\": true, \"size\": 2097152}";
    @memcpy(payload[0..head.len], head);
    @memset(payload[head.len..], ' ');
    var d: Daemon = .{};
    try d.start(&tmp, "long", .{ .payload = &payload, .lent = lent });
    defer d.stop();
    try t.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 1 << 20));
}

test "a refusal that carries a size is refused, not read as a grant" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    defer d.stop();
    const lent = try memfd("refused");
    defer _ = linux.close(lent);
    // "not enough free carveout" carries a size and a descriptor, so trusting the size alone would read it as a grant.
    try d.start(&tmp, "refused-size", .{
        .payload = "{\"ok\": false, \"error\": \"not enough free carveout\", \"base\": 10739515392, \"size\": 2145386496, \"free\": 0, \"largest\": 0}",
        .lent = lent,
    });
    try std.testing.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 4 << 20));
}

test "an ask between two slices is rounded up, so the grant is never less than planned" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    defer d.stop();
    const lent = try memfd("round-up");
    defer _ = linux.close(lent);
    try d.start(&tmp, "round-up", .{ .payload = "{\"ok\": true, \"base\": 10739515392, \"size\": 4194304}", .lent = lent });
    const ask = (3 << 20) + 1;
    const s = try slab(d.path, "kindlingai_1", ask);
    defer _ = linux.close(s.fd);
    defer _ = linux.close(s.sock);
    try std.testing.expect(s.size >= ask); // never quietly less than asked
    d.stop();
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, d.observed.request[0..d.observed.request_len], .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(i64, 4194304), parsed.value.object.get("size").?.integer); // rounded up on the wire
}

test "a refusal followed by the daemon closing the connection is still the cause, not a broken pipe" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    defer d.stop();
    // The real daemon refuses and then closes: the reply arrives with an EOF behind it, and the reply is the cause.
    try d.start(&tmp, "close-after-refusal", .{
        .payload = "{\"ok\": false, \"error\": \"not enough free carveout\", \"base\": 10739515392, \"size\": 2145386496, \"free\": 0, \"largest\": 0}",
        .close_after_reply = true,
    });
    try std.testing.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 4 << 20));
}

test "a grant smaller than the ask is SliceTooSmall, not a smaller pool" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("half");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "short", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = lent });
    defer d.stop();
    try t.expectError(error.SliceTooSmall, slab(d.path, "kindlingai_1", 4 << 20));
}

test "a reply with no descriptor is DaemonRefused" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    try d.start(&tmp, "nofd", .{ .payload = "{\"ok\": true, \"base\": 10739515392, \"size\": 2097152}" });
    defer d.stop();
    try t.expectError(error.DaemonRefused, slab(d.path, "kindlingai_1", 1 << 20));
}

test "a missing socket is NoDaemon" {
    const t = std.testing;
    try t.expectError(error.NoDaemon, slab(".zig-cache/tmp/absent/d.sock", "kindlingai_1", 1 << 20));
}

test "the daemon sees the socket close as the release" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("held");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "release", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = lent, .wait_close = true });
    defer d.stop();
    const s = try slab(d.path, "kindlingai_1", 1 << 20);
    _ = linux.close(s.fd); // the mapping would hold its own reference; the socket is the claim
    _ = linux.close(s.sock);
    d.stop();
    try t.expect(d.observed.failed == null);
    try t.expect(d.observed.closed);
}

/// A driver whose only reachable call fails: a claim that gets past the protocol half stops here, not in CUDA.
const NoCuda = struct {
    var calls: u32 = 0;
    fn ctxGetDevice(_: *abi.Device) callconv(.c) abi.Result {
        calls += 1;
        return 201; // CUDA_ERROR_INVALID_CONTEXT
    }
    fn driver() Driver {
        var d: Driver = undefined;
        d.api.cuCtxGetDevice = ctxGetDevice;
        return d;
    }
};

test "claim refuses a missing socket, a refused claim and a short grant before it touches CUDA" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const d = NoCuda.driver();
    NoCuda.calls = 0;
    try t.expectError(error.NoDaemon, claim(&d, ".zig-cache/tmp/absent/d.sock", "kindlingai_1", 1 << 20));
    const lent = try memfd("half");
    defer _ = linux.close(lent);
    var refused: Daemon = .{};
    try refused.start(&tmp, "claim-refused", .{ .payload = "{\"ok\": false, \"error\": \"unknown op\"}" });
    defer refused.stop();
    try t.expectError(error.DaemonRefused, claim(&d, refused.path, "kindlingai_1", 1 << 20));
    var short: Daemon = .{};
    try short.start(&tmp, "claim-short", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = lent });
    defer short.stop();
    try t.expectError(error.SliceTooSmall, claim(&d, short.path, "kindlingai_1", 4 << 20));
    try t.expectEqual(@as(u32, 0), NoCuda.calls); // every refusal above happened before any driver call
}

test "a claim the protocol half accepts reaches the driver, and a driver that cannot answer is VmmUnavailable" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const lent = try memfd("vmm");
    defer _ = linux.close(lent);
    var d: Daemon = .{};
    try d.start(&tmp, "vmm", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = lent });
    defer d.stop();
    const drv = NoCuda.driver();
    NoCuda.calls = 0;
    try t.expectError(error.VmmUnavailable, claim(&drv, d.path, "kindlingai_1", 1 << 20));
    try t.expectEqual(@as(u32, 1), NoCuda.calls);
}

test "a key file's first line is the key the daemon is sent" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "key", .data = "per-box-key\nnot the key\n" });
    var buf: [128]u8 = undefined;
    const key_file = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}/key", .{tmp.sub_path});
    var d: Daemon = .{};
    try d.start(&tmp, "keyed", .{ .payload = "{\"ok\": false, \"error\": \"unknown op\"}" });
    defer d.stop();
    const drv = NoCuda.driver();
    const want: carveout.Request = .{ .backend = .daemon, .bytes = lend.gran, .socket = d.path, .key = "the-default", .key_file = key_file };
    try t.expectError(error.DaemonRefused, carveout.Carveout.from(&drv, want));
    d.stop();
    var parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, d.observed.request[0..d.observed.request_len], .{});
    defer parsed.deinit();
    try t.expectEqualStrings("per-box-key", parsed.value.object.get("key").?.string);
}

test "a second descriptor in the reply is closed, not leaked" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const first = try memfd("one!");
    defer _ = linux.close(first);
    const second = try memfd("two!");
    defer _ = linux.close(second);
    const baseline = openFds(); // before the daemon exists, so its listener is not in the count
    var d: Daemon = .{};
    // wait_close holds the daemon's connection open, so the count below is not racing its close.
    try d.start(&tmp, "two", .{ .payload = "{\"ok\": true, \"size\": 2097152}", .lent = first, .extra = second, .wait_close = true });
    defer d.stop();
    const s = try slab(d.path, "kindlingai_1", 1 << 20);
    // the listener, the client's socket, the daemon's connection and the one descriptor kept: the second was closed
    try t.expectEqual(baseline + 4, openFds());
    _ = linux.close(s.sock);
    _ = linux.close(s.fd);
    d.stop();
    try t.expectEqual(baseline, openFds());
}

test "a daemon that accepts and never answers is DaemonSilent, and the client lets go of its socket" {
    const t = std.testing;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var d: Daemon = .{};
    try d.start(&tmp, "silent", .{ .payload = "", .silent = true });
    defer d.stop();
    const before = openFds();
    const saved = lend.wait_s;
    lend.wait_s = 1;
    defer lend.wait_s = saved;
    try t.expectError(error.DaemonSilent, slab(d.path, "kindlingai_1", 1 << 20));
    d.thread.?.join(); // the daemon's own connection ends once it saw the client close
    d.thread = null;
    try t.expect(d.observed.closed);
    try t.expectEqual(before, openFds());
}

test "a request to a daemon that has gone away is DaemonRefused" {
    const t = std.testing;
    var fds: [2]linux.fd_t = undefined;
    try t.expectEqual(linux.E.SUCCESS, linux.errno(linux.socketpair(linux.AF.UNIX, linux.SOCK.SEQPACKET, 0, &fds)));
    _ = linux.close(fds[1]);
    defer _ = linux.close(fds[0]);
    try t.expectError(error.DaemonRefused, lend.send(fds[0], "kindlingai_1", 2 << 20));
}
