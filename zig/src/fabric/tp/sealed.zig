//! Sealed records under the socket helpers: once a proven connection is added, `sock` seals every byte it moves on it.
const std = @import("std");
const c = std.c;
const posix = std.posix;
const log = @import("log.zig");
const sock = @import("sock.zig");
const auth = @import("sys").rank_auth;
const seal = auth.seal;

/// Connections sealed at once: rank 0 of the largest group holds two for each other rank.
const max_lines = 256;

/// One connection's seal. Senders take turns; one thread reads, as it would the socket itself.
pub const Line = struct {
    seal: seal.Seal,
    arriving: seal.Arriving = .{},
    sending: std.atomic.Value(bool) = .init(false),
    ended: std.atomic.Value(bool) = .init(false),

    /// Ends the line at both ends: nothing after a record that was cut short or did not open would open either.
    fn end(l: *Line, fd: sock.Fd) void {
        l.ended.store(true, .release);
        _ = c.shutdown(fd, posix.SHUT.RDWR);
    }

    /// `bytes` in sealed records, each whole on the socket before `deadline` (null: no limit); `flags` join each send.
    pub fn send(l: *Line, fd: sock.Fd, bytes: []const u8, deadline: ?u64, flags: u32) sock.Error!void {
        while (l.sending.swap(true, .acquire)) std.atomic.spinLoopHint();
        defer l.sending.store(false, .release);
        if (l.ended.load(.acquire)) return error.Closed;
        var wire: [seal.max_record]u8 = undefined;
        var at: usize = 0;
        while (at < bytes.len) {
            const n = @min(bytes.len - at, seal.max_plain);
            const len = l.seal.seal(bytes[at..][0..n], &wire);
            sendRaw(fd, wire[0..len], deadline, flags) catch |e| {
                l.end(fd);
                return e;
            };
            at += n;
        }
    }

    /// One record of `bytes` if the socket takes it now. False: nothing of it left, or the line has ended.
    pub fn trySend(l: *Line, fd: sock.Fd, bytes: []const u8) bool {
        if (bytes.len == 0) return true;
        if (bytes.len > seal.max_plain or l.sending.swap(true, .acquire)) return false;
        defer l.sending.store(false, .release);
        if (l.ended.load(.acquire)) return false;
        var wire: [seal.max_record]u8 = undefined;
        const len = l.seal.seal(bytes, &wire);
        const n = c.send(fd, &wire, len, posix.MSG.NOSIGNAL | posix.MSG.DONTWAIT);
        if (n == @as(isize, @intCast(len))) return true;
        if (n <= 0) {
            l.seal.sent -= 1; // no byte of it left, so its place in the count goes to the next record
            return false;
        }
        // a part of it left: the rest follows within a second, or the peer could open nothing after it
        sendRaw(fd, wire[@intCast(n)..len], sock.nowNs() + std.time.ns_per_s, 0) catch {
            l.end(fd);
            return false;
        };
        return true;
    }

    /// Fills `out` from records arriving before `deadline` (null: no limit); one that does not open ends the line.
    pub fn recv(l: *Line, fd: sock.Fd, out: []u8, deadline: ?u64) sock.Error!void {
        var at: usize = 0;
        while (true) {
            at += l.seal.take(out[at..]);
            if (at == out.len) return;
            if (l.ended.load(.acquire)) return error.Closed;
            const want = l.arriving.want() catch return l.refuse(fd);
            if (deadline) |d| try sock.waitReadable(fd, d);
            const n = c.recv(fd, want.ptr, want.len, 0);
            if (n <= 0) {
                if (n < 0 and posix.errno(n) == .INTR) continue;
                return error.Closed;
            }
            l.arriving.fed(&l.seal, @intCast(n)) catch return l.refuse(fd);
        }
    }

    fn refuse(l: *Line, fd: sock.Fd) sock.Error {
        log.warn("tp: a record on a rank link did not open under the group's key, so the link is closed", .{});
        l.end(fd);
        return error.Closed;
    }
};

fn sendRaw(fd: sock.Fd, bytes: []const u8, deadline: ?u64, flags: u32) sock.Error!void {
    const wait: u32 = if (deadline == null) 0 else posix.MSG.DONTWAIT;
    var at: usize = 0;
    while (at < bytes.len) {
        if (deadline) |d| try sock.waitWritable(fd, d);
        const n = c.send(fd, bytes[at..].ptr, bytes.len - at, posix.MSG.NOSIGNAL | wait | flags);
        if (n < 0 and (posix.errno(n) == .INTR or (deadline != null and posix.errno(n) == .AGAIN))) continue;
        if (n <= 0) return error.Closed;
        at += @intCast(n);
    }
}

const Slot = struct { fd: std.atomic.Value(sock.Fd) = .init(-1), line: std.atomic.Value(?*Line) = .init(null) };
var slots: [max_lines]Slot = @splat(.{});
/// Slots ever taken: a group with no key adds none, and its sockets are looked up in no slot.
var used: std.atomic.Value(usize) = .init(0);
var changing: std.atomic.Value(bool) = .init(false);

/// `fd`'s seal, or null for a socket that carries its bytes as they are.
pub fn of(fd: sock.Fd) ?*Line {
    if (fd < 0) return null;
    for (slots[0..used.load(.acquire)]) |*s| if (s.fd.load(.acquire) == fd) return s.line.load(.acquire);
    return null;
}

/// From here on, `sock` sends and receives `fd`'s bytes as records sealed under the key proof's `session`.
pub fn add(fd: sock.Fd, session: *const auth.Key, connector: bool) error{SocketFailed}!void {
    const line = std.heap.page_allocator.create(Line) catch return error.SocketFailed;
    line.* = .{ .seal = .init(session, connector) };
    while (changing.swap(true, .acquire)) std.atomic.spinLoopHint();
    defer changing.store(false, .release);
    forget(fd);
    for (&slots, 0..) |*s, i| if (s.fd.load(.acquire) < 0) {
        s.line.store(line, .release);
        s.fd.store(fd, .release);
        if (i >= used.load(.acquire)) used.store(i + 1, .release);
        return;
    };
    std.heap.page_allocator.destroy(line);
    return error.SocketFailed;
}

/// Forgets `fd`'s seal: the socket closes, or its number now names a new one. No thread may still be using it.
pub fn drop(fd: sock.Fd) void {
    if (fd < 0 or used.load(.acquire) == 0) return;
    while (changing.swap(true, .acquire)) std.atomic.spinLoopHint();
    defer changing.store(false, .release);
    forget(fd);
}

fn forget(fd: sock.Fd) void {
    for (slots[0..used.load(.acquire)]) |*s| if (s.fd.load(.acquire) == fd) {
        s.fd.store(-1, .release);
        const line = s.line.swap(null, .acq_rel) orelse continue;
        std.crypto.secureZero(u8, std.mem.asBytes(line)); // its keys leave memory with it
        std.heap.page_allocator.destroy(line);
    };
}
