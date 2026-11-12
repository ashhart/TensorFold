//! The display-memory lending daemon's client: one claim over its SEQPACKET socket, mapped with CUDA's VMM.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const arena = @import("arena.zig");

pub const Error = error{ NoDaemon, DaemonSilent, DaemonRefused, SliceTooSmall, VmmUnavailable, MapFailed };

/// The daemon's answer before any mapping: the connection that holds the claim, the lent descriptor, the granted bytes.
pub const Slab = struct { sock: linux.fd_t, fd: linux.fd_t, size: usize };

/// A claim mapped into the current context; `fd` is the connection whose close gives the memory back to the daemon.
pub const Claim = struct { fd: linux.fd_t, dev: abi.DevicePtr, size: usize, vmm: arena.Vmm };

/// A request's JSON: a key longer than this cannot be sent, and the daemon would refuse it as unknown anyway.
const request_max = 4096;
/// A reply's JSON: the daemon's own replies are a few hundred bytes.
pub const reply_max = 4096;
/// CMSG_SPACE for two descriptors is 40 bytes, and the daemon sends one.
const control_max = 64;

/// struct sockaddr_un: the family, then the path (108 bytes on Linux).
const SockaddrUn = extern struct { family: linux.sa_family_t = linux.AF.UNIX, path: [path_max]u8 };
/// The bytes a sockaddr_un's path holds, terminator included.
const path_max = 108;

/// Claims `want` bytes from the daemon at `path` and maps the grant for the current device; the socket stays open.
pub fn claim(d: *const Driver, path: []const u8, key: []const u8, want: usize) Error!Claim {
    // Linux alone has the daemon and the driver library the mapping opens by name: elsewhere there is nobody to ask.
    if (comptime builtin.os.tag != .linux) return error.NoDaemon;
    const s = try slab(path, key, want);
    errdefer _ = linux.close(s.sock);
    errdefer _ = linux.close(s.fd);
    var device: abi.Device = 0;
    if (d.api.cuCtxGetDevice(&device) != abi.success) return error.VmmUnavailable;
    var vmm = arena.Vmm.open(d, device, s.size) catch return error.VmmUnavailable;
    errdefer vmm.close();
    const base = vmm.mapSlice(s.fd, s.size) catch return error.MapFailed;
    _ = linux.close(s.fd); // the mapping holds its own reference to the memory
    return .{ .fd = s.sock, .dev = base, .size = s.size, .vmm = vmm };
}

/// Unmaps the slice, then closes the connection: the close is what gives the memory back to the daemon.
pub fn release(c: Claim) void {
    if (comptime builtin.os.tag != .linux) return; // no claim was ever made there
    var vmm = c.vmm; // the mapping is a value here, and its methods take a pointer to it
    vmm.unmapSlice(c.dev, c.size);
    vmm.close();
    _ = linux.close(c.fd);
}

/// A lending daemon lends whole slices of this size; an ask is rounded UP to it, so a grant is never short.
pub const gran: usize = 2 << 20;

/// The protocol half: one alloc request, its reply, and the descriptor that came with it.
pub fn slab(path: []const u8, key: []const u8, want: usize) Error!Slab {
    // Whole slices, rounded up: the daemon rounds up too, so an unrounded ask can pass what is free and be refused.
    if (want == 0) return error.SliceTooSmall;
    const ask = std.mem.alignForward(usize, want, gran);
    const sock = try connect(path);
    errdefer _ = linux.close(sock);
    try send(sock, key, ask);
    const reply = try receive(sock);
    errdefer _ = linux.close(reply.fd);
    if (reply.size < want or reply.size == 0) return error.SliceTooSmall; // less than planned is a refusal
    // The reply's size is the authority. A seekable descriptor's own length is checked too, its offset put back.
    const saved = linux.lseek(reply.fd, 0, linux.SEEK.CUR);
    if (linux.errno(saved) == .SUCCESS) {
        const end = linux.lseek(reply.fd, 0, linux.SEEK.END);
        _ = linux.lseek(reply.fd, @intCast(saved), linux.SEEK.SET);
        if (linux.errno(end) == .SUCCESS and @as(usize, @intCast(end)) < reply.size) return error.SliceTooSmall;
    }
    return .{ .sock = sock, .fd = reply.fd, .size = reply.size };
}

/// A connected SEQPACKET socket: the daemon refuses any other type, and a missing or refused path is no daemon.
fn connect(path: []const u8) Error!linux.fd_t {
    if (path.len == 0 or path.len >= path_max) return error.NoDaemon;
    var addr: SockaddrUn = .{ .path = @splat(0) };
    @memcpy(addr.path[0..path.len], path);
    const rc = linux.socket(linux.AF.UNIX, linux.SOCK.SEQPACKET | linux.SOCK.CLOEXEC, 0);
    if (linux.errno(rc) != .SUCCESS) return error.NoDaemon;
    const fd: linux.fd_t = @intCast(rc);
    errdefer _ = linux.close(fd);
    // A daemon that accepts and never answers must not hang the load: both directions give up after `wait_s`.
    const tv: linux.timeval = .{ .sec = wait_s, .usec = 0 };
    _ = linux.setsockopt(fd, linux.SOL.SOCKET, linux.SO.RCVTIMEO, std.mem.asBytes(&tv).ptr, @sizeOf(linux.timeval));
    _ = linux.setsockopt(fd, linux.SOL.SOCKET, linux.SO.SNDTIMEO, std.mem.asBytes(&tv).ptr, @sizeOf(linux.timeval));
    const len: linux.socklen_t = @intCast(@offsetOf(SockaddrUn, "path") + path.len + 1);
    if (linux.errno(linux.connect(fd, @ptrCast(&addr), len)) != .SUCCESS) return error.NoDaemon;
    return fd;
}

/// Seconds the daemon has to answer; a variable so a host test need not wait this long.
pub var wait_s: isize = 10;

/// One request datagram: the key, the op and the ask, as JSON.
pub fn send(sock: linux.fd_t, key: []const u8, want: usize) Error!void {
    var buf: [request_max]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    const Request = struct { key: []const u8, op: []const u8, size: usize };
    std.json.Stringify.value(Request{ .key = key, .op = "alloc", .size = want }, .{}, &w) catch return error.DaemonRefused;
    const text = w.buffered();
    const rc = linux.sendto(sock, text.ptr, text.len, 0, null, 0);
    if (linux.errno(rc) != .SUCCESS or rc != text.len) return error.DaemonRefused;
}

const Reply = struct { size: usize, fd: linux.fd_t };

/// One reply datagram: its JSON and the descriptor in its ancillary data; a truncated or fd-less reply is refused.
fn receive(sock: linux.fd_t) Error!Reply {
    var buf: [reply_max]u8 = undefined;
    var control: [control_max]u8 align(linux.cmsg_align) = undefined;
    var iov = [_]std.posix.iovec{.{ .base = &buf, .len = buf.len }};
    var msg: linux.msghdr = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = &control, .controllen = control.len, .flags = 0 };
    const rc = linux.recvmsg(sock, &msg, 0);
    if (linux.errno(rc) == .AGAIN) return error.DaemonSilent;
    if (linux.errno(rc) != .SUCCESS) return error.DaemonRefused;
    if (msg.flags & linux.MSG.TRUNC != 0) return error.DaemonRefused;
    const lent = takeFd(control[0..msg.controllen]) orelse return error.DaemonRefused;
    errdefer _ = linux.close(lent);
    const size = grantedSize(buf[0..rc]) orelse return error.DaemonRefused;
    return .{ .size = size, .fd = lent };
}

/// The first descriptor in the ancillary data; any others are closed, so a daemon that sends two leaks nothing.
fn takeFd(control: []const u8) ?linux.fd_t {
    var found: ?linux.fd_t = null;
    var off: usize = 0;
    while (off + @sizeOf(linux.cmsghdr) <= control.len) {
        const h: *const linux.cmsghdr = @ptrCast(@alignCast(control[off..].ptr));
        if (h.len < @sizeOf(linux.cmsghdr) or off + h.len > control.len) break;
        if (h.level == linux.SOL.SOCKET and h.type == linux.SCM.RIGHTS) {
            const data = control[off + @sizeOf(linux.cmsghdr) .. off + h.len];
            var i: usize = 0;
            while (i + @sizeOf(linux.fd_t) <= data.len) : (i += @sizeOf(linux.fd_t)) {
                const fd: linux.fd_t = @as(*const linux.fd_t, @ptrCast(@alignCast(data[i..].ptr))).*;
                if (found == null) found = fd else _ = linux.close(fd);
            }
        }
        off += std.mem.alignForward(usize, h.len, linux.cmsg_align);
    }
    return found;
}

/// The granted size of a reply: `ok` true and an integer `size`; null for a refusal or anything else.
fn grantedSize(text: []const u8) ?usize {
    // A parsed object costs several times the text it came from, so a datagram of the largest size still fits.
    var scratch: [8 * reply_max]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&scratch);
    const doc = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), text, .{}) catch return null;
    const obj = switch (doc) {
        .object => |o| o,
        else => return null,
    };
    const ok = obj.get("ok") orelse return null;
    if (ok != .bool or !ok.bool) return null;
    const size = obj.get("size") orelse return null;
    const n = switch (size) {
        .integer => |i| i,
        else => return null,
    };
    if (n < 0) return null;
    return std.math.cast(usize, n);
}

test {
    if (builtin.os.tag == .linux) _ = @import("carveout_lend_test.zig");
}
