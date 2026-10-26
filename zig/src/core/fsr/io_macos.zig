//! A bounded asynchronous read group over uncached descriptors, drained before any slot can be published.
const std = @import("std");
const builtin = @import("builtin");
const reader = @import("reader.zig");
const platform = @import("sys");

pub fn readMany(files: anytype, requests: []const reader.Request, stats: *reader.Stats) !void {
    if (comptime builtin.os.tag != .macos) return error.FsrNeedsMacOS;
    return group(Darwin, files, requests, stats);
}

pub const Darwin = struct {
    pub const Event = extern struct { notify: c_int, signal: c_int, value: ?*anyopaque, function: ?*const anyopaque, attributes: ?*anyopaque };
    pub const Control = extern struct { fd: c_int, offset: i64, buffer: ?*volatile anyopaque, bytes: usize, priority: c_int, event: Event, opcode: c_int };
    extern "c" fn aio_read(*Control) c_int;
    extern "c" fn aio_error(*const Control) c_int;
    extern "c" fn aio_return(*Control) isize;
    extern "c" fn aio_suspend([*]const ?*const Control, c_int, ?*const std.c.timespec) c_int;
    pub fn prepare(c: *Control, fd: c_int, request: reader.Request) void {
        c.* = std.mem.zeroes(Control);
        c.fd = fd;
        c.offset = @intCast(request.span.offset);
        c.buffer = request.destination.ptr;
        c.bytes = request.destination.len;
    }
    pub fn submit(c: *Control) bool {
        return aio_read(c) == 0;
    }
    pub fn status(c: *Control) c_int {
        return aio_error(c);
    }
    pub fn result(c: *Control) isize {
        return aio_return(c);
    }
    pub fn wait(c: *Control) void {
        var list = [_]?*const Control{c};
        _ = aio_suspend(&list, 1, null);
    }
};

pub fn group(comptime Aio: type, files: anytype, requests: []const reader.Request, stats: *reader.Stats) !void {
    if (requests.len > reader.width) return error.InvalidExpertComponents;
    for (requests) |request| {
        if (request.span.file >= files.len) return error.NoFile;
        if (request.span.length != request.destination.len or request.span.offset > std.math.maxInt(i64))
            return error.InvalidTensorSpan;
    }
    var controls: [reader.width]Aio.Control = undefined;
    var queued: usize = 0;
    for (requests, 0..) |request, i| {
        const c = &controls[i];
        Aio.prepare(c, files[request.span.file].fd, request);
        if (!Aio.submit(c)) break;
        stats.calls += 1;
        queued += 1;
    }
    var failure: ?anyerror = null;
    for (requests[0..queued], 0..) |request, i| {
        const c = &controls[i];
        var status = Aio.status(c);
        while (status == @backingInt(platform.posix.E.INPROGRESS)) {
            Aio.wait(c);
            status = Aio.status(c);
        }
        const got = Aio.result(c);
        if (status != 0 or got < 0) {
            failure = error.ReadFailed;
            continue;
        }
        const n: usize = @intCast(got);
        if (n > request.destination.len) {
            failure = error.InvalidReadCount;
            continue;
        }
        stats.bytes += n;
        if (n != request.destination.len) {
            finish(files, request, n, stats) catch |err| {
                failure = failure orelse err;
            };
        }
    }
    if (failure) |err| return err;
    // Resource pressure or unsupported asynchronous reads fall back only after every submitted request drains.
    for (requests[queued..]) |request| try finish(files, request, 0, stats);
}

fn finish(files: anytype, request: reader.Request, start: usize, stats: *reader.Stats) !void {
    var at = start;
    while (at < request.destination.len) {
        const n = try files[request.span.file].readAt(0, request.destination[at..], request.span.offset + at);
        stats.calls += 1;
        if (n == 0) return error.ShortRead;
        if (n > request.destination.len - at) return error.InvalidReadCount;
        stats.bytes += n;
        at += n;
    }
}
