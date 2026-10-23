//! A kept-alive connection's arena between requests: a long request's memory goes back, the connection stays open.
const std = @import("std");
const posix = @import("sys").posix; // the layer serve.zig installs its stop handlers with
const server = @import("server.zig");
const d = @import("drain_test.zig");
const testing = std.testing;

/// The bytes `child` holds for the server's threads, counted as they allocate, resize and free.
const Live = struct {
    child: std.mem.Allocator,
    bytes: std.atomic.Value(isize) = .init(0),

    fn allocator(l: *Live) std.mem.Allocator {
        return .{ .ptr = l, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn add(ctx: *anyopaque, from: usize, to: usize) void {
        const l: *Live = @ptrCast(@alignCast(ctx));
        _ = l.bytes.fetchAdd(@as(isize, @intCast(to)) - @as(isize, @intCast(from)), .acq_rel);
    }

    fn alloc(ctx: *anyopaque, n: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const p = @as(*Live, @ptrCast(@alignCast(ctx))).child.rawAlloc(n, al, ra) orelse return null;
        add(ctx, 0, n);
        return p;
    }

    fn resize(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, n: usize, ra: usize) bool {
        if (!@as(*Live, @ptrCast(@alignCast(ctx))).child.rawResize(m, al, n, ra)) return false;
        add(ctx, m.len, n);
        return true;
    }

    fn remap(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const p = @as(*Live, @ptrCast(@alignCast(ctx))).child.rawRemap(m, al, n, ra) orelse return null;
        add(ctx, m.len, n);
        return p;
    }

    fn free(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, ra: usize) void {
        @as(*Live, @ptrCast(@alignCast(ctx))).child.rawFree(m, al, ra);
        add(ctx, m.len, 0);
    }
};

/// A non-streaming count_tokens request whose one message holds `size` bytes; its reply is read whole, kept alive.
fn countTokens(r: *std.Io.Reader, w: *std.Io.Writer, size: usize) !void {
    const open = "{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"";
    const close = "\"}]}";
    const head = "POST /v1/messages/count_tokens HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n";
    try w.print("{s}Content-Length: {d}\r\n\r\n{s}", .{ head, open.len + size + close.len, open });
    try w.splatByteAll('x', size);
    try w.writeAll(close);
    try w.flush();
    try testing.expect(std.mem.startsWith(u8, try r.takeDelimiterInclusive('\n'), "HTTP/1.1 200 "));
    var length: usize = 0;
    while (true) {
        const line = std.mem.trimEnd(u8, try r.takeDelimiterInclusive('\n'), "\r\n");
        if (line.len == 0) break;
        try testing.expect(!std.ascii.eqlIgnoreCase(line, "Connection: close"));
        if (std.ascii.startsWithIgnoreCase(line, "Content-Length: ")) length = try std.fmt.parseInt(usize, line[16..], 10);
    }
    try testing.expect(std.mem.indexOf(u8, try r.take(length), "\"input_tokens\"") != null);
}

test "Connection arena: after two long non-streaming requests a kept-alive connection holds at most the limit" {
    const io = testing.io;
    var live: Live = .{ .child = testing.allocator };
    var tt: d.TestText = .{};
    var env = std.process.Environ.Map.init(testing.allocator);
    defer env.deinit();
    try env.put("TENSORFOLD_NO_LIVE", "1");
    var slow: d.Slow = .{ .gpa = testing.allocator, .io = io };
    var x: d.Served = .{};
    const t = try d.start(live.allocator(), io, &env, &slow, &tt, &x);
    const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(d.listen_port.load(.acquire)) };
    const stream = try addr.connect(io, .{ .mode = .stream });
    var wbuf: [4096]u8 = undefined;
    var w = std.Io.net.Stream.Writer.init(stream, io, &wbuf);
    var rbuf: [4096]u8 = undefined;
    var r = std.Io.net.Stream.Reader.init(stream, io, &rbuf);
    const held = blk: {
        defer {
            stream.close(io);
            posix.raise(.TERM) catch {};
            t.join();
        }
        // two short requests: the second runs in what the first left, so `before` is a connection at rest
        try countTokens(&r.interface, &w.interface, 16);
        try countTokens(&r.interface, &w.interface, 16);
        const before = live.bytes.load(.acquire);
        try countTokens(&r.interface, &w.interface, 8 << 20);
        try countTokens(&r.interface, &w.interface, 8 << 20);
        // the connection's arena is reset before it reads the next request, so this one starts after that reset
        try countTokens(&r.interface, &w.interface, 16);
        break :blk live.bytes.load(.acquire) - before;
    };
    try testing.expectEqual(@as(u8, 0), x.code);
    if (held > server.connection_arena_limit) std.debug.print("the idle connection held {d} bytes more\n", .{held});
    try testing.expect(held <= server.connection_arena_limit);
}
