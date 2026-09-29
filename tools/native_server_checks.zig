const std = @import("std");
const long_request = "{\"prompt\":\"Count upwards, one number per line.\",\"max_tokens\":200000,\"ignore_eos\":true,\"temperature\":0,\"stream\":true}";

fn connect(io: std.Io, port: u16) !std.Io.net.Stream {
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    return address.connect(io, .{ .mode = .stream });
}

fn headers(io: std.Io, socket: std.Io.net.Stream, length: usize) !void {
    var buffer: [2048]u8 = undefined;
    var writer = socket.writer(io, &buffer);
    try writer.interface.print("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{length});
    try writer.interface.flush();
}

fn write(io: std.Io, socket: std.Io.net.Stream, bytes: []const u8) !void {
    var buffer: [2048]u8 = undefined;
    var writer = socket.writer(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}

fn post(io: std.Io, port: u16, body: []const u8) !std.Io.net.Stream {
    const socket = try connect(io, port);
    errdefer socket.close(io);
    try headers(io, socket, body.len);
    try write(io, socket, body);
    return socket;
}

fn readAll(a: std.mem.Allocator, io: std.Io, socket: std.Io.net.Stream) ![]u8 {
    var buffer: [8192]u8 = undefined;
    var reader = socket.reader(io, &buffer);
    return reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
}

fn assertCancelled(bytes: []const u8) !void {
    if (std.mem.indexOf(u8, bytes, "\"finish_reason\":\"length\"") != null or std.mem.indexOf(u8, bytes, "\"finish_reason\":\"stop\"") != null) return error.CancelledRequestCompleted;
    if (bytes.len != 0 and std.mem.indexOf(u8, bytes, "RequestTimedOut") == null and std.mem.indexOf(u8, bytes, "ServerStopping") == null and std.mem.indexOf(u8, bytes, "data: ") == null) return error.MissingCancellation;
}

const Scenario = struct {
    init: std.process.Init,
    child: std.process.Child,
    idle: bool,

    fn run(s: *Scenario) anyerror!void {
        const io = s.init.io;
        const a = s.init.arena.allocator();
        var stderr_buffer: [8192]u8 = undefined;
        var stderr = s.child.stderr.?.reader(io, &stderr_buffer);
        const prefix = "Native inference listening at http://127.0.0.1:";
        const port = while (true) {
            const line = try stderr.interface.takeSentinel('\n');
            if (std.mem.indexOf(u8, line, prefix)) |start| {
                const value = line[start + prefix.len ..];
                const end = std.mem.indexOfScalar(u8, value, ' ') orelse return error.InvalidListenAddress;
                break try std.fmt.parseInt(u16, value[0..end], 10);
            }
        };
        if (s.idle) {
            try std.posix.kill(s.child.id.?, .INT);
            if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
            std.debug.print("PASS: SIGINT exits an idle server cleanly\n", .{});
            return;
        }

        const slow_head = try connect(io, port);
        defer slow_head.close(io);
        try write(io, slow_head, "POST /v1/completions HTTP/1.1\r\n");
        const slow_body = try connect(io, port);
        defer slow_body.close(io);
        try headers(io, slow_body, 9999);
        try write(io, slow_body, "{");
        const queued = try connect(io, port);
        defer queued.close(io);
        try headers(io, queued, long_request.len);
        try std.Io.sleep(io, .fromMilliseconds(250), .awake);
        const active = try post(io, port, long_request);
        defer active.close(io);
        try std.Io.sleep(io, .fromMilliseconds(100), .awake);
        try write(io, queued, long_request);
        try assertCancelled(try readAll(a, io, queued));
        try assertCancelled(try readAll(a, io, active));
        _ = try readAll(a, io, slow_head);
        _ = try readAll(a, io, slow_body);
        // A GPU operation already in flight may finish after the socket deadline.
        var recovered = false;
        for (0..16) |_| {
            const recovery = try post(io, port, "{\"prompt\":\"Hello\",\"max_tokens\":0}");
            defer recovery.close(io);
            const response = try readAll(a, io, recovery);
            if (std.mem.indexOf(u8, response, "200 OK") != null and std.mem.indexOf(u8, response, "\"completion_tokens\":0") != null) {
                recovered = true;
                break;
            }
            try assertCancelled(response);
        }
        if (!recovered) return error.ServerDidNotRecover;
        std.debug.print("PASS: deadlines stop active/queued inference and partial requests; next request succeeds\n", .{});

        const generating = try post(io, port, long_request);
        defer generating.close(io);
        var buffer: [8192]u8 = undefined;
        var reader = generating.reader(io, &buffer);
        while (true) {
            const line = try reader.interface.takeSentinel('\n');
            if (std.mem.startsWith(u8, line, "data: ")) break;
        }
        const waiting = try post(io, port, long_request);
        defer waiting.close(io);
        const stalled = try connect(io, port);
        defer stalled.close(io);
        try write(io, stalled, "POST /v1/completions HTTP/1.1\r\n");
        try std.Io.sleep(io, .fromMilliseconds(100), .awake);
        try std.posix.kill(s.child.id.?, .TERM);
        try assertCancelled(try reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024)));
        try assertCancelled(try readAll(a, io, waiting));
        _ = try readAll(a, io, stalled);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        std.debug.print("PASS: SIGTERM cancels active/queued inference, releases stalled clients and exits cleanly\n", .{});
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.ExpectedExecutableAndModel;
    for ([_]bool{ false, true }) |idle| {
        var scenario = Scenario{ .init = init, .idle = idle, .child = try std.process.spawn(init.io, .{ .argv = &.{ args[1], "serve", args[2], "--port", "0", "--request-timeout-seconds", if (idle) "0" else "2", "--shutdown-grace-seconds", "1", "--no-thinking" }, .stderr = .pipe }) };
        defer scenario.child.kill(init.io);
        const Event = union(enum) { done: anyerror!void, timeout: std.Io.Cancelable!void };
        var events: [2]Event = undefined;
        var select = std.Io.Select(Event).init(init.io, &events);
        defer select.cancelDiscard();
        try select.concurrent(.done, Scenario.run, .{&scenario});
        try select.concurrent(.timeout, std.Io.sleep, .{ init.io, std.Io.Duration.fromSeconds(90), .awake });
        switch (try select.await()) {
            .done => |result| try result,
            .timeout => return error.ServerLifecycleCheckTimedOut,
        }
    }
}
