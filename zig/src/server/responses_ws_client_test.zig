//! A small masked-frame WebSocket client and HTTP helpers for the Responses socket tests.
const std = @import("std");
const json = @import("json");
const ws_frame = @import("ws_frame.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

pub const Frame = struct { opcode: ws_frame.Opcode, payload: []u8 };

/// One client socket; it lives where ``open`` put it (its reader and writer point at its buffers).
pub const Client = struct {
    io: std.Io,
    stream: std.Io.net.Stream,
    rbuf: [16384]u8 = undefined,
    wbuf: [16384]u8 = undefined,
    reader: std.Io.net.Stream.Reader = undefined,
    writer: std.Io.net.Stream.Writer = undefined,
    /// The upgrade reply's status.
    status: u16 = 0,
    /// The close code the server sent (null until a close frame arrives).
    closed: ?u16 = null,

    /// Connects and sends the upgrade with ``key`` as its bearer token (null: none).
    pub fn open(c: *Client, a: Allocator, io: std.Io, port: u16, key: ?[]const u8) !void {
        const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(port) };
        c.* = .{ .io = io, .stream = try addr.connect(io, .{ .mode = .stream }) };
        c.reader = .init(c.stream, io, &c.rbuf);
        c.writer = .init(c.stream, io, &c.wbuf);
        const auth = if (key) |k| try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{k}) else "";
        const head = try std.fmt.allocPrint(a, "GET /v1/responses HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" ++
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n{s}\r\n", .{auth});
        try c.writer.interface.writeAll(head);
        try c.writer.interface.flush();
        const first = try c.reader.interface.takeDelimiterInclusive('\n');
        c.status = try std.fmt.parseInt(u16, first[9..12], 10);
        while (true) {
            const line = try c.reader.interface.takeDelimiterInclusive('\n');
            if (std.mem.eql(u8, line, "\r\n")) break;
        }
    }

    pub fn close(c: *Client) void {
        c.stream.close(c.io);
    }

    /// One frame, masked unless ``masked`` is false (a protocol error the server must refuse).
    pub fn sendFrame(c: *Client, a: Allocator, opcode: ws_frame.Opcode, payload: []const u8, masked: bool) !void {
        const key: [4]u8 = .{ 0x11, 0x22, 0x33, 0x44 };
        var head: [ws_frame.max_head]u8 = undefined;
        const h = ws_frame.Header.encode(&head, true, opcode, payload.len, if (masked) key else null);
        const body = try a.dupe(u8, payload);
        if (masked) ws_frame.unmask(body, key, 0);
        try c.writer.interface.writeAll(h);
        try c.writer.interface.writeAll(body);
        try c.writer.interface.flush();
    }

    pub fn send(c: *Client, a: Allocator, text: []const u8) !void {
        return c.sendFrame(a, .text, text, true);
    }

    /// The next server frame, whole.
    pub fn frame(c: *Client, a: Allocator) !Frame {
        const r = &c.reader.interface;
        const two = try r.takeArray(2);
        const opcode: ws_frame.Opcode = @fromBackingInt(@as(u4, @truncate(two[0])));
        var len: u64 = two[1] & 0x7f;
        if (len == 126) len = std.mem.readInt(u16, try r.takeArray(2), .big) else if (len == 127) len = std.mem.readInt(u64, try r.takeArray(8), .big);
        const payload = try a.alloc(u8, @intCast(len));
        try r.readSliceAll(payload);
        if (opcode == .close and payload.len >= 2) c.closed = std.mem.readInt(u16, payload[0..2], .big);
        return .{ .opcode = opcode, .payload = payload };
    }

    /// The next event object; pings are skipped, and a close frame is ``error.Closed``.
    pub fn event(c: *Client, a: Allocator) !Value {
        while (true) {
            const f = try c.frame(a);
            switch (f.opcode) {
                .text => return (try json.parse(a, f.payload)).ok,
                .close => return error.Closed,
                else => {},
            }
        }
    }

    /// Events through the turn's last: completed, incomplete, failed, an ``error`` or a steer refusal.
    pub fn turn(c: *Client, a: Allocator, text: []const u8) ![]Value {
        try c.send(a, text);
        return c.until(a);
    }

    pub fn until(c: *Client, a: Allocator) ![]Value {
        var out: std.ArrayList(Value) = .empty;
        while (true) {
            const e = try c.event(a);
            try out.append(a, e);
            const kind = e.strField("type").?;
            for ([_][]const u8{ "response.completed", "response.incomplete", "response.failed", "error", "response.steer.failed" }) |end| {
                if (std.mem.eql(u8, kind, end)) return out.items;
            }
        }
    }

    /// Reads frames until the server's close and returns its code.
    pub fn closeCode(c: *Client, a: Allocator) !?u16 {
        while (true) {
            const f = c.frame(a) catch return c.closed;
            if (f.opcode == .close) return c.closed;
        }
    }
};

/// The last event of a turn.
pub fn last(events: []Value) Value {
    return events[events.len - 1];
}

/// The final response a turn's last event carries.
pub fn final(events: []Value) Value {
    return last(events).get("response").?;
}

/// An error event's ``error.code``.
pub fn code(events: []Value) []const u8 {
    return last(events).get("error").?.strField("code") orelse "";
}

/// An integer field of a final response's usage.
pub fn usage(events: []Value, field: []const u8) i64 {
    return final(events).get("usage").?.get(field).?.int64().?;
}

pub const Reply = struct { status: u16, body: []const u8 };

/// One HTTP request with ``key`` as its bearer token; the connection closes after it.
pub fn http(a: Allocator, io: std.Io, port: u16, method: []const u8, path: []const u8, key: ?[]const u8, body: []const u8) !Reply {
    const addr: std.Io.net.IpAddress = .{ .ip4 = .loopback(port) };
    const stream = try addr.connect(io, .{ .mode = .stream });
    defer stream.close(io);
    const auth = if (key) |k| try std.fmt.allocPrint(a, "Authorization: Bearer {s}\r\n", .{k}) else "";
    const head = try std.fmt.allocPrint(a, "{s} {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n{s}Content-Length: {d}\r\nConnection: close\r\n\r\n{s}", .{ method, path, auth, body.len, body });
    var wbuf: [4096]u8 = undefined;
    var w = std.Io.net.Stream.Writer.init(stream, io, &wbuf);
    try w.interface.writeAll(head);
    try w.interface.flush();
    var rbuf: [16384]u8 = undefined;
    var r = std.Io.net.Stream.Reader.init(stream, io, &rbuf);
    const text = try r.interface.allocRemaining(a, .unlimited);
    const split = std.mem.indexOf(u8, text, "\r\n\r\n") orelse return error.Head;
    return .{ .status = try std.fmt.parseInt(u16, text[9..12], 10), .body = text[split + 4 ..] };
}

/// The ``data:`` objects of an SSE body, in order.
pub fn sseEvents(a: Allocator, body: []const u8) ![]Value {
    var out: std.ArrayList(Value) = .empty;
    var lines = std.mem.splitScalar(u8, body, '\n');
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, "data: ")) try out.append(a, (try json.parse(a, line[6..])).ok);
    return out.items;
}

/// A copy of ``v`` without the fields that differ between two runs of one turn: ids, times and runtime stats.
pub fn normalized(a: Allocator, v: Value) !Value {
    switch (v) {
        .array => |items| {
            const out = try a.alloc(Value, items.len);
            for (items, out) |item, *slot| slot.* = try normalized(a, item);
            return .{ .array = out };
        },
        .object => |o| {
            const copy = try json.newObject(a);
            for (o.keys(), o.values()) |k, x| {
                const varying = for ([_][]const u8{ "id", "item_id", "created_at", "completed_at", "tensorfold" }) |name| {
                    if (std.mem.eql(u8, k, name)) break true;
                } else false;
                if (!varying) try copy.put(a, k, try normalized(a, x));
            }
            return .{ .object = copy };
        },
        else => return v,
    }
}
