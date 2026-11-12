//! RFC 6455 frames: incremental header parsing, payload unmasking, message assembly and unmasked server frames.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Conn = @import("http_conn.zig").Conn;
const http_body = @import("http_body.zig");

pub const Opcode = enum(u4) {
    continuation = 0,
    text = 1,
    binary = 2,
    close = 8,
    ping = 9,
    pong = 10,
    _,

    pub fn control(op: Opcode) bool {
        return @backingInt(op) >= 8;
    }
};

/// A frame error names the close code that answers it.
pub const Error = error{ ProtocolError, UnsupportedData, InvalidPayload, MessageTooBig };

/// The close code for a frame error: 1002, 1003, 1007 or 1009.
pub fn closeCode(e: Error) u16 {
    return switch (e) {
        error.ProtocolError => 1002,
        error.UnsupportedData => 1003,
        error.InvalidPayload => 1007,
        error.MessageTooBig => 1009,
    };
}

pub const max_control = 125;
pub const max_head = 14;

pub const Header = struct {
    fin: bool,
    rsv: u3,
    opcode: Opcode,
    masked: bool,
    payload: u64,
    key: [4]u8 = .{ 0, 0, 0, 0 },
    /// The length used its shortest encoding (RFC 6455 section 5.2 requires it).
    minimal: bool = true,

    pub const Parsed = struct { header: Header, len: usize };

    /// The header at the start of ``bytes`` and its length; null until the bytes hold all of it.
    pub fn parse(bytes: []const u8) ?Parsed {
        if (bytes.len < 2) return null;
        var h: Header = .{
            .fin = bytes[0] & 0x80 != 0,
            .rsv = @intCast((bytes[0] >> 4) & 0x7),
            .opcode = @fromBackingInt(@intCast(@as(u4, @truncate(bytes[0])))),
            .masked = bytes[1] & 0x80 != 0,
            .payload = bytes[1] & 0x7f,
        };
        var at: usize = 2;
        if (h.payload == 126) {
            if (bytes.len < 4) return null;
            h.payload = std.mem.readInt(u16, bytes[2..4], .big);
            h.minimal = h.payload >= 126;
            at = 4;
        } else if (h.payload == 127) {
            if (bytes.len < 10) return null;
            h.payload = std.mem.readInt(u64, bytes[2..10], .big);
            h.minimal = h.payload > std.math.maxInt(u16);
            at = 10;
        }
        if (h.masked) {
            if (bytes.len < at + 4) return null;
            h.key = bytes[at..][0..4].*;
            at += 4;
        }
        return .{ .header = h, .len = at };
    }

    /// Encodes a header into ``out``; servers send no ``mask``, clients must.
    pub fn encode(out: *[max_head]u8, fin: bool, opcode: Opcode, len: u64, mask: ?[4]u8) []const u8 {
        out[0] = @as(u8, if (fin) 0x80 else 0) | @backingInt(opcode);
        const bit: u8 = if (mask != null) 0x80 else 0;
        var at: usize = 2;
        if (len < 126) {
            out[1] = bit | @as(u8, @intCast(len));
        } else if (len <= std.math.maxInt(u16)) {
            out[1] = bit | 126;
            std.mem.writeInt(u16, out[2..4], @intCast(len), .big);
            at = 4;
        } else {
            out[1] = bit | 127;
            std.mem.writeInt(u64, out[2..10], len, .big);
            at = 10;
        }
        if (mask) |key| {
            out[at..][0..4].* = key;
            at += 4;
        }
        return out[0..at];
    }
};

/// XORs ``payload`` with ``key``, where ``offset`` is the payload's position within its frame.
pub fn unmask(payload: []u8, key: [4]u8, offset: usize) void {
    for (payload, offset..) |*b, i| b.* ^= key[i & 3];
}

/// Frames up to this size go out in one write from the stack; larger ones write the header, then the payload.
const small_frame = 16 * 1024;

/// One unmasked server frame; it keeps no buffer, so a large event leaves no capacity behind on the connection.
pub fn writeFrame(conn: *Conn, opcode: Opcode, payload: []const u8) error{Closed}!void {
    var head: [max_head]u8 = undefined;
    const h = Header.encode(&head, true, opcode, payload.len, null);
    if (h.len + payload.len <= small_frame) {
        var whole: [small_frame]u8 = undefined;
        @memcpy(whole[0..h.len], h);
        @memcpy(whole[h.len..][0..payload.len], payload);
        return conn.writeAll(whole[0 .. h.len + payload.len]);
    }
    try conn.writeAll(h);
    return conn.writeAll(payload);
}

/// A close frame with ``code`` and ``reason``, cut to fit a control frame.
pub fn closeFrame(conn: *Conn, code: u16, reason: []const u8) error{Closed}!void {
    var payload: [max_control]u8 = undefined;
    std.mem.writeInt(u16, payload[0..2], code, .big);
    const cut = utf8Prefix(reason, max_control - 2);
    @memcpy(payload[2..][0..cut.len], cut);
    return writeFrame(conn, .close, payload[0 .. 2 + cut.len]);
}

/// The longest prefix of ``text`` within ``max`` bytes that ends on a character boundary.
fn utf8Prefix(text: []const u8, max: usize) []const u8 {
    if (text.len <= max) return text;
    var end = max;
    while (end > 0 and text[end] & 0xc0 == 0x80) end -= 1;
    return text[0..end];
}

/// What a finished frame gives the session.
pub const Event = union(enum) {
    /// A whole text message; the caller owns it and frees it with the assembler's allocator.
    text: []u8,
    /// Ping and pong payloads point into the assembler and live until its next ``push``.
    ping: []const u8,
    pong: []const u8,
    /// A close frame: its code (null when it carried none) and UTF-8 reason.
    close: struct { code: ?u16, reason: []const u8 },
};

pub const Step = struct { used: usize, event: ?Event = null };

/// Joins client frames into messages. It parses a header once its bytes are present and takes payload in any chunks.
pub const Assembler = struct {
    gpa: Allocator,
    limit: usize = http_body.limit,
    message: std.ArrayList(u8) = .empty,
    /// A text message has started and its final fragment has not arrived.
    open: bool = false,
    /// The frame whose payload is being read, and how much of it has arrived.
    frame: ?Header = null,
    got: u64 = 0,
    control: [max_control]u8 = undefined,

    pub fn deinit(s: *Assembler) void {
        s.message.deinit(s.gpa);
    }

    /// Whether the assembler is between messages: no frame or fragmented message is half read.
    pub fn idle(s: *const Assembler) bool {
        return s.frame == null and !s.open;
    }

    /// Takes what it can from ``bytes``: a header once whole, then payload. ``used`` 0 and no event means read more.
    pub fn push(s: *Assembler, bytes: []const u8) (Error || Allocator.Error)!Step {
        var used: usize = 0;
        if (s.frame == null) {
            const p = Header.parse(bytes) orelse return .{ .used = 0 };
            try s.begin(p.header);
            used = p.len;
        }
        const h = s.frame.?;
        const take: usize = @intCast(@min(h.payload - s.got, bytes.len - used));
        const chunk = bytes[used..][0..take];
        const offset: usize = @intCast(s.got);
        if (h.opcode.control()) {
            const dest = s.control[offset..][0..take];
            @memcpy(dest, chunk);
            unmask(dest, h.key, offset);
        } else {
            const start = s.message.items.len;
            try s.message.appendSlice(s.gpa, chunk);
            unmask(s.message.items[start..], h.key, offset);
        }
        s.got += take;
        used += take;
        if (s.got < h.payload) return .{ .used = used };
        s.frame = null;
        s.got = 0;
        return .{ .used = used, .event = try s.end(h) };
    }

    fn begin(s: *Assembler, h: Header) (Error || Allocator.Error)!void {
        if (h.rsv != 0 or !h.masked or !h.minimal or h.payload >> 63 != 0) return error.ProtocolError;
        switch (h.opcode) {
            .close, .ping, .pong => if (!h.fin or h.payload > max_control) return error.ProtocolError,
            .continuation => if (!s.open) return error.ProtocolError,
            .text => if (s.open) return error.ProtocolError,
            .binary => return if (s.open) error.ProtocolError else error.UnsupportedData,
            _ => return error.ProtocolError,
        }
        if (!h.opcode.control() and h.payload > s.limit - s.message.items.len) return error.MessageTooBig;
        if (h.opcode == .text) s.open = true;
        s.frame = h;
        s.got = 0;
    }

    fn end(s: *Assembler, h: Header) (Error || Allocator.Error)!?Event {
        const n: usize = @intCast(h.payload);
        switch (h.opcode) {
            .ping => return .{ .ping = s.control[0..n] },
            .pong => return .{ .pong = s.control[0..n] },
            .close => return try closing(s.control[0..n]),
            else => {},
        }
        if (!h.fin) return null;
        s.open = false;
        if (!std.unicode.utf8ValidateSlice(s.message.items)) {
            s.message.clearRetainingCapacity();
            return error.InvalidPayload;
        }
        return .{ .text = try s.message.toOwnedSlice(s.gpa) };
    }
};

/// A close frame's payload: none, or a valid code and a UTF-8 reason.
fn closing(payload: []const u8) Error!Event {
    if (payload.len == 0) return .{ .close = .{ .code = null, .reason = "" } };
    if (payload.len == 1) return error.ProtocolError;
    const code = std.mem.readInt(u16, payload[0..2], .big);
    const valid = switch (code) {
        1000...1003, 1007...1014, 3000...4999 => true,
        else => false,
    };
    if (!valid) return error.ProtocolError;
    if (!std.unicode.utf8ValidateSlice(payload[2..])) return error.InvalidPayload;
    return .{ .close = .{ .code = code, .reason = payload[2..] } };
}

const testing = std.testing;

/// A masked client frame, as a browser or SDK sends it.
fn clientFrame(a: Allocator, fin: bool, opcode: Opcode, payload: []const u8, key: [4]u8) ![]u8 {
    var head: [max_head]u8 = undefined;
    const h = Header.encode(&head, fin, opcode, payload.len, key);
    const out = try std.mem.concat(a, u8, &.{ h, payload });
    unmask(out[h.len..], key, 0);
    return out;
}

/// Feeds ``wire`` in chunks of ``step`` bytes and collects every event.
fn feed(s: *Assembler, wire: []const u8, step: usize, events: *std.ArrayList(Event)) !void {
    var held: std.ArrayList(u8) = .empty;
    defer held.deinit(testing.allocator);
    var at: usize = 0;
    while (true) {
        const step_r = try s.push(held.items);
        const rest = held.items.len - step_r.used;
        std.mem.copyForwards(u8, held.items[0..rest], held.items[step_r.used..]);
        held.shrinkRetainingCapacity(rest);
        if (step_r.event) |e| try events.append(testing.allocator, e);
        if (step_r.used > 0 or step_r.event != null) continue;
        if (at == wire.len) return;
        const n = @min(step, wire.len - at);
        try held.appendSlice(testing.allocator, wire[at..][0..n]);
        at += n;
    }
}

fn freeEvents(s: *Assembler, events: *std.ArrayList(Event)) void {
    for (events.items) |e| if (e == .text) s.gpa.free(e.text);
    events.deinit(testing.allocator);
}

test "ws_frame: 7, 16 and 64-bit lengths round-trip through encode and parse" {
    var head: [max_head]u8 = undefined;
    const key = [4]u8{ 1, 2, 3, 4 };
    for ([_]u64{ 0, 125, 126, 65535, 65536, 1 << 40 }, [_]usize{ 6, 6, 8, 8, 14, 14 }) |len, size| {
        const h = Header.encode(&head, true, .text, len, key);
        try testing.expectEqual(size, h.len);
        try testing.expect(Header.parse(h[0 .. h.len - 1]) == null);
        const p = Header.parse(h).?;
        try testing.expectEqual(size, p.len);
        try testing.expectEqual(len, p.header.payload);
        try testing.expectEqualSlices(u8, &key, &p.header.key);
        try testing.expect(p.header.fin and p.header.masked and p.header.opcode == .text);
    }
    try testing.expectEqual(@as(usize, 2), Header.encode(&head, false, .ping, 3, null).len);
    try testing.expectEqual(@as(usize, 10), Header.encode(&head, true, .binary, 70000, null).len);
}

test "ws_frame: masks hold across every chunk split, for every message length class" {
    const a = testing.allocator;
    for ([_]usize{ 5, 300, 70000 }) |len| {
        const text = try a.alloc(u8, len);
        defer a.free(text);
        for (text, 0..) |*b, i| b.* = 'a' + @as(u8, @intCast(i % 26));
        const wire = try clientFrame(a, true, .text, text, .{ 0x37, 0xfa, 0x21, 0x3d });
        defer a.free(wire);
        for ([_]usize{ 1, 3, 7, 4096, wire.len }) |step| {
            var s: Assembler = .{ .gpa = a };
            defer s.deinit();
            var events: std.ArrayList(Event) = .empty;
            defer freeEvents(&s, &events);
            try feed(&s, wire, step, &events);
            try testing.expectEqual(@as(usize, 1), events.items.len);
            try testing.expectEqualStrings(text, events.items[0].text);
            try testing.expect(s.idle());
        }
    }
}

test "ws_frame: fragments join, and control frames between them pass through" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const x = arena.allocator();
    const wire = try std.mem.concat(x, u8, &.{
        try clientFrame(x, false, .text, "{\"type\":", .{ 9, 8, 7, 6 }),
        try clientFrame(x, true, .ping, "hi", .{ 1, 1, 1, 1 }),
        try clientFrame(x, false, .continuation, "\"response.", .{ 5, 4, 3, 2 }),
        try clientFrame(x, true, .continuation, "create\"}", .{ 0xff, 0, 0xff, 0 }),
        try clientFrame(x, true, .pong, "", .{ 2, 2, 2, 2 }),
        try clientFrame(x, true, .close, "\x03\xe8bye", .{ 3, 3, 3, 3 }),
    });
    var s: Assembler = .{ .gpa = a };
    defer s.deinit();
    var events: std.ArrayList(Event) = .empty;
    defer freeEvents(&s, &events);
    try feed(&s, wire, 5, &events);
    try testing.expectEqual(@as(usize, 4), events.items.len);
    try testing.expectEqualStrings("{\"type\":\"response.create\"}", events.items[1].text);
    try testing.expect(events.items[2] == .pong);
    try testing.expectEqual(@as(?u16, 1000), events.items[3].close.code);
}

fn expectRefused(want: u16, frames: []const []const u8) !void {
    const a = testing.allocator;
    var s: Assembler = .{ .gpa = a, .limit = 64 };
    defer s.deinit();
    var events: std.ArrayList(Event) = .empty;
    defer freeEvents(&s, &events);
    const wire = try std.mem.concat(a, u8, frames);
    defer a.free(wire);
    feed(&s, wire, wire.len, &events) catch |e| return testing.expectEqual(want, closeCode(@errorCast(e)));
    return error.TestExpectedError;
}

test "ws_frame: control-frame rules and close codes 1002, 1003, 1007 and 1009" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const x = arena.allocator();
    const k = [4]u8{ 0xa1, 0xb2, 0xc3, 0xd4 };
    var head: [max_head]u8 = undefined;
    const unmasked = try std.mem.concat(x, u8, &.{ Header.encode(&head, true, .text, 2, null), "{}" });
    try expectRefused(1002, &.{unmasked});
    try expectRefused(1002, &.{try clientFrame(x, false, .ping, "", k)});
    try expectRefused(1002, &.{try clientFrame(x, true, .ping, &@as([126]u8, @splat('p')), k)});
    try expectRefused(1002, &.{try clientFrame(x, true, .continuation, "x", k)});
    try expectRefused(1002, &.{ try clientFrame(x, false, .text, "x", k), try clientFrame(x, true, .text, "y", k) });
    try expectRefused(1002, &.{try clientFrame(x, true, @fromBackingInt(@intCast(3)), "", k)});
    const rsv = try clientFrame(x, true, .text, "x", k);
    rsv[0] |= 0x40;
    try expectRefused(1002, &.{rsv});
    try expectRefused(1002, &.{try clientFrame(x, true, .close, "\x03", k)});
    try expectRefused(1002, &.{try clientFrame(x, true, .close, "\x03\xed", k)});
    try expectRefused(1002, &.{try clientFrame(x, true, .close, "\x03\xf7", k)}); // 1015 is never sent
    // a length in a longer form than it needs: 1 byte as a 16-bit length, 200 bytes as a 64-bit length
    const short16 = [_]u8{ 0x81, 0x80 | 126, 0, 1 } ++ k ++ [_]u8{'x' ^ 0xa1};
    try expectRefused(1002, &.{&short16});
    const short64 = [_]u8{ 0x81, 0x80 | 127, 0, 0, 0, 0, 0, 0, 0, 200 } ++ k;
    try expectRefused(1002, &.{&short64});
    try expectRefused(1003, &.{try clientFrame(x, true, .binary, "x", k)});
    try expectRefused(1007, &.{try clientFrame(x, true, .text, "\xc3\x28", k)});
    try expectRefused(1007, &.{ try clientFrame(x, false, .text, "\xe2\x82", k), try clientFrame(x, true, .continuation, "", k) });
    try expectRefused(1007, &.{try clientFrame(x, true, .close, "\x03\xe8\xff", k)});
    try expectRefused(1009, &.{try clientFrame(x, true, .text, &@as([65]u8, @splat('a')), k)});
    try expectRefused(1009, &.{ try clientFrame(x, false, .text, &@as([40]u8, @splat('a')), k), try clientFrame(x, true, .continuation, &@as([25]u8, @splat('a')), k) });
}

test "ws_frame: a valid split UTF-8 character across fragments is accepted" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const x = arena.allocator();
    const wire = try std.mem.concat(x, u8, &.{
        try clientFrame(x, false, .text, "\xe2\x82", .{ 1, 2, 3, 4 }),
        try clientFrame(x, true, .continuation, "\xac", .{ 4, 3, 2, 1 }),
    });
    var s: Assembler = .{ .gpa = a };
    defer s.deinit();
    var events: std.ArrayList(Event) = .empty;
    defer freeEvents(&s, &events);
    try feed(&s, wire, 1, &events);
    try testing.expectEqualStrings("\xe2\x82\xac", events.items[0].text);
}

const Capture = struct {
    bytes: std.ArrayList(u8) = .empty,
    writes: usize = 0,
    fn write(ctx: *anyopaque, data: []const u8) error{Closed}!void {
        const w: *Capture = @ptrCast(@alignCast(ctx));
        w.writes += 1;
        w.bytes.appendSlice(testing.allocator, data) catch return error.Closed;
    }
};

test "ws_frame: server frames are unmasked, one write each, and close carries code and reason" {
    var c = try Conn.init(testing.allocator, -1, "fixture");
    defer c.deinit();
    var cap: Capture = .{};
    defer cap.bytes.deinit(testing.allocator);
    c.write_hooks = .{ .ctx = &cap, .write = Capture.write };
    const big: [300]u8 = @splat('z');
    try writeFrame(&c, .text, &big);
    try testing.expectEqual(@as(usize, 1), cap.writes);
    const p = Header.parse(cap.bytes.items).?;
    try testing.expect(!p.header.masked and p.header.fin and p.header.opcode == .text);
    try testing.expectEqual(@as(u64, 300), p.header.payload);
    try testing.expectEqualSlices(u8, &big, cap.bytes.items[p.len..]);
    cap.bytes.clearRetainingCapacity();
    const long: [200]u8 = @splat('x');
    try closeFrame(&c, 1009, &long);
    const q = Header.parse(cap.bytes.items).?;
    try testing.expectEqual(Opcode.close, q.header.opcode);
    try testing.expectEqual(@as(u64, max_control), q.header.payload);
    try testing.expectEqualSlices(u8, "\x03\xf1", cap.bytes.items[q.len..][0..2]);
    cap.bytes.clearRetainingCapacity();
    cap.writes = 0;
    const huge = try testing.allocator.alloc(u8, small_frame + 1);
    defer testing.allocator.free(huge);
    @memset(huge, 'h');
    try writeFrame(&c, .text, huge);
    try testing.expectEqual(@as(usize, 2), cap.writes); // header, then payload: no buffer kept on the connection
    try testing.expectEqual(@as(usize, 0), c.out.capacity);
    const r = Header.parse(cap.bytes.items).?;
    try testing.expectEqual(@as(u64, small_frame + 1), r.header.payload);
    try testing.expectEqualSlices(u8, huge, cap.bytes.items[r.len..]);
}

test "ws_frame: the service-restart close codes 1012, 1013 and 1014 are valid" {
    for ([_]u16{ 1012, 1013, 1014 }) |code| {
        var payload: [2]u8 = undefined;
        std.mem.writeInt(u16, &payload, code, .big);
        const e = try closing(&payload);
        try testing.expectEqual(@as(?u16, code), e.close.code);
    }
}
