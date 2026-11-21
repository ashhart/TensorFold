//! A Responses socket's reader: frames from the connection buffer into whole queued messages, without blocking a turn.
const std = @import("std");
const ws_frame = @import("ws_frame.zig");
const http_body = @import("http_body.zig");
const Session = @import("responses_ws.zig").Session;

/// Messages a socket holds unhandled: at most this many, and ``max_queued_bytes`` in all.
pub const max_queued = 16;
pub const max_queued_bytes = http_body.limit;

/// Takes every whole piece the buffer holds; when none progresses, reads what the socket has within ``wait_ms``.
pub fn pump(s: *Session, wait_ms: i32) void {
    if (s.gone) return;
    if (!take(s) and !s.gone) fill(s, wait_ms);
    if (!s.gone and s.deadline.phase == .body and s.now() >= s.deadline.end) s.closeWith(1008, "message read timeout");
}

/// One socket read within ``wait_ms``, then what it brought; EOF or a failed read ends the session.
fn fill(s: *Session, wait_ms: i32) void {
    const got = s.conn.readSome(wait_ms) catch |e| switch (e) {
        error.Timeout => return,
        error.Closed => {
            s.gone = true;
            return;
        },
    };
    if (got == 0) {
        s.gone = true;
        return;
    }
    _ = take(s);
}

/// The cancellation check of a running turn: what the socket has ready is read and parsed, never waited for.
pub fn pollGone(ctx: *anyopaque) bool {
    const s: *Session = @ptrCast(@alignCast(ctx));
    pump(s, 0);
    return s.gone or s.conn.broken;
}

/// Feeds buffered bytes to the assembler until it needs more; true when anything was consumed.
fn take(s: *Session) bool {
    var progressed = false;
    while (!s.gone) {
        const bytes = s.conn.pending();
        if (bytes.len == 0) break;
        if (s.assembler.idle() and s.deadline.phase != .body) s.deadline.body(s.now(), s.conn.timeouts.request);
        if (s.assembler.frame == null) if (ws_frame.Header.parse(bytes)) |p| {
            if (!p.header.opcode.control()) s.deadline.declared(p.header.payload, s.conn.timeouts.request);
        };
        // checked before the bytes count, so a message that ends after its deadline is never queued
        if (s.deadline.phase == .body and s.now() >= s.deadline.end) {
            s.closeWith(1008, "message read timeout");
            return true;
        }
        const step = s.assembler.push(bytes) catch |e| {
            if (e == error.OutOfMemory) s.closeWith(1011, "out of memory") else s.closeWith(ws_frame.closeCode(@errorCast(e)), "frame error");
            return true;
        };
        s.conn.consume(step.used);
        if (step.used > 0) progressed = true;
        if (step.event) |event| {
            progressed = true;
            // a whole frame that leaves no message open ends the deadline; a partial header keeps it running
            if (s.assembler.idle()) s.deadline = .{};
            handle(s, event);
        } else if (step.used == 0) break;
    }
    return progressed;
}

fn handle(s: *Session, event: ws_frame.Event) void {
    switch (event) {
        .text => |message| queue(s, message),
        .ping => |payload| ws_frame.writeFrame(s.conn, .pong, payload) catch {
            s.gone = true;
        },
        .pong => {},
        .close => |c| s.closeWith(c.code orelse 1000, ""),
    }
}

/// A whole message joins the queue, or past either bound closes the socket with 1008.
fn queue(s: *Session, message: []u8) void {
    if (s.queue.items.len >= max_queued or message.len > max_queued_bytes - s.queued_bytes) {
        s.gpa.free(message);
        return s.closeWith(1008, "too many queued messages");
    }
    s.queue.append(s.gpa, message) catch {
        s.gpa.free(message);
        return s.closeWith(1011, "out of memory");
    };
    s.queued_bytes += message.len;
}

const testing = std.testing;
const http = @import("http_conn.zig");

/// Scripted socket bytes on a fake clock: each piece arrives ``delay`` ms after the read that asks for it.
const Script = struct {
    pieces: []const []const u8,
    delays: []const i32,
    at: usize = 0,
    ms: i64 = 0,
    sent: std.ArrayList(u8) = .empty,

    fn cast(ctx: *anyopaque) *Script {
        return @ptrCast(@alignCast(ctx));
    }
    fn now(ctx: *anyopaque) i64 {
        return cast(ctx).ms;
    }
    fn read(ctx: *anyopaque, out: []u8, wait_ms: i32) http.IoError!usize {
        const s = cast(ctx);
        if (s.at == s.pieces.len) return error.Timeout;
        if (s.delays[s.at] >= wait_ms) {
            s.ms += wait_ms;
            return error.Timeout;
        }
        s.ms += s.delays[s.at];
        const piece = s.pieces[s.at];
        @memcpy(out[0..piece.len], piece);
        s.at += 1;
        return piece.len;
    }
    fn write(ctx: *anyopaque, data: []const u8) error{Closed}!void {
        cast(ctx).sent.appendSlice(testing.allocator, data) catch return error.Closed;
    }

    /// Pumps until the script is spent or the session ends; true when it closed with 1008 and queued nothing.
    fn timedOut(script: *Script) !bool {
        var c = try http.Conn.init(testing.allocator, -1, "script");
        defer c.deinit();
        defer script.sent.deinit(testing.allocator);
        c.quiet_log = true;
        c.read_hooks = .{ .ctx = script, .now = now, .read = read };
        c.write_hooks = .{ .ctx = script, .write = write };
        c.timeouts.request = .{ .body_ms = 100, .body_bytes_per_s = 1_000_000 };
        var global: std.atomic.Value(usize) = .init(0);
        var s: Session = .{
            .srv = undefined,
            .conn = &c,
            .gpa = testing.allocator,
            .cache = .{ .gpa = testing.allocator, .global = &global },
            .assembler = .{ .gpa = testing.allocator },
            .started_ms = 0,
            .idle_since_ms = 0,
        };
        defer {
            for (s.queue.items) |m| testing.allocator.free(m);
            s.queue.deinit(testing.allocator);
            s.assembler.deinit();
            s.cache.deinit();
        }
        while (!s.gone and script.at < script.pieces.len) pump(&s, 1000);
        const close_1008 = [_]u8{ 0x88, 0x02 + "message read timeout".len, 0x03, 0xf0 };
        return s.close_sent and s.queue.items.len == 0 and std.mem.startsWith(u8, script.sent.items, &close_1008);
    }
};

test "a message read deadline holds across a partial header and a late last byte" {
    // a masked text frame of "{}" with a zero mask key
    const whole = [_]u8{ 0x81, 0x82, 0, 0, 0, 0, '{', '}' };
    var on_time: Script = .{ .pieces = &.{ whole[0..1], whole[1..] }, .delays = &.{ 0, 50 } };
    try testing.expect(!try on_time.timedOut());
    var header_drip: Script = .{ .pieces = &.{ whole[0..1], whole[1..] }, .delays = &.{ 0, 150 } };
    try testing.expect(try header_drip.timedOut());
    var last_byte: Script = .{ .pieces = &.{ whole[0..7], whole[7..] }, .delays = &.{ 0, 150 } };
    try testing.expect(try last_byte.timedOut());
}
