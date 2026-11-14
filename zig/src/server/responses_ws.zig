//! A Responses WebSocket: one socket's session loop, its limits, and how it closes (the design's "Session").
const std = @import("std");
const builtin = @import("builtin");
const auth_scope = @import("auth_scope.zig");
const ws_frame = @import("ws_frame.zig");
const ws_upgrade = @import("ws_upgrade.zig");
const ws_cache = @import("ws_cache.zig");
const http_deadline = @import("http_deadline.zig");
const wait = @import("wait_reason.zig");
const read = @import("responses_ws_read.zig");
const events = @import("responses_ws_events.zig");
const turn = @import("responses_ws_turn.zig");
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Allocator = std.mem.Allocator;

/// A socket lives this long from its upgrade (OpenAI's 60-minute connection limit).
pub const lifetime_ms: i64 = 60 * 60 * 1000;
/// Tests only: a shorter lifetime, read in test builds alone.
pub var test_lifetime_ms: ?i64 = null;
/// Tests only: the server the last socket opened on, so a test can read its store and cache budget.
pub var test_server: std.atomic.Value(?*Server) = .init(null);
/// The session waits on the socket at most this long before it checks drain, expiry and the idle timeout again.
pub const step_ms: i64 = 1000;
/// Named lanes one connection accepts; the default lane does not count.
pub const max_lanes = 32;

const limit_message = "Responses websocket connection limit reached (60 minutes). Create a new websocket connection to continue.";

/// The key that opened the socket: its digest, and the label and scopes it had at its last check.
pub const Principal = struct { digest: [32]u8, label: []u8, scopes: auth_scope.Scopes };

pub const Session = struct {
    srv: *Server,
    conn: *Conn,
    gpa: Allocator,
    cache: ws_cache.Cache,
    assembler: ws_frame.Assembler,
    /// Whole client messages not yet handled, oldest first; each owned by ``gpa``.
    queue: std.ArrayList([]u8) = .empty,
    queued_bytes: usize = 0,
    /// The read deadline of the message being read (phase ``none`` between messages).
    deadline: http_deadline.Budget = .{},
    /// Named lanes seen on this socket, each name owned by ``gpa``.
    lanes: std.array_hash_map.String(void) = .empty,
    started_ms: i64,
    /// When the last message's handling ended (or the upgrade): the idle timeout counts from here.
    idle_since_ms: i64,
    gone: bool = false,
    close_sent: bool = false,
    /// Null when the upgrade passed without a key (no key store, or keys disabled).
    principal: ?Principal = null,

    pub fn now(s: *const Session) i64 {
        return s.conn.nowMs();
    }

    /// Sends a close frame once and ends the session.
    pub fn closeWith(s: *Session, code: u16, reason: []const u8) void {
        if (!s.close_sent and !s.conn.broken) ws_frame.closeFrame(s.conn, code, reason) catch {};
        s.close_sent = true;
        s.gone = true;
    }

    fn deinit(s: *Session) void {
        s.conn.gone_hook = null;
        for (s.queue.items) |m| s.gpa.free(m);
        s.queue.deinit(s.gpa);
        for (s.lanes.keys()) |name| s.gpa.free(name);
        s.lanes.deinit(s.gpa);
        s.assembler.deinit();
        s.cache.deinit();
        if (s.principal) |p| s.gpa.free(p.label);
    }

    fn lifetime() i64 {
        if (builtin.is_test) if (test_lifetime_ms) |ms| return ms;
        return lifetime_ms;
    }

    fn expired(s: *const Session) bool {
        return s.now() - s.started_ms >= lifetime();
    }

    /// Drain and expiry, checked at every wait step and after every message; true when the session closed.
    fn ending(s: *Session) bool {
        if (s.srv.draining.load(.seq_cst)) {
            events.connectionError(s, .{ .status = 503, .kind = "server_error", .code = wait.restarting.code, .message = wait.restarting.message });
            s.closeWith(1012, "server restarting");
            return true;
        }
        if (s.expired()) {
            events.connectionError(s, .{ .status = 400, .kind = "invalid_request_error", .code = "websocket_connection_limit_reached", .message = limit_message });
            s.closeWith(1000, "connection limit reached");
            return true;
        }
        return false;
    }

    /// The wait for the next step: at most a second, and never past the idle timeout, a message deadline or expiry.
    fn stepWait(s: *const Session) i32 {
        const t = s.now();
        var left: i64 = step_ms;
        left = @min(left, lifetime() - (t - s.started_ms));
        if (s.assembler.idle()) left = @min(left, @as(i64, s.conn.timeouts.idle_ms) - (t - s.idle_since_ms));
        if (s.deadline.phase == .body) left = @min(left, s.deadline.end - t);
        return @intCast(std.math.clamp(left, 1, step_ms));
    }

    /// Between messages: wait for bytes in one-second steps until a message is queued or the session ends.
    fn awaitMessage(s: *Session) void {
        while (!s.gone and s.queue.items.len == 0) {
            if (s.ending()) return;
            if (s.assembler.idle() and s.now() - s.idle_since_ms >= s.conn.timeouts.idle_ms) {
                s.closeWith(1001, "idle timeout");
                return;
            }
            read.pump(s, s.stepWait());
        }
    }

    /// Messages one at a time in arrival order; drain and expiry are checked before each, so a close drops the rest.
    fn run(s: *Session) void {
        while (!s.gone) {
            s.awaitMessage();
            if (s.gone or s.ending()) break;
            const message = s.queue.orderedRemove(0);
            s.queued_bytes -= message.len;
            defer s.gpa.free(message);
            turn.message(s, message);
            s.idle_since_ms = s.now();
            if (s.conn.broken) s.gone = true;
        }
    }
};

/// The upgrade answered, then the session until it closes; the HTTP connection closes after it.
pub fn serve(srv: *Server, conn: *Conn, a: Allocator) void {
    if (builtin.is_test) test_server.store(srv, .release); // before the 101, so a test that reads the 101 sees this server
    if (!ws_upgrade.answer(conn, a)) return;
    conn.budget = .{};
    const started = conn.nowMs();
    var s: Session = .{
        .srv = srv,
        .conn = conn,
        .gpa = srv.gpa,
        .cache = .{ .gpa = srv.gpa, .global = &srv.ws_cache_bytes },
        .assembler = .{ .gpa = srv.gpa },
        .started_ms = started,
        .idle_since_ms = started,
    };
    defer s.deinit();
    const upgrade_label = conn.key_label;
    defer conn.key_label = upgrade_label; // the session's own copy dies with it
    if (conn.key_digest) |digest| if (conn.key_label) |label| {
        const owned = srv.gpa.dupe(u8, label) catch return;
        s.principal = .{ .digest = digest, .label = owned, .scopes = conn.key_scopes };
        conn.key_label = owned;
    };
    s.run();
}
