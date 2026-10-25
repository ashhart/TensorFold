//! Keep a silent streaming response alive while the ordinary request cancellation path remains active.
const Conn = @import("http_conn.zig").Conn;
const Out = @import("openai.zig").Out;
/// The HTTP fixture observes an attempted due beat, including one correctly suppressed before the head.
pub const Testing = if (@import("builtin").is_test) struct {
    pub threadlocal var after_due_beat: ?struct { ctx: *anyopaque, call: *const fn (*anyopaque, ?bool, bool) void } = null;
} else struct {};

pub const Gone = struct {
    conn: *Conn,
    /// The stream a keepalive goes to once it has been silent `every_ns` (0: never)
    out: ?Out = null,
    every_ns: i96 = 0,
    /// When supplied, no keepalive bytes are sent until the response open has completed.
    opened: ?*const bool = null,
    pub fn check(g: Gone) bool {
        return g.conn.peerGone();
    }

    /// A keepalive if the stream has been silent long enough; a failed write means the client left.
    pub fn beat(g: Gone, silent_ns: i96) error{Closed}!bool {
        const o = g.out orelse return false;
        const send = o.vt.keepalive orelse return false;
        if (g.every_ns <= 0 or silent_ns < g.every_ns) return false;
        var sent = false;
        defer if (@import("builtin").is_test) {
            if (Testing.after_due_beat) |observer| observer.call(observer.ctx, if (g.opened) |p| p.* else null, sent);
        };
        if (g.opened) |opened| if (!opened.*) return false;
        try send(o.ctx);
        sent = true;
        return true;
    }
};
