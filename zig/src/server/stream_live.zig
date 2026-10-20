//! Keep a silent streaming response alive while the ordinary request cancellation path remains active.
const Conn = @import("http_conn.zig").Conn;
const Out = @import("openai.zig").Out;
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
        if (g.opened) |opened| if (!opened.*) return false;
        try send(o.ctx);
        return true;
    }
};
