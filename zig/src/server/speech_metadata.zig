const std = @import("std");
const Conn = @import("http_conn.zig").Conn;

pub const Metadata = struct {
    enabled: bool,
    redraws: u8 = 0,
    register_on: bool = false,
    register_tripped: bool = false,
    register_redrawn: bool = false,

    pub fn add(m: Metadata, c: *Conn) !void {
        if (!m.enabled) return;
        var buf: [3]u8 = undefined;
        try c.addHeader("X-Silence-Redraws", std.fmt.bufPrint(&buf, "{d}", .{m.redraws}) catch unreachable);
    }

    pub fn addWhole(m: Metadata, c: *Conn) !void {
        try m.add(c);
        if (m.register_on) {
            try c.addHeader("X-Register-Trip", if (m.register_tripped) "1" else "0");
            try c.addHeader("X-Register-Redraws", if (m.register_redrawn) "1" else "0");
        }
    }
};
