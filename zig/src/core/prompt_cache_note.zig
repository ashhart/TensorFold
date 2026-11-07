//! A test's ear on the prompt cache's log: under test `pc.note` hands each line to this thread's sink, if one is set.
const std = @import("std");

/// The lines a test heard, back to back, each ending in a newline (what passes `bytes` is cut).
pub const Heard = struct {
    bytes: [4096]u8 = undefined,
    len: usize = 0,

    pub fn text(h: *const Heard) []const u8 {
        return h.bytes[0..h.len];
    }
};

/// The current thread's sink (null: lines go nowhere, as before).
pub threadlocal var sink: ?*Heard = null;

pub fn hear(comptime fmt: []const u8, args: anytype) void {
    const h = sink orelse return;
    var w: std.Io.Writer = .fixed(h.bytes[h.len..]);
    w.print(fmt ++ "\n", args) catch {};
    h.len += w.end;
}
