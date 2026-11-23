//! A hand-off's export point: the stream's mark hook, called one token before the prompt's end.
const std = @import("std");
const lanes = @import("lanes");

/// The cut a prefill arms its target with: `run` fires the stream's mark hook there, during the pass.
pub const Cut = struct {
    s: *lanes.Stream,
    at: u32,
};

/// `s`'s cut (its prompt less one token) when a pass exports there: two tokens at least, a hook, and a mark.
pub fn of(s: *lanes.Stream) ?Cut {
    const ids = s.prompt();
    if (ids.len < 2 or s.reuse.hook == null) return null;
    const at: u32 = @intCast(ids.len - 1);
    if (std.mem.indexOfScalar(u32, s.reuse.marks, at) == null) return null;
    return .{ .s = s, .at = at };
}

pub fn run(ptr: *anyopaque) void {
    const c: *Cut = @ptrCast(@alignCast(ptr));
    const k = c.s.reuse.hook orelse return;
    k.at(k.ptr, c.s, c.at);
}
