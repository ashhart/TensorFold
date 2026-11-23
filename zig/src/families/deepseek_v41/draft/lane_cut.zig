//! A hand-off's export point: the stream's mark hook, called one token before the prompt's end.
const std = @import("std");
const lanes = @import("lanes");

/// The cut a prefill arms its target with: `run` fires the stream's mark hook there, during the pass.
pub const Cut = struct {
    s: *lanes.Stream,
    at: u32,

    pub fn run(ptr: *anyopaque) void {
        const c: *Cut = @ptrCast(@alignCast(ptr));
        const k = c.s.reuse.hook orelse return;
        k.at(k.ptr, c.s, c.at);
    }
};

/// Arms `x`'s `slot` with a hand-off cut when `s`'s last prompt token is a mark; false: this prompt exports nothing.
pub fn armCut(x: anytype, slot: u32, s: *lanes.Stream) bool {
    const ids = s.prompt();
    if (ids.len < 2 or s.reuse.hook == null or std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(ids.len - 1)) == null) return false;
    x.cuts[slot] = .{ .s = s, .at = @intCast(ids.len - 1) };
    x.target.arm(slot, .{ .ptr = &x.cuts[slot], .run = Cut.run });
    return true;
}
