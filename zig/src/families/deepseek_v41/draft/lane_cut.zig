//! DeepSeek-V4.1's hand-off export point on a slot: the stream's mark hook, run by the target at the prompt's cut.
const std = @import("std");
const ln = @import("lanes.zig");
const Stream = @import("lanes").Stream;

/// A hand-off's export point: the stream's mark hook, called one token before the prompt's end.
pub const Cut = struct {
    s: *Stream,
    at: u32,

    fn run(ptr: *anyopaque) void {
        const c: *Cut = @ptrCast(@alignCast(ptr));
        const k = c.s.reuse.hook orelse return;
        k.at(k.ptr, c.s, c.at);
    }

    fn progress(ptr: *anyopaque, at: u64) void {
        const c: *Cut = @ptrCast(@alignCast(ptr));
        const k = c.s.reuse.progress orelse return;
        k.at(k.ptr, c.s, @intCast(at));
    }
};

/// A hand-off stream's export point armed on its slot (the stream's marks hold the cut); false: not a hand-off.
pub fn arm(x: *ln.Lanes, slot: u32, s: *Stream) bool {
    const ids = s.prompt();
    if (ids.len < 2 or s.reuse.hook == null or std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(ids.len - 1)) == null) return false;
    x.cuts[slot] = .{ .s = s, .at = @intCast(ids.len - 1) };
    x.target.arm(slot, .{ .ptr = &x.cuts[slot], .run = Cut.run, .progress = Cut.progress });
    return true;
}
