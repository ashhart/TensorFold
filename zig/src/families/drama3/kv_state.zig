//! Per-lane logical position, reset at admission and advanced only by committed prompt rows and audio frames.
const std = @import("std");
pub const State = struct {
    prompt: usize = 0,
    frames: usize = 0,
    started: bool = false,

    pub fn reset(s: *State) void {
        s.* = .{};
    }
};

test "a reused lane starts at the same logical state as a fresh lane" {
    var s: State = .{ .prompt = 128, .frames = 9, .started = true };
    s.reset();
    try std.testing.expectEqualDeep(State{}, s);
}
