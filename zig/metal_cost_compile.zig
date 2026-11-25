//! Force every native timing branch to compile; this object never executes or links Metal.
const std = @import("std");
const tf = @import("tensorfold");
const qwen = @import("src/native/qwen27_lanes.zig");

export fn qwenCostCompile(b: *qwen.Metal, io: *std.Io) u8 {
    @import("src/native/qwen27_costs.zig").measure(b, io.*) catch return 1;
    return 0;
}
export fn nemoCostCompile(b: *tf.nemotron.backend.Metal, io: *std.Io) u8 {
    tf.nemotron.timing.measure(b, io.*) catch return 1;
    return 0;
}
export fn glmCostCompile(sl: *tf.glm.slots.Slots, gpa: *std.mem.Allocator, io: *std.Io) u8 {
    _ = tf.glm.timing.measure(sl, gpa.*, io.*) catch return 1;
    return 0;
}
