//! Lane admission leaves the complete bounded trainer resident budget available before selecting the stream count.
const std = @import("std");
const cfg = @import("config.zig");
pub const max_rows = 256;
pub fn hostBytes(c: *const cfg.Config, pack: usize) !usize {
    var problem: cfg.Problem = .{};
    const geom = try @import("cuda_attention.zig").Geometry.of(c, &problem);
    const idw = try geom.ids();
    return c.layers * pack * c.hidden * 2 + 2 * 3 * c.layers * c.hidden * 4 + pack * 24 + (c.hidden + c.wide() + c.streams) * 2 + idw * 4;
}
pub fn bytes(comptime B: type, a: std.mem.Allocator, c: *const cfg.Config, capacity: usize) !u64 {
    const pack: usize = @min(max_rows, capacity);
    if (pack < 2) return error.Qwen4TrainingRows;
    var tape = try @import("train_tape.zig").Tape(B).describe(a, undefined, c, pack);
    defer tape.deinitDrained();
    const plan = try @import("train_buffers.zig").Plan(B).make(c, pack);
    return std.math.add(u64, tape.bytes, try std.math.add(u64, try B.allocationBytes(plan.bytes), try hostBytes(c, pack)));
}
