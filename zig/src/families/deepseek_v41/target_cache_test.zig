const std = @import("std");
const target = @import("target.zig");
const fwd = @import("forward.zig");

test "CUDA target reports a fully reused prompt and clears the receipt before a rejected request" {
    const ids = [_]u32{ 11, 22, 33 };
    var forward: fwd.Forward = undefined;
    forward.slot.pos = ids.len;
    forward.slot.pending = null;
    var gpu: target.GpuTarget = .{ .f = &forward, .resume_first = 99 };
    const t = gpu.target();
    try std.testing.expectEqual(@as(u32, 99), try t.prefill(0, &ids, null, ids.len));
    try std.testing.expectEqual(@as(u64, ids.len), t.resumedRows(0));
    try std.testing.expectEqual(@as(u64, 0), t.resumedRows(1));
    try std.testing.expectError(error.Sampling, t.prefill(0, &ids, .{ .seed = 1 }, ids.len));
    try std.testing.expectEqual(@as(u64, 0), t.resumedRows(0));
}
