//! Read-only layer prefetch through core; uploads and page drops retain their loader's order.
const std = @import("std");
const core = @import("core");
pub fn begin(ck: *const core.Checkpoint, i: usize, count: usize) !core.checkpoint_read_ahead.Scope {
    if (count == 0) return .{};
    var buf: [160]u8 = undefined;
    return core.checkpoint_read_ahead.checkpoint(ck, try std.fmt.bufPrint(&buf, "model.layers.{d}.", .{i}), count);
}
