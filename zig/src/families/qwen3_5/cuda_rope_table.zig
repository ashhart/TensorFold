//! The engine's rope table on the device: what the target's and the MTP layer's rope kernels read (null: plain rope).
const std = @import("std");
const cuda = @import("cuda");
const Engine = @import("cuda_engine.zig").Engine;
const Rope = @import("cuda_rope.zig").Rope;

/// The engine ropes with `rope` from its next forward: the table uploaded, graphs captured with the old one dropped.
pub fn set(e: *Engine, rope: Rope) !void {
    free(e);
    e.g.rope = rope;
    switch (rope) {
        .plain => {},
        .yarn => |y| {
            e.rope_buf = try cuda.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(&y.table));
            e.rope_table = e.rope_buf.?.ptr;
            std.debug.print("info: YaRN rope, factor {d} over the trained {d} positions: windows up to {d}\n", .{ y.factor, y.trained, e.g.context });
        },
    }
    // a captured graph keeps the table address it was captured with: each shape captures again on its next run
    var it = e.graphs.valueIterator();
    while (it.next()) |g| g.deinit();
    e.graphs.clearRetainingCapacity();
}

pub fn free(e: *Engine) void {
    if (e.rope_buf) |*b| b.free();
    e.rope_buf = null;
    e.rope_table = 0;
}
