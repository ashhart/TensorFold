//! A native CUDA host's halt: its lane host stops serving with `reason`.
const Host = @import("cuda.zig").Host;

pub fn halt(ctx: *anyopaque, reason: []const u8) void {
    const h: *Host = @ptrCast(@alignCast(ctx));
    h.host.halt(reason);
}
