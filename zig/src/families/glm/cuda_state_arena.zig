//! GLM's CUDA state arena: one zeroed device allocation handed out field by field.
const std = @import("std");
const cuda = @import("cuda");

/// Sub-allocations of one device allocation, 256-byte aligned like the driver's own.
pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    pub fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

/// Allocates `sizes` (bytes) in one zeroed device buffer and hands out each field's address in order.
pub fn arena(d: *const cuda.Driver, sizes: []const usize, out: []u64) !Arena {
    var total: usize = 0;
    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, @max(total, 256)) };
    errdefer a.buf.free();
    try a.buf.fill8(0, null);
    for (sizes, out) |n, *p| p.* = a.take(n);
    return a;
}
