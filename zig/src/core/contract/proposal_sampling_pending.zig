//! Pitched proposal sampling remains unmeasured until its CUDA device gate supplies resource and arithmetic receipts.
const R = @import("registry.zig").CudaRecord;
const limits = @import("limits.zig");
pub const records = [_]R{
    .{ .launch = .{ .name = "sample tf_draw_pitched", .symbol = "tf_draw_pitched", .image = "sample", .threads = 1024 }, .usage = @splat(.{ .regs = 0, .static = 0 }) },
};
test "proposal pitched sampling has no invented GPU resource qualification" {
    const std = @import("std");
    try std.testing.expectEqual(limits.sms.len, try @import("registry.zig").checkCuda(std.testing.allocator, &records, true));
}
