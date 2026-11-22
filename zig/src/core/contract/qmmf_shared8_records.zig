//! MXFP8 direct-store decode geometry; compiled usage is explicitly pending for every SM and compiler.
const std = @import("std");
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "qmmf MXFP8 shared down half16 stride", .symbol = "tf_qmmf_shared8_16_half", .image = "qmmf", .threads = 128, .dynamic = 16896, .min_sm = 80 }, .usage = registry.none },
    .{ .launch = .{ .name = "qmmf MXFP8 shared down 16 stride", .symbol = "tf_qmmf_shared8_16", .image = "qmmf", .threads = 128, .dynamic = 25088, .min_sm = 80 }, .usage = registry.none },
};
test "decode fusion records: resources remain pending rather than borrowing contiguous receipts" {
    try std.testing.expectEqual(@as(usize, 10), try registry.checkCuda(std.testing.allocator, &records, true));
    for (records) |record| for ([_]@import("limits.zig").Nvcc{ .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |release| {
        try std.testing.expectEqual(registry.none, registry.usageFor(record, release));
    };
}
