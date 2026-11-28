//! The new sparse bias launch needs measured resource receipts before landing.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "sample tf_bias_rows", .symbol = "tf_bias_rows", .image = "sample", .threads = 256 }, .usage = registry.none },
};
