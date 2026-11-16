//! Routed checkpoint half-product rotation, nvcc 13.0.88.
const CudaRecord = @import("registry.zig").CudaRecord;
pub const records = [_]CudaRecord{
    .{ .launch = .{ .name = "exl3 routed half rotation", .symbol = "tf_exl3_routed_half_rotate", .image = "exl3_routed_half", .threads = 32 }, .usage = .{ .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 26, .static = 0, .bound = 32 }, .{ .regs = 20, .static = 0, .bound = 32 }, .{ .regs = 20, .static = 0, .bound = 32 } } },
};
