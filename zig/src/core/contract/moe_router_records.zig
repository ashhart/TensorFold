//! Expert-first router launch; CUDA 13.0.88 cubin resources measured on all five targets.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "moe tf_moe_router_expert_first", .symbol = "tf_moe_router_expert_first", .image = "moe", .threads = 256 }, .usage = .{
        .{ .regs = 32, .static = 16512, .bound = 256 },
        .{ .regs = 48, .static = 16512, .bound = 256 },
        .{ .regs = 48, .static = 16512, .bound = 256 },
        .{ .regs = 63, .static = 16512, .bound = 256 },
        .{ .regs = 63, .static = 16512, .bound = 256 },
    }, .later_pending = true },
};
