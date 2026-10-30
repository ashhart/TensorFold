//! nvcc 13.2 five-SM receipts for Qwen prefill; nvcc 13.3 produced the same usage.
const Override = @import("../registry.zig").Override;
pub const records = [_]Override{
    .{ .name = "qmm_prefill many bit-exact BF16 launch fusion", .usage = .{ .{ .regs = 172, .static = 0, .bound = 256 }, .{ .regs = 172, .static = 0, .bound = 256 }, .{ .regs = 172, .static = 0, .bound = 256 }, .{ .regs = 128, .static = 1024, .bound = 256 }, .{ .regs = 128, .static = 0, .bound = 256 } } },
    .{ .name = "qmm_prefill 2x2 bit-exact BF16 GB10 large tile", .usage = .{ .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 255, .static = 0, .bound = 128 }, .{ .regs = 220, .static = 0, .bound = 128 }, .{ .regs = 220, .static = 0, .bound = 128 } } },
};
