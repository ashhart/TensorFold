//! mHC launches; nvcc 13.0 resources live in measurements/main_130.zig. Releases 13.1 through 13.3 remain pending.
const registry = @import("registry.zig");
const R = registry.CudaRecord;
const O = registry.Override;
pub const cuda = [_]R{
    .{ .launch = .{ .name = "qwen35_mhc enter", .symbol = "qwen35_mhc_enter", .image = "qwen35_mhc", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "qwen35_mhc route", .symbol = "qwen35_mhc_route", .image = "qwen35_mhc", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "qwen35_mhc merge", .symbol = "qwen35_mhc_merge", .image = "qwen35_mhc", .threads = 256 }, .usage = registry.none },
    .{ .launch = .{ .name = "qwen35_mhc leave", .symbol = "qwen35_mhc_leave", .image = "qwen35_mhc", .threads = 256 }, .usage = registry.none },
};
pub const cuda_134 = [_]O{
    .{ .name = "qwen35_mhc enter", .usage = .{ .{ .regs = 18, .static = 0 }, .{ .regs = 18, .static = 0 }, .{ .regs = 18, .static = 0 }, .{ .regs = 17, .static = 0 }, .{ .regs = 17, .static = 0 } } },
    .{ .name = "qwen35_mhc route", .usage = .{ .{ .regs = 40, .static = 1068 }, .{ .regs = 40, .static = 1068 }, .{ .regs = 40, .static = 1068 }, .{ .regs = 39, .static = 1068 }, .{ .regs = 39, .static = 1068 } } },
    .{ .name = "qwen35_mhc merge", .usage = .{ .{ .regs = 28, .static = 0 }, .{ .regs = 29, .static = 0 }, .{ .regs = 29, .static = 0 }, .{ .regs = 28, .static = 0 }, .{ .regs = 28, .static = 0 } } },
    .{ .name = "qwen35_mhc leave", .usage = .{ .{ .regs = 22, .static = 0 }, .{ .regs = 24, .static = 0 }, .{ .regs = 24, .static = 0 }, .{ .regs = 18, .static = 0 }, .{ .regs = 18, .static = 0 } } },
};
