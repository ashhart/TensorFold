//! tf_exl3_plan_finish_after: nvcc 13.0's usage; 13.2 pending, 13.3 in measurements/pending_133.zig.
const registry = @import("registry.zig");
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "exl3_experts tf_exl3_plan_finish_after", .symbol = "tf_exl3_plan_finish_after", .image = "exl3_experts", .threads = 32 }, .usage = .{ .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 }, .{ .regs = 40, .static = 0, .bound = 0 } }, .later_pending = true },
};
