//! nvcc 13.3.73, sample.cu's production flags, independently compiled SASS for all five served targets.
const Override = @import("../registry.zig").Override;
pub const records = [_]Override{
    .{ .name = "sample tf_draw_repetition", .usage = .{ .{ .regs = 51, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 } } },
    .{ .name = "sample tf_greedy_repetition", .usage = .{ .{ .regs = 29, .static = 9752, .bound = 1024 }, .{ .regs = 31, .static = 9752, .bound = 1024 }, .{ .regs = 31, .static = 9752, .bound = 1024 }, .{ .regs = 40, .static = 9752, .bound = 1024 }, .{ .regs = 40, .static = 9752, .bound = 1024 } } },
};
