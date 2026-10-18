//! nvcc 13.2 usage measured on the candidate; 13.0 remains unmeasured for these launches.
const Override = @import("../registry.zig").Override;
pub const records = [_]Override{
    .{ .name = "glm_glue tf_glm_top1", .usage = .{ .{ .regs = 29, .static = 8192, .bound = 1024 }, .{ .regs = 30, .static = 8192, .bound = 1024 }, .{ .regs = 30, .static = 8192, .bound = 1024 }, .{ .regs = 36, .static = 8192, .bound = 1024 }, .{ .regs = 36, .static = 8192, .bound = 1024 } } },
    .{ .name = "sample tf_draw", .usage = .{ .{ .regs = 52, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 }, .{ .regs = 64, .static = 9752, .bound = 1024 } } },
    .{ .name = "sample tf_grammar_mask", .usage = .{ .{ .regs = 28, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 26, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 }, .{ .regs = 17, .static = 0, .bound = 256 } } },
    .{ .name = "sample tf_logprob_rows", .usage = .{ .{ .regs = 27, .static = 520, .bound = 1024 }, .{ .regs = 28, .static = 520, .bound = 1024 }, .{ .regs = 28, .static = 520, .bound = 1024 }, .{ .regs = 40, .static = 520, .bound = 1024 }, .{ .regs = 40, .static = 520, .bound = 1024 } } },
};
