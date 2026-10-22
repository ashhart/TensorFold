//! Decode attention usage supplied from nvcc 13.3; other compiler releases remain unmeasured.
const Override = @import("../registry.zig").Override;
pub const records = [_]Override{
    .{ .name = "qwen35 tf_q35_attn_partial", .usage = .{ .{ .regs = 167, .static = 11648, .bound = 128 }, .{ .regs = 167, .static = 11648, .bound = 128 }, .{ .regs = 167, .static = 11648, .bound = 128 }, .{ .regs = 168, .static = 11648, .bound = 128 }, .{ .regs = 168, .static = 11648, .bound = 128 } } },
    .{ .name = "qwen35 tf_q35_attn_rows", .usage = .{ .{ .regs = 254, .static = 6912, .bound = 256 }, .{ .regs = 254, .static = 6912, .bound = 256 }, .{ .regs = 254, .static = 6912, .bound = 256 }, .{ .regs = 246, .static = 6912, .bound = 256 }, .{ .regs = 246, .static = 6912, .bound = 256 } } },
    .{ .name = "qwen35_kv8 tf_q35_attn_partial8", .usage = .{ .{ .regs = 199, .static = 44672, .bound = 128 }, .{ .regs = 199, .static = 44672, .bound = 128 }, .{ .regs = 207, .static = 44672, .bound = 128 }, .{ .regs = 197, .static = 44672, .bound = 128 }, .{ .regs = 197, .static = 44672, .bound = 128 } } },
    .{ .name = "qwen35_kv8 tf_q35_attn_rows8", .usage = .{ .{ .regs = 255, .static = 39936, .bound = 256 }, .{ .regs = 255, .static = 39936, .bound = 256 }, .{ .regs = 255, .static = 39936, .bound = 256 }, .{ .regs = 245, .static = 39936, .bound = 256 }, .{ .regs = 245, .static = 39936, .bound = 256 } } },
};
