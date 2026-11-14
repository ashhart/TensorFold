//! Nemotron's batched draw entries; usage from the full kernel contract on a GB10 (sm_80/86/89/120/121).
const registry = @import("registry.zig");

pub const records = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "cuda sample tf_draw_ids_rows", .symbol = "tf_draw_ids_rows", .image = "sample", .threads = 1024 }, .usage = .{ .{ .regs = 64, .static = 19504, .bound = 1024 }, .{ .regs = 64, .static = 19504, .bound = 1024 }, .{ .regs = 64, .static = 19504, .bound = 1024 }, .{ .regs = 64, .static = 19504, .bound = 1024 }, .{ .regs = 64, .static = 19504, .bound = 1024 } } },
    .{ .launch = .{ .name = "nemotron_ops tf_nemo_gather_rows", .symbol = "tf_nemo_gather_rows", .image = "nemotron_ops", .threads = 256 }, .usage = .{ .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 24, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "nemotron_ops tf_nemo_gather_last", .symbol = "tf_nemo_gather_last", .image = "nemotron_ops", .threads = 256 }, .usage = .{ .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 22, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 }, .{ .regs = 36, .static = 0, .bound = 0 } } },
    .{ .launch = .{ .name = "nemotron_ops tf_nemo_scatter_rows", .symbol = "tf_nemo_scatter_rows", .image = "nemotron_ops", .threads = 32 }, .usage = .{ .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 14, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 }, .{ .regs = 16, .static = 0, .bound = 0 } } },
};
