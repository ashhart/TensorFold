//! Spark2.5's CUDA launch records: act_rows' head gate and exact GELU, at each launcher's two block sizes.
const CudaRecord = @import("registry.zig").CudaRecord;
const Usage = @FieldType(CudaRecord, "usage");

const gate_usage: Usage = .{ .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 }, .{ .regs = 25, .static = 0, .bound = 256 } };
const gelu_usage: Usage = .{ .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 29, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 }, .{ .regs = 32, .static = 0, .bound = 256 } };

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "act_rows tf_head_gate_rows", .symbol = "tf_head_gate_rows", .image = "act_rows", .threads = 256 }, .usage = gate_usage },
    .{ .launch = .{ .name = "act_rows tf_head_gate_rows, 16 rows or fewer", .symbol = "tf_head_gate_rows", .image = "act_rows", .threads = 64 }, .usage = gate_usage },
    .{ .launch = .{ .name = "act_rows tf_gelu_erf_rows", .symbol = "tf_gelu_erf_rows", .image = "act_rows", .threads = 256 }, .usage = gelu_usage },
    .{ .launch = .{ .name = "act_rows tf_gelu_erf_rows, 16 rows or fewer", .symbol = "tf_gelu_erf_rows", .image = "act_rows", .threads = 128 }, .usage = gelu_usage },
};
