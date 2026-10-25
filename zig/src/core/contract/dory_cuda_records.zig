//! Dory launches await device resource receipts for each supported compiler and SM.
const CudaRecord = @import("registry.zig").CudaRecord;

pub const cuda = [_]CudaRecord{
    .{ .launch = .{ .name = "norms tf_l2_rows", .symbol = "tf_l2_rows", .image = "norms", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
    .{ .launch = .{ .name = "norms tf_slerp_rows", .symbol = "tf_slerp_rows", .image = "norms", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
    .{ .launch = .{ .name = "qk_rows tf_qk_rows_l2 at dim 256", .symbol = "tf_qk_rows_l2", .image = "qk_rows", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
    .{ .launch = .{ .name = "act_rows tf_gate_l2_rows", .symbol = "tf_gate_l2_rows", .image = "act_rows", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
    .{ .launch = .{ .name = "act_rows tf_swiglu_l2_rows", .symbol = "tf_swiglu_l2_rows", .image = "act_rows", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_1", .symbol = "tf_dense_rows_scaled_1", .image = "dense_rows", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_4", .symbol = "tf_dense_rows_scaled_4", .image = "dense_rows", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_16", .symbol = "tf_dense_rows_scaled_16", .image = "dense_rows", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
    .{ .launch = .{ .name = "dense rows tf_dense_rows_scaled_32", .symbol = "tf_dense_rows_scaled_32", .image = "dense_rows", .threads = 256 }, .usage = @import("registry.zig").none, .later_pending = true },
};
