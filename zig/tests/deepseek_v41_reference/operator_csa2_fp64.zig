//! CSA2 FP64 qualification, grouped by operator boundary.
test {
    _ = @import("operator_csa2_fp64_pool.zig");
    _ = @import("operator_csa2_fp64_sparse.zig");
    _ = @import("operator_csa2_fp64_sink.zig");
    _ = @import("operator_csa2_fp64_rope.zig");
    _ = @import("operator_csa2_fp64_index.zig");
}
