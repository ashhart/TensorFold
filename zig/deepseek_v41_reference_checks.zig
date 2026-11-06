//! Independent CPU arithmetic boundaries used by the resident-checkpoint teacher oracle.
test {
    _ = @import("tests/deepseek_v41_reference/projection.zig");
    _ = @import("tests/deepseek_v41_reference/math.zig");
    _ = @import("tests/deepseek_v41_reference/attention_math.zig");
    _ = @import("tests/deepseek_v41_reference/qualify.zig");
    _ = @import("tests/deepseek_v41_reference/operator_hc_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_hc_split_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_hc_amplify_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_csa2_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_csa2_pack_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_csa2_sparse_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_csa2_rope_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_csa2_index_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_exl3_projection_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_combine_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_affine_i8_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/wo_a_fusion.zig");
    _ = @import("tests/deepseek_v41_reference/row_invariance.zig");
    _ = @import("tests/deepseek_v41_reference/exl3_int8_tile.zig");
}
