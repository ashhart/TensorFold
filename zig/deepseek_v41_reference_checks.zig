//! Independent CPU arithmetic boundaries used by the resident-checkpoint teacher oracle.
test {
    _ = @import("tests/deepseek_v41_reference/projection.zig");
    _ = @import("tests/deepseek_v41_reference/math.zig");
    _ = @import("tests/deepseek_v41_reference/attention_math.zig");
    _ = @import("tests/deepseek_v41_reference/qualify.zig");
    _ = @import("tests/deepseek_v41_reference/operator_hc_fp64.zig");
    _ = @import("tests/deepseek_v41_reference/operator_csa2_fp64.zig");
}
