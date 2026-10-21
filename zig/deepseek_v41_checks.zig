//! Source-built V4.1 operator qualification; real Metal execution, no checkpoint needed.
test {
    _ = @import("src/core/exl3_projection_gpu_test.zig");
    _ = @import("src/families/deepseek_v41/engram.zig");
    _ = @import("src/families/deepseek_v41/csa2.zig");
    _ = @import("src/families/deepseek_v41/ops_check.zig");
    _ = @import("src/families/deepseek_v41/load.zig");
    _ = @import("src/families/deepseek_v41/index_precision_test.zig");
    _ = @import("src/families/deepseek_v41/config.zig");
    _ = @import("src/families/deepseek_v41/profiler.zig");
    _ = @import("src/core/gpu_counters.zig");
    _ = @import("src/core/expert_groups.zig");
}
