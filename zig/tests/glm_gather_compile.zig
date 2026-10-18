//! Compile the GPU gather gate's test bodies without running them or invoking a shader compiler.
const checks = @import("glm_rows_check.zig");
export fn glmGatherCompile(g: *const checks.Gpu) u8 {
    @import("glm_gather_check.zig").check(g) catch return 1;
    return 0;
}
