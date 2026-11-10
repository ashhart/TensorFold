//! Validate GLM's TF_GLM_* settings on every rank before its CUDA open reads ranks, NCCL or weights.
const environment = @import("sys").environment;

/// Every `.glm` row of the shared table, parsed exactly as its later reader parses it.
pub fn startup() !void {
    try environment.checkScope(.glm, environment.Process{});
}

test {
    _ = @import("cuda_env_test.zig");
}
