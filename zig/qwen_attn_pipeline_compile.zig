//! Analyze actual startup and contract bodies with inert images; no CUDA or NVCC execution.
export fn pipelineContract() usize {
    return @intFromPtr(&@import("contract").run);
}
export fn pipelineStartup() usize {
    return @intFromPtr(&@import("qwen35").engine.Engine.init);
}
