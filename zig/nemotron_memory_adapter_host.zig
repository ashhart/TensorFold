//! Analyze the Metal learner's real snapshot callbacks without running a device or loading a checkpoint.
export fn compileNemotronMemoryAdapter() usize {
    return @intFromPtr(&@import("tensorfold").nemotron.memory_adapter.Adapter.backend);
}
