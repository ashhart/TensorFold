//! Qwen3.5-2B on the CPU: the host-memory model, its state, forward and lane backend; Metal's 2B is qwen3_5.zig.
pub const config = @import("config.zig");
pub const Model = @import("cpu_model.zig").Model;
pub const state = @import("cpu_state.zig");
pub const forward = @import("cpu_forward.zig");
pub const backend = @import("cpu_backend.zig");
pub const kernels = @import("cpu_kernels.zig");
pub const ops = @import("cpu_ops.zig");
pub const mixers = @import("cpu_mixers.zig");
pub const qualify = @import("cpu_qualify.zig");
pub const pool = @import("cpu_pool.zig");

test {
    _ = kernels;
    _ = ops;
    _ = mixers;
    _ = @import("cpu_model.zig");
    _ = state;
    _ = pool;
    _ = backend;
    _ = qualify;
    _ = @import("draw.zig");
    _ = config;
}
