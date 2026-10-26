//! Nemotron-Labs Dory 1.0 (a looped nGPT stack) on CUDA: config, weights, schedule, memory, float64 reference.

pub const config = @import("config.zig");
pub const Config = config.Config;
pub const names = @import("names.zig");
pub const schedule = @import("schedule.zig");
pub const memory = @import("memory.zig");
/// The float64 forward the kernels and the CUDA model are checked against.
pub const ref = @import("ref.zig");
pub const cuda_weights = @import("cuda_weights.zig");
pub const cuda_model = @import("cuda_model.zig");
pub const cuda_backend = @import("cuda_backend.zig");
pub const cuda_snap = @import("cuda_snap.zig");
pub const cuda_pages = @import("cuda_pages.zig");
pub const cuda_run = @import("cuda_run.zig");
/// The native CUDA registry's entry (native/cuda.zig).
pub const native = @import("cuda_native.zig");

test {
    _ = @import("config.zig");
    _ = @import("names.zig");
    _ = @import("schedule.zig");
    _ = @import("memory.zig");
    _ = @import("ref.zig");
    _ = @import("cuda_weights.zig");
    _ = @import("cuda_model.zig");
    _ = @import("cuda_kernels.zig");
    _ = @import("cuda_backend.zig");
    _ = @import("cuda_snap.zig");
    _ = @import("cuda_pages.zig");
    _ = @import("cuda_run.zig");
    _ = @import("cuda_native.zig");
}
