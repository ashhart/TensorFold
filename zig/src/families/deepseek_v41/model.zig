//! Backend-neutral checkpoint owners. CUDA rank slicing is an explicit capability adapter.
pub const Config = @import("config.zig").Config;
pub const Plan = @import("plan.zig").CheckpointPlan;
pub const CudaRankPlan = @import("plan.zig").Plan;
pub const Image = @import("named.zig").Named;
