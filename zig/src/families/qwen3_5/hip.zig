//! Qwen3.5 / Qwen3.6 text checkpoints for the native HIP engine: configuration, host index and upload.

pub const config = @import("weights/config.zig");
pub const table = @import("weights/table.zig");
pub const host = @import("weights/host.zig");
pub const checkpoint = @import("weights/checkpoint.zig");
pub const weights = @import("backend/hip/model/weights.zig");
pub const Config = config.Config;
pub const Spec = config.Spec;
pub const Checkpoint = checkpoint.Checkpoint;
pub const Model = weights.Model;
pub const view = @import("backend/hip/model/view.zig");
pub const state = @import("backend/hip/forward/state.zig");
pub const pages = @import("backend/hip/forward/pages.zig");
pub const forward = @import("backend/hip/forward/forward.zig");
pub const window = @import("backend/hip/forward/window.zig");
pub const plan = @import("backend/hip/forward/plan.zig");
pub const moe = @import("backend/hip/forward/moe.zig");
pub const sample = @import("backend/hip/engine/sample.zig");
pub const draw = @import("backend/hip/engine/draw.zig");
pub const bridge = @import("backend/hip/model/bridge.zig");
pub const engine = @import("backend/hip/engine/engine.zig");
pub const memory = @import("backend/hip/engine/memory.zig");
pub const prefix = @import("backend/hip/engine/prefix.zig");
pub const radix = @import("engine_api").prompt_radix;
pub const hip_lanes = @import("backend/hip/engine/hip_lanes.zig");
pub const mtp = @import("backend/hip/engine/mtp.zig");
pub const worker = @import("backend/hip/engine/worker.zig");
pub const slicing = @import("weights/slicing.zig");
pub const reduce = @import("backend/hip/forward/reduce.zig");
pub const native_engine = @import("backend/hip/engine/native.zig");
/// The registry entries of the native server: the dense and the sparse checkpoints.
pub const native = native_engine.Native("qwen3_5");
pub const native_moe = native_engine.Native("qwen3_5_moe");

test {
    _ = config;
    _ = table;
    _ = @import("weights/shard.zig");
    _ = @import("weights/projection.zig");
    _ = @import("weights/checkpoint_test.zig");
    _ = @import("weights/real_test.zig");
    _ = view;
    _ = moe;
    _ = sample;
    _ = prefix;
    _ = radix;
    _ = native_engine;
    _ = pages;
    _ = plan;
    _ = slicing;
    // the engine and its workers compile on a host with no GPU
    @import("std").testing.refAllDecls(hip_lanes.Hip);
    @import("std").testing.refAllDecls(worker.Worker);
    @import("std").testing.refAllDecls(engine.Engine);
}
