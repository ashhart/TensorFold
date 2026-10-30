//! Nemotron-H (Nemotron 3.5 Lightning) for the native HIP engine: admission, the plan, the loader and the forward.
pub const config = @import("config.zig");
pub const Config = config.Config;
pub const tensors = @import("hip/tensors.zig");
pub const plan = @import("hip/plan.zig");
pub const admit = @import("hip/admit.zig");
pub const model = @import("hip/model.zig");
pub const load = @import("hip/load.zig");
pub const pages = @import("hip/pages.zig");
/// The Mamba-2 mixer's and the MoE layer's interfaces, which their kernels fill.
pub const mamba = @import("hip/mamba.zig");
pub const moe = @import("hip/moe.zig");
pub const forward = @import("hip/forward.zig");
pub const round = @import("hip/round.zig");
pub const graphs = @import("hip/graphs.zig");
pub const costs = @import("hip/costs.zig");
pub const engine = @import("hip/engine.zig");
/// The lane backend the native server drives, its draft head and its prompt cache.
pub const lanes = @import("hip/lanes.zig");
pub const head = @import("hip/head.zig");
pub const prefix = @import("hip/prefix.zig");
pub const worker = @import("hip/worker.zig");
/// The registry entry of the native server.
pub const native = @import("hip/native.zig");

test {
    _ = @import("hip/prompt_test.zig");
    _ = config;
    _ = tensors;
    _ = plan;
    _ = admit;
    _ = model;
    _ = pages;
    _ = mamba;
    _ = moe;
    _ = forward;
    _ = round;
    _ = graphs;
    _ = costs;
    _ = native;
    _ = lanes;
    _ = head;
    _ = prefix;
    _ = worker;
    @import("std").testing.refAllDecls(load);
    @import("std").testing.refAllDecls(forward);
    @import("std").testing.refAllDecls(engine.Engine);
    @import("std").testing.refAllDecls(pages.Stream);
    @import("std").testing.refAllDecls(pages.Pool);
    @import("std").testing.refAllDecls(native);
}
