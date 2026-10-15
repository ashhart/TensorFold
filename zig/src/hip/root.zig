//! The HIP backend as one module: runtime, the model-free kernels' launches and ops, and what an engine opens.

pub const abi = @import("abi.zig");
pub const runtime = @import("runtime.zig");
pub const Runtime = runtime.Runtime;
pub const Error = runtime.Error;
pub const Context = @import("context.zig").Context;
pub const memory = @import("memory.zig");
pub const DeviceBuffer = memory.DeviceBuffer;
pub const HostBuffer = memory.HostBuffer;
pub const usage = memory.usage;
pub const Usage = memory.Usage;
pub const Stream = @import("stream.zig").Stream;
pub const Module = @import("module.zig").Module;
pub const Function = @import("module.zig").Function;
pub const launch = @import("launch.zig");
pub const graph = @import("graph.zig");
pub const graph_cache = @import("graph_cache.zig");
pub const graph_split = @import("graph_split.zig");
pub const Arena = @import("arena.zig").Arena;
pub const caps = @import("caps.zig");
pub const Caps = caps.Caps;
pub const kernels = @import("kernels.zig");
pub const launches = @import("launches.zig");
pub const Launcher = launches.Launcher;
/// The stream handle a dry run passes: launches do nothing on it.
pub const counting = @import("launches/util.zig").counting;
pub const raw = @import("raw.zig");
pub const ops = @import("ops/ops.zig");
pub const plan_ops = @import("ops/plan.zig");
pub const policy = @import("policy.zig");
pub const Policy = policy.Policy;
pub const admission = @import("admission.zig");
pub const Device = @import("device.zig").Device;
pub const Upload = @import("upload.zig").Upload;
pub const quant = @import("core").quant;
pub const registry = @import("core").registry;
pub const collective = @import("collective.zig");
pub const moe = @import("moe.zig");
pub const reduce = @import("reduce.zig");
/// Joins this rank's tensor-parallel group: RCCL's (comm/), behind collective.zig.
pub const joinGroup = @import("comm/group.zig").join;
pub const joinRun = @import("comm/group.zig").joinRun;
/// What this rank runs, digested for its join (executable, config.json, weights).
pub const identity = @import("comm/identity.zig");
pub const Startup = @import("startup.zig").Startup;

test {
    _ = policy;
    _ = admission;
    _ = Device;
    _ = collective;
    _ = graph_cache;
    _ = moe;
    _ = reduce;
    _ = @import("comm/rccl.zig");
    _ = @import("comm/link.zig");
    _ = @import("comm/link_test.zig");
    _ = @import("comm/group.zig");
    _ = @import("comm/identity.zig");
    _ = @import("agree.zig");
    _ = @import("startup.zig");
}
