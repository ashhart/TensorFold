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
pub const rccl = @import("comm/rccl.zig");
pub const link = @import("comm/link.zig");
pub const Group = @import("comm/group.zig").Group;

test {
    _ = policy;
    _ = admission;
    _ = Device;
    _ = rccl;
    _ = link;
    _ = Group;
}
