//! The assembly inventory has geometry queries but cannot create a device, buffer, queue or pipeline.
const real = @import("real_metal");
pub const Device = struct {
    units: bool,
    gen: u32,
    pub fn tensorUnits(d: Device) bool {
        return d.units;
    }
    pub fn generation(d: Device) u32 {
        return d.gen;
    }
};
pub const objc = real.objc;
pub const types = real.types;
pub const Size = real.Size;
pub const ResourceOptions = real.ResourceOptions;
pub const ResourceUsage = real.ResourceUsage;
pub const DispatchType = real.DispatchType;
pub const Buffer = real.Buffer;
pub const MappedFile = real.MappedFile;
pub const Library = real.Library;
pub const CompileOptions = real.CompileOptions;
pub const Pipeline = real.Pipeline;
pub const Queue = real.Queue;
pub const CommandBuffer = real.CommandBuffer;
pub const ComputeEncoder = real.ComputeEncoder;
pub const BlitEncoder = real.BlitEncoder;
pub const SharedEvent = real.SharedEvent;
pub const Event = real.Event;
pub const IndirectCommandBuffer = real.IndirectCommandBuffer;
pub const IndirectCommand = real.IndirectCommand;
pub const ResidencySet = real.ResidencySet;
pub const resident = real.resident;
pub const residency_policy = real.residency_policy;
pub const keepalive = real.keepalive;
pub const clock = real.clock;
pub const pass_metrics = real.pass_metrics;
pub const counters = real.counters;
pub const ane = real.ane;
pub const CommandSessionFor = real.CommandSessionFor;
pub const CommandSession = real.CommandSession;
