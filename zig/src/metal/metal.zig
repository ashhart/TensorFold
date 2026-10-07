//! TensorFold's Metal runtime: our own Objective-C bindings over Metal and Foundation, no MLX.
const std = @import("std");

pub const objc = @import("objc.zig");
pub const types = @import("types.zig");
pub const Size = types.Size;
pub const ResourceOptions = types.ResourceOptions;
pub const ResourceUsage = types.ResourceUsage;
pub const DispatchType = types.DispatchType;

pub const Device = @import("device.zig").Device;
pub const Buffer = @import("device.zig").Buffer;
pub const MappedFile = @import("device.zig").MappedFile;
pub const Library = @import("library.zig").Library;
pub const CompileOptions = @import("library.zig").CompileOptions;
pub const Pipeline = @import("library.zig").Pipeline;
pub const Queue = @import("command.zig").Queue;
pub const CommandBuffer = @import("command.zig").CommandBuffer;
pub const ComputeEncoder = @import("command.zig").ComputeEncoder;
pub const SharedEvent = @import("sync.zig").SharedEvent;
pub const Event = @import("sync.zig").Event;
pub const IndirectCommandBuffer = @import("icb.zig").IndirectCommandBuffer;
pub const IndirectCommand = @import("icb.zig").Command;
pub const ResidencySet = @import("residency.zig").ResidencySet;
pub const keepalive = @import("keepalive.zig");
pub const clock = @import("clock.zig");

test {
    std.testing.refAllDecls(@This());
}
