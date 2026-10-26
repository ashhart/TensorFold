//! The Vulkan backend as one module: the loader and instance, device selection, buffers, submission and dispatch.

/// The Vulkan C API's types and constants (vulkan_core.h without prototypes).
pub const c = @import("vk");
pub const runtime = @import("runtime.zig");
pub const Runtime = runtime.Runtime;
pub const Error = runtime.Error;
pub const check = runtime.check;
pub const device = @import("device.zig");
pub const Device = device.Device;
pub const Features = device.Features;
pub const memory = @import("memory.zig");
pub const Buffer = memory.Buffer;
pub const commands = @import("commands.zig");
pub const Commands = commands.Commands;
pub const barrier = commands.barrier;
pub const upload = commands.upload;
pub const download = commands.download;
pub const orExit = commands.orExit;
pub const dispatch = @import("dispatch.zig");

test {
    const std = @import("std");
    inline for (.{ runtime, Runtime, device, Device, memory, Buffer, commands, Commands, dispatch, dispatch.Pipeline }) |T|
        std.testing.refAllDecls(T);
}
