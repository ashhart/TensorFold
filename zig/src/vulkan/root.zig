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
pub const fill = commands.fill;
pub const zero = commands.zero;
pub const recordFill = commands.recordFill;
pub const orExit = commands.orExit;
pub const dispatch = @import("dispatch.zig");
pub const params = @import("stage1_params");
pub const tensor = @import("tensor.zig");
pub const arena = @import("arena.zig");
pub const ops = @import("ops.zig");
pub const ops_q4 = @import("ops_q4.zig");
pub const norm_mlp = @import("norm_mlp.zig");
pub const attention = @import("attention.zig");
pub const Attention = attention.Attention;
pub const gdn = @import("gdn.zig");
pub const Gdn = gdn.Gdn;
pub const greedy = @import("greedy.zig");
pub const Greedy = greedy.Greedy;

test {
    const std = @import("std");
    inline for (.{ runtime, Runtime, device, Device, memory, Buffer, commands, Commands, dispatch, dispatch.Pipeline, params, tensor, tensor.View, arena, ops, ops_q4, norm_mlp, norm_mlp.NormMlp, attention, Attention, gdn, Gdn, greedy, Greedy }) |T|
        std.testing.refAllDecls(T);
}
