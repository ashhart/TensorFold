//! Indirect command buffers for compute: dispatches encoded once on the host, replayed by the GPU.
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const Device = @import("device.zig").Device;
const Buffer = @import("device.zig").Buffer;
const Pipeline = @import("library.zig").Pipeline;

const Id = objc.Id;

pub const Error = error{NoIndirectCommandBuffer};

pub const IndirectCommandBuffer = struct {
    id: Id,
    count: usize,

    /// Room for `count` compute dispatches, each binding at most `max_buffers` buffers of its own.
    pub fn init(device: Device, count: usize, max_buffers: usize) Error!IndirectCommandBuffer {
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTLIndirectCommandBufferDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        objc.msg(void, desc, "setCommandTypes:", .{types.IndirectCommandType.concurrent_dispatch | types.IndirectCommandType.concurrent_dispatch_threads});
        objc.msg(void, desc, "setInheritPipelineState:", .{false});
        objc.msg(void, desc, "setInheritBuffers:", .{false});
        objc.msg(void, desc, "setMaxKernelBufferBindCount:", .{max_buffers});
        const id = objc.msg(?Id, device.id, "newIndirectCommandBufferWithDescriptor:maxCommandCount:options:", .{ desc, count, types.ResourceOptions.shared });
        return .{ .id = id orelse return error.NoIndirectCommandBuffer, .count = count };
    }

    pub fn command(self: IndirectCommandBuffer, index: usize) Command {
        std.debug.assert(index < self.count);
        return .{ .id = objc.msg(Id, self.id, "indirectComputeCommandAtIndex:", .{index}) };
    }

    pub fn reset(self: IndirectCommandBuffer) void {
        objc.msg(void, self.id, "resetWithRange:", .{types.Range{ .location = 0, .length = self.count }});
    }

    pub fn deinit(self: IndirectCommandBuffer) void {
        objc.release(self.id);
    }
};

/// One dispatch in an indirect command buffer; re-encoding it changes the next replay's arguments.
pub const Command = struct {
    id: Id,

    pub fn setPipeline(self: Command, pipeline: Pipeline) void {
        objc.msg(void, self.id, "setComputePipelineState:", .{pipeline.id});
    }

    pub fn setBuffer(self: Command, buffer: Buffer, offset: usize, index: usize) void {
        objc.msg(void, self.id, "setKernelBuffer:offset:atIndex:", .{ buffer.id, offset, index });
    }

    pub fn dispatchGroups(self: Command, groups: types.Size, threads: types.Size) void {
        objc.msg(void, self.id, "concurrentDispatchThreadgroups:threadsPerThreadgroup:", .{ groups, threads });
    }

    pub fn dispatchThreads(self: Command, grid: types.Size, threads: types.Size) void {
        objc.msg(void, self.id, "concurrentDispatchThreads:threadsPerThreadgroup:", .{ grid, threads });
    }

    /// This dispatch starts only after every earlier one in the buffer completes.
    pub fn setBarrier(self: Command) void {
        objc.msg(void, self.id, "setBarrier", .{});
    }

    pub fn clearBarrier(self: Command) void {
        objc.msg(void, self.id, "clearBarrier", .{});
    }
};
