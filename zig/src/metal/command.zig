//! Command queues, command buffers and compute encoders (serial or concurrent dispatch).
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const Buffer = @import("device.zig").Buffer;
const Pipeline = @import("library.zig").Pipeline;
const SharedEvent = @import("sync.zig").SharedEvent;
const Event = @import("sync.zig").Event;
const IndirectCommandBuffer = @import("icb.zig").IndirectCommandBuffer;
const ResidencySet = @import("residency.zig").ResidencySet;

const Id = objc.Id;
const Size = types.Size;

pub const Queue = struct {
    id: Id,

    /// A command buffer that retains what it references (autoreleased: keep a pool around it).
    pub fn commandBuffer(self: Queue) CommandBuffer {
        return .{ .id = objc.msg(Id, self.id, "commandBuffer", .{}) };
    }

    /// A command buffer that does not retain its resources: the caller keeps them alive until it completes.
    pub fn commandBufferUnretained(self: Queue) CommandBuffer {
        return .{ .id = objc.msg(Id, self.id, "commandBufferWithUnretainedReferences", .{}) };
    }

    /// Make `set` resident for every command buffer of this queue.
    pub fn addResidencySet(self: Queue, set: ResidencySet) void {
        objc.msg(void, self.id, "addResidencySet:", .{set.id});
    }

    pub fn removeResidencySet(self: Queue, set: ResidencySet) void {
        objc.msg(void, self.id, "removeResidencySet:", .{set.id});
    }

    pub fn deinit(self: Queue) void {
        objc.release(self.id);
    }
};

pub const CommandBuffer = struct {
    id: Id,

    pub fn compute(self: CommandBuffer, dispatch: types.DispatchType) ComputeEncoder {
        return .{ .id = objc.msg(Id, self.id, "computeCommandEncoderWithDispatchType:", .{@backingInt(dispatch)}) };
    }

    /// The GPU waits here until `event` reaches `value`.
    pub fn waitFor(self: CommandBuffer, event: SharedEvent, value: u64) void {
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    /// The GPU sets `event` to `value` once the work before this point completes.
    pub fn signal(self: CommandBuffer, event: SharedEvent, value: u64) void {
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    /// waitFor and signal with a GPU-only event.
    pub fn waitForEvent(self: CommandBuffer, event: Event, value: u64) void {
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    pub fn signalEvent(self: CommandBuffer, event: Event, value: u64) void {
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    pub fn useResidencySet(self: CommandBuffer, set: ResidencySet) void {
        objc.msg(void, self.id, "useResidencySet:", .{set.id});
    }

    pub fn commit(self: CommandBuffer) void {
        objc.msg(void, self.id, "commit", .{});
    }

    pub fn wait(self: CommandBuffer) void {
        objc.msg(void, self.id, "waitUntilCompleted", .{});
    }

    pub fn status(self: CommandBuffer) types.CommandBufferStatus {
        return @fromBackingInt(@intCast(objc.msg(usize, self.id, "status", .{})));
    }

    /// The error text when the command buffer failed, else null.
    pub fn failure(self: CommandBuffer) ?[*:0]const u8 {
        if (self.status() != .@"error") return null;
        return objc.errorText(objc.msg(?Id, self.id, "error", .{}));
    }

    /// GPU execution span in seconds (host clock), valid once completed.
    pub fn gpuSeconds(self: CommandBuffer) f64 {
        return self.gpuEnd() - self.gpuStart();
    }

    pub fn gpuStart(self: CommandBuffer) f64 {
        return objc.msg(f64, self.id, "GPUStartTime", .{});
    }

    pub fn gpuEnd(self: CommandBuffer) f64 {
        return objc.msg(f64, self.id, "GPUEndTime", .{});
    }

    pub fn retain(self: CommandBuffer) CommandBuffer {
        return .{ .id = objc.retain(self.id) };
    }

    pub fn release(self: CommandBuffer) void {
        objc.release(self.id);
    }
};

pub const ComputeEncoder = struct {
    id: Id,

    pub fn setPipeline(self: ComputeEncoder, pipeline: Pipeline) void {
        objc.msg(void, self.id, "setComputePipelineState:", .{pipeline.id});
    }

    pub fn setBuffer(self: ComputeEncoder, buffer: Buffer, offset: usize, index: usize) void {
        objc.msg(void, self.id, "setBuffer:offset:atIndex:", .{ buffer.id, offset, index });
    }

    /// Small constants copied into the command stream (at most 4 KB).
    pub fn setBytes(self: ComputeEncoder, bytes: []const u8, index: usize) void {
        objc.msg(void, self.id, "setBytes:length:atIndex:", .{ bytes.ptr, bytes.len, index });
    }

    pub fn setValue(self: ComputeEncoder, value: anytype, index: usize) void {
        self.setBytes(std.mem.asBytes(&value), index);
    }

    pub fn setThreadgroupMemory(self: ComputeEncoder, len: usize, index: usize) void {
        objc.msg(void, self.id, "setThreadgroupMemoryLength:atIndex:", .{ len, index });
    }

    /// Whole threadgroups: `groups` of `threads` each.
    pub fn dispatchGroups(self: ComputeEncoder, groups: Size, threads: Size) void {
        objc.msg(void, self.id, "dispatchThreadgroups:threadsPerThreadgroup:", .{ groups, threads });
    }

    /// Whole threadgroups of `threads` each, their counts (3 u32) read from `args` at `offset` when the dispatch runs.
    pub fn dispatchIndirect(self: ComputeEncoder, args: Buffer, offset: usize, threads: Size) void {
        objc.msg(void, self.id, "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:", .{ args.id, offset, threads });
    }

    /// A grid of `grid` threads in groups of `threads` (edge groups may be partial), as MLX dispatches.
    pub fn dispatchThreads(self: ComputeEncoder, grid: Size, threads: Size) void {
        objc.msg(void, self.id, "dispatchThreads:threadsPerThreadgroup:", .{ grid, threads });
    }

    /// Later dispatches see every buffer write before this point (concurrent encoders).
    pub fn barrier(self: ComputeEncoder) void {
        objc.msg(void, self.id, "memoryBarrierWithScope:", .{types.BarrierScope.buffers});
    }

    /// Declare a resource used through indirect commands or argument buffers.
    pub fn useResource(self: ComputeEncoder, buffer: Buffer, usage: usize) void {
        objc.msg(void, self.id, "useResource:usage:", .{ buffer.id, usage });
    }

    pub fn execute(self: ComputeEncoder, icb: IndirectCommandBuffer, first: usize, count: usize) void {
        objc.msg(void, self.id, "executeCommandsInBuffer:withRange:", .{ icb.id, types.Range{ .location = first, .length = count } });
    }

    pub fn end(self: ComputeEncoder) void {
        objc.msg(void, self.id, "endEncoding", .{});
    }
};
