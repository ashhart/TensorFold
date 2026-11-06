//! Command queues, command buffers and compute encoders (serial or concurrent dispatch), on Metal 3 or Metal 4.
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const metal4 = @import("metal4.zig");
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
    /// A Metal 4 context's queue: its command buffers come from the context, which owns them and the queue.
    m4: ?*metal4.Context = null,

    pub fn deviceId(self: Queue) Id {
        return objc.msg(Id, self.id, "device", .{});
    }

    /// A command buffer that retains what it references (autoreleased: keep a pool around it).
    pub fn commandBuffer(self: Queue) CommandBuffer {
        if (self.m4) |c| return c.commandBuffer();
        return .{ .id = objc.msg(Id, self.id, "commandBuffer", .{}) };
    }

    /// A command buffer that does not retain its resources: the caller keeps them alive until it completes.
    pub fn commandBufferUnretained(self: Queue) CommandBuffer {
        if (self.m4) |c| return c.commandBuffer();
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
        if (self.m4 == null) objc.release(self.id);
    }
};

pub const CommandBuffer = struct {
    id: Id,
    m4: ?*metal4.Context = null,
    slot: u8 = 0,

    pub fn compute(self: CommandBuffer, dispatch: types.DispatchType) ComputeEncoder {
        if (self.m4) |c| return .{ .id = c.encoder(self.id), .m4 = c, .serial = dispatch == .serial };
        return .{ .id = objc.msg(Id, self.id, "computeCommandEncoderWithDispatchType:", .{@backingInt(dispatch)}) };
    }

    /// A blit encoder on this buffer, for copies between buffers on the GPU.
    pub fn blit(self: CommandBuffer) BlitEncoder {
        if (self.m4 != null) @panic("metal 4: no blit encoders; copies stay on Metal 3");
        return .{ .id = objc.msg(Id, self.id, "blitCommandEncoder", .{}) };
    }

    /// The GPU waits here until `event` reaches `value`.
    pub fn waitFor(self: CommandBuffer, event: SharedEvent, value: u64) void {
        if (self.m4 != null) @panic("metal 4: events between command buffers stay on Metal 3");
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    /// The GPU sets `event` to `value` once the work before this point completes.
    pub fn signal(self: CommandBuffer, event: SharedEvent, value: u64) void {
        if (self.m4 != null) @panic("metal 4: events between command buffers stay on Metal 3");
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    /// waitFor and signal with a GPU-only event.
    pub fn waitForEvent(self: CommandBuffer, event: Event, value: u64) void {
        if (self.m4 != null) @panic("metal 4: events between command buffers stay on Metal 3");
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    pub fn signalEvent(self: CommandBuffer, event: Event, value: u64) void {
        if (self.m4 != null) @panic("metal 4: events between command buffers stay on Metal 3");
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    pub fn useResidencySet(self: CommandBuffer, set: ResidencySet) void {
        objc.msg(void, self.id, "useResidencySet:", .{set.id});
    }

    pub fn commit(self: CommandBuffer) void {
        @import("pass_metrics.zig").commit();
        if (self.m4) |c| return c.commit(self.slot);
        objc.msg(void, self.id, "commit", .{});
    }

    pub fn wait(self: CommandBuffer) void {
        const metrics = @import("pass_metrics.zig");
        const start = metrics.time();
        if (self.m4) |c| c.wait(self.slot) else objc.msg(void, self.id, "waitUntilCompleted", .{});
        if (start != 0) metrics.wait(start, self.gpuSeconds());
    }

    pub fn status(self: CommandBuffer) types.CommandBufferStatus {
        if (self.m4) |c| return c.status(self.slot);
        return @fromBackingInt(@intCast(objc.msg(usize, self.id, "status", .{})));
    }

    /// The error text when the command buffer failed, else null.
    pub fn failure(self: CommandBuffer) ?[*:0]const u8 {
        if (self.m4) |c| return c.failure(self.slot);
        if (self.status() != .@"error") return null;
        return objc.errorText(objc.msg(?Id, self.id, "error", .{}));
    }

    /// GPU execution span in seconds (host clock), valid once completed.
    pub fn gpuSeconds(self: CommandBuffer) f64 {
        return self.gpuEnd() - self.gpuStart();
    }

    pub fn gpuStart(self: CommandBuffer) f64 {
        if (self.m4) |c| return c.start[self.slot];
        return objc.msg(f64, self.id, "GPUStartTime", .{});
    }

    pub fn gpuEnd(self: CommandBuffer) f64 {
        if (self.m4) |c| return c.end[self.slot];
        return objc.msg(f64, self.id, "GPUEndTime", .{});
    }

    /// A Metal 4 context owns its command buffers, so they need no retain.
    pub fn retain(self: CommandBuffer) CommandBuffer {
        if (self.m4 != null) return self;
        return .{ .id = objc.retain(self.id) };
    }

    pub fn release(self: CommandBuffer) void {
        if (self.m4 == null) objc.release(self.id);
    }
};

/// Copies between buffers on the GPU.
pub const BlitEncoder = struct {
    id: Id,

    pub fn copy(self: BlitEncoder, from: Buffer, from_offset: usize, to: Buffer, to_offset: usize, size: usize) void {
        objc.msg(void, self.id, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", .{ from.id, from_offset, to.id, to_offset, size });
    }

    pub fn fill(self: BlitEncoder, buffer: Buffer, offset: usize, size: usize, value: u8) void {
        objc.msg(void, self.id, "fillBuffer:range:value:", .{ buffer.id, types.Range{ .location = offset, .length = size }, value });
    }

    pub fn end(self: BlitEncoder) void {
        objc.msg(void, self.id, "endEncoding", .{});
    }
};

pub const ComputeEncoder = struct {
    id: Id,
    m4: ?*metal4.Context = null,
    /// Metal 4 encoders run their dispatches side by side, so a serial one puts a barrier after each.
    serial: bool = false,

    pub fn deviceId(self: ComputeEncoder) Id {
        if (self.m4) |c| return c.device.id;
        return objc.msg(Id, self.id, "device", .{});
    }

    pub fn setPipeline(self: ComputeEncoder, pipeline: Pipeline) void {
        objc.msg(void, self.id, "setComputePipelineState:", .{pipeline.id});
    }

    pub fn setBuffer(self: ComputeEncoder, buffer: Buffer, offset: usize, index: usize) void {
        if (self.m4) |c| return c.bind(buffer, offset, index);
        objc.msg(void, self.id, "setBuffer:offset:atIndex:", .{ buffer.id, offset, index });
    }

    /// Small constants copied into the command stream (at most 4 KB).
    pub fn setBytes(self: ComputeEncoder, bytes: []const u8, index: usize) void {
        if (self.m4) |c| return c.constant(bytes, index);
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
        self.dispatched();
    }

    /// Whole threadgroups of `threads` each, their counts (3 u32) read from `args` at `offset` when the dispatch runs.
    pub fn dispatchIndirect(self: ComputeEncoder, args: Buffer, offset: usize, threads: Size) void {
        if (self.m4) |c| {
            c.dispatchIndirect(self.id, args, offset, threads);
            return self.dispatched();
        }
        objc.msg(void, self.id, "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:", .{ args.id, offset, threads });
    }

    /// A grid of `grid` threads in groups of `threads` (edge groups may be partial), as MLX dispatches.
    pub fn dispatchThreads(self: ComputeEncoder, grid: Size, threads: Size) void {
        objc.msg(void, self.id, "dispatchThreads:threadsPerThreadgroup:", .{ grid, threads });
        self.dispatched();
    }

    fn dispatched(self: ComputeEncoder) void {
        const c = self.m4 orelse return;
        c.pending = true;
        if (self.serial) c.barrier(self.id);
    }

    /// Later dispatches see every buffer write before this point (concurrent encoders).
    pub fn barrier(self: ComputeEncoder) void {
        if (self.m4) |c| return c.barrier(self.id);
        objc.msg(void, self.id, "memoryBarrierWithScope:", .{types.BarrierScope.buffers});
    }

    /// Declare a resource used through indirect commands or argument buffers.
    pub fn useResource(self: ComputeEncoder, buffer: Buffer, usage: usize) void {
        if (self.m4) |c| {
            _ = c.resident(buffer);
            return;
        }
        objc.msg(void, self.id, "useResource:usage:", .{ buffer.id, usage });
    }

    pub fn execute(self: ComputeEncoder, icb: IndirectCommandBuffer, first: usize, count: usize) void {
        if (self.m4 != null) @panic("metal 4: indirect command buffers stay on Metal 3");
        objc.msg(void, self.id, "executeCommandsInBuffer:withRange:", .{ icb.id, types.Range{ .location = first, .length = count } });
    }

    pub fn end(self: ComputeEncoder) void {
        objc.msg(void, self.id, "endEncoding", .{});
    }

    /// The command buffer this encoder was opened from, for the event ops a seam encodes BETWEEN encoders.
    pub fn commandBuffer(self: ComputeEncoder) CommandBuffer {
        if (self.m4 != null) @panic("metal 4: seams stay on Metal 3");
        return .{ .id = objc.msg(Id, self.id, "commandBuffer", .{}) };
    }

    /// Whether this encoder dispatches serially or concurrently: a seam reopens it in the same mode.
    pub fn dispatchType(self: ComputeEncoder) types.DispatchType {
        if (self.m4 != null) return if (self.serial) .serial else .concurrent;
        return @fromBackingInt(@intCast(objc.msg(usize, self.id, "dispatchType", .{})));
    }
};
