//! Command queues, command buffers and compute encoders (serial or concurrent), on Metal 3 or Metal 4 (metal4.zig).
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const Buffer = @import("device.zig").Buffer;
const Pipeline = @import("library.zig").Pipeline;
const SharedEvent = @import("sync.zig").SharedEvent;
const Event = @import("sync.zig").Event;
const IndirectCommandBuffer = @import("icb.zig").IndirectCommandBuffer;
const ResidencySet = @import("residency.zig").ResidencySet;
const metal4 = @import("metal4.zig");

const Id = objc.Id;
const Size = types.Size;

pub const Queue = struct {
    id: Id,
    m4: ?*metal4.State = null,

    pub fn deviceId(self: Queue) Id {
        return objc.msg(Id, self.id, "device", .{});
    }

    /// A command buffer that retains what it references (autoreleased: keep a pool around it).
    pub fn commandBuffer(self: Queue) CommandBuffer {
        if (self.m4) |q| return CommandBuffer.of(q.acquire());
        return .{ .id = objc.msg(Id, self.id, "commandBuffer", .{}) };
    }

    /// A command buffer that does not retain its resources: the caller keeps them alive until it completes.
    pub fn commandBufferUnretained(self: Queue) CommandBuffer {
        if (self.m4) |q| return CommandBuffer.of(q.acquire());
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
        if (self.m4) |q| return q.close();
        objc.release(self.id);
    }

    pub fn isMetal4(self: Queue) bool {
        return self.m4 != null;
    }

    /// Names a round of work for the Metal 4 timeline (TF_METAL4_TIMELINE); nothing on Metal 3.
    pub fn roundBegin(self: Queue, name: []const u8) void {
        if (self.m4) |q| q.roundBegin(name);
    }

    /// After the round's last wait.
    pub fn roundEnd(self: Queue) void {
        if (self.m4) |q| q.roundEnd();
    }
};

pub const CommandBuffer = struct {
    id: Id,
    m4: ?*metal4.Slot = null,
    /// the slot's use this handle names
    use: u32 = 0,

    fn of(s: *metal4.Slot) CommandBuffer {
        return .{ .id = s.cb, .m4 = s, .use = s.use };
    }

    pub fn compute(self: CommandBuffer, dispatch: types.DispatchType) ComputeEncoder {
        if (self.m4) |s| return .{ .id = s.compute(dispatch == .serial), .m4 = s };
        return .{ .id = objc.msg(Id, self.id, "computeCommandEncoderWithDispatchType:", .{@backingInt(dispatch)}) };
    }

    /// A blit encoder on this buffer, for copies between buffers on the GPU.
    pub fn blit(self: CommandBuffer) BlitEncoder {
        if (self.m4) |s| return .{ .id = s.blitEncoder(), .m4 = s };
        return .{ .id = objc.msg(Id, self.id, "blitCommandEncoder", .{}) };
    }

    /// The GPU waits here until `event` reaches `value`.
    pub fn waitFor(self: CommandBuffer, event: SharedEvent, value: u64) void {
        if (self.m4) |s| return s.waitEvent(event.id, value);
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    /// The GPU sets `event` to `value` once the work before this point completes.
    pub fn signal(self: CommandBuffer, event: SharedEvent, value: u64) void {
        if (self.m4) |s| return s.signalEvent(event.id, value);
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    /// waitFor and signal with a GPU-only event.
    pub fn waitForEvent(self: CommandBuffer, event: Event, value: u64) void {
        if (self.m4) |s| return s.waitEvent(event.id, value);
        objc.msg(void, self.id, "encodeWaitForEvent:value:", .{ event.id, value });
    }

    pub fn signalEvent(self: CommandBuffer, event: Event, value: u64) void {
        if (self.m4) |s| return s.signalEvent(event.id, value);
        objc.msg(void, self.id, "encodeSignalEvent:value:", .{ event.id, value });
    }

    pub fn useResidencySet(self: CommandBuffer, set: ResidencySet) void {
        objc.msg(void, self.id, "useResidencySet:", .{set.id});
    }

    pub fn commit(self: CommandBuffer) void {
        @import("pass_metrics.zig").commit();
        if (self.m4) |s| return s.commit();
        objc.msg(void, self.id, "commit", .{});
    }

    pub fn wait(self: CommandBuffer) void {
        const metrics = @import("pass_metrics.zig");
        const start = metrics.time();
        if (self.m4) |s| s.wait(self.use) else objc.msg(void, self.id, "waitUntilCompleted", .{});
        if (start != 0) metrics.wait(start, self.gpuSeconds());
    }

    pub fn status(self: CommandBuffer) types.CommandBufferStatus {
        if (self.m4) |s| return s.status(self.use);
        return @fromBackingInt(@intCast(objc.msg(usize, self.id, "status", .{})));
    }

    /// The error text when the command buffer failed, else null; on Metal 4, any failure on its queue not yet reported.
    pub fn failure(self: CommandBuffer) ?[*:0]const u8 {
        if (self.m4) |s| return s.failure(self.use);
        if (self.status() != .@"error") return null;
        return objc.errorText(objc.msg(?Id, self.id, "error", .{}));
    }

    /// GPU execution span in seconds (host clock), valid once completed.
    pub fn gpuSeconds(self: CommandBuffer) f64 {
        return self.gpuEnd() - self.gpuStart();
    }

    pub fn gpuStart(self: CommandBuffer) f64 {
        if (self.m4) |s| return if (s.times(self.use, true)) |t| t[0] else 0;
        return objc.msg(f64, self.id, "GPUStartTime", .{});
    }

    pub fn gpuEnd(self: CommandBuffer) f64 {
        if (self.m4) |s| return if (s.times(self.use, true)) |t| t[1] else 0;
        return objc.msg(f64, self.id, "GPUEndTime", .{});
    }

    /// The GPU's start and end once reported, without waiting for the report (Metal 4), else null.
    pub fn gpuTimes(self: CommandBuffer) ?[2]f64 {
        if (self.m4) |s| return s.times(self.use, false);
        return .{ self.gpuStart(), self.gpuEnd() };
    }

    /// Metal 4's command buffers belong to their queue's slots: retain and release do nothing there.
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
    m4: ?*metal4.Slot = null,

    pub fn copy(self: BlitEncoder, from: Buffer, from_offset: usize, to: Buffer, to_offset: usize, size: usize) void {
        if (self.m4) |s| return s.copy(self.id, from.id, from_offset, to.id, to_offset, size);
        objc.msg(void, self.id, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", .{ from.id, from_offset, to.id, to_offset, size });
    }

    pub fn fill(self: BlitEncoder, buffer: Buffer, offset: usize, size: usize, value: u8) void {
        if (self.m4) |s| return s.fill(self.id, buffer.id, offset, size, value);
        objc.msg(void, self.id, "fillBuffer:range:value:", .{ buffer.id, types.Range{ .location = offset, .length = size }, value });
    }

    pub fn end(self: BlitEncoder) void {
        objc.msg(void, self.id, "endEncoding", .{});
    }
};

pub const ComputeEncoder = struct {
    id: Id,
    m4: ?*metal4.Slot = null,

    pub fn deviceId(self: ComputeEncoder) Id {
        if (self.m4) |s| return s.q.device;
        return objc.msg(Id, self.id, "device", .{});
    }

    pub fn setPipeline(self: ComputeEncoder, pipeline: Pipeline) void {
        objc.msg(void, self.id, "setComputePipelineState:", .{pipeline.id});
        if (self.m4) |s| s.pipeline(pipeline.id);
    }

    pub fn setBuffer(self: ComputeEncoder, buffer: Buffer, offset: usize, index: usize) void {
        if (self.m4) |s| return s.bind(buffer.id, offset, index);
        objc.msg(void, self.id, "setBuffer:offset:atIndex:", .{ buffer.id, offset, index });
    }

    /// Small constants copied into the command stream (at most 4 KB); on Metal 4 into the slot's constants pages.
    pub fn setBytes(self: ComputeEncoder, bytes: []const u8, index: usize) void {
        if (self.m4) |s| return s.bytes(bytes, index);
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
        if (self.m4) |s| s.dispatched(self.id);
    }

    /// Whole threadgroups of `threads` each, their counts (3 u32) read from `args` at `offset` when the dispatch runs.
    pub fn dispatchIndirect(self: ComputeEncoder, args: Buffer, offset: usize, threads: Size) void {
        if (self.m4) |s| return s.dispatchIndirect(self.id, args.id, offset, threads);
        objc.msg(void, self.id, "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:", .{ args.id, offset, threads });
    }

    /// A grid of `grid` threads in groups of `threads` (edge groups may be partial), as MLX dispatches.
    pub fn dispatchThreads(self: ComputeEncoder, grid: Size, threads: Size) void {
        objc.msg(void, self.id, "dispatchThreads:threadsPerThreadgroup:", .{ grid, threads });
        if (self.m4) |s| s.dispatched(self.id);
    }

    /// Later dispatches see every buffer write before this point (concurrent encoders).
    pub fn barrier(self: ComputeEncoder) void {
        if (self.m4) |s| return s.barrier(self.id);
        objc.msg(void, self.id, "memoryBarrierWithScope:", .{types.BarrierScope.buffers});
    }

    /// Declare a resource used through indirect commands or argument buffers (on Metal 4: it joins the residency set).
    pub fn useResource(self: ComputeEncoder, buffer: Buffer, usage: usize) void {
        if (self.m4) |s| {
            _ = s.q.address(buffer.id);
            return;
        }
        objc.msg(void, self.id, "useResource:usage:", .{ buffer.id, usage });
    }

    pub fn execute(self: ComputeEncoder, icb: IndirectCommandBuffer, first: usize, count: usize) void {
        objc.msg(void, self.id, "executeCommandsInBuffer:withRange:", .{ icb.id, types.Range{ .location = first, .length = count } });
        if (self.m4) |s| s.dispatched(self.id);
    }

    pub fn end(self: ComputeEncoder) void {
        if (self.m4) |s| s.ending(self.id);
        objc.msg(void, self.id, "endEncoding", .{});
    }

    /// The command buffer this encoder was opened from, for the event ops a seam encodes BETWEEN encoders.
    pub fn commandBuffer(self: ComputeEncoder) CommandBuffer {
        if (self.m4) |s| return CommandBuffer.of(s);
        return .{ .id = objc.msg(Id, self.id, "commandBuffer", .{}) };
    }

    /// Whether this encoder dispatches serially or concurrently: a seam reopens it in the same mode.
    pub fn dispatchType(self: ComputeEncoder) types.DispatchType {
        if (self.m4) |s| return if (s.serial) .serial else .concurrent;
        return @fromBackingInt(@intCast(objc.msg(usize, self.id, "dispatchType", .{})));
    }
};
