//! Metal counter sampling; owned resources must outlive every command buffer using them.
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const device = @import("device.zig");
const command = @import("command.zig");

extern "c" var MTLCommonCounterSetTimestamp: objc.Id;
extern "c" var MTLCommonCounterTimestamp: objc.Id;

pub const error_value: u64 = std.math.maxInt(u64);
pub const max_samples: usize = 131072;
const max_counter_sets: usize = 64;
pub const TimestampResult = extern struct { timestamp: u64 };
pub const SamplingPoint = enum(usize) { stage = 0, draw = 1, dispatch = 2, tile_dispatch = 3, blit = 4 };
pub const Mode = enum { unsupported, dispatch, stage };
pub const ClockSample = struct { cpu: u64 = 0, gpu: u64 = 0 };

pub const Capability = struct {
    timestamp_set: bool = false,
    timestamp_counter: bool = false,
    clock_calibration: bool = false,
    stage_boundary: bool = false,
    dispatch_boundary: bool = false,

    pub fn mode(self: Capability) Mode {
        if (!self.timestamp_set or !self.timestamp_counter or !self.clock_calibration) return .unsupported;
        if (self.dispatch_boundary) return .dispatch;
        if (self.stage_boundary) return .stage;
        return .unsupported;
    }
};

fn timestampSet(dev: device.Device) ?objc.Id {
    const sets = objc.msg(?objc.Id, dev.id, "counterSets", .{}) orelse return null;
    const count = objc.msg(usize, sets, "count", .{});
    if (count > max_counter_sets) return null;
    for (0..count) |i| {
        const set = objc.msg(objc.Id, sets, "objectAtIndex:", .{i});
        const name = objc.msg(objc.Id, set, "name", .{});
        if (objc.msg(bool, name, "isEqualToString:", .{MTLCommonCounterSetTimestamp})) return set;
    }
    return null;
}

fn hasTimestamp(set: objc.Id) bool {
    const counters = objc.msg(objc.Id, set, "counters", .{});
    const count = objc.msg(usize, counters, "count", .{});
    if (count > max_counter_sets) return false;
    for (0..count) |i| {
        const counter = objc.msg(objc.Id, counters, "objectAtIndex:", .{i});
        const name = objc.msg(objc.Id, counter, "name", .{});
        if (objc.msg(bool, name, "isEqualToString:", .{MTLCommonCounterTimestamp})) return true;
    }
    return false;
}

/// Device queries, never architecture-name assumptions; no GPU allocations or command submissions.
pub fn query(dev: device.Device) Capability {
    var result: Capability = .{};
    if (!dev.responds("counterSets") or !dev.responds("newCounterSampleBufferWithDescriptor:error:")) return result;
    if (timestampSet(dev)) |set| {
        result.timestamp_set = true;
        result.timestamp_counter = hasTimestamp(set);
    }
    result.clock_calibration = dev.responds("sampleTimestamps:gpuTimestamp:");
    if (dev.responds("supportsCounterSampling:")) {
        result.stage_boundary = objc.msg(bool, dev.id, "supportsCounterSampling:", .{@backingInt(SamplingPoint.stage)});
        result.dispatch_boundary = objc.msg(bool, dev.id, "supportsCounterSampling:", .{@backingInt(SamplingPoint.dispatch)});
    }
    return result;
}

/// Apple's CPU clock samples are nanoseconds; GPU ticks require two-point calibration, not a guessed tick rate.
pub fn sampleClocks(dev: device.Device) ClockSample {
    var result: ClockSample = .{};
    objc.msg(void, dev.id, "sampleTimestamps:gpuTimestamp:", .{ &result.cpu, &result.gpu });
    return result;
}

pub const SampleBuffer = struct {
    id: objc.Id,
    count: usize,

    /// Shared opaque samples support CPU resolution after command completion, including stage-only devices.
    pub fn init(dev: device.Device, count: usize) !SampleBuffer {
        if (count == 0 or count > max_samples) return error.InvalidSampleCount;
        const capability = query(dev);
        if (capability.mode() == .unsupported) return error.UnsupportedGpuCounters;
        const allocated = objc.msg(?objc.Id, objc.class("MTLCounterSampleBufferDescriptor"), "alloc", .{}) orelse
            return error.CounterSampleBufferCreationFailed;
        const desc = objc.msg(?objc.Id, allocated, "init", .{}) orelse return error.CounterSampleBufferCreationFailed;
        defer objc.release(desc);
        objc.msg(void, desc, "setCounterSet:", .{timestampSet(dev).?});
        objc.msg(void, desc, "setStorageMode:", .{@as(usize, 0)});
        objc.msg(void, desc, "setSampleCount:", .{count});
        var err: ?objc.Id = null;
        const id = objc.msg(?objc.Id, dev.id, "newCounterSampleBufferWithDescriptor:error:", .{ desc, &err }) orelse {
            std.log.err("GPU counter sample buffer: {s}", .{objc.errorText(err)});
            return error.CounterSampleBufferCreationFailed;
        };
        return .{ .id = id, .count = count };
    }

    pub fn sample(self: SampleBuffer, encoder: command.ComputeEncoder, index: usize) !void {
        if (index >= self.count) return error.CounterSampleIndexOutOfBounds;
        objc.msg(void, encoder.id, "sampleCountersInBuffer:atSampleIndex:withBarrier:", .{ self.id, index, true });
    }

    /// Caller must ensure GPU completion; borrows a bounded NSData result without a GPU blit or second copy.
    pub fn resolveCompleted(self: SampleBuffer, count: usize) !Resolved {
        if (count == 0 or count > self.count) return error.InvalidSampleCount;
        const data = objc.msg(?objc.Id, self.id, "resolveCounterRange:", .{
            types.Range{ .location = 0, .length = count },
        }) orelse return error.CounterResolveFailed;
        if (objc.msg(usize, data, "length", .{}) != count * @sizeOf(TimestampResult))
            return error.InvalidCounterResultLength;
        return .{ .id = objc.retain(data), .count = count };
    }

    pub fn deinit(self: SampleBuffer) void {
        objc.release(self.id);
    }
};

pub const Resolved = struct {
    id: objc.Id,
    count: usize,

    pub fn values(self: Resolved) []const TimestampResult {
        const bytes = objc.msg([*]const u8, self.id, "bytes", .{});
        const values_: [*]const TimestampResult = @ptrCast(@alignCast(bytes));
        return values_[0..self.count];
    }

    pub fn deinit(self: Resolved) void {
        objc.release(self.id);
    }
};

pub const ComputePass = struct {
    id: objc.Id,
    attachment: objc.Id,

    /// Reusable descriptor; Metal snapshots it when creating an encoder. All timing encoders are serial.
    pub fn init() !ComputePass {
        const allocated = objc.msg(?objc.Id, objc.class("MTLComputePassDescriptor"), "alloc", .{}) orelse
            return error.CounterPassCreationFailed;
        const id = objc.msg(?objc.Id, allocated, "init", .{}) orelse return error.CounterPassCreationFailed;
        objc.msg(void, id, "setDispatchType:", .{@backingInt(types.DispatchType.serial)});
        const array = objc.msg(objc.Id, id, "sampleBufferAttachments", .{});
        const attachment = objc.msg(objc.Id, array, "objectAtIndexedSubscript:", .{@as(usize, 0)});
        return .{ .id = id, .attachment = attachment };
    }

    pub fn compute(self: ComputePass, cb: command.CommandBuffer, samples: SampleBuffer, start: usize, end: usize) !command.ComputeEncoder {
        if (start >= samples.count or end >= samples.count) return error.CounterSampleIndexOutOfBounds;
        objc.msg(void, self.attachment, "setSampleBuffer:", .{samples.id});
        objc.msg(void, self.attachment, "setStartOfEncoderSampleIndex:", .{start});
        objc.msg(void, self.attachment, "setEndOfEncoderSampleIndex:", .{end});
        return .{ .id = objc.msg(?objc.Id, cb.id, "computeCommandEncoderWithDescriptor:", .{self.id}) orelse
            return error.CounterEncoderCreationFailed };
    }

    pub fn deinit(self: ComputePass) void {
        objc.release(self.id);
    }
};

pub const Fence = struct {
    id: objc.Id,

    pub fn init(dev: device.Device) !Fence {
        return .{ .id = objc.msg(?objc.Id, dev.id, "newFence", .{}) orelse return error.CounterFenceCreationFailed };
    }

    pub fn update(self: Fence, encoder: command.ComputeEncoder) void {
        objc.msg(void, encoder.id, "updateFence:", .{self.id});
    }

    pub fn waitCompute(self: Fence, encoder: command.ComputeEncoder) void {
        objc.msg(void, encoder.id, "waitForFence:", .{self.id});
    }

    pub fn deinit(self: Fence) void {
        objc.release(self.id);
    }
};

test "counter capability prefers dispatch and refuses missing timestamp data" {
    var capability = Capability{ .timestamp_set = true, .timestamp_counter = true, .clock_calibration = true };
    try std.testing.expectEqual(Mode.unsupported, capability.mode());
    capability.stage_boundary = true;
    try std.testing.expectEqual(Mode.stage, capability.mode());
    capability.dispatch_boundary = true;
    try std.testing.expectEqual(Mode.dispatch, capability.mode());
    capability.timestamp_counter = false;
    try std.testing.expectEqual(Mode.unsupported, capability.mode());
}

test "sample buffer capacity rejects invalid counts before device access" {
    try std.testing.expectError(error.InvalidSampleCount, SampleBuffer.init(undefined, 0));
    try std.testing.expectError(error.InvalidSampleCount, SampleBuffer.init(undefined, max_samples + 1));
    const samples = SampleBuffer{ .id = undefined, .count = 2 };
    try std.testing.expectError(error.CounterSampleIndexOutOfBounds, samples.sample(undefined, 2));
    try std.testing.expectError(error.InvalidSampleCount, samples.resolveCompleted(3));
}
