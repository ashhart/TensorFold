//! The engine's Metal module for the Hy4-flash speed probe. Everything is the real module except ComputeEncoder,
//! which forwards to a real encoder and, while a Splitter is splitting, turns each barrier into an encoder boundary
//! whose start and end the GPU timestamps. Without splitting a barrier stays a barrier, so timed runs match the engine.
const std = @import("std");
const real = @import("real_metal");

pub const objc = real.objc;
pub const types = real.types;
pub const Size = real.Size;
pub const ResourceOptions = real.ResourceOptions;
pub const ResourceUsage = real.ResourceUsage;
pub const DispatchType = real.DispatchType;
pub const Device = real.Device;
pub const Buffer = real.Buffer;
pub const MappedFile = real.MappedFile;
pub const Library = real.Library;
pub const CompileOptions = real.CompileOptions;
pub const Pipeline = real.Pipeline;
pub const Queue = real.Queue;
pub const CommandBuffer = real.CommandBuffer;
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

pub const max_segments = 2048; // two 8-byte stamps each: Metal caps a sample buffer at 32 KiB

/// Timing only (TF_HY4_SKIP): dispatches of these pipelines are dropped, so a run's GPU time without them shows
/// what they cost unprofiled. The tokens are then wrong.
pub var skip_ids: []const usize = &.{};
pub const max_names = 6;

/// One encoder between two barriers: the pipelines it set (first few) and how many dispatches it made.
pub const Segment = struct {
    pipelines: [max_names]usize = @splat(0),
    named: u8 = 0,
    dispatches: u16 = 0,
};

pub const Splitter = struct {
    cb: real.CommandBuffer,
    cur: real.ComputeEncoder,
    split: bool,
    pass: ?counters.ComputePass = null,
    samples: ?counters.SampleBuffer = null,
    count: usize = 0,
    segments: []Segment,
    current: usize = 0,

    /// A plain concurrent encoder: barriers stay barriers.
    pub fn plain(cb: real.CommandBuffer, segments: []Segment) Splitter {
        return .{ .cb = cb, .cur = cb.compute(.concurrent), .split = false, .segments = segments };
    }

    /// Timed serial encoders, one per barrier-delimited segment; `samples` holds two stamps a segment.
    pub fn timed(cb: real.CommandBuffer, pass: counters.ComputePass, samples: counters.SampleBuffer, segments: []Segment) !Splitter {
        var s: Splitter = .{ .cb = cb, .cur = undefined, .split = true, .pass = pass, .samples = samples, .segments = segments };
        try s.open();
        return s;
    }

    fn open(self: *Splitter) !void {
        if (self.count >= self.segments.len or 2 * self.count + 1 >= self.samples.?.count) return error.TooManyHy4ProbeSegments;
        self.segments[self.count] = .{};
        self.cur = try self.pass.?.compute(self.cb, self.samples.?, 2 * self.count, 2 * self.count + 1);
    }

    pub fn encoder(self: *Splitter) ComputeEncoder {
        return .{ .s = self };
    }

    fn note(self: *Splitter, p: Pipeline) void {
        if (!self.split) return;
        const seg = &self.segments[self.count];
        if (seg.named < max_names) {
            seg.pipelines[seg.named] = @intFromPtr(p.id);
            seg.named += 1;
        }
    }

    fn dispatched(self: *Splitter) void {
        if (self.split) self.segments[self.count].dispatches +|= 1;
    }

    fn barrier(self: *Splitter) void {
        if (!self.split) return self.cur.barrier();
        if (self.segments[self.count].dispatches == 0) return;
        self.cur.end();
        self.count += 1;
        self.open() catch @panic("hy4 probe: out of timed segments");
    }

    /// End the last encoder. Returns how many timed segments the command buffer holds.
    pub fn finish(self: *Splitter) usize {
        self.cur.end();
        if (!self.split) return 0;
        return self.count + @intFromBool(self.segments[self.count].dispatches != 0);
    }
};

pub const ComputeEncoder = struct {
    s: *Splitter,

    fn skipped(self: ComputeEncoder) bool {
        for (skip_ids) |id| if (id == self.s.current) return true;
        return false;
    }

    pub fn deviceId(self: ComputeEncoder) objc.Id {
        return self.s.cur.deviceId();
    }
    pub fn setPipeline(self: ComputeEncoder, pipeline: Pipeline) void {
        self.s.current = @intFromPtr(pipeline.id);
        self.s.note(pipeline);
        self.s.cur.setPipeline(pipeline);
    }
    pub fn setBuffer(self: ComputeEncoder, buffer: Buffer, offset: usize, index: usize) void {
        self.s.cur.setBuffer(buffer, offset, index);
    }
    pub fn setBytes(self: ComputeEncoder, bytes: []const u8, index: usize) void {
        self.s.cur.setBytes(bytes, index);
    }
    pub fn setValue(self: ComputeEncoder, value: anytype, index: usize) void {
        self.s.cur.setBytes(std.mem.asBytes(&value), index);
    }
    pub fn setThreadgroupMemory(self: ComputeEncoder, len: usize, index: usize) void {
        self.s.cur.setThreadgroupMemory(len, index);
    }
    pub fn dispatchGroups(self: ComputeEncoder, groups: Size, threads: Size) void {
        if (self.skipped()) return;
        self.s.dispatched();
        self.s.cur.dispatchGroups(groups, threads);
    }
    pub fn dispatchIndirect(self: ComputeEncoder, args: Buffer, offset: usize, threads: Size) void {
        if (self.skipped()) return;
        self.s.dispatched();
        self.s.cur.dispatchIndirect(args, offset, threads);
    }
    pub fn dispatchThreads(self: ComputeEncoder, grid: Size, threads: Size) void {
        if (self.skipped()) return;
        self.s.dispatched();
        self.s.cur.dispatchThreads(grid, threads);
    }
    pub fn barrier(self: ComputeEncoder) void {
        self.s.barrier();
    }
    pub fn useResource(self: ComputeEncoder, buffer: Buffer, usage: usize) void {
        self.s.cur.useResource(buffer, usage);
    }
    pub fn execute(self: ComputeEncoder, icb: IndirectCommandBuffer, first: usize, count: usize) void {
        self.s.dispatched();
        self.s.cur.execute(icb, first, count);
    }
    pub fn end(self: ComputeEncoder) void {
        _ = self.s.finish();
    }
    pub fn commandBuffer(self: ComputeEncoder) CommandBuffer {
        return self.s.cur.commandBuffer();
    }
    pub fn dispatchType(self: ComputeEncoder) DispatchType {
        return self.s.cur.dispatchType();
    }
};
