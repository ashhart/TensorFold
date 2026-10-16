//! Same-command-buffer GPU timestamps, bounded resources and no per-span CPU waits.
const std = @import("std");
const mtl = @import("metal");
const counters = mtl.counters;

pub const Class = @import("gpu_profile.zig").Class;
pub const class_count = @typeInfo(Class).@"enum".field_names.len;
pub const max_span_capacity: usize = 65536;
pub const Options = struct { enabled: bool = false, max_spans: usize = 512 };
pub const Span = struct {
    kind: Class = .dense,
    begin_timestamp: u64 = 0,
    end_timestamp: u64 = 0,
    gpu_seconds: f64 = 0,
};
pub const Aggregate = struct { calls: u64 = 0, gpu_seconds: f64 = 0 };
const State = enum { idle, recording, resolved, read, broken };

/// Two reference samples convert GPU ticks to nanoseconds; never use CPU span time.
pub const Calibration = struct {
    before: counters.ClockSample,
    after: counters.ClockSample,

    pub fn secondsPerTick(self: Calibration) !f64 {
        if (self.after.cpu <= self.before.cpu or self.after.gpu <= self.before.gpu) return error.InvalidClockCalibration;
        const cpu_delta: f64 = @floatFromInt(self.after.cpu - self.before.cpu);
        const gpu_delta: f64 = @floatFromInt(self.after.gpu - self.before.gpu);
        return cpu_delta / gpu_delta * 1e-9;
    }

    pub fn duration(self: Calibration, begin: u64, end: u64) !f64 {
        if (begin == 0 or end == 0 or begin == counters.error_value or end == counters.error_value) return error.InvalidCounterSample;
        if (end < begin) return error.ReversedCounterSample;
        if (begin < self.before.gpu or end > self.after.gpu) return error.CounterOutsideCalibration;
        const ticks: f64 = @floatFromInt(end - begin);
        return ticks * try self.secondsPerTick();
    }
};

const Resources = struct {
    device: mtl.Device,
    samples: counters.SampleBuffer,
    pass: ?counters.ComputePass,
    fence: counters.Fence,

    fn init(device: mtl.Device, mode: counters.Mode, sample_count: usize) !Resources {
        const samples = try counters.SampleBuffer.init(device, sample_count);
        errdefer samples.deinit();
        const fence = try counters.Fence.init(device);
        errdefer fence.deinit();
        const pass = if (mode == .stage) try counters.ComputePass.init() else null;
        return .{
            .device = .{ .id = mtl.objc.retain(device.id) },
            .samples = samples,
            .pass = pass,
            .fence = fence,
        };
    }

    fn deinit(self: Resources) void {
        if (self.pass) |pass| pass.deinit();
        self.fence.deinit();
        self.samples.deinit();
        self.device.deinit();
    }
};

pub const Recorder = struct {
    allocator: std.mem.Allocator,
    resources: ?Resources = null,
    hardware: counters.Capability = .{},
    spans: []Span = &.{},
    span_count: usize = 0,
    active_span: ?usize = null,
    command_buffer: ?mtl.CommandBuffer = null,
    encoder_open: bool = false,
    state: State = .idle,
    clock_before: counters.ClockSample = .{},
    calibration: ?Calibration = null,
    totals: [class_count]Aggregate = @splat(.{}),

    /// Init-only bounded storage; disabled init allocates nothing.
    pub fn init(allocator: std.mem.Allocator, device: mtl.Device, options: Options) !Recorder {
        var self = Recorder{ .allocator = allocator };
        if (!options.enabled) return self;
        if (options.max_spans == 0 or options.max_spans > max_span_capacity) return error.InvalidSpanCapacity;
        self.hardware = counters.query(device);
        const mode = self.hardware.mode();
        if (mode == .unsupported) return error.UnsupportedGpuCounters;
        self.spans = try allocator.alloc(Span, options.max_spans);
        errdefer allocator.free(self.spans);
        self.resources = try Resources.init(device, mode, options.max_spans * 2);
        return self;
    }

    pub fn enabled(self: *const Recorder) bool {
        return self.resources != null;
    }

    /// For disabled Recorders this is intentionally unqueried; use metal.counters.query(device) to inspect hardware.
    pub fn capability(self: *const Recorder) counters.Capability {
        return self.hardware;
    }

    /// Bind the caller's uncommitted command buffer with an open compute encoder; do not commit inside this API.
    pub fn beginCommand(self: *Recorder, cb: mtl.CommandBuffer) !void {
        if (!self.enabled()) {
            self.encoder_open = true;
            return;
        }
        if (self.state != .idle and self.state != .read) return error.CounterRecordingInFlight;
        if (cb.status() != .not_enqueued) return error.CounterCommandAlreadySubmitted;
        self.command_buffer = cb.retain();
        self.clock_before = counters.sampleClocks(self.resources.?.device);
        self.calibration = null;
        self.span_count = 0;
        self.active_span = null;
        self.totals = @splat(.{});
        self.encoder_open = true;
        self.state = .recording;
    }

    /// Non-nested spans; stage mode rotates fenced encoders, requiring resource and pipeline rebinding.
    pub fn beginSpan(self: *Recorder, encoder: *mtl.ComputeEncoder, kind: Class) !void {
        if (!self.enabled()) return;
        if (self.state != .recording or !self.encoder_open) return error.CounterNotRecording;
        if (self.active_span != null) return error.NestedCounterSpan;
        if (self.span_count == self.spans.len) return error.CounterSpanCapacityExceeded;
        const resources = self.resources.?;
        const index = self.span_count;
        if (self.hardware.mode() == .dispatch) {
            try resources.samples.sample(encoder.*, index * 2);
        } else {
            resources.fence.update(encoder.*);
            encoder.end();
            self.encoder_open = false;
            encoder.* = resources.pass.?.compute(self.command_buffer.?, resources.samples, index * 2, index * 2 + 1) catch |err| {
                self.state = .broken;
                return err;
            };
            self.encoder_open = true;
            resources.fence.waitCompute(encoder.*);
        }
        self.spans[index] = .{ .kind = kind };
        self.span_count += 1;
        self.active_span = index;
    }

    /// On stage-only devices the returned uninstrumented encoder still waits for all writes of the previous span.
    pub fn endSpan(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        if (!self.enabled()) return;
        if (self.state != .recording or !self.encoder_open) return error.CounterNotRecording;
        const index = self.active_span orelse return error.NoActiveCounterSpan;
        const resources = self.resources.?;
        if (self.hardware.mode() == .dispatch) {
            try resources.samples.sample(encoder.*, index * 2 + 1);
        } else {
            resources.fence.update(encoder.*);
            encoder.end();
            self.encoder_open = false;
            const cb = self.command_buffer.?;
            const id = mtl.objc.msg(?mtl.objc.Id, cb.id, "computeCommandEncoderWithDispatchType:", .{
                @backingInt(mtl.DispatchType.serial),
            }) orelse {
                self.state = .broken;
                return error.CounterEncoderCreationFailed;
            };
            encoder.* = .{ .id = id };
            self.encoder_open = true;
            resources.fence.waitCompute(encoder.*);
        }
        self.active_span = null;
    }

    /// Ends the final encoder before the caller's existing commit/wait; resolution requires completion.
    pub fn finishEncoding(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        if (!self.enabled()) {
            encoder.end();
            self.encoder_open = false;
            return;
        }
        if (self.state != .recording or !self.encoder_open) return error.CounterNotRecording;
        if (self.active_span != null) return error.UnclosedCounterSpan;
        encoder.end();
        self.encoder_open = false;
        self.state = .resolved;
    }

    /// Cancels unsubmitted work, closing any surviving encoder, including after encoder creation failure.
    pub fn cancelEncoding(self: *Recorder, encoder: *mtl.ComputeEncoder) !void {
        if (self.command_buffer) |cb| {
            if (cb.status() != .not_enqueued) return error.CounterCommandAlreadySubmitted;
        }
        if (self.encoder_open) {
            encoder.end();
            self.encoder_open = false;
        }
        try self.discardAfterCompletion();
    }

    /// No wait or fallback; results are borrowed until beginCommand, discardAfterCompletion or deinit.
    pub fn readAfterCompletion(self: *Recorder) ![]const Span {
        if (!self.enabled()) return &.{};
        if (self.state == .read) return self.spans[0..self.span_count];
        if (self.state != .resolved) return error.CounterResultsNotEncoded;
        const cb = self.command_buffer.?;
        const status = cb.status();
        if (status == .@"error") return error.CounterCommandFailed;
        if (status != .completed) return error.CounterCommandNotCompleted;
        const resources = self.resources.?;
        const calibration = Calibration{
            .before = self.clock_before,
            .after = counters.sampleClocks(resources.device),
        };
        if (self.span_count == 0) {
            cb.release();
            self.command_buffer = null;
            self.state = .read;
            return &.{};
        }
        const resolved = try resources.samples.resolveCompleted(self.span_count * 2);
        defer resolved.deinit();
        const results = resolved.values();
        // Validate every sample before publishing partial totals; retrying cannot duplicate accumulated calls.
        for (self.spans[0..self.span_count], 0..) |_, i| {
            _ = try calibration.duration(results[i * 2].timestamp, results[i * 2 + 1].timestamp);
        }
        for (self.spans[0..self.span_count], 0..) |*span, i| {
            span.begin_timestamp = results[i * 2].timestamp;
            span.end_timestamp = results[i * 2 + 1].timestamp;
            span.gpu_seconds = try calibration.duration(span.begin_timestamp, span.end_timestamp);
            const aggregate = &self.totals[@backingInt(span.kind)];
            aggregate.calls += 1;
            aggregate.gpu_seconds += span.gpu_seconds;
        }
        self.calibration = calibration;
        cb.release();
        self.command_buffer = null;
        self.state = .read;
        return self.spans[0..self.span_count];
    }

    pub fn aggregates(self: *const Recorder) [class_count]Aggregate {
        return self.totals;
    }

    /// Discards closed, unsubmitted or completed recordings; in-flight resources are never reused.
    pub fn discardAfterCompletion(self: *Recorder) !void {
        if (self.encoder_open) return error.CounterEncoderStillOpen;
        if (self.command_buffer) |cb| {
            switch (cb.status()) {
                .not_enqueued, .completed, .@"error" => {},
                else => return error.CounterCommandNotCompleted,
            }
            cb.release();
            self.command_buffer = null;
        }
        self.state = .idle;
        self.span_count = 0;
        self.active_span = null;
        self.totals = @splat(.{});
        self.calibration = null;
    }

    /// Unique ownership: do not copy; cancel unsubmitted work or await completion before destruction.
    pub fn deinit(self: *Recorder) !void {
        try self.discardAfterCompletion();
        if (self.resources) |resources| {
            resources.deinit();
            self.allocator.free(self.spans);
        }
        self.* = .{ .allocator = self.allocator };
    }
};

test "calibration converts GPU ticks rather than CPU span time" {
    const calibration = Calibration{ .before = .{ .cpu = 1000, .gpu = 100 }, .after = .{ .cpu = 5000, .gpu = 300 } };
    try std.testing.expectApproxEqAbs(@as(f64, 1e-6), try calibration.duration(150, 200), 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0), try calibration.duration(175, 175), 1e-15);
    try std.testing.expectError(error.InvalidCounterSample, calibration.duration(counters.error_value, 200));
    try std.testing.expectError(error.ReversedCounterSample, calibration.duration(200, 150));
    try std.testing.expectError(error.CounterOutsideCalibration, calibration.duration(99, 200));
    const invalid = Calibration{ .before = .{ .cpu = 1000, .gpu = 100 }, .after = .{ .cpu = 1000, .gpu = 300 } };
    try std.testing.expectError(error.InvalidClockCalibration, invalid.secondsPerTick());
}

test "disabled recorder allocates nothing and never touches a device or encoder for spans" {
    var recorder = try Recorder.init(std.testing.failing_allocator, undefined, .{ .enabled = false });
    defer recorder.deinit() catch unreachable;
    var encoder: mtl.ComputeEncoder = undefined;
    try recorder.beginSpan(&encoder, .attention);
    try recorder.endSpan(&encoder);
    try std.testing.expect(!recorder.enabled());
    try std.testing.expectEqual(@as(usize, 0), (try recorder.readAfterCompletion()).len);
    for (recorder.aggregates()) |aggregate| try std.testing.expectEqual(@as(u64, 0), aggregate.calls);
}

test "calibration rejects missing samples, both reversed clocks and both range boundaries" {
    const c = Calibration{ .before = .{ .cpu = 1000, .gpu = 100 }, .after = .{ .cpu = 5000, .gpu = 300 } };
    try std.testing.expectError(error.InvalidCounterSample, c.duration(0, 200));
    try std.testing.expectError(error.InvalidCounterSample, c.duration(100, counters.error_value));
    try std.testing.expectError(error.CounterOutsideCalibration, c.duration(150, 301));
    try std.testing.expectApproxEqAbs(@as(f64, 4e-6), try c.duration(100, 300), 1e-15);
    const bad_gpu = Calibration{ .before = c.before, .after = .{ .cpu = 5000, .gpu = 99 } };
    const bad_cpu = Calibration{ .before = c.before, .after = .{ .cpu = 999, .gpu = 300 } };
    try std.testing.expectError(error.InvalidClockCalibration, bad_gpu.secondsPerTick());
    try std.testing.expectError(error.InvalidClockCalibration, bad_cpu.secondsPerTick());
}

test "enabled capacity validation occurs before any device access" {
    try std.testing.expectError(error.InvalidSpanCapacity, Recorder.init(std.testing.failing_allocator, undefined, .{
        .enabled = true,
        .max_spans = 0,
    }));
    try std.testing.expectError(error.InvalidSpanCapacity, Recorder.init(std.testing.failing_allocator, undefined, .{
        .enabled = true,
        .max_spans = max_span_capacity + 1,
    }));
}
