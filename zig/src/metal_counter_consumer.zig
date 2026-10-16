//! Model-free dependent FP32 operators with untracked resources and one externally completed command buffer.
const std = @import("std");
const mtl = @import("metal");
pub const counters = @import("core/gpu_counters.zig");
const source = @import("kernel_sources").counter_probe;
pub const element_count: u32 = 4096;
const classes = [_]counters.Class{ .dense, .activation, .copy };

pub const Report = struct {
    spans: [3]counters.Span,
    aggregates: [counters.class_count]counters.Aggregate,
    calibration: counters.Calibration,
    in_flight_rejected: bool,
};

pub const Consumer = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    pipeline: mtl.Pipeline,
    a: mtl.Buffer,
    b: mtl.Buffer,
    indirect: mtl.Buffer,
    gate: mtl.SharedEvent,
    next_gate: u64 = 1,

    pub fn init(device: mtl.Device) !Consumer {
        const queue = try device.queue();
        errdefer queue.deinit();
        const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer library.deinit();
        const pipeline = try mtl.Pipeline.init(device, library, "tf_counter_affine", false);
        errdefer pipeline.deinit();
        const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const a = try device.buffer(element_count * @sizeOf(f32), options);
        errdefer a.deinit();
        const b = try device.buffer(element_count * @sizeOf(f32), options);
        errdefer b.deinit();
        const indirect = try device.buffer(3 * @sizeOf(u32), options);
        errdefer indirect.deinit();
        @memcpy(indirect.slice(u32, 3), &[_]u32{ element_count / 256, 1, 1 });
        const gate = try device.sharedEvent();
        return .{
            .device = .{ .id = mtl.objc.retain(device.id) },
            .queue = queue,
            .pipeline = pipeline,
            .a = a,
            .b = b,
            .indirect = indirect,
            .gate = gate,
        };
    }

    pub fn deinit(self: Consumer) void {
        self.gate.deinit();
        self.indirect.deinit();
        self.b.deinit();
        self.a.deinit();
        self.pipeline.deinit();
        self.queue.deinit();
        self.device.deinit();
    }

    /// The only CPU wait is the consumer's original final completion wait, never a span wait.
    pub fn execute(self: *Consumer, recorder: *counters.Recorder) !void {
        for (self.a.slice(f32, element_count), 0..) |*value, i| value.* = initial(i);
        @memset(self.b.slice(f32, element_count), std.math.nan(f32));
        const cb = self.queue.commandBuffer();
        const gate_value = self.next_gate;
        self.next_gate += 1;
        cb.waitFor(self.gate, gate_value);
        try recorder.beginCommand(cb);
        var encoder = cb.compute(.serial);
        var submitted = false;
        errdefer {
            if (submitted) {
                self.gate.set(gate_value);
                cb.wait();
                recorder.discardAfterCompletion() catch {};
            } else recorder.cancelEncoding(&encoder) catch {};
        }
        const scales = [_]f32{ 2, 1, 0.5 };
        const biases = [_]f32{ 1, -3, 2 };
        for (classes, 0..) |kind, i| {
            try recorder.beginSpan(&encoder, kind);
            const input = if (i % 2 == 0) self.a else self.b;
            const output = if (i % 2 == 0) self.b else self.a;
            encoder.setPipeline(self.pipeline);
            encoder.setBuffer(input, 0, 0);
            encoder.setBuffer(output, 0, 1);
            encoder.setValue(element_count, 2);
            encoder.setValue(scales[i], 3);
            encoder.setValue(biases[i], 4);
            encoder.useResource(input, mtl.ResourceUsage.read);
            encoder.useResource(output, mtl.ResourceUsage.write);
            encoder.useResource(self.indirect, mtl.ResourceUsage.read);
            if (i == 1) {
                encoder.dispatchIndirect(self.indirect, 0, mtl.Size.of(256, 1, 1));
            } else encoder.dispatchThreads(mtl.Size.of(element_count, 1, 1), mtl.Size.of(256, 1, 1));
            encoder.barrier();
            try recorder.endSpan(&encoder);
        }
        try recorder.finishEncoding(&encoder);
        cb.commit();
        submitted = true;
        if (recorder.enabled()) {
            try expectFailure(error.CounterCommandNotCompleted, recorder.readAfterCompletion());
            try expectFailure(error.CounterRecordingInFlight, recorder.beginCommand(cb));
            try expectFailure(error.CounterCommandNotCompleted, recorder.discardAfterCompletion());
            try expectFailure(error.CounterCommandNotCompleted, recorder.deinit());
        }
        self.gate.set(gate_value);
        cb.wait();
        if (cb.failure() != null) return error.CounterConsumerGpuFailure;
        _ = try recorder.readAfterCompletion();
        for (self.b.slice(f32, element_count), 0..) |value, i| {
            const expected = initial((i + 3) % element_count) + 1;
            if (@as(u32, @bitCast(value)) != @as(u32, @bitCast(expected))) return error.CounterConsumerOutputMismatch;
        }
    }

    pub fn probe(self: *Consumer, allocator: std.mem.Allocator) !Report {
        var plain = try counters.Recorder.init(allocator, self.device, .{});
        defer plain.deinit() catch unreachable;
        try self.execute(&plain);
        var expected_bits: [element_count]u32 = undefined;
        for (self.b.slice(f32, element_count), &expected_bits) |value, *bits| bits.* = @bitCast(value);
        var instrumented = try counters.Recorder.init(allocator, self.device, .{ .enabled = true, .max_spans = 3 });
        defer instrumented.deinit() catch unreachable;
        try self.execute(&instrumented);
        for (self.b.slice(f32, element_count), expected_bits) |value, bits| {
            if (@as(u32, @bitCast(value)) != bits) return error.CounterInstrumentationChangedBits;
        }
        const spans = try instrumented.readAfterCompletion();
        if (spans.len != 3) return error.CounterConsumerSpanCountMismatch;
        var report: Report = .{
            .spans = undefined,
            .aggregates = instrumented.aggregates(),
            .calibration = instrumented.calibration orelse return error.MissingClockCalibration,
            .in_flight_rejected = true,
        };
        @memcpy(&report.spans, spans);
        var seconds: f64 = 0;
        for (report.spans, classes) |span, kind| {
            if (span.kind != kind) return error.CounterConsumerClassMismatch;
            seconds += span.gpu_seconds;
        }
        if (seconds <= 0 or !std.math.isFinite(seconds)) return error.InvalidCounterSample;
        return report;
    }
};

fn initial(index: usize) f32 {
    const value: i32 = @as(i32, @intCast(index % 127)) - 63;
    return @floatFromInt(value);
}

fn expectFailure(expected: anyerror, result: anytype) !void {
    if (result) |_| {
        return error.CounterExpectedFailureMissing;
    } else |err| if (err != expected) return err;
}

fn supportedDevice() !mtl.Device {
    const device = mtl.Device.init() catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    if (mtl.counters.query(device).mode() == .unsupported) {
        device.deinit();
        return error.SkipZigTest;
    }
    return device;
}

test "sampled dependent direct and indirect operators preserve FP32 bits and repeat without stale samples" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    const first = try consumer.probe(std.testing.allocator);
    const second = try consumer.probe(std.testing.allocator);
    for (first.spans, second.spans, classes) |a, b, kind| {
        try std.testing.expectEqual(kind, a.kind);
        try std.testing.expectEqual(kind, b.kind);
        try std.testing.expect(b.begin_timestamp >= a.end_timestamp);
        try std.testing.expectEqual(@as(u64, 1), second.aggregates[@backingInt(kind)].calls);
    }
    try std.testing.expect(first.in_flight_rejected and second.in_flight_rejected);
}

test "bounded recorder rejects nesting, overflow, open spans and premature reuse then cancels and reuses" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 1 });
    defer recorder.deinit() catch unreachable;
    const cb = queue.commandBuffer();
    try recorder.beginCommand(cb);
    var encoder = cb.compute(.serial);
    errdefer recorder.cancelEncoding(&encoder) catch {};
    try std.testing.expectError(error.CounterEncoderStillOpen, recorder.deinit());
    try std.testing.expectError(error.CounterEncoderStillOpen, recorder.discardAfterCompletion());
    try std.testing.expectError(error.CounterRecordingInFlight, recorder.beginCommand(cb));
    try std.testing.expectError(error.NoActiveCounterSpan, recorder.endSpan(&encoder));
    try recorder.beginSpan(&encoder, .dense);
    try std.testing.expectError(error.NestedCounterSpan, recorder.beginSpan(&encoder, .copy));
    try std.testing.expectError(error.UnclosedCounterSpan, recorder.finishEncoding(&encoder));
    try recorder.endSpan(&encoder);
    try std.testing.expectError(error.CounterSpanCapacityExceeded, recorder.beginSpan(&encoder, .copy));
    try recorder.finishEncoding(&encoder);
    try std.testing.expectError(error.CounterCommandNotCompleted, recorder.readAfterCompletion());
    try recorder.cancelEncoding(&encoder);
    const next = queue.commandBuffer();
    try recorder.beginCommand(next);
    encoder = next.compute(.serial);
    try recorder.finishEncoding(&encoder);
    next.commit();
    next.wait();
    if (next.failure() != null) return error.CounterConsumerGpuFailure;
    try std.testing.expectEqual(@as(usize, 0), (try recorder.readAfterCompletion()).len);
    for (recorder.aggregates()) |aggregate| try std.testing.expectEqual(@as(u64, 0), aggregate.calls);
    try recorder.discardAfterCompletion();
}

test "completed recorder reuses bounded samples and read results do not double count" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try supportedDevice();
    defer device.deinit();
    var consumer = try Consumer.init(device);
    defer consumer.deinit();
    var recorder = try counters.Recorder.init(std.testing.allocator, device, .{ .enabled = true, .max_spans = 3 });
    defer recorder.deinit() catch unreachable;
    try consumer.execute(&recorder);
    const first = try recorder.readAfterCompletion();
    const prior_end = first[2].end_timestamp;
    _ = try recorder.readAfterCompletion();
    for (classes) |kind| try std.testing.expectEqual(@as(u64, 1), recorder.aggregates()[@backingInt(kind)].calls);
    try consumer.execute(&recorder);
    const second = try recorder.readAfterCompletion();
    try std.testing.expectEqual(@as(usize, 3), second.len);
    try std.testing.expect(second[0].begin_timestamp >= prior_end);
    for (classes) |kind| try std.testing.expectEqual(@as(u64, 1), recorder.aggregates()[@backingInt(kind)].calls);
}
