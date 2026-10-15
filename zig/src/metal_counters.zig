//! A bounded model-free Metal capability and timestamp probe; no weights or engine integration.
const std = @import("std");
const mtl = @import("metal");
const consumer = @import("metal_counter_consumer.zig");
const usage = "Usage: tf-metal-counters [--capability | --help]\n";

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var buffer: [8192]u8 = undefined;
    var output = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    defer output.interface.flush() catch {};
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        try output.interface.writeAll(usage);
        return 0;
    }
    const capability_only = args.len == 2 and std.mem.eql(u8, args[1], "--capability");
    if (args.len != 1 and !capability_only) {
        try std.json.Stringify.value(.{ .status = "error", .@"error" = "InvalidArguments" }, .{}, &output.interface);
        try output.interface.writeByte('\n');
        return 2;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch |err| {
        try std.json.Stringify.value(.{ .status = "unsupported", .@"error" = @errorName(err) }, .{}, &output.interface);
        try output.interface.writeByte('\n');
        return 2;
    };
    defer device.deinit();
    const capability = mtl.counters.query(device);
    const mode = capability.mode();
    if (capability_only or mode == .unsupported) {
        try std.json.Stringify.value(.{
            .status = if (mode == .unsupported) "unsupported" else "capability",
            .mode = mode,
            .capability = capability,
            .stage_sampling_exercised = false,
            .dispatch_sampling_exercised = false,
        }, .{}, &output.interface);
        try output.interface.writeByte('\n');
        return if (capability_only) 0 else 2;
    }
    var operators = consumer.Consumer.init(device) catch |err| return reportError(&output.interface, capability, err);
    defer operators.deinit();
    const report = operators.probe(init.gpa) catch |err| return reportError(&output.interface, capability, err);
    try std.json.Stringify.value(.{
        .status = "ok",
        .mode = mode,
        .capability = capability,
        .stage_sampling_exercised = mode == .stage,
        .dispatch_sampling_exercised = mode == .dispatch,
        .elements = consumer.element_count,
        .dependent_stages = report.spans.len,
        .instrumented_fp32_bit_identity = true,
        .independent_expected_fp32_bits = true,
        .in_flight_read_reset_deinit_rejected = report.in_flight_rejected,
        .timing_source = "calibrated_gpu_counter_samples",
        .clock_calibration = report.calibration,
        .seconds_per_gpu_tick = try report.calibration.secondsPerTick(),
        .spans = report.spans,
        .aggregate_classes = std.meta.tags(consumer.counters.Class),
        .aggregates = report.aggregates,
    }, .{}, &output.interface);
    try output.interface.writeByte('\n');
    return 0;
}

fn reportError(output: *std.Io.Writer, capability: mtl.counters.Capability, err: anyerror) !u8 {
    try std.json.Stringify.value(.{
        .status = "error",
        .@"error" = @errorName(err),
        .mode = capability.mode(),
        .capability = capability,
    }, .{}, output);
    try output.writeByte('\n');
    return 1;
}
