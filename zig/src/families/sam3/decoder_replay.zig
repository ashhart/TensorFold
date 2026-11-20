const std = @import("std");
const st = @import("safetensors");
const ops = @import("ops.zig");
const Store = @import("checkpoint.zig").Store;

const Capture = struct {
    quality: [4]f32 = @splat(0),
    seen: bool = false,

    fn observe(context: *anyopaque, name: []const u8, value: ops.Tensor, _: []const usize) !void {
        if (!std.mem.eql(u8, name, "decoder.quality")) return;
        const self: *Capture = @ptrCast(@alignCast(context));
        if (value.count != self.quality.len or value.dtype != .f32) return error.DecoderReplayShape;
        @memcpy(&self.quality, value.slice(f32, value.count));
        self.seen = true;
    }
};

fn upload(r: *ops.Runtime, file: *st.File, key: []const u8) !ops.Feature {
    const value = file.get(key) orelse return error.MissingDecoderReplayInput;
    if (value.dtype != .f32 or value.rank != 4 or value.dim(0) != 1) return error.DecoderReplayShape;
    const tensor = try r.alloc(value.numel());
    @memcpy(tensor.slice(u8, value.bytes.len), value.bytes);
    try tensor.checkFinite(key);
    return ops.feature(tensor, value.dim(1), value.dim(2), value.dim(3));
}

pub fn run(init: std.process.Init, g: *@import("tensor_metal").Gpu, args: []const [:0]const u8) !void {
    if (args.len != 3) return error.ExpectedCheckpointReferenceAndReceipt;
    const a = init.arena.allocator();
    var store = try Store.open(init.gpa, init.io, args[0]);
    defer store.deinit();
    var reference = try st.File.open(init.gpa, init.io, args[1]);
    defer reference.close(init.io);
    var capture = Capture{};
    var r = ops.Runtime{ .g = g, .store = &store, .a = a, .config = try store.config(), .precision = .fp32, .observer = .{ .context = &capture, .write = Capture.observe } };
    defer r.deinit();
    if (r.config.version != .sam3) return error.DecoderReplayRequiresSam3;
    const features = [3]ops.Feature{
        try upload(&r, &reference, "pyramid.propagation.0"),
        try upload(&r, &reference, "pyramid.propagation.1"),
        try upload(&r, &reference, "pyramid.propagation.2"),
    };
    const image = try upload(&r, &reference, "decoder.image");
    for (features, 0..) |feature, index| {
        const side = @as(usize, 288) >> @as(u6, @intCast(index));
        if (feature.height != side or feature.width != side or feature.channels != 256) return error.DecoderReplayShape;
    }
    if (image.height != 72 or image.width != 72 or image.channels != 256) return error.DecoderReplayShape;
    const truth = reference.get("result.pred_masks") orelse return error.MissingDecoderReplayReference;
    const quality = reference.get("decoder.result.1") orelse return error.MissingDecoderReplayReference;
    if (!truth.is(.f32, &.{ 1, 288, 288, 1 }) or !quality.is(.f32, &.{ 1, 3 })) return error.DecoderReplayShape;
    g.compact_gemm = true;
    const output = try @import("decoder.zig").decode(&r, features, image.tensor, .{});
    if (!capture.seen or output.masks.tensor.count != truth.numel()) return error.DecoderReplayShape;
    const expected_quality: [3]f32 = std.mem.bytesToValue([3]f32, quality.bytes);
    var quality_in_range = true;
    var quality_maximum_error: f64 = 0;
    for (capture.quality) |value| quality_in_range = quality_in_range and std.math.isFinite(value) and value >= 0 and value <= 1;
    for (capture.quality[1..], expected_quality) |actual, expected| quality_maximum_error = @max(quality_maximum_error, @abs(@as(f64, actual) - expected));
    var native_choice: usize = 1;
    var reference_choice: usize = 0;
    for (2..4) |i| if (capture.quality[i] > capture.quality[native_choice]) {
        native_choice = i;
    };
    for (1..3) |i| if (expected_quality[i] > expected_quality[reference_choice]) {
        reference_choice = i;
    };
    var maximum: f64 = 0;
    var mean: f64 = 0;
    var intersection: usize = 0;
    var union_: usize = 0;
    for (0..truth.numel()) |i| {
        const expected: f32 = @bitCast(std.mem.readInt(u32, truth.bytes[i * 4 ..][0..4], .little));
        const actual = output.masks.tensor.get(i);
        if (!std.math.isFinite(expected) or !std.math.isFinite(actual)) return error.NonFiniteDecoderReplay;
        const difference = @abs(@as(f64, expected) - actual);
        maximum = @max(maximum, difference);
        mean += difference;
        intersection += @intFromBool(actual > 0 and expected > 0);
        union_ += @intFromBool(actual > 0 or expected > 0);
    }
    mean /= @floatFromInt(truth.numel());
    const iou = if (union_ == 0) 1 else @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(union_));
    const exact = std.mem.eql(u8, truth.bytes, output.masks.tensor.slice(u8, truth.bytes.len));
    const report = try std.json.Stringify.valueAlloc(a, .{ .scope = "decoder on identical captured FP32 pyramid and image inputs", .exact_masks = exact, .quality_in_range = quality_in_range, .quality_maximum_error = quality_maximum_error, .native_quality = capture.quality, .reference_quality = expected_quality, .native_candidate = native_choice, .reference_candidate = reference_choice + 1, .mask_maximum_error = maximum, .mask_mean_error = mean, .mask_iou = iou }, .{ .whitespace = .indent_2 });
    const file = try std.Io.Dir.cwd().createFile(init.io, args[2], .{ .exclusive = true });
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, report);
    try file.writeStreamingAll(init.io, "\n");
    std.debug.print("Decoder matched inputs: candidate {d} vs {d}; mask IoU {d:.9}, maximum {e}, mean {e}\n", .{ native_choice, reference_choice + 1, iou, maximum, mean });
    if (!quality_in_range) return error.DecoderQualityOutsideUnitInterval;
    if (native_choice != reference_choice + 1) return error.DecoderReplayCandidateDiffers;
    if (!exact) return error.DecoderReplayMasksDiffer;
}
