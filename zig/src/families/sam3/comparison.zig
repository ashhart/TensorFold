const std = @import("std");

pub const Quality = struct {
    frames: usize = 0,
    pixels: usize = 0,
    exact_pixels: usize = 0,
    iou_sum: f64 = 0,
    minimum_iou: f64 = 1,
    worst_frame: ?usize = null,

    pub fn add(self: *Quality, frame: usize, actual: []const u8, expected: []const u8) !void {
        if (actual.len == 0 or actual.len != expected.len) return error.MaskShapeMismatch;
        var intersection: usize = 0;
        var union_size: usize = 0;
        var exact: usize = 0;
        for (actual, expected) |x, y| {
            if ((x != 0 and x != 255) or (y != 0 and y != 255)) return error.ExpectedBinaryMasks;
            intersection += @intFromBool(x != 0 and y != 0);
            union_size += @intFromBool(x != 0 or y != 0);
            exact += @intFromBool(x == y);
        }
        const iou = if (union_size == 0) 1 else @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(union_size));
        self.frames += 1;
        self.pixels += actual.len;
        self.exact_pixels += exact;
        self.iou_sum += iou;
        if (iou < self.minimum_iou) {
            self.minimum_iou = iou;
            self.worst_frame = frame;
        }
    }

    pub fn meanIou(self: Quality) f64 {
        return if (self.frames == 0) 0 else self.iou_sum / @as(f64, @floatFromInt(self.frames));
    }

    pub fn addPrediction(self: *Quality, frame: usize, id: u64, prediction: anytype, expected: []const u8) !void {
        if (prediction.frame_index != frame or prediction.objects.len != 1 or prediction.objects[0].id != id) return error.ReferenceObjectMismatch;
        try self.add(frame, prediction.objects[0].mask.pixels, expected);
    }
};

pub const Gate = struct {
    min_mean_iou: ?f64 = null,
    min_frame_iou: ?f64 = null,

    pub fn set(self: *Gate, flag: []const u8, value: []const u8) !bool {
        const target = if (std.mem.eql(u8, flag, "--min-mean-iou")) &self.min_mean_iou else if (std.mem.eql(u8, flag, "--min-frame-iou")) &self.min_frame_iou else return false;
        if (target.* != null) return error.DuplicateComparisonThreshold;
        const number = std.fmt.parseFloat(f64, value) catch return error.InvalidIouThreshold;
        if (!std.math.isFinite(number) or number < 0 or number > 1) return error.InvalidIouThreshold;
        target.* = number;
        return true;
    }

    pub fn validate(self: Gate) !void {
        if (self.min_mean_iou != null and self.min_frame_iou == null) return error.FrameIouThresholdRequired;
        for ([_]?f64{ self.min_mean_iou, self.min_frame_iou }) |maybe| if (maybe) |value| {
            if (!std.math.isFinite(value) or value < 0 or value > 1) return error.InvalidIouThreshold;
        };
    }

    pub fn check(self: Gate, quality: Quality, expected_frames: usize) !void {
        try self.validate();
        if (expected_frames == 0) return error.EmptyMaskComparison;
        if (quality.frames != expected_frames) return error.IncompleteMaskComparison;
        if (self.min_frame_iou) |minimum| {
            if (quality.minimum_iou < minimum) return error.MaskComparisonFailed;
            if (self.min_mean_iou) |mean| if (quality.meanIou() < mean) return error.MaskComparisonFailed;
        } else if (quality.exact_pixels != quality.pixels) return error.MaskComparisonFailed;
    }
};

pub fn validateVideo(has_reference: bool, frames: usize, objects: usize, exercise_lifecycle: bool, gate: Gate) !void {
    try gate.validate();
    if (!has_reference) {
        if (gate.min_frame_iou != null or gate.min_mean_iou != null) return error.ExpectedReferenceMasks;
        return;
    }
    if (frames < 2) return error.EmptyMaskComparison;
    if (objects != 1) return error.SingleObjectReferenceRequired;
    if (exercise_lifecycle) return error.ReferenceLifecycleMismatch;
}

test "SAM comparison rejects a bad frame even when the mean is high" {
    var quality = Quality{};
    for (0..100) |frame| try quality.add(frame, &.{ 255, 0 }, &.{ 255, 0 });
    try quality.add(100, &.{ 0, 255 }, &.{ 255, 0 });
    try std.testing.expect(quality.meanIou() > 0.99);
    try std.testing.expectEqual(@as(?usize, 100), quality.worst_frame);
    try std.testing.expectError(error.MaskComparisonFailed, (Gate{ .min_mean_iou = 0.99, .min_frame_iou = 0.9 }).check(quality, 101));
    try std.testing.expectError(error.MaskComparisonFailed, (Gate{}).check(quality, 101));
}

test "SAM comparison requires full frame coverage and rejects softened output masks" {
    var quality = Quality{};
    try quality.add(9, &.{ 255, 0 }, &.{ 255, 0 });
    try (Gate{}).check(quality, 1);
    try std.testing.expectError(error.IncompleteMaskComparison, (Gate{}).check(quality, 2));
    try std.testing.expectError(error.EmptyMaskComparison, (Gate{}).check(.{}, 0));
    try std.testing.expectError(error.ExpectedBinaryMasks, quality.add(10, &.{ 255, 1 }, &.{ 255, 0 }));
    try std.testing.expectError(error.MaskShapeMismatch, quality.add(10, &.{255}, &.{ 255, 0 }));
    try std.testing.expectEqual(@as(usize, 1), quality.frames);
}

test "SAM explicit IoU thresholds pass measured approximate masks and validate their bounds" {
    var quality = Quality{};
    try quality.add(0, &.{ 255, 255, 0 }, &.{ 255, 0, 255 });
    try (Gate{ .min_frame_iou = 0.3 }).check(quality, 1);
    try std.testing.expectError(error.MaskComparisonFailed, (Gate{ .min_frame_iou = 0.4 }).check(quality, 1));
    var gate = Gate{};
    for ([_][]const u8{ "nan", "inf", "-0.1", "1.1" }) |value| try std.testing.expectError(error.InvalidIouThreshold, gate.set("--min-frame-iou", value));
    try std.testing.expect(try gate.set("--min-mean-iou", "0.9"));
    try std.testing.expectError(error.FrameIouThresholdRequired, gate.validate());
    try std.testing.expect(try gate.set("--min-frame-iou", "0.8"));
    try gate.validate();
    try std.testing.expectError(error.DuplicateComparisonThreshold, gate.set("--min-frame-iou", "0.8"));
}

test "SAM reference comparisons refuse seed-only and changing-object workloads" {
    try std.testing.expectError(error.EmptyMaskComparison, validateVideo(true, 1, 1, false, .{}));
    try std.testing.expectError(error.SingleObjectReferenceRequired, validateVideo(true, 2, 2, false, .{}));
    try std.testing.expectError(error.ReferenceLifecycleMismatch, validateVideo(true, 2, 1, true, .{}));
    try std.testing.expectError(error.ExpectedReferenceMasks, validateVideo(false, 2, 1, false, .{ .min_frame_iou = 0.9 }));
    try validateVideo(true, 2, 1, false, .{});
    try validateVideo(false, 2, 2, true, .{});
}

test "SAM comparison fails missing objects, wrong identities and wrong frames" {
    const Object = struct { id: u64, mask: struct { pixels: []const u8 } };
    const Prediction = struct { frame_index: usize, objects: []const Object };
    const object = Object{ .id = 42, .mask = .{ .pixels = &.{ 255, 0 } } };
    var quality = Quality{};
    try std.testing.expectError(error.ReferenceObjectMismatch, quality.addPrediction(7, 42, Prediction{ .frame_index = 7, .objects = &.{} }, &.{ 255, 0 }));
    try std.testing.expectError(error.ReferenceObjectMismatch, quality.addPrediction(7, 99, Prediction{ .frame_index = 7, .objects = &.{object} }, &.{ 255, 0 }));
    try std.testing.expectError(error.ReferenceObjectMismatch, quality.addPrediction(8, 42, Prediction{ .frame_index = 7, .objects = &.{object} }, &.{ 255, 0 }));
    try std.testing.expectEqual(@as(usize, 0), quality.frames);
    try quality.addPrediction(7, 42, Prediction{ .frame_index = 7, .objects = &.{object} }, &.{ 255, 0 });
    try (Gate{}).check(quality, 1);
}
