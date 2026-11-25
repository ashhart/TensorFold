const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;

pub const View = struct { stream: *lanes.Stream, chunk_start: usize, end: usize, valid_from: usize, epoch: u64 };

pub fn normalizedFeatures(view: ?View, mtp: bool, stream: *lanes.Stream, epoch: u64, cache_written: u64, start: u64, count: u32, buffer: u64, row_bytes: usize) ?be.Features.Normalized {
    const v = view orelse return null;
    const requested_end = std.math.add(u64, start, count) catch return null;
    const chunk_start: u64 = @intCast(v.chunk_start);
    const end: u64 = @intCast(v.end);
    if (!mtp or v.stream != stream or v.epoch != epoch or cache_written != end or start < chunk_start or requested_end > end) return null;
    const norm_start = @max(start, @as(u64, @intCast(v.valid_from)));
    const norm_end = @min(requested_end, end);
    const norm_rows = if (norm_end > norm_start) norm_end - norm_start else 0;
    const byte_offset = std.math.mul(u64, norm_start - chunk_start, @intCast(row_bytes)) catch return null;
    return .{ .buffer = buffer, .byte_offset = byte_offset, .start = norm_start, .rows = @intCast(norm_rows), .row_bytes = @intCast(row_bytes) };
}

pub fn directSuffixRequired(expected: ?View, stream: *lanes.Stream, start: u64, count: u32) bool {
    const v = expected orelse return false;
    if (v.stream != stream or stream.prompt_len == 0) return false;
    const request_end = std.math.add(u64, start, count) catch return true;
    const required_start: u64 = @intCast(v.valid_from);
    const prompt_last: u64 = @intCast(stream.prompt_len - 1);
    const required_end = @min(@as(u64, @intCast(v.end)), prompt_last);
    return required_start < required_end and start < required_end and request_end > required_start;
}

fn finalPromptRow(stream: *lanes.Stream, start: u64, count: u32) bool {
    return stream.prompt_len > 0 and count == 1 and start == @as(u64, @intCast(stream.prompt_len - 1));
}

pub fn normalizedFeaturesOrExpired(expected: ?View, view: ?View, mtp: bool, stream: *lanes.Stream, epoch: u64, cache_written: u64, start: u64, count: u32, buffer: u64, row_bytes: usize) !?be.Features.Normalized {
    if (finalPromptRow(stream, start, count)) return null;
    const normalized = normalizedFeatures(view, mtp, stream, epoch, cache_written, start, count, buffer, row_bytes);
    if (normalized == null and directSuffixRequired(expected, stream, start, count)) return error.MtpNormalizedViewExpired;
    return normalized;
}

pub fn release(normalized: *?View, expected: *?View, stream: *lanes.Stream) void {
    if (expected.*) |view| if (view.stream == stream) {
        expected.* = null;
        normalized.* = null;
    };
}

test "MTP normalized prefill views reject stale stream, epoch, cache, and range" {
    var a: lanes.Stream = undefined;
    var b: lanes.Stream = undefined;
    const stream: *lanes.Stream = &a;
    const other: *lanes.Stream = &b;
    const view: View = .{ .stream = stream, .chunk_start = 100, .end = 200, .valid_from = 150, .epoch = 7 };
    const valid = normalizedFeatures(view, true, stream, 7, 200, 140, 40, 0x1000, 64).?;
    try std.testing.expectEqual(@as(u64, 150), valid.start);
    try std.testing.expectEqual(@as(u32, 30), valid.rows);
    try std.testing.expectEqual(@as(u64, 3200), valid.byte_offset);
    try std.testing.expect(normalizedFeatures(view, false, stream, 7, 200, 140, 40, 0x1000, 64) == null);
    try std.testing.expect(normalizedFeatures(view, true, other, 7, 200, 140, 40, 0x1000, 64) == null);
    try std.testing.expect(normalizedFeatures(view, true, stream, 8, 200, 140, 40, 0x1000, 64) == null);
    try std.testing.expect(normalizedFeatures(view, true, stream, 7, 199, 140, 40, 0x1000, 64) == null);
    try std.testing.expect(normalizedFeatures(view, true, stream, 7, 200, 99, 40, 0x1000, 64) == null);
    try std.testing.expect(normalizedFeatures(view, true, stream, 7, 200, 180, 21, 0x1000, 64) == null);
    try std.testing.expect(normalizedFeatures(view, true, stream, 7, 200, std.math.maxInt(u64) - 2, 4, 0x1000, 64) == null);
}

test "MTP normalized suffix expiry fails closed while prefix and final-row paths remain valid" {
    var target: lanes.Stream = undefined;
    var other: lanes.Stream = undefined;
    target.prompt_len = 8192;
    other.prompt_len = 8192;
    const stream: *lanes.Stream = &target;
    const view: View = .{ .stream = stream, .chunk_start = 4096, .end = 8192, .valid_from = 7935, .epoch = 7 };

    const valid = try normalizedFeaturesOrExpired(view, view, true, stream, 7, 8192, 4096, 4095, 0x1000, 64);
    try std.testing.expectEqual(@as(u64, 7935), valid.?.start);
    try std.testing.expectEqual(@as(u32, 256), valid.?.rows);

    try std.testing.expectError(error.MtpNormalizedViewExpired, normalizedFeaturesOrExpired(view, null, true, stream, 7, 8192, 4096, 4095, 0x1000, 64));
    try std.testing.expectError(error.MtpNormalizedViewExpired, normalizedFeaturesOrExpired(view, view, true, stream, 8, 8192, 4096, 4095, 0x1000, 64));
    try std.testing.expect((try normalizedFeaturesOrExpired(view, null, true, stream, 7, 8192, 4096, 3839, 0x1000, 64)) == null);
    try std.testing.expect((try normalizedFeaturesOrExpired(view, view, true, stream, 7, 8192, 8191, 1, 0x1000, 64)) == null);
    try std.testing.expect((try normalizedFeaturesOrExpired(view, null, true, stream, 8, 8192, 8191, 1, 0x1000, 64)) == null);
    try std.testing.expect((try normalizedFeaturesOrExpired(view, null, true, &other, 7, 8192, 4096, 4095, 0x1000, 64)) == null);
    try std.testing.expect((try normalizedFeaturesOrExpired(null, null, true, stream, 7, 8192, 4096, 4095, 0x1000, 64)) == null);
}

test "MTP normalized views release only with their stream" {
    var target: lanes.Stream = undefined;
    var other: lanes.Stream = undefined;
    var normalized: ?View = .{ .stream = &target, .chunk_start = 100, .end = 200, .valid_from = 150, .epoch = 7 };
    var expected = normalized;
    release(&normalized, &expected, &other);
    try std.testing.expect(normalized != null and expected != null);
    release(&normalized, &expected, &target);
    try std.testing.expect(normalized == null and expected == null);
}
