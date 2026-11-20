const std = @import("std");
const runtime = @import("runtime.zig");
const elementwise = @import("elementwise.zig");
const spatial = @import("spatial.zig");

const Runtime = runtime.Runtime;
const Tensor = runtime.Tensor;
const FeatureMap = runtime.FeatureMap;

pub const max_slots = 16;
pub const absent_logit: f32 = -1024;
const fill_presence: f32 = -10;
const fill_quality: f32 = 0;

pub const Pick = enum { primary, best_alternative, best_any };

pub const TensorBuffer = [4]Tensor;

pub const Outcome = struct {
    mask_logits: FeatureMap,
    full_logits: ?FeatureMap = null,
    /// Map the frame's masks are read from when it differs from the full or mask logits (a mask-prompt passthrough).
    readout_logits: ?FeatureMap = null,
    pointers: Tensor,
    slots: usize,
    presence: [max_slots]f32,
    quality: [max_slots]f32,

    pub fn tensors(self: Outcome, buffer: *TensorBuffer) []const Tensor {
        buffer[0] = self.mask_logits.tensor;
        buffer[1] = self.pointers;
        var count: usize = 2;
        for ([_]?FeatureMap{ self.full_logits, self.readout_logits }) |maybe| if (maybe) |map| {
            buffer[count] = map.tensor;
            count += 1;
        };
        return buffer[0..count];
    }

    /// The logits masks are read from: readout, else full, else mask logits.
    pub fn shown(self: Outcome) FeatureMap {
        return self.readout_logits orelse self.full_logits orelse self.mask_logits;
    }
};

pub fn pick(quality: []const f32, rule: Pick) usize {
    const first: usize = switch (rule) {
        .primary => return 0,
        .best_alternative => 1,
        .best_any => 0,
    };
    if (quality.len <= first) return 0;
    var best = first;
    for (first + 1..quality.len) |i| {
        if (quality[i] > quality[best]) best = i;
    }
    return best;
}

pub fn present(logit: f32) bool {
    return !(logit <= 0);
}

pub fn single(rt: *Runtime, group: Outcome, slot: usize) !Outcome {
    const width = rt.config.hidden;
    if (slot >= group.slots or group.pointers.count != group.slots * width) return error.SamOutcomeShape;
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    var result = Outcome{
        .mask_logits = try column(rt, group.mask_logits, group.slots, slot),
        .full_logits = if (group.full_logits) |full| try column(rt, full, group.slots, slot) else null,
        .readout_logits = if (group.readout_logits) |shown_map| try column(rt, shown_map, group.slots, slot) else null,
        .pointers = try elementwise.concat(rt, &.{runtime.part(group.pointers, slot * width, width)}),
        .slots = 1,
        .presence = @splat(0),
        .quality = @splat(0),
    };
    result.presence[0] = group.presence[slot];
    result.quality[0] = group.quality[slot];
    return result;
}

fn column(rt: *Runtime, map: FeatureMap, slots: usize, slot: usize) !FeatureMap {
    const pixels = map.height * map.width;
    if (map.channels != slots or map.tensor.count != pixels * slots) return error.SamOutcomeShape;
    const values = try elementwise.columns(rt, map.tensor, pixels, slots, slot, 1);
    return FeatureMap.of(values, map.height, map.width, 1);
}

const Fills = struct { presence: [max_slots]f32, quality: [max_slots]f32 };

fn fillArrays(parts: []const ?Outcome) Fills {
    var fills = Fills{ .presence = @splat(fill_presence), .quality = @splat(fill_quality) };
    for (parts, 0..) |maybe, s| if (maybe) |part| {
        fills.presence[s] = part.presence[0];
        fills.quality[s] = part.quality[0];
    };
    return fills;
}

fn fullSize(parts: []const ?Outcome) ?[2]usize {
    for (parts) |maybe| if (maybe) |part| if (part.full_logits) |full| return .{ full.height, full.width };
    return null;
}

fn readoutSize(parts: []const ?Outcome) ?[2]usize {
    for (parts) |maybe| if (maybe) |part| if (part.readout_logits) |map| return .{ map.height, map.width };
    return null;
}

fn checkPart(part: Outcome, side: usize, width: usize) !void {
    const logits = part.mask_logits;
    if (part.slots != 1 or logits.height != side or logits.width != side or logits.channels != 1 or
        logits.tensor.count != side * side or part.pointers.count != width) return error.SamOutcomeShape;
    for ([_]?FeatureMap{ part.full_logits, part.readout_logits }) |maybe| if (maybe) |map| {
        if (map.channels != 1 or map.tensor.count != map.height * map.width) return error.SamOutcomeShape;
    };
}

pub fn merge(rt: *Runtime, parts: []const ?Outcome) !Outcome {
    const n = parts.len;
    if (n == 0 or n > max_slots) return error.SamOutcomeShape;
    const side = rt.config.maskSide();
    const width = rt.config.hidden;
    var missing = false;
    for (parts) |maybe| {
        if (maybe) |part| try checkPart(part, side, width) else missing = true;
    }
    const size = fullSize(parts);
    const shown_size = readoutSize(parts);
    const m = rt.mark();
    errdefer rt.rewind(m, &.{});
    const fill_mask: ?Tensor = if (missing) try elementwise.hostFill(rt, side * side, absent_logit) else null;
    const fill_pointer: ?Tensor = if (missing) try elementwise.hostFill(rt, width, 0) else null;
    const fill_full: ?Tensor = if (missing and size != null) try elementwise.hostFill(rt, size.?[0] * size.?[1], absent_logit) else null;
    const fill_shown: ?Tensor = if (missing and shown_size != null) try elementwise.hostFill(rt, shown_size.?[0] * shown_size.?[1], absent_logit) else null;

    const opened = rt.gpu.batch == null;
    if (opened) try rt.gpu.beginBatch();
    errdefer if (opened and rt.gpu.batch != null) rt.gpu.endBatch() catch {};
    var slot_logits: [max_slots]Tensor = undefined;
    var pointers: [max_slots]Tensor = undefined;
    for (parts, 0..) |maybe, s| {
        slot_logits[s] = if (maybe) |part| part.mask_logits.tensor else fill_mask.?;
        pointers[s] = if (maybe) |part| part.pointers else fill_pointer.?;
    }
    const fills = fillArrays(parts);
    var result = Outcome{
        .mask_logits = FeatureMap.of(try stack(rt, slot_logits[0..n], side * side), side, side, n),
        .pointers = try elementwise.concat(rt, pointers[0..n]),
        .slots = n,
        .presence = fills.presence,
        .quality = fills.quality,
    };
    if (size) |hw| {
        var full: [max_slots]Tensor = undefined;
        for (parts, 0..) |maybe, s| full[s] = if (maybe) |part| try fullPart(rt, part, hw) else fill_full.?;
        result.full_logits = FeatureMap.of(try stack(rt, full[0..n], hw[0] * hw[1]), hw[0], hw[1], n);
    }
    if (shown_size) |hw| {
        var shown: [max_slots]Tensor = undefined;
        for (parts, 0..) |maybe, s| shown[s] = if (maybe) |part| try readoutPart(rt, part, hw) else fill_shown.?;
        result.readout_logits = FeatureMap.of(try stack(rt, shown[0..n], hw[0] * hw[1]), hw[0], hw[1], n);
    }
    if (opened) try rt.gpu.endBatch();
    var buffer: TensorBuffer = undefined;
    rt.rewind(m, result.tensors(&buffer));
    return result;
}

// Full logits already at the target size enter unchanged; every other source is resampled bilinearly.
fn fullPart(rt: *Runtime, part: Outcome, size: [2]usize) !Tensor {
    if (part.full_logits) |full| if (full.height == size[0] and full.width == size[1]) return full.tensor;
    const source = part.full_logits orelse part.mask_logits;
    return (try spatial.resample(rt, source, size[0], size[1], .bilinear)).tensor;
}

// A slot without its own readout map contributes its shown logits resampled bilinearly to the readout size.
fn readoutPart(rt: *Runtime, part: Outcome, size: [2]usize) !Tensor {
    const source = part.shown();
    if (source.height == size[0] and source.width == size[1]) return source.tensor;
    return (try spatial.resample(rt, source, size[0], size[1], .bilinear)).tensor;
}

fn stack(rt: *Runtime, slots: []const Tensor, pixels: usize) !Tensor {
    const rows = try elementwise.concat(rt, slots);
    return elementwise.transpose(rt, rows, slots.len, pixels);
}

fn testPart(presence: f32, quality: f32, full: ?[2]usize) Outcome {
    const empty = Tensor{ .buffer = undefined, .count = 0, .dtype = .f32 };
    var part = Outcome{
        .mask_logits = FeatureMap.of(empty, 288, 288, 1),
        .full_logits = if (full) |hw| FeatureMap.of(empty, hw[0], hw[1], 1) else null,
        .pointers = empty,
        .slots = 1,
        .presence = @splat(0),
        .quality = @splat(0),
    };
    part.presence[0] = presence;
    part.quality[0] = quality;
    return part;
}

test "SAM candidate rule scans upward with strict comparison" {
    const four = [_]f32{ 0.2, 0.7, 0.95, 0.6 };
    try std.testing.expectEqual(@as(usize, 2), pick(&four, .best_alternative));
    try std.testing.expectEqual(@as(usize, 0), pick(&four, .primary));
    try std.testing.expectEqual(@as(usize, 2), pick(&four, .best_any));
    try std.testing.expectEqual(@as(usize, 1), pick(&.{ 0.1, 0.5, 0.3, 0.5 }, .best_alternative));
    try std.testing.expectEqual(@as(usize, 0), pick(&.{ 0.9, 0.9, 0.2 }, .best_any));
    try std.testing.expectEqual(@as(usize, 2), pick(&.{ 0.2, 0.1, 0.9 }, .best_any));
    try std.testing.expectEqual(@as(usize, 3), pick(&.{ 0.5, 0.4, std.math.nan(f32), 0.45 }, .best_alternative));
}

test "SAM presence treats NaN as present and signed zeros as absent" {
    for ([_]f32{ -std.math.inf(f32), -3.5, -0.0, 0.0 }) |logit| try std.testing.expect(!present(logit));
    for ([_]f32{ std.math.floatTrueMin(f32), 7, std.math.inf(f32), std.math.nan(f32) }) |logit|
        try std.testing.expect(present(logit));
}

test "SAM merge fills missing slots and takes the full-logit size from the first slot that has one" {
    var parts: [16]?Outcome = @splat(null);
    parts[0] = testPart(3.5, 0.25, null);
    parts[3] = testPart(-2, 0.75, .{ 540, 960 });
    const fills = fillArrays(&parts);
    for (0..16) |s| {
        const expected_presence: f32 = switch (s) {
            0 => 3.5,
            3 => -2,
            else => -10,
        };
        const expected_quality: f32 = switch (s) {
            0 => 0.25,
            3 => 0.75,
            else => 0,
        };
        try std.testing.expectEqual(expected_presence, fills.presence[s]);
        try std.testing.expectEqual(expected_quality, fills.quality[s]);
    }
    try std.testing.expectEqual([2]usize{ 540, 960 }, fullSize(&parts).?);
    parts[1] = testPart(1, 1, .{ 7, 5 });
    try std.testing.expectEqual([2]usize{ 7, 5 }, fullSize(&parts).?);
    parts[1] = null;
    parts[3] = testPart(1, 1, null);
    try std.testing.expectEqual(@as(?[2]usize, null), fullSize(&parts));
}
