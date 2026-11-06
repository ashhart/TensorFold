//! Qwen3.5's attention-layer pattern from a text config: one rule for `layer_types` and `full_attention_interval`.
const std = @import("std");

/// Each refusal names the key at fault; a contradiction names both.
pub const Error = error{ BadLayerTypes, UnsupportedLayerTypes, BadFullAttentionInterval, LayerTypesContradictFullAttentionInterval };

/// transformers' `full_attention_interval` when config.json names none.
pub const default_interval = 4;

/// `full_attention_interval`: absent; unset (null, 0 or false); or every `n`-th layer is full attention.
const Interval = union(enum) { absent, unset, every: usize };

/// Into `full` (true: full attention): `layer_types`, checked against a given interval; else every interval-th layer.
pub fn read(full: []bool, tower: std.json.ObjectMap) Error!void {
    const interval = try intervalOf(tower.get("full_attention_interval"));
    const list = tower.get("layer_types") orelse .null;
    if (list == .null) {
        const n = switch (interval) {
            .absent => default_interval,
            .unset => return error.BadFullAttentionInterval,
            .every => |n| n,
        };
        for (full, 0..) |*f, i| f.* = every(n, i);
        return;
    }
    if (list != .array or list.array.items.len != full.len) return error.BadLayerTypes;
    for (list.array.items, full) |item, *f| {
        if (item != .string) return error.BadLayerTypes;
        if (std.mem.eql(u8, item.string, "full_attention")) f.* = true else if (std.mem.eql(u8, item.string, "linear_attention")) f.* = false else return error.UnsupportedLayerTypes;
    }
    if (interval == .every) for (full, 0..) |f, i| if (f != every(interval.every, i)) return error.LayerTypesContradictFullAttentionInterval;
}

/// Whether layer `i` is full attention when every `n`-th layer is.
pub fn every(n: usize, i: usize) bool {
    return (i + 1) % n == 0;
}

/// A positive whole number (4 or 4.0) is an interval; null, 0 and false leave it unset; anything else is refused.
fn intervalOf(v: ?std.json.Value) Error!Interval {
    return switch (v orelse return .absent) {
        .null => .unset,
        .bool => |b| if (b) error.BadFullAttentionInterval else .unset,
        .integer => |i| if (i == 0) .unset else if (i > 0) .{ .every = std.math.cast(usize, i) orelse return error.BadFullAttentionInterval } else error.BadFullAttentionInterval,
        .float => |f| if (f == 0) .unset else if (f >= 1 and f <= 1 << 32 and f == @floor(f)) .{ .every = @intFromFloat(f) } else error.BadFullAttentionInterval,
        else => error.BadFullAttentionInterval,
    };
}

const t = std.testing;
const four = &[_]bool{ false, false, false, true, false, false, false, true };
const types4 = "\"layer_types\":[\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"," ++
    "\"linear_attention\",\"linear_attention\",\"linear_attention\",\"full_attention\"]";

/// `read` of a tower whose JSON text is `fmt` filled with `args` inside braces, for `layers` layers.
fn pattern(buf: *[64]bool, comptime fmt: []const u8, args: anytype, layers: usize) Error![]const bool {
    const text = std.fmt.allocPrint(t.allocator, "{{" ++ fmt ++ "}}", args) catch unreachable;
    defer t.allocator.free(text);
    const doc = std.json.parseFromSlice(std.json.Value, t.allocator, text, .{}) catch unreachable;
    defer doc.deinit();
    try read(buf[0..layers], doc.value.object);
    return buf[0..layers];
}

test "no layer_types, or null: every fourth layer when the interval is absent, every n-th when it names n" {
    var b: [64]bool = undefined;
    try t.expectEqualSlices(bool, four, try pattern(&b, "", .{}, 8));
    try t.expectEqualSlices(bool, four, try pattern(&b, "\"layer_types\":null", .{}, 8));
    try t.expectEqualSlices(bool, four, try pattern(&b, "\"full_attention_interval\":4.0", .{}, 8));
    try t.expectEqualSlices(bool, &.{ false, true, false, true }, try pattern(&b, "\"full_attention_interval\":2", .{}, 4));
    try t.expectEqualSlices(bool, &.{ true, true }, try pattern(&b, "\"layer_types\":null,\"full_attention_interval\":1", .{}, 2));
    try t.expectEqualSlices(bool, &.{ false, false }, try pattern(&b, "\"full_attention_interval\":9", .{}, 2));
}

test "no layer_types, or null, and an unset or malformed interval: refused by the interval's name" {
    var b: [64]bool = undefined;
    for ([_][]const u8{ "null", "0", "false", "true", "-4", "4.5", "\"4\"", "[4]", "{}", "0.5" }) |v| {
        try t.expectError(error.BadFullAttentionInterval, pattern(&b, "\"full_attention_interval\":{s}", .{v}, 8));
        try t.expectError(error.BadFullAttentionInterval, pattern(&b, "\"layer_types\":null,\"full_attention_interval\":{s}", .{v}, 8));
    }
}

test "layer_types read as written, in any order, when the interval is absent or unset (null, 0, false)" {
    var b: [64]bool = undefined;
    const mixed = "\"layer_types\":[\"full_attention\",\"linear_attention\",\"full_attention\"]";
    for ([_][]const u8{ "", ",\"full_attention_interval\":null", ",\"full_attention_interval\":0", ",\"full_attention_interval\":false" }) |interval| {
        try t.expectEqualSlices(bool, &.{ true, false, true }, try pattern(&b, mixed ++ "{s}", .{interval}, 3));
    }
}

test "layer_types and an interval: accepted when they agree, refused by both names when they do not" {
    var b: [64]bool = undefined;
    try t.expectEqualSlices(bool, four, try pattern(&b, types4 ++ ",\"full_attention_interval\":4", .{}, 8));
    try t.expectEqualSlices(bool, four, try pattern(&b, types4 ++ ",\"full_attention_interval\":4.0", .{}, 8));
    for ([_][]const u8{ "1", "2", "3", "5", "8", "9" }) |n| {
        try t.expectError(error.LayerTypesContradictFullAttentionInterval, pattern(&b, types4 ++ ",\"full_attention_interval\":{s}", .{n}, 8));
    }
    for ([_][]const u8{ "4.5", "\"4\"" }) |v| {
        try t.expectError(error.BadFullAttentionInterval, pattern(&b, types4 ++ ",\"full_attention_interval\":{s}", .{v}, 8));
    }
}

test "layer_types that are not a list of the layers' kinds: the wrong length, a non-string, another kind" {
    var b: [64]bool = undefined;
    try t.expectError(error.BadLayerTypes, pattern(&b, types4, .{}, 7));
    try t.expectError(error.BadLayerTypes, pattern(&b, types4, .{}, 9));
    for ([_][]const u8{ "true", "\"full_attention\"", "{}", "4", "[\"full_attention\",null]", "[\"full_attention\",4]" }) |v| {
        try t.expectError(error.BadLayerTypes, pattern(&b, "\"layer_types\":{s}", .{v}, 2));
    }
    for ([_][]const u8{ "sliding_attention", "indexed_attention", "Full_attention", "" }) |kind| {
        try t.expectError(error.UnsupportedLayerTypes, pattern(&b, "\"layer_types\":[\"linear_attention\",\"{s}\"]", .{kind}, 2));
    }
}
