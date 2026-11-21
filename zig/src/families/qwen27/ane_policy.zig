//! The fp16 Neural Engine prompt share is selected explicitly, only on a tensor-unit GPU.
const std = @import("std");
pub const default_columns: u32 = 0;
/// Loading the optional fp16 share does not establish exact output identity with the GPU path.
pub const exact_output_qualified = false;
/// 32 MiB bounds four-slice 27B private overhead across 20 fresh processes and the first 8K prompt.
pub const build_runtime_overhead: ?u64 = 32 << 20;

pub fn columns(value: ?[]const u8) !u32 {
    return if (value) |text| std.fmt.parseInt(u32, text, 10) else default_columns;
}

pub fn requestedOnChip(n: u32, tensor_units: bool) bool {
    return n > 0 and tensor_units;
}

/// Explicit ANE column share allowed on Apple GPU generation 15+ (M3 and newer) when opted in via TF_ANE_COLUMNS.
pub fn requestedOnGeneration(n: u32, generation: u32) bool {
    return n > 0 and generation >= 15;
}

/// Strict opt-in: `TF_ANE_INT8` defaults to false everywhere and only `"1"` or `"true"` enables W8A8 INT8 slices.
pub fn int8(value: ?[]const u8) !bool {
    const raw = value orelse return false;
    if (raw.len == 0 or std.mem.eql(u8, raw, "0") or std.mem.eql(u8, raw, "false")) return false;
    if (std.mem.eql(u8, raw, "1") or std.mem.eql(u8, raw, "true")) return true;
    return error.InvalidAneInt8;
}

/// Looped checkpoints keep `TF_ANE_INT8` opt-in too, identical to non-looped checkpoints.
pub fn int8For(value: ?[]const u8, _: bool) !bool {
    return int8(value);
}

test "ANE policy: the default is off and an explicit column count selects the opt-in" {
    try std.testing.expectEqual(@as(u32, 0), default_columns);
    try std.testing.expectEqual(@as(u32, 0), try columns(null));
    try std.testing.expectEqual(@as(u32, 0), try columns("0"));
    try std.testing.expectEqual(@as(u32, 3072), try columns("3072"));
    try std.testing.expect(!exact_output_qualified);
}

test "ANE policy: older GPUs never request a share, and tensor units alone do not turn it on" {
    for ([_]u32{ 0, 64, 3072 }) |n| try std.testing.expect(!requestedOnChip(n, false));
    try std.testing.expect(!requestedOnChip(0, true));
    try std.testing.expect(requestedOnChip(3072, true));
}

test "ANE policy: malformed and overflowing overrides keep refusing" {
    try std.testing.expectError(error.InvalidCharacter, columns(""));
    try std.testing.expectError(error.Overflow, columns("-1"));
    try std.testing.expectError(error.InvalidCharacter, columns("no"));
    try std.testing.expectError(error.Overflow, columns("4294967296"));
}

test "ANE policy: the measured private-overhead price is 32 MiB independent of columns" {
    try std.testing.expectEqual(@as(?u64, 32 << 20), build_runtime_overhead);
}

test "ANE policy: generation gating and strict opt-in TF_ANE_INT8 parsing" {
    try std.testing.expect(!requestedOnGeneration(1024, 14));
    try std.testing.expect(!requestedOnGeneration(0, 15));
    try std.testing.expect(requestedOnGeneration(1024, 15));
    try std.testing.expect(requestedOnGeneration(2048, 17));
    try std.testing.expect(!try int8(null));
    try std.testing.expect(!try int8(""));
    try std.testing.expect(!try int8("0"));
    try std.testing.expect(!try int8("false"));
    try std.testing.expect(try int8("1"));
    try std.testing.expect(try int8("true"));
    try std.testing.expectError(error.InvalidAneInt8, int8("yes"));
    try std.testing.expect(!try int8For(null, true));
    try std.testing.expect(!try int8For("", true));
    try std.testing.expect(!try int8For("0", true));
    try std.testing.expect(!try int8For("false", true));
    try std.testing.expect(try int8For("1", true));
    try std.testing.expect(try int8For("1", false));
    try std.testing.expect(try int8For("true", true));
}
