//! Source selection for the shared M5 attention product; other kernels retain their own rounding policy.
const std = @import("std");

pub const env = "TF_ATTENTION_PV_FP32";
pub const define = "#define TF_ATTN_PV_FP32 1\n";

/// Only an explicit 1 enables the mode; an unset or zero value keeps the existing product.
pub fn parse(value: ?[]const u8) !bool {
    const text = value orelse return false;
    if (std.mem.eql(u8, text, "0")) return false;
    if (std.mem.eql(u8, text, "1")) return true;
    return error.AttentionPrecisionValue;
}

/// Read while building the engine's libraries, before serving begins.
pub fn requested() !bool {
    return parse(if (@import("sys").c.getenv(env)) |v| std.mem.span(v) else null);
}

/// Expand fragments with an explicit mode, so contracts never inherit a caller's environment.
pub fn source(a: std.mem.Allocator, units: bool, text: []const u8, header: []const u8, full: bool) ![]u8 {
    const body = try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", header);
    defer a.free(body);
    const head = if (!units) "#define TF_SIMD_FRAGS 1\n" else if (full) define else "";
    return std.mem.concat(a, u8, &.{ head, body });
}

/// Keep the default identity and separate opt-in states even when a short calibration probe misses the difference.
pub fn hashMode(h: *std.hash.Wyhash, full: bool) void {
    if (full) h.update("attention-pv-fp32-v1");
}

test "attention precision: unset and zero preserve the default; only one opts in" {
    try std.testing.expect(!try parse(null));
    try std.testing.expect(!try parse("0"));
    try std.testing.expect(try parse("1"));
    for ([_][]const u8{ "", "true", "01", "-1", "2" }) |v| try std.testing.expectError(error.AttentionPrecisionValue, parse(v));
}

test "attention precision: source modes and the older-chip source are explicit" {
    const a = std.testing.allocator;
    const text = "#include \"../nax.h\"\nBODY";
    const off = try source(a, true, text, "HEADER", false);
    defer a.free(off);
    const on = try source(a, true, text, "HEADER", true);
    defer a.free(on);
    const old = try source(a, false, text, "HEADER", true);
    defer a.free(old);
    try std.testing.expectEqualStrings("HEADER\nBODY", off);
    try std.testing.expectEqualStrings(define ++ "HEADER\nBODY", on);
    try std.testing.expectEqualStrings("#define TF_SIMD_FRAGS 1\nHEADER\nBODY", old);
}

test "attention precision: kept-state identity separates the mode without moving the default" {
    var base = std.hash.Wyhash.init(17);
    base.update("checkpoint");
    var off = base;
    hashMode(&off, false);
    var on = base;
    hashMode(&on, true);
    try std.testing.expectEqual(base.final(), off.final());
    try std.testing.expect(on.final() != off.final());
}
