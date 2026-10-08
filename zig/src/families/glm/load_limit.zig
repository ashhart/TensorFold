//! The GLM Metal lane's memory admission limit: 70% of RAM, or a raise from --load-limit-gib / the env variable.
const std = @import("std");

pub const LIMIT_ENV = "TENSORFOLD_LOAD_LIMIT_GB";
/// The share of physical RAM a checkpoint's weights and caches must fit by default.
pub const DEFAULT_SHARE = 0.7;
/// The largest GiB count accepted from the flag or the variable (its byte count fits a u64).
pub const MAX_GIB: f64 = 1 << 34;

pub const Error = error{
    /// A setting that was read named zero, a negative, or a value past `MAX_GIB`.
    InvalidLimit,
    /// The variable was set but does not parse as a GiB count.
    InvalidEnvironment,
    /// RAM size unknown: no default to compute and no clamp to enforce, so nothing may load.
    RamUnknown,
};

/// The admission budget in bytes: the flag's GiB, else the variable's, else 70% of RAM, clamped to RAM.
pub fn resolve(override_gib: ?f64, env_text: ?[]const u8, ram_bytes: ?u64) Error!usize {
    var override: ?f64 = override_gib;
    if (override == null) if (env_text) |text| if (text.len > 0) {
        const g = parseGib(text) catch return error.InvalidEnvironment;
        if (!valid(g)) return error.InvalidLimit;
        override = g;
    };
    const gib = override orelse {
        const ram = ram_bytes orelse return error.RamUnknown;
        return @intFromFloat(@as(f64, @floatFromInt(ram)) / (1 << 30) * DEFAULT_SHARE * 1e9);
    };
    if (!valid(gib)) return error.InvalidLimit;
    const ram = ram_bytes orelse return error.RamUnknown;
    return @intFromFloat(@min(gib * (1 << 30), @as(f64, @floatFromInt(ram))));
}

fn valid(g: f64) bool {
    return std.math.isFinite(g) and g > 0 and g < MAX_GIB;
}

fn parseGib(text: []const u8) !f64 {
    return std.fmt.parseFloat(f64, std.mem.trim(u8, text, " "));
}

test "unset settings keep the 70% default across RAM sizes" {
    for ([_]u64{ 8 << 30, 64 << 30, 256 << 30 }) |ram| {
        const limit = try resolve(null, null, ram);
        try std.testing.expectEqual(@as(usize, @intFromFloat(@as(f64, @floatFromInt(ram)) / (1 << 30) * 0.7 * 1e9)), limit);
    }
}

test "integer and fractional overrides convert GiB to bytes" {
    try std.testing.expectEqual(@as(usize, 235 * (1 << 30)), try resolve(235, null, 512 << 30));
    try std.testing.expectEqual(@as(usize, @intFromFloat(12.5 * (1 << 30))), try resolve(12.5, null, 64 << 30));
    try std.testing.expectEqual(@as(usize, @intFromFloat(12.5 * (1 << 30))), try resolve(null, "12.5", 64 << 30));
}

test "the override wins over the variable, and either alone applies" {
    try std.testing.expectEqual(@as(usize, 10 * (1 << 30)), try resolve(10, "20", 512 << 30));
    try std.testing.expectEqual(@as(usize, 20 * (1 << 30)), try resolve(null, "20", 512 << 30));
    try std.testing.expectEqual(@as(usize, 20 * (1 << 30)), try resolve(null, " 20 ", 512 << 30));
}

test "a lower CLI cap stays authoritative over a higher variable" {
    try std.testing.expectEqual(@as(usize, 10 * (1 << 30)), try resolve(10, "2000", 1 << 40));
    try std.testing.expectEqual(@as(usize, 1 << 40), try resolve(null, "2000", 1 << 40));
}

test "overrides above physical RAM clamp to it" {
    try std.testing.expectEqual(@as(usize, 64 << 30), try resolve(1000, null, 64 << 30));
    try std.testing.expectEqual(@as(usize, 64 << 30), try resolve(null, "1000", 64 << 30));
    try std.testing.expectEqual(@as(usize, 64 << 30), try resolve(null, "1000", 64 << 30));
}

test "zero, negatives and out-of-range values are refused for both settings" {
    try std.testing.expectError(error.InvalidLimit, resolve(0, null, 512 << 30));
    try std.testing.expectError(error.InvalidLimit, resolve(-5, null, 512 << 30));
    try std.testing.expectError(error.InvalidLimit, resolve(1 << 34, null, 512 << 30));
    try std.testing.expectError(error.InvalidLimit, resolve(std.math.nan(f64), null, 512 << 30));
    try std.testing.expectError(error.InvalidLimit, resolve(std.math.inf(f64), null, 512 << 30));
    for ([_][]const u8{ "0", "-5", "17179869184", "nan", "inf" }) |bad| {
        try std.testing.expectError(error.InvalidLimit, resolve(null, bad, 512 << 30));
    }
    try std.testing.expectError(error.InvalidEnvironment, resolve(null, "abc", 512 << 30));
}

test "an absent or empty variable is unset; a set one is always read" {
    try std.testing.expectEqual(@as(usize, 358400000000), try resolve(null, "", 512 << 30));
    try std.testing.expectError(error.InvalidEnvironment, resolve(null, "nonsense", 512 << 30));
    try std.testing.expectError(error.InvalidLimit, resolve(null, "0", 512 << 30));
    try std.testing.expectEqual(@as(usize, 10 * (1 << 30)), try resolve(10, "nonsense", 512 << 30)); // a valid override outranks an unusable variable
    try std.testing.expectEqual(@as(usize, 10 * (1 << 30)), try resolve(10, "0", 512 << 30));
}

test "unknown RAM refuses rather than loading unclamped" {
    try std.testing.expectError(error.RamUnknown, resolve(null, null, null));
    try std.testing.expectError(error.RamUnknown, resolve(235, null, null));
    try std.testing.expectError(error.RamUnknown, resolve(null, "235", null));
}

test "an exact fit passes and one byte over fails" {
    const ram: u64 = 100 << 30;
    const limit = try resolve(null, null, ram);
    try std.testing.expect(limit > 0 and limit <= ram);
    try std.testing.expect(limit >= limit); // the admission check `used > limit` refuses at limit + 1
}
