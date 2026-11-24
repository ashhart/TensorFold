//! The EXL3 chain's settings: bounds, exact words and switches, the profile path reader and the video token budget.
const std = @import("std");
const e = @import("environment.zig");
const sys = @import("sys.zig");
const t = std.testing;
const a = t.allocator;

test "environment: Flash Next EXL3 rows keep their bounds, exact words and exact switches" {
    for ([_][2][]const u8{
        .{ "TF_FLASHNEXT_CHUNK", "0" },
        .{ "TF_FLASHNEXT_CHUNK", "16385" },
        .{ "TF_FLASHNEXT_CHUNK_SOLO", "16385" },
        .{ "TF_FLASHNEXT_EXL3_STREAMS", "0" },
        .{ "TF_FLASHNEXT_EXPERT_RESERVE_MB", "17592186044416" },
        .{ "TF_FLASHNEXT_CACHE_TELEMETRY", "true" },
        .{ "TF_FLASHNEXT_EXL3_PROFILE_SIZING", "yes" },
        .{ "TF_FLASHNEXT_SNAP_HOST", "01" },
        .{ "TF_FLASHNEXT_PLE_READ", "Pread" },
        .{ "TF_FLASHNEXT_EXL3_ROUTED", "DEVICE" },
        .{ "TF_FLASHNEXT_EXL3_ARCH", "1" },
        .{ "TENSORFOLD_MEMORY_RESERVE_GIB", "-0.5" },
    }) |c| try t.expectError(error.InvalidEnvironment, e.parse(c[0], c[1]));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_FLASHNEXT_EXL3_ROUTED", "host"));
    try t.expectEqual(@as(?u64, 0), try e.parse("TF_FLASHNEXT_PLE_READ", "map"));
}

test "environment: readPath reads the process environment, blank as unset, and refuses control characters" {
    const name = "TF_FLASHNEXT_EXL3_PROFILE_DIR";
    const previous: ?[:0]u8 = if (sys.getenv(name)) |raw| try a.dupeSentinel(u8, raw, 0) else null;
    defer if (previous) |raw| {
        _ = sys.env.setenv(name, raw, 1);
        a.free(raw);
    } else {
        _ = sys.env.unsetenv(name);
    };
    _ = sys.env.unsetenv(name);
    try t.expectEqual(@as(?[]const u8, null), try e.readPath(name));
    for ([_][:0]const u8{ "", "  " }) |blank| {
        try t.expectEqual(@as(c_int, 0), sys.env.setenv(name, blank, 1));
        try t.expectEqual(@as(?[]const u8, null), try e.readPath(name));
    }
    try t.expectEqual(@as(c_int, 0), sys.env.setenv(name, " /profiles ", 1));
    try t.expectEqualStrings(" /profiles ", (try e.readPath(name)).?);
    try t.expectEqual(@as(c_int, 0), sys.env.setenv(name, "/pro\nfiles", 1));
    try t.expectError(error.InvalidEnvironment, e.readPath(name));
}

test "environment: video token budgets retain the full unsigned count and refuse malformed input" {
    const name = "TENSORFOLD_VIDEO_TOKENS";
    try t.expectEqual(std.math.maxInt(u64), try e.value(u64, name, "18446744073709551615", 0));
    try t.expectEqual(@as(u64, 0), try e.value(u64, name, "0", 7));
    for ([_][]const u8{ "-1", "18446744073709551616", "32k" }) |raw|
        try t.expectError(error.InvalidEnvironment, e.value(u64, name, raw, 7));
}
