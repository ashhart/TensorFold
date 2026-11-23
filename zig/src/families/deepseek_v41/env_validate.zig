//! Validate optional DeepSeek settings before opening resources, including settings read only by later paths.
const std = @import("std");
const sys = @import("sys");
const environment = sys.environment;
const lookup = @import("draft/lookup.zig");

/// Every TF_DSV41_* row of the shared table and DeepSeek's cross-setting rules, parsed as each later reader parses it.
pub fn check(env: anytype) !void {
    try environment.checkScope(.dsv41, env);
    _ = try lookup.Settings.checkCopyDrafter(env);
}

pub fn refusal(a: std.mem.Allocator, env: anytype) !?[]const u8 {
    if (try environment.refusal(a, env, .{ .dsv41 = true })) |why| return why;
    _ = lookup.Settings.checkCopyDrafter(env) catch return try a.dupe(u8, lookup.Settings.copy_drafter_lookup_refusal);
    return null;
}

pub fn startup() !void {
    try check(environment.Process{});
}

/// BIAS_VL has always selected the bias when nonempty, including 0; validation does not reinterpret that value.
pub fn imageBiasEnv() !bool {
    const raw = sys.getenv("TF_DSV41_BIAS_VL");
    try environment.check("TF_DSV41_BIAS_VL", raw);
    return raw != null and std.mem.trim(u8, raw.?, " \t\r\n").len > 0;
}

test {
    _ = @import("env_validate_test.zig");
}
