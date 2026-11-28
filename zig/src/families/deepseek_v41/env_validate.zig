//! Validate optional DeepSeek settings before opening resources, including settings read only by later paths.
const std = @import("std");
const sys = @import("sys");
const environment = sys.environment;
const lookup = @import("draft/lookup.zig");

/// TF_EXPERIMENTAL stays at the model/server admission gates; other DSV41 tools historically ignored it.
fn WithoutExperimental(comptime Env: type) type {
    return struct {
        env: Env,
        pub fn get(self: @This(), name: [:0]const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_EXPERIMENTAL")) return null;
            return self.env.get(name);
        }
    };
}

/// Every TF_DSV41_* row of the shared table and DeepSeek's cross-setting rules, parsed as each later reader parses it.
pub fn check(env: anytype) !void {
    try environment.checkScope(.dsv41, WithoutExperimental(@TypeOf(env)){ .env = env });
    _ = try lookup.Settings.checkCopyDrafter(env);
}

pub fn refusal(a: std.mem.Allocator, env: anytype) !?[]const u8 {
    if (try environment.refusal(a, WithoutExperimental(@TypeOf(env)){ .env = env }, .{ .dsv41 = true })) |why| return why;
    _ = lookup.Settings.checkCopyDrafter(env) catch return try a.dupe(u8, lookup.Settings.copy_drafter_lookup_refusal);
    return null;
}

pub fn startup() !void {
    try check(environment.Process{});
    _ = try @import("graphs.zig").Settings.fromEnv();
}

/// Model admission retains the experimental gate's exact public refusal.
pub fn modelAdmission() !void {
    const quiet = @import("builtin").is_test;
    const experimental = environment.rawText("TF_EXPERIMENTAL");
    _ = environment.parse("TF_EXPERIMENTAL", experimental) catch {
        if (!quiet) std.log.err("{s}", .{sys.experimental.deepseek_message});
        return error.ExperimentalDeepSeekRequired;
    };
    if (sys.experimental.refusal("deepseek_v41", experimental)) |text| {
        if (!quiet) std.log.err("{s}", .{text});
        return error.ExperimentalDeepSeekRequired;
    }
}

/// The model's stall setting retains its seconds-to-nanoseconds conversion and overflow refusal.
pub fn stallTimeoutNs() !u64 {
    const seconds = environment.read(u64, "TF_DSV41_STALL_EXIT_S", 0) catch return error.BadStallTimeout;
    return std.math.mul(u64, seconds, std.time.ns_per_s) catch return error.BadStallTimeout;
}

/// Engram's present blank cache count was invalid whenever its directory reader opened it; unset keeps 65536.
pub fn engramCache(raw: ?[]const u8) !u32 {
    if (raw != null and std.mem.trim(u8, raw.?, " \t\r\n").len == 0) return error.InvalidEnvironment;
    return environment.value(u32, "TF_DSV41_ENGRAM_CACHE", raw, 65536);
}

/// Empty and spaces-only profiling rows stay no-ops; other whitespace-only text keeps the legacy parse refusal.
pub fn profilingRows(raw: ?[]const u8) !?[]const u8 {
    if (raw) |text| {
        if (std.mem.trim(u8, text, " ").len != 0 and std.mem.trim(u8, text, " \t\r\n").len == 0) return error.InvalidCharacter;
    }
    return environment.path("TF_DSV41_DECODE_TP_PROF_ROWS", raw);
}

/// BIAS_VL has always selected the bias when nonempty, including 0; validation does not reinterpret that value.
pub fn imageBiasEnv() !bool {
    const raw = environment.rawText("TF_DSV41_BIAS_VL");
    try environment.check("TF_DSV41_BIAS_VL", raw);
    return raw != null and std.mem.trim(u8, raw.?, " \t\r\n").len > 0;
}

test {
    _ = @import("env_validate_test.zig");
}
