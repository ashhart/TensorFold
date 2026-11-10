//! GLM's TF_GLM_* rows: each reader's name is a glm row; blank, 0 and 1 keep their meaning and other text is refused.
const std = @import("std");
const sys = @import("sys");
const e = sys.environment;
const env = sys.env;
const t = std.testing;

/// The names GLM's CUDA readers pass to the shared parser, and the default each reader keeps.
const Name = struct { name: [:0]const u8, default: bool = false };
const switches = [_]Name{
    .{ .name = "TF_GLM_HOST_CACHE" },
    .{ .name = "TF_GLM_VISION_HOST" },
    .{ .name = "TF_GLM_PROF" },
    .{ .name = "TF_GLM_LINK_WAIT" },
    .{ .name = "TF_GLM_PDL", .default = true },
    .{ .name = "TF_GLM_PROMPT_STEPS", .default = true },
    .{ .name = "TF_GLM_SHARED", .default = true },
};
const counts = [_][:0]const u8{"TF_GLM_DFLASH2_LOG"};
const reals = [_][:0]const u8{"TF_GLM_DRAFT_CONFIDENCE"};
/// The window GLM may serve past config.json's: GLM's raisedCeiling reads it at the open, after these rows' check.
const windows = [_][:0]const u8{"TF_GLM_CONTEXT_CEILING"};

/// The row by name without `rule`, so a missing row fails an assertion instead of panicking.
fn row(name: []const u8) !e.Rule {
    for (e.rules) |r| if (std.mem.eql(u8, r.name, name)) return r;
    std.debug.print("{s} has no row in platform/environment.zig\n", .{name});
    return error.TestUnexpectedResult;
}

const Fixture = struct {
    name: []const u8,
    raw: ?[]const u8,
    pub fn get(f: Fixture, key: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, key, f.name)) f.raw else null;
    }
};

fn check(name: []const u8, raw: []const u8, valid: bool) !void {
    const fx: Fixture = .{ .name = name, .raw = raw };
    const why = try e.refusal(t.allocator, fx, .{ .glm = true });
    defer if (why) |w| t.allocator.free(w);
    // a non-GLM CUDA start checks the cuda scope only, so a GLM typo never stops it
    try t.expect(try e.refusal(t.allocator, fx, .{ .cuda = true, .dsv41 = true }) == null);
    if (valid) {
        try e.checkScope(.glm, fx);
        return t.expect(why == null);
    }
    try t.expectError(error.InvalidEnvironment, e.checkScope(.glm, fx));
    try t.expect(std.mem.startsWith(u8, why.?, name));
    if (std.mem.indexOfAny(u8, raw, "\t\r\n") == null) try t.expect(std.mem.indexOf(u8, why.?, raw) != null);
    try t.expect(std.mem.endsWith(u8, why.?, (try row(name)).forms));
}

test "GLM strict env: every GLM reader's name is a glm row, and the glm scope holds nothing else" {
    for (switches) |s| try t.expectEqual(e.Kind.flag, (try row(s.name)).kind);
    for (counts) |name| try t.expectEqual(e.Kind.u64, (try row(name)).kind);
    for (reals) |name| try t.expectEqual(e.Kind.real, (try row(name)).kind);
    for (windows) |name| try t.expectEqual(e.Kind.u64, (try row(name)).kind);
    var n: usize = 0;
    for (e.rules) |r| if (r.scope == .glm) {
        try t.expect(std.mem.startsWith(u8, r.name, "TF_GLM_"));
        n += 1;
    };
    try t.expectEqual(switches.len + counts.len + reals.len + windows.len, n);
}

test "GLM strict env: switches keep unset, blank, 0 and 1, and refuse words and padded digits by name" {
    for (switches) |s| {
        _ = try row(s.name);
        for ([_]?[]const u8{ null, "", " ", " \t\r\n" }) |raw| {
            try t.expectEqual(s.default, try e.value(bool, s.name, raw, s.default));
            if (raw) |v| try check(s.name, v, true);
        }
        try t.expect(!try e.value(bool, s.name, "0", true));
        try t.expect(try e.value(bool, s.name, "1", false));
        for ([_][]const u8{ "yes", "true", "on", "off", "2", "01", "-1", " 1", "1 ", "0\t", "bad" }) |raw| try check(s.name, raw, false);
    }
}

test "GLM strict env: the DFlash2 log takes a whole count, as its old parseInt did, and refuses the rest by name" {
    for (counts) |name| {
        _ = try row(name);
        try t.expectEqual(@as(u64, 0), try e.value(u64, name, null, 0));
        try t.expectEqual(@as(u64, 0), try e.value(u64, name, " ", 0));
        try t.expectEqual(@as(u64, 4), try e.value(u64, name, "+4", 0));
        try t.expectEqual(@as(u64, std.math.maxInt(u64)), try e.value(u64, name, "18446744073709551615", 0));
        for ([_][]const u8{ "0", "16" }) |raw| try check(name, raw, true);
        for ([_][]const u8{ "all", "-1", "1.5", " 4", "0x10", "18446744073709551616" }) |raw| try check(name, raw, false);
    }
}

test "GLM strict env: the actual readers and the open's check keep defaults, 0 and 1, and refuse a typo" {
    const Case = struct { name: [:0]const u8, read: *const fn () anyerror!bool };
    const cases = [_]Case{
        .{ .name = "TF_GLM_HOST_CACHE", .read = @import("cuda_snapshot_pages.zig").onHost },
        .{ .name = "TF_GLM_VISION_HOST", .read = @import("cuda_vision_weights.zig").onHost },
        .{ .name = "TF_GLM_PROF", .read = @import("cuda_prof.zig").Prof.wanted },
    };
    const startup = @import("cuda_env.zig").startup;
    for (cases) |c| {
        _ = try row(c.name);
        _ = env.unsetenv(c.name);
        defer _ = env.unsetenv(c.name);
        try t.expect(!try c.read());
        _ = env.setenv(c.name, "0", 1);
        try t.expect(!try c.read());
        _ = env.setenv(c.name, "1", 1);
        try t.expect(try c.read());
        try startup();
        _ = env.setenv(c.name, "\t", 1);
        try t.expect(!try c.read());
        try startup();
        _ = env.setenv(c.name, "yes", 1);
        try t.expectError(error.InvalidEnvironment, c.read());
        try t.expectError(error.InvalidEnvironment, startup());
    }
    // the open's refusal reaches the server in GLM's words, not as a failed kernel load
    try t.expect(@import("cuda_native.zig").explain(null, error.InvalidEnvironment) != null);
}

test "GLM strict env: the DFlash2 cut is a finite chance from 0 to 1, read as a float by the actual reader" {
    for (reals) |name| {
        _ = try row(name);
        try t.expectEqual(@as(f64, 0.7), try e.value(f64, name, null, 0.7));
        try t.expectEqual(@as(f64, 0.7), try e.value(f64, name, " \t", 0.7));
        for ([_][]const u8{ "0", "0.25", "0.7", "1", "1e-3" }) |raw| {
            try t.expectEqual(try std.fmt.parseFloat(f64, raw), try e.value(f64, name, raw, 0.7));
            try check(name, raw, true);
        }
        for ([_][]const u8{ "nan", "-nan", "inf", "-inf", "1e999", "-0.1", "-1", "1.01", "2", "text", " 0.5", "0.5 " }) |raw| try check(name, raw, false);
    }
    const confidence = @import("dflash.zig").confidence;
    _ = env.unsetenv("TF_GLM_DRAFT_CONFIDENCE");
    defer _ = env.unsetenv("TF_GLM_DRAFT_CONFIDENCE");
    try t.expectEqual(@as(f64, 0.7), try confidence());
    _ = env.setenv("TF_GLM_DRAFT_CONFIDENCE", "0.5", 1);
    try t.expectEqual(@as(f64, 0.5), try confidence());
    _ = env.setenv("TF_GLM_DRAFT_CONFIDENCE", "1.5", 1);
    try t.expectError(error.InvalidEnvironment, confidence());
    try t.expectError(error.InvalidEnvironment, @import("cuda_env.zig").startup());
}

test "GLM strict env: the context ceiling is a whole token count from 1 to 2147483616; unset or blank keeps config.json's" {
    for (windows) |name| {
        _ = try row(name);
        for ([_]?[]const u8{ null, "", " \t" }) |raw| {
            try t.expectEqual(@as(?u64, null), try e.value(?u64, name, raw, null));
            if (raw) |v| try check(name, v, true);
        }
        for ([_][]const u8{ "1", "2097152", "2147483616" }) |raw| {
            try t.expectEqual(@as(?u64, try std.fmt.parseInt(u64, raw, 10)), try e.value(?u64, name, raw, null));
            try check(name, raw, true);
        }
        for ([_][]const u8{ "0", "2147483617", "+2097152", " 2097152", "2097152 ", "2M", "0x200000", "2097152.0" }) |raw| try check(name, raw, false);
    }
}

test "GLM strict env: the open's check refuses a malformed ceiling before ranks, NCCL or weights" {
    const startup = @import("cuda_env.zig").startup;
    const name = "TF_GLM_CONTEXT_CEILING";
    _ = try row(name);
    _ = env.unsetenv(name);
    defer _ = env.unsetenv(name);
    try startup();
    _ = env.setenv(name, "2097152", 1);
    try startup();
    _ = env.setenv(name, "2M", 1);
    try t.expectError(error.InvalidEnvironment, startup());
}
