//! GPU-family and backend defaults for the existing exact grouped shared-round path.
const std = @import("std");
const environment = @import("sys").environment;
const name = "TF_FN_GX_SHARED";

/// The device generation is applegpu_gN from the existing Metal detector, not an MTLGPUFamily number.
pub fn defaultFor(backend: []const u8, generation: u32) bool {
    return std.mem.eql(u8, backend, "metal") and generation == 15;
}

pub fn parse(backend: []const u8, generation: u32, raw: ?[]const u8) !bool {
    return environment.value(bool, name, raw, defaultFor(backend, generation));
}

pub fn on(backend: []const u8, generation: u32) !bool {
    return environment.read(bool, name, defaultFor(backend, generation));
}

test "Flash Next grouped shared: defaults follow backend and GPU generation for unset and blank" {
    const cases = environment.valid_cases.gx_shared_defaults;
    for (cases) |c| for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| {
        try std.testing.expectEqual(c.enabled, try parse(c.backend, c.generation, raw));
    };
}

test "Flash Next grouped shared: registered 0 and 1 override every backend default" {
    const cases = environment.valid_cases;
    for (cases.gx_shared_defaults) |c| {
        var checked: usize = 0;
        for (cases.valid) |v| if (std.mem.eql(u8, v.name, name)) {
            try std.testing.expectEqual(v.n != 0, try parse(c.backend, c.generation, v.raw));
            checked += 1;
        };
        try std.testing.expectEqual(@as(usize, 2), checked);
    }
}

test "Flash Next grouped shared: strict flags and actual Metal loader retain one policy" {
    for ([_][]const u8{ "2", "-1", "true", "false", "on", "yes", "00", "1 ", " 0" }) |raw| {
        try std.testing.expectError(error.InvalidEnvironment, parse("metal", 17, raw));
        try std.testing.expectError(error.InvalidEnvironment, parse("cuda", 15, raw));
    }
    const loader = @embedFile("engine_load.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, loader, "r.gx_shared = try @import(\"gx_shared.zig\").on(\"metal\", r.device.generation());"));
}

test "Flash Next grouped shared: M5 default is off and M3 alone opts in" {
    try std.testing.expect(!try parse("metal", 17, null));
    try std.testing.expect(try parse("metal", 15, null));
    for ([_]u32{ 0, 9, 10, 13, 14, 16, 18 }) |generation| {
        try std.testing.expect(!try parse("metal", generation, null));
    }
}
