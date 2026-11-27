//! Backend defaults for the existing exact grouped shared-round path.
const std = @import("std");
const environment = @import("sys").environment;
const name = "TF_FN_GX_SHARED";

/// Only Metal changes its default; explicit values retain the shared strict flag parser.
pub fn defaultFor(backend: []const u8) bool {
    return std.mem.eql(u8, backend, "metal");
}

pub fn parse(backend: []const u8, raw: ?[]const u8) !bool {
    return environment.value(bool, name, raw, defaultFor(backend));
}

pub fn on(backend: []const u8) !bool {
    return environment.read(bool, name, defaultFor(backend));
}

test "Flash Next grouped shared: defaults are backend-specific for unset and blank" {
    const cases = environment.valid_cases.gx_shared_defaults;
    for (cases) |c| for ([_]?[]const u8{ null, "", " ", "\t", " \t\r\n" }) |raw| {
        try std.testing.expectEqual(c.enabled, try parse(c.backend, raw));
    };
}

test "Flash Next grouped shared: registered 0 and 1 override every backend default" {
    const cases = environment.valid_cases;
    for (cases.gx_shared_defaults) |c| {
        var checked: usize = 0;
        for (cases.valid) |v| if (std.mem.eql(u8, v.name, name)) {
            try std.testing.expectEqual(v.n != 0, try parse(c.backend, v.raw));
            checked += 1;
        };
        try std.testing.expectEqual(@as(usize, 2), checked);
    }
}

test "Flash Next grouped shared: strict flags and actual Metal loader retain one policy" {
    for ([_][]const u8{ "2", "-1", "true", "false", "on", "yes", "00", "1 ", " 0" }) |raw| {
        try std.testing.expectError(error.InvalidEnvironment, parse("metal", raw));
        try std.testing.expectError(error.InvalidEnvironment, parse("cuda", raw));
    }
    const loader = @embedFile("engine_load.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, loader, "r.gx_shared = try @import(\"gx_shared.zig\").on(\"metal\");"));
}
