//! The pass's pool-sizing settings reach their consumers through platform/environment.zig.
const std = @import("std");
const env = @import("sys").env;
const pass = @import("cuda_exl3_pass.zig");

const SavedEnv = struct {
    name: [:0]const u8,
    value: ?[:0]u8,

    fn take(name: [:0]const u8) !SavedEnv {
        const old = if (@import("sys").getenv(name)) |v| try std.testing.allocator.dupeSentinel(u8, v, 0) else null;
        return .{ .name = name, .value = old };
    }

    fn set(s: SavedEnv, value: ?[:0]const u8) !void {
        if (value) |v| try std.testing.expectEqual(@as(c_int, 0), env.setenv(s.name, v, 1)) else _ = env.unsetenv(s.name);
    }

    fn restore(s: SavedEnv) void {
        if (s.value) |v| {
            _ = env.setenv(s.name, v, 1);
            std.testing.allocator.free(v);
        } else _ = env.unsetenv(s.name);
    }
};

test "exl3: UNIFORM_LINES obeys zero and one and refuses other text" {
    const saved = try SavedEnv.take("TF_FLASHNEXT_EXL3_UNIFORM_LINES");
    defer saved.restore();
    for ([_]?[:0]const u8{ null, "", " ", "\t" }) |unset| {
        try saved.set(unset);
        try std.testing.expect(!try pass.uniformLines());
    }
    try saved.set("0");
    try std.testing.expect(!try pass.uniformLines());
    try saved.set("1");
    try std.testing.expect(try pass.uniformLines());
    for ([_][:0]const u8{ "2", "yes", "true", "on", " 1", "1 " }) |text| {
        try saved.set(text);
        try std.testing.expectError(error.InvalidEnvironment, pass.uniformLines());
    }
}
