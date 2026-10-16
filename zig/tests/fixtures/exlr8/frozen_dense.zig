//! sha256 of each dense case's windows (widths 1-16 in order) from the first build; never regenerate in a test.
const std = @import("std");

const Case = struct { name: []const u8, windows: []const u8 };
pub const cases = [_]Case{
    .{ .name = "int4 g32 by output f32", .windows = "b5000f101b8be9aa35847cdaa6ee503dd7b370a03946b3199cb1309765e839e4" },
    .{ .name = "int4 g32 by input f32", .windows = "b2845fdbc27d65060794e35f58bd110a4f451f73f572034ff078c8c799fbad3d" },
    .{ .name = "MXFP8 bf16", .windows = "66e64bbc40739696bd368b8293a171f21ddc05e64469c4b8eb3618c7481b9633" },
    .{ .name = "FP8 128x128 blocks f32", .windows = "7a701a995416d8b230681fe177125dc07ddb6b7c2d70d2b117bb12a243283237" },
    .{ .name = "FP8 a scale a row f32", .windows = "879923b520aa984e4a92e0d0ec2f63441e3855cea3acd1ee32748164ff64d50d" },
    .{ .name = "bf16 f32", .windows = "1bb937faed77e88e4b75921fac0a62e49f8f3c45fa214e17c2b6b893c271af6a" },
    .{ .name = "fp32 bf16", .windows = "9c6badc018376da562423c504ad7c67b392d1eece59c18ad8c2f92f9ddb3f73a" },
};

/// A case's windows against its frozen digest.
pub fn compare(name: []const u8, digest: []const u8) !void {
    for (cases) |c| if (std.mem.eql(u8, c.name, name)) {
        if (std.mem.eql(u8, c.windows, digest)) return;
        std.debug.print("{s}: the windows differ from their frozen bytes\n", .{name});
        return error.FrozenWindowMismatch;
    };
    return error.MissingFrozenCase;
}

test "an unknown case is refused" {
    try std.testing.expectError(error.MissingFrozenCase, compare("none", ""));
}
