//! The opt-in speech output flags, parsed beside the serve parser's own flag dispatch.
const std = @import("std");
const cli = @import("cli.zig");

pub const flags = [_]cli.Flag{
    .{ .name = "--speech-speed", .kind = .store_true, .native = true },
};

pub const Flags = struct {
    /// --speech-speed: a speech request's `speed` from 0.25 to 4 stretches the finished take.
    speed: bool = false,
};

/// True when `name` is a speech output flag, now set.
pub fn apply(f: *Flags, name: []const u8) bool {
    if (!std.mem.eql(u8, name, "--speech-speed")) return false;
    f.speed = true;
    return true;
}

test "speech CLI: --speech-speed defaults off and is a switch" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    try std.testing.expect(!(try cli.parse(a, &.{"m"}, &u)).speech.speed);
    try std.testing.expect((try cli.parse(a, &.{ "m", "--speech-speed" }, &u)).speech.speed);
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "m", "--speech-speed=1" }, &u));
}
