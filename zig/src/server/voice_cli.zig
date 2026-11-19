//! The opt-in voice flags of a speech server, parsed beside the serve parser's own flag dispatch.
const std = @import("std");
const cli = @import("cli.zig");

pub const flags = [_]cli.Flag{
    .{ .name = "--request-voices", .kind = .store_true, .native = true },
    .{ .name = "--manage-voices", .kind = .store_true, .native = true },
};

/// The served catalog's settings: --voices DIR, --default-voice NAME and the two flags.
pub const Config = struct {
    dir: ?[]const u8 = null,
    default: []const u8 = "default",
    inline_voices: bool = false,
    manage: bool = false,
};

pub const Flags = struct {
    /// --request-voices: a speech request's `voice` may be an attested voice object instead of a catalog name.
    inline_voices: bool = false,
    /// --manage-voices: admin keys add (POST /v1/voices) and delete (DELETE /v1/voices/NAME) catalog voices.
    manage: bool = false,
};

/// True when `name` is a voice flag, now set.
pub fn apply(f: *Flags, name: []const u8) bool {
    if (std.mem.eql(u8, name, "--request-voices")) f.inline_voices = true else if (std.mem.eql(u8, name, "--manage-voices")) f.manage = true else return false;
    return true;
}

test "voice CLI: --request-voices and --manage-voices default off and are switches" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    try std.testing.expectEqual(Flags{}, (try cli.parse(a, &.{"m"}, &u)).voice);
    try std.testing.expectEqual(Flags{ .inline_voices = true }, (try cli.parse(a, &.{ "m", "--request-voices" }, &u)).voice);
    try std.testing.expectEqual(Flags{ .inline_voices = true, .manage = true }, (try cli.parse(a, &.{ "m", "--manage-voices", "--request-voices" }, &u)).voice);
}
