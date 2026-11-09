//! serve --prefill-from, --mcdma-link and --mcdma-produce (#1594 step 2b): a GB10 prefills for the Mac over mcdma-rpcd.
const std = @import("std");

/// The three flags' rows in serve's table: the Mac pulls, and a GB10 build that serves exports produces.
pub fn flags(comptime Flag: type, comptime produce: bool) [3]Flag {
    const mac = @import("builtin").os.tag == .macos;
    return .{ .{ .name = "--prefill-from", .native = mac }, .{ .name = "--mcdma-link", .native = mac }, .{ .name = "--mcdma-produce", .native = produce } };
}

/// cli.zig's apply for these flags: a malformed value is a usage error naming the flag; false for any other flag.
pub fn flag(a: std.mem.Allocator, out: anytype, name: []const u8, value: []const u8, u: anytype) !bool {
    return apply(out, name, value) catch |err| @import("cli.zig").fail(u, a, "argument {s}: {s}", .{ name, @errorName(err) });
}

/// `out` is the Args' lane_remote.Flags; false for any other flag.
pub fn apply(out: anytype, name: []const u8, value: []const u8) !bool {
    if (std.mem.eql(u8, name, "--prefill-from")) {
        const uri = std.Uri.parse(value) catch return error.ExpectedHttpUrl;
        if (!std.ascii.eqlIgnoreCase(uri.scheme, "http") or uri.host == null) return error.ExpectedHttpUrl;
        out.url = value;
    } else if (std.mem.eql(u8, name, "--mcdma-link")) {
        if (!validLink(value)) return error.ExpectedLinkName;
        out.link = value;
    } else if (std.mem.eql(u8, name, "--mcdma-produce")) {
        if (!validLink(value)) return error.ExpectedLinkName;
        out.produce = value;
    } else return false;
    return true;
}

/// mcdma-rpcd's link names: 1-20 characters of [A-Za-z0-9_-].
fn validLink(name: []const u8) bool {
    if (name.len < 1 or name.len > 20) return false;
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return false;
    return true;
}

test "the remote prefill flags take an http URL and an mcdma-rpcd link name, and refuse others by name" {
    var f: struct { url: ?[]const u8 = null, link: ?[]const u8 = null, produce: ?[]const u8 = null } = .{};
    try std.testing.expect(try apply(&f, "--prefill-from", "http://192.0.2.20:8000"));
    try std.testing.expect(try apply(&f, "--mcdma-link", "tf2b"));
    try std.testing.expectEqualStrings("tf2b", f.link.?);
    try std.testing.expect(try apply(&f, "--mcdma-produce", "tf2b"));
    try std.testing.expectError(error.ExpectedHttpUrl, apply(&f, "--prefill-from", "https://gb10:8000"));
    try std.testing.expectError(error.ExpectedHttpUrl, apply(&f, "--prefill-from", "gb10:8000"));
    try std.testing.expectError(error.ExpectedLinkName, apply(&f, "--mcdma-link", "a/b"));
    try std.testing.expect(!try apply(&f, "--context", "4096"));
}
