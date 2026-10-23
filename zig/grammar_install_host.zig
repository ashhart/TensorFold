//! Check the production install destinations without building xgrammar or a native server.
const std = @import("std");
const layout = @import("grammar_layout");

test "grammar install: Linux is the native server's sibling lib directory" {
    try sibling(layout.linux);
}

test "grammar install: macOS is the native server's sibling lib directory" {
    try sibling(layout.macos);
}

test "grammar install: Windows is the native server's sibling lib directory" {
    try sibling(layout.windows);
}

fn sibling(dir: []const u8) !void {
    const a = std.testing.allocator;
    for ([_][]const u8{ "zig-out", "/opt/tensorfold", "custom prefix" }) |prefix| {
        const expected = try std.fs.path.resolve(a, &.{ prefix, layout.server_dir, "..", "lib" });
        defer a.free(expected);
        const actual = try std.fs.path.resolve(a, &.{ prefix, dir });
        defer a.free(actual);
        try std.testing.expectEqualStrings(expected, actual);
    }
}
