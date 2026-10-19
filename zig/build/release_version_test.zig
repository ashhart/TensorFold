//! Compare actual source metadata instead of a second hardcoded release string.
const std = @import("std");
test "source version matches the newest changelog release heading" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    var diagnostic: std.zon.parse.Diagnostics = undefined;
    const manifest = try std.zon.parse.fromSlice(struct { version: []const u8 }, .{ .gpa = a, .arena = arena.allocator(), .source = @embedFile("release_manifest"), .diagnostics = &diagnostic, .ignore_unknown_fields = true });
    _ = try std.SemanticVersion.parse(manifest.version);
    var newest: ?std.SemanticVersion = null;
    var heading: []const u8 = "";
    var lines = std.mem.splitScalar(u8, @embedFile("release_changelog"), '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "## ")) continue;
        const text = std.mem.trim(u8, line[3..], " \t\r");
        const version = std.SemanticVersion.parse(text) catch continue;
        if (newest == null or version.order(newest.?) == .gt) {
            newest = version;
            heading = text;
        }
    }
    try std.testing.expect(newest != null);
    try std.testing.expectEqualStrings(heading, manifest.version);
}
