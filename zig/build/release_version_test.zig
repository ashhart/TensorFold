//! Compare actual source metadata instead of a second hardcoded release string.
const std = @import("std");
fn sourceVersion(a: std.mem.Allocator, arena: std.mem.Allocator) ![]const u8 {
    var diagnostic: std.zon.parse.Diagnostics = undefined;
    const manifest = try std.zon.parse.fromSlice(struct { version: []const u8 }, .{ .gpa = a, .arena = arena, .source = @embedFile("release_manifest"), .diagnostics = &diagnostic, .ignore_unknown_fields = true });
    _ = try std.SemanticVersion.parse(manifest.version);
    return manifest.version;
}
test "source version matches the newest changelog release heading" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const source = try sourceVersion(a, arena.allocator());
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
    try std.testing.expectEqualStrings(heading, source);
}
/// A doc with no `V=` line fails too, so a reworded example cannot drop out of the check unseen.
fn expectInstallVersion(doc: []const u8, source: []const u8) !void {
    var seen: usize = 0;
    var lines = std.mem.splitScalar(u8, doc, '\n');
    while (lines.next()) |line| {
        const text = std.mem.trim(u8, line, " \t\r");
        if (!std.mem.startsWith(u8, text, "V=")) continue;
        seen += 1;
        try std.testing.expectEqualStrings(source, text[2..]);
    }
    try std.testing.expect(seen > 0);
}
test "install examples in the README and runbook name the source version" {
    const a = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const source = try sourceVersion(a, arena.allocator());
    try expectInstallVersion(@embedFile("release_readme"), source);
    try expectInstallVersion(@embedFile("release_runbook"), source);
}
